"""Hyperparameter search orchestration for auto-tuning.

Coordinates candidate generation, profiling, and selection of optimal
configuration for a given model and topology.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from typing import Callable

from megatext.autotune.profiler import ProfileResult, _first_error_line, profile_candidate
from megatext.autotune.strategies import (
    BATCH_SIZES,
    REMAT_POLICIES,
    SA_BLOCK_SIZES,
    Candidate,
    ModelConstraints,
)
from megatext.autotune.topology import TPUTopology, detect_topology

CandidateEvaluator = Callable[[dict, Candidate, int, int], ProfileResult]


def _parse_positive_int(value, default: int) -> int:
    """Best-effort parser for positive integer config values."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _effective_sa_block_limit(config_overrides: dict, refine_sa_backward: bool = False) -> int:
    """Return the largest sa_block worth searching for the current config."""
    seq_len = _parse_positive_int(config_overrides.get("max_target_length", 4096), 4096)
    sliding_window = _parse_positive_int(config_overrides.get("sliding_window_size", 0), 0)
    if sliding_window > 0:
        factor = 2 if refine_sa_backward else 1
        return min(seq_len, sliding_window * factor)
    return seq_len


def _backward_sa_block_candidates(sa_block_size: int, refine_sa_backward: bool = False) -> list[int]:
    """Return staged backward block candidates for a forward sa_block."""
    candidates = [sa_block_size]
    if not refine_sa_backward:
        return candidates
    half = sa_block_size // 2
    if half >= 512 and half != sa_block_size:
        candidates.append(half)
    return candidates


@dataclass
class AutoTuneConfig:
    """Configuration for automated hyperparameter search."""

    scope: str = "batch_remat"
    warmup_steps: int = 3
    num_profile_steps: int = 3
    include_sa_block: bool = False
    refine_sa_backward: bool = False


@dataclass
class SearchResult:
    """Result of a hyperparameter search."""

    best_candidate: Candidate | None
    best_result: ProfileResult | None
    all_results: list[ProfileResult]
    topology: TPUTopology
    search_time_seconds: float

    def summary(self) -> str:
        """Human-readable summary of search results."""
        evaluated = len(self.all_results)
        valid = [r for r in self.all_results if r.succeeded]
        failed = [r for r in self.all_results if not r.succeeded]
        lines = [
            f"Auto-tune Search Results",
            f"{'=' * 50}",
            f"Topology: {self.topology.platform} {self.topology.chip_name}, "
            f"{self.topology.device_count} devices, {self.topology.num_slices} slice(s)",
            f"Evaluated: {evaluated} ({len(valid)} succeeded, {len(failed)} failed)",
            f"Search time: {self.search_time_seconds:.1f}s",
            f"",
        ]

        if self.best_result is not None:
            lines += [
                f"Best configuration:",
                f"  {self.best_candidate}",
                f"  TFLOPs/s/device: {self.best_result.tflops_per_device:.1f}",
                f"  Step time: {self.best_result.mean_step_time_seconds:.3f}s",
                f"  Peak memory: {self.best_result.peak_memory_gb:.1f} GB",
            ]
        else:
            lines.append("No feasible configuration found.")

        if valid:
            valid.sort(key=lambda r: r.tflops_per_device, reverse=True)
            frontier = pareto_frontier(valid)
            lines += [f"", f"Pareto frontier (memory asc):"]
            for r in sorted(frontier, key=lambda item: (item.peak_memory_gb, -item.tflops_per_device)):
                lines.append(
                    f"  mem={r.peak_memory_gb:.1f}GB, "
                    f"{r.tflops_per_device:.1f} TFLOPs/s - {r.candidate}"
                )
            lines += [f"", f"All feasible results (by TFLOPs/s):"]
            for i, r in enumerate(valid):
                lines.append(
                    f"  {i + 1}. {r.tflops_per_device:.1f} TFLOPs/s, "
                    f"{r.mean_step_time_seconds:.3f}s, "
                    f"mem={r.peak_memory_gb:.1f}GB - "
                    f"{r.candidate}"
                )

        if failed:
            lines += [f"", f"Failed candidates:"]
            for r in failed:
                detail = _first_error_line(r.error) or r.returncode_detail or r.exit_class
                stage = f", stage={r.failure_stage}" if r.failure_stage else ""
                lines.append(
                    f"  {r.candidate} - {detail}{stage} "
                    f"(mem={r.peak_memory_gb:.1f}GB)"
                )

        return "\n".join(lines)


def run_search(
    config_overrides: dict,
    autotune_config: AutoTuneConfig | None = None,
    topology: TPUTopology | None = None,
    max_batch_size: int = 8,
    evaluator: CandidateEvaluator | None = None,
) -> SearchResult:
    """Run auto-tuning search. Always optimizes for TFLOPs/s/device.

    Two independent modes based on autotune_config.scope:
      - batch_remat: search remat policies x batch sizes with OOM pruning
      - parallelism: search TP/FSDP splits

    Args:
        config_overrides: Dict of key=value overrides for megatext pyconfig.
        autotune_config: AutoTuneConfig instance. Required.
        topology: Pre-detected topology (auto-detected if None).

    Returns:
        SearchResult with the best configuration found.
    """
    if autotune_config is None:
        autotune_config = AutoTuneConfig()
    if evaluator is None:
        evaluator = profile_candidate

    start_time = time.monotonic()

    if topology is None:
        topology = detect_topology()

    constraints = ModelConstraints.from_config_dict(config_overrides, topology)
    all_results: list[ProfileResult] = []
    num_steps = autotune_config.num_profile_steps
    warmup_steps = autotune_config.warmup_steps
    best_result: ProfileResult | None = None

    scope = autotune_config.scope
    batch_sizes = [b for b in BATCH_SIZES if b <= max_batch_size]

    # -- Stage 1: Batch + remat (+ optional sa_block) --------------------
    if scope in ("batch_remat", "all"):
        include_sa = autotune_config.include_sa_block
        sa_block_limit = _effective_sa_block_limit(
            config_overrides,
            refine_sa_backward=autotune_config.refine_sa_backward,
        )

        if include_sa:
            valid_sa_blocks = [s for s in SA_BLOCK_SIZES if s <= sa_block_limit] or [SA_BLOCK_SIZES[0]]
        else:
            valid_sa_blocks = [int(config_overrides.get("sa_block_q", 512))]

        logging.info(
            "=== Batch + remat search (sa_blocks=%s, sa_block_limit=%s, max_batch=%s) ===",
            valid_sa_blocks,
            sa_block_limit,
            batch_sizes[-1],
        )

        best_result = _search_batch_remat(
            config_overrides,
            all_results,
            num_steps,
            warmup_steps,
            valid_sa_blocks=valid_sa_blocks,
            batch_sizes=batch_sizes,
            refine_sa_backward=autotune_config.refine_sa_backward,
            evaluator=evaluator,
        )

        if best_result is None:
            logging.warning("All batch+remat combinations failed (OOM).")

    # -- Stage 2: Parallelism --------------------------------------------
    if scope in ("parallelism", "all"):
        tp_values = constraints.valid_tp_values()
        logging.info(f"=== Parallelism search (TP={tp_values}) ===")

        if best_result is not None:
            base = best_result.candidate
        else:
            base = Candidate.from_config_dict(config_overrides)

        skip_base_tp = scope == "all"
        stage_best = _search_parallelism(
            config_overrides,
            base,
            tp_values,
            all_results,
            num_steps,
            warmup_steps,
            skip_base_tp,
            evaluator=evaluator,
        )

        if stage_best is not None:
            if best_result is None or stage_best.tflops_per_device > best_result.tflops_per_device:
                best_result = stage_best
        elif best_result is None:
            logging.warning("All parallelism candidates failed (OOM).")

    if scope not in ("batch_remat", "parallelism", "all"):
        raise ValueError(f"Unknown scope: {scope!r}. Use 'batch_remat', 'parallelism', or 'all'.")

    search_time = time.monotonic() - start_time

    return SearchResult(
        best_candidate=best_result.candidate if best_result else None,
        best_result=best_result,
        all_results=all_results,
        topology=topology,
        search_time_seconds=search_time,
    )


def _fmt_result(result: ProfileResult) -> str:
    """Format a profile result for logging."""
    stage = f", stage={result.failure_stage}" if result.failure_stage else ""
    detail = _first_error_line(result.error)
    memory_detail = ""
    if result.compile_peak_memory_gb > 0.0 or result.runtime_peak_hbm_gb > 0.0:
        memory_detail = (
            f", compile={result.compile_peak_memory_gb:.1f}, "
            f"runtime={result.runtime_peak_hbm_gb:.1f}"
        )
    if result.infra_error:
        return (
            f"{result.candidate}: FAILED "
            f"({result.returncode_detail or detail or result.error}{stage})"
        )
    if result.confirmed_oom or result.error:
        return (
            f"{result.candidate}: FAILED "
            f"({memory_detail[2:] if memory_detail else ''}"
            f"{stage}"
            f"{', reason=' + detail if detail else ''})"
        )
    return (
        f"{result.candidate}: "
        f"{result.tflops_per_device:.1f} TFLOPs/s, "
        f"mem={result.peak_memory_gb:.1f}GB"
    )


def pareto_frontier(results: list[ProfileResult]) -> list[ProfileResult]:
    """Return non-dominated feasible points maximizing TFLOPs and minimizing memory."""
    frontier: list[ProfileResult] = []
    best_tflops = float("-inf")
    for result in sorted(results, key=lambda item: (item.peak_memory_gb, -item.tflops_per_device)):
        if result.tflops_per_device > best_tflops:
            frontier.append(result)
            best_tflops = result.tflops_per_device
    return frontier


def _search_batch_remat(
    config_overrides: dict,
    all_results: list[ProfileResult],
    num_steps: int,
    warmup_steps: int = 3,
    valid_sa_blocks: list[int] | None = None,
    batch_sizes: list[int] | None = None,
    refine_sa_backward: bool = False,
    evaluator: CandidateEvaluator | None = None,
) -> ProfileResult | None:
    """Search over sa_block × remat × batch with OOM-monotone ceiling shrink.

    Memory is monotonically increasing along all 3 axes:
      sa_block_sizes: ascending (larger block = more memory)
      BATCH_SIZES:    ascending (larger batch = more memory)
      REMAT_POLICIES: [full, ..., minimal] (ascending memory)

    Performance (TFLOPs) is monotone for sa_block and remat, but NOT for batch
    (can peak then drop). So we linear-scan all feasible batches per remat.

    Algorithm:
      Phase 1: Binary search for max feasible sa_block (OOM is monotone).
      Phase 2: For each feasible sa_block (largest first):
        For each remat (least memory first), scan batches 1..ceiling.
        OOM breaks the scan (batch OOM is monotone).
        Ceiling shrinks across remats (remat OOM is monotone).
    """
    best_result: ProfileResult | None = None

    if valid_sa_blocks is None:
        valid_sa_blocks = [512]
    if batch_sizes is None:
        batch_sizes = BATCH_SIZES
    if evaluator is None:
        evaluator = profile_candidate

    _cache: dict[tuple, ProfileResult] = {}

    def _evaluate(candidate: Candidate) -> ProfileResult:
        result = evaluator(config_overrides, candidate, num_steps=num_steps, warmup_steps=warmup_steps)
        all_results.append(result)
        logging.info(_fmt_result(result))
        return result

    def _probe(candidate: Candidate) -> ProfileResult:
        key = (
            candidate.remat_policy,
            candidate.per_device_batch_size,
            candidate.sa_block_size,
            candidate.sa_block_backward_size,
        )
        if key in _cache:
            return _cache[key]
        result = _evaluate(candidate)
        _cache[key] = result
        return result

    # ── Phase 1: Binary search for max feasible forward sa_block ──────────
    max_sa_idx = len(valid_sa_blocks) - 1
    if len(valid_sa_blocks) > 1:
        logging.info("Phase 1: Forward SA block binary search")
        lo, hi = 0, max_sa_idx
        best_sa_idx = -1
        while lo <= hi:
            mid = (lo + hi) // 2
            sa_block = valid_sa_blocks[mid]
            result = _probe(Candidate(
                remat_policy=REMAT_POLICIES[0],
                per_device_batch_size=batch_sizes[0],
                sa_block_size=sa_block,
                sa_block_backward_size=_backward_sa_block_candidates(
                    sa_block, refine_sa_backward=refine_sa_backward
                )[-1],
            ))
            if not result.succeeded:
                hi = mid - 1
            else:
                best_sa_idx = mid
                lo = mid + 1
        if best_sa_idx < 0:
            logging.warning("All SA block sizes infeasible at batch=1, remat=full.")
            return None
        max_sa_idx = best_sa_idx
        logging.info(f"Phase 1 result: best sa_block_fwd={valid_sa_blocks[max_sa_idx]}")

    # ── Phase 2: Backward refinement + remat/batch scan ───────────────────
    # For each forward block, we only refine backward splash blocks over
    # {same, half}. That captures the common "forward can stay large, dkv must
    # shrink" case without exploding the search space.
    for si in range(max_sa_idx, -1, -1):
        sa_block = valid_sa_blocks[si]
        backward_candidates = _backward_sa_block_candidates(
            sa_block,
            refine_sa_backward=refine_sa_backward,
        )

        for sa_block_bwd in backward_candidates:
            logging.info(
                "Phase 2: Scan (sa_block_fwd=%s, sa_block_bwd=%s)",
                sa_block,
                sa_block_bwd,
            )

            batch_ceiling = len(batch_sizes)  # exclusive upper bound

            for remat in REMAT_POLICIES:
                if batch_ceiling <= 0:
                    break

                new_ceiling = 0
                previous_tflops = None
                for bi in range(batch_ceiling):
                    result = _probe(Candidate(
                        remat_policy=remat,
                        per_device_batch_size=batch_sizes[bi],
                        sa_block_size=sa_block,
                        sa_block_backward_size=sa_block_bwd,
                    ))
                    if not result.succeeded:
                        break  # larger batches are pruned after the first failure
                    new_ceiling = bi + 1
                    if best_result is None or result.tflops_per_device > best_result.tflops_per_device:
                        best_result = result
                    if previous_tflops is not None and result.tflops_per_device < previous_tflops:
                        # Batch throughput is assumed unimodal. Once it declines, larger batches
                        # cannot improve the Pareto frontier because they use more memory too.
                        break
                    previous_tflops = result.tflops_per_device

                batch_ceiling = new_ceiling

    return best_result


def _search_parallelism(
    config_overrides: dict,
    base: Candidate,
    tp_values: list[int],
    all_results: list[ProfileResult],
    num_steps: int,
    warmup_steps: int = 3,
    skip_base_tp: bool = False,
    evaluator: CandidateEvaluator | None = None,
) -> ProfileResult | None:
    """Search TP/FSDP splits with fixed batch and remat."""
    best_result: ProfileResult | None = None
    if evaluator is None:
        evaluator = profile_candidate

    for tp in tp_values:
        if skip_base_tp and tp == base.ici_tensor_parallelism:
            continue
        candidate = replace(base, ici_tensor_parallelism=tp, ici_fsdp_parallelism=-1)
        result = evaluator(config_overrides, candidate, num_steps, warmup_steps=warmup_steps)
        all_results.append(result)
        logging.info(_fmt_result(result))
        if result.succeeded:
            if best_result is None or result.tflops_per_device > best_result.tflops_per_device:
                best_result = result

    return best_result


def apply_search_result(
    config_overrides: dict,
    result: SearchResult,
) -> dict:
    """Apply the best candidate from a search result to config overrides.

    Returns a new dict with the best settings merged in.
    """
    if result.best_candidate is None:
        raise ValueError("No feasible configuration found -- cannot apply.")
    overrides = result.best_candidate.to_overrides()
    merged = dict(config_overrides)
    merged.update(overrides)
    return merged


# -- CLI entry point ------------------------------------------------------


def _parse_train_script(script_path: str) -> list[str]:
    """Extract key=value args from a training bash script.

    Finds lines with `python -m megatext.trainers.pretrain`
    and extracts the key=value arguments that follow.
    """
    import re
    import shlex

    with open(script_path) as f:
        content = f.read()

    # Join continuation lines (backslash + newline)
    content = content.replace("\\\n", " ")

    overrides = []
    for line in content.splitlines():
        line = line.strip()
        if not re.search(r"python\d?\s+-m\s+megatext\.trainers\.pretrain", line):
            continue

        # Extract tokens after the module name
        try:
            tokens = shlex.split(line)
        except ValueError:
            continue

        # Find the module arg and collect everything after it
        for i, tok in enumerate(tokens):
            if tok == "megatext.trainers.pretrain":
                for tok in tokens[i + 1:]:
                    if "=" in tok and "$" not in tok:
                        overrides.append(tok)
                break
        break  # only parse the first matching command

    return overrides


def _parse_cli_overrides(raw_overrides: list[str]) -> dict:
    """Parse key=value CLI overrides into a flat dict.

    Simple parsing: all key=value pairs are treated as pyconfig overrides.
    Args without '=' are appended to the previous arg's value (space-separated).
    """
    # First pass: merge args without '=' into the previous arg
    merged: list[str] = []
    for arg in raw_overrides:
        if "=" in arg:
            merged.append(arg)
        elif merged:
            merged[-1] += f" {arg}"
        else:
            raise ValueError(f"Invalid override (expected key=value): {arg}")

    result: dict = {}
    for override in merged:
        key, value = override.split("=", 1)
        # Attempt numeric/bool parsing
        result[key] = _parse_value(value)

    return result


def _parse_value(value: str):
    """Parse a string value to int, float, bool, or leave as str."""
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def main() -> None:
    """CLI entry point for auto-tuning.

    Usage:
        # From a training script (recommended):
        python -m megatext.autotune.search gke/jobs/train/my_job.sh --scope batch_remat

        # Direct overrides:
        python -m megatext.autotune.search --scope batch_remat model_name=qwen3 ici_fsdp_parallelism=-1
    """
    import argparse
    import tempfile

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger("megatext").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(
        prog="autotune",
        description="Auto-tune training hyperparameters. Pass a train script (.sh) or key=value overrides.",
    )
    parser.add_argument("config", nargs="?", default=None,
                        help="Train script (.sh) or first key=value override")
    parser.add_argument("overrides", nargs="*", help="Config overrides as key=value")
    parser.add_argument("--scope", default="all", choices=["all", "batch_remat", "parallelism"],
                        help="What to auto-tune (default: all).")
    parser.add_argument("--num-profile-steps", type=int, default=3, help="Profile steps per candidate (default: 3).")
    parser.add_argument("--warmup-steps", type=int, default=3, help="Warmup steps per candidate (default: 3).")
    parser.add_argument("--include-sa-block", action="store_true",
                        help="Include splash attention block size in batch_remat search.")
    parser.add_argument(
        "--refine-sa-backward",
        action="store_true",
        help="Refine splash backward blocks with {same, half} candidates. Disabled by default.",
    )
    parser.add_argument("--max-batch-size", type=int, default=8, help="Max per-device batch size to search (default: 8).")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: temp dir).")
    args = parser.parse_args()

    config_path = args.config
    all_overrides = list(args.overrides)

    # If config is a .sh train script, extract overrides from it
    if config_path is not None and config_path.endswith(".sh"):
        logging.info(f"Parsing train script: {config_path}")
        script_overrides = _parse_train_script(config_path)
        logging.info(f"Extracted overrides: {script_overrides}")
        all_overrides = script_overrides + all_overrides
        config_path = None
    elif config_path is not None and "=" in config_path:
        all_overrides.insert(0, config_path)
        config_path = None

    topology = detect_topology()
    print(
        f"Topology: {topology.platform} {topology.chip_name}, "
        f"{topology.device_count} devices, {topology.hbm_per_device_gb:.0f} GB HBM/device"
    )

    output_dir = args.output_dir
    tmp_dir = None
    if output_dir is None:
        tmp_dir = tempfile.mkdtemp(prefix="autotune-")
        output_dir = tmp_dir

    try:
        # Parse CLI overrides and force autotune-specific settings
        config_overrides = _parse_cli_overrides(all_overrides)
        config_overrides.update({
            "steps": 10,
            "dataset_type": "synthetic",
            "enable_checkpointing": False,
            "enable_tensorboard": False,
            "base_output_directory": output_dir,
            "run_name": "autotune",
        })

        autotune_cfg = AutoTuneConfig(
            scope=args.scope,
            warmup_steps=args.warmup_steps,
            num_profile_steps=args.num_profile_steps,
            include_sa_block=args.include_sa_block,
            refine_sa_backward=args.refine_sa_backward,
        )
        result = run_search(config_overrides, autotune_config=autotune_cfg, topology=topology, max_batch_size=args.max_batch_size)
        print()
        print(result.summary())
        if result.best_candidate is not None:
            print()
            print("Recommended overrides for your training script:")
            for k, v in result.best_candidate.to_overrides().items():
                print(f"  {k}={v}")
    finally:
        if tmp_dir is not None:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
