"""Hyperparameter search orchestration for auto-tuning.

Coordinates candidate generation, profiling, and selection of optimal
configuration for a given model and topology.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace

from megatext.autotune.profiler import ProfileResult, profile_candidate
from megatext.autotune.strategies import (
    BATCH_SIZES,
    REMAT_POLICIES,
    SA_BLOCK_SIZES,
    Candidate,
    ModelConstraints,
)
from megatext.autotune.topology import TPUTopology, detect_topology


@dataclass
class AutoTuneConfig:
    """Configuration for automated hyperparameter search."""

    enabled: bool = False
    scope: str = "batch_remat"
    num_profile_steps: int = 10
    include_sa_block: bool = False


@dataclass
class SearchResult:
    """Result of a hyperparameter search."""

    best_candidate: Candidate | None
    best_result: ProfileResult | None
    all_results: list[ProfileResult]
    topology: TPUTopology
    search_time_seconds: float
    total_candidates: int = 0
    search_space: str = ""

    def summary(self) -> str:
        """Human-readable summary of search results."""
        evaluated = len(self.all_results)
        pruned = self.total_candidates - evaluated
        lines = [
            f"Auto-tune Search Results",
            f"{'=' * 50}",
            f"Topology: {self.topology.platform} {self.topology.chip_name}, "
            f"{self.topology.device_count} devices, {self.topology.num_slices} slice(s)",
            f"Search space: {self.search_space}",
            f"Total candidates: {self.total_candidates}, "
            f"evaluated: {evaluated}" + (f" ({pruned} pruned)" if pruned > 0 else ""),
            f"Search time: {self.search_time_seconds:.1f}s",
            f"",
            f"Fixed settings: dataset=synthetic, checkpointing=off, "
            f"allow_split_physical_axes=true, ga=1",
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
            lines.append("No feasible configuration found. All candidates OOM.")

        lines += [
            f"",
            f"All results (sorted by TFLOPs/s/device):",
        ]

        valid = [r for r in self.all_results if not r.oom and r.error is None]
        valid.sort(key=lambda r: r.tflops_per_device, reverse=True)
        for i, r in enumerate(valid[:10]):
            lines.append(
                f"  {i + 1}. {r.tflops_per_device:.1f} TFLOPs/s, "
                f"{r.mean_step_time_seconds:.3f}s, "
                f"mem={r.peak_memory_gb:.1f}GB - {r.candidate}"
            )

        oom_count = sum(1 for r in self.all_results if r.oom)
        err_results = [r for r in self.all_results if r.error and not r.oom]
        if oom_count > 0:
            lines.append(f"\nOOM: {oom_count} candidates")
        if err_results:
            lines.append(f"\nErrors ({len(err_results)}):")
            for r in err_results:
                lines.append(f"  {r.error} - {r.candidate}")

        return "\n".join(lines)


def run_search(
    config_overrides: dict,
    autotune_config: AutoTuneConfig | None = None,
    topology: TPUTopology | None = None,
    max_batch_size: int = 8,
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

    start_time = time.monotonic()

    if topology is None:
        topology = detect_topology()

    constraints = ModelConstraints.from_config_dict(config_overrides, topology)
    all_results: list[ProfileResult] = []
    num_steps = autotune_config.num_profile_steps
    best_result: ProfileResult | None = None
    total_candidates = 0
    search_space_parts = []

    scope = autotune_config.scope
    batch_sizes = [b for b in BATCH_SIZES if b <= max_batch_size]

    # -- Stage 1: Batch + remat (+ optional sa_block) --------------------
    if scope in ("batch_remat", "all"):
        include_sa = autotune_config.include_sa_block
        seq_len = int(config_overrides.get("max_target_length", 4096))

        if include_sa:
            valid_sa_blocks = [s for s in SA_BLOCK_SIZES if s <= seq_len] or [SA_BLOCK_SIZES[0]]
        else:
            valid_sa_blocks = [int(config_overrides.get("sa_block_q", 512))]

        stage_total = len(valid_sa_blocks) * len(batch_sizes) * len(REMAT_POLICIES)
        total_candidates += stage_total
        search_space_parts.append(
            f"sa_blocks={valid_sa_blocks}, batch_sizes={batch_sizes}, remat_policies={REMAT_POLICIES}"
        )
        stage_name = "SA block + batch + remat" if include_sa else "Batch + remat"
        logging.info(f"=== {stage_name} search ===")
        logging.info(f"Search space: {stage_total} combinations")

        best_result = _search_batch_remat(config_overrides, all_results, num_steps, valid_sa_blocks=valid_sa_blocks, batch_sizes=batch_sizes)

        if best_result is None:
            logging.warning("All batch+remat combinations failed (OOM). Try reducing model size or max_target_length.")

    # -- Stage 2: Parallelism --------------------------------------------
    if scope in ("parallelism", "all"):
        tp_values = constraints.valid_tp_values()
        total_candidates += len(tp_values)
        search_space_parts.append(f"tp_values={tp_values}")
        logging.info(f"=== Parallelism search ===")
        logging.info(f"Search space: {len(tp_values)} TP values: {tp_values}")

        # Use best from stage 1 as base (for 'all'), or user config (for 'parallelism')
        if best_result is not None:
            base = best_result.candidate
        else:
            base = Candidate.from_config_dict(config_overrides)

        # scope=all: base was already profiled in stage 1, skip it
        # scope=parallelism: base hasn't been profiled, include it
        skip_base_tp = scope == "all"
        stage_best = _search_parallelism(config_overrides, base, tp_values, all_results, num_steps, skip_base_tp)

        if stage_best is not None:
            if best_result is None or stage_best.tflops_per_device > best_result.tflops_per_device:
                best_result = stage_best
        elif best_result is None:
            logging.warning("All parallelism candidates failed (OOM).")

    if scope not in ("batch_remat", "parallelism", "all"):
        raise ValueError(f"Unknown scope: {scope!r}. Use 'batch_remat', 'parallelism', or 'all'.")

    search_time = time.monotonic() - start_time

    if best_result is None:
        logging.error("No feasible configuration found. All candidates OOM.")
        return SearchResult(
            best_candidate=None,
            best_result=None,
            all_results=all_results,
            topology=topology,
            search_time_seconds=search_time,
            total_candidates=total_candidates,
            search_space="; ".join(search_space_parts),
        )

    return SearchResult(
        best_candidate=best_result.candidate,
        best_result=best_result,
        all_results=all_results,
        topology=topology,
        search_time_seconds=search_time,
        total_candidates=total_candidates,
        search_space="; ".join(search_space_parts),
    )


_SEP = "=" * 70


def _log_result(msg: str, eval_count: int, remaining: int, pruned: int = 0) -> None:
    logging.info(_SEP)
    logging.info(msg)
    parts = [f"evaluated={eval_count}", f"remaining={remaining}"]
    if pruned > 0:
        parts.append(f"pruned={pruned}")
    logging.info(f"({', '.join(parts)})")
    logging.info(_SEP)


def _search_batch_remat(
    config_overrides: dict,
    all_results: list[ProfileResult],
    num_steps: int,
    valid_sa_blocks: list[int] | None = None,
    batch_sizes: list[int] | None = None,
) -> ProfileResult | None:
    """Search over sa_block_size x batch_size x remat_policy.

    All axes are monotonically increasing in memory:
      sa_block_sizes: ascending (larger block = more memory, faster)
      BATCH_SIZES:    [1, 2, 3, ..., 16] (ascending)
      REMAT_POLICIES: [full, ..., minimal]  (ascending memory)

    For each (sa_block, batch) pair (largest first), binary search over remat
    to find the fastest remat that fits. Profile that combo.
    """
    best_result: ProfileResult | None = None
    eval_count = 0
    oom_count = 0
    pruned_count = 0

    if valid_sa_blocks is None:
        valid_sa_blocks = [512]
    if batch_sizes is None:
        batch_sizes = BATCH_SIZES

    total = len(valid_sa_blocks) * len(batch_sizes) * len(REMAT_POLICIES)

    # Binary search for max feasible sa_block using (full, batch=1) probe.
    max_sa_idx = len(valid_sa_blocks) - 1
    if len(valid_sa_blocks) > 1:
        logging.info("SA block feasibility (binary search)")
        lo, hi = 0, max_sa_idx
        best_sa_idx = -1
        while lo <= hi:
            mid = (lo + hi) // 2
            probe = Candidate(
                remat_policy=REMAT_POLICIES[0],
                per_device_batch_size=batch_sizes[0],
                sa_block_size=valid_sa_blocks[mid],
            )
            eval_count += 1
            result = profile_candidate(config_overrides, probe, num_steps=num_steps)
            _remaining_count = total - eval_count - pruned_count
            if result.oom or result.error:
                all_results.append(result)
                oom_count += 1
                _log_result(f"{probe}: OOM", eval_count, _remaining_count, pruned_count)
                hi = mid - 1
            else:
                all_results.append(result)
                best_sa_idx = mid
                lo = mid + 1
                _log_result(f"{probe}: {result.tflops_per_device:.1f} TFLOPs/s", eval_count, _remaining_count, pruned_count)
        if best_sa_idx < 0:
            _remaining_count = total - eval_count - pruned_count
            _log_result("All sa_block sizes infeasible", eval_count, _remaining_count, pruned_count)
            return None
        max_sa_idx = best_sa_idx
        pruned_sa = len(valid_sa_blocks) - (max_sa_idx + 1)
        pruned_count += pruned_sa * len(batch_sizes) * len(REMAT_POLICIES)

    # Profile each feasible sa_block (largest first = best performance)
    for si in range(max_sa_idx, -1, -1):
        sa_block = valid_sa_blocks[si]

        max_batch_idx = len(batch_sizes) - 1

        for remat in REMAT_POLICIES:
            if max_batch_idx < 0:
                pruned_count += len(batch_sizes)
                break

            lo, hi = 0, max_batch_idx
            best_batch_idx = -1
            best_batch_result = None

            while lo <= hi:
                mid = (lo + hi) // 2
                candidate = Candidate(
                    remat_policy=remat,
                    per_device_batch_size=batch_sizes[mid],
                    sa_block_size=sa_block,
                )
                eval_count += 1
                result = profile_candidate(config_overrides, candidate, num_steps=num_steps)
                _remaining_count = total - eval_count - pruned_count
                if result.oom or result.error:
                    all_results.append(result)
                    oom_count += 1
                    _log_result(f"{candidate}: OOM", eval_count, _remaining_count, pruned_count)
                    hi = mid - 1
                else:
                    all_results.append(result)
                    _log_result(f"{candidate}: {result.tflops_per_device:.1f} TFLOPs/s", eval_count, _remaining_count, pruned_count)
                    best_batch_idx = mid
                    best_batch_result = result
                    lo = mid + 1

            if best_batch_idx < 0:
                max_batch_idx = -1
                continue

            max_batch_idx = best_batch_idx

            if not best_batch_result.oom and not best_batch_result.error:
                if best_result is None or best_batch_result.tflops_per_device > best_result.tflops_per_device:
                    best_result = best_batch_result

    return best_result


def _search_parallelism(
    config_overrides: dict,
    base: Candidate,
    tp_values: list[int],
    all_results: list[ProfileResult],
    num_steps: int,
    skip_base_tp: bool = False,
) -> ProfileResult | None:
    """Search TP/FSDP splits with fixed batch and remat."""
    best_result: ProfileResult | None = None
    total = len(tp_values)
    eval_count = 0
    oom_count = 0

    for tp in tp_values:
        if skip_base_tp and tp == base.ici_tensor_parallelism:
            continue  # skip -- already profiled in stage 1
        candidate = replace(base, ici_tensor_parallelism=tp, ici_fsdp_parallelism=-1)
        eval_count += 1
        result = profile_candidate(config_overrides, candidate, num_steps)
        all_results.append(result)
        remaining = total - eval_count - oom_count
        if result.oom or result.error:
            oom_count += 1
            _log_result(f"{candidate}: OOM", eval_count, remaining)
        else:
            if best_result is None or result.tflops_per_device > best_result.tflops_per_device:
                best_result = result
            _log_result(f"{candidate}: {result.tflops_per_device:.1f} TFLOPs/s", eval_count, remaining)

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
                overrides.extend(tok for tok in tokens[i + 1:] if "=" in tok)
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
    parser.add_argument("--num-profile-steps", type=int, default=5, help="Profile steps per candidate (default: 5).")
    parser.add_argument("--include-sa-block", action="store_true",
                        help="Include splash attention block size in batch_remat search.")
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
            enabled=True,
            scope=args.scope,
            num_profile_steps=args.num_profile_steps,
            include_sa_block=args.include_sa_block,
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
