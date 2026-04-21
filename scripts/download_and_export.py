"""Download a Megatext checkpoint from GCS and convert it to HuggingFace format.

Combines `gcloud storage rsync` for resumable parallel download with the existing
megatext-to-hf conversion pipeline, so a single command goes from a GCS
checkpoint path to a local HF-ready directory.

Usage examples:

  # Auto-detect latest step, download + convert:
  python custom/download_and_export.py \
    --gcs-path gs://bucket/checkpoints/pretrain-qwen3-8b \
    --hf-config-path checkpoints/config/Qwen3-SWA-8B \
    --output-dir /tmp/qwen3-8b-hf

  # Multiple steps (pipelined: download and conversion overlap):
  python custom/download_and_export.py \
    --gcs-path gs://bucket/checkpoints/pretrain-qwen3-8b \
    --checkpoint-step 4000 6000 8000 \
    --hf-config-path checkpoints/config/Qwen3-SWA-8B \
    --output-dir /tmp/qwen3-8b-hf

  # Skip download (already have local checkpoint):
  python custom/download_and_export.py \
    --gcs-path gs://bucket/checkpoints/pretrain-qwen3-8b \
    --local-dir /data/checkpoints/pretrain-qwen3-8b \
    --checkpoint-step 8000 \
    --hf-config-path checkpoints/config/Qwen3-SWA-8B \
    --output-dir /tmp/qwen3-8b-hf \
    --skip-download
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import typing

log = logging.getLogger(__name__)

# Will be set in main() — file objects for log output redirection.
_subprocess_log: typing.TextIO | None = None
_subprocess_log_path: str | None = None
_convert_log_path: str | None = None


def _init_logs() -> None:
    """Create timestamped log files for download and convert output."""
    global _subprocess_log, _subprocess_log_path, _convert_log_path
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "download_and_export")
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%y%m%d-%H%M%S")

    # Download log: captures gcloud subprocess stdout/stderr
    _subprocess_log_path = os.path.join(log_dir, f"{ts}-download.log")
    _subprocess_log = open(_subprocess_log_path, "w")
    log.info("Download log: %s", _subprocess_log_path)

    # Convert log: captures conversion logging output
    _convert_log_path = os.path.join(log_dir, f"{ts}-convert.log")
    fmt = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    fh = logging.FileHandler(_convert_log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logging.getLogger("megatext.conversion").addHandler(fh)
    logging.getLogger("absl").addHandler(fh)
    log.info("Convert log: %s", _convert_log_path)


# ── GCS helpers ──────────────────────────────────────────────────────────────


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    """Run a subprocess, redirecting stdout/stderr to the log file."""
    log.info("$ %s", " ".join(cmd))
    if _subprocess_log:
        _subprocess_log.write(f"$ {' '.join(cmd)}\n")
        _subprocess_log.flush()
    return subprocess.run(
        cmd,
        stdout=_subprocess_log,
        stderr=_subprocess_log,
        check=check,
    )


def _gcs_ls(gcs_path: str) -> list[str]:
    """List immediate children under a GCS prefix (directory-like listing)."""
    if not gcs_path.endswith("/"):
        gcs_path += "/"
    result = subprocess.run(
        ["gcloud", "storage", "ls", gcs_path],
        capture_output=True, text=True, check=True,
    )
    return [line.strip().rstrip("/") for line in result.stdout.splitlines() if line.strip()]


def resolve_latest_step(gcs_path: str) -> int:
    """Find the latest (numerically highest) checkpoint step under a GCS root."""
    children = _gcs_ls(gcs_path)
    steps: list[int] = []
    for child in children:
        name = child.rsplit("/", 1)[-1]
        if name.isdigit():
            steps.append(int(name))
    if not steps:
        raise FileNotFoundError(
            f"No numeric step directories found under {gcs_path}. "
            "Make sure --gcs-path points to the checkpoint root "
            "(e.g. gs://bucket/checkpoints/run-name)."
        )
    latest = max(steps)
    log.info("Available steps: %s — using latest: %d", sorted(steps), latest)
    return latest


def download_checkpoint(
    gcs_path: str,
    local_dir: str,
    step: int,
) -> str:
    """Download a checkpoint step from GCS to a local directory.

    Uses ``gcloud storage rsync -r`` so that interrupted downloads can be
    resumed without re-transferring already-completed files.

    Returns the local path to the downloaded step directory.
    """
    gcs_step_path = f"{gcs_path.rstrip('/')}/{step}"
    local_step_path = os.path.join(local_dir, str(step))
    os.makedirs(local_step_path, exist_ok=True)

    log.info("[step %d] Syncing %s -> %s ...", step, gcs_step_path, local_step_path)
    t0 = time.time()
    _run(["gcloud", "storage", "rsync", "-r", gcs_step_path, local_step_path])
    elapsed = time.time() - t0
    log.info("[step %d] Download completed in %.1fs", step, elapsed)

    return local_step_path


# ── Conversion ───────────────────────────────────────────────────────────────


def convert_to_hf(
    megatext_model_path: str,
    output_dir: str,
    hf_config_path: str,
    scan_layers: bool = True,
    hf_token: str | None = None,
    tokenizer_path: str | None = None,
) -> str:
    """Run the megatext-to-hf conversion."""
    log.info("Converting Megatext checkpoint -> HuggingFace format...")
    log.info("  megatext_model_path = %s", megatext_model_path)
    log.info("  output_dir          = %s", output_dir)
    log.info("  hf_config_path      = %s", hf_config_path)
    log.info("  scan_layers         = %s", scan_layers)
    if tokenizer_path:
        log.info("  tokenizer_path      = %s", tokenizer_path)

    from megatext.conversion.convert import megatext_to_hf

    t0 = time.time()
    result = megatext_to_hf(
        megatext_model_path=megatext_model_path,
        output_dir=output_dir,
        hf_config_path=hf_config_path,
        scan_layers=scan_layers,
        hf_token=hf_token,
        tokenizer_path=tokenizer_path,
    )
    elapsed = time.time() - t0
    log.info("Conversion completed in %.1fs", elapsed)
    return result


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download Megatext checkpoint(s) from GCS and convert to "
            "HuggingFace format. Multiple steps are pipelined: the next "
            "download runs while the current step is being converted."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Source
    parser.add_argument(
        "--gcs-path",
        required=True,
        help=(
            "GCS checkpoint root path (e.g. gs://bucket/checkpoints/run-name). "
            "The tool will look for numeric step subdirectories under this path."
        ),
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        nargs="+",
        required=True,
        help=(
            "Checkpoint step(s) to process. Multiple values are pipelined: "
            "download and conversion overlap across steps "
            "(e.g. --checkpoint-step 4000 6000 8000). "
            "Use -1 for the latest available step."
        ),
    )

    # Download
    parser.add_argument(
        "--local-dir",
        default=None,
        help=(
            "Local directory to download the checkpoint into. "
            "Defaults to <project-root>/.cache/megatext-checkpoints/<run-name>."
        ),
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip the GCS download step (use an existing local checkpoint).",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete the downloaded Megatext checkpoint after conversion to save disk space.",
    )

    # Conversion
    parser.add_argument(
        "--hf-config-path",
        required=True,
        help=(
            "Pre-prepared HF model/config template path or HF hub ID. "
            "Used for the runtime HF config/model class and tokenizer artifacts. "
            "(e.g. checkpoints/config/Qwen3-SWA-8B or Qwen/Qwen3-8B)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory for converted HF checkpoint(s). "
            "Defaults to <project-root>/checkpoints/<run-name>."
        ),
    )
    parser.add_argument(
        "--scan-layers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether the checkpoint uses scanned layers (default: True).",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace access token (default: $HF_TOKEN env var).",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help=(
            "Optional path or HF hub ID to load the tokenizer from. "
            "If omitted, the tokenizer is copied from --hf-config-path."
        ),
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="Skip the conversion step (download only).",
    )

    return parser


def _patch_pwd() -> None:
    """Patch pwd.getpwuid/grp.getgrgid for environments with unknown UIDs.

    Shared filesystems may have files owned by UIDs not in /etc/passwd.
    Libraries like etils call pwd.getpwuid(st.st_uid) which raises KeyError.
    """
    import grp
    import pwd
    import types

    _orig_pw = pwd.getpwuid
    _orig_gr = grp.getgrgid

    def _pw(uid):
        try:
            return _orig_pw(uid)
        except KeyError:
            return types.SimpleNamespace(
                pw_name=str(uid), pw_uid=uid, pw_gid=uid,
                pw_gecos="", pw_dir="/tmp", pw_shell="/bin/bash",
            )

    def _gr(gid):
        try:
            return _orig_gr(gid)
        except KeyError:
            return types.SimpleNamespace(gr_name=str(gid), gr_gid=gid, gr_mem=[])

    pwd.getpwuid = _pw
    grp.getgrgid = _gr


def main() -> None:
    _patch_pwd()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = build_parser()
    args = parser.parse_args()

    _init_logs()

    gcs_path: str = args.gcs_path.rstrip("/")
    # e.g. gs://bucket/.../pretrain-qwen3-swa-8b/checkpoints
    #   -> gcs_parent = "pretrain-qwen3-swa-8b", gcs_leaf = "checkpoints"
    gcs_parent_path, gcs_leaf = gcs_path.rsplit("/", 1)
    run_name = gcs_parent_path.rsplit("/", 1)[-1]

    steps = args.checkpoint_step
    if -1 in steps:
        latest = resolve_latest_step(gcs_path)
        steps = [latest if s == -1 else s for s in steps]
        log.info("Resolved -1 to latest step: %d", latest)
    log.info("Steps to process: %s", steps)

    default_cache = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ".cache", "megatext-checkpoints",
    )
    local_dir = args.local_dir or os.path.join(default_cache, run_name, gcs_leaf)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = args.output_dir or os.path.join(project_root, "checkpoints", run_name)

    _SENTINEL = None  # signals that the download thread is done

    # Queue: download thread produces (step, local_path | Exception),
    #        main thread consumes and runs conversion sequentially.
    ready: queue.Queue[tuple[int, str | Exception] | None] = queue.Queue()
    abort = threading.Event()

    def _download_worker() -> None:
        """Download steps one by one, pushing each to the queue as it finishes."""
        consecutive_failures = 0
        for s in steps:
            if abort.is_set():
                ready.put((s, RuntimeError("Aborted due to previous failures.")))
                continue

            local_step_path = os.path.join(local_dir, str(s))
            if args.skip_download:
                if not os.path.isdir(local_step_path):
                    ready.put((s, FileNotFoundError(
                        f"--skip-download was set but {local_step_path} does not exist."
                    )))
                    consecutive_failures += 1
                else:
                    log.info("[step %d] Skipping download, using %s", s, local_step_path)
                    ready.put((s, local_step_path))
                    consecutive_failures = 0
            else:
                try:
                    path = download_checkpoint(gcs_path, local_dir, s)
                    ready.put((s, path))
                    consecutive_failures = 0
                except Exception as exc:
                    log.error("[step %d] Download failed (see %s for details)", s, _subprocess_log_path)
                    ready.put((s, exc))
                    consecutive_failures += 1

            if consecutive_failures >= 3:
                log.error("3 consecutive download failures — aborting remaining steps.")
                abort.set()

        ready.put(_SENTINEL)

    # Start download thread
    dl_thread = threading.Thread(target=_download_worker, daemon=True)
    dl_thread.start()

    # Main thread: consume from queue, convert sequentially
    results: dict[int, str] = {}
    errors: dict[int, Exception] = {}

    while True:
        item = ready.get()
        if item is _SENTINEL:
            break

        s, path_or_exc = item

        if isinstance(path_or_exc, Exception):
            errors[s] = path_or_exc
            continue

        local_path = path_or_exc

        if args.skip_convert:
            results[s] = local_path
            log.info("[step %d] Skipping conversion.", s)
            continue

        step_output_dir = (
            os.path.join(output_dir, str(s)) if (args.output_dir is None or len(steps) > 1)
            else output_dir
        )
        try:
            convert_to_hf(
                megatext_model_path=local_path,
                output_dir=step_output_dir,
                hf_config_path=args.hf_config_path,
                scan_layers=args.scan_layers,
                hf_token=args.hf_token,
                tokenizer_path=args.tokenizer_path,
            )
            results[s] = step_output_dir
            log.info("[step %d] HF checkpoint saved to: %s", s, step_output_dir)
        except Exception as exc:
            errors[s] = exc
            log.error("[step %d] Conversion failed: %s", s, exc, exc_info=True)

        # Cleanup only after successful conversion to free disk for next step
        if args.cleanup and not args.skip_download and s in results:
            log.info("[step %d] Cleaning up %s", s, local_path)
            shutil.rmtree(local_path, ignore_errors=True)

    dl_thread.join()

    # ── Summary ─────────────────────────────────────────────────────────
    log.info("=" * 60)
    for s in sorted(results):
        log.info("  step %d -> %s", s, results[s])
    if errors:
        log.error("Failed steps:")
        for s in sorted(errors):
            log.error("  step %d: %s", s, errors[s])
        sys.exit(1)

    if _subprocess_log:
        _subprocess_log.close()
        log.info("Full subprocess log: %s", _subprocess_log_path)
    log.info("All done!")


if __name__ == "__main__":
    main()
