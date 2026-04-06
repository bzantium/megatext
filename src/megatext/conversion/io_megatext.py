"""Orbax checkpoint I/O for Megatext format.

Saves/loads flat param dicts as orbax checkpoints compatible with Megatext's
training loop. No Megatext config dependency.
"""

from __future__ import annotations

import os

from etils import epath
import jax
import numpy as np
import orbax.checkpoint as ocp

from megatext.utils import logging as max_logging


def _nest_dict(flat: dict[str, np.ndarray]) -> dict:
    """Nest flat dict by splitting keys on '-'.

    "params-decoder-layers-mlp-wi_0-kernel" →
    {"params": {"decoder": {"layers": {"mlp": {"wi_0": {"kernel": array}}}}}}
    """
    nested: dict = {}
    for key, value in flat.items():
        parts = key.split("-")
        d = nested
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return nested


def _flatten_dict(nested: dict, prefix: str = "") -> dict:
    """Flatten nested dict to flat dict with '-'-joined keys."""
    flat: dict = {}
    for key, value in nested.items():
        full_key = f"{prefix}-{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_dict(value, full_key))
        else:
            flat[full_key] = value
    return flat


def _swap_scan_axis(
    weights: dict[str, np.ndarray], src: int, dst: int,
) -> dict[str, np.ndarray]:
    """Move the scan (layers) axis between converter order and Megatext order.

    Megatext uses param_scan_axis=1; our converter stacks at axis 0.
    Scanned params are identified by '-layers-' in their flat key.
    """
    return {
        k: np.moveaxis(np.asarray(v), src, dst) if "-layers-" in k and v.ndim >= 2 else np.asarray(v)
        for k, v in weights.items()
    }


def _add_root_prefix(weights: dict[str, np.ndarray], prefix: str) -> dict[str, np.ndarray]:
    return {f"{prefix}-{key}": value for key, value in weights.items()}


def save_megatext_checkpoint(
    weights: dict[str, np.ndarray],
    output_dir: str,
    *,
    use_ocdbt: bool = True,
    use_zarr3: bool = True,
) -> str:
    """Save flat param dict as orbax checkpoint.

    Structure matches Megatext's create_orbax_checkpoint_manager():
    item_names=("items",) with PyTreeCheckpointHandler, producing
    output_dir/0/items/... so Megatext can restore via load_params_from_path.

    Returns:
        Path to the checkpoint directory (output_dir itself).
    """
    nested_params = _nest_dict(_swap_scan_axis(weights, src=0, dst=1))

    train_state = {
        "step": np.int32(0),
        "params": nested_params,
    }

    train_state_jax = jax.tree.map(
        lambda x: jax.numpy.asarray(x) if isinstance(x, np.ndarray) else x,
        train_state,
    )

    os.makedirs(output_dir, exist_ok=True)

    options = ocp.CheckpointManagerOptions(max_to_keep=1)
    mgr = ocp.CheckpointManager(
        output_dir,
        item_names=("items",),
        item_handlers={"items": ocp.PyTreeCheckpointHandler()},
        options=options,
    )

    mgr.save(
        0,
        args=ocp.args.Composite(items=ocp.args.PyTreeSave(train_state_jax)),
    )
    mgr.wait_until_finished()

    max_logging.log(f"Saved checkpoint to {output_dir}")
    return output_dir


def load_megatext_checkpoint(checkpoint_path: str, step: int | None = None) -> dict[str, np.ndarray]:
  """Load orbax checkpoint → flat dict with '-'-joined keys.

  Args:
    checkpoint_path: Path to checkpoint directory (parent of step dirs).
    step: Specific checkpoint step to restore. If None, restores the latest step.

  Returns:
    Flat dict mapping Megatext param keys to numpy arrays.
  """
  mgr = ocp.CheckpointManager(
      checkpoint_path,
      item_names=("items",),
      item_handlers={"items": ocp.PyTreeCheckpointHandler()},
  )
  restore_step = mgr.latest_step() if step is None else step
  if restore_step is None:
    raise FileNotFoundError(f"No checkpoint found in {checkpoint_path}")

  return load_megatext_checkpoint_params(
      os.path.join(checkpoint_path, str(restore_step), "items"),
  )


def _cpu_abstract_from_metadata(tree):
  """Convert Orbax metadata tree to a CPU-restorable abstract pytree."""
  try:
    cpu_device = jax.devices("cpu")[0]
  except (RuntimeError, ValueError, IndexError):
    cpu_device = jax.devices()[0]
  cpu_sharding = jax.sharding.SingleDeviceSharding(cpu_device)
  if isinstance(tree, dict):
    return {key: _cpu_abstract_from_metadata(value) for key, value in tree.items()}
  if hasattr(tree, "shape") and hasattr(tree, "dtype"):
    return jax.ShapeDtypeStruct(shape=tree.shape, dtype=tree.dtype, sharding=cpu_sharding)
  return tree


def load_megatext_checkpoint_params(items_path: str) -> dict[str, np.ndarray]:
  """Load params subtree from a Megatext Orbax checkpoint step onto CPU."""
  checkpoint_path = epath.Path(items_path)
  ckptr = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
  metadata = ckptr.metadata(checkpoint_path)
  abstract_params = _cpu_abstract_from_metadata(metadata.item_metadata.tree["params"])
  restore_args = ocp.checkpoint_utils.construct_restore_args(abstract_params)
  restored = ckptr.restore(
      checkpoint_path,
      item={"params": abstract_params},
      transforms={},
      restore_args={"params": restore_args},
  )

  params = restored["params"]
  if isinstance(params, dict) and "params" in params:
    params = params["params"]

  return _add_root_prefix(_swap_scan_axis(_flatten_dict(params), src=1, dst=0), "params")
