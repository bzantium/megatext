# Copyright 2023-2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Debug utilities for inspecting model shardings and config."""

import jax
import jax.numpy as jnp
import numpy as np
import psutil
from packaging.version import Version

from megatext.utils import logging as max_logging
from megatext.utils import sharding
from megatext.utils.logging import add_text_to_summary_writer


def print_shardings_params(params, params_sharding, mesh, logical_annotations=None):
  """
  Print state shardings comparing Logical Definition vs Physical Result.
  """
  if not hasattr(params, "params"):
    params = {"params": params}
  if not hasattr(params_sharding, "params"):
    params_sharding = {"params": params_sharding}
  if logical_annotations and not hasattr(logical_annotations, "params"):
    logical_annotations = {"params": logical_annotations}

  leaves_params, _ = jax.tree_util.tree_flatten_with_path(params)
  leaves_sharding, _ = jax.tree_util.tree_flatten_with_path(params_sharding)
  leaves_logical, _ = jax.tree_util.tree_flatten_with_path(logical_annotations)

  for (path, leaf_val), (_, leaf_sharding), (_, leaf_logical_val) in zip(leaves_params, leaves_sharding, leaves_logical):
    path_str = "/".join(str(p.key if hasattr(p, "key") else p.name) for p in path)
    shape = jax.typeof(leaf_val)
    pspec = sharding.remove_size_one_mesh_axis(leaf_sharding.spec, mesh)
    pspec_str = str(tuple(pspec))
    logical_str = str(leaf_logical_val)

    message = f" {path_str}\n" f"    Shape:     {shape}\n" f"    Logical:   {logical_str}\n" f"    Physical:  {pspec_str}"
    max_logging.info(message)

  print(flush=True)


def add_config_to_summary_writer(config, summary_writer):
  """Writes config params to tensorboard"""
  if jax.process_index() == 0:
    for key, value in config.get_keys().items():
      add_text_to_summary_writer(key, str(value), summary_writer)


# ---------------------------------------------------------------------------
# Memory/debug utilities (moved from max_utils.py)
# ---------------------------------------------------------------------------


def with_memory_kind(t, memory_kind):
  return jax.tree_util.tree_map(lambda x: x.with_memory_kind(kind=memory_kind), t)


def calculate_num_params_from_pytree(params):
  params_sizes = jax.tree_util.tree_map(jax.numpy.size, params)
  total_parameters = jax.tree_util.tree_reduce(lambda x, y: x + y, params_sizes)
  assert total_parameters >= 0
  return total_parameters


def device_space():
  """Version guard for jax.memory.Space.Device."""
  # See b/436565838 for more.
  if Version(jax.__version__) >= Version("0.7.1"):
    return jax.memory.Space.Device  # pytype: disable=module-attr
  else:
    return jax._src.sharding_impls.TransferToMemoryKind("device")  # pylint: disable=protected-access # pytype: disable=module-attr


def calculate_total_params_per_chip(params):
  """Calculate total params per chip."""

  def calculate_leaf_params_per_chip(arr):
    shard = arr.addressable_shards[0]
    return np.prod(shard.data.shape)

  params_sizes_per_chip = jax.tree_util.tree_map(calculate_leaf_params_per_chip, params)
  total_parameters_per_chip = jax.tree_util.tree_reduce(lambda x, y: x + y, params_sizes_per_chip)
  return total_parameters_per_chip


def _bytes_of(x):
  """Return the number of bytes used by a single leaf in a pytree.
  Handles concrete arrays (NumPy/JAX), abstract shapes, scalars, and None.
  Unknown types default to 0.
  """
  # Abstract JAX values: compute bytes from shape x dtype size.
  if isinstance(x, jax.ShapeDtypeStruct):
    # jnp.dtype() normalizes to a consistent dtype object (e.g., handles bfloat16)
    return int(np.prod(x.shape)) * int(jnp.dtype(x.dtype).itemsize)

  # Concrete arrays (NumPy, JAX): rely on their native nbytes property.
  if hasattr(x, "nbytes"):
    return int(x.nbytes)

  # Python scalars (int, float, bool): convert to a NumPy array to measure size.
  if isinstance(x, (int, float, bool)):
    return int(np.array(x).nbytes)

  # None or unsupported leaf types: count as zero bytes.
  if x is not None:
    max_logging.log(f"Unsupported leaf type in calculate_bytes_from_pytree: {type(x)}")

  return 0


def calculate_bytes_from_pytree(params):
  """Return the total memory footprint (in bytes) of all leaves in a pytree.

  Each leaf is measured using `_bytes_of`. Non-array or unsupported types
  contribute 0 unless they are scalars.
  """
  return sum(map(_bytes_of, jax.tree_util.tree_leaves(params)))


def summarize_size_from_pytree(params):
  num_params = calculate_num_params_from_pytree(params)
  num_bytes = calculate_bytes_from_pytree(params)
  return num_params, num_bytes, num_bytes / num_params


def summarize_pytree_data(params, name="Params", raw=False):
  """Generate basic metrics of a given Pytree."""
  num_params, total_param_size, avg_param_size = summarize_size_from_pytree(params)
  if not raw:
    num_params_in_billions = num_params / 1e9
    total_param_size_in_gb = total_param_size / 1e9
    print(
        f"{name} stats: \n"
        f"\tTotal number of params: {num_params_in_billions:.3f} billion \n"
        f"\tTotal memory usage: {total_param_size_in_gb:.3f} GB \n"
        f"\tAvg size: {avg_param_size:.3f} bytes\n"
    )
  else:
    print(
        f"{name} stats: \n"
        f"\tTotal number of params: {num_params:.3f} \n"
        f"\tTotal memory usage: {total_param_size:.3f} bytes \n"
        f"\tAvg size: {avg_param_size:.3f} bytes\n"
    )
  return num_params, total_param_size, avg_param_size


def print_mem_stats(label: str):
  max_logging.log(f"\nMemstats: {label}:")
  try:
    for d in jax.local_devices():
      stats = d.memory_stats()
      used = round(stats["bytes_in_use"] / 2**30, 2)
      limit = round(stats["bytes_limit"] / 2**30, 2)
      max_logging.log(f"\tUsing (GB) {used} / {limit} ({used/limit:%}) on {d}")
  except (RuntimeError, KeyError, TypeError) as ex:
    max_logging.log(f"\tMemstats unavailable, error: {ex}")


def print_cpu_ram_stats(label: str):
  """Print stats of CPU RAM usage/availability."""
  max_logging.log(f"\nRAMstats: {label}:")
  try:
    ram = psutil.virtual_memory()

    total = round(ram.total / 2**30, 2)
    available = round(ram.available / 2**30, 2)
    used = round(ram.used / 2**30, 2)

    max_logging.log(f"\tUsing (GB) {used} / {total} ({used/total:%}) -->  Available:{available}")
  except (RuntimeError, KeyError, TypeError) as ex:
    max_logging.log(f"\tRAM stats unavailable, error: {ex}")


def print_compiled_memory_stats(compiled_stats):
  """Prints a summary of the compiled memory statistics."""
  if compiled_stats is None:
    return

  def bytes_to_gb(num_bytes):
    return num_bytes / (1024**3)

  output_gb = bytes_to_gb(compiled_stats.output_size_in_bytes)
  temp_gb = bytes_to_gb(compiled_stats.temp_size_in_bytes)
  argument_gb = bytes_to_gb(compiled_stats.argument_size_in_bytes)
  alias_gb = bytes_to_gb(compiled_stats.alias_size_in_bytes)
  host_temp_gb = bytes_to_gb(compiled_stats.host_temp_size_in_bytes)
  total_gb = output_gb + temp_gb + argument_gb - alias_gb

  max_logging.log(
      f"Total memory size: {total_gb:.1f} GB, Output size: {output_gb:.1f} GB, Temp size: {temp_gb:.1f} GB, "
      f"Argument size: {argument_gb:.1f} GB, Host temp size: {host_temp_gb:.1f} GB."
  )


def print_system_information():
  """Print system information of the current environment.
  Note that this will initialize the JAX backend."""
  max_logging.log(f"System Information: Jax Version: {jax.__version__}")
  max_logging.log(f"System Information: Jaxlib Version: {jax.lib.__version__}")
  max_logging.log(f"System Information: Jax Backend: {jax.extend.backend.get_backend().platform_version}")


def print_non_trivial_mesh_axis(mesh):
  """Print mesh axis if its axis size is larger than one."""
  for mesh_axis, axis_size in mesh.shape.items():
    if axis_size > 1:
      print(f"{mesh_axis}: {axis_size}", flush=True)


def print_pytree_shape(print_str, ptree):
  print("\n")
  print(print_str)
  print(jax.tree_util.tree_map(lambda x: x.shape, ptree))


def print_model_vars(print_str, model_vars):
  for k in model_vars:
    print(f"{print_str} key{k}:")
    print(f"\t {model_vars[k]}")
