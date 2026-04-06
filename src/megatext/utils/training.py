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

"""Training-related utilities for intermediate values and jaxpr debugging."""

import argparse
import functools
from functools import partial
from contextlib import contextmanager
import os

import flax
import jax
from jax.sharding import PartitionSpec as P
import jax.numpy as jnp
import numpy as np

from megatext.common.common_types import MODEL_MODE_PREFILL, MODEL_MODE_AUTOREGRESSIVE, MODEL_MODE_TRAIN
from megatext.utils import storage as gcs_utils
from megatext.utils import logging as max_logging


OVERWRITE_WITH_GRADIENT = "_overwrite_with_gradient"


def get_nested_value(dictionary, nested_key, default=None):
  """
  Retrieves a value from a nested key in a dictionary.

  Args:
      dictionary: The dictionary to search in.
      nested_key: A tuple representing the nested key, e.g., ('level1', 'level2', 'key').
      default: The value to return if the nested key is not found.

  Returns:
      The value associated with the nested key, or the default value if not found.
  """
  current_level = dictionary

  for key in nested_key:
    if not isinstance(current_level, dict) or key not in current_level:
      return default
    current_level = current_level[key]
  return current_level


def get_intermediate_value(model, nested_key, default=None, clear=False):
  """
  Retrieves an intermediate value from an NNX model. This functions has context about
  where the intermediate value is located.

  Args:
    model: The NNX model.
    nested_key: A string representing the nested key, e.g., hidden_states_norm_out
    default: The value to return if the nested key is not found.
    clear: Clears the intermediate value from the model.

  Returns:
    The value associated with the nested key, or the default value if not found.
  """
  intermediate_value = default
  match nested_key:
    case "out_projection_activations":
      if nested_key in model.decoder.layers["self_attention"]:
        intermediate_value = model.decoder.layers["self_attention"][nested_key].get_value()[-1]
        if clear:
          del model.decoder.layers["self_attention"][nested_key]
    case _:
      # Default case to handle any unknown nested keys
      raise ValueError(f"Incorrect nested_key: {nested_key}")

  return intermediate_value


def maybe_dump_jaxpr(config, p_train_step, train_step_inputs):
  """Dump jaxpr to local then upload to GCS."""
  if not config.dump_jaxpr:
    return
  max_logging.log("Tracing train_step to jaxpr...")

  # We use the p_train_step (the JIT-decorated function)
  p_train_jaxpr = jax.make_jaxpr(p_train_step)(*train_step_inputs)

  local_filename = "train_step.jaxpr"
  local_path = os.path.join(config.dump_jaxpr_local_dir, local_filename)

  os.makedirs(config.dump_jaxpr_local_dir, exist_ok=True)

  # pylint: disable=unspecified-encoding
  with open(local_path, "w") as f:
    f.write(str(p_train_jaxpr))

  max_logging.log(f"Jaxpr dumped locally to {local_path}")

  if config.dump_jaxpr_gcs_dir:
    gcs_utils.upload_dump(
        config.dump_jaxpr_local_dir,
        config.dump_jaxpr_gcs_dir,
        module_name=local_filename,
        delete_local_after=config.dump_jaxpr_delete_local_after,  # Keeping local for debugging
        all_host_upload=False,  # Only upload from lead host (Host 0)
    )


# ---------------------------------------------------------------------------
# Training utilities (moved from max_utils.py)
# ---------------------------------------------------------------------------


def str2bool(v):
  """Parses a string representation of a boolean value into a Python boolean."""
  if isinstance(v, bool):
    return v
  if v.lower() == "true":
    return True
  elif v.lower() == "false":
    return False
  else:
    raise argparse.ArgumentTypeError("Boolean value expected (e.g., True or False).")


# Cross entropy implementation is taken from original T5X codebase:
# https://github.com/google-research/t5x/blob/ace831eea1e2742b4299cd1a9af7e4f302038351/t5x/losses.py#L25-L101
@jax.custom_vjp
def cross_entropy_with_logits(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    z_loss: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Computes cross entropy loss with stable custom gradient.
  Computes a stabilized-gradient version of:
    -jnp.sum(targets * nn.log_softmax(logits), axis=-1)
  If z_loss > 0, then an auxiliary loss equal to z_loss*log(z)^2
  will be added to the cross entropy loss (z = softmax normalization constant).
  The two uses of z_loss are:
  1. To keep the logits from drifting too far from zero, which can cause
     unacceptable roundoff errors in bfloat16.
  2. To encourage the logits to be normalized log-probabilities.
  Args:
    logits: [batch, length, num_classes] float array.
    targets: categorical one-hot targets [batch, length, num_classes] float
      array.
    z_loss: coefficient for auxiliary z-loss loss term.
  Returns:
    tuple with the total loss and the z_loss, both
    float arrays with shape [batch, length].
  """
  logits_sum = jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
  log_softmax = logits - logits_sum
  loss = -jnp.sum(targets * log_softmax, axis=-1)
  # Add auxiliary z-loss term.
  log_z = jnp.squeeze(logits_sum, axis=-1)
  total_z_loss = z_loss * jax.lax.square(log_z)
  loss += total_z_loss
  return loss, total_z_loss


def _cross_entropy_with_logits_fwd(logits: jnp.ndarray, targets: jnp.ndarray, z_loss: float = 0.0) -> tuple[
    tuple[jnp.ndarray, jnp.ndarray],
    tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ],
]:
  """Forward-mode of `cross_entropy_with_logits`."""
  max_logit = logits.max(axis=-1, keepdims=True)
  shifted = logits - max_logit
  exp_shifted = jnp.exp(shifted)
  sum_exp = jnp.sum(exp_shifted, axis=-1, keepdims=True)
  log_softmax = shifted - jnp.log(sum_exp)
  loss = -jnp.sum(targets * log_softmax, axis=-1)
  # Add auxiliary z-loss term.
  log_z = jnp.squeeze(jnp.log(sum_exp) + max_logit, axis=-1)
  total_z_loss = z_loss * jax.lax.square(log_z)
  loss += total_z_loss
  return (loss, total_z_loss), (
      logits,
      targets,
      z_loss,
      exp_shifted,
      sum_exp,  # pytype: disable=bad-return-type  #jax-ndarray
      log_z,
  )


def _cross_entropy_with_logits_bwd(
    res: tuple[
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
        jnp.ndarray,
    ],
    g: tuple[jnp.ndarray, jnp.ndarray],
) -> tuple[jnp.ndarray, None, None]:
  """Backward-mode of `cross_entropy_with_logits`."""
  g = g[0]  # Ignore z_loss component as that is only used for logging.
  logits, targets, z_loss, exp_shifted, sum_exp, log_z = res
  # z-loss term adds the (2 * z_loss * log_z) factor.
  deriv = jnp.expand_dims(1 + 2 * z_loss * log_z, -1) * exp_shifted / sum_exp - targets
  g_logits = jnp.expand_dims(g, axis=-1) * deriv

  return (
      jnp.asarray(g_logits, logits.dtype),
      None,  # we don't need gradients on targets
      None,  # we don't need gradients on z_loss
  )  # sets z-loss coeff gradient to 0


cross_entropy_with_logits.defvjp(_cross_entropy_with_logits_fwd, _cross_entropy_with_logits_bwd)


def l2norm_pytree(x):
  """L2 norm of a pytree of arrays."""
  return jnp.sqrt(jax.tree_util.tree_reduce(lambda total, leaf: total + jnp.sum(jnp.square(leaf)), x, initializer=0.0))


def get_batch_seq_len_for_mode(config, model_mode):
  """
  Resolves the batch size and sequence length based on the model's operational mode.

  Args:
    config: A configuration object with model parameters.
    model_mode: The current operational mode
                (e.g., PREFILL, AUTOREGRESSIVE, TRAIN).

  Returns:
    A tuple of (batch_size, seq_len).
  """
  if model_mode == MODEL_MODE_PREFILL:
    batch_size = 1
    seq_len = config.max_prefill_predict_length

  elif model_mode == MODEL_MODE_AUTOREGRESSIVE:
    batch_size = config.micro_batch_size_to_train_on
    seq_len = 1

  elif model_mode == MODEL_MODE_TRAIN:
    batch_size = config.micro_batch_size_to_train_on
    seq_len = config.max_target_length

  else:
    raise ValueError(f"Unknown model_mode: {model_mode}")

  return batch_size, seq_len


@contextmanager
def maybe_get_transformer_engine_context(config):
  """Runs a transformer engine context engine manager for GPUs only."""
  if config.hardware in ["gpu", "gpu_multiprocess"]:
    with transformer_engine_context():
      yield
  else:
    with dummy_context_manager():
      yield


@contextmanager
def dummy_context_manager():
  """A context manager that does nothing."""
  yield


@contextmanager
def transformer_engine_context():
  """If TransformerEngine is available, this context manager will provide
  the library with Megatext-specific details needed for correcct operation."""
  try:
    from transformer_engine.jax.sharding import global_shard_guard, MeshResource  # pylint: disable=import-outside-toplevel
    mesh_resource = MeshResource(  # pytype: disable=wrong-arg-types
        dp_resource="data",
        tp_resource="tensor",
        fsdp_resource="fsdp",
        pp_resource=None,
        cp_resource="context",
    )
    with global_shard_guard(mesh_resource):
      yield
  except (ImportError, AttributeError):
    yield


def unbox_logicallypartioned(boxed_pytree):
  """Unboxes the flax.LogicallyPartitioned pieces

  Args:
    boxed_pytree: a pytree that includes LogicallyPartitioned
      leaves.
  Returns:
    a pytree where all all LogicallyPartitioned leaves have been unboxed.
  """
  return jax.tree_util.tree_map(
      lambda x: x.unbox() if isinstance(x, flax.linen.spmd.LogicallyPartitioned) else x,
      boxed_pytree,
      is_leaf=lambda k: isinstance(k, flax.linen.spmd.LogicallyPartitioned),
  )


def permute_to_match_megatext_rope(arr):
  """Permutes the Huggingface Rope to match the Megatext logic."""
  assert arr.shape[-1] % 2 == 0, "The last dimension for rope has to be even."
  evens, odds = np.split(arr, 2, axis=arr.ndim - 1)  # pylint: disable=W0632
  x = np.empty_like(arr)
  x[..., ::2] = evens
  x[..., 1::2] = odds
  return x


def unpermute_from_match_megatext_rope(arr, model_size):
  """
  Function to get the RoPE values in correct ordering
  """
  if model_size[:8] != "llama3.1":
    return arr
  evens = arr[..., ::2]
  odds = arr[..., 1::2]
  return jax.numpy.concatenate((evens, odds), axis=arr.ndim - 1)


@partial(jax.jit, static_argnames=("cp_size", "seq_dim", "to_contiguous"))
def reorder_sequence(tensor, cp_size: int, seq_dim: int = 1, to_contiguous: bool = False):
  """Reorders the sequence of the tensor. For example, with cp_size=2,
  [0, 1, 2, 3, 4, 5, 6, 7] -> [0, 1, 6, 7, 2, 3, 4, 5]
  and backward
  [0, 1, 6, 7, 2, 3, 4, 5] -> [0, 1, 2, 3, 4, 5, 6, 7]
  """

  if tensor is None:
    return tensor

  seq_len = tensor.shape[seq_dim]
  group_size = seq_len // (2 * cp_size)

  if cp_size % 2 != 0:
    raise ValueError(f"{cp_size=} must be a multiple of 2.")

  if seq_len % (cp_size * 2) != 0:
    raise ValueError(f"{tensor.shape=} is not a multiple of {cp_size*2=}")

  ori_tensor_shape = tensor.shape
  reshaped = tensor.reshape(
      *ori_tensor_shape[:seq_dim],
      2 * cp_size,
      group_size,
      *ori_tensor_shape[seq_dim + 1 :],
  )

  if not to_contiguous:
    first_half = jnp.arange(cp_size)
    second_half = jnp.arange(2 * cp_size - 1, cp_size - 1, -1)
    src_indices = jnp.stack([first_half, second_half], axis=1).reshape(-1)

  else:
    half = cp_size // 2
    first_pair = [4 * r for r in range(half)]
    second_pair = [4 * r + 2 for r in range(half)]
    third_pair = [2 * cp_size - 1 - 4 * r for r in range(half)]
    fourth_pair = [i - 2 for i in third_pair]
    first_block = first_pair + third_pair
    second_block = second_pair + fourth_pair
    src_indices = jnp.stack([jnp.array(first_block), jnp.array(second_block)], axis=1).reshape(-1)

  reordered = jnp.take(reshaped, src_indices, axis=seq_dim)
  return reordered.reshape(ori_tensor_shape)


@partial(jax.jit, static_argnums=1)
def reorder_causal_load_balanced(batch, cp_size):
  """Reorders the example batch sequences"""
  return {
      key: reorder_sequence(
          value,
          cp_size=cp_size,
      )
      if key
      in ["inputs", "targets", "inputs_position", "targets_position", "inputs_segmentation", "targets_segmentation"]
      else value
      for key, value in batch.items()
  }


@staticmethod
def reorder_mask_load_balancing(tensor, cp_size: int, seq_dim: int):
  """
  Reorders a tensor for load balancing the compute of causal attention.
  This function works on numpy arrays instead of jax.numpy arrays.
  """

  seq_len = tensor.shape[seq_dim]
  group_size = seq_len // (2 * cp_size)

  if cp_size % 2 != 0:
    raise ValueError(f"{cp_size=} must be a multiple of 2.")

  if seq_len % (cp_size * 2) != 0:
    raise ValueError(f"{tensor.shape=} is not a multiple of {cp_size*2=}")

  ori_tensor_shape = tensor.shape
  reshaped = tensor.reshape(
      *ori_tensor_shape[:seq_dim],
      2 * cp_size,
      group_size,
      *ori_tensor_shape[seq_dim + 1 :],
  )

  first_half = np.arange(cp_size)
  second_half = np.arange(2 * cp_size - 1, cp_size - 1, -1)
  src_indices = np.stack([first_half, second_half], axis=1).reshape(-1)
  reordered = np.take(reshaped, src_indices, axis=seq_dim)
  return reordered.reshape(ori_tensor_shape)


def generate_representative_group_sizes(target_m: int, g: int) -> tuple[int, ...]:
  """Generate group sizes for a given target m."""
  np.random.seed(0)
  repr_val = np.random.uniform(size=(g,))
  repr_val = np.random.binomial(1, 0.9, (g,)) * repr_val
  repr_val = np.int32((repr_val / np.sum(repr_val)) * target_m)
  repr_val[0] += target_m - np.sum(repr_val)
  return tuple(map(int, repr_val))


def unscan_train_state_params(params, sharding, mesh, scan_axis, layer_groups):
  """
  Unrolls scanned parameter groups into per-layer entries.

  Args:
    train_state: training state with scanned `params`
    mesh: the mesh to use for sharding output
    scan_axis: axis along which scanning was applied (usually 0)
    layer_groups: list of tuples like:
      [("dense_layers", 4), ("moe_layers", 12)]
  """
  params_copy = params.unfreeze() if hasattr(params, "unfreeze") else params
  decoder = params_copy["params"]["decoder"]
  sharding_decoder = sharding["params"]["decoder"]

  def strip_scan_axis(pspec: P) -> P:
    """Removes the element at `scan_axis` from a PartitionSpec tuple."""
    spec_tuple = tuple(pspec)
    return P(*(spec_tuple[:scan_axis] + spec_tuple[scan_axis + 1 :]))

  for layer_name, num_layers in layer_groups:
    scanned_layers = decoder[layer_name]
    scanned_sharding = sharding_decoder[layer_name]

    unscanned_sharding_spec = jax.tree_util.tree_map(
        strip_scan_axis, jax.tree_util.tree_map(lambda x: x.spec, scanned_sharding)
    )
    unscanned_sharding = jax.tree_util.tree_map(lambda ps: jax.sharding.NamedSharding(mesh, ps), unscanned_sharding_spec)

    layer_pytrees = [
        jax.tree_util.tree_map(functools.partial(jnp.take, indices=i, axis=scan_axis), scanned_layers)
        for i in range(num_layers)
    ]

    for i, layer_params in enumerate(layer_pytrees):
      resharded_params = jax.device_put(layer_params, unscanned_sharding)
      decoder[f"{layer_name}_{i}"] = resharded_params

    del decoder[layer_name]


def rescan_train_state_params(params, source_shardings, scan_axis, layer_groups):
  """
  Reconstruct scanned layers from per-layer entries using minimal HBM.

  Args:
    train_state: training state with unrolled {layer_name}_{i} entries
    scan_axis: axis to scan over
    layer_groups: list of (layer_name, num_layers)
    mesh: jax.sharding.Mesh for out_shardings
  """
  decoder = params["params"]["decoder"]
  sharding = source_shardings["params"]["decoder"]

  for layer_name, num_layers in layer_groups:

    def stack_layers(*layers):
      return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=scan_axis), *layers)

    compiled_stack = jax.jit(
        stack_layers,
        out_shardings=sharding[layer_name],
    )

    layer_list = [decoder.pop(f"{layer_name}_{i}") for i in range(num_layers)]
    scanned = compiled_stack(*layer_list)
    decoder[layer_name] = scanned


def cast_dtype_from_to(nest, src, dst):
  """All items in nest with dtype src are casted to dtype dst."""
  return jax.tree_util.tree_map(lambda t: t.astype(dst) if t.dtype == src else t, nest)


def find_nans_and_infs(pytree):
  def finder(x):
    return jnp.any(jnp.isinf(x) | jnp.isnan(x))

  bad_pytree = jax.tree_util.tree_map(finder, pytree)
  return jax.tree_util.tree_flatten(bad_pytree)


def delete_pytree(p):
  def delete_leaf(leaf):
    if isinstance(leaf, jax.Array):
      leaf.delete()
    del leaf

  jax.tree_util.tree_map(delete_leaf, p)


def parse_custom_args(argv):
  """Load multiple YAML config files from command line arguments."""
  configs = []
  current_argv = []
  python_script = argv[0]
  for arg in argv[1:]:
    if arg.endswith((".yaml", ".yml")):
      if current_argv:
        configs.append(current_argv)
      current_argv = [python_script, arg]
    else:
      current_argv.append(arg)
  if current_argv:
    configs.append(current_argv)
  return configs


def should_prevent_cse_in_remat(config):
  """Determines whether to prevent common subexpression elimination (CSE) in remat.

  CSE should not be prevented when:
  1. Layers are being scanned (scan_layers=True), OR
  2. Gradient accumulation is enabled (gradient_accumulation_steps > 1) on GPU hardware
  """
  if config.scan_layers:
    return False
  if config.gradient_accumulation_steps > 1 and config.hardware in ("gpu", "gpu_multiprocess"):
    return False
  return True


def get_abstract_param(model, config):
  """Get abstract model structure (name, shape) without materializing the weights to save memory."""
  import numpy as np
  import jax.numpy as jnp
  from flax.linen import partitioning as nn_partitioning
  from megatext.multimodal import processor as mm_processor

  with model.mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    key = jax.random.PRNGKey(0)
    input_shape = (config.micro_batch_size_to_train_on, config.max_target_length)
    image_shape = mm_processor.get_dummy_image_shape_for_init(
        config.decoder_block, batch_size=config.micro_batch_size_to_train_on
    )
    audio_shape = mm_processor.get_dummy_audio_shape_for_init(config)
  abstract_vars = jax.eval_shape(
      model.init,
      {"params": key, "dropout": key, "aqt": key},
      jnp.ones(input_shape, dtype=jnp.int32),
      jnp.ones(input_shape, dtype=jnp.int32),
      encoder_images=np.ones(image_shape, dtype=jnp.int32) if config.use_multimodal else None,
      encoder_audios=np.ones(audio_shape, dtype=jnp.float32) if config.use_audio else None,
  )
  return abstract_vars
