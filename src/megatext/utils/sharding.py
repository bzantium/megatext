# Copyright 2025-2026 Google LLC
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

# pylint: disable=line-too-long, disable=bare-except, consider-using-generator
""" Utils that are only interesting to Megatext and sharding related. """

import collections
from collections.abc import Iterable, Sequence
import os
import socket
import time
from typing import Any

from etils import epath
from flax import linen as nn

import jax
from jax.core import Tracer
from jax.experimental import mesh_utils
from jax.sharding import PartitionSpec as P, NamedSharding, reshard

import numpy as np
import optax
import orbax.checkpoint as ocp
from orbax.checkpoint.experimental.emergency.multi_tier_checkpointing import initialization

from megatext.common.common_types import ShardMode
from megatext.utils import logging as max_logging
from megatext.utils.training import reorder_causal_load_balanced

import inspect  # for debugging only
from pathlib import Path

initialize_multi_tier_checkpointing = initialization.initialize_multi_tier_checkpointing
HYBRID_RING_64X4 = "hybrid_ring_64x4"
HYBRID_RING_32X8 = "hybrid_ring_32x8"

_LOGGED_ACTIVATION_SHARDINGS = set()
_ACTIVATION_SHARDINGS_DUMP = []


def clear_input_shardings_dump():
  """Clear the input shardings dump"""
  _LOGGED_ACTIVATION_SHARDINGS.clear()
  _ACTIVATION_SHARDINGS_DUMP.clear()


def get_input_data_sharding(config, mesh):
  """Get the input data sharding for the model"""
  return create_sharding(mesh, config.input_data_sharding_logical_axes, rules=config.logical_axis_rules)


def _get_sharding_desc(inputs, extra_stack_level):
  """Get the inputs sharding description using inspect module"""
  frame = inspect.currentframe()
  # Traverse back extra_stack_level times:
  for _ in range(1 + extra_stack_level):
    if frame is not None:
      frame = frame.f_back
  if frame is not None:
    callers_local_vars = frame.f_locals.items()

    x = [var_name for var_name, var_val in callers_local_vars if var_val is inputs]
    if len(x) > 0:
      caller_path_full = inspect.stack()[1 + extra_stack_level].filename
      # Use pathlib.Path to easily extract just the filename from the full path.
      caller_filename = Path(caller_path_full).name
      return f"{caller_filename[:-3]}/{x[0]}"
  return "Unknown"


def maybe_shard_with_name(
    inputs, named_sharding, shard_mode, debug_sharding=False, extra_stack_level=0, sharding_desc="", logical_axes=None
):
  """
  In auto shardmode, this function hints inputs follow given named_sharding.
  In explicit shardmode, this function enforces inputs following named_sharding.
  sharding_desc is description of inputs of upper layer(s) of caller (with the form of <filename>/<variable>).
   It is used as key in log/dump files when debug_sharding==true
  """
  if inputs is None:
    return None
  if (
      debug_sharding and isinstance(inputs, Tracer) and isinstance(named_sharding, NamedSharding)
  ):  # only print pspec for JitTracer
    if not sharding_desc:
      sharding_desc = _get_sharding_desc(inputs, extra_stack_level + 1)

    if not logical_axes:
      logical_axes = "Unknown"
    elif isinstance(logical_axes, list):
      logical_axes = tuple(logical_axes)

    pspec = remove_size_one_mesh_axis(getattr(named_sharding, "spec"), getattr(named_sharding, "mesh"))
    log_key = (sharding_desc, str(jax.typeof(inputs)), tuple(pspec), extra_stack_level)
    if log_key not in _LOGGED_ACTIVATION_SHARDINGS:
      max_logging.info(f"{sharding_desc} Logical: {log_key[1]:.<60} {logical_axes}.", stacklevel=3 + extra_stack_level)
      max_logging.info(f"{sharding_desc} Physical: {log_key[1]:.<60} {log_key[2]}.", stacklevel=3 + extra_stack_level)
      _LOGGED_ACTIVATION_SHARDINGS.add(log_key)

      _ACTIVATION_SHARDINGS_DUMP.append(
          {
              f"{sharding_desc}: {log_key[1]}": {
                  "logic_axes": f"{logical_axes}",
                  "PartitionSpec": f"P{log_key[2]}",
              }
          }
      )
  if shard_mode == ShardMode.EXPLICIT:
    return reshard(inputs, named_sharding)
  else:
    return jax.lax.with_sharding_constraint(inputs, named_sharding)


def maybe_shard_with_logical(
    inputs, logical_axes, mesh, shard_mode, rules=None, debug_sharding=False, extra_stack_level=0, sharding_desc=""
):
  """
  A wrapper of maybe_shard_with_name when logical axes are inputs
  sharding_desc is description of inputs of upper layer(s) of caller (with the form of <filename>/<variable>).
   It is used as key in log/dump files when debug_sharding==true
  """
  if inputs is None:
    return None

  if debug_sharding and not sharding_desc:
    sharding_desc = _get_sharding_desc(inputs, extra_stack_level + 1)

  named_sharding = create_sharding(mesh, logical_axes, rules=rules)

  return maybe_shard_with_name(
      inputs,
      named_sharding,
      shard_mode,
      debug_sharding=debug_sharding,
      extra_stack_level=extra_stack_level + 1,
      sharding_desc=sharding_desc,
      logical_axes=logical_axes,
  )


def remove_size_one_mesh_axis(spec, mesh):
  """
  Removes mesh axes from a PartitionSpec (P) where the axis size is 1.

  This is a common optimization to simplify sharding by excluding redundant axes.
  Function originally from jax._src.core:
  https://github.com/jax-ml/jax/blob/main/jax/_src/core.py
  """
  if spec is None:
    return None
  new_spec = []  # type: ignore
  for s in spec:
    if s is None or s == P.UNCONSTRAINED:
      new_spec.append(s)  # type: ignore
    elif isinstance(s, tuple):
      new_spec.append(tuple(i for i in s if mesh.shape.get(i, 1) != 1))
    else:
      new_spec.append(None if mesh.shape.get(s, 1) == 1 else s)  # type: ignore
  return P(*new_spec, unreduced=spec.unreduced, reduced=spec.reduced)


def logical_to_mesh_axes(logical_names, mesh, rules=None):
  """Remove size one mesh axes given logical names."""
  tensor_spec = nn.logical_to_mesh_axes(logical_names, rules=rules)
  return remove_size_one_mesh_axis(tensor_spec, mesh)


def logical_to_mesh(tree, mesh, rules=None):
  """Remove size one mesh axes given logical pspec pytree."""
  if tree is None:
    return None
  return jax.tree.map(
      lambda x: logical_to_mesh_axes(x, mesh, rules=rules),
      tree,
      is_leaf=lambda x: isinstance(x, P),
  )


def logical_to_mesh_sharding(tree, mesh, rules=None):
  """Return sharding pytree given logical specs pytree"""
  if tree is None:
    return None
  return jax.tree.map(
      lambda x: NamedSharding(mesh, x),
      logical_to_mesh(tree, mesh, rules=rules),
      is_leaf=lambda x: isinstance(x, P),
  )


def create_sharding(mesh, logical_names, rules=None):
  """Create NamedSharding with given logical names."""
  return NamedSharding(mesh, logical_to_mesh_axes(logical_names, mesh, rules=rules))


def get_mesh_axes_used_by_tensor_spec(tensor_sharding_spec):
  """
  Extracts the set of mesh axis names that a tensor's PartitionSpec uses.

  This function inspects a tensor's sharding specification (PartitionSpec) and
  identifies which mesh axes are actively used for sharding. If a tensor is not
  sharded (i.e., fully replicated), the resulting set will be empty.

  Args:
    tensor_sharding_spec: The PartitionSpec of a tensor, which defines how it's partitioned across the mesh.
    It can be None or contain strings and iterables representing the mesh axes.
    all_mesh_axis_names: A collection of all available mesh axis names in the current device mesh.

  Returns:
    A set of strings, where each string is a mesh axis name used by the
    tensor's sharding spec. Returns an empty set for unsharded tensors.
  """
  # Flatten the sharding spec, as it can contain nested iterables (e.g., ('data', 'mdl')).
  tensor_sharding_spec = sum(
      [
          [axis] if isinstance(axis, str) else list(axis) if isinstance(axis, Iterable) else []
          for axis in tensor_sharding_spec
      ],
      [],
  )
  return tensor_sharding_spec


def _get_nontrival_mesh_axes(mesh):
  """
  Returns mesh axes from config that are valid and have more than one shard.

  This function identifies which of the predefined potential sharding axes are
  actually present in the current device mesh and are configured with a size
  greater than one (i.e., are actually sharded).

  Args:
    mesh: The device mesh object, which contains information about the mesh topology, including axis names and their sizes.

  Returns:
    A set of strings, where each string is a mesh axis name that is both
    pre-configured as a target for sharding and has more than one shard in the mesh.
  """

  target_sharding_axes_config = [
      "fsdp",
      "fsdp_transpose",
      "sequence",
      "context",
      "context_autoregressive",
      "tensor",
      "tensor_transpose",
      "tensor_sequence",
      "stage",
      "expert",
  ]

  # Filter the target axes to find those that exist in the current mesh
  # and have a size greater than 1, meaning they are actually used for sharding.
  return {axis for axis in target_sharding_axes_config if axis in mesh.axis_names and mesh.shape[axis] > 1}


def _analyze_sharding(params, mesh, valid_target_mesh_axes):
  """
  Analyzes parameters to find which are unsharded on any valid mesh axis.

  This function iterates through all parameters in a model, checking their
  sharding specifications. It identifies parameters that are not sharded along any
  of the provided valid target axes (i.e., they are fully replicated across these axes).

  Args:
    params: A PyTree of model parameters.
    mesh: The device mesh object.
    valid_target_mesh_axes: A set of mesh axis names that are considered valid targets for sharding.

  Returns:
    A tuple containing:
      - unsharded_params_total_size (int): The total size (number of elements) of all parameters found to be
        unsharded on the target axes.
      - problematic_tensors_details (list): A list of dictionaries, where each
        dictionary contains details about a tensor that is not sharded on any of the target axes.
  """
  unsharded_params_total_size = 0  # Initialize a counter for the size of unsharded parameters.
  problematic_tensors_details = []  # Initialize a list to store details of problematic tensors.

  # Get a flattened list of all parameters (leaves) in the PyTree, along with their paths.
  all_params_leaves = jax.tree_util.tree_leaves_with_path(params)

  for path, p_leaf in all_params_leaves:  # Iterate over each parameter leaf
    param_name_str = jax.tree_util.keystr(path)  # Convert the tree path to a readable string

    # Check that sharding and spec exist and are valid
    sharding = getattr(p_leaf, "sharding", None)
    spec = getattr(sharding, "spec", None)
    assert sharding is not None and spec is not None and isinstance(spec, P), (
        f"Parameter '{param_name_str}' is missing a valid '.sharding.spec'."
        "Expected 'p_leaf.sharding.spec' to be a non-null 'partitionspec'."
    )

    current_sharding_spec = p_leaf.sharding.spec  # Extract the current tensor's sharding spec
    # Identify axes used for sharding
    mesh_axes_used = get_mesh_axes_used_by_tensor_spec(current_sharding_spec)
    # Check if the parameter is sharded on all the valid target axes.
    is_sharded_on_all_target_axis = all(axis in mesh_axes_used for axis in valid_target_mesh_axes)

    # If the parameter is not sharded on all of the target axes, it's considered "problematic."
    if not is_sharded_on_all_target_axis:
      unsharded_params_total_size += p_leaf.size  # Add to total unsharded parameter size
      unsharded_axes = set(valid_target_mesh_axes) - set(mesh_axes_used)
      # Add detailed info to list of problematic tensors
      problematic_tensors_details.append(
          {
              "name": param_name_str,  # Tensor name
              "size": p_leaf.size,  # tensor size
              "shape": p_leaf.shape,  # tensor shape
              "spec": str(current_sharding_spec),  # Tensor sharding spec as string
              "available_axes": sorted(list(valid_target_mesh_axes)),  # Axes that could be used for sharding
              "unsharded_axes": sorted(list(unsharded_axes)),  # Unsharded axes
          }
      )
  # Return the total size of unsharded parameters and the list of problematic tensors.
  return unsharded_params_total_size, problematic_tensors_details  # Return results


def _raise_if_unsharded_exceeds_tolerance(unsharded_size, total_size, tolerance, problematic_tensors_details):
  """
  Raises an AssertionError if the percentage of unsharded parameters exceeds the given tolerance.

  This function calculates the proportion of model parameters that are unsharded
  and compares it against a specified tolerance. If the tolerance is exceeded,
  it constructs and raises a detailed error message.

  Args:
    unsharded_size: The total size of parameters not sharded on target axes.
    total_size: The total size of all parameters in the model.
    tolerance: A float (e.g., 0.05 for 5%) representing the maximum allowed percentage of unsharded parameters.
    problematic_tensors_details: A list of details about the unsharded tensors,
    used to generate an informative error message.

  Raises:
    AssertionError: If the percentage of unsharded parameters is greater than the tolerance.
  """
  if total_size <= 0:
    raise ValueError("Total size must be greater than zero.")

  # Calculate the percentage of unsharded parameters.
  unsharded_param_perc = unsharded_size / total_size

  # If the percentage is over the tolerance, prepare and raise an error.
  if unsharded_param_perc > tolerance:
    # Sort the problematic tensors by size to show the largest ones first.
    problematic_tensors_details.sort(key=lambda x: x["size"], reverse=True)

    # Begin constructing the error message.
    error_msg_lines = [
        f"Unsharded parameter percentage ({unsharded_param_perc:.2%})" f"exceeds tolerance ({tolerance:.2%})."
    ]
    # Add a header explaining the issue.
    error_msg_lines.append(
        "The following large tensors are replicated (unsharded) but could be sharded on at "
        "least one of the available axes:"
    )
    # Add details for the top 5 largest problematic tensors.
    for detail in problematic_tensors_details[:5]:  # Show top 5 largest problematic tensors
      error_msg_lines.append(
          f" - Name: {detail['name']}(Size: {detail['size']}, Shape: {detail['spec']}, Spec: {detail['spec']}) "
          f" is unsharded on axis: {detail['unsharded_axes']}"
          f" could be sharded on: {detail['available_axes']}"
      )

    # Raise the assertion error with the combined, formatted message.
    raise AssertionError("\n".join(error_msg_lines))


def assert_params_sufficiently_sharded(params, mesh, tolerance):
  """
  Asserts that the total size of replicated parameters is within a given tolerance.

  This is the main function that orchestrates the sharding analysis. It determines
  the total number of parameters, identifies valid sharding axes, analyzes the
  sharding of all parameters, and then raises an error if the amount of
  unsharded parameters exceeds the specified tolerance.

  Args:
    params: A PyTree of model parameters.
    mesh: The device mesh object.
    tolerance: A float representing the maximum allowed percentage of unsharded parameters.
  """
  # Calculate the total size of all parameters in the model.
  from megatext.utils.debug import calculate_bytes_from_pytree
  total_num_params = calculate_bytes_from_pytree(params)

  # Get the set of nontrival mesh axes that can be used for sharding.
  valid_target_mesh_axes = _get_nontrival_mesh_axes(mesh)
  # If there are no valid axes to shard along, there's nothing to check, so we can exit.
  if not valid_target_mesh_axes:
    return  # Exit early

  # Analyze the parameters to find the total size of unsharded parameters
  # and get details on which tensors are problematic.
  unsharded_params_total_size, problematic_tensors_details = _analyze_sharding(params, mesh, valid_target_mesh_axes)

  # Check if the amount of unsharded parameters is within the tolerance and
  # raise an exception if it is not.
  _raise_if_unsharded_exceeds_tolerance(
      unsharded_params_total_size, total_num_params, tolerance, problematic_tensors_details
  )


def add_data_to_sharding(mesh, path, aval, sharding):
  """Adds 'data' dimension to sharding spec if compatible and not already present.

  This function attempts to add data parallelism to a sharding specification by finding
  a dimension that is divisible by the 'data' mesh axis size and doesn't conflict with
  existing partitioning (e.g., tensor parallelism).
  This function is mainly used to add data parallelism to the optimizer state for Zero-1 style sharding.

  Args:
    mesh: The device mesh
    path: JAX tree path to the value being sharded
    aval: Abstract value with shape information
    sharding: Current NamedSharding to potentially augment

  Returns:
    NamedSharding: Updated sharding with 'data' dimension added, or original if unchanged

  Raises:
    AssertionError: If sharding is not NamedSharding or shape cannot be sharded
  """
  if not isinstance(sharding, jax.sharding.NamedSharding):
    raise AssertionError(f"Expected NamedSharding, found {sharding} of {type(sharding)=} at {jax.tree_util.keystr(path)}")
  try:
    sharded_shape = sharding.shard_shape(aval.shape)
  except Exception as e:
    raise AssertionError(f"Could not shard {jax.tree_util.keystr(path)} of shape={aval.shape} with {sharding=}") from e
  pspec = sharding.spec

  if "data" in jax.tree.leaves(pspec):
    return sharding

  for idx, (size, partition) in enumerate(zip(sharded_shape, pspec)):
    if partition is None:
      partition = ()

    if isinstance(partition, str):
      partition = (partition,)

    if size % mesh.shape["data"] == 0 and (partition is None or "tensor" not in partition):
      added_component = ("data",) + partition
      new_pspec = jax.sharding.PartitionSpec(*(pspec[:idx] + (added_component,) + pspec[idx + 1 :]))
      new_sharding = jax.sharding.NamedSharding(sharding.mesh, new_pspec)
      return new_sharding
  return sharding


def maybe_update_params_sharding_with_opt(config, state_mesh_shardings):
  """Updates parameter sharding configuration when optimizer state sharding is enabled.

  When shard_optimizer_over_data is enabled (Zero-1 style sharding), this function
  extracts the optimizer state shardings from the Adam optimizer's first moment (mu)
  and merges them with the parameter shardings. This ensures parameter sharding is
  consistent with how the optimizer state is distributed across the compute mesh.

  Args:
    config: Configuration object with shard_optimizer_over_data flag
    state_mesh_shardings: Train state mesh shardings containing params and opt_state

  Returns:
    A tuple of (prev_params_shardings, updated_state_mesh_shardings):
      - prev_params_shardings: Original parameter shardings before the update
      - updated_state_mesh_shardings: State mesh shardings with updated params field
        (unchanged if shard_optimizer_over_data is False)
  """
  prev_params_shardings = state_mesh_shardings.params
  if config.shard_optimizer_over_data:
    if isinstance(state_mesh_shardings.opt_state, optax.ScaleByAdamState):
      sharded_fp32_params = state_mesh_shardings.opt_state.mu
    elif isinstance(state_mesh_shardings.opt_state, tuple) and isinstance(
        state_mesh_shardings.opt_state[0], optax.ScaleByAdamState
    ):
      sharded_fp32_params = state_mesh_shardings.opt_state[0].mu
    else:
      raise NotImplementedError(f"Could not find optimizer state shardings from {type(state_mesh_shardings.opt_state)}")
    if "params" not in sharded_fp32_params.keys():
      # When quantization=fp8 is enabled the sharded_fp32_params
      # are not wrapped in `params`. Here we wrap them back.
      sharded_fp32_params = {"params": sharded_fp32_params}
    state_mesh_shardings = state_mesh_shardings.replace(params=dict(prev_params_shardings, **sharded_fp32_params))
  return prev_params_shardings, state_mesh_shardings


def logical_axis_rules_pp_act_as_dp(logical_rules):
  """Add stage as a physical axes before data for each rule, so stage acts just like data instead of PP.
  This is used when we want to pipeline only a subset of layers, and leave the rest like DP.
  """
  new_rules = []
  for key, physical_axes in logical_rules:
    if isinstance(physical_axes, str):
      physical_axes = (physical_axes,)
    else:
      physical_axes = tuple(physical_axes)
    new_physical_axes = tuple(axis for axis in physical_axes if axis != "stage")
    if "data" in new_physical_axes:
      data_idx = new_physical_axes.index("data")
      new_physical_axes = new_physical_axes[0:data_idx] + ("stage",) + new_physical_axes[data_idx:]
    new_rules.append((key, new_physical_axes))
  return tuple(new_rules)


def get_formatted_sharding_annotations(params, mesh=None):
  """
  Generates a readable string report of sharding annotations for all parameters.

  This function iterates through a PyTree of model parameters and inspects the
  sharding information attached to each parameter (leaf). It creates a
  human-readable summary that is useful for debugging sharding configurations.

  Args:
    params: The PyTree of model parameters to inspect.
    mesh: (Optional) The device mesh. If provided, its axis names and shape
          are included in the report for additional context.

  Returns:
    A single string containing the formatted report of sharding annotations
    for every parameter, with each entry on a new line.
  """
  # Initialize a list to hold the lines of the report, starting with a title.
  annotation_lines = ["Comprehensice Weight Sharding Annotations:"]

  # If a mesh object is provided, add its details to the report header.
  if mesh:
    annotation_lines.append(f"Mesh axes: {mesh.axis_names}, Mesh shape: {mesh.shape}")
    annotation_lines.append("-" * 30)

  # Get a flattened list of all parameters (leaves) and their corresponding paths in the PyTree.
  all_params_leaves = jax.tree_util.tree_leaves_with_path(params)

  # Loop through each parameter leaf in the flattened list.
  for path, p_leaf in all_params_leaves:
    # Convert the parameter's path (a sequence of keys) into a readable string name.
    param_name_str = jax.tree_util.keystr(path)
    # Get the shape of the parameter as a string.
    shape_str = str(p_leaf.shape)
    # Set a default description for sharding, in case none is found.
    sharding_desc = "N/A"

    # Check if the parameter leaf has a 'sharding' attribute.
    if hasattr(p_leaf, "sharding"):
      # Case 1: Standard JAX sharding with a PartitionSpec.
      if hasattr(p_leaf.sharding, "spec") and p_leaf.sharding.spec is not None:
        # The spec is a tuple (PartitionSpec), format it for readability.
        spec_parts = []
        for item in p_leaf.sharding.spec:
          # Represent None as "Replicated" to make it explicit.
          spec_parts.append(str(item) if item is not None else "Replicated")
        sharding_desc = f"PartitionSpec({', '.join(spec_parts)})"
      # Case 2: The parameter is explicitly marked as fully replicated.
      elif hasattr(p_leaf.sharding, "spec") and p_leaf.sharding.spec is None:
        sharding_desc = "Fully Replicated (spec is None)"
      # Case 3: A generic fallback if a sharding object exists but has no recognized spec attribute.
      else:
        # Print the string representation of the sharding object itself.
        sharding_desc = str(p_leaf.sharding)
    # Case 4: The parameter has no .sharding attribute at all.
    else:
      sharding_desc = "No .sharding attribute found"

    # Append the formatted details for the current parameter to our list of lines.
    annotation_lines.append(f" - Param: {param_name_str}\n" f"   Shape: {shape_str}\n" f"   Sharding: {sharding_desc}")
  # Join all the collected lines into a single string, separated by newlines.
  return "\n".join(annotation_lines)


def remove_fsdp_sharding(sharding_tree):
  """Recursively traverses the sharding tree to remove fsdp axes."""

  def _remove_fsdp_from_partition_spec(named_sharding):
    """Removes 'fsdp' and 'fsdp_transpose' from a PartitionSpec."""
    if isinstance(named_sharding, jax.sharding.NamedSharding):
      new_spec = []
      # Iterate through each axis in the original PartitionSpec.
      for axis in named_sharding.spec:
        if axis is None:
          new_spec.append(None)
        elif isinstance(axis, str):
          # If the axis is 'fsdp', replace it with None to signify replication.
          if axis not in ("fsdp", "fsdp_transpose"):
            new_spec.append(axis)
          else:
            new_spec.append(None)
        elif isinstance(axis, (list, tuple)):
          # If the axis is a collection, filter out 'fsdp'.
          new_axis = [a for a in axis if a not in ("fsdp", "fsdp_transpose")]
          new_spec.append(tuple(new_axis))
        else:
          raise ValueError(f"Unsupported_axis_type: {type(axis)}")
        # Return a new sharding object with the modified spec.
      return jax.sharding.NamedSharding(named_sharding.mesh, jax.sharding.PartitionSpec(*new_spec))
    return named_sharding

  return jax.tree.map(_remove_fsdp_from_partition_spec, sharding_tree)


def get_physical_spec_no_fsdp(full_logical, mesh, logical_axis_rules):
  """
  Generates a physical sharding spec for fully replicated weights.

  This function computes a target sharding layout where model parameters are fully
  replicated across the 'fsdp' mesh axis. It starts with the original logical
  sharding and removes any rules that shard along the 'fsdp' or
  'fsdp_transpose' axes.

  Replacing a sharding axis with `None` in a PartitionSpec instructs JAX to
  replicate the array data along that physical mesh dimension. The resulting
  specification is used as a target layout for an all-gather operation.

  Args:
    full_logical: A PyTree of logical PartitionSpecs for the model parameters.
    mesh: The JAX device mesh.
    logical_axis_rules: Rules for converting logical axes to physical mesh axes.

  Returns:
    A PyTree of physical `jax.sharding.NamedSharding` objects that describe a
    layout where parameters are fully gathered (replicated) across the 'fsdp'
    mesh axis.
  """

  # Convert the high-level logical spec to a physical one using default rules.
  physical = logical_to_mesh_sharding(full_logical, mesh=mesh, rules=logical_axis_rules)
  # Apply the function to remove the FSDP sharding, defining our target layout.
  physical_no_fsdp = remove_fsdp_sharding(physical)
  return physical_no_fsdp


def all_gather_over_fsdp(variables, sharding_info, mesh, logical_axis_rules, shard_mode):
  """Performs an all-gather on FSDP-sharded variables via a sharding constraint.
  This function triggers an all-gather operation on the model's parameters.
  It does so by applying a sharding constraint that specifies a fully
  replicated layout.

  The JAX compiler satisfies this constraint by automatically inserting the
  necessary `all-gather` collective communication operations into the
  computation graph, effectively gathering the sharded weights.

  Args:
    variables: The PyTree of model parameters, currently sharded across devices.
    sharding_info: The logical partition spec of the currently sharded `variables`.
    mesh: The JAX device mesh.
    logical_axis_rules: Rules for converting logical axes to physical mesh axes.
    shard_mode: auto or explicit shard mode.

  Returns:
    The model's variables with the all-gather operation applied, resulting
    in the weights being fully replicated on all devices in the 'fsdp' mesh.
  """
  # Get the target physical layout (weights fully replicated).
  physical_constraint_no_fsdp = get_physical_spec_no_fsdp(sharding_info, mesh, logical_axis_rules)
  # Apply the constraint to the model's current variables. This tells JAX to
  # gather the weights into this layout.
  return maybe_shard_with_name(variables, physical_constraint_no_fsdp, shard_mode=shard_mode)


def create_device_mesh(config, devices=None):
  """Creates a device mesh with each slice in its own data parallel group. If there is only one slice, uses two replicas"""
  from jax.experimental import mesh_utils

  import numpy as np

  if devices is None:
    devices = jax.devices()
  if config.subslice_shape and config.enable_single_controller and config.num_slices == 1:
    max_logging.log(f"Trying to create a subslice with shape: {config.subslice_shape}")
    subslice_shape = tuple(int(x) for x in config.subslice_shape.split(","))
    device_coords = [device.coords for device in devices]
    device_coords_np = np.array(device_coords)

    # Find the minimum coordinates to start the subslice
    min_coords = device_coords_np.min(axis=0)

    subslice_devices = []
    for device in devices:
      coords = device.coords
      if all(min_coords[i] <= coords[i] < min_coords[i] + subslice_shape[i] for i in range(len(subslice_shape))):
        subslice_devices.append(device)
    devices = subslice_devices

  num_devices = len(devices)
  num_slices = 1 if config.inference_benchmark_test else config.num_slices
  num_devices_per_slice = num_devices // num_slices

  multi_slice_env = num_slices > 1

  # Find possible unspecified parallelisms
  ici_parallelism = fill_unspecified_mesh_axes(config.ici_parallelism.copy(), num_devices_per_slice, "ICI")

  allow_split_physical_axes = config.allow_split_physical_axes if config.allow_split_physical_axes else False

  if multi_slice_env:
    dcn_parallelism = fill_unspecified_mesh_axes(config.dcn_parallelism.copy(), num_slices, "DCN")
    if is_valid_custom_mesh(ici_parallelism, config.custom_mesh):
      mesh = create_custom_device_mesh(ici_parallelism, dcn_parallelism, devices, config.custom_mesh)
    else:
      mesh = mesh_utils.create_hybrid_device_mesh(
          ici_parallelism,
          dcn_parallelism,
          devices,
          allow_split_physical_axes=allow_split_physical_axes,
      )
  else:
    if allow_split_physical_axes:
      if is_valid_custom_mesh(ici_parallelism, config.custom_mesh):
        mesh = mesh_utils.create_device_mesh(
            [16, 16],
            devices,
            contiguous_submeshes=False,
            allow_split_physical_axes=False,
        )
        mesh = reshape_mesh_to_rings(mesh, config.custom_mesh)
        mesh = np.reshape(mesh, ici_parallelism)
      else:
        mesh = mesh_utils.create_device_mesh(
            ici_parallelism,
            devices,
            contiguous_submeshes=False,
            allow_split_physical_axes=allow_split_physical_axes,
        )
    else:
      mesh = mesh_utils.create_device_mesh(
          ici_parallelism,
          devices,
      )
      if config.optimize_mesh_for_tpu_v6e:
        mesh = optimize_mesh_for_tpu_v6e(mesh, devices)

  max_logging.log(f"Num_devices: {num_devices}, shape {mesh.shape}")

  return mesh


def shard_reorder_causal_load_balanced(batch, cp_size, shard_mode):
  """Shard the output of the reordered sequence."""
  reordered = reorder_causal_load_balanced(batch, cp_size)
  for _, v in batch.items():
    if isinstance(v, jax.Array):
      reordered = maybe_shard_with_name(reordered, v.sharding, shard_mode)
      break
  return reordered


def get_reorder_callable(cp_size, shard_mode):
  """Creates a callable that can be used with map() to reorder batches."""
  import functools

  return functools.partial(shard_reorder_causal_load_balanced, cp_size=cp_size, shard_mode=shard_mode)


# ---------------------------------------------------------------------------
# Functions moved from max_utils.py
# ---------------------------------------------------------------------------


def maybe_initialize_jax_distributed_system(raw_keys):
  """The best recipe to initialize the Jax Distributed System has varied over time. We keep a layer of
  indirection in Megatext to avoid breaking the call sites unnecessarily.

  Currently jax.distributed.initialize() fully works as expected!

  For CPUs, we call jax.distributed.initialize() explicitly, with the specified arguments.
  """
  if raw_keys["skip_jax_distributed_system"]:
    max_logging.log("Skipping jax distributed system due to skip_jax_distributed_system=True flag.")
    return
  if raw_keys["enable_single_controller"]:
    max_logging.log("Skipping jax distributed system since its not needed for single controller.")
    return
  if jax.distributed.is_initialized():
    max_logging.log("Jax distributed system is already initialized.")
    return
  if raw_keys["inference_benchmark_test"]:
    # Disable initialization for inference benmark test.
    return
  if raw_keys["compile_topology"]:
    # Don't initialize jax distributed with AOT compilation
    return
  if is_gpu_backend(raw_keys):
    max_logging.log("Attempting to initialize the jax distributed system for GPU backend...")
    initialize_jax_for_gpu(raw_keys)
    max_logging.log("Jax distributed system initialized on GPU!")
  elif is_cpu_backend(raw_keys):
    max_logging.log("Attempting to initialize the jax distributed system for CPU backend...")
    initialize_jax_for_cpu(raw_keys)
    max_logging.log("Jax distributed system initialized on CPUs!")
  elif raw_keys["enable_multi_tier_checkpointing"]:
    max_logging.log("Attempting to initialize the jax distributed system for multi-tier " "checkpointing...")
    initialize_multi_tier_checkpointing(
        local_checkpoint_directory=raw_keys["local_checkpoint_directory"],
        backup_interval_minutes=raw_keys["multi_tier_checkpointing_backup_interval_minutes"],
        run_name=raw_keys["run_name"],
        jax_initialization_timeout_seconds=raw_keys["jax_distributed_initialization_timeout"],
        data_parallelism=raw_keys["mtc_data_parallelism"],
    )
    max_logging.log("Jax distributed system initialized for multi-tier checkpointing!")
  elif (raw_keys["enable_checkpointing"] and raw_keys["compile_topology_num_slices"] == -1) or raw_keys[
      "hardware"
  ] == "gpu_multiprocess":
    max_logging.log("Attempting to initialize the jax distributed system...")
    if not raw_keys["enable_emergency_checkpoint"]:
      jax.distributed.initialize(initialization_timeout=raw_keys["jax_distributed_initialization_timeout"])
    else:
      if raw_keys["hardware"] == "gpu_multiprocess":
        max_logging.log("Initializing jax distribtued to support local checkpointing with" " GPUs...")
        jax.distributed.initialize(initialization_timeout=raw_keys["jax_distributed_initialization_timeout"])
        ocp.multihost.initialize_runtime_to_distributed_ids()
        ocp.multihost.initialize_distributed_to_device_ids()
      else:
        initialize_jax_for_tpu_with_emergency_checkpointing(raw_keys)
    max_logging.log("Jax distributed system initialized!")


def initialize_jax_for_gpu(raw_keys):
  """Jax distributed initialize for GPUs."""
  if os.environ.get("JAX_COORDINATOR_IP") is not None:
    coordinator_ip = str(os.getenv("JAX_COORDINATOR_IP"))
    coordinator_port = str(os.getenv("JAX_COORDINATOR_PORT"))
    devices = os.getenv("CUDA_VISIBLE_DEVICES")
    if devices is not None:
      try:
        devices = [int(x) for x in devices.split(",")]
      except (ValueError, TypeError) as e:
        max_logging.log(f"Error parsing CUDA_VISIBLE_DEVICES: {e}")
        devices = None

    jax.distributed.initialize(
        coordinator_address=f"{coordinator_ip}:{coordinator_port}",
        num_processes=int(os.getenv("NNODES")),
        process_id=int(os.getenv("NODE_RANK")),
        initialization_timeout=raw_keys["jax_distributed_initialization_timeout"],
        local_device_ids=devices,
    )
    max_logging.log(f"JAX global devices: {jax.devices()}")


def initialize_jax_for_cpu(raw_keys):
  """Jax distributed initialize for CPUs. Includes retries until the coordinator is ready."""
  coordinator_ip_address = get_coordinator_ip_address()
  coordinator_address = coordinator_ip_address + ":1234"  # JAX coordinator port used in XPK
  # Env variables to be set in XPK or otherwise
  job_index = int(os.environ.get("JOB_INDEX"))
  job_completion_index = int(os.environ.get("JOB_COMPLETION_INDEX"))
  processes_in_job = int(os.environ.get("PROCESSES_IN_JOB"))
  pid = job_index * processes_in_job + job_completion_index
  max_logging.log(f" Jax process id is {pid} ")
  # Explicit initialize is needed only for CPUs
  jax.distributed.initialize(
      coordinator_address=coordinator_address,
      process_id=pid,
      num_processes=int(os.environ.get("JAX_PROCESS_COUNT")),
      initialization_timeout=raw_keys["jax_distributed_initialization_timeout"],
  )


def initialize_jax_for_tpu_with_emergency_checkpointing(raw_keys):
  """Initialize JAX distributed runtime for TPUs when emergency checkpointing is used.
  The information required to initialize JAX distributed runtime will be written by GKE to
  the local checkpoint directory. This function retrieves that information and initializes
  JAX distributed runtime.
  """
  process_id, coordinator_address = _retrieve_jax_init_info(raw_keys)

  if process_id != "" and coordinator_address != "":
    max_logging.log(
        f"Using {process_id} as the process_id and {coordinator_address} as the"
        " coordinator_address to initialize JAX distributed runtime..."
    )
    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        process_id=int(process_id),
        initialization_timeout=raw_keys["jax_distributed_initialization_timeout"],
    )

    ocp.multihost.initialize_runtime_to_distributed_ids()
    ocp.multihost.initialize_distributed_to_device_ids()


def _retrieve_jax_init_info(raw_keys):
  """Retrieve JAX init info from a local file."""
  JAX_INIT_INFO_FILE = "jax-init-info.txt"
  local_jax_init_info_file = epath.Path(raw_keys["local_checkpoint_directory"]) / JAX_INIT_INFO_FILE
  # Allow time for the JAX init info file to be populated by GKE. This is needed because the file is
  # only populated when the worker with process id of 0 is determined. After a disruption, although some
  # workers might be up and running, the init info file won't be populated until the node with process id
  # of 0 is known and this could take time. Using 900 seconds for now and it needs to be increased if the
  # "repair" time is longer.
  for i in range(900):
    if local_jax_init_info_file.exists():
      return local_jax_init_info_file.read_text().split("\n")[:2]
    max_logging.log(f"Unable to locate {JAX_INIT_INFO_FILE} after {i} seconds, sleeping for 1 second before retrying...")
    time.sleep(1)
  max_logging.log(
      f"Unable to locate {JAX_INIT_INFO_FILE} after 900 seconds," "returning empty process id and coordinator address."
  )
  return "", ""


def get_num_slices(raw_keys):
  """Calculate num_slices based on number of devices."""
  if raw_keys.get("num_slices", -1) != -1:
    max_logging.log(f"Using num_slices={raw_keys['num_slices']} per user request.")
    return raw_keys["num_slices"]
  if raw_keys["hardware"] == "cpu":
    max_logging.log(" Setting num_slices=1 for CPU hardware type")
    return 1
  if int(raw_keys["compile_topology_num_slices"]) > 0:
    return raw_keys["compile_topology_num_slices"]
  else:
    devices = jax.devices()
    try:
      return 1 + max(d.slice_index for d in devices)
    except (ValueError, AttributeError):
      return 1


def is_cpu_backend(raw_keys):
  """Determine whether Megatext is intended to run on a CPU backend."""
  return raw_keys["hardware"] == "cpu"


def is_gpu_backend(raw_keys):
  """Determine whether Megatext is intended to run on a GPU backend."""
  return raw_keys["hardware"] == "gpu"


def get_coordinator_ip_address():
  """Get coordinator IP Address with retries"""
  coordinator_address = ""
  coordinator_ip_address = ""
  if os.environ.get("JAX_COORDINATOR_ADDRESS") is not None:
    coordinator_address = os.environ.get("JAX_COORDINATOR_ADDRESS")
    coordinator_found = False
    lookup_attempt = 1
    max_coordinator_lookups = 50
    while not coordinator_found and lookup_attempt <= max_coordinator_lookups:
      try:
        coordinator_ip_address = socket.gethostbyname(coordinator_address)
        coordinator_found = True
      except socket.gaierror:
        max_logging.log(
            f"Failed to recognize coordinator address {coordinator_address} on attempt {lookup_attempt}, retrying..."
        )
        lookup_attempt += 1
        time.sleep(5)
  max_logging.log(f"Coordinator IP address: {coordinator_ip_address}")
  return coordinator_ip_address


def fill_unspecified_mesh_axes(parallelism_vals, target_product, parallelism_type):
  """Evaluates unspecified DCN/ICI parallelism values"""
  if -1 in parallelism_vals:
    assert (
        parallelism_vals.count(-1) == 1
    ), f"Found unspecified values (-1) for more than one {parallelism_type}\
      parallelism axis. At most one axis can be unspecified."

    determined_val = target_product / np.prod(parallelism_vals) * -1

    assert (
        determined_val >= 1 and determined_val.is_integer
    ), f"Unspecified value unable to be determined with the given\
      {parallelism_type} parallelism values"

    parallelism_vals[parallelism_vals.index(-1)] = int(determined_val)

  target_type = "slices" if parallelism_type == "DCN" else "devices per slice"
  assert np.prod(parallelism_vals) == target_product, (
      f"Number of {target_type} {target_product} does not match"
      f" the product of the {parallelism_type} parallelism {np.prod(parallelism_vals)}"
  )

  return parallelism_vals


def reshape_mesh_to_rings(a, strategy):
  """Reshape device mesh to rings for 64x4 or 32x8 mesh shape"""
  b = []
  if strategy == HYBRID_RING_64X4:
    for i in range(8):
      b.append([])
      for j in range(8):
        a_i = i * 2
        a_j = j * 2
        # forms a ring of size 4
        b[i].append([a[a_i, a_j], a[a_i, a_j + 1], a[a_i + 1, a_j + 1], a[a_i + 1, a_j]])
    b = np.array(b)
    b = np.reshape(b, (64, 4))
  elif strategy == HYBRID_RING_32X8:
    for i in range(8):
      b.append([])
      for j in range(4):
        a_i = i * 2
        a_j = j * 4
        # forms a ring of size 8
        b[i].append(
            [
                a[a_i, a_j],
                a[a_i, a_j + 1],
                a[a_i, a_j + 2],
                a[a_i, a_j + 3],
                a[a_i + 1, a_j + 3],
                a[a_i + 1, a_j + 2],
                a[a_i + 1, a_j + 1],
                a[a_i + 1, a_j],
            ]
        )
    b = np.array(b)
    b = np.reshape(b, (32, 8))
  else:
    raise ValueError(f"The strategy {strategy} to reshape the mesh is not implemented.")
  return b


def create_custom_device_mesh(
    mesh_shape: Sequence[int],
    dcn_mesh_shape: Sequence[int],
    devices: Sequence[Any],
    custom_strategy: str,
    process_is_granule: bool = False,
    should_sort_granules_by_key: bool = True,
) -> np.ndarray:
  """Custom device mesh for 64x4 ici parallelism"""
  assert len(devices) % 256 == 0, f"This custom mesh is not valid for {len(devices)} devices"
  attr = "process_index" if process_is_granule else "slice_index"
  if not hasattr(devices[0], attr):
    raise ValueError(f"Device {devices[0]} does not have attribute {attr}. See" " `process_is_granule` option.")
  granule_dict = collections.defaultdict(list)
  for dev in devices:
    granule_dict[getattr(dev, attr)].append(dev)
  granules = (
      [granule_dict[key] for key in sorted(granule_dict.keys())] if should_sort_granules_by_key else granule_dict.values()
  )
  if np.prod(dcn_mesh_shape) != len(granules):
    raise ValueError(f"Number of slices {len(granules)} must equal the product of " f"dcn_mesh_shape {dcn_mesh_shape}")
  per_granule_meshes = [
      mesh_utils.create_device_mesh(
          [16, 16],
          granule,
          allow_split_physical_axes=False,
      )
      for granule in granules
  ]

  per_granule_meshes = [np.reshape(reshape_mesh_to_rings(x, custom_strategy), mesh_shape) for x in per_granule_meshes]
  # TODO(jekbradbury): handle non-uniform DCN topologies
  granule_mesh = np.arange(len(granules)).reshape(dcn_mesh_shape)
  blocks = np.vectorize(lambda i: per_granule_meshes[i], otypes=[object])(granule_mesh)
  device_mesh = np.block(blocks.tolist())
  return device_mesh


def is_valid_custom_mesh(ici_parallelism, strategy):
  """Checks if the given strategy and ICI parallelism are valid."""
  if not strategy:
    return False

  valid_strategies = {
      HYBRID_RING_64X4: [1, 4, 64],
      HYBRID_RING_32X8: [1, 8, 32],
  }

  if strategy in valid_strategies:
    if sorted(set(ici_parallelism)) == valid_strategies[strategy]:
      return True
    else:
      raise ValueError(f"Invalid custom_mesh:{strategy} chosen for ICI mesh shape {ici_parallelism}")
  else:
    raise ValueError(f"The strategy {strategy} to reshape the mesh is invalid.")


def optimize_mesh_for_tpu_v6e(mesh, devices):
  """Apply transformations to the mesh to optimize for TPU v6e"""
  if devices[0].device_kind != "TPU v6 lite":
    return mesh
  num_devices = len(devices)
  mesh_is_1d_ring = num_devices in mesh.shape
  if not mesh_is_1d_ring:
    return mesh
  # check that the physical topology is 2x4
  device_coords = [d.coords for d in devices]
  coord_size = len(device_coords[0])
  max_coords = tuple(max(dc[i] for dc in device_coords) for i in range(coord_size))
  min_coords = tuple(min(dc[i] for dc in device_coords) for i in range(coord_size))
  dims = tuple(h - l + 1 for (h, l) in zip(max_coords, min_coords))
  if dims != (2, 4, 1):
    return mesh
  axis_idx = mesh.shape.index(num_devices)
  new_mesh = np.moveaxis(mesh, axis_idx, 0)
  new_mesh[4:] = new_mesh[-1:3:-1]
  new_mesh = np.moveaxis(new_mesh, 0, axis_idx)
  max_logging.log("Optimized the mesh for TPU v6e")
  return new_mesh
