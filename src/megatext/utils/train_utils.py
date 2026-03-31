# Copyright 2023–2025 Google LLC
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

# pylint: disable=bare-except, consider-using-generator
""" Utils that are only interesting for training in MaxText. """

import os
import jax
import functools
from flax.linen import partitioning as nn_partitioning
from megatext.common import checkpointing
from megatext.common.data_loader import create_dataloader
from megatext.common.goodput import GoodputEvent, maybe_record_goodput
from megatext.optimizers import optimizers
from megatext.schedulers import create_learning_rate_schedule
from megatext.utils import logging as max_logging
from megatext.utils import max_utils
from megatext.utils import megatext_utils
from megatext.utils import model_factory as model_creation_utils
from megatext.utils import sharding
from megatext.utils.rampup_batch import create_rampup_manager
from megatext.trainers.diloco import diloco


def create_training_tools(config, model, mesh):
  """Creates the init_rng, optimizer, learning rate schedule, and checkpoint manager."""
  init_rng = jax.random.PRNGKey(config.init_weights_seed)
  learning_rate_schedule = create_learning_rate_schedule(config)
  # pass in model for muon
  tx = optimizers.get_optimizer(config, learning_rate_schedule, model)
  logger = checkpointing.setup_checkpoint_logger(config)
  if config.enable_multi_tier_checkpointing:
    checkpoint_manager = checkpointing.create_orbax_emergency_replicator_checkpoint_manager(
        config.local_checkpoint_directory,
        config.local_checkpoint_period,
        mesh,
    )
  elif config.enable_emergency_checkpoint:
    abstract_state, _, _ = megatext_utils.get_abstract_state(model, tx, config, init_rng, mesh, is_training=True)
    checkpoint_manager = checkpointing.create_orbax_emergency_checkpoint_manager(
        config.local_checkpoint_directory,
        config.checkpoint_dir,
        mesh,
        abstract_state,
        config.local_checkpoint_period,
        config.checkpoint_period,
        logger,
    )
  else:
    # TODO(b/368121306): Remove this once zarr3 support is plumbed on the backend
    use_ocdbt = config.checkpoint_storage_use_ocdbt
    use_zarr3 = config.checkpoint_storage_use_zarr3
    if config.enable_single_controller and not config.colocated_python_checkpointing:
      use_ocdbt, use_zarr3 = False, False

    checkpoint_dir = ""
    if config.enable_checkpointing:
      checkpoint_dir = config.checkpoint_dir
    checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
        checkpoint_dir,
        config.enable_checkpointing,
        config.async_checkpointing,
        config.checkpoint_period,
        config.dataset_type,
        logger,
        use_ocdbt,
        use_zarr3,
        config.enable_continuous_checkpointing,
        config.max_num_checkpoints_to_keep,
        config.checkpoint_storage_concurrent_gb,
        config.enable_single_controller,
        config.colocated_python_checkpointing,
        config.enable_single_replica_ckpt_restoring,
    )

  return init_rng, checkpoint_manager, learning_rate_schedule, tx


def jit_train_step(config, model, state, state_mesh_shardings, data_sharding, train_step, params_shardings):
  """Returns a JIT-compiled train step function, which is loaded from a file if specified in the config."""
  if config.enable_diloco:
    functional_train = train_step
    in_shardings = (state_mesh_shardings, data_sharding, None)  # State, batch, rng
    out_shardings = (state_mesh_shardings, None)  # State, metrics
    static_argnums = ()  # We partial out the static argnums of model and config
    donate_argnums = 0  # This is the index of the state - we allow the compiler to make use of this memory.
  else:
    (
        functional_train,
        in_shardings,
        out_shardings,
        static_argnums,
        donate_argnums,
    ) = megatext_utils.get_functional_train_with_signature(
        train_step, data_sharding, state_mesh_shardings, model, config, params_shardings
    )

  # Define the compilation of functional_train, either by loading the compiled version or wrapping a new one in a jit
  if config.compiled_trainstep_file != "":
    max_logging.log("Loading the compiled function...")
    execution_devices = model.mesh.devices.flatten().tolist()
    # Need to pass train signature and state to determine i/o shapes of train_state for now.
    p_train_step = megatext_utils.load_compiled(config, functional_train, state, execution_devices)
    max_logging.log("Loaded compiled function!")
  else:
    p_train_step = jax.jit(
        functional_train,
        in_shardings=in_shardings,
        out_shardings=out_shardings,
        static_argnums=static_argnums,
        donate_argnums=donate_argnums,
    )

  return p_train_step


def jit_eval_step(config, model, state_mesh_shardings, data_sharding, eval_step):
  """Returns a JIT-compiled eval step function."""
  (
      functional_eval,
      in_shardings,
      out_shardings,
      static_argnums,
      donate_argnums,
  ) = megatext_utils.get_functional_eval_with_signature(eval_step, data_sharding, state_mesh_shardings, model, config)

  p_eval_step = None
  if config.compiled_trainstep_file == "":
    p_eval_step = jax.jit(
        functional_eval,
        in_shardings=in_shardings,
        out_shardings=out_shardings,
        static_argnums=static_argnums,
        donate_argnums=donate_argnums,
    )

  return p_eval_step


def jit_train_and_eval_step(
    config,
    model,
    mesh,
    state,
    state_mesh_shardings,
    train_step,
    eval_step=None,
    eval_data_iterator=None,
    params_shardings=None,
):
  """Returns a JIT-compiled train and eval step function."""
  if config.enable_diloco:
    train_step_partial = functools.partial(train_step, model, config, state_mesh_shardings, params_shardings)
    train_step = diloco.build_diloco_train_step(config, train_step_partial, mesh=mesh)
  data_sharding = sharding.get_input_data_sharding(config, mesh)
  p_train_step = jit_train_step(config, model, state, state_mesh_shardings, data_sharding, train_step, params_shardings)
  p_eval_step = None
  if eval_data_iterator:
    p_eval_step = jit_eval_step(config, model, state_mesh_shardings, data_sharding, eval_step)

  return p_train_step, p_eval_step


def setup_train_loop(config, recorder, devices=None):
  """Set up prerequisites for the training loop -

      checkpoint_manager, PRNG keys, Mesh, Model and optimizer.
      Set up data iterator and tokenizer, initialize the model.

  Args: config recorder

  Returns:
    init_rng:
    checkpoint_manager: Orbax checkpointer
    state_mesh_annotations: the mesh annotations for the train state
    model:
    mesh:
    learning_rate_schedule:
    data_iterator:
    data_loader:
    rampup_manager: the class managing rampup batch sizes
    state: the initialized train state
  """
  # pylint: disable=import-outside-toplevel
  from megatext.data.input_pipeline_interface import create_data_iterator

  with maybe_record_goodput(recorder, GoodputEvent.TPU_INIT):
    model = model_creation_utils.from_config(config, devices)
    mesh = model.mesh
    init_rng, checkpoint_manager, learning_rate_schedule, tx = create_training_tools(config, model, mesh)

  with maybe_record_goodput(recorder, GoodputEvent.TRAINING_PREPARATION):
    data_iterator, eval_data_iterator = create_data_iterator(config, mesh)
    rampup_manager = create_rampup_manager(config, checkpoint_manager)
    data_loader = create_dataloader(config, mesh, data_iterator, recorder, rampup_manager)
    context_parallel_size = mesh.shape.get("context", 1)
    # Check if context parallelism is being used with sequence packing
    if context_parallel_size > 1 and config.packing and config.dataset_type != "synthetic":
      raise ValueError(
          "Context parallelism cannot be used with sequence packing. "
          "Disable sequence packing (set packing=False). "
          "Context parallelism with packing support will be added soon."
      )

    # Apply reordering wrapper to data iterators if context parallelism is enabled
    with jax.set_mesh(mesh):
      if context_parallel_size > 1 and config.context_parallel_load_balance:
        data_iterator = map(megatext_utils.get_reorder_callable(context_parallel_size, config.shard_mode), data_iterator)
        if eval_data_iterator:
          eval_data_iterator = map(
              megatext_utils.get_reorder_callable(context_parallel_size, config.shard_mode),
              eval_data_iterator,
          )

    state, _, state_mesh_shardings, data_iterator = megatext_utils.setup_training_state(
        model, data_iterator, tx, config, init_rng, mesh, checkpoint_manager
    )

    if config.enable_diloco:
      with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
        state, outer_opt_state_sharding = diloco.build_diloco_state(config, lambda: state, mesh=mesh)

        # create state_mesh_shardings for the DilocoState
        inner_state_shardings = diloco.add_diloco_to_sharding(state_mesh_shardings)
        state_mesh_shardings = diloco.DiLoCoTrainState(
            inner_state_shardings,
            state_mesh_shardings.params,
            outer_opt_state_sharding,
            jax.sharding.NamedSharding(mesh=state_mesh_shardings.step.mesh, spec=jax.sharding.PartitionSpec()),
        )

    # TODO(aireenmei, hengtaoguo): support sharding in vit for multimodal
    if not config.using_pipeline_parallelism and not config.use_multimodal:
      # The vocab tensor(s) of shape [vocab, embed] (and transpose) are not sharded by stage
      sharding.assert_params_sufficiently_sharded(state.params, mesh, config.sharding_tolerance)

    # print weights sharding info under debug sharding mode
    if config.debug_sharding:
      logical_annotations = megatext_utils.get_logical_annotations(model, tx, config, init_rng, mesh, is_training=True)
      max_utils.print_non_trivial_mesh_axis(model.mesh)
      megatext_utils.print_shardings_params(
          state.params, state_mesh_shardings.params, model.mesh, logical_annotations.params
      )

  return (
      init_rng,
      checkpoint_manager,
      state_mesh_shardings,
      model,
      mesh,
      learning_rate_schedule,
      data_iterator,
      data_loader,
      rampup_manager,
      eval_data_iterator,
      state,
  )


def validate_train_config(config):
  """Validates the configuration is set correctly for 'train.py'."""

  assert config.run_name, "Erroring out, need a real run_name"
  if config.dataset_path and not config.dataset_path.startswith("gs://"):
    max_logging.log("WARNING: 'dataset_path' might be pointing your local file system")
  if not config.base_output_directory.startswith("gs://"):
    max_logging.log("WARNING: 'base_output_directory' might be pointing your local file" " system")
  assert config.steps > 0, "You must set steps or learning_rate_schedule_steps to a positive integer."

  if config.quantization in ("fp8", "nanoo_fp8"):
    # pylint: disable=line-too-long
    assert config.gradient_accumulation_steps == 1, (
        "fp8 can't be used with gradient_accumulation_steps right now. Please"
        " use other quantization or set gradient_accumulation_steps to 1"
    )

  if config.packing and config.dataset_type == "synthetic":
    max_logging.log(
        "WARNING: Sequence packing is essentially ignored for synthetic data. "
        "Please use a real dataset to use sequence packing."
    )


def get_functional_train_with_signature(
    train_step, data_sharding, state_mesh_shardings, model, config, params_shardings=None
):
  """Get the shardings (both state and data) for `train_step`."""
  functional_train = functools.partial(train_step, model, config, state_mesh_shardings, params_shardings)
  functional_train.__name__ = "train_step"
  in_shardings = (state_mesh_shardings, data_sharding, None)  # State, batch, rng
  out_shardings = (state_mesh_shardings, None)  # State, metrics
  static_argnums = ()  # We partial out the static argnums of model and config
  donate_argnums = 0  # This is the index of the state - we allow the compiler to make use of this memory.
  return functional_train, in_shardings, out_shardings, static_argnums, donate_argnums


def get_functional_eval_with_signature(eval_step, data_sharding, state_mesh_shardings, model, config):
  """Get the shardings (both state and data) for `eval_step`."""
  functional_eval = functools.partial(eval_step, model, config)
  functional_eval.__name__ = "eval_step"
  in_shardings = (state_mesh_shardings, data_sharding, None)  # State, batch, rng
  out_shardings = None  # metrics
  static_argnums = ()  # We partial out the static argnums of model, config
  donate_argnums = ()  # state will be kept instead of being donated in eval_step
  return functional_eval, in_shardings, out_shardings, static_argnums, donate_argnums


def get_shaped_batch(config):
  """Return the shape of the batch - this is what eval_shape would return for the
  output of create_data_iterator, but eval_shape doesn't work, see b/306901078."""
  import jax.numpy as jnp

  from megatext.multimodal import processor as mm_processor

  if config.enable_diloco:
    batch_shape = (
        config.num_diloco_replicas,
        config.global_batch_size_to_load // config.num_diloco_replicas,
        config.max_target_length,
    )
  else:
    batch_shape = (config.global_batch_size_to_load, config.max_target_length)
  shaped_batch = {}
  shaped_batch["inputs"] = jax.ShapeDtypeStruct(batch_shape, jnp.int32)
  shaped_batch["inputs_position"] = jax.ShapeDtypeStruct(batch_shape, jnp.int32)
  shaped_batch["inputs_segmentation"] = jax.ShapeDtypeStruct(batch_shape, jnp.int32)
  shaped_batch["targets"] = jax.ShapeDtypeStruct(batch_shape, jnp.int32)
  shaped_batch["targets_position"] = jax.ShapeDtypeStruct(batch_shape, jnp.int32)
  shaped_batch["targets_segmentation"] = jax.ShapeDtypeStruct(batch_shape, jnp.int32)
  if config.use_multimodal:
    image_shape = mm_processor.get_dummy_image_shape_for_init(
        config.decoder_block, batch_size=config.micro_batch_size_to_train_on
    )
    shaped_batch["images"] = jax.ShapeDtypeStruct(image_shape, jnp.int32)
    shaped_batch["image_masks"] = jax.ShapeDtypeStruct(image_shape[:2], jnp.int32)
  if config.use_audio:
    audio_shape = mm_processor.get_dummy_audio_shape_for_init(config)
    shaped_batch["audios"] = jax.ShapeDtypeStruct(audio_shape, jnp.float32)
  return shaped_batch


def should_prevent_cse_in_remat(config):
  """Determines whether to prevent common subexpression elimination (CSE) in remat.

  CSE should not be prevented when:
  1. Layers are being scanned (scan_layers=True), OR
  2. Gradient accumulation is enabled (gradient_accumulation_steps > 1) on GPU hardware

  Args:
    config: Configuration object with scan_layers, gradient_accumulation_steps, and hardware

  Returns:
    bool: True if CSE should be prevented, False otherwise
  """
  if config.scan_layers:
    return False

  if config.gradient_accumulation_steps > 1 and config.hardware in ("gpu", "gpu_multiprocess"):
    return False

  return True


def load_compiled(config, partial_train, state, execution_devices):
  """# Loading a serialized compiled train step function."""
  import pickle

  from jax.experimental.serialize_executable import deserialize_and_load

  # Currently partial_train and state  are needed to reconstruct
  # input/output shapes to construct the in_trees and out_trees for load API
  # Parker is working on a serializing these
  def load_serialized_compiled(save_name):
    with open(save_name, "rb") as f:
      serialized_compiled = pickle.load(f)
    return serialized_compiled

  def get_train_input_output_trees(func, input_args, input_kwargs):
    _, in_tree_recreated = jax.tree_util.tree_flatten((input_args, input_kwargs))
    out_shaped = jax.eval_shape(func, *input_args, **input_kwargs)
    _, out_tree_recreated = jax.tree_util.tree_flatten(out_shaped)
    return in_tree_recreated, out_tree_recreated

  serialized_compiled = load_serialized_compiled(config.compiled_trainstep_file)
  shaped_batch = get_shaped_batch(config)
  example_rng = jax.random.PRNGKey(0)
  shaped_input_args = (state, shaped_batch, example_rng)
  shaped_input_kwargs = {}
  in_tree, out_tree = get_train_input_output_trees(partial_train, shaped_input_args, shaped_input_kwargs)
  p_train_step = deserialize_and_load(serialized_compiled, in_tree, out_tree, execution_devices=execution_devices)
  return p_train_step


def apply_gradient_clipping(raw_grads, state, clipping_threshold):
  """Applies gradient clipping to raw gradients, with special handing for FLAX fp8 stats.

  Args:
    raw_grads: A pytree of raw gradients.
    state: The current optimizer state.
    clipping_threshold: The gradient clipping threshold.

  Returns:
    A pytree of clipped gradients.
  """
  import optax

  from megatext.utils.training import OVERWRITE_WITH_GRADIENT

  gradient_clip_transformation = optax.clip_by_global_norm(clipping_threshold)
  if OVERWRITE_WITH_GRADIENT in raw_grads:
    # Scales + Amax History for Delayed Tensor Scaling SHOULD NOT be clipped or affect clipping
    fp8_stats = raw_grads.pop(OVERWRITE_WITH_GRADIENT)
    grads, _ = gradient_clip_transformation.update(raw_grads, state, None)
    grads[OVERWRITE_WITH_GRADIENT] = fp8_stats  # pytype: disable=unsupported-operands
    raw_grads[OVERWRITE_WITH_GRADIENT] = fp8_stats  # pytype: disable=unsupported-operands
  else:
    grads, _ = gradient_clip_transformation.update(raw_grads, state, None)

  return grads


def update_state_param(state, target_path, value):
  """
  Updates a specific parameter in state.params at the given path.

  Args:
      state: The current TrainState.
      target_path: A tuple of keys matching the structure inside state.params.
      value: The value to apply.
  """

  def create_jax_path(target_path):
    path = []
    for k in target_path:
      path.append(jax.tree_util.DictKey(key=k))
    return tuple(path)

  def _apply_update(path, param):
    if path == updated_target_path:
      return param + value
    return param

  updated_target_path = create_jax_path(target_path)
  new_params = jax.tree_util.tree_map_with_path(_apply_update, state.params)
  return state.replace(params=new_params)


def init_decode_state(apply_fn, params):
  """Init train state with null opt state for decode."""
  from flax.training import train_state

  state = train_state.TrainState(step=0, apply_fn=apply_fn, params=params, tx=None, opt_state={})  # type: ignore
  return state


def init_training_state(apply_fn, params, tx):
  """Init train state with null opt state for decode."""
  from flax.training import train_state

  state = train_state.TrainState.create(apply_fn=apply_fn, params=params, tx=tx)
  return state


def init_initial_state(model, tx, config, is_training, key):
  """
  We pass in "static" objects like model, tx, config as JAX compares them by
  object hash, and instantiating them inside causes pjit top-level annotations
  to fail to match as pytree prefixes if we re-instantiate.

  Args: model, tx, config, is_training, key
  """
  import numpy as np
  import jax.numpy as jnp

  from megatext.multimodal import processor as mm_processor

  input_shape = (config.micro_batch_size_to_train_on, config.max_target_length)
  image_shape = mm_processor.get_dummy_image_shape_for_init(
      config.decoder_block, batch_size=config.micro_batch_size_to_train_on
  )
  audio_shape = mm_processor.get_dummy_audio_shape_for_init(config)
  # Split the master key into independent keys for each RNG collection
  # Reference: https://flax-linen.readthedocs.io/en/latest/guides/flax_fundamentals/rng_guide.html
  params_key, dropout_key, aqt_key = jax.random.split(key, 3)

  model_vars = model.init(
      {"params": params_key, "dropout": dropout_key, "aqt": aqt_key},
      np.ones(input_shape, dtype=jnp.int32),
      np.ones(input_shape, dtype=jnp.int32),
      encoder_images=np.ones(image_shape, dtype=jnp.int32) if config.use_multimodal else None,
      encoder_audios=np.ones(audio_shape, dtype=jnp.float32) if config.use_audio else None,
      # nnx_method="no_op",
  )
  if is_training:
    return init_training_state(model.apply, model_vars, tx)
  return init_decode_state(model.apply, model_vars)


def get_abstract_param(model, config):
  """Get abstract model structure (name, shape) without materializing the weights to save memory"""
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


def setup_decode_state(model, config, rng, mesh, checkpoint_manager):
  """Setup decode state by loading params from a checkpoint.
  Args:
    model: the flax model to initialize
    config: config object
    rng: jax.prng key
    mesh: jax.devices() mesh
    checkpoint_manager: Checkpoint manager

  Returns:
    state: state with decode params loaded from the checkpoint
    state_mesh_annotations: the mesh annotations for the state
  """
  from flax.linen import partitioning as nn_partitioning

  from megatext.common import checkpointing
  from megatext.utils import max_utils

  if not config.load_parameters_path:
    # generate random params
    max_logging.log("No decode checkpoint specified - generating random weights.")
    state, state_mesh_annotations, _, _ = setup_initial_state(
        model, None, None, config, rng, mesh, checkpoint_manager, False
    )
  else:
    # Load params from checkpoint
    max_logging.log(f"Loading decode params from {config.load_parameters_path}")
    unboxed_abstract_state, state_mesh_annotations, _ = get_abstract_state(model, None, config, rng, mesh, False)
    with nn_partitioning.axis_rules(config.logical_axis_rules):
      params = checkpointing.load_params_from_path(
          config.load_parameters_path,
          unboxed_abstract_state.params,
          config.checkpoint_storage_concurrent_gb,
          config.checkpoint_storage_use_ocdbt,
          config.checkpoint_storage_use_zarr3,
      )
    state = init_decode_state(None, params)

  state = max_utils.unbox_logicallypartioned(state)
  return state, state_mesh_annotations


def setup_training_state(model, data_iterator, tx, config, rng, mesh, checkpoint_manager):
  is_training = True
  return setup_initial_state(
      model,
      data_iterator,
      tx,
      config,
      rng,
      mesh,
      checkpoint_manager,
      is_training,
  )


def setup_initial_state(
    model,
    data_iterator,
    tx,
    config,
    rng,
    mesh,
    checkpoint_manager,
    is_training=True,
):
  """We initialize the model and optimizer state, and optionally load from a
  checkpoint as necessary.

  Args:
    model: the flax model to initialize
    tx: the optax.GradientTransformation
    config: config object
    rng: jax.prng key
    mesh: jax.devices() mesh
    checkpoint_manager: an Orbax checkpointing.CheckpointManager object
    is_training: True to initialize training state, False for decode state

  Returns:
    state: the initialized train state
    state_mesh_annotations: the mesh annotations for the train state
  """
  import functools as _functools

  from flax.linen import partitioning as nn_partitioning

  import orbax.checkpoint.experimental.emergency.checkpoint_manager as emergency_checkpoint_manager
  import orbax.checkpoint.experimental.emergency.replicator_checkpoint_manager as emergency_replicator_checkpoint_manager

  from megatext.common import checkpointing
  from megatext.utils import max_utils

  unboxed_abstract_state, state_mesh_annotations, state_mesh_shardings = get_abstract_state(
      model, tx, config, rng, mesh, is_training
  )

  # Initialization
  with nn_partitioning.axis_rules(config.logical_axis_rules):
    restored, raw_params = checkpointing.load_state_if_possible(
        checkpoint_manager,
        data_iterator,
        config.load_parameters_path,
        config.load_full_state_path,
        config.checkpoint_storage_concurrent_gb,
        unboxed_abstract_state,
        config.enable_single_replica_ckpt_restoring,
        config.dataset_type,
        use_ocdbt=config.checkpoint_storage_use_ocdbt,
        use_zarr3=config.checkpoint_storage_use_zarr3,
        enable_orbax_v1=config.enable_orbax_v1,
        checkpoint_conversion_fn=config.checkpoint_conversion_fn,
        source_checkpoint_layout=config.source_checkpoint_layout,
        expansion_factor_real_data=config.expansion_factor_real_data,
    )

    if restored:
      if isinstance(
          checkpoint_manager,
          (
              emergency_checkpoint_manager.CheckpointManager,
              emergency_replicator_checkpoint_manager.ReplicatorCheckpointManager,
          ),
      ):
        state = restored
      else:
        # The update of data_iterator state happens in place, no need to assign explicitly
        state = restored["items"]
    else:
      init_state_partial = _functools.partial(init_initial_state, model, tx, config, is_training)
      init_state_partial.__name__ = "initialize_state"
      # pylint: disable=not-callable
      state = jax.jit(
          init_state_partial,
          in_shardings=None,
          out_shardings=state_mesh_shardings,
      )(rng)
      if raw_params:  # If we loaded a partial state, we need to merge it.
        state = state.replace(params=raw_params)

  state = max_utils.unbox_logicallypartioned(state)

  return state, state_mesh_annotations, state_mesh_shardings, data_iterator


def get_logical_annotations(model, tx, config, rng, mesh, is_training=True):
  import functools as _functools

  from flax import linen as nn
  from flax.linen import partitioning as nn_partitioning

  init_state_partial = _functools.partial(init_initial_state, model, tx, config, is_training, rng)

  with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
    abstract_state = jax.eval_shape(init_state_partial)
    logical_annotations = nn.get_partition_spec(abstract_state)
  return logical_annotations


def get_abstract_state(model, tx, config, rng, mesh, is_training=True):
  """Get a shaped abstraction of the state (including optimizer)"""
  import functools as _functools

  from flax import linen as nn
  from flax.linen import partitioning as nn_partitioning

  from megatext.utils import max_utils
  from megatext.utils import sharding as _sharding

  init_state_partial = _functools.partial(init_initial_state, model, tx, config, is_training, rng)

  with nn_partitioning.axis_rules(config.logical_axis_rules):
    abstract_state = jax.eval_shape(init_state_partial)

  state_logical_annotations = nn.get_partition_spec(abstract_state)

  state_mesh_shardings = nn.logical_to_mesh_sharding(state_logical_annotations, mesh, config.logical_axis_rules)
  if is_training and config.shard_optimizer_over_data:
    # Add data to sharding for optimizer state
    state_mesh_shardings = state_mesh_shardings.replace(
        opt_state=jax.tree.map_with_path(
            _functools.partial(_sharding.add_data_to_sharding, mesh),
            max_utils.unbox_logicallypartioned(abstract_state).opt_state,
            state_mesh_shardings.opt_state,
        )
    )
  if is_training and config.optimizer_memory_host_offload:
    opt_state = jax.tree_util.tree_map(lambda x: x.with_memory_kind(kind="pinned_host"), state_mesh_shardings.opt_state)
    state_mesh_shardings = state_mesh_shardings.replace(opt_state=opt_state)
  if is_training and config.parameter_memory_host_offload:
    assert config.param_scan_axis == 0, "You must set the scan axis 0 to enable parameter offloading."

    def move(path, x):
      max_logging.log(f"max_utils.py: Moving {path} to host")
      return x.with_memory_kind(kind="pinned_host")

    params = jax.tree_util.tree_map_with_path(move, state_mesh_shardings.params)
    state_mesh_shardings = state_mesh_shardings.replace(params=params)

  abstract_sharded_state = jax.jit(init_state_partial, in_shardings=None, out_shardings=state_mesh_shardings).eval_shape()

  unboxed_abstract_sharded_state = max_utils.unbox_logicallypartioned(abstract_sharded_state)
  # Initialization
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
    state_mesh_annotations = nn.logical_to_mesh(state_logical_annotations)
  return (
      unboxed_abstract_sharded_state,
      state_mesh_annotations,
      state_mesh_shardings,
  )


def get_prefill_kv_cache_annotations(model, config, rng, mesh, page_state=None):
  """Get a shaped abstraction of the state (including optimizer)"""
  import jax.numpy as jnp

  from flax import linen as nn
  from flax.linen import partitioning as nn_partitioning

  from megatext.common.common_types import MODEL_MODE_PREFILL
  from megatext.multimodal import processor as mm_processor

  def init_kv_cache(model, config):
    input_shape = (
        config.micro_batch_size_to_train_on,
        config.max_prefill_predict_length,
    )
    image_shape = mm_processor.get_dummy_image_shape_for_init(
        config.decoder_block, batch_size=config.micro_batch_size_to_train_on
    )
    audio_shape = mm_processor.get_dummy_audio_shape_for_init(config)

    model_vars = model.init(
        {"params": rng, "dropout": rng, "aqt": rng},
        jnp.ones(input_shape),
        jnp.ones(input_shape),
        encoder_images=jnp.ones(image_shape) if config.use_multimodal else None,
        encoder_audios=jnp.ones(audio_shape) if config.use_audio else None,
        model_mode=MODEL_MODE_PREFILL,
        slot=0,
        page_state=page_state,
    )
    return model_vars["cache"]

  with nn_partitioning.axis_rules(config.logical_axis_rules):
    init_kv_cache_partial = functools.partial(init_kv_cache, model, config)
    abstract_state = jax.eval_shape(init_kv_cache_partial)
  state_logical_annotations = nn.get_partition_spec(abstract_state)
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
    state_mesh_annotations = nn.logical_to_mesh(state_logical_annotations)
  return state_mesh_annotations


def get_kv_cache_annotations(model, config, rng, mesh, page_state=None):
  """Get a shaped abstraction of the state (including optimizer)"""
  import jax.numpy as jnp

  from flax import linen as nn
  from flax.linen import partitioning as nn_partitioning

  from megatext.common.common_types import MODEL_MODE_AUTOREGRESSIVE
  from megatext.multimodal import processor as mm_processor

  def init_kv_cache(model, config):
    input_shape = (config.micro_batch_size_to_train_on, 1)
    image_shape = mm_processor.get_dummy_image_shape_for_init(
        config.decoder_block, batch_size=config.micro_batch_size_to_train_on
    )
    audio_shape = mm_processor.get_dummy_audio_shape_for_init(config)

    model_vars = model.init(
        {"params": rng, "dropout": rng, "aqt": rng},
        jnp.ones(input_shape),
        jnp.ones(input_shape),
        encoder_images=jnp.ones(image_shape) if config.use_multimodal else None,
        encoder_audios=jnp.ones(audio_shape) if config.use_audio else None,
        model_mode=MODEL_MODE_AUTOREGRESSIVE,
        slot=0,
        page_state=page_state,
    )
    return model_vars["cache"]

  with nn_partitioning.axis_rules(config.logical_axis_rules):
    init_kv_cache_partial = functools.partial(init_kv_cache, model, config)
    abstract_state = jax.eval_shape(init_kv_cache_partial)
  state_logical_annotations = nn.get_partition_spec(abstract_state)
  with jax.set_mesh(mesh), nn_partitioning.axis_rules(config.logical_axis_rules):
    state_mesh_annotations = nn.logical_to_mesh(state_logical_annotations)
  return state_mesh_annotations


def save_quantized_checkpoint_if_configured(config, params):
  """Save quantized checkpoint if configured"""
  from megatext.common import checkpointing

  assert config.quantization, "quantization must be configured"
  if config.save_quantized_params_path:
    checkpointing.save_params_to_path(
        checkpoint_dir=config.save_quantized_params_path,
        params=params,
        use_ocdbt=config.checkpoint_storage_use_ocdbt,
        use_zarr3=config.checkpoint_storage_use_zarr3,
    )
  else:
    max_logging.log("Skipping saving quantized checkpoint as save_quantized_params_path is null.")
