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

"""Re-export shim for backward compatibility.

All functions have been moved to focused modules:
  - sharding: create_device_mesh, get_input_data_sharding, assert_params_sufficiently_sharded,
              add_data_to_sharding, maybe_update_params_sharding_with_opt, all_gather_over_fsdp,
              shard_reorder_causal_load_balanced, get_reorder_callable
  - flops: calculate_tflops_training_per_device, calculate_tokens_training_per_device, and all
           other TFLOP calculation functions
  - train_utils: get_functional_train_with_signature, get_functional_eval_with_signature,
                 get_shaped_batch, should_prevent_cse_in_remat, load_compiled,
                 apply_gradient_clipping, update_state_param, init_decode_state,
                 init_training_state, init_initial_state, get_abstract_param,
                 setup_decode_state, setup_training_state, setup_initial_state,
                 get_logical_annotations, get_abstract_state,
                 get_prefill_kv_cache_annotations, get_kv_cache_annotations,
                 save_quantized_checkpoint_if_configured
  - debug: print_shardings_params, add_config_to_summary_writer
  - training: get_nested_value, get_intermediate_value, maybe_dump_jaxpr, OVERWRITE_WITH_GRADIENT
"""

# pylint: disable=unused-import

# --- sharding ---
from megatext.utils.sharding import create_device_mesh  # noqa: F401
from megatext.utils.sharding import get_input_data_sharding  # noqa: F401
from megatext.utils.sharding import assert_params_sufficiently_sharded  # noqa: F401
from megatext.utils.sharding import add_data_to_sharding  # noqa: F401
from megatext.utils.sharding import maybe_update_params_sharding_with_opt  # noqa: F401
from megatext.utils.sharding import all_gather_over_fsdp  # noqa: F401
from megatext.utils.sharding import shard_reorder_causal_load_balanced  # noqa: F401
from megatext.utils.sharding import get_reorder_callable  # noqa: F401

# --- flops ---
from megatext.utils.flops import calculate_tflops_training_per_device  # noqa: F401
from megatext.utils.flops import calculate_tokens_training_per_device  # noqa: F401
from megatext.utils.flops import calculate_gemma2_tflops_training_per_device  # noqa: F401
from megatext.utils.flops import calculate_mixed_attention_model_tflops_training_per_device  # noqa: F401
from megatext.utils.flops import _calculate_chunked_attention_flops_per_layer  # noqa: F401
from megatext.utils.flops import calculate_llama4_attention_tflops  # noqa: F401
from megatext.utils.flops import calculate_indexer_mask_ratio  # noqa: F401
from megatext.utils.flops import calculate_indexer_tflops_per_device  # noqa: F401
from megatext.utils.flops import calculate_mla_tflops_per_device  # noqa: F401
from megatext.utils.flops import calculate_ffn_mamtul_tflops_per_device  # noqa: F401
from megatext.utils.flops import calculate_routed_and_shared_ffn_tflops_per_device  # noqa: F401
from megatext.utils.flops import get_dense_moe_layers  # noqa: F401
from megatext.utils.flops import calculate_gated_delta_net_flops_per_device  # noqa: F401
from megatext.utils.flops import calculate_gemma3_vision_layers_tflops_per_device  # noqa: F401
from megatext.utils.flops import calculate_llama4_vision_layers_tflops_per_device  # noqa: F401
from megatext.utils.flops import calculate_engram_tflops  # noqa: F401
from megatext.utils.flops import calculate_vision_encoder_tflops  # noqa: F401
from megatext.utils.flops import calculate_prefill_tflops_per_device  # noqa: F401

# --- train_utils ---
from megatext.utils.train_utils import get_functional_train_with_signature  # noqa: F401
from megatext.utils.train_utils import get_functional_eval_with_signature  # noqa: F401
from megatext.utils.train_utils import get_shaped_batch  # noqa: F401
from megatext.utils.train_utils import should_prevent_cse_in_remat  # noqa: F401
from megatext.utils.train_utils import load_compiled  # noqa: F401
from megatext.utils.train_utils import apply_gradient_clipping  # noqa: F401
from megatext.utils.train_utils import update_state_param  # noqa: F401
from megatext.utils.train_utils import init_decode_state  # noqa: F401
from megatext.utils.train_utils import init_training_state  # noqa: F401
from megatext.utils.train_utils import init_initial_state  # noqa: F401
from megatext.utils.train_utils import get_abstract_param  # noqa: F401
from megatext.utils.train_utils import setup_decode_state  # noqa: F401
from megatext.utils.train_utils import setup_training_state  # noqa: F401
from megatext.utils.train_utils import setup_initial_state  # noqa: F401
from megatext.utils.train_utils import get_logical_annotations  # noqa: F401
from megatext.utils.train_utils import get_abstract_state  # noqa: F401
from megatext.utils.train_utils import get_prefill_kv_cache_annotations  # noqa: F401
from megatext.utils.train_utils import get_kv_cache_annotations  # noqa: F401
from megatext.utils.train_utils import save_quantized_checkpoint_if_configured  # noqa: F401

# --- debug ---
from megatext.utils.debug import print_shardings_params  # noqa: F401
from megatext.utils.debug import add_config_to_summary_writer  # noqa: F401

# --- training ---
from megatext.utils.training import get_nested_value  # noqa: F401
from megatext.utils.training import get_intermediate_value  # noqa: F401
from megatext.utils.training import maybe_dump_jaxpr  # noqa: F401
from megatext.utils.training import OVERWRITE_WITH_GRADIENT  # noqa: F401

# --- schedulers (pre-existing re-export) ---
from megatext.schedulers import create_learning_rate_schedule  # noqa: F401
