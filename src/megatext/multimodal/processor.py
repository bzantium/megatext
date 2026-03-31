# Copyright 2023–2026 Google LLC
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

"""Multimodal data preprocessor router."""

from megatext.multimodal import utils as mm_utils


def preprocess_mm_data(config):
  """Preprocesses multimodal data based on the provided configuration.
  Routes to the appropriate preprocessing function based on the model name.

  Args:
    config: A `pyconfig.Config` object containing configuration parameters.

  Returns:
    A `PreprocessorOutput` object containing the processed multimodal data.
  """
  processor_outputs = mm_utils.PreprocessorOutput()

  if config.decoder_block == "gemma3":
    from megatext.multimodal.processor_gemma3 import preprocess_mm_data_gemma3  # pylint: disable=import-outside-toplevel

    images = [mm_utils.load_image_from_path(p) for p in config.image_path.split(",")]
    processor_outputs = preprocess_mm_data_gemma3(images)
  elif config.decoder_block == "llama4":
    from megatext.multimodal.processor_llama4 import preprocess_mm_data_llama4  # pylint: disable=import-outside-toplevel

    images = [mm_utils.load_image_from_path(p) for p in config.image_path.split(",")]
    processor_outputs = preprocess_mm_data_llama4(images)
  elif config.decoder_block == "qwen3":
    from megatext.multimodal.processor_qwen3_omni import preprocess_mm_data_qwen3_omni  # pylint: disable=import-outside-toplevel

    processor_outputs = preprocess_mm_data_qwen3_omni(config)
  else:
    raise ValueError(f"Model {config.model} not supported for multimodal preprocessing.")

  return processor_outputs


def preprocess_image_for_training(image, decoder_block):
  """Preprocesses a single image for training based on the decoder block type."""
  if decoder_block == "gemma3":
    from megatext.multimodal.processor_gemma3 import preprocess_mm_data_gemma3  # pylint: disable=import-outside-toplevel

    return preprocess_mm_data_gemma3(image)
  elif decoder_block == "llama4":
    from megatext.multimodal.processor_llama4 import preprocess_mm_data_llama4  # pylint: disable=import-outside-toplevel

    return preprocess_mm_data_llama4(image)
  else:
    raise ValueError(f"Decoder block {decoder_block} not supported for image preprocessing.")


def get_image_offsets(config, processor_output: mm_utils.PreprocessorOutput | None):
  """Get the increase in total token count after inserting image token placeholders"""
  if config.decoder_block == "gemma3":
    from megatext.multimodal.processor_gemma3 import get_image_offsets_gemma3  # pylint: disable=import-outside-toplevel

    return get_image_offsets_gemma3(processor_output)
  elif config.decoder_block == "llama4":
    from megatext.multimodal.processor_llama4 import get_image_offsets_llama4  # pylint: disable=import-outside-toplevel

    return get_image_offsets_llama4(processor_output)
  elif config.decoder_block == "qwen3":
    from megatext.multimodal.processor_qwen3_omni import get_mm_offsets_qwen3_omni  # pylint: disable=import-outside-toplevel

    return get_mm_offsets_qwen3_omni(config, processor_output)
  else:
    return 0


def reformat_prompt(prompt, image_placeholder, decoder_block, num_images, video_placeholder="<|video|>", num_videos=0):
  """Reformat prompt for different models."""
  if decoder_block == "gemma3":
    from megatext.multimodal.processor_gemma3 import reformat_prompt_gemma3  # pylint: disable=import-outside-toplevel

    return reformat_prompt_gemma3(prompt, image_placeholder, num_images)
  elif decoder_block == "llama4":
    from megatext.multimodal.processor_llama4 import reformat_prompt_llama4  # pylint: disable=import-outside-toplevel

    return reformat_prompt_llama4(prompt, image_placeholder, num_images)
  elif decoder_block == "qwen3":
    from megatext.multimodal.processor_qwen3_omni import reformat_prompt_qwen3_omni  # pylint: disable=import-outside-toplevel

    return reformat_prompt_qwen3_omni(
        prompt=prompt,
        image_placeholder=image_placeholder,
        num_images=num_images,
        video_placeholder=video_placeholder,
        num_videos=num_videos,
    )
  else:
    return prompt


def reformat_response(response, decoder_block):
  """Reformat response for different models."""
  if decoder_block == "llama4":
    formatted_response = f"{response}<|eot|>"
    return formatted_response
  elif decoder_block == "gemma3":
    formatted_response = f"{response}<end_of_turn>"
    return formatted_response
  elif decoder_block == "qwen3":
    formatted_response = f"{response}<|im_end|>"
    return formatted_response
  else:
    return response


def prepare_text_for_image_fusion(tokens, config, processor_output=None):
  """Prepare text by adding extra tokens for image fusion based on the model."""
  if config.decoder_block == "gemma3":
    from megatext.multimodal.processor_gemma3 import add_extra_tokens_for_images_gemma3  # pylint: disable=import-outside-toplevel

    return add_extra_tokens_for_images_gemma3(tokens, max_num_images=processor_output.num_images)
  elif config.decoder_block == "llama4":
    from megatext.multimodal.processor_llama4 import add_extra_tokens_for_images_llama4  # pylint: disable=import-outside-toplevel

    return add_extra_tokens_for_images_llama4(tokens, processor_output)
  elif config.decoder_block == "qwen3":
    from megatext.multimodal.processor_qwen3_omni import add_extra_tokens_for_qwen3_omni  # pylint: disable=import-outside-toplevel

    return add_extra_tokens_for_qwen3_omni(tokens, config, processor_output)
  else:
    raise ValueError(f"Model {config.model} does not support multimodal inference.")


def get_dummy_image_shape_for_init(decoder_block, batch_size=1, num_image_per_sequence=1):
  """Return the shape of the dummy image for specific model's initialization."""
  image_shape = ()
  if decoder_block == "gemma3":
    from megatext.multimodal.processor_gemma3 import get_dummy_image_shape_for_init_gemma3  # pylint: disable=import-outside-toplevel

    image_shape = get_dummy_image_shape_for_init_gemma3(batch_size, num_image_per_sequence)
  elif decoder_block == "llama4":
    from megatext.multimodal.processor_llama4 import get_dummy_image_shape_for_init_llama4  # pylint: disable=import-outside-toplevel

    image_shape = get_dummy_image_shape_for_init_llama4(batch_size, num_image_per_sequence)
  elif decoder_block == "qwen3":
    from megatext.multimodal.processor_qwen3_omni import get_dummy_image_shape_for_init_qwen3_omni  # pylint: disable=import-outside-toplevel

    image_shape = get_dummy_image_shape_for_init_qwen3_omni(batch_size)
  return image_shape


def get_dummy_audio_shape_for_init(config):
  """Return the shape of the dummy audio for specific model's initialization.

  Args:
    config: Model configuration containing audio parameters

  Returns:
    Tuple representing audio shape: (batch, num_mel_bins, audio_length)
    Returns empty tuple if audio is not configured for the model
  """
  audio_shape = ()
  if config.decoder_block == "qwen3":
    from megatext.multimodal.processor_qwen3_omni import get_dummy_audio_shape_for_init_qwen3_omni  # pylint: disable=import-outside-toplevel

    audio_shape = get_dummy_audio_shape_for_init_qwen3_omni(config)

  return audio_shape


def get_bidirectional_mask_vision(config, decoder_input_tokens):
  """Get the bidirectional mask for specific models."""
  bidirectional_mask_vision = None
  if config.decoder_block == "gemma3":
    from megatext.multimodal.processor_gemma3 import GEMMA_TOKEN_PLACEHOLDER  # pylint: disable=import-outside-toplevel

    bidirectional_mask_vision = decoder_input_tokens == GEMMA_TOKEN_PLACEHOLDER
  elif config.decoder_block == "llama4":
    from megatext.multimodal.processor_llama4 import LLAMA4_PATCH_TOKEN  # pylint: disable=import-outside-toplevel

    bidirectional_mask_vision = decoder_input_tokens == LLAMA4_PATCH_TOKEN
  elif config.decoder_block == "qwen3":
    from megatext.multimodal.processor_qwen3_omni import QWEN3_OMNI_IMAGE_TOKEN, QWEN3_OMNI_VIDEO_TOKEN  # pylint: disable=import-outside-toplevel

    # Create bidirectional_mask for vision/video token merging
    bidirectional_mask_vision = (decoder_input_tokens == QWEN3_OMNI_IMAGE_TOKEN) | (
        decoder_input_tokens == QWEN3_OMNI_VIDEO_TOKEN
    )
    # Create image/video mask for deepstack visual embedding injection
  return bidirectional_mask_vision


def get_bidirectional_mask_audio(config, decoder_input_tokens):
  """Get the bidirectional mask for specific models."""
  bidirectional_mask_audio = None
  if config.decoder_block == "qwen3":
    from megatext.multimodal.processor_qwen3_omni import QWEN3_OMNI_AUDIO_TOKEN  # pylint: disable=import-outside-toplevel

    # Create bidirectional_mask for audio token merging
    bidirectional_mask_audio = decoder_input_tokens == QWEN3_OMNI_AUDIO_TOKEN
  return bidirectional_mask_audio
