"""Generic scannable block for multi-layer decoder scan bodies.

Models with a scan body that covers more than one logical decoder layer can
subclass ``ScannableBlock`` and supply a ``layer_cls`` together with a
``layer_kwargs_fn(i, config)`` that selects per-layer arguments such as
attention type or MoE layout.

The default body size still falls back to
``config.inhomogeneous_layer_cycle_interval`` for existing alternating-layer
models, but subclasses can override ``scan_body_layer_count`` when the body
size is model-specific.
"""

from __future__ import annotations

from typing import Any, Callable

from flax import nnx
import jax.numpy as jnp


Config = Any
Quant = Any


class ScannableBlock(nnx.Module):
  """Generic scannable block that wraps N sub-layers into one scan iteration."""

  @classmethod
  def scan_body_layer_count(cls, config: Config) -> int:
    return getattr(config, "inhomogeneous_layer_cycle_interval", 1)

  def __init__(
      self,
      config: Config,
      mesh,
      model_mode: str,
      quant: Quant | None = None,
      *,
      layer_cls: type,
      layer_kwargs_fn: Callable[[int, Any], dict[str, Any]],
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.mesh = mesh
    self._scan_body_layer_count = self.scan_body_layer_count(config)
    for i in range(self._scan_body_layer_count):
      kwargs = layer_kwargs_fn(i, config)
      layer = layer_cls(
          config=config,
          mesh=mesh,
          model_mode=model_mode,
          quant=quant,
          rngs=rngs,
          **kwargs,
      )
      setattr(self, f"layers_{i}", layer)

  def _apply_layer(
      self,
      layer: nnx.Module,
      inputs: jnp.ndarray,
      decoder_segment_ids: jnp.ndarray | None,
      decoder_positions: jnp.ndarray | None,
      deterministic: bool,
      model_mode: str,
      runtime_context: dict[str, Any] | None = None,
      **kwargs,
  ):
    if runtime_context is not None:
      kwargs = {**runtime_context, **kwargs}
    return layer(
        inputs,
        decoder_segment_ids,
        decoder_positions,
        deterministic,
        model_mode,
        **kwargs,
    )

  def __call__(
      self,
      inputs: jnp.ndarray,
      decoder_segment_ids: jnp.ndarray | None,
      decoder_positions: jnp.ndarray | None,
      deterministic: bool,
      model_mode: str,
      runtime_context: dict[str, Any] | None = None,
      **kwargs,
  ):
    y = inputs
    if isinstance(y, tuple):
      y = y[0]
    for i in range(self._scan_body_layer_count):
      layer = getattr(self, f"layers_{i}")
      y = self._apply_layer(
          layer,
          y,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
          runtime_context=runtime_context,
          **kwargs,
      )
      if self.config.scan_layers:
        y = y[0]
    if self.config.scan_layers:
      return y, None
    return y
