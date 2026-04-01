"""Generic scannable block for inhomogeneous decoder layer patterns.

Provides a single ScannableBlock class that replaces model-specific scannable
block implementations. Each model supplies a `layer_cls` and a
`layer_kwargs_fn(i, config)` that returns per-layer keyword arguments
(e.g. attention_type, is_moe_layer).
"""

from __future__ import annotations

from typing import Any, Callable

from flax import nnx
import jax.numpy as jnp


Config = Any
Quant = Any


class ScannableBlock(nnx.Module):
  """Generic scannable block that wraps N sub-layers into one scan iteration.

  Each sub-layer is created from ``layer_cls`` with per-layer kwargs determined
  by ``layer_kwargs_fn(i, config)``.

  Subclasses only need to provide ``layer_cls`` and ``layer_kwargs_fn`` via
  ``__init__`` — the ``__call__`` loop is fully generic.
  """

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
    cycle = config.inhomogeneous_layer_cycle_interval
    for i in range(cycle):
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

  def __call__(
      self,
      inputs: jnp.ndarray,
      decoder_segment_ids: jnp.ndarray | None,
      decoder_positions: jnp.ndarray | None,
      deterministic: bool,
      model_mode: str,
      **kwargs,
  ):
    y = inputs
    if isinstance(y, tuple):
      y = y[0]
    for i in range(self.config.inhomogeneous_layer_cycle_interval):
      layer = getattr(self, f"layers_{i}")
      y = layer(
          y,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
          **kwargs,
      )
      if self.config.scan_layers:
        y = y[0]
    if self.config.scan_layers:
      return y, None
    return y
