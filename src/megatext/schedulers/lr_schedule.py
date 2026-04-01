"""Learning rate schedules for training."""

import jax.numpy as jnp
import optax

from megatext.configs.types import LearningRateScheduleType, WsdDecayStyle


def _cosine_decay(init_lr, final_lr, len_steps):
  """Cosine decay from init_lr to final_lr over len_steps."""
  def schedule(step):
    pct = step / (len_steps - 1) if len_steps > 1 else 1.0
    a = 0.5 * (jnp.cos(jnp.pi * pct) + 1)
    return init_lr * a + final_lr * (1 - a)
  return schedule


def _warmup_and_tail(lr, warmup_steps, total_steps, steps, decay_pieces, decay_boundaries):
  """Wrap decay pieces with linear warmup prefix and constant-zero tail."""
  pieces = []
  boundaries = []

  if warmup_steps > 0:
    pieces.append(optax.linear_schedule(init_value=0.0, end_value=lr, transition_steps=warmup_steps))
    boundaries.append(warmup_steps)

  pieces.extend(decay_pieces)
  boundaries.extend(decay_boundaries)

  constant_zero_steps = steps - total_steps
  if constant_zero_steps > 0:
    pieces.append(optax.constant_schedule(0.0))
    boundaries.append(total_steps)

  return optax.join_schedules(pieces, boundaries)


def create_cosine_schedule(
    lr: float,
    final_lr: float,
    warmup_steps: int,
    total_steps: int,
    steps: int,
) -> optax.Schedule:
  """Cosine learning rate schedule with linear warmup.

  Inspired by Llama2, see https://arxiv.org/pdf/2307.09288.pdf section 2.2

  1) Linear warmup 0 → lr over [0, warmup_steps]
  2) Cosine decay lr → final_lr over [warmup_steps, total_steps]
  3) Constant 0 from total_steps to steps (if steps > total_steps)
  """
  cos_steps = total_steps - warmup_steps
  decay_pieces = []
  decay_boundaries = []
  if cos_steps > 0:
    decay_pieces.append(_cosine_decay(lr, final_lr, cos_steps))
    decay_boundaries.append(warmup_steps + cos_steps)

  return _warmup_and_tail(lr, warmup_steps, total_steps, steps, decay_pieces, decay_boundaries)


def create_wsd_schedule(
    lr: float,
    final_lr: float,
    warmup_steps: int,
    total_steps: int,
    steps: int,
    decay_steps: int,
    decay_style: WsdDecayStyle = WsdDecayStyle.LINEAR,
) -> optax.Schedule:
  """Warmup-Stable-Decay (WSD) learning rate schedule.

  1) Linear warmup 0 → lr over [0, warmup_steps]
  2) Constant lr for [warmup_steps, total_steps - decay_steps]
  3) Decay lr → final_lr over [total_steps - decay_steps, total_steps]
     using linear or cosine decay based on decay_style
  4) Constant 0 from total_steps to steps (if steps > total_steps)
  """
  stable_steps = total_steps - warmup_steps - decay_steps
  decay_pieces = []
  decay_boundaries = []

  if stable_steps > 0:
    decay_pieces.append(optax.constant_schedule(lr))
    decay_boundaries.append(warmup_steps + stable_steps)
  if decay_steps > 0:
    if decay_style == WsdDecayStyle.LINEAR:
      decay_pieces.append(optax.linear_schedule(init_value=lr, end_value=final_lr, transition_steps=decay_steps - 1))
    else:
      decay_pieces.append(_cosine_decay(lr, final_lr, decay_steps))
    decay_boundaries.append(warmup_steps + stable_steps + decay_steps)

  return _warmup_and_tail(lr, warmup_steps, total_steps, steps, decay_pieces, decay_boundaries)


def create_learning_rate_schedule(config) -> optax.Schedule:
  """Create learning rate schedule from config. Dispatches to cosine or WSD."""
  lr = config.learning_rate
  final_lr = config.final_learning_rate

  if config.lr_schedule_type == LearningRateScheduleType.COSINE:
    return create_cosine_schedule(lr, final_lr, config.warmup_steps, config.learning_rate_schedule_steps, config.steps)
  else:
    return create_wsd_schedule(
        lr, final_lr, config.warmup_steps, config.learning_rate_schedule_steps, config.steps,
        config.wsd_decay_steps, config.wsd_decay_style,
    )
