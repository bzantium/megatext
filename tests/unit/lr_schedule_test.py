# Copyright 2023-2026 Google LLC
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

"""Unit tests for megatext.schedulers.lr_schedule."""

import jax.numpy as jnp
import numpy.testing as npt

from megatext.configs.types import WsdDecayStyle
from megatext.schedulers.lr_schedule import (
    create_cosine_schedule,
    create_learning_rate_schedule,
    create_wsd_schedule,
)


def _eval(schedule, step):
    """Evaluate schedule at a given step, returning a plain float."""
    return float(schedule(jnp.array(step)))


# ---------------------------------------------------------------------------
# Cosine schedule
# ---------------------------------------------------------------------------


def test_cosine_warmup_start_near_zero():
    schedule = create_cosine_schedule(lr=1e-3, final_lr=1e-4, warmup_steps=10, total_steps=100, steps=100)
    npt.assert_allclose(_eval(schedule, 0), 0.0, atol=1e-7)


def test_cosine_warmup_end_equals_peak():
    schedule = create_cosine_schedule(lr=1e-3, final_lr=1e-4, warmup_steps=10, total_steps=100, steps=100)
    npt.assert_allclose(_eval(schedule, 10), 1e-3, rtol=1e-5)


def test_cosine_lr_decreases_after_warmup():
    schedule = create_cosine_schedule(lr=1e-3, final_lr=1e-4, warmup_steps=10, total_steps=100, steps=100)
    assert _eval(schedule, 30) < _eval(schedule, 10)


def test_cosine_final_value():
    schedule = create_cosine_schedule(lr=1e-3, final_lr=1e-4, warmup_steps=10, total_steps=100, steps=100)
    npt.assert_allclose(_eval(schedule, 99), 1e-4, rtol=1e-4)


def test_cosine_mid_decay_between_peak_and_final():
    schedule = create_cosine_schedule(lr=1e-3, final_lr=1e-4, warmup_steps=10, total_steps=100, steps=100)
    mid = 10 + (100 - 10) // 2
    lr_mid = _eval(schedule, mid)
    assert 1e-4 < lr_mid < 1e-3


def test_cosine_zero_phase_after_schedule():
    schedule = create_cosine_schedule(lr=1e-3, final_lr=1e-4, warmup_steps=10, total_steps=100, steps=150)
    npt.assert_allclose(_eval(schedule, 120), 0.0, atol=1e-7)


# ---------------------------------------------------------------------------
# WSD schedule
# ---------------------------------------------------------------------------


def test_wsd_warmup_start_near_zero():
    schedule = create_wsd_schedule(lr=1e-3, final_lr=1e-4, warmup_steps=10, total_steps=100, steps=100, decay_steps=10)
    npt.assert_allclose(_eval(schedule, 0), 0.0, atol=1e-7)


def test_wsd_warmup_end_equals_peak():
    schedule = create_wsd_schedule(lr=1e-3, final_lr=1e-4, warmup_steps=10, total_steps=100, steps=100, decay_steps=10)
    npt.assert_allclose(_eval(schedule, 10), 1e-3, rtol=1e-5)


def test_wsd_stable_phase_constant():
    schedule = create_wsd_schedule(lr=1e-3, final_lr=1e-4, warmup_steps=10, total_steps=100, steps=100, decay_steps=10)
    # stable_steps = 100 - 10 - 10 = 80, stable phase: steps 10..89
    npt.assert_allclose(_eval(schedule, 50), 1e-3, rtol=1e-5)


def test_wsd_decay_phase_decreases():
    schedule = create_wsd_schedule(lr=1e-3, final_lr=1e-4, warmup_steps=10, total_steps=100, steps=100, decay_steps=10)
    # decay starts at step 90
    assert _eval(schedule, 99) < _eval(schedule, 90)


def test_wsd_cosine_decay_style():
    schedule = create_wsd_schedule(
        lr=1e-3, final_lr=1e-4, warmup_steps=10, total_steps=100, steps=100,
        decay_steps=10, decay_style=WsdDecayStyle.COSINE,
    )
    assert _eval(schedule, 99) < _eval(schedule, 90)


# ---------------------------------------------------------------------------
# create_learning_rate_schedule (config dispatch)
# ---------------------------------------------------------------------------


def test_config_dispatch_cosine():
    from types import SimpleNamespace
    from megatext.configs.types import LearningRateScheduleType

    config = SimpleNamespace(
        learning_rate=1e-3,
        learning_rate_final_fraction=0.1,
        warmup_steps=10,
        learning_rate_schedule_steps=100,
        steps=100,
        lr_schedule_type=LearningRateScheduleType.COSINE,
    )
    schedule = create_learning_rate_schedule(config)
    npt.assert_allclose(_eval(schedule, 0), 0.0, atol=1e-7)
    npt.assert_allclose(_eval(schedule, 10), 1e-3, rtol=1e-5)


def test_config_dispatch_wsd():
    from types import SimpleNamespace
    from megatext.configs.types import LearningRateScheduleType

    config = SimpleNamespace(
        learning_rate=1e-3,
        learning_rate_final_fraction=0.1,
        warmup_steps=10,
        learning_rate_schedule_steps=100,
        steps=100,
        lr_schedule_type=LearningRateScheduleType.WSD,
        wsd_decay_steps=10,
        wsd_decay_style=WsdDecayStyle.LINEAR,
    )
    schedule = create_learning_rate_schedule(config)
    npt.assert_allclose(_eval(schedule, 50), 1e-3, rtol=1e-5)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_warmup_steps_zero():
    schedule = create_cosine_schedule(lr=1e-3, final_lr=1e-4, warmup_steps=0, total_steps=100, steps=100)
    npt.assert_allclose(_eval(schedule, 0), 1e-3, rtol=1e-5)


def test_single_step():
    schedule = create_cosine_schedule(lr=1e-3, final_lr=1e-4, warmup_steps=0, total_steps=1, steps=1)
    assert _eval(schedule, 0) > 0


def test_warmup_linearly_increases():
    schedule = create_cosine_schedule(lr=1e-3, final_lr=1e-4, warmup_steps=50, total_steps=100, steps=100)
    lrs = [_eval(schedule, s) for s in range(50)]
    for i in range(1, len(lrs)):
        assert lrs[i] > lrs[i - 1], f"LR did not increase at warmup step {i}"


def test_schedule_steps_greater_than_steps():
    schedule = create_cosine_schedule(lr=1e-3, final_lr=1e-4, warmup_steps=10, total_steps=200, steps=100)
    assert _eval(schedule, 99) > 0
