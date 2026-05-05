import random

import numpy as np
import torch

from datalus.infrastructure.checkpointing import (
    capture_rng_state,
    restore_rng_state,
    seed_everything,
)


def test_rng_state_roundtrip():
    seed_everything(123)
    state = capture_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(1))
    restore_rng_state(state)
    actual = (random.random(), np.random.rand(), torch.rand(1))
    assert expected[0] == actual[0]
    assert expected[1] == actual[1]
    assert torch.equal(expected[2], actual[2])
