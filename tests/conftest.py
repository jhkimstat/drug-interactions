from itertools import product

import numpy as np
import pytest

from factorregression.state import Data, ModelSpec


@pytest.fixture
def problem():
    X = np.array(list(product((0, 1), repeat=5)), dtype=float)
    y = np.random.default_rng(81).binomial(1, 0.5, len(X))
    return Data(X, y), ModelSpec(5, 3, {2: 2, 3: 3})
