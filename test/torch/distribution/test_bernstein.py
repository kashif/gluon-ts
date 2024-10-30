# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from typing import List, Tuple

import torch
import numpy as np
import pytest

from gluonts.torch.distributions import (
    BernsteinQuantileDistribution,
    BernsteinQuantileOutput,
)

@pytest.mark.parametrize(
    "distr, alpha, quantile, target, crps",
    [
        (
            BernsteinQuantileDistribution(
                coefficients=torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32).reshape(1, 3),
                degree=2
            ),
            [0.0, 0.25, 0.5, 0.75, 1.0],
            [0.0, 0.25, 0.5, 0.75, 1.0],  # Expected quantile values
            [0.5],
            [0.0833],  # Expected CRPS value
        ),
        (
            BernsteinQuantileDistribution(
                coefficients=torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32).reshape(1, 3),
                degree=2
            ),
            [0.0, 0.25, 0.5, 0.75, 1.0],
            [0.0, 0.5, 1.0, 1.5, 2.0],  # Expected quantile values
            [1.0],
            [0.1667],  # Expected CRPS value
        ),
    ],
)
def test_values(
    distr: BernsteinQuantileDistribution,
    alpha: List[float],
    quantile: List[float],
    target: List[float],
    crps: List[float],
):
    """Test quantile values and CRPS computation"""
    target = torch.tensor(target).reshape(len(target))
    alpha = torch.tensor(alpha).reshape(len(alpha), len(target))
    quantile = torch.tensor(quantile).reshape((len(quantile), len(target)))
    crps = torch.tensor(crps)

    # Test quantile values
    assert torch.allclose(
        distr.quantile(alpha),
        quantile,
        rtol=1e-3,
        atol=1e-3,
    )

    # Test CRPS computation
    assert torch.allclose(
        distr.crps(target),
        crps,
        rtol=1e-3,
        atol=1e-3,
    )

@pytest.mark.parametrize(
    "batch_shape, degree, num_samples",
    [
        ((3, 4, 5), 5, 100),
        ((1,), 2, 1),
        ((10,), 3, 10),
        ((10, 5), 4, 10),
    ],
)
def test_shapes(
    batch_shape: Tuple,
    degree: int,
    num_samples: int,
):
    """Test shape handling"""
    coefficients = torch.ones((*batch_shape, degree + 1), dtype=torch.float32)
    target = torch.ones(batch_shape, dtype=torch.float32)

    distr = BernsteinQuantileDistribution(coefficients=coefficients, degree=degree)

    # Test batch shape computation
    assert distr.batch_shape == batch_shape

    # Test sample shapes
    samples = distr.sample()
    assert samples.shape == batch_shape

    samples = distr.sample((num_samples,))
    assert samples.shape == (num_samples, *batch_shape)

    # Test quantile shapes
    alpha = torch.rand(batch_shape)
    assert distr.quantile(alpha).shape == batch_shape

    # Test CRPS shape
    assert distr.crps(target).shape == batch_shape

@pytest.mark.parametrize(
    "batch_shape, degree, num_samples",
    [
        ((1000,), 3, 100),
        ((500, 2), 4, 10),
    ],
)
@pytest.mark.parametrize(
    "atol",
    [1e-1],  # Larger tolerance due to sampling
)
def test_consistency(
    batch_shape: Tuple,
    degree: int,
    num_samples: int,
    atol: float,
):
    """Test quantile-cdf consistency and monotonicity"""
    distr_out = BernsteinQuantileOutput(degree=degree)
    args_proj = distr_out.get_args_proj(in_features=30)

    # Generate random inputs
    net_out = torch.normal(mean=0.0, std=1.0, size=(*batch_shape, 30))
    args = args_proj(net_out)
    distr = distr_out.distribution(args)

    # Test quantile(cdf(y)) ≈ y
    y = torch.normal(mean=0.0, std=1.0, size=batch_shape)
    y_approx = distr.quantile(distr.cdf(y))
    assert torch.max(torch.abs(y_approx - y)) < atol

    # Test cdf(quantile(alpha)) ≈ alpha
    alpha = torch.rand(size=batch_shape)
    alpha_approx = distr.cdf(distr.quantile(alpha))
    assert torch.max(torch.abs(alpha_approx - alpha)) < atol

    # Test monotonicity of quantile function
    alpha1 = torch.rand(size=batch_shape)
    alpha2 = alpha1 + 0.1  # Ensure alpha2 > alpha1
    assert torch.all(distr.quantile(alpha2) >= distr.quantile(alpha1))

def test_robustness():
    """Test handling of extreme values"""
    distr_out = BernsteinQuantileOutput(degree=3)
    args_proj = distr_out.get_args_proj(in_features=30)

    # Test with large inputs
    net_out = torch.normal(mean=0.0, size=(100, 30), std=1e2)
    args = args_proj(net_out)
    distr = distr_out.distribution(args)

    # Test quantile function with extreme probabilities
    alpha = torch.tensor([0.0, 1.0])
    q = distr.quantile(alpha)
    assert torch.all(torch.isfinite(q))

    # Test CDF with extreme values
    y = torch.normal(mean=0.0, size=(100,), std=1e2)
    p = distr.cdf(y)
    assert torch.all(torch.isfinite(p))
    assert torch.all(p >= 0) and torch.all(p <= 1)
