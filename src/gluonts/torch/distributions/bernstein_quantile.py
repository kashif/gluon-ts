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

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.distributions import Distribution, AffineTransform, TransformedDistribution

from gluonts.core.component import validated
from .distribution_output import DistributionOutput


class BernsteinQuantileDistribution(Distribution):
    r"""
    Distribution class for quantile function approximation using Bernstein polynomials.
    
    Parameters
    ----------
    coefficients
        Tensor of shape (*batch_shape, degree+1) containing the coefficients of
        Bernstein basis polynomials.
    degree
        Degree of Bernstein polynomials.
    """
    
    def __init__(
        self,
        coefficients: torch.Tensor,
        degree: int,
        validate_args: bool = False,
    ) -> None:
        self.coefficients = coefficients
        self.degree = degree
        
        batch_shape = coefficients.shape[:-1]
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def bernstein_basis(self, alpha: torch.Tensor, k: int) -> torch.Tensor:
        """Compute k-th Bernstein basis polynomial of degree n."""
        n = self.degree
        # Compute binomial coefficient
        coef = torch.exp(
            torch.lgamma(torch.tensor(n + 1.))
            - torch.lgamma(torch.tensor(k + 1.))
            - torch.lgamma(torch.tensor(n - k + 1.))
        )
        return coef * (alpha ** k) * ((1 - alpha) ** (n - k))

    def quantile(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        Evaluate quantile function at specified levels using Bernstein polynomials.
        
        Parameters
        ----------
        alpha
            Tensor of shape (*batch_shape) containing quantile levels in [0,1]
        
        Returns
        -------
        Tensor
            Quantile values of shape (*batch_shape)
        """
        # Ensure alpha is in [0,1]
        alpha = torch.clamp(alpha, 0, 1)
        
        # Expand alpha for broadcasting
        alpha_expanded = alpha.unsqueeze(-1)
        
        # Compute all Bernstein basis polynomials
        basis_values = torch.stack([
            self.bernstein_basis(alpha_expanded, k)
            for k in range(self.degree + 1)
        ], dim=-1)
        
        # Compute quantile values as linear combination of basis polynomials
        return torch.sum(basis_values * self.coefficients, dim=-1)

    def cdf(self, y: torch.Tensor) -> torch.Tensor:
        """
        Approximate the CDF using binary search on the quantile function.
        
        Parameters
        ----------
        y
            Tensor of shape (*batch_shape) containing values
            
        Returns
        -------
        Tensor
            CDF values of shape (*batch_shape)
        """
        # Initialize search bounds
        lower = torch.zeros_like(y)
        upper = torch.ones_like(y)
        
        # Binary search
        for _ in range(10):  # Number of iterations for desired precision
            mid = (lower + upper) / 2
            q_mid = self.quantile(mid)
            lower = torch.where(q_mid < y, mid, lower)
            upper = torch.where(q_mid < y, upper, mid)
            
        return (lower + upper) / 2

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """
        Generate random samples using inverse transform sampling.
        """
        alpha = torch.rand(
            sample_shape + self.batch_shape,
            device=self.coefficients.device,
        )
        return self.quantile(alpha)

    def crps(self, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the Continuous Ranked Probability Score.
        """
        # Approximate CRPS using numerical integration
        alpha = torch.linspace(0, 1, 100, device=y.device)
        quantiles = self.quantile(alpha)
        
        # Compute integrand
        indicator = (quantiles.unsqueeze(-1) >= y.unsqueeze(-2)).float()
        integrand = (indicator - alpha.unsqueeze(-1)) ** 2
        
        # Numerical integration using trapezoidal rule
        return torch.trapz(integrand, alpha, dim=-2)


class BernsteinQuantileOutput(DistributionOutput):
    r"""
    Distribution output class for quantile function approximation using Bernstein polynomials.
    
    Parameters
    ----------
    degree
        Degree of Bernstein polynomials to use.
    """
    
    distr_cls: type = BernsteinQuantileDistribution
    
    @validated()
    def __init__(self, degree: int) -> None:
        super().__init__()
        
        assert isinstance(degree, int) and degree > 0, \
            "degree must be a positive integer"
        
        self.degree = degree
        self.args_dim: Dict[str, int] = {"coefficients": degree + 1}

    def domain_map(self, coefficients: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Ensures coefficients are monotonically increasing by applying cumulative sum
        of positive values.
        """
        # Apply softplus and cumsum to ensure monotonicity
        return (F.softplus(coefficients).cumsum(dim=-1),)

    def distribution(
        self,
        distr_args,
        loc: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
    ) -> Distribution:
        """
        Create distribution instance with given parameters.
        """
        coefficients = distr_args[0]
        distr = self.distr_cls(coefficients, self.degree)
        
        if scale is None:
            return distr
        else:
            return TransformedDistribution(
                distr, [AffineTransform(loc=loc, scale=scale)]
            )

    @property
    def event_shape(self) -> Tuple:
        return ()
