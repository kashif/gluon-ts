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

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F

from gluonts.core.component import validated
from gluonts.time_feature import get_lags_for_frequency
from gluonts.torch.distributions import (
    DistributionOutput,
    StudentTOutput,
)
from gluonts.torch.scaler import Scaler, MeanScaler, NOPScaler
from gluonts.torch.modules.feature import FeatureEmbedder
from gluonts.torch.util import (
    lagged_sequence_values,
    repeat_along_dim,
    take_last,
    unsqueeze_expand,
)
from gluonts.itertools import prod
from gluonts.model import Input, InputSpec

from flash_stu.utils.numerics import nearest_power_of_two
from flash_stu.utils.stu_utils import get_spectral_filters
from flash_stu.layers.stu_layer import STULayer
from flash_stu.config import FlashSTUConfig


# class STU(nn.Module):
#     """Simple STU Layer in PyTorch with support for d_in != d_out."""

#     def __init__(
#         self,
#         d_in: int = 256,
#         d_out: int = 256,
#         input_len: int = 1024,
#         num_eigh: int = 24,
#         auto_reg_k_u: int = 3,
#         auto_reg_k_y: int = 2,
#         learnable_m_y: bool = True,
#     ) -> None:
#         super().__init__()
#         self.d_in = d_in
#         self.d_out = d_out
#         self.input_len = input_len
#         self.eigh = self.get_top_hankel_eigh(input_len, num_eigh)
#         self.k = num_eigh
#         self.auto_reg_k_u = auto_reg_k_u
#         self.auto_reg_k_y = auto_reg_k_y
#         self.learnable_m_y = learnable_m_y
#         self.m_x_var = 1.0 / (float(self.d_out) ** 0.5)

#         # Initialize parameters
#         self.init_m_y = nn.Parameter(
#             torch.zeros(self.d_out, self.auto_reg_k_y, self.d_out),
#             requires_grad=learnable_m_y,
#         )
#         # Initialize m_u using trunc_normal_ and scaled with m_x_var
#         m_u = nn.Parameter(
#             nn.init.trunc_normal_(
#                 torch.empty(self.d_out, self.d_in, self.auto_reg_k_u)
#             )
#         )
#         self.m_u = m_u * self.m_x_var

#         self.m_phi = nn.Parameter(torch.zeros(self.d_in * self.k, self.d_out))

#         # Initialize state
#         self.reset_state(batch_size=1)

#     def reset_state(self, batch_size: int) -> None:
#         """Reset the state for a new batch."""
#         self.state = {
#             "y": torch.zeros(batch_size, self.auto_reg_k_y, self.d_out),
#             "x": torch.zeros(batch_size, self.auto_reg_k_u, self.d_in),
#         }

#     def get_state(self) -> dict:
#         """Get the current state."""
#         return {k: v.clone() for k, v in self.state.items()}

#     def set_state(self, state: dict) -> None:
#         """Set the current state."""
#         self.state = {k: v.clone() for k, v in state.items()}

#     def get_top_hankel_eigh(
#         self, n: int, k: int
#     ) -> tuple[torch.Tensor, torch.Tensor]:
#         """Get top k eigenvalues and eigenvectors of spectral Hankel matrix."""

#         def get_hankel_matrix(
#             seq_len: int, use_hankel_L: bool = False
#         ) -> np.ndarray:
#             entries = np.arange(1, seq_len + 1, dtype=np.float64)
#             i_plus_j = entries[:, None] + entries[None, :]

#             if use_hankel_L:
#                 sgn = (-1.0) ** (i_plus_j - 2.0) + 1.0
#                 denom = (i_plus_j + 3.0) * (i_plus_j - 1.0) * (i_plus_j + 1.0)
#                 return sgn * (8.0 / denom)
#             else:
#                 return 2.0 / (i_plus_j**3 - i_plus_j)

#         hankel_matrix = get_hankel_matrix(n)
#         eig_vals, eig_vecs = np.linalg.eigh(hankel_matrix)
#         return (
#             torch.from_numpy(eig_vals[-k:]).float(),
#             torch.from_numpy(eig_vecs[:, -k:]).float(),
#         )

#     def compute_x_tilde(self, inputs: torch.Tensor) -> torch.Tensor:
#         """Project input sequence into spectral basis."""
#         eig_vals, eig_vecs = self.eigh
#         b, l, _ = inputs.shape

#         # Use only the relevant part of eig_vecs
#         eig_vecs_trunc = eig_vecs[:l, :]

#         # Compute convolution
#         x_tilde = torch.einsum("lk,bld->bkld", eig_vecs_trunc.to(inputs.device), inputs)

#         # Apply eigenvalue scaling
#         x_tilde *= eig_vals.view(1, -1, 1, 1).to(inputs.device) ** 0.25

#         # Reshape
#         x_tilde = x_tilde.reshape(b, l, -1)

#         # Shift
#         x_tilde = torch.roll(x_tilde, shifts=2, dims=1)
#         x_tilde[:, :2] = 0  # Zero out the first two elements

#         return x_tilde

#     def compute_ar_x_preds(self, x: torch.Tensor) -> torch.Tensor:
#         """Compute the auto-regressive component of spectral SSM."""
#         b, l, _ = x.shape
#         o = torch.einsum("oik,bli->bklo", self.m_u.to(x.device), x)

#         # Roll and mask
#         o = torch.stack(
#             [
#                 torch.roll(o[:, i], shifts=i, dims=1)
#                 for i in range(self.auto_reg_k_u)
#             ],
#             dim=1,
#         )
#         mask = (
#             torch.triu(torch.ones(self.auto_reg_k_u, l))
#             .unsqueeze(0)
#             .unsqueeze(-1)
#             .to(x.device)
#         )

#         return torch.sum(o * mask, dim=1)

#     def forward(
#         self, inputs: torch.Tensor, initial_state: dict = None
#     ) -> tuple[torch.Tensor, dict]:
#         """Forward pass with support for variable input lengths up to input_len."""
#         b, l, d_in = inputs.shape
#         assert (
#             d_in == self.d_in
#         ), f"Input dimension {d_in} does not match expected dimension {self.d_in}"
#         assert (
#             l <= self.input_len
#         ), f"Input sequence length {l} exceeds maximum length {self.input_len}"

#         # Set initial state if provided, otherwise reset
#         if initial_state is not None:
#             self.set_state(initial_state)
#         else:
#             self.reset_state(batch_size=b)

#         # Create a fresh copy of the state for this forward pass
#         current_state = {k: v.clone() for k, v in self.state.items()}

#         # Pad input if necessary
#         if l < self.input_len:
#             pad_length = self.input_len - l
#             inputs_padded = torch.nn.functional.pad(
#                 inputs, (0, 0, 0, pad_length)
#             )
#         else:
#             inputs_padded = inputs

#         x_tilde = self.compute_x_tilde(inputs_padded)
#         delta_phi = torch.einsum("blk,ko->blo", x_tilde, self.m_phi)
#         delta_ar_u = self.compute_ar_x_preds(inputs_padded)
        
#         # Use the current_state copy for compute_y_t
#         output = self.compute_y_t(self.init_m_y, delta_phi + delta_ar_u, current_state)

#         # Update state['x'] in the copy
#         current_state["x"] = torch.roll(current_state["x"], shifts=-l, dims=1)
#         current_state["x"][:, -l:] = inputs[:, -self.auto_reg_k_u:]

#         # Update the internal state with the final values
#         self.state = {k: v.clone() for k, v in current_state.items()}

#         # Return only the non-padded output and the final state
#         return output[:, :l, :], self.get_state()

#     def compute_y_t(
#         self, m_y: torch.Tensor, deltas: torch.Tensor, current_state: dict
#     ) -> torch.Tensor:
#         """Compute sequence of y_t given a series of deltas and m_y."""
#         b, l, _ = deltas.shape
#         ys = []
#         current_state_y = current_state["y"].clone()

#         for i in range(l):
#             output = (
#                 torch.einsum("oky,bky->bo", m_y.to(deltas.device), current_state_y.to(deltas.device))
#                 + deltas[:, i]
#             )
#             ys.append(output)
            
#             # Create new tensor for the rolled state
#             next_state_y = torch.roll(current_state_y.clone(), 1, dims=1)
#             next_state_y[:, 0] = output
#             current_state_y = next_state_y

#         # Update the passed state dictionary
#         current_state["y"] = current_state_y
#         return torch.stack(ys, dim=1)

#     def step(
#         self, x: torch.Tensor, state: Optional[dict] = None
#     ) -> tuple[torch.Tensor, dict]:
#         """Auto-regressive step function using current state."""
#         if state is not None:
#             current_state = {k: v.clone() for k, v in state.items()}
#         else:
#             current_state = {k: v.clone() for k, v in self.state.items()}
        
#         b, d_in = x.shape
#         assert (
#             d_in == self.d_in
#         ), f"Input dimension {d_in} does not match expected dimension {self.d_in}"
#         assert (
#             b == current_state["x"].shape[0]
#         ), "Batch size mismatch with current state"

#         # Update state['x']
#         next_state_x = torch.roll(current_state["x"].clone(), shifts=-1, dims=1)
#         next_state_x[:, -1] = x
#         current_state["x"] = next_state_x

#         # Compute x_tilde for the current step
#         x_expanded = x.unsqueeze(1)  # Shape: (b, 1, d_in)
#         x_tilde = self.compute_x_tilde(x_expanded)[:, 0]  # Shape: (b, k * d_in)

#         # Compute delta_phi
#         delta_phi = torch.matmul(x_tilde, self.m_phi)

#         # Compute delta_ar_u using the current state
#         delta_ar_u = torch.einsum("oik,bki->bo", self.m_u, current_state["x"])

#         # Compute y_t using the current state
#         y_t = (
#             torch.einsum("oky,bky->bo", self.init_m_y.to(x.device), current_state["y"].to(x.device))
#             + delta_phi.to(x.device)
#             + delta_ar_u.to(x.device)
#         )

#         # Update state['y']
#         next_state_y = torch.roll(current_state["y"].clone(), shifts=1, dims=1)
#         next_state_y[:, 0] = y_t
#         current_state["y"] = next_state_y

#         # Update the internal state
#         self.state = {k: v.clone() for k, v in current_state.items()}

#         return y_t, current_state


# class MLP(nn.Module):
#     def __init__(self, config, dtype=None):
#         # https://arxiv.org/pdf/2002.05202
#         super().__init__()
#         dtype = dtype if dtype is not None else config.torch_dtype
#         self.hidden_size = config.n_embd
#         self.intermediate_size = config.n_embd * config.mlp_scale
#         self.gate_proj = nn.Linear(
#             self.hidden_size,
#             self.intermediate_size,
#             bias=config.bias,
#             dtype=dtype,
#         )
#         self.up_proj = nn.Linear(
#             self.hidden_size,
#             self.intermediate_size,
#             bias=config.bias,
#             dtype=dtype,
#         )
#         self.down_proj = nn.Linear(
#             self.intermediate_size,
#             self.hidden_size,
#             bias=config.bias,
#             dtype=dtype,
#         )
#         self.dropout = nn.Dropout(
#             config.dropout
#         )  # TODO: Write Issue in Liger-Kernel repo to support Dropout

#     def forward(self, x):
#         gate = self.gate_proj(x)
#         gate = F.gelu(gate, approximate="tanh")
#         up = self.up_proj(x)
#         fuse = gate * up
#         outputs = self.down_proj(fuse)
#         outputs = self.dropout(outputs)
#         return outputs


# class STULayer(nn.Module):
#     def __init__(self, config, phi, n):
#         super(STULayer, self).__init__()
#         self.stu_norm = nn.RMSNorm(config.n_embd, dtype=config.torch_dtype)

#         self.stu = STU(config, phi, n)
#         self.mlp_norm = nn.RMSNorm(config.n_embd, dtype=config.torch_dtype)
#         self.mlp = MLP(config, dtype=config.torch_dtype)

#         # TODO: Write Issue in Liger-Kernel repo to support user-defined dtype for MLP
#         self.stu_norm = self.stu_norm.to(dtype=config.torch_dtype)
#         self.mlp = self.mlp.to(dtype=config.torch_dtype)
#         self.mlp_norm = self.mlp_norm.to(dtype=config.torch_dtype)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = x + self.stu(self.stu_norm(x))
#         x = x + self.mlp(self.mlp_norm(x))
#         return x


class STUModel(nn.Module):
    """
    Module implementing the STU model https://arxiv.org/abs/2312.06837.


    Parameters
    ----------
    freq
        String indicating the sampling frequency of the data to be processed.
    context_length
        Length of the STU unrolling prior to the forecast date.
    prediction_length
        Number of time points to predict.
    num_feat_dynamic_real
        Number of dynamic real features that will be provided to ``forward``.
    num_feat_static_real
        Number of static real features that will be provided to ``forward``.
    num_feat_static_cat
        Number of static categorical features that will be provided to
        ``forward``.
    cardinality
        List of cardinalities, one for each static categorical feature.
    embedding_dimension
        Dimension of the embedding space, one for each static categorical
        feature.
    num_layers
        Number of layers in the STU.
    hidden_size
        Size of the hidden layers in the STU.
    dropout_rate
        Dropout rate to be applied at training time.
    distr_output
        Type of distribution to be output by the model at each time step
    lags_seq
        Indices of the lagged observations that the STU takes as input. For
        example, ``[1]`` indicates that the STU only takes the observation at
        time ``t-1`` to produce the output for time ``t``; instead,
        ``[1, 25]`` indicates that the STU takes observations at times ``t-1``
        and ``t-25`` as input.
    scaling
        Whether to apply mean scaling to the observations (target).
    default_scale
        Default scale that is applied if the context length window is
        completely unobserved. If not set, the scale in this case will be
        the mean scale in the batch.
    num_parallel_samples
        Number of samples to produce when unrolling the STU in the prediction
        time range.
    nonnegative_pred_samples
        Should final prediction samples be non-negative? If yes, an activation
        function is applied to ensure non-negative. Observe that this is applied
        only to the final samples and this is not applied during training.
    """

    @validated()
    def __init__(
        self,
        freq: str,
        context_length: int,
        prediction_length: int,
        num_feat_dynamic_real: int = 1,
        num_feat_static_real: int = 1,
        num_feat_static_cat: int = 1,
        cardinality: List[int] = [1],
        embedding_dimension: Optional[List[int]] = None,
        num_layers: int = 2,
        hidden_size: int = 40,
        dropout_rate: float = 0.1,
        distr_output: DistributionOutput = StudentTOutput(),
        lags_seq: Optional[List[int]] = None,
        scaling: bool = True,
        default_scale: Optional[float] = None,
        num_parallel_samples: int = 100,
        nonnegative_pred_samples: bool = False,
    ) -> None:
        super().__init__()

        assert distr_output.event_shape == ()
        assert num_feat_dynamic_real > 0
        assert num_feat_static_real > 0
        assert num_feat_static_cat > 0
        assert len(cardinality) == num_feat_static_cat
        assert (
            embedding_dimension is None
            or len(embedding_dimension) == num_feat_static_cat
        )

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.distr_output = distr_output
        self.param_proj = distr_output.get_args_proj(hidden_size)
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_feat_static_cat = num_feat_static_cat
        self.num_feat_static_real = num_feat_static_real
        self.embedding_dimension = (
            embedding_dimension
            if embedding_dimension is not None or cardinality is None
            else [min(50, (cat + 1) // 2) for cat in cardinality]
        )
        self.lags_seq = lags_seq or get_lags_for_frequency(freq_str=freq)
        self.lags_seq = [l - 1 for l in self.lags_seq]
        self.num_parallel_samples = num_parallel_samples
        self.past_length = self.context_length + max(self.lags_seq)
        self.embedder = FeatureEmbedder(
            cardinalities=cardinality,
            embedding_dims=self.embedding_dimension,
        )
        if scaling:
            self.scaler: Scaler = MeanScaler(
                dim=-1, keepdim=True, default_scale=default_scale
            )
        else:
            self.scaler = NOPScaler(dim=-1, keepdim=True)
        self.input_size = len(self.lags_seq) + self._number_of_features

        self.register_buffer("phi", get_spectral_filters(self.context_length + self.prediction_length, 24))
        n = nearest_power_of_two(self.context_length * 2 - 1)

        config = FlashSTUConfig(
            d_in=self.input_size,
            d_out=hidden_size,
            seq_len=self.context_length + self.prediction_length,
            num_eigh=24,
        )
        self.stu = STULayer(
            config, 
            self.phi,
            n,
        )

        self.nonnegative_pred_samples = nonnegative_pred_samples

    def describe_inputs(self, batch_size=1) -> InputSpec:
        return InputSpec(
            {
                "feat_static_cat": Input(
                    shape=(batch_size, self.num_feat_static_cat),
                    dtype=torch.long,
                ),
                "feat_static_real": Input(
                    shape=(batch_size, self.num_feat_static_real),
                    dtype=torch.float,
                ),
                "past_time_feat": Input(
                    shape=(
                        batch_size,
                        self._past_length,
                        self.num_feat_dynamic_real,
                    ),
                    dtype=torch.float,
                ),
                "past_target": Input(
                    shape=(batch_size, self._past_length),
                    dtype=torch.float,
                ),
                "past_observed_values": Input(
                    shape=(batch_size, self._past_length),
                    dtype=torch.float,
                ),
                "future_time_feat": Input(
                    shape=(
                        batch_size,
                        self.prediction_length,
                        self.num_feat_dynamic_real,
                    ),
                    dtype=torch.float,
                ),
            },
            zeros_fn=torch.zeros,
        )

    @property
    def _number_of_features(self) -> int:
        return (
            sum(self.embedding_dimension)
            + self.num_feat_dynamic_real
            + self.num_feat_static_real
            + 1  # the log(scale)
        )

    @property
    def _past_length(self) -> int:
        return self.context_length + max(self.lags_seq)

    def prepare_input(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_target: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        context = past_target[..., -self.context_length :]
        observed_context = past_observed_values[..., -self.context_length :]

        input, _, scale = self.scaler(context, observed_context)
        future_length = future_time_feat.shape[-2]
        if future_length > 1:
            assert future_target is not None
            input = torch.cat(
                (input, future_target[..., : future_length - 1] / scale),
                dim=-1,
            )
        prior_input = past_target[..., : -self.context_length] / scale

        lags = lagged_sequence_values(
            self.lags_seq, prior_input, input, dim=-1
        )

        time_feat = torch.cat(
            (
                take_last(past_time_feat, dim=-2, num=self.context_length - 1),
                future_time_feat,
            ),
            dim=-2,
        )

        embedded_cat = self.embedder(feat_static_cat)
        static_feat = torch.cat(
            (embedded_cat, feat_static_real, scale.log()),
            dim=-1,
        )
        expanded_static_feat = unsqueeze_expand(
            static_feat, dim=-2, size=time_feat.shape[-2]
        )

        features = torch.cat((expanded_static_feat, time_feat), dim=-1)

        return torch.cat((lags, features), dim=-1), scale, static_feat

    def unroll_lagged_stu(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_target: Optional[torch.Tensor] = None,
    ) -> Tuple[
        Tuple[torch.Tensor, ...],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
    ]:
        """
        Applies the underlying STU to the provided target data and covariates.

        Parameters
        ----------
        feat_static_cat
            Tensor of static categorical features,
            shape: ``(batch_size, num_feat_static_cat)``.
        feat_static_real
            Tensor of static real features,
            shape: ``(batch_size, num_feat_static_real)``.
        past_time_feat
            Tensor of dynamic real features in the past,
            shape: ``(batch_size, past_length, num_feat_dynamic_real)``.
        past_target
            Tensor of past target values,
            shape: ``(batch_size, past_length)``.
        past_observed_values
            Tensor of observed values indicators,
            shape: ``(batch_size, past_length)``.
        future_time_feat
            Tensor of dynamic real features in the future,
            shape: ``(batch_size, prediction_length, num_feat_dynamic_real)``.
        future_target
            (Optional) tensor of future target values,
            shape: ``(batch_size, prediction_length)``.

        Returns
        -------
        Tuple
            A tuple containing, in this order:
            - Parameters of the output distribution
            - Scaling factor applied to the target
            - Raw output of the STU
            - Static input to the STU
            - Output state from the STU
        """
        stu_input, scale, static_feat = self.prepare_input(
            feat_static_cat,
            feat_static_real,
            past_time_feat,
            past_target,
            past_observed_values,
            future_time_feat,
            future_target,
        )

        output, new_state = self.stu(stu_input)

        params = self.param_proj(output)
        return params, scale, output, static_feat, new_state

    @torch.jit.ignore
    def output_distribution(
        self, params, scale=None, trailing_n=None
    ) -> torch.distributions.Distribution:
        """
        Instantiate the output distribution.

        Parameters
        ----------
        params
            Tuple of distribution parameters.
        scale
            (Optional) scale tensor.
        trailing_n
            If set, the output distribution is created only for the last
            ``trailing_n`` time points.

        Returns
        -------
        torch.distributions.Distribution
            Output distribution from the model.
        """
        sliced_params = params
        if trailing_n is not None:
            sliced_params = [p[:, -trailing_n:] for p in params]
        return self.distr_output.distribution(sliced_params, scale=scale)

    def post_process_samples(self, samples: torch.Tensor) -> torch.Tensor:
        """
        Method to enforce domain-specific constraints on the generated samples.
        For example, we can enforce forecasts to be nonnegative.
        Parameters
        ----------
        samples
            Tensor of samples
        Returns
        -------
            Tensor of processed samples with the same shape.
        """

        if self.nonnegative_pred_samples:
            return torch.relu(samples)

        return samples

    def forward(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: torch.Tensor,
        num_parallel_samples: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Invokes the model on input data, and produce outputs future samples.

        Parameters
        ----------
        feat_static_cat
            Tensor of static categorical features,
            shape: ``(batch_size, num_feat_static_cat)``.
        feat_static_real
            Tensor of static real features,
            shape: ``(batch_size, num_feat_static_real)``.
        past_time_feat
            Tensor of dynamic real features in the past,
            shape: ``(batch_size, past_length, num_feat_dynamic_real)``.
        past_target
            Tensor of past target values,
            shape: ``(batch_size, past_length)``.
        past_observed_values
            Tensor of observed values indicators,
            shape: ``(batch_size, past_length)``.
        future_time_feat
            (Optional) tensor of dynamic real features in the past,
            shape: ``(batch_size, prediction_length, num_feat_dynamic_real)``.
        num_parallel_samples
            How many future samples to produce.
            By default, self.num_parallel_samples is used.
        """
        if num_parallel_samples is None:
            num_parallel_samples = self.num_parallel_samples

        params, scale, _, static_feat, state = self.unroll_lagged_stu(
            feat_static_cat,
            feat_static_real,
            past_time_feat,
            past_target,
            past_observed_values,
            future_time_feat[:, :1],
        )

        repeated_scale = scale.repeat_interleave(
            repeats=num_parallel_samples, dim=0
        )
        repeated_static_feat = static_feat.repeat_interleave(
            repeats=num_parallel_samples, dim=0
        ).unsqueeze(dim=1)
        repeated_past_target = (
            past_target.repeat_interleave(repeats=num_parallel_samples, dim=0)
            / repeated_scale
        )
        repeated_time_feat = future_time_feat.repeat_interleave(
            repeats=num_parallel_samples, dim=0
        )
        repeated_state = {
            key: value.repeat_interleave(repeats=num_parallel_samples, dim=0)
            for key, value in state.items()
        }
        repeated_params = [
            s.repeat_interleave(repeats=num_parallel_samples, dim=0)
            for s in params
        ]
        distr = self.output_distribution(
            repeated_params, trailing_n=1, scale=repeated_scale
        )
        next_sample = distr.sample()
        future_samples = [next_sample]

        for k in range(1, self.prediction_length):
            scaled_next_sample = next_sample / repeated_scale
            next_features = torch.cat(
                (repeated_static_feat, repeated_time_feat[:, k : k + 1]),
                dim=-1,
            )
            next_lags = lagged_sequence_values(
                self.lags_seq, repeated_past_target, scaled_next_sample, dim=-1
            )
            stu_input = torch.cat((next_lags, next_features), dim=-1)

            output, repeated_state = self.stu.step(stu_input.squeeze(dim=1), repeated_state)

            repeated_past_target = torch.cat(
                (repeated_past_target, scaled_next_sample), dim=1
            )

            params = self.param_proj(output.unsqueeze(dim=1))
            distr = self.output_distribution(params, scale=repeated_scale)
            next_sample = distr.sample()
            future_samples.append(next_sample)

        future_samples_concat = torch.cat(future_samples, dim=1)

        future_samples_concat = self.post_process_samples(
            future_samples_concat
        )

        return future_samples_concat.reshape(
            (-1, num_parallel_samples, self.prediction_length)
        )

    def log_prob(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_target: torch.Tensor,
    ) -> torch.Tensor:
        return -self.loss(
            feat_static_cat=feat_static_cat,
            feat_static_real=feat_static_real,
            past_time_feat=past_time_feat,
            past_target=past_target,
            past_observed_values=past_observed_values,
            future_time_feat=future_time_feat,
            future_target=future_target,
            future_observed_values=torch.ones_like(future_target),
            aggregate_by=torch.sum,
        )

    def loss(
        self,
        feat_static_cat: torch.Tensor,
        feat_static_real: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_target: torch.Tensor,
        future_observed_values: torch.Tensor,
        aggregate_by=torch.mean,
    ) -> torch.Tensor:
        extra_dims = len(future_target.shape) - len(past_target.shape)
        extra_shape = future_target.shape[:extra_dims]
        batch_shape = future_target.shape[: extra_dims + 1]

        repeats = prod(extra_shape)
        feat_static_cat = repeat_along_dim(feat_static_cat, 0, repeats)
        feat_static_real = repeat_along_dim(feat_static_real, 0, repeats)
        past_time_feat = repeat_along_dim(past_time_feat, 0, repeats)
        past_target = repeat_along_dim(past_target, 0, repeats)
        past_observed_values = repeat_along_dim(
            past_observed_values, 0, repeats
        )
        future_time_feat = repeat_along_dim(future_time_feat, 0, repeats)

        future_target_reshaped = future_target.reshape(
            -1,
            *future_target.shape[extra_dims + 1 :],
        )
        future_observed_reshaped = future_observed_values.reshape(
            -1,
            *future_observed_values.shape[extra_dims + 1 :],
        )

        params, scale, _, _, _ = self.unroll_lagged_stu(
            feat_static_cat,
            feat_static_real,
            past_time_feat,
            past_target,
            past_observed_values,
            future_time_feat,
            future_target_reshaped,
        )

        context_target = take_last(
            past_target, dim=-1, num=self.context_length - 1
        )
        target = torch.cat(
            (context_target, future_target_reshaped),
            dim=1,
        )
        context_observed = take_last(
            past_observed_values, dim=-1, num=self.context_length - 1
        )
        observed_values = torch.cat(
            (context_observed, future_observed_reshaped), dim=1
        )
        loss_values = self.distr_output.loss(
            target=target, distr_args=params, scale=scale
        )
        loss_values = loss_values * observed_values

        loss_values = loss_values.reshape(*batch_shape, *loss_values.shape[1:])

        return aggregate_by(
            loss_values,
            dim=tuple(range(extra_dims + 1, len(future_target.shape))),
        )
