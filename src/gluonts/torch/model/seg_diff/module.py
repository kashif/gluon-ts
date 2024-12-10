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

from typing import Optional, Tuple
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from gluonts.core.component import validated
from gluonts.model import Input, InputSpec
from gluonts.torch.distributions import StudentTOutput
from gluonts.torch.scaler import StdScaler, MeanScaler, NOPScaler
from gluonts.torch.util import take_last, unsqueeze_expand, weighted_average
from gluonts.torch.model.simple_feedforward import make_linear_layer


class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)


ACT2CLS = {
    "gelu": nn.GELU,
    "leaky_relu": nn.LeakyReLU,
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "sigmoid": nn.Sigmoid,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "tanh": nn.Tanh,
}
ACT2FN = ClassInstantier(ACT2CLS)


class Patch(nn.Module):
    def __init__(self, patch_size: int, patch_stride: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]

        if length % self.patch_size != 0:
            padding_size = (
                *x.shape[:-1],
                self.patch_size - (length % self.patch_size),
            )
            padding = torch.full(
                size=padding_size,
                fill_value=torch.nan,
                dtype=x.dtype,
                device=x.device,
            )
            x = torch.concat((padding, x), dim=-1)

        x = x.unfold(
            dimension=-1, size=self.patch_size, step=self.patch_stride
        )
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        out_dim: int,
        act_fn_name: str,
        dropout_p: float = 0.0,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout_p)
        self.hidden_layer = nn.Linear(in_dim, h_dim)
        self.act = ACT2FN[act_fn_name]
        self.output_layer = nn.Linear(h_dim, out_dim)
        self.residual_layer = nn.Linear(in_dim, out_dim)

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor):
        hid = self.act(self.hidden_layer(x))
        out = self.dropout(self.output_layer(hid))
        res = self.residual_layer(x)

        out = out + res

        if self.use_layer_norm:
            return self.layer_norm(out)
        return out


class Flow(nn.Module):
    def __init__(self, cond_dim: int, out_dim: int, h: int):
        super().__init__()

        self.linear = nn.Linear(out_dim + cond_dim + 1, h)
        self.act = ACT2FN["gelu"]
        self.output_layer = nn.Linear(h, out_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor):
        x = torch.cat((x_t, t, cond), dim=-1)
        x = self.linear(x)
        x = self.act(x)
        x = self.output_layer(x)
        return x

    @torch.inference_mode()
    def step(
        self,
        x_t: torch.Tensor,
        t_start: float,
        t_end: float,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Performs one step of the flow matching process.

        Args:
            x_t: Input tensor to evolve
            t_start: Starting time
            t_end: Ending time
            cond: Conditioning tensor from transformer decoder
        """
        # Expand t_start to match batch dimension
        t_start = torch.full((x_t.shape[0], 1), t_start, device=x_t.device)
        t_mid = t_start + (t_end - t_start) / 2

        # First half step
        v1 = self(x_t=x_t, t=t_start, cond=cond)
        x_mid = x_t + v1 * (t_end - t_start) / 2

        # Second half step
        v2 = self(x_t=x_mid, t=t_mid, cond=cond)
        x_end = x_t + v2 * (t_end - t_start)

        return x_end


class SegDiffModel(nn.Module):
    """
    Module implementing the SegDiff model for forecasting.

    Parameters
    ----------


    num_feat_dynamic_real
        Number of dynamic real features in the data (default: 0).
    distr_output
        Distribution to use to evaluate observations and sample predictions.
        Default: ``StudentTOutput()``.
    """

    @validated()
    def __init__(
        self,
        prediction_length: int,
        context_length_multiplier: int,
        patch_len: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_feat_dynamic_real: int,
        dropout: float,
        activation: str,
        norm_first: bool,
        num_decoder_layers: int,
        scaling: str,
        dropout_rate: float = 0.1,
        num_parallel_samples: int = 100,
        flow_hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        self.prediction_length = prediction_length

        self.context_length = context_length_multiplier * patch_len
        self.patch_len = patch_len
        self.d_model = d_model
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_parallel_samples = num_parallel_samples

        if scaling == "mean":
            self.scaler = MeanScaler(keepdim=True)
        elif scaling == "std":
            self.scaler = StdScaler(keepdim=True)
        else:
            self.scaler = NOPScaler(keepdim=True)

        self.patch = Patch(
            patch_size=patch_len,
            patch_stride=patch_len,
        )

        self.input_patch_embedding = ResidualBlock(
            in_dim=patch_len + 2 + num_feat_dynamic_real * patch_len,
            h_dim=dim_feedforward,
            out_dim=d_model,
            act_fn_name=activation,
            dropout_p=dropout_rate,
        )

        layer_norm_eps: float = 1e-5
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
            norm_first=norm_first,
        )
        decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.decoder = nn.TransformerEncoder(
            decoder_layer, num_decoder_layers, decoder_norm
        )

        self.flow = Flow(
            cond_dim=d_model, out_dim=patch_len, h=flow_hidden_dim
        )

    def describe_inputs(self, batch_size=1) -> InputSpec:
        if self.num_feat_dynamic_real > 0:
            input_spec_feat = {
                "past_time_feat": Input(
                    shape=(
                        batch_size,
                        self.context_length,
                        self.num_feat_dynamic_real,
                    ),
                    dtype=torch.float,
                ),
                "future_time_feat": Input(
                    shape=(
                        batch_size,
                        self.patch_len,
                        self.num_feat_dynamic_real,
                    ),
                    dtype=torch.float,
                ),
            }
        else:
            input_spec_feat = {}

        return InputSpec(
            {
                "past_target": Input(
                    shape=(batch_size, self.context_length), dtype=torch.float
                ),
                "past_observed_values": Input(
                    shape=(batch_size, self.context_length), dtype=torch.float
                ),
                **input_spec_feat,
            },
            torch.zeros,
        )

    def params_from_decoder_output(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_target: Optional[torch.Tensor] = None,
        future_observed_values: Optional[torch.Tensor] = None,
        past_time_feat: Optional[torch.Tensor] = None,
        future_time_feat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if future_target is not None and future_observed_values is not None:
            past_target = torch.cat((past_target, future_target), dim=1)
            past_observed_values = torch.cat(
                (past_observed_values, future_observed_values), dim=1
            )

        # scale the input
        past_target_scaled, loc, scale = self.scaler(
            past_target, past_observed_values
        )
        patched_past_target = self.patch(past_target_scaled)

        # do patching for time features as well
        if self.num_feat_dynamic_real > 0:
            time_feat = torch.cat((past_time_feat, future_time_feat), dim=1)
            patched_time_feat = self.patch(time_feat)

        # add loc and scale to past_target_patches as additional features
        log_abs_loc = loc.sign() * loc.abs().log1p()
        log_scale = scale.log()

        expanded_static_feat = unsqueeze_expand(
            torch.cat([log_abs_loc, log_scale], dim=-1),
            dim=1,
            size=patched_past_target.shape[1],
        )
        inputs = torch.cat((patched_past_target, expanded_static_feat), dim=-1)

        if self.num_feat_dynamic_real > 0:
            inputs = torch.cat((inputs, patched_time_feat), dim=-1)
        # project the input embeddings to the model dimension
        input_embeddings = self.input_patch_embedding(inputs)

        # causal mask for the transformer decoder
        mask = nn.Transformer.generate_square_subsequent_mask(
            input_embeddings.shape[1], device=input_embeddings.device
        )
        # transformer encoder with positional encoding
        dec_out = self.decoder(input_embeddings, is_causal=True, mask=mask)

        # # Project decoder output to condition the flow
        # flow_cond = self.proj(dec_out)
        return dec_out, loc, scale

    def loss(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_target: torch.Tensor,
        future_observed_values: torch.Tensor,
        past_time_feat: Optional[torch.Tensor] = None,
        future_time_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        flow_cond, loc, scale = self.params_from_decoder_output(
            past_target=past_target,
            past_observed_values=past_observed_values,
            past_time_feat=past_time_feat,
            future_time_feat=future_time_feat,
            future_target=future_target,
            future_observed_values=future_observed_values,
        )
        # Get patches for target
        target = self.patch(
            (torch.cat((past_target, future_target), dim=1) - loc) / scale
        )

        # Flow matching loss
        x_1 = target[:, 1:, :]  # Target patches
        x_0 = torch.randn_like(x_1)  # Random noise source distribution
        # t is a tensor of shape (batch_size, num_patches, 1)
        t = torch.rand((x_1.shape[0], x_1.shape[1], 1), device=x_1.device)

        x_t = (1 - t) * x_0 + t * x_1
        dx_t = x_1 - x_0

        # Condition flow on decoder output
        flow_out = self.flow(t=t, x_t=x_t, cond=flow_cond[:, :-1, :])

        return F.mse_loss(flow_out, dx_t)

    def forward(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_time_feat: Optional[torch.Tensor] = None,
        future_time_feat: Optional[torch.Tensor] = None,
        num_parallel_samples: Optional[int] = None,
    ):
        if num_parallel_samples is None:
            num_parallel_samples = self.num_parallel_samples

        # Get initial flow conditioning from decoder
        flow_cond, loc, scale = self.params_from_decoder_output(
            past_target=past_target,
            past_observed_values=past_observed_values,
            past_time_feat=past_time_feat,
        )

        # Initialize samples for each batch
        batch_size = past_target.shape[0]

        # Sample initial noise for each batch and parallel sample
        x = torch.randn(
            batch_size * num_parallel_samples,
            self.patch_len,
            device=past_target.device,
        )

        # Setup time steps for flow
        n_steps = 8
        time_steps = torch.linspace(0, 1.0, n_steps + 1, device=x.device)

        # Get last decoder output and repeat for parallel samples
        last_cond = flow_cond[:, -1, :].repeat_interleave(
            num_parallel_samples, dim=0
        )

        # Evolve the samples through time using the flow
        for i in range(n_steps):
            x = self.flow.step(
                x_t=x,
                t_start=time_steps[i],
                t_end=time_steps[i + 1],
                cond=last_cond,
            )

        # Reshape and scale the samples
        next_sample = x.view(
            batch_size, num_parallel_samples, self.patch_len
        ) * scale.unsqueeze(1) + loc.unsqueeze(1)
        future_samples = [next_sample]
        total_samples = self.patch_len

        # Repeat interleave inputs for parallel sampling
        repeat_past_target = past_target.repeat_interleave(
            num_parallel_samples, dim=0
        )
        repeat_past_observed_values = past_observed_values.repeat_interleave(
            num_parallel_samples, dim=0
        )
        if past_time_feat is not None:
            repeat_past_time_feat = past_time_feat.repeat_interleave(
                num_parallel_samples, dim=0
            )
        if future_time_feat is not None:
            repeat_future_time_feat = future_time_feat.repeat_interleave(
                num_parallel_samples, dim=0
            )

        # Continue sampling until prediction length is reached
        while total_samples < self.prediction_length:
            # Get updated conditioning by feeding previous samples back through decoder
            future_samples_flat = torch.cat(
                future_samples, dim=-1
            )  # Combine all generated patches
            future_samples_flat = future_samples_flat.view(
                batch_size * num_parallel_samples, -1
            )

            flow_cond, loc, scale = self.params_from_decoder_output(
                past_target=repeat_past_target,
                past_observed_values=repeat_past_observed_values,
                past_time_feat=repeat_past_time_feat
                if past_time_feat is not None
                else None,
                future_target=future_samples_flat,
                future_observed_values=torch.ones_like(future_samples_flat),
                future_time_feat=repeat_future_time_feat
                if future_time_feat is not None
                else None,
            )

            # Sample new noise for next patch
            x = torch.randn(
                batch_size * num_parallel_samples,
                self.patch_len,
                device=past_target.device,
            )

            # Use updated conditioning from decoder
            last_cond = flow_cond[:, -1, :]

            # Evolve the new samples
            for i in range(n_steps):
                x = self.flow.step(
                    x_t=x,
                    t_start=time_steps[i],
                    t_end=time_steps[i + 1],
                    cond=last_cond,
                )

            # Scale and store the samples
            next_sample = x.view(
                batch_size, num_parallel_samples, self.patch_len
            ) * scale.view(batch_size, num_parallel_samples, -1) + loc.view(
                batch_size, num_parallel_samples, -1
            )
            future_samples.append(next_sample)
            total_samples += self.patch_len

        # Concatenate and trim to prediction length
        future_samples_concat = torch.cat(future_samples, dim=-1)
        return future_samples_concat[..., : self.prediction_length]
