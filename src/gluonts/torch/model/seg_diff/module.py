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
                size=padding_size, fill_value=torch.nan, dtype=x.dtype, device=x.device
            )
            x = torch.concat((padding, x), dim=-1)

        x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
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
        distr_output=StudentTOutput(),
    ) -> None:
        super().__init__()

        self.prediction_length = prediction_length

        self.context_length = context_length_multiplier * patch_len
        self.patch_len = patch_len
        self.context_length_multiplier = context_length_multiplier
        self.d_model = d_model
        self.distr_output = distr_output
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

        self.proj = nn.Linear(d_model, patch_len * context_length_multiplier)

        self.args_proj = self.distr_output.get_args_proj(context_length_multiplier)

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
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:

        if future_target is not None and future_observed_values is not None:
            past_target = torch.cat((past_target, future_target), dim=1)
            past_observed_values = torch.cat((past_observed_values, future_observed_values), dim=1)
            
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
        log_abs_loc = loc.abs().log1p()
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
        mask = nn.Transformer.generate_square_subsequent_mask(input_embeddings.shape[1], device=input_embeddings.device)
        # transformer encoder with positional encoding
        dec_out = self.decoder(input_embeddings, is_causal=True, mask=mask)
        
        num_patches = dec_out.shape[1]
        # flatten and project to [batch_size, num_patches, patch_len=prediction_length, self.context_length_multiplier]
        dec_proj = self.proj(dec_out).reshape(-1, num_patches, self.patch_len, self.context_length_multiplier)

        # project to distribution for each predciction length by mapping the last dimension to the distribution  parameters
        distr_args = self.args_proj(dec_proj)
        return distr_args, loc, scale

    def loss(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_target: torch.Tensor,
        future_observed_values: torch.Tensor,
        past_time_feat: Optional[torch.Tensor] = None,
        future_time_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        distr_args, loc, scale = self.params_from_decoder_output(
            past_target=past_target,
            past_observed_values=past_observed_values,
            past_time_feat=past_time_feat,
            future_time_feat=future_time_feat,
            future_target=future_target,
            future_observed_values=future_observed_values,
        )
        # take all but the last distribution arguments for the future
        # shape [[B, T, patch_len], [B, T, patch_len], ...]  are all the arguments
        distr_args = [params[:, :-1, :] for params in distr_args]

        target = self.patch(torch.cat((past_target, future_target), dim=1))
        observed_values = self.patch(torch.cat((past_observed_values, future_observed_values), dim=1))

        loss = self.distr_output.loss(
            target=target[:, 1:, :], distr_args=distr_args, loc=loc.unsqueeze(-1), scale=scale.unsqueeze(-1)
        )
        return weighted_average(loss, weights=observed_values[:, 1:, :], dim=-1)

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

        distr_args, loc, scale = self.params_from_decoder_output(
            past_target=past_target,
            past_observed_values=past_observed_values,
            past_time_feat=past_time_feat,
        )

        # repeat the parameters for each parallel sample
        repeated_scale = scale.repeat_interleave(repeats=num_parallel_samples, dim=0)
        repeated_loc = loc.repeat_interleave(repeats=num_parallel_samples, dim=0)
        
        repeated_past_target = past_target.repeat_interleave(repeats=num_parallel_samples, dim=0)
        repeated_past_observed_values = past_observed_values.repeat_interleave(repeats=num_parallel_samples, dim=0)

        repeated_past_time_feat = past_time_feat.repeat_interleave(repeats=num_parallel_samples, dim=0) if past_time_feat is not None else None
        repeated_future_time_feat = future_time_feat.repeat_interleave(repeats=num_parallel_samples, dim=0) if future_time_feat is not None else None

        # take the very last distribution arguments to sample the next patch_len time steps
        repeated_distr_args = [params[:, -1, ...].repeat_interleave(repeats=num_parallel_samples, dim=0) for params in distr_args]

        next_sample = self.distr_output.distribution(repeated_distr_args, loc=repeated_loc, scale=repeated_scale).sample()
        future_samples = [next_sample]
        total_samples = self.patch_len

        # sample the next patch_len time steps until the prediction length is reached
        while total_samples < self.prediction_length:
            repeated_past_target = torch.cat(
                (repeated_past_target, next_sample),
                dim=1,
            )
            repeated_past_observed_values = torch.cat(
                (repeated_past_observed_values, torch.ones_like(next_sample)),
                dim=1,
            )
            if repeated_past_time_feat is not None and repeated_future_time_feat is not None:
                repeated_past_time_feat = torch.cat(
                    (repeated_past_time_feat, repeated_future_time_feat[:, total_samples - next_sample.shape[1] : total_samples]),
                    dim=1,
                )
            distr_args, _, _ = self.params_from_decoder_output(
                past_target=repeated_past_target,
                past_observed_values=repeated_past_observed_values,
                past_time_feat=repeated_past_time_feat,
            )
            repeated_distr_args = [params[:, -1, ...] for params in distr_args]
            next_sample = self.distr_output.distribution(repeated_distr_args, loc=repeated_loc, scale=repeated_scale).sample()
            future_samples.append(next_sample)
            total_samples += self.patch_len

        future_samples_concat = torch.cat(future_samples, dim=1)

        # Trim any extra predictions
        future_samples_concat = future_samples_concat[:, :self.prediction_length]

        # reshape the samples to the desired shape
        return future_samples_concat.reshape(
            (-1, num_parallel_samples, self.prediction_length)
        )
        