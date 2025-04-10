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

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Independent, Normal

from gluonts.core.component import validated
from gluonts.model import Input, InputSpec
from gluonts.torch.scaler import StdScaler, MeanScaler, NOPScaler
# from gluonts.torch.util import unsqueeze_expand

from flow_matching.path import CondOTProbPath
from flow_matching.solver import ODESolver
# from .transport import Transport, ModelType, PathType, WeightType, Sampler
from .ttt import Block, TTTConfig, TTTCache


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
        # Ensure input is at least 3D
        if x.ndim == 1:
            x = x.unsqueeze(0).unsqueeze(-1)  # [L] -> [1, L, 1]
        elif x.ndim == 2:
            x = x.unsqueeze(-1)  # [B, L] -> [B, L, 1]

        batch_size, seq_len, feat_dim = x.shape

        # Handle padding if needed
        if seq_len % self.patch_size != 0:
            padding_size = self.patch_size - (seq_len % self.patch_size)
            padding = torch.full(
                size=(batch_size, padding_size, feat_dim),
                fill_value=torch.nan,
                dtype=x.dtype,
                device=x.device,
            )
            x = torch.cat((padding, x), dim=1)
            seq_len = x.shape[1]

        # Unfold along sequence dimension
        x = x.unfold(
            dimension=1, size=self.patch_size, step=self.patch_stride
        )  # [B, num_patches, patch_size, Feature]

        # Reshape to [B, num_patches, patch_size * Feature]
        return x.reshape(batch_size, -1, self.patch_size * feat_dim)


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


class VelocityModel(nn.Module):
    def __init__(
        self,
        cond_dim: int,
        feat_dim: int,
        hidden_dim: int,
        time_embed_dim: int = 8,
        act_fn_name: str = "gelu",
        cond_drop_prob: float = 0.2,  # Add conditioning dropout probability
        cfg_scale: float = 1.5,
    ):
        super().__init__()

        act_fn = ACT2FN[act_fn_name]
        self.cond_drop_prob = cond_drop_prob
        self.cfg_scale = cfg_scale

        # Time embedding network
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            act_fn,
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Conditioning network for better feature extraction
        self.cond_net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Main velocity network with skip connections
        self.net = nn.Sequential(
            nn.Linear(feat_dim + hidden_dim + time_embed_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Dropout(0.1),  # Add some regularization
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, feat_dim),
        )

        # Initialize weights for better gradient flow
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor):
        # # Handle different time tensor shapes
        # if t.ndim == 0:  # scalar time
        #     t = t.view(1)

        # # Expand t to match batch dimensions of x
        # while t.ndim < x.ndim:
        #     t = t.unsqueeze(-1)
        t = t.expand(*x.shape[:-1], 1)

        # Get time embeddings
        t_embed = self.time_embed(t)

        # Get conditioning features
        cond_features = self.cond_net(cond)

        if self.training:
            # Drop conditioning features with probability cond_drop_prob
            mask = (
                torch.rand(
                    cond_features.shape[:2], device=cond_features.device
                )
                > self.cond_drop_prob
            )
            cond_features = cond_features * mask.unsqueeze(-1)

            # Concatenate all inputs and compute conditioned velocity
            inputs = torch.cat([x, t_embed, cond_features], dim=-1)
            return self.net(inputs)
        else:
            # Compute null velocity
            null_cond_features = torch.zeros_like(cond_features)
            null_inputs = torch.cat([x, t_embed, null_cond_features], dim=-1)
            null_output = self.net(null_inputs)

            # Concatenate all inputs and compute conditioned velocity
            inputs = torch.cat([x, t_embed, cond_features], dim=-1)
            cond_output = self.net(inputs)

            return null_output + self.cfg_scale * (cond_output - null_output)


class Flow(nn.Module):
    def __init__(
        self,
        cond_dim: int,
        feat_dim: int,
        hidden_dim: int,
        act_fn_name: str = "gelu",
    ):
        super().__init__()

        # Define MLP for velocity field
        self.velocity_model = VelocityModel(
            cond_dim, feat_dim, hidden_dim, act_fn_name=act_fn_name
        )
        # Flow matching components
        self.prob_path = CondOTProbPath()

    def compute_loss(
        self, x_0: torch.Tensor, x_1: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        """Compute flow matching loss."""
        batch, seq_len, feat_dim = x_0.shape
        t = torch.rand((batch * seq_len,), device=x_0.device)

        # Get path sample from probability path with scheduler outputs
        path_sample = self.prob_path.sample(
            t=t, x_0=x_0.reshape(-1, feat_dim), x_1=x_1.reshape(-1, feat_dim)
        )

        # Get velocity field prediction
        v_t = self.velocity_model(
            path_sample.x_t.view(batch, seq_len, feat_dim),
            path_sample.t.view(batch, seq_len, 1),
            cond,
        )

        # Flow matching loss
        return F.mse_loss(v_t, path_sample.dx_t.view(batch, seq_len, feat_dim))


# class Flow(nn.Module):
#     def __init__(self, cond_dim: int, out_dim: int, h: int):
#         super().__init__()

#         # Define MLP for velocity field
#         self.velocity_model = VelocityModel(cond_dim, out_dim, h)

#         # Create transport object with velocity model type and linear path
#         self.transport = Transport(
#             model_type=ModelType.VELOCITY,
#             path_type=PathType.LINEAR,
#             loss_type=WeightType.NONE,
#             train_eps=0.0,
#             sample_eps=0.0,
#         )

#         # Create sampler for generating samples
#         self.sampler = Sampler(self.transport)

#     def compute_loss(
#         self, x_1: torch.Tensor, cond: torch.Tensor
#     ) -> torch.Tensor:
#         """Compute flow matching loss."""
#         # Use transport training loss
#         terms = self.transport.training_losses(
#             model=self.velocity_model,
#             x1=x_1,
#             model_kwargs={"cond": cond},
#         )
#         return terms["loss"].mean()

#     # def sample(
#     #     self,
#     #     x_init: torch.Tensor,
#     #     cond: torch.Tensor,
#     #     method: str = "dopri5",
#     #     step_size: float = 0.05,
#     #     return_intermediates: bool = False,
#     #     time_grid: Optional[torch.Tensor] = None,
#     # ) -> torch.Tensor:
#     #     """
#     #     Generate samples using the ODE solver.

#     #     Args:
#     #         x_init: Initial noise tensor
#     #         cond: Conditioning tensor
#     #         method: ODE solver method ('dopri5', 'euler', 'heun', etc.)
#     #         step_size: Step size for fixed-step solvers
#     #         return_intermediates: Whether to return intermediate states
#     #         time_grid: Optional time points for sampling. If None, uses default grid

#     #     Returns:
#     #         Generated samples
#     #     """
#     #     if time_grid is None:
#     #         time_grid = torch.linspace(0, 1.0, int(1.0/step_size) + 1, device=x_init.device)

#     #     # Get ODE sampler with specified method
#     #     ode_sampler = self.sampler.sample_ode(
#     #         sampling_method=method,
#     #         num_steps=len(time_grid) if method in ['euler', 'heun'] else 50,
#     #         atol=1e-5,
#     #         rtol=1e-5,
#     #     )

#     #     # Sample using the velocity model
#     #     samples = ode_sampler(
#     #         x_init,
#     #         model=self.velocity_model,
#     #         cond=cond,
#     #     )

#     #     if return_intermediates:
#     #         return samples
#     #     return samples[-1]  # Return only final state if intermediates not requested

#     def sample(
#         self,
#         x_init: torch.Tensor,
#         cond: torch.Tensor,
#         method: str = "Euler",
#         step_size: float = 0.05,
#         return_intermediates: bool = False,
#         time_grid: Optional[torch.Tensor] = None,
#         diffusion_form: str = "linear",
#         diffusion_norm: float = 1.0,
#     ) -> torch.Tensor:
#         """
#         Generate samples using the SDE solver.

#         Args:
#             x_init: Initial noise tensor
#             cond: Conditioning tensor
#             method: SDE solver method ('Euler', 'Heun')
#             step_size: Step size for fixed-step solvers
#             return_intermediates: Whether to return intermediate states
#             time_grid: Optional time points for sampling. If None, uses default grid
#             diffusion_form: Form of diffusion coefficient ('linear', 'constant', 'SBDM', etc.)
#             diffusion_norm: Scale of the diffusion coefficient

#         Returns:
#             Generated samples
#         """
#         num_steps = (
#             int(1.0 / step_size) + 1 if time_grid is None else len(time_grid)
#         )

#         # Get SDE sampler with specified method
#         sde_sampler = self.sampler.sample_sde(
#             sampling_method=method,
#             diffusion_form=diffusion_form,
#             diffusion_norm=diffusion_norm,
#             last_step="Mean",  # Use mean for last step correction
#             last_step_size=step_size,
#             num_steps=num_steps,
#         )

#         # Sample using the velocity model
#         samples = sde_sampler(
#             x_init,
#             model=self.velocity_model,
#             cond=cond,
#         )

#         # ode_sampler = self.sampler.sample_ode(
#         #     sampling_method=method,
#         #     num_steps=num_steps,
#         #     diffusion_form=diffusion_form,
#         #     diffusion_norm=diffusion_norm,
#         # )

#         # samples = ode_sampler(
#         #     x_init,
#         #     model=self.velocity_model,
#         #     cond=cond,
#         # )

#         if return_intermediates:
#             return samples
#         # Return only final state if intermediates not requested
#         return samples[-1]


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
        n_steps: int = 10,
    ) -> None:
        super().__init__()

        self.prediction_length = prediction_length

        self.context_length = context_length_multiplier * patch_len
        self.patch_len = patch_len
        self.d_model = d_model
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_parallel_samples = num_parallel_samples
        self.n_steps = n_steps

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

        # layer_norm_eps: float = 1e-5
        # decoder_layer = nn.TransformerEncoderLayer(
        #     d_model=d_model,
        #     nhead=nhead,
        #     dim_feedforward=dim_feedforward,
        #     dropout=dropout,
        #     activation="gelu",
        #     layer_norm_eps=layer_norm_eps,
        #     batch_first=True,
        #     norm_first=norm_first,
        # )
        # decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

        # self.decoder = nn.TransformerEncoder(
        #     decoder_layer, num_decoder_layers, decoder_norm
        # )
        self.config = TTTConfig(
            hidden_size=d_model,
            num_attention_heads=nhead,
            intermediate_size=dim_feedforward,
            num_hidden_layers=num_decoder_layers,
            hidden_act=activation,
            ttt_layer_type="mlp",
        )
        self.layers = nn.ModuleList(
            [
                Block(self.config, layer_idx)
                for layer_idx in range(num_decoder_layers)
            ]
        )

        self.flow = Flow(
            cond_dim=d_model,
            feat_dim=patch_len,
            hidden_dim=flow_hidden_dim,
            act_fn_name=activation,
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
        cache_params: Optional[TTTCache] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if future_target is not None and future_observed_values is not None:
            past_target = torch.cat((past_target, future_target), dim=1)
            past_observed_values = torch.cat(
                (past_observed_values, future_observed_values), dim=1
            )

        patched_target = self.patch(past_target)

        # scale the input
        target_scaled, loc, scale = self.scaler(
            patched_target, ~patched_target.isnan()
        )

        # do patching for time features as well
        # if self.num_feat_dynamic_real > 0:
        #     time_feat = torch.cat((past_time_feat, future_time_feat), dim=1)
        #     patched_time_feat = self.patch(time_feat)

        # add loc and scale to past_target_patches as additional features
        log_abs_loc = loc.sign() * loc.abs().log1p()
        log_scale = scale.log()

        # expanded_static_feat = unsqueeze_expand(
        #     torch.cat([log_abs_loc, log_scale], dim=-1),
        #     dim=1,
        #     size=patched_target.shape[1],
        # )
        static_feat = torch.cat([log_abs_loc, log_scale], dim=-1)
        inputs = torch.cat((target_scaled, static_feat), dim=-1)

        if future_time_feat is not None:
            past_time_feat = torch.cat(
                (past_time_feat, future_time_feat), dim=1
            )
        patched_time_feat = self.patch(past_time_feat)[:, 1:, :]

        if future_target is not None:
            # shift the time featur patches by one and pad the very last patch with zeros:
            patched_time_feat = torch.cat(
                (
                    patched_time_feat,
                    torch.zeros_like(patched_time_feat[:, -1, :]).unsqueeze(1),
                ),
                dim=1,
            )

        # if self.num_feat_dynamic_real > 0:
        #     inputs = torch.cat((inputs, patched_time_feat), dim=-1)
        # project the input embeddings to the model dimension
        input_embeddings = self.input_patch_embedding(
            torch.cat((inputs, patched_time_feat), dim=-1)
        )

        # causal mask for the transformer decoder
        mask = nn.Transformer.generate_square_subsequent_mask(
            input_embeddings.shape[1], device=input_embeddings.device
        )
        # transformer encoder with positional encoding
        hidden_states = input_embeddings
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=mask,
                position_ids=torch.arange(
                    input_embeddings.shape[1], device=input_embeddings.device
                ).unsqueeze(0),
                cache_params=cache_params,
            )
        # dec_out = self.decoder(input_embeddings, is_causal=True, mask=mask)

        # # Project decoder output to condition the flow
        # flow_cond = self.proj(dec_out)
        return hidden_states, target_scaled, loc, scale, cache_params

    def loss(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_target: torch.Tensor,
        future_observed_values: torch.Tensor,
        past_time_feat: Optional[torch.Tensor] = None,
        future_time_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        flow_cond, target_scaled, _, _, _ = self.params_from_decoder_output(
            past_target=past_target,
            past_observed_values=past_observed_values,
            future_target=future_target,
            future_observed_values=future_observed_values,
            past_time_feat=past_time_feat,
            future_time_feat=future_time_feat,
        )

        # # Get patches for target
        # target = self.patch(
        #     (torch.cat((past_target, future_target), dim=1) - loc) / scale
        # )

        # Flow matching loss
        # Target patches
        x_1 = target_scaled[:, 1:, :]

        # source distribution
        x_0 = torch.randn_like(x_1)  # Random noise source distribution
        # x_0 = target[:, :-1, :] + torch.randn_like(target[:, :-1, :]) * 0.7

        return self.flow.compute_loss(
            x_1=x_1, x_0=x_0, cond=flow_cond[:, :-1, :]
        )

    def log_prob(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_target: torch.Tensor,
        future_observed_values: torch.Tensor,
        past_time_feat: Optional[torch.Tensor] = None,
        future_time_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = past_target.device
        # gaussian_log_density = MultivariateNormal(
        #     torch.zeros(self.patch_len, device=device),
        #     torch.eye(self.patch_len, device=device),
        # ).log_prob
        gaussian_log_density = Independent(
            Normal(
                torch.zeros(self.patch_len, device=device),
                torch.ones(self.patch_len, device=device),
            ),
            1,
        ).log_prob

        flow_cond, target_scaled, loc, scale, _ = self.params_from_decoder_output(
            past_target=past_target,
            past_observed_values=past_observed_values,
            past_time_feat=past_time_feat,
            future_time_feat=future_time_feat,
            future_target=future_target,
            future_observed_values=future_observed_values,
        )
        # Get patches for target
        # target = self.patch(
        #     (torch.cat((past_target, future_target), dim=1) - loc) / scale
        # )

        # Flow matching loss
        x_1 = target_scaled[:, 1:, :]  # Target patches
        cond = flow_cond[:, :-1, :]

        solver = ODESolver(self.flow.velocity_model)

        _, exact_log_p = solver.compute_likelihood(
            x_1=x_1.reshape(-1, self.patch_len),
            cond=cond.reshape(-1, self.d_model),
            method="midpoint",
            step_size=0.05,
            exact_divergence=True,
            log_p0=gaussian_log_density,
        )
        return -exact_log_p.mean()

    @torch.inference_mode()
    def forward(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_time_feat: Optional[torch.Tensor] = None,
        future_time_feat: Optional[torch.Tensor] = None,
        num_parallel_samples: Optional[int] = None,
        cache_params: Optional[TTTCache] = None,
    ):
        if num_parallel_samples is None:
            num_parallel_samples = self.num_parallel_samples

        # Initialize TTTCache if not provided
        if cache_params is None:
            cache_params = TTTCache(self, batch_size=past_target.shape[0], device=past_target.device)

        # Get initial flow conditioning from decoder
        flow_cond, _, past_loc, past_scale, cache_params = self.params_from_decoder_output(
            past_target=past_target,
            past_observed_values=past_observed_values,
            past_time_feat=past_time_feat,
            future_time_feat=future_time_feat[:, : self.patch_len]
            if future_time_feat is not None
            else None,
            cache_params=cache_params,
        )
        loc = past_loc[:, -1, :]
        scale = past_scale[:, -1, :]

        # Initialize samples for each batch
        batch_size = past_target.shape[0]

        # Sample initial noise for each batch and parallel sample
        x = torch.randn(
            batch_size * num_parallel_samples,
            self.patch_len,
            device=past_target.device,
        )
        # add it to the very last patch of past_target of size self.patch_len
        # x = x + (
        #     (past_target[:, -self.patch_len :] - loc) / scale
        # ).repeat_interleave(num_parallel_samples, dim=0)

        # # the very last patch from past_target
        # x = (
        #     (self.patch(past_target)[:, -1, :] - loc) / scale
        # ).repeat_interleave(num_parallel_samples, dim=0)

        T = torch.linspace(0, 1, self.n_steps + 1, device=x.device)
        # Get last decoder output and repeat for parallel samples
        last_cond = flow_cond[:, -1, :].repeat_interleave(
            num_parallel_samples, dim=0
        )

        # solver
        solver = ODESolver(self.flow.velocity_model)

        # Evolve the samples through time using the flow
        x = solver.sample(
            x_init=x,
            cond=last_cond,
            method="midpoint",
            step_size=0.05,
            return_intermediates=False,
            time_grid=T,
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
            # Calculate the current offset for future_time_feat
            time_feat_offset = min(
                total_samples + self.patch_len, self.prediction_length
            )
            current_future_time_feat = (
                repeat_future_time_feat[:, total_samples:time_feat_offset]
                if future_time_feat is not None
                else None
            )

            # Update the cache sequence length offset for the next autoregressive step
            cache_params.seqlen_offset += future_samples_flat.shape[1] // self.patch_len

            flow_cond, _, _, _, cache_params = self.params_from_decoder_output(
                past_target=repeat_past_target,
                past_observed_values=repeat_past_observed_values,
                past_time_feat=repeat_past_time_feat
                if past_time_feat is not None
                else None,
                future_target=future_samples_flat,
                future_observed_values=torch.ones_like(future_samples_flat),
                future_time_feat=current_future_time_feat,
                cache_params=cache_params,
            )

            # Sample new source sample for next patch
            x = torch.randn(
                batch_size * num_parallel_samples,
                self.patch_len,
                device=past_target.device,
            )

            # # the very last patch from past_target
            # x = (future_samples[-1].view(batch_size*num_parallel_samples, -1) - loc) / scale

            # Use updated conditioning from decoder
            last_cond = flow_cond[:, -1, :]

            # Evolve the new samples
            # for i in range(self.n_steps):
            #     x = self.flow.step(
            #         x_t=x,
            #         t_start=time_steps[i],
            #         t_end=time_steps[i + 1],
            #         cond=last_cond,
            #     )
            x = solver.sample(
                x_init=x,
                cond=last_cond,
                method="midpoint",
                step_size=0.05,
                return_intermediates=False,
                time_grid=T,
            )

            # Scale and store the samples
            next_sample = x.view(
                batch_size, num_parallel_samples, self.patch_len
            ) * scale.view(batch_size, 1, -1) + loc.view(batch_size, 1, -1)
            future_samples.append(next_sample)
            total_samples += self.patch_len

        # Concatenate and trim to prediction length
        future_samples_concat = torch.cat(future_samples, dim=-1)
        return future_samples_concat[..., : self.prediction_length]
