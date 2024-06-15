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

from typing import Tuple

import torch
from torch import nn

from gluonts.core.component import validated
from gluonts.model import Input, InputSpec
from gluonts.torch.distributions import StudentTOutput
from gluonts.torch.scaler import StdScaler, MeanScaler, NOPScaler
from gluonts.torch.model.simple_feedforward import make_linear_layer
from gluonts.torch.util import weighted_average


class MovingAvg(nn.Module):
    """
    Moving average block to highlight the trend of time series.

    Parameters:
    - kernel_size (int): The size of the kernel for the average pooling operation.
    - stride (int): The stride of the average pooling operation.

    Attributes:
    - kernel_size (int): The size of the kernel for the average pooling operation.

    Methods:
    - forward(x): Performs the forward pass of the moving average block.

    """

    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(
            kernel_size=kernel_size, stride=stride, padding=0
        )

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, ...].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, ...].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    """
    Series decomposition block.

    This class represents a series decomposition block that decomposes a time series into its trend and residual components.
    It takes a kernel size as input, which determines the size of the moving average window used for trend estimation.

    Parameters:
    -----------
    kernel_size : int
        The size of the moving average window used for trend estimation.

    Methods:
    --------
    forward(x):
        Performs the forward pass of the series decomposition block.

    Attributes:
    -----------
    moving_avg : MovingAvg
        The moving average module used for trend estimation.
    """

    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DFT_series_decomp(nn.Module):
    """
    Series decomposition block

    This class represents a series decomposition block that performs decomposition of a time series into its seasonal and trend components using Discrete Fourier Transform (DFT).

    Parameters:
    - top_k (int): The number of top frequencies to keep during decomposition. Default is 5.

    Methods:
    - forward(x): Performs the forward pass of the series decomposition block.

    Returns:
    - x_season (torch.Tensor): The seasonal component of the input time series.
    - x_trend (torch.Tensor): The trend component of the input time series.
    """

    def __init__(self, top_k=5):
        super().__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, 5)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend


class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(
        self, context_length, down_sampling_window, down_sampling_layers
    ):
        super().__init__()

        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        context_length // (down_sampling_window**i),
                        context_length // (down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        context_length // (down_sampling_window ** (i + 1)),
                        context_length // (down_sampling_window ** (i + 1)),
                    ),
                )
                for i in range(down_sampling_layers)
            ]
        )

    def forward(self, season_list):
        # mixing high->low
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(
        self, context_length, down_sampling_window, down_sampling_layers
    ):
        super().__init__()

        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        context_length // (down_sampling_window ** (i + 1)),
                        context_length // (down_sampling_window**i),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        context_length // (down_sampling_window**i),
                        context_length // (down_sampling_window**i),
                    ),
                )
                for i in reversed(range(down_sampling_layers))
            ]
        )

    def forward(self, trend_list):
        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    def __init__(
        self,
        context_length,
        down_sampling_window,
        down_sampling_layers,
        d_model,
        d_ff,
        decomp_method,
        kernel_size,
        top_k,
    ):
        super().__init__()

        if decomp_method == "moving_avg":
            self.decompsition = SeriesDecomp(kernel_size)
        elif decomp_method == "dft_decomp":
            self.decompsition = DFT_series_decomp(top_k)
        else:
            raise ValueError("unknown decomp_method")

        # Mixing season
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(
            context_length, down_sampling_window, down_sampling_layers
        )

        # Mixing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(
            context_length, down_sampling_window, down_sampling_layers
        )

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_ff),
            nn.GELU(),
            nn.Linear(in_features=d_ff, out_features=d_model),
        )

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decompsition(x)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        # bottom-up season mixing
        out_season_list = self.mixing_multi_scale_season(season_list)
        # top-down trend mixing
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for ori, out_season, out_trend, length in zip(
            x_list, out_season_list, out_trend_list, length_list
        ):
            out = out_season + out_trend
            out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
        return out_list


class TimeMixerModel(nn.Module):
    """
    Module implementing a feed-forward model form the paper
    https://arxiv.org/pdf/2205.13504.pdf extended for probabilistic
    forecasting.

    Parameters
    ----------
    prediction_length
        Number of time points to predict.
    context_length
        Number of time steps prior to prediction time that the model.
    hidden_dimension
        Size of last hidden layers in the feed-forward network.
    distr_output
        Distribution to use to evaluate observations and sample predictions.
    """

    @validated()
    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        down_sampling_window: int,
        down_sampling_layers: int,
        e_layers: int,
        d_model: int,
        d_ff: int,
        down_sampling_method: str = "max",
        decomp_method: str = "dft_decomp",
        distr_output=StudentTOutput(),
        kernel_size: int = 25,
        top_k: int = 5,
        scaling: str = "mean",
    ) -> None:
        super().__init__()

        assert prediction_length > 0
        assert context_length > 0

        self.prediction_length = prediction_length
        self.context_length = context_length

        self.down_sampling_method = down_sampling_method

        self.pdm_blocks = nn.ModuleList(
            [
                PastDecomposableMixing(
                    context_length=context_length,
                    down_sampling_window=down_sampling_window,
                    d_model=d_model,
                    d_ff=d_ff,
                    decomp_method=decomp_method,
                    kernel_size=kernel_size,
                    top_k=top_k,
                )
                for _ in range(e_layers)
            ]
        )

        self.distr_output = distr_output
        if scaling == "mean":
            self.scaler = MeanScaler(keepdim=True)
        elif scaling == "std":
            self.scaler = StdScaler(keepdim=True)
        else:
            self.scaler = NOPScaler(keepdim=True)

        self.predict_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    prediction_length // (down_sampling_window**i),
                    prediction_length,
                )
                for i in range(down_sampling_layers + 1)
            ]
        )
        self.args_proj = self.distr_output.get_args_proj(d_model)

    def describe_inputs(self, batch_size=1) -> InputSpec:
        return InputSpec(
            {
                "past_target": Input(
                    shape=(batch_size, self.context_length), dtype=torch.float
                ),
                "past_observed_values": Input(
                    shape=(batch_size, self.context_length), dtype=torch.float
                ),
            },
            torch.zeros,
        )

    def forward(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
        # scale the input
        past_target_scaled, loc, scale = self.scaler(
            past_target, past_observed_values
        )

        return distr_args, loc, scale

    def loss(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        future_target: torch.Tensor,
        future_observed_values: torch.Tensor,
    ) -> torch.Tensor:
        distr_args, loc, scale = self(
            past_target=past_target, past_observed_values=past_observed_values
        )
        loss = self.distr_output.loss(
            target=future_target, distr_args=distr_args, loc=loc, scale=scale
        )
        return weighted_average(loss, weights=future_observed_values, dim=-1)
