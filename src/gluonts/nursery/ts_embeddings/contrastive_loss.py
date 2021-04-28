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


import torch
import torch.nn.functional as F
import numpy
from typing import Optional, Any, Tuple


class NT_Xent_Loss(torch.nn.modules.loss._Loss):
    """
    NT-Xent loss as used in SimCLR. Extension of the code used in:
    https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries/blob/master/losses/triplet_loss.py
    """

    def __init__(
        self,
        compared_length,  # this is the maximum length of the positive example (and also of the negative example)
        temperature=1.0,
    ):
        super().__init__()
        self.compared_length = compared_length
        if self.compared_length is None:
            self.compared_length = numpy.inf
        self.temperature = temperature

    def get_representations(
        self, batch, encoder
    ) -> Tuple[torch.tensor, torch.Tensor]:
        """
        Return representations for the time series and randomly selected subwindows
        """
        batch_size = batch.size(0)
        length = self.compared_length

        # Choice of length of positive and negative samples
        length_pos_neg = numpy.random.randint(1, high=length + 1)

        # We choose for each batch example a random interval in the time
        # series, which is the 'anchor'
        random_length = numpy.random.randint(
            length_pos_neg, high=length + 1
        )  # Length of anchors
        beginning_batches = numpy.random.randint(
            0, high=length - random_length + 1, size=batch_size
        )  # Start of anchors

        # The positive samples are chosen at random in the chosen anchors
        beginning_samples_pos = numpy.random.randint(
            0, high=random_length - length_pos_neg + 1, size=batch_size
        )  # Start of positive samples in the anchors
        # Start of positive samples in the batch examples
        beginning_positive = beginning_batches + beginning_samples_pos
        # End of positive samples in the batch examples
        end_positive = beginning_positive + length_pos_neg

        representation = encoder(
            torch.cat(
                [
                    batch[
                        j : j + 1,
                        :,
                        beginning_batches[j] : beginning_batches[j]
                        + random_length,
                    ]
                    for j in range(batch_size)
                ]
            )
        )  # Anchors representations.

        positive_representation = encoder(
            torch.cat(
                [
                    batch[
                        j : j + 1,
                        :,
                        end_positive[j] - length_pos_neg : end_positive[j],
                    ]
                    for j in range(batch_size)
                ]
            )
        )  # Positive samples representations

        return representation, positive_representation

    def forward(self, representation, augmented_representation):
        # norm_rep = F.normalize(representation, p=2, dim=-1)
        # norm_augmented_rep = F.normalize(augmented_representation, p=2, dim=-1)
        # all = torch.cat([norm_rep, norm_augmented_rep], dim=0)
        all = torch.cat([representation, augmented_representation], dim=0)

        n = all.size(0)

        cos_dist = torch.einsum("ax,bx->ab", all, all)
        S = torch.exp(cos_dist / self.temperature)

        mask = ~torch.eye(n, device=S.device).bool()
        neg = S.masked_select(mask).view(n, -1).sum(dim=-1)
        # neg = torch.clip(neg, eps, numpy.inf)

        pos = torch.exp(
            torch.sum(representation * augmented_representation, dim=-1)
            / self.temperature
        )
        pos = torch.cat([pos, pos], dim=0)

        # todo check whether this should be safe-guarded against NaNs
        loss = -torch.log(pos / neg).mean()

        return loss


class BarlowTwins(torch.nn.Module):
    def __init__(self, compared_length, out_channels, lambd=1e-2, scale_loss=1.0):
        super().__init__()

        self.compared_length = compared_length
        self.lambd = lambd
        self.scale_loss = scale_loss
        self.bn = torch.nn.BatchNorm1d(out_channels, affine=False)

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def get_representations(
        self, batch, encoder
    ) -> Tuple[torch.tensor, torch.Tensor]:
        """
        Return representations for the time series and randomly selected subwindows
        """
        batch_size = batch.size(0)
        length = self.compared_length

        # Choice of length of positive and negative samples
        length_pos_neg = numpy.random.randint(1, high=length + 1)

        # We choose for each batch example a random interval in the time
        # series, which is the 'anchor'
        random_length = numpy.random.randint(
            length_pos_neg, high=length + 1
        )  # Length of anchors
        beginning_batches = numpy.random.randint(
            0, high=length - random_length + 1, size=batch_size
        )  # Start of anchors

        # The positive samples are chosen at random in the chosen anchors
        beginning_samples_pos = numpy.random.randint(
            0, high=random_length - length_pos_neg + 1, size=batch_size
        )  # Start of positive samples in the anchors
        # Start of positive samples in the batch examples
        beginning_positive = beginning_batches + beginning_samples_pos
        # End of positive samples in the batch examples
        end_positive = beginning_positive + length_pos_neg

        representation = encoder(
            torch.cat(
                [
                    batch[
                        j : j + 1,
                        :,
                        beginning_batches[j] : beginning_batches[j]
                        + random_length,
                    ]
                    for j in range(batch_size)
                ]
            )
        )  # Anchors representations

        positive_representation = encoder(
            torch.cat(
                [
                    batch[
                        j : j + 1,
                        :,
                        end_positive[j] - length_pos_neg : end_positive[j],
                    ]
                    for j in range(batch_size)
                ]
            )
        )  # Positive samples representations

        return representation, positive_representation

    def forward(self, z1, z2):
        N = z1.size(0)

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)
        c.div_(N)

        # use scale-loss to multiply the loss by a constant factor
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().mul(self.scale_loss)
        off_diag = self.off_diagonal(c).pow_(2).sum().mul(self.scale_loss)
        loss = on_diag + self.lambd * off_diag
        return loss