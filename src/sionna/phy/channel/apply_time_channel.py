#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Block for applying channel responses to channel inputs in the time domain"""

from typing import Optional, Union

import torch

from sionna.phy import Block
from sionna.phy.utils import insert_dims
from sionna.phy.channel.awgn import AWGN
from sionna.phy.channel.utils import _toeplitz

__all__ = ["ApplyTimeChannel"]


class ApplyTimeChannel(Block):
    # pylint: disable=line-too-long
    r"""
    Apply time domain channel responses ``h_time`` to channel inputs ``x``,
    by filtering the channel inputs with time-variant channel responses.

    For each batch example, ``num_time_samples`` + ``l_tot`` - 1 time steps of a
    channel realization are required to filter the channel inputs.

    The channel output consists of ``num_time_samples`` + ``l_tot`` - 1
    time samples, as it is the result of filtering the channel input of length
    ``num_time_samples`` with the time-variant channel filter of length
    ``l_tot``. In the case of a single-input single-output link and given a sequence of channel
    inputs :math:`x_0,\cdots,x_{N_B-1}`, where :math:`N_B` is ``num_time_samples``, this
    layer outputs

    .. math::
        y_b = \sum_{\ell = 0}^{L_{\text{tot}}-1} x_{b-\ell} \bar{h}_{b,\ell} + w_b

    where :math:`L_{\text{tot}}` corresponds ``l_tot``, :math:`w_b` to the additive noise, and
    :math:`\bar{h}_{b,\ell}` to the :math:`\ell^{th}` tap of the :math:`b^{th}` channel sample.
    This layer outputs :math:`y_b` for :math:`b` ranging from 0 to
    :math:`N_B + L_{\text{tot}} - 2`, and :math:`x_{b}` is set to 0 for :math:`b \geq N_B`.

    For multiple-input multiple-output (MIMO) links, the channel output is computed for each antenna
    of each receiver and by summing over all the antennas of all transmitters.

    :param num_time_samples: Number of time samples forming the channel input (:math:`N_B`)
    :param l_tot: Length of the channel filter (:math:`L_{\text{tot}} = L_{\text{max}} - L_{\text{min}} + 1`)
    :param precision: `None` (default) | `"single"` | `"double"`.
        Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input x: [batch size, num_tx, num_tx_ant, num_time_samples], `torch.complex`.
        Channel inputs.

    :input h_time: [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_samples + l_tot - 1, l_tot], `torch.complex`.
        Channel responses.
        For each batch example, ``num_time_samples`` + ``l_tot`` - 1 time steps of a
        channel realization are required to filter the channel inputs.

    :input no: `None` (default) | `torch.Tensor`, `torch.float`.
        Scalar or tensor whose shape can be broadcast to the shape of the channel outputs:
        [batch size, num_rx, num_rx_ant, num_time_samples + l_tot - 1].
        The (optional) noise power ``no`` is per complex dimension. If ``no`` is a
        scalar, noise of the same variance will be added to the outputs.
        If ``no`` is a tensor, it must have a shape that can be broadcast to
        the shape of the channel outputs. This allows, e.g., adding noise of
        different variance to each example in a batch. If ``no`` has a lower
        rank than the channel outputs, then ``no`` will be broadcast to the
        shape of the channel outputs by adding dummy dimensions after the
        last axis.

    :output y: [batch size, num_rx, num_rx_ant, num_time_samples + l_tot - 1], `torch.complex`.
        Channel outputs.
        The channel output consists of ``num_time_samples`` + ``l_tot`` - 1
        time samples, as it is the result of filtering the channel input of length
        ``num_time_samples`` with the time-variant channel filter of length
        ``l_tot``.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.channel import ApplyTimeChannel

        num_time_samples = 100
        l_tot = 27
        batch_size = 32
        num_tx, num_tx_ant = 1, 4
        num_rx, num_rx_ant = 1, 2

        apply_channel = ApplyTimeChannel(num_time_samples=num_time_samples, l_tot=l_tot)

        x = torch.randn(batch_size, num_tx, num_tx_ant, num_time_samples, dtype=torch.complex64)
        h_time = torch.randn(batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant,
                             num_time_samples + l_tot - 1, l_tot, dtype=torch.complex64)
        y = apply_channel(x, h_time)
        print(y.shape)
        # torch.Size([32, 1, 2, 126])
    """

    def __init__(
        self,
        num_time_samples: int,
        l_tot: int,
        precision: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(precision=precision, device=device, **kwargs)
        self._awgn = AWGN(precision=self.precision, device=self.device)
        self._num_time_samples = num_time_samples
        self._l_tot = l_tot

        # The channel transfer function is implemented by first gathering from
        # the vector of transmitted baseband symbols
        # x = [x_0,...,x_{num_time_samples-1}]^T the symbols that are then
        # multiplied by the channel tap coefficients.
        # We build here the matrix of indices G, with size
        # `num_time_samples + l_tot - 1` x `l_tot` that is used to perform this
        # gathering.
        # For example, if there are 4 channel taps
        # h = [h_0, h_1, h_2, h_3]^T
        # and `num_time_samples` = 10 time steps then G would be
        #       [[0, 10, 10, 10]
        #        [1,  0, 10, 10]
        #        [2,  1,  0, 10]
        #        [3,  2,  1,  0]
        #        [4,  3,  2,  1]
        #        [5,  4,  3,  2]
        #        [6,  5,  4,  3]
        #        [7,  6,  5,  4]
        #        [8,  7,  6,  5]
        #        [9,  8,  7,  6]
        #        [10, 9,  8,  7]
        #        [10,10,  9,  8]
        #        [10,10, 10,  9]
        # Note that G is a Toeplitz matrix.
        # In this example, the index `num_time_samples`=10 corresponds to the
        # zero symbol. The vector of transmitted symbols is padded with one
        # zero at the end.
        first_column = torch.cat(
            [
                torch.arange(0, num_time_samples, dtype=torch.int64,
                             device=self.device),
                torch.full([l_tot - 1], num_time_samples, dtype=torch.int64,
                           device=self.device),
            ]
        )
        first_row = torch.cat(
            [
                torch.zeros(1, dtype=torch.int64, device=self.device),
                torch.full([l_tot - 1], num_time_samples, dtype=torch.int64,
                           device=self.device),
            ]
        )
        # Register as buffer for CUDA graph compatibility
        self.register_buffer("_g", _toeplitz(first_column, first_row))

    @property
    def num_time_samples(self) -> int:
        """Number of time samples"""
        return self._num_time_samples

    @property
    def l_tot(self) -> int:
        """Length of the channel filter"""
        return self._l_tot

    def call(
        self,
        x: torch.Tensor,
        h_time: torch.Tensor,
        no: Optional[Union[float, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Apply time channel to input signal.

        :param x: Channel inputs
        :param h_time: Channel responses
        :param no: Noise power per complex dimension

        :output y: Channel outputs
        """
        # Preparing the channel input for broadcasting and matrix multiplication
        # Pad with one zero at the end (for the zero symbol index)
        x = torch.nn.functional.pad(x, (0, 1))

        # Add dimensions for num_rx and num_rx_ant
        x = insert_dims(x, 2, axis=1)

        # Gather using advanced indexing
        x = x[..., self._g]

        # Apply the channel response
        y = (h_time * x).sum(dim=-1)

        # Sum over TX antennas and TX
        y = y.sum(dim=4).sum(dim=3)

        # Add AWGN if requested
        if no is not None:
            y = self._awgn(y, no)

        return y
