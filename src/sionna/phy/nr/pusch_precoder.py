#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""PUSCH Precoding for the 5G NR module of Sionna PHY."""

from typing import List, Optional
import numpy as np
import torch

from sionna.phy import Block


__all__ = ["PUSCHPrecoder"]


class PUSCHPrecoder(Block):
    r"""Precodes a batch of modulated symbols mapped onto a resource grid
    for PUSCH transmissions.

    Each transmitter is assumed to have its own precoding matrix.

    :param precoding_matrices: List of length `num_tx` containing one
        two-dimensional precoding matrix per transmitter. Each matrix must
        have shape [num_antenna_ports, num_layers], and all matrices must have
        the same shape. Internally, the matrices are stacked to shape
        [num_tx, num_antenna_ports, num_layers].
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation.

    :input inputs: [batch_size, num_tx, num_layers, num_symbols_per_slot, num_subcarriers], `torch.complex`.
        Batch of resource grids to be precoded.

    :output x_precoded: [batch_size, num_tx, num_antenna_ports, num_symbols_per_slot, num_subcarriers], `torch.complex`.
        Batch of precoded resource grids.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        import numpy as np
        from sionna.phy.nr import PUSCHPrecoder

        # Create precoding matrices for 2 transmitters, 2 antenna ports, 1 layer
        w1 = np.array([[1], [0]], dtype=complex) / np.sqrt(2)
        w2 = np.array([[1], [1]], dtype=complex) / 2
        precoding_matrices = [w1, w2]

        precoder = PUSCHPrecoder(precoding_matrices)
        x = torch.randn(4, 2, 1, 14, 48, dtype=torch.complex64)
        y = precoder(x)
        print(y.shape)
        # torch.Size([4, 2, 2, 14, 48])
    """

    def __init__(
        self,
        precoding_matrices: List,
        precision: Optional[str] = None,
        device: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(precision=precision, device=device, **kwargs)

        self._num_tx = len(precoding_matrices)

        # Check that all precoding matrices have the same shape
        shape = precoding_matrices[0].shape
        w_list = []
        for w in precoding_matrices:
            if w.shape[0] != shape[0] or w.shape[1] != shape[1]:
                raise ValueError(
                    "All precoding matrices must have the same shape")
            w_list.append(w)

        # w has shape: [num_tx, num_antenna_ports, num_layers]
        # Convert list of numpy arrays to a single numpy array first for performance
        # Register as buffer for CUDAGraph compatibility
        self.register_buffer("_w", torch.tensor(
            np.array(w_list), dtype=self.cdtype, device=self.device))

    def build(self, input_shape: tuple) -> None:
        """Validate input shape."""
        _, num_tx, num_layers, _, _ = input_shape
        if num_tx != len(self._w):
            raise ValueError(
                f"Input has {num_tx} transmitters but precoding configured "
                f"for {len(self._w)}")
        if num_layers != self._w[0].shape[1]:
            raise ValueError(
                f"Precoding matrices for {self._w[0].shape[1]} layers but "
                f"input has {num_layers}")

    def call(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply precoding.

        :param inputs: Resource grids to precode.

        :output x_precoded: Precoded resource grids.
        """
        # inputs: [batch_size, num_tx, num_layers, num_symbols_per_slot, num_subcarriers]

        # Change ordering: [batch_size, num_symbols, num_subcarriers, num_tx, num_layers]
        inputs = inputs.permute(0, 3, 4, 1, 2)

        # Add dimension for matrix multiplication
        inputs = inputs.unsqueeze(-1)

        # Precode: [batch_size, num_symbols, num_subcarriers, num_tx, num_antenna_ports]
        z = torch.matmul(self._w, inputs).squeeze(-1)

        # Re-order: [batch_size, num_tx, num_antenna_ports, num_symbols, num_subcarriers]
        z = z.permute(0, 3, 4, 1, 2)

        return z

