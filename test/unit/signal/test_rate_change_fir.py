#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for FIR interpolation and decimation blocks."""

import unittest

import torch
from sionna.phy.config import config
from sionna.phy.signal import (
    CustomFilter,
    DecimatingFIR,
    Downsampling,
    InterpolatingFIR,
    RootRaisedCosineFilter,
    UpFirDn,
    Upsampling,
)


class TestRateChangeFIR(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device(config.device)

    def _signal(self, complex_input: bool) -> torch.Tensor:
        torch.manual_seed(123)
        x = torch.randn(3, 29, device=self.device)
        if complex_input:
            x = x + 1j * torch.randn(3, 29, device=self.device)
        return x

    def _rrc(self, sps: int) -> torch.Tensor:
        f = RootRaisedCosineFilter(
            8, sps, 0.25, normalize=False, precision="single", device=str(self.device)
        )
        return f.coefficients.detach()

    def test_interpolating_fir_matches_explicit_upsample_and_filter(self) -> None:
        for complex_input in (False, True):
            for sps in (2, 4):
                x = self._signal(complex_input)
                h = self._rrc(sps)
                ref = CustomFilter(sps, h, normalize=False)(
                    Upsampling(sps)(x), padding="full"
                )
                y = InterpolatingFIR(sps, h)(x)
                self.assertEqual(y.shape[-1], ref.shape[-1] - (sps - 1))
                err = torch.max(torch.abs(ref[..., : y.shape[-1]] - y)).item()
                tail = torch.max(torch.abs(ref[..., y.shape[-1] :])).item()
                self.assertLess(err, 1e-5)
                self.assertEqual(tail, 0.0)

    def test_decimating_fir_matches_full_filter_then_downsample(self) -> None:
        for complex_input in (False, True):
            for sps in (2, 4):
                x = self._signal(complex_input)
                h = torch.flip(self._rrc(sps), dims=[0]) / float(sps)
                ref = Downsampling(sps, offset=7, num_symbols=20)(
                    CustomFilter(sps, h, normalize=False)(x, padding="full")
                )
                y = DecimatingFIR(sps, h, offset=7, num_symbols=20)(x)
                self.assertEqual(y.shape, ref.shape)
                err = torch.max(torch.abs(ref - y)).item()
                self.assertLess(err, 1e-6)

    def test_upfirdn_matches_explicit_up_filter_down_path(self) -> None:
        for complex_input in (False, True):
            for sps in (2, 4):
                for offset in (0, 3, 9):
                    x = self._signal(complex_input)
                    h = self._rrc(sps)
                    ref = CustomFilter(sps, h, normalize=False)(
                        Upsampling(sps)(x), padding="full"
                    )
                    ref = Downsampling(sps, offset=offset, num_symbols=20)(ref)
                    y = UpFirDn(h, up=sps, down=sps, offset=offset, num_symbols=20)(x)
                    self.assertEqual(y.shape, ref.shape)
                    err = torch.max(torch.abs(ref - y)).item()
                    self.assertLess(err, 1e-5)

    def test_nonlast_axis_complex_taps_and_trailing_samples(self) -> None:
        x = self._signal(True).T.contiguous().requires_grad_()
        h = torch.tensor([1 + 2j, -0.5j, 0.25 - 1j], device=self.device)
        for up, down, offset, count in (
            (3, 1, 0, 7),
            (3, 2, 1, None),
            (1, 3, 2, None),
        ):
            full = torch.zeros(
                x.shape[0] * up, x.shape[1], dtype=x.dtype, device=self.device
            )
            full[::up] = x
            expected = torch.stack(
                [
                    torch.nn.functional.conv1d(
                        torch.nn.functional.pad(
                            full[:, column].reshape(1, 1, -1),
                            (h.numel() - 1, h.numel() - 1),
                        ),
                        h.flip(0).reshape(1, 1, -1),
                    ).flatten()[offset::down]
                    for column in range(x.shape[1])
                ],
                dim=1,
            )
            if count is not None:
                expected = expected[:count]
            actual = UpFirDn(
                h,
                up=up,
                down=down,
                offset=offset,
                num_symbols=count,
                axis=0,
                device=str(self.device),
            )(x)
            torch.testing.assert_close(actual, expected)
            actual.abs().sum().backward(retain_graph=True)
            self.assertIsNotNone(x.grad)

    def test_empty_phase_and_invalid_arguments(self) -> None:
        x = self._signal(False)
        h = torch.tensor([1.0, 2.0], device=self.device)
        y = DecimatingFIR(3, h, offset=100, device=str(self.device))(x)
        self.assertEqual(y.shape, (3, 0))
        self.assertEqual(
            UpFirDn(h, up=2, down=3, num_symbols=0, device=str(self.device))(x).shape,
            (3, 0),
        )
        with self.assertRaisesRegex(ValueError, "coefficients"):
            InterpolatingFIR(2, torch.empty(0))
        with self.assertRaisesRegex(ValueError, "num_symbols"):
            DecimatingFIR(2, h, num_symbols=-1)


if __name__ == "__main__":
    unittest.main()
