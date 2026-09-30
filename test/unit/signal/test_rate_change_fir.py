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

    @staticmethod
    def _complete(block, x: torch.Tensor, axis: int = -1) -> torch.Tensor:
        return torch.cat((block(x), block.flush()), dim=axis)

    def test_interpolating_fir_matches_explicit_upsample_and_filter(self) -> None:
        for complex_input in (False, True):
            for sps in (2, 4):
                x = self._signal(complex_input)
                h = self._rrc(sps)
                ref = CustomFilter(sps, h, normalize=False)(
                    Upsampling(sps)(x), padding="full"
                )
                y = self._complete(InterpolatingFIR(sps, h), x)
                self.assertEqual(y.shape, ref.shape)
                self.assertLess(torch.max(torch.abs(ref - y)).item(), 1e-5)

    def test_decimating_fir_matches_full_filter_then_downsample(self) -> None:
        for complex_input in (False, True):
            for sps in (2, 4):
                x = self._signal(complex_input)
                h = torch.flip(self._rrc(sps), dims=[0]) / float(sps)
                ref = Downsampling(sps, offset=7, num_symbols=20)(
                    CustomFilter(sps, h, normalize=False)(x, padding="full")
                )
                y = self._complete(DecimatingFIR(sps, h, offset=7, num_symbols=20), x)
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
                    y = self._complete(
                        UpFirDn(h, up=sps, down=sps, offset=offset, num_symbols=20), x
                    )
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
            actual = self._complete(
                UpFirDn(
                    h,
                    up=up,
                    down=down,
                    offset=offset,
                    num_symbols=count,
                    axis=0,
                    device=str(self.device),
                ),
                x,
                axis=0,
            )
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

    def test_decimation_phase_across_single_sample_chunks(self) -> None:
        x = self._signal(False)
        h = torch.tensor([0.5, -1.0, 0.25, 0.75, 0.125], device=self.device)
        filtered = CustomFilter(4, h, normalize=False)(x, padding="full")
        for offset in (0, 1, 4, 19, 31, 100):
            for limit in (None, 0, 3):
                with self.subTest(offset=offset, limit=limit):
                    expected = Downsampling(4, offset=offset, num_symbols=limit)(
                        filtered
                    )
                    block = DecimatingFIR(4, h, offset=offset, num_symbols=limit)
                    parts = [block(chunk) for chunk in x.split(1, dim=-1)]
                    parts.append(block.flush())
                    torch.testing.assert_close(torch.cat(parts, dim=-1), expected)

    def test_chunked_processing_matches_whole_vector_and_flush_resets(self) -> None:
        x = self._signal(True).T.contiguous()
        h = torch.tensor(
            [0.25 + 0.5j, 1 - 0.25j, -0.5, 0.125j, 0.75], device=self.device
        )
        cases = (
            lambda: InterpolatingFIR(3, h, axis=0, device=str(self.device)),
            lambda: InterpolatingFIR(4, h[:1], axis=0, device=str(self.device)),
            lambda: DecimatingFIR(3, h, offset=2, axis=0, device=str(self.device)),
            lambda: DecimatingFIR(
                4, h[:2], offset=5, num_symbols=4, axis=0, device=str(self.device)
            ),
            lambda: UpFirDn(h, up=3, down=2, offset=1, axis=0, device=str(self.device)),
            lambda: UpFirDn(
                h[:2],
                up=4,
                down=3,
                offset=2,
                num_symbols=6,
                axis=0,
                device=str(self.device),
            ),
        )
        chunks = torch.split(x, (1, 4, 2, 7, 15), dim=0)
        for make_block in cases:
            whole = make_block()
            expected_head = whole(x)
            expected_tail = whole.flush()
            block = make_block()
            parts = [block(chunk) for chunk in chunks]
            torch.testing.assert_close(torch.cat(parts, dim=0), expected_head)
            actual_tail = block.flush()
            torch.testing.assert_close(actual_tail, expected_tail)
            expected = torch.cat((expected_head, expected_tail), dim=0)
            # flush() resets both FIR history and the decimation phase.
            torch.testing.assert_close(self._complete(block, x, axis=0), expected)


if __name__ == "__main__":
    unittest.main()
