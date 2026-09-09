#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for sionna.phy.mimo.stream_management module."""

import pytest
import numpy as np

from sionna.phy.mimo.stream_management import StreamManagement


class TestStreamManagement:
    """Tests for StreamManagement class."""

    def test_basic_single_user(self):
        """Test basic single-user MIMO configuration."""
        # 1 RX, 1 TX, 2 streams
        rx_tx_association = np.array([[1]])
        sm = StreamManagement(rx_tx_association, num_streams_per_tx=2)

        assert sm.num_rx == 1
        assert sm.num_tx == 1
        assert sm.num_streams_per_tx == 2
        assert sm.num_streams_per_rx == 2
        assert sm.num_tx_per_rx == 1
        assert sm.num_rx_per_tx == 1

    def test_two_user_symmetric(self):
        """Test symmetric 2-user configuration."""
        # 2 RX, 2 TX, each TX sends to one RX
        rx_tx_association = np.array([[1, 0], [0, 1]])
        sm = StreamManagement(rx_tx_association, num_streams_per_tx=2)

        assert sm.num_rx == 2
        assert sm.num_tx == 2
        assert sm.num_streams_per_tx == 2
        assert sm.num_streams_per_rx == 2
        assert sm.num_tx_per_rx == 1
        assert sm.num_rx_per_tx == 1

    def test_multi_user_mu_mimo(self):
        """Test MU-MIMO configuration with 1 TX, 2 RX."""
        # 2 RX, 1 TX, TX sends to both RX
        rx_tx_association = np.array([[1], [1]])
        sm = StreamManagement(rx_tx_association, num_streams_per_tx=4)

        assert sm.num_rx == 2
        assert sm.num_tx == 1
        assert sm.num_streams_per_tx == 4
        assert sm.num_streams_per_rx == 2  # 1*4/2 = 2
        assert sm.num_tx_per_rx == 1
        assert sm.num_rx_per_tx == 2

    def test_precoding_ind(self):
        """Test precoding indices computation."""
        rx_tx_association = np.array([[1, 0], [0, 1]])
        sm = StreamManagement(rx_tx_association, num_streams_per_tx=1)

        # TX 0 sends to RX 0, TX 1 sends to RX 1
        expected = np.array([[0], [1]], dtype=np.int32)
        np.testing.assert_array_equal(sm.precoding_ind, expected)

    def test_stream_association(self):
        """Test stream association matrix structure."""
        rx_tx_association = np.array([[1, 0], [0, 1]])
        sm = StreamManagement(rx_tx_association, num_streams_per_tx=2)

        # Check shape
        assert sm.stream_association.shape == (2, 2, 2)

        # RX 0 gets streams from TX 0 only
        np.testing.assert_array_equal(sm.stream_association[0, 0, :], [1, 1])
        np.testing.assert_array_equal(sm.stream_association[0, 1, :], [0, 0])

        # RX 1 gets streams from TX 1 only
        np.testing.assert_array_equal(sm.stream_association[1, 0, :], [0, 0])
        np.testing.assert_array_equal(sm.stream_association[1, 1, :], [1, 1])

    def test_tx_stream_ids(self):
        """Test TX stream ID mapping."""
        rx_tx_association = np.array([[1, 0], [0, 1]])
        sm = StreamManagement(rx_tx_association, num_streams_per_tx=2)

        # TX 0 gets streams 0,1; TX 1 gets streams 2,3
        expected = np.array([[0, 1], [2, 3]])
        np.testing.assert_array_equal(sm.tx_stream_ids, expected)

    def test_rx_stream_ids(self):
        """Test RX stream ID mapping."""
        rx_tx_association = np.array([[1, 0], [0, 1]])
        sm = StreamManagement(rx_tx_association, num_streams_per_tx=2)

        # RX 0 gets streams 0,1 (from TX 0)
        # RX 1 gets streams 2,3 (from TX 1)
        expected = np.array([[0, 1], [2, 3]], dtype=np.int32)
        np.testing.assert_array_equal(sm.rx_stream_ids, expected)

    def test_stream_ind(self):
        """Test stream reordering indices."""
        rx_tx_association = np.array([[1, 0], [0, 1]])
        sm = StreamManagement(rx_tx_association, num_streams_per_tx=2)

        # Since RX gets streams in the same order they're transmitted,
        # stream_ind should be identity permutation
        expected = np.array([0, 1, 2, 3])
        np.testing.assert_array_equal(sm.stream_ind, expected)

    def test_detection_indices(self):
        """Test detection desired/undesired indices."""
        rx_tx_association = np.array([[1, 0], [0, 1]])
        sm = StreamManagement(rx_tx_association, num_streams_per_tx=2)

        # Total elements in stream_association: 2*2*2 = 8
        # Desired indices where stream_association == 1
        # Undesired indices where stream_association == 0
        assert len(sm.detection_desired_ind) + len(sm.detection_undesired_ind) == 8

    def test_invalid_binary_association(self):
        """Test that non-binary association raises error."""
        rx_tx_association = np.array([[1, 2], [0, 1]])
        with pytest.raises(ValueError):
            StreamManagement(rx_tx_association, num_streams_per_tx=1)

    def test_asymmetric_rx_association_raises(self):
        """Test that asymmetric RX association raises error."""
        # RX 0 has 2 TXs, RX 1 has 1 TX - should fail
        rx_tx_association = np.array([[1, 1], [1, 0]])
        with pytest.raises(ValueError):
            StreamManagement(rx_tx_association, num_streams_per_tx=1)

    def test_asymmetric_tx_association_raises(self):
        """Test that asymmetric TX association raises error."""
        # TX 0 has 2 RXs, TX 1 has 1 RX - should fail
        rx_tx_association = np.array([[1, 0], [1, 1]])
        with pytest.raises(ValueError):
            StreamManagement(rx_tx_association, num_streams_per_tx=1)

    def test_broadcast_configuration(self):
        """Test broadcast configuration (1 TX to all RX)."""
        # 3 RX, 1 TX broadcasting to all
        rx_tx_association = np.array([[1], [1], [1]])
        sm = StreamManagement(rx_tx_association, num_streams_per_tx=3)

        assert sm.num_rx == 3
        assert sm.num_tx == 1
        assert sm.num_streams_per_tx == 3
        assert sm.num_streams_per_rx == 1  # 1*3/3 = 1
        assert sm.num_rx_per_tx == 3

    def test_many_to_many_stream_mapping(self):
        """Streams are uniformly and uniquely assigned on active links."""
        sm = StreamManagement(
            np.ones((2, 2), dtype=np.int32), num_streams_per_tx=4
        )

        expected_association = np.array(
            [
                [[1, 1, 0, 0], [1, 1, 0, 0]],
                [[0, 0, 1, 1], [0, 0, 1, 1]],
            ],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(
            sm.stream_association, expected_association
        )
        np.testing.assert_array_equal(
            sm.rx_stream_ids,
            np.array([[0, 1, 4, 5], [2, 3, 6, 7]], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            sm.detection_desired_ind,
            np.array([0, 1, 4, 5, 10, 11, 14, 15]),
        )
        np.testing.assert_array_equal(
            sm.detection_undesired_ind,
            np.array([2, 3, 6, 7, 8, 9, 12, 13]),
        )

        # Every TX stream has exactly one RX owner and reordering recovers
        # transmitter-major stream order.
        np.testing.assert_array_equal(
            sm.stream_association.sum(axis=0),
            np.ones((2, 4), dtype=np.int32),
        )
        np.testing.assert_array_equal(
            sm.rx_stream_ids.ravel()[sm.stream_ind],
            sm.tx_stream_ids.ravel(),
        )

    def test_many_to_many_requires_integral_stream_split(self):
        """Streams must divide evenly over a transmitter's receivers."""
        with pytest.raises(ValueError, match="must be divisible"):
            StreamManagement(
                np.ones((2, 2), dtype=np.int32), num_streams_per_tx=3
            )

    def test_interfering_streams(self):
        """Test number of interfering streams calculation."""
        # Diagonal configuration: 4 TXs, 4 RXs
        rx_tx_association = np.eye(4, dtype=np.int32)
        sm = StreamManagement(rx_tx_association, num_streams_per_tx=2)

        # Total streams: 4 TX * 2 streams = 8
        # Streams per RX: 8 / 4 = 2
        # Interfering streams: 8 - 2 = 6
        assert sm.num_interfering_streams_per_rx == 6

    def test_property_setter(self):
        """Test that rx_tx_association can be set after construction."""
        rx_tx_association = np.array([[1]])
        sm = StreamManagement(rx_tx_association, num_streams_per_tx=1)
        assert sm.num_rx == 1
        assert sm.num_tx == 1

        # Update association
        new_association = np.array([[1, 0], [0, 1]])
        sm.rx_tx_association = new_association
        assert sm.num_rx == 2
        assert sm.num_tx == 2

    def test_larger_configuration(self):
        """Test larger MIMO configuration."""
        # 4 RX, 4 TX, diagonal association
        rx_tx_association = np.eye(4, dtype=np.int32)
        sm = StreamManagement(rx_tx_association, num_streams_per_tx=2)

        assert sm.num_rx == 4
        assert sm.num_tx == 4
        assert sm.num_streams_per_tx == 2
        assert sm.num_streams_per_rx == 2
        assert sm.num_tx_per_rx == 1
        assert sm.num_rx_per_tx == 1
        assert sm.num_interfering_streams_per_rx == 6  # 4*2 - 2 = 6

