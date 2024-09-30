#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import pytest
import unittest
import numpy as np
import tensorflow as tf
import sionna
from sionna.mimo import EPDetector
from sionna.mapping import Constellation, Mapper
from sionna.utils import BinarySource, QAMSource, compute_ser, compute_ber, ebnodb2no, sim_ber
from sionna.channel import FlatFadingChannel
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder

class MIMODetection(tf.keras.Model):
    def __init__(self,
                 num_tx,
                 num_rx_ant,
                 num_bits_per_symbol,
                 output,
                 coded,
                 dtype=tf.complex64):
        super().__init__()
        self._dtype = dtype
        self._n = (2000//num_bits_per_symbol)*num_bits_per_symbol
        self._k = 1750
        self._coderate = self._k/self._n
        self._num_tx = num_tx
        self._num_rx_ant = num_rx_ant
        self._num_bits_per_symbol = num_bits_per_symbol
        self._output = output
        self._coded = coded

        self._binary_source = BinarySource()

        if self._coded:
            self._encoder = LDPC5GEncoder(self._k, self._n, num_bits_per_symbol=num_bits_per_symbol, dtype=dtype.real_dtype)
            self._decoder = LDPC5GDecoder(self._encoder, hard_out=False)
        if self._output=="symbol":
            self._hard_out = True
        else:
            self._hard_out = False

        self._mapper = Mapper("qam", self._num_bits_per_symbol, return_indices=True, dtype=dtype)
        self._channel = FlatFadingChannel(self._num_tx,
                                          self._num_rx_ant,
                                          add_awgn=True,
                                          return_channel=True,
                                          dtype=dtype)
        ep_detector = EPDetector(self._output, num_bits_per_symbol, hard_out=self._hard_out, dtype=dtype)
        self._detector = ep_detector

    def call(self, batch_size, ebno_db):

        if self._coded:
            b = self._binary_source([batch_size, self._num_tx, self._k])
            c = self._encoder(b)
        else:
            c = self._binary_source([batch_size, self._num_tx, self._n])

        shape = tf.shape(c)
        x, x_ind = self._mapper(c)
        x = tf.reshape(x, [-1, self._num_tx])
        no =  tf.cast(self._num_tx, tf.float32)*tf.pow(10.0, -ebno_db/10.0)
        y, h = self._channel([x, no])
        s = tf.cast(no*tf.eye(self._num_rx_ant), self._dtype)
        det_out = self._detector([y, h, s])

        if self._output=="bit":
            llr = tf.reshape(det_out, shape)
            if self._coded:
                b_hat = self._decoder(llr)
                return b, b_hat
            else:
                return c, llr
        elif self._output=="symbol":
            x_hat = tf.reshape(det_out, tf.shape(x_ind))
            return x_ind, x_hat

class TestEPDetector(unittest.TestCase):
    def test_wrong_parameters(self):
        with self.assertRaises(AssertionError):
            "Neither constellation nor constellation_type"
            EPDetector("bit", 4, dtype=tf.float32)

        with self.assertRaises(AssertionError):
            "Wrong output"
            EPDetector("sym", 4)

        with self.assertRaises(AssertionError):
            "Wrong number of iterations"
            EPDetector("sym", 4, l=0)

        with self.assertRaises(AssertionError):
            "Beta out of bounds"
            EPDetector("sym", 4, beta=1.1)

    def test_symbol_errors(self):
        """Test that we get no symbol errors on a noise free channel"""
        sionna.config.xla_compat = False
        num_tx = 3
        num_rx_ant = 7
        num_bits_per_symbols = [2,4,6,8]
        batch_size = 100
        for num_bits_per_symbol in num_bits_per_symbols:
            qam_source = QAMSource(num_bits_per_symbol, return_indices=True)
            channel = FlatFadingChannel(num_tx, num_rx_ant, add_awgn=False, return_channel=True)
            kbest = EPDetector("symbol", num_bits_per_symbol, hard_out=True)
            x, x_ind = qam_source([batch_size, num_tx])
            y, h = channel(x)
            s = tf.cast(1e-4*tf.eye(num_rx_ant), tf.complex64)
            x_ind_hat = kbest([y, h, s])
            self.assertEqual(0, compute_ser(x_ind, x_ind_hat))

    def test_no_bit_errors(self):
        "Test that we get no uncoded bit errors on a noise free channel"
        sionna.config.xla_compat = False
        num_tx = 3
        num_rx_ant = 7
        num_bits_per_symbols = [2,4,6,8]
        batch_size = 100
        for num_bits_per_symbol in num_bits_per_symbols:
            qam_source = QAMSource(num_bits_per_symbol, return_indices=True, return_bits=True)
            channel = FlatFadingChannel(num_tx, num_rx_ant, add_awgn=False, return_channel=True)
            kbest = EPDetector("bit", num_bits_per_symbol, hard_out=True)
            x, _, b = qam_source([batch_size, num_tx])
            y, h = channel(x)
            s = tf.cast(1e-4*tf.eye(num_rx_ant), tf.complex64)
            b_hat = kbest([y, h, s])
            self.assertEqual(0, compute_ber(b, b_hat))

    def test_all_execution_modes(self):
        "Test that the detector work in all execution modes"
        def evaluate(ep):
            def func():
                return ep(100, 20.0)
            _, x_hat = func()
            self.assertFalse(np.any(np.isnan(x_hat)))
            _, x_hat = tf.function(func, jit_compile=False)()
            self.assertFalse(np.any(np.isnan(x_hat)))
            # Avoid numerical errors (NaN, Inf) in low precision (complex64)
            if ep._dtype == tf.complex128:
                _, x_hat = tf.function(func, jit_compile=True)()
                self.assertFalse(np.any(np.isnan(x_hat)))

        for dtype in [tf.complex64, tf.complex128]:
            evaluate(MIMODetection(1, 1, 4, "bit", False, dtype))
            evaluate(MIMODetection(3, 3, 2, "bit", True, dtype))
            evaluate(MIMODetection(3, 6, 4, "symbol", False, dtype))
            evaluate(MIMODetection(3, 5, 4, "bit", False, dtype))
            evaluate(MIMODetection(2, 6, 4, "bit", True, dtype))
            evaluate(MIMODetection(4, 8, 4, "symbol", False, dtype))

    @pytest.mark.usefixtures("only_gpu")
    def test_ser_against_references(self):
        sionna.config.xla_compat = False
        def sim(ep, snr_db):
            ser, _ = sim_ber(ep,
                        [snr_db],
                        batch_size=64,
                        max_mc_iter=1000,
                        num_target_block_errors=1000,
                        soft_estimates=False,
                        graph_mode="graph",
                        verbose=False)
            return ser[0]
        # Values taken from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9832663
        # Fig. 8 (a)
        ser = sim(MIMODetection(4, 8, 2, "symbol", False, tf.complex64), 12.0)
        self.assertGreaterEqual(ser, 4e-5)
        self.assertLessEqual(ser, 5e-5)

        # Fig. 8 (b)
        ser = sim(MIMODetection(6, 8, 2, "symbol", False, tf.complex64), 13.0)
        self.assertGreaterEqual(ser, 1.5e-4)
        self.assertLessEqual(ser, 2.5e-4)

        # Fig. 8 (c)
        ser = sim(MIMODetection(8, 8, 2, "symbol", False, tf.complex64), 18.0)
        self.assertGreaterEqual(ser, 3e-5)
        self.assertLessEqual(ser, 4e-5)

        # Fig. 9 (c)
        ser = sim(MIMODetection(32, 32, 2, "symbol", False, tf.complex64), 13.0)
        self.assertGreaterEqual(ser, 7e-5)
        self.assertLessEqual(ser, 8e-5)

        # Fig. 10 (c)
        ser = sim(MIMODetection(32, 32, 4, "symbol", False, tf.complex64), 27.0)
        self.assertGreaterEqual(ser, 9e-5)
        self.assertLessEqual(ser, 1e-4)

        # Fig. 11 (c)
        ser = sim(MIMODetection(8, 8, 6, "symbol", False, tf.complex128), 40.0)
        self.assertGreaterEqual(ser, 3e-4)
        self.assertLessEqual(ser, 4e-4)