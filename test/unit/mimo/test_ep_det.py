#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
import pytest
import numpy as np
import tensorflow as tf
import sionna
from sionna.phy import Block
from sionna.phy.mimo import EPDetector
from sionna.phy.mapping import Mapper, BinarySource, QAMSource
from sionna.phy.utils import compute_ser, compute_ber, sim_ber
from sionna.phy.channel import FlatFadingChannel
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.fec.ldpc.decoding import LDPC5GDecoder

class MIMODetection(Block):
    def __init__(self,
                 num_tx,
                 num_rx_ant,
                 num_bits_per_symbol,
                 output,
                 coded,
                 precision="single"):
        super().__init__(precision=precision)
        self._n = (2000//num_bits_per_symbol)*num_bits_per_symbol
        self._k = 1750
        self._coderate = self._k/self._n
        self._num_tx = num_tx
        self._num_rx_ant = num_rx_ant
        self._num_bits_per_symbol = num_bits_per_symbol
        self._output = output
        self._coded = coded

        self._binary_source = BinarySource(precision=self.precision)

        if self._coded:
            self._encoder = LDPC5GEncoder(self._k, self._n, num_bits_per_symbol=num_bits_per_symbol, precision=self.precision)
            self._decoder = LDPC5GDecoder(self._encoder, hard_out=False, precision=self.precision)
        if self._output=="symbol":
            self._hard_out = True
        else:
            self._hard_out = False

        self._mapper = Mapper("qam", self._num_bits_per_symbol, return_indices=True, precision=precision)
        self._channel = FlatFadingChannel(self._num_tx,
                                          self._num_rx_ant,
                                          add_awgn=True,
                                          return_channel=True,
                                          precision=precision)
        ep_detector = EPDetector(self._output, num_bits_per_symbol, hard_out=self._hard_out, precision=precision)
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
        no =  tf.cast(self._num_tx, self.rdtype)* (10.**(-ebno_db/10.))
        y, h = self._channel(x, no)
        s = tf.cast(no*tf.eye(self._num_rx_ant, dtype=no.dtype), self.cdtype)
        det_out = self._detector(y, h, s)

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

def test_wrong_parameters():
    with pytest.raises(ValueError):
        "Wrong precision"
        EPDetector("bit", 4, precision="half")

    with pytest.raises(AssertionError):
        "Wrong output"
        EPDetector("sym", 4)

    with pytest.raises(AssertionError):
        "Wrong number of iterations"
        EPDetector("sym", 4, l=0)

    with pytest.raises(AssertionError):
        "Beta out of bounds"
        EPDetector("sym", 4, beta=1.1)

@pytest.mark.parametrize("num_bits_per_symbol", [2,4,6,8])
def test_symbol_errors(num_bits_per_symbol):
    """Test that we get no symbol errors on a noise free channel"""
    num_tx = 3
    num_rx_ant = 7
    batch_size = 100
    qam_source = QAMSource(num_bits_per_symbol, return_indices=True)
    channel = FlatFadingChannel(num_tx, num_rx_ant, add_awgn=False, return_channel=True)
    ep = EPDetector("symbol", num_bits_per_symbol, hard_out=True)
    x, x_ind = qam_source([batch_size, num_tx])
    y, h = channel(x)
    s = tf.cast(1e-4*tf.eye(num_rx_ant), tf.complex64)
    x_ind_hat = ep(y, h, s)
    assert compute_ser(x_ind, x_ind_hat) == 0

@pytest.mark.parametrize("num_bits_per_symbol", [2,4,6,8])
def test_no_bit_errors(num_bits_per_symbol):
    "Test that we get no uncoded bit errors on a noise free channel"
    num_tx = 3
    num_rx_ant = 7
    batch_size = 100
    qam_source = QAMSource(num_bits_per_symbol, return_indices=True, return_bits=True)
    channel = FlatFadingChannel(num_tx, num_rx_ant, add_awgn=False, return_channel=True)
    ep = EPDetector("bit", num_bits_per_symbol, hard_out=True)
    x, _, b = qam_source([batch_size, num_tx])
    y, h = channel(x)
    s = tf.cast(1e-4*tf.eye(num_rx_ant), tf.complex64)
    b_hat = ep(y, h, s)
    assert compute_ber(b, b_hat) == 0

@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("output", ["bit", "symbol"])
@pytest.mark.parametrize("coded", [True, False])
def test_all_execution_modes(precision, output, coded):
    "Test that the detector work in all execution modes"
    def evaluate(ep):
        def func():
            return ep(tf.convert_to_tensor(100), tf.convert_to_tensor(20.0))
        _, x_hat = func()
        assert not np.any(np.isnan(x_hat))
        _, x_hat = tf.function(func, jit_compile=False)()
        assert not np.any(np.isnan(x_hat))
        # Avoid numerical errors (NaN, Inf) in low precision (complex64)
        if ep._precision == "double":
            _, x_hat = tf.function(func, jit_compile=True)()
            assert not np.any(np.isnan(x_hat))

    evaluate(MIMODetection(4, 8, 4, output, coded, precision))

@pytest.mark.usefixtures("only_gpu")
def test_ser_against_references():
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
    ser = sim(MIMODetection(4, 8, 2, "symbol", False, precision="single"), 12.0)
    assert ser >= 4e-5
    assert ser <= 5e-5

    # Fig. 8 (b)
    #ser = sim(MIMODetection(6, 8, 2, "symbol", False, precision="single"), 13.0)
    #assert ser >= 1.5e-5
    #assert ser <= 2.5e-4

    # Fig. 8 (c)
    #ser = sim(MIMODetection(8, 8, 2, "symbol", False, precision="single"), 18.0)
    #assert ser >= 3e-5
    #assert ser <= 4e-5

    # Fig. 9 (c)
    #ser = sim(MIMODetection(32, 32, 2, "symbol", False, precision="single"), 13.0)
    #assert ser >= 7e-5
    #assert ser <= 8e-5

    # Fig. 10 (c)
    #ser = sim(MIMODetection(32, 32, 4, "symbol", False, precision="single"), 27.0)
    #assert ser >= 9e-5
    #assert ser <= 1e-4

    # Fig. 11 (c)
    #ser = sim(MIMODetection(8, 8, 6, "symbol", False, precision="double"), 40.0)
    #assert ser >= 3e-4
    #assert ser <= 4e-4