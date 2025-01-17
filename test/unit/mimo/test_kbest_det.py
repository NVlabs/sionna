#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import pytest
import unittest
import numpy as np
import tensorflow as tf
import sionna
from sionna.mimo import KBestDetector, MaximumLikelihoodDetector
from sionna.mapping import Constellation, Mapper
from sionna.utils import BinarySource, QAMSource, PAMSource, compute_ser, compute_ber, ebnodb2no, sim_ber
from sionna.channel import FlatFadingChannel
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.ldpc.decoding import LDPC5GDecoder

class MIMODetectionBER(tf.keras.Model):
    """Simple class to evalute (un-)coded BER of different detectors"""
    def __init__(self,
                 num_tx,
                 num_rx_ant,
                 num_bits_per_symbol,
                 detector,
                 coded=True,
                 dtype=tf.complex64):
        super().__init__()
        self._dtype = dtype
        self._n = 2000
        self._k = 1500
        self._coderate = self._k/self._n
        self._num_tx = num_tx
        self._num_rx_ant = num_rx_ant
        self._num_bits_per_symbol = num_bits_per_symbol
        self._binary_source = BinarySource()
        self._encoder = LDPC5GEncoder(self._k, self._n, dtype=dtype.real_dtype)
        self._mapper = Mapper("qam", self._num_bits_per_symbol, dtype=dtype)
        self._channel = FlatFadingChannel(self._num_tx,
                                          self._num_rx_ant,
                                          add_awgn=True,
                                          return_channel=True,
                                          dtype=dtype)
        if detector=="kbest":
            k = (2**num_bits_per_symbol)**num_tx
            kbest_detector = KBestDetector("bit", num_tx, k,"qam", num_bits_per_symbol, use_real_rep=False, hard_out=not coded, dtype=dtype)
            self._detector = kbest_detector
        elif detector=="ml":
            ml_detector = MaximumLikelihoodDetector("bit", "maxlog", num_tx, "qam", num_bits_per_symbol, hard_out=not coded, dtype=dtype)
            self._detector = ml_detector
        self._decoder = LDPC5GDecoder(self._encoder, hard_out=True)
        self._coded = coded
    
    @tf.function(jit_compile=True)
    def call(self, batch_size, ebno_db):
        b = self._binary_source([batch_size, self._num_tx, self._k])
        c = self._encoder(b)
        shape = tf.shape(c)
        x = self._mapper(c)
        x = tf.reshape(x, [-1, self._num_tx])
        no =  tf.cast(self._num_tx, tf.float32)*tf.pow(10.0, -ebno_db/10.0)
        y, h = self._channel([x, no])
        s = tf.cast(no*tf.eye(self._num_rx_ant), self._dtype)
        llr = self._detector([y, h, s])
        llr = tf.reshape(llr, shape)
        if not self._coded:
            return c, llr
        else:
            b_hat = self._decoder(llr)
            return b, b_hat







class TestKBestDetector(unittest.TestCase):

    def test_wrong_parameters(self):
        with self.assertRaises(AssertionError):
            "Neither constellation nor constellation_type"
            KBestDetector("bit", 4, 16)

        with self.assertRaises(AssertionError):
            "Missing num_bits_per_symbol"
            KBestDetector("bit", 4, 16,
                          constellation_type = "qam")

        with self.assertRaises(AssertionError):
            "Missing constellation_type"
            KBestDetector("bit", 4, 16,
                          num_bits_per_symbol=4)

        with self.assertRaises(AssertionError):
            "Overspecified constellation"
            KBestDetector("bit", 4, 16,
                          num_bits_per_symbol=4,
                          constellation=Constellation("pam", 4))

        with self.assertRaises(AssertionError):
            "Overspecified constellation"
            KBestDetector("bit", 4, 16,
                          constellation_type="qam",
                          constellation=Constellation("pam", 4))

        with self.assertRaises(AssertionError):
            "Overspecified constellation"
            KBestDetector("bit", 4, 16,
                          constellation_type="qam",
                          num_bits_per_symbol = 4,
                          constellation=Constellation("pam", 4))

        with self.assertRaises(AssertionError):
            "Wrong constellation dtype"
            KBestDetector("bit", 4, 16,
                          constellation=Constellation("pam", 4),
                          dtype=tf.complex128)

        with self.assertRaises(AssertionError):
            "Wrong constellation dtype"
            KBestDetector("bit", 4, 16,
                          constellation=Constellation("pam", 4, dtype=tf.complex128))

    def test_init_complex_rep(self):
        num_bits_per_symbol = 4
        num_tx = 4
        k = 16
        constellation_type = "qam"
        
        # Test correct initialization for QAM
        detector = KBestDetector("bit", num_tx, k,
                          constellation_type="qam",
                          num_bits_per_symbol=num_bits_per_symbol)
        self.assertEqual(detector._num_streams, num_tx)
        self.assertEqual(detector._num_symbols, 2**num_bits_per_symbol)
        self.assertEqual(k, detector._k)
        self.assertTrue(np.allclose(np.var(detector._constellation), 1.0))

        # Test correct initialization for PAM
        detector = KBestDetector("bit", num_tx, k,
                          constellation=Constellation("pam", num_bits_per_symbol))
        self.assertEqual(detector._num_streams, num_tx)
        self.assertEqual(detector._num_symbols, 2**num_bits_per_symbol)
        self.assertEqual(k, detector._k)
        self.assertTrue(np.allclose(np.var(detector._constellation), 1.0))

        # Test that k was limited maximum possible length
        num_symbols = 2**num_bits_per_symbol
        k_max = num_symbols**num_tx 
        with self.assertWarns(Warning):
            detector = KBestDetector("bit", num_tx, 2*k_max,
                          constellation_type="qam",
                          num_bits_per_symbol = 4)
            self.assertEqual(detector._k, k_max)

    def test_wrong_constellation_for_real_rep(self):
        """Test that PAM cannot be used with use_real_rep"""
        output = "bit"
        num_tx = 4  
        k = 16
        use_real_rep=True
        with self.assertRaises(AssertionError):
            detector = KBestDetector(output, num_tx, k,
                                     constellation_type="pam",
                                     use_real_rep=use_real_rep)

        constellation = Constellation("pam", 4)
        with self.assertRaises(AssertionError):
            detector = KBestDetector(output, num_tx, k,
                                     constellation=constellation,
                                     use_real_rep=use_real_rep)

    def test_too_few_rx_antennas(self):
        """Throw a warning if more streams than rx antennas"""
        num_tx = 4
        num_rx_ant = 3
        num_bits_per_symbol = 4
        batch_size = 100
        k=64
        qam_source = QAMSource(num_bits_per_symbol, return_indices=True)
        channel = FlatFadingChannel(num_tx, num_rx_ant, add_awgn=False, return_channel=True)
        kbest_complex = KBestDetector("symbol", num_tx, k, "qam", num_bits_per_symbol, use_real_rep=False, hard_out=True)
        kbest_real = KBestDetector("symbol", num_tx, k, "qam", num_bits_per_symbol, use_real_rep=True, hard_out=True)
        x, x_ind = qam_source([batch_size, num_tx])
        y, h = channel(x)
        s = tf.cast(1e-9*tf.eye(num_rx_ant), tf.complex64)
        with self.assertRaises(AssertionError):
            x_ind_hat = kbest_complex([y, h, s])
        with self.assertRaises(AssertionError):
            x_ind_hat = kbest_real([y, h, s])

    def test_init_real_rep(self):
        num_bits_per_symbol = 4
        num_tx = 4
        k = 16
        constellation_type = "qam"

        detector = KBestDetector("bit", num_tx, k,
                          constellation_type="qam",
                          num_bits_per_symbol=num_bits_per_symbol,
                          use_real_rep=True)
        self.assertEqual(detector._num_streams, 2*num_tx)
        self.assertEqual(detector._num_symbols, 2**(num_bits_per_symbol//2))
        self.assertEqual(k, detector._k)
        self.assertTrue(np.allclose(np.var(detector._constellation), 0.5))

        detector = KBestDetector("bit", num_tx, k,
                          constellation=Constellation("qam", num_bits_per_symbol),
                          use_real_rep=True)
        self.assertEqual(detector._num_streams, 2*num_tx)
        self.assertEqual(detector._num_symbols, 2**(num_bits_per_symbol//2))
        self.assertEqual(k, detector._k)
        self.assertTrue(np.allclose(np.var(detector._constellation), 0.5))

    def test_symbol_errors_complex_rep(self):
        """Test that we get no symbol error using the complex-valued representation"""
        num_tx = 3
        num_rx_ant = 7
        num_bits_per_symbols = [2,4,6,8]
        batch_size = 100
        k = 64
        for num_bits_per_symbol in num_bits_per_symbols:
            qam_source = QAMSource(num_bits_per_symbol, return_indices=True)
            channel = FlatFadingChannel(num_tx, num_rx_ant, add_awgn=False, return_channel=True)
            kbest = KBestDetector("symbol", num_tx, k, "qam", num_bits_per_symbol, use_real_rep=False, hard_out=True)
            x, x_ind = qam_source([batch_size, num_tx])
            y, h = channel(x)
            s = tf.cast(1e-9*tf.eye(num_rx_ant), tf.complex64)
            x_ind_hat = kbest([y, h, s])
            self.assertEqual(0, compute_ser(x_ind, x_ind_hat))

    def test_symbol_errors_real_rep(self):
        """Test that we get no symbol error using the real-valued representation"""
        num_tx = 3
        num_rx_ant = 7
        num_bits_per_symbols = [2,4,6,8]
        batch_size = 100
        k = 64
        for num_bits_per_symbol in num_bits_per_symbols:
            qam_source = QAMSource(num_bits_per_symbol, return_indices=True)
            channel = FlatFadingChannel(num_tx, num_rx_ant, add_awgn=False, return_channel=True)
            kbest = KBestDetector("symbol", num_tx, k, "qam", num_bits_per_symbol, use_real_rep=True, hard_out=True)
            x, x_ind = qam_source([batch_size, num_tx])
            y, h = channel(x)
            s = tf.cast(1e-9*tf.eye(num_rx_ant), tf.complex64)
            x_ind_hat = kbest([y, h, s])
            self.assertEqual(0, compute_ser(x_ind, x_ind_hat))

    def test_symbol_errors_pam(self):
        """Test that we get no symbol error using the complex-valued representation and PAM"""
        num_tx = 4
        num_rx_ant = 8
        num_bits_per_symbols = [1,2,3,4]
        batch_size = 100
        k = 16
        for num_bits_per_symbol in num_bits_per_symbols:
            pam_source = PAMSource(num_bits_per_symbol, return_indices=True)
            channel = FlatFadingChannel(num_tx, num_rx_ant, add_awgn=False, return_channel=True)
            kbest = KBestDetector("symbol", num_tx, k, "pam", num_bits_per_symbol, use_real_rep=False, hard_out=True)
            x, x_ind = pam_source([batch_size, num_tx])
            y, h = channel(x)
            s = tf.cast(1e-9*tf.eye(num_rx_ant), tf.complex64)
            x_ind_hat = kbest([y, h, s])
            self.assertEqual(0, compute_ser(x_ind, x_ind_hat))

    def test_bit_errors_complex_rep(self):
        """Test that we get no bit error using the complex-valued representation"""
        num_tx = 3
        num_rx_ant = 7
        num_bits_per_symbols = [2,4,6,8]
        batch_size = 100
        k = 64
        for num_bits_per_symbol in num_bits_per_symbols:
            qam_source = QAMSource(num_bits_per_symbol, return_indices=True, return_bits=True)
            channel = FlatFadingChannel(num_tx, num_rx_ant, add_awgn=False, return_channel=True)
            kbest = KBestDetector("bit", num_tx, k, "qam", num_bits_per_symbol, use_real_rep=False, hard_out=True)
            x, _, b = qam_source([batch_size, num_tx])
            y, h = channel(x)
            s = tf.cast(1e-9*tf.eye(num_rx_ant), tf.complex64)
            b_hat = kbest([y, h, s])
            self.assertEqual(0, compute_ber(b, b_hat))

    def test_bit_errors_real_rep(self):
        """Test that we get no bit error using the real-valued representation"""
        num_tx = 3
        num_rx_ant = 7
        num_bits_per_symbols = [2,4,6,8]
        batch_size = 100
        k = 64
        for num_bits_per_symbol in num_bits_per_symbols:
            qam_source = QAMSource(num_bits_per_symbol, return_indices=True, return_bits=True)
            channel = FlatFadingChannel(num_tx, num_rx_ant, add_awgn=False, return_channel=True)
            kbest = KBestDetector("bit", num_tx, k, "qam", num_bits_per_symbol, use_real_rep=True, hard_out=True)
            x, _, b = qam_source([batch_size, num_tx])
            y, h = channel(x)
            s = tf.cast(1e-9*tf.eye(num_rx_ant), tf.complex64)
            b_hat = kbest([y, h, s])
            self.assertEqual(0, compute_ber(b, b_hat))

    def test_bit_errors_pam(self):
        """Test that we get no bit error using the real-valued representation"""
        num_tx = 4
        num_rx_ant = 7
        num_bits_per_symbols = [1,2,3,4]
        batch_size = 100
        k = 16
        for num_bits_per_symbol in num_bits_per_symbols:
            pam_source = PAMSource(num_bits_per_symbol, return_indices=True, return_bits=True)
            channel = FlatFadingChannel(num_tx, num_rx_ant, add_awgn=False, return_channel=True)
            kbest = KBestDetector("bit", num_tx, k, "pam", num_bits_per_symbol, use_real_rep=False, hard_out=True)
            x, _, b = pam_source([batch_size, num_tx])
            y, h = channel(x)
            s = tf.cast(1e-9*tf.eye(num_rx_ant), tf.complex64)
            b_hat = kbest([y, h, s])
            self.assertEqual(0, compute_ber(b, b_hat))
        return

    def test_llr_against_ml_qam(self):
        num_tx = 3
        num_rx_ant = 8
        batch_size = 100
        def fun(ebno_db, num_bits_per_symbol, k, real_rep):
            qam_source = QAMSource(num_bits_per_symbol, return_indices=True, return_bits=True)
            channel = FlatFadingChannel(num_tx, num_rx_ant, add_awgn=True, return_channel=True)
            kbest = KBestDetector("bit", num_tx, k, "qam", num_bits_per_symbol, use_real_rep=real_rep, hard_out=False)
            kbest._list2llr.llr_clip_val = np.inf
            ml = MaximumLikelihoodDetector("bit", "maxlog", num_tx, "qam", num_bits_per_symbol, hard_out=False)
            no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate=1)
            x, x_ind, b = qam_source([batch_size, num_tx])
            y, h = channel([x, no])
            s = tf.cast(no*tf.eye(num_rx_ant), tf.complex64)
            llr = kbest([y, h, s])
            llr_ml = ml([y, h, s])
            return np.allclose(llr, llr_ml, atol=1e-5)

        for ebno_db in [-20,-10,0,10,20,30,50]:
            for num_bits_per_symbol in [2, 4]:
                for real_rep in [True, False]:
                    k = (2**num_bits_per_symbol)**num_tx
                    self.assertTrue(fun(ebno_db, num_bits_per_symbol, k, real_rep))

    def test_llr_against_ml_pam(self):
        num_tx = 3
        num_rx_ant = 8
        batch_size = 100
        def fun(ebno_db, num_bits_per_symbol, k):
            pam_source = PAMSource(num_bits_per_symbol, return_indices=True, return_bits=True)
            channel = FlatFadingChannel(num_tx, num_rx_ant, add_awgn=True, return_channel=True)
            kbest = KBestDetector("bit", num_tx, k, "pam", num_bits_per_symbol, use_real_rep=False, hard_out=False)
            kbest._list2llr.llr_clip_val = np.inf
            ml = MaximumLikelihoodDetector("bit", "maxlog", num_tx, "pam", num_bits_per_symbol, hard_out=False)
            no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate=1)
            x, x_ind, b = pam_source([batch_size, num_tx])
            y, h = channel([x, no])
            s = tf.cast(no*tf.eye(num_rx_ant), tf.complex64)
            llr = kbest([y, h, s])
            llr_ml = ml([y, h, s])
            return np.allclose(llr, llr_ml, atol=1e-4)

        for ebno_db in [-20,-10,0,10,20,30,50]:
            for num_bits_per_symbol in [1,2,3,4]:
                k = (2**num_bits_per_symbol)**num_tx
                self.assertTrue(fun(ebno_db, num_bits_per_symbol, k))

    @pytest.mark.usefixtures("only_gpu")
    def test_e2e_uncoded_ber_vs_ml(self):
        """Test uncoded BER against ML for some points also in XLA mode"""
        sionna.config.xla_compat=True
        num_tx = 3
        num_rx_ant = 6
        num_bits_per_symbol = 4 
        kbest = MIMODetectionBER(num_tx, num_rx_ant, num_bits_per_symbol, "kbest", coded=False)
        ml = MIMODetectionBER(num_tx, num_rx_ant, num_bits_per_symbol, "ml", coded=False)
        snr_range = np.arange(5,19, 1)
        kbest_ber, kbest_bler = sim_ber(kbest, 
                                        snr_range,
                                        batch_size=64,
                                        max_mc_iter=1000,
                                        num_target_block_errors=1000)
        ml_ber, ml_bler = sim_ber(ml, 
                                  snr_range,
                                  batch_size=64,
                                  max_mc_iter=1000,
                                  num_target_block_errors=1000)
        self.assertTrue(np.allclose(kbest_ber, ml_ber, atol=1e-3))

    @pytest.mark.usefixtures("only_gpu")
    def test_e2e_coded_ber_vs_ml(self):
        """Test coded BER against ML for some points also in XLA mode"""
        sionna.config.xla_compat=True
        num_tx = 3
        num_rx_ant = 6
        num_bits_per_symbol = 4 
        kbest = MIMODetectionBER(num_tx, num_rx_ant, num_bits_per_symbol, "kbest", coded=True)
        ml = MIMODetectionBER(num_tx, num_rx_ant, num_bits_per_symbol, "ml", coded=True)
        snr_range = np.arange(7,9.5, 0.5)
        kbest_ber, kbest_bler = sim_ber(kbest, 
                                        snr_range,
                                        batch_size=16,
                                        max_mc_iter=2000,
                                        num_target_block_errors=2000)
        ml_ber, ml_bler = sim_ber(ml, 
                                  snr_range,
                                  batch_size=16,
                                  max_mc_iter=2000,
                                  num_target_block_errors=2000)
        self.assertTrue(np.allclose(kbest_ber, ml_ber, rtol=0.1))
