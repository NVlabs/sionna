#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import pytest
import unittest
import numpy as np
import tensorflow as tf
from sionna import config
from sionna.mimo import LinearDetector, MMSEPICDetector
from sionna.channel import FlatFadingChannel, exp_corr_mat, PerColumnModel
from sionna.utils import BinarySource, sim_ber, ebnodb2no
from sionna.mapping import Mapper

@pytest.mark.usefixtures("only_gpu")
class TestMMSEPICDetector(unittest.TestCase):

    # Number of bits per symbol for modulation
    NUM_BITS_PER_SYMBOL = 4

    # Channel correlation exponent
    CHANNEL_CORR_A = 0.8

    # Max error :
    MAX_ERR = 5e-2


    def run_e2e(self, det, batch_dims, num_rx_ant, num_tx_ant, ebno_dbs, exec_mode, dtype):

        num_bits_per_symbol = TestMMSEPICDetector.NUM_BITS_PER_SYMBOL
        batch_dims = tf.cast(batch_dims, tf.int32)

        #
        # Transmitter
        #
        binary_source = BinarySource(dtype=dtype.real_dtype)
        mapper = Mapper("qam", num_bits_per_symbol, dtype=dtype)

        #
        # Channel
        #
        spatial_corr_mat = exp_corr_mat(TestMMSEPICDetector.CHANNEL_CORR_A,
                                    num_rx_ant, dtype)
        spatial_corr = PerColumnModel(spatial_corr_mat)
        channel = FlatFadingChannel(num_tx_ant, num_rx_ant,
                                    spatial_corr=spatial_corr,
                                    return_channel=True,
                                    dtype=dtype)

        #
        # Detector
        #
        if det == 'mmse-pic':
            # MMSE-PIC
            detector = MMSEPICDetector(demapping_method="maxlog",
                                        num_iter=1,
                                        output="bit",
                                        constellation_type="qam",
                                        num_bits_per_symbol=num_bits_per_symbol,
                                        dtype=dtype)
        elif det == 'lmmse':
            # LMMSE
            detector = LinearDetector(equalizer="lmmse",
                                        output="bit",
                                        demapping_method="maxlog",
                                        constellation_type="qam",
                                        num_bits_per_symbol=num_bits_per_symbol,
                                        dtype=dtype)

        # Bits shape
        bits_shape = tf.concat([batch_dims, [num_tx_ant, num_bits_per_symbol]], axis=0)
        # Null prior
        prior = tf.zeros(bits_shape, dtype.real_dtype)
        # Noise covariance
        s = tf.eye(num_rx_ant, dtype=dtype)

        def _run(batch_size, ebno_db):
            # `batch_size` is ignored

            no = ebnodb2no(ebno_db, num_bits_per_symbol, 1.0)

            #
            # Transmitter
            #

            bits = binary_source(bits_shape)
            x = mapper(bits)
            x = tf.squeeze(x, axis=-1)

            #
            # Channel
            #
            y,h = channel((x, no))

            #
            # Detector
            #
            s_ = tf.cast(no, dtype)*s
            if det == 'mmse-pic':
                llrs = detector((y, h, prior, s_))
            elif det == 'lmmse':
                llrs = detector((y, h, s_))

            return bits, llrs

        # Compile according to the specified execution mode
        if exec_mode == 'eager':
            _run_c = _run
        elif exec_mode == 'graph':
            _run_c = tf.function(_run)
        elif exec_mode == 'xla':
            _run_c = tf.function(_run, jit_compile=True)

        # Run over the range of N0s
        ber,_ = sim_ber(_run_c, ebno_dbs, 1,
                        max_mc_iter=100,
                        num_target_bit_errors=1000,
                        soft_estimates=True,
                        early_stop=False,
                        dtype=dtype)

        return ber

    def run_test(self, batch_dims, num_rx_ant, num_tx_ant, ebno_dbs):

        #
        # Test eager - simple precision
        #
        config.seed = config.seed
        ber_lmmse = self.run_e2e('lmmse',
                                    batch_dims,
                                    num_rx_ant,
                                    num_tx_ant,
                                    ebno_dbs,
                                    'eager',
                                    tf.complex64)
        config.seed = config.seed
        ber_mmse_pic = self.run_e2e('mmse-pic',
                                    batch_dims,
                                    num_rx_ant,
                                    num_tx_ant,
                                    ebno_dbs,
                                    'eager',
                                    tf.complex64)
        max_err = np.max(np.abs(ber_lmmse-ber_mmse_pic)/np.abs(ber_lmmse))
        self.assertLess(max_err, TestMMSEPICDetector.MAX_ERR)

        #
        # Test graph - simple precision
        #
        config.seed = config.seed
        ber_lmmse = self.run_e2e('lmmse',
                                batch_dims,
                                num_rx_ant,
                                num_tx_ant,
                                ebno_dbs,
                                'graph',
                                tf.complex64)
        config.seed = config.seed
        ber_mmse_pic = self.run_e2e('mmse-pic',
                                    batch_dims,
                                    num_rx_ant,
                                    num_tx_ant,
                                    ebno_dbs,
                                    'graph',
                                    tf.complex64)
        max_err = np.max(np.abs(ber_lmmse-ber_mmse_pic)/np.abs(ber_lmmse))
        self.assertLess(max_err, TestMMSEPICDetector.MAX_ERR)

        #
        # Test xla - simple precision
        #
        # sionna.Config.xla_compat = True
        # ber_lmmse = self.run_e2e('lmmse',
        #                         batch_dims,
        #                         num_rx_ant,
        #                         num_tx_ant,
        #                         ebno_dbs,
        #                         'xla',
        #                         tf.complex64)
        # ber_mmse_pic = self.run_e2e('mmse-pic',
        #                             batch_dims,
        #                             num_rx_ant,
        #                             num_tx_ant,
        #                             ebno_dbs,
        #                             'xla',
        #                             tf.complex64)
        # max_err = np.max(np.abs(ber_lmmse-ber_mmse_pic)/np.abs(ber_lmmse))
        # self.assertTrue(max_err < TestMMSEPICDetector.MAX_ERR)
        # sionna.Config.xla_compat = False

        #
        # Test eager - double precision
        #
        config.seed = config.seed
        ber_lmmse = self.run_e2e('lmmse',
                                batch_dims,
                                num_rx_ant,
                                num_tx_ant,
                                ebno_dbs,
                                'eager',
                                tf.complex128)
        config.seed = config.seed
        ber_mmse_pic = self.run_e2e('mmse-pic',
                                    batch_dims,
                                    num_rx_ant,
                                    num_tx_ant,
                                    ebno_dbs,
                                    'eager',
                                    tf.complex128)
        max_err = np.max(np.abs(ber_lmmse-ber_mmse_pic)/np.abs(ber_lmmse))
        self.assertLess(max_err, TestMMSEPICDetector.MAX_ERR)

        #
        # Test graph - double precision
        #
        config.seed = config.seed
        ber_lmmse = self.run_e2e('lmmse',
                                batch_dims,
                                num_rx_ant,
                                num_tx_ant,
                                ebno_dbs,
                                'graph',
                                tf.complex128)
        config.seed = config.seed
        ber_mmse_pic = self.run_e2e('mmse-pic',
                                    batch_dims,
                                    num_rx_ant,
                                    num_tx_ant,
                                    ebno_dbs,
                                    'graph',
                                    tf.complex128)
        max_err = np.max(np.abs(ber_lmmse-ber_mmse_pic)/np.abs(ber_lmmse))
        self.assertLess(max_err, TestMMSEPICDetector.MAX_ERR)

        #
        # Test xla - double precision
        #
        # sionna.Config.xla_compat = True
        # ber_lmmse = self.run_e2e('lmmse',
        #                         batch_dims,
        #                         num_rx_ant,
        #                         num_tx_ant,
        #                         ebno_dbs,
        #                         'xla',
        #                         tf.complex128)
        # ber_mmse_pic = self.run_e2e('mmse-pic',
        #                             batch_dims,
        #                             num_rx_ant,
        #                             num_tx_ant,
        #                             ebno_dbs,
        #                             'xla',
        #                             tf.complex128)
        # max_err = np.max(np.abs(ber_lmmse-ber_mmse_pic))
        # self.assertTrue(max_err < TestMMSEPICDetector.MAX_ERR)
        # sionna.Config.xla_compat = False

    def test_one_time_one(self):
        self.run_test([64], 1, 1, [20.0])

    def test_one_time_n(self):
        self.run_test([64], 16, 1, [-5.0])

    def test_m_time_n(self):
        self.run_test([64], 16, 4, [0.0])

    def test_batch_dims(self):
        detector = MMSEPICDetector(demapping_method="maxlog",
                                    num_iter=1,
                                    output="bit",
                                    constellation_type="qam",
                                    num_bits_per_symbol=2,
                                    dtype=tf.complex64)
        # Arbitrary batch dims [8,4,3]
        # 16 rx antennas
        # 2 tx antennas
        y = config.tf_rng.normal([8,4,3,16,2])
        y = tf.complex(y[...,0], y[...,1])
        h = config.tf_rng.normal([8,4,3,16,2,2])
        h = tf.complex(h[...,0], h[...,1])
        # Covariance matrix is the identity matrix
        s = tf.eye(16, dtype=tf.complex64)
        # Zero prior
        # 2 tx
        prior = tf.zeros([8,4,3,2,2])

        # Run the detector
        llrs = detector((y,h,prior,s))

        # Test output shape
        self.assertEqual(llrs.shape, [8,4,3,2,2])

    def test_xla(self):

        detector = MMSEPICDetector(demapping_method="maxlog",
                                    num_iter=1,
                                    output="bit",
                                    constellation_type="qam",
                                    num_bits_per_symbol=2,
                                    dtype=tf.complex64)


        @tf.function(jit_compile=True)
        def _run_xla():

            # 16 rx antennas
            # 2 tx antennas
            y = config.tf_rng.normal([64,16,2])
            y = tf.complex(y[...,0], y[...,1])
            h = config.tf_rng.normal([64,16,2,2])
            h = tf.complex(h[...,0], h[...,1])
            # Covariance matrix is the identity matrix
            s = tf.eye(16, dtype=tf.complex64)
            # Zero prior
            # 2 tx
            prior = tf.zeros([64,2,2])

            # Run the detector
            llrs = detector((y,h,prior,s))

        # Run in XLA
        _run_xla()

    def test_prior_symbols(self):

        detector = MMSEPICDetector(demapping_method="maxlog",
                                    num_iter=1,
                                    output="symbol",
                                    constellation_type="qam",
                                    num_bits_per_symbol=2, # QPSK
                                    dtype=tf.complex64)

        # 16 rx antennas
        # 2 tx antennas
        y = config.tf_rng.normal([64,16,2])
        y = tf.complex(y[...,0], y[...,1])
        h = config.tf_rng.normal([64,16,2,2])
        h = tf.complex(h[...,0], h[...,1])
        # Covariance matrix is the identity matrix
        s = tf.eye(16, dtype=tf.complex64)
        # Zero prior
        # 2 tx
        prior = config.tf_rng.normal([64,2,4]) # QPSK

        # Run the detector
        logits = detector((y,h,prior,s))

        # Test output shape
        self.assertEqual(logits.shape, [64,2,4])

    def test_multiple_iterations(self):

        detector = MMSEPICDetector(demapping_method="maxlog",
                                    num_iter=3,
                                    output="bit",
                                    constellation_type="qam",
                                    num_bits_per_symbol=2, # QPSK
                                    dtype=tf.complex64)

        # 16 rx antennas
        # 2 tx antennas
        y = config.tf_rng.normal([64,16,2])
        y = tf.complex(y[...,0], y[...,1])
        h = config.tf_rng.normal([64,16,2,2])
        h = tf.complex(h[...,0], h[...,1])
        # Covariance matrix is the identity matrix
        s = tf.eye(16, dtype=tf.complex64)
        # Zero prior
        # 2 tx
        prior = config.tf_rng.normal([64,2,2]) # QPSK

        # Run the detector
        logits = detector((y,h,prior,s))

        # Test output shape
        self.assertEqual(logits.shape, [64,2,2])
