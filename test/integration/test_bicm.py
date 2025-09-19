#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""Integration tests for Bit-Interleaved Coded Modulation"""

import unittest
import numpy as np
import tensorflow as tf

from sionna.phy import Block
from sionna.phy.mapping import Constellation, Mapper, Demapper, BinarySource
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.fec.interleaving import RandomInterleaver, Deinterleaver
from sionna.phy.fec.scrambling import Scrambler, Descrambler
from sionna.phy.utils import ebnodb2no, hard_decisions
from sionna.phy.utils.plotting import PlotBER
from sionna.phy.channel import AWGN

class LDPC_QAM_AWGN(Block):
    """System model for channel coding BER simulations.

    This model allows to simulate BERs over an AWGN channel with
    QAM modulation. It can enable/disable multiple options to analyze all-zero
    codeword simulations.

    If active, the system uses the 5G LDPC encoder/decoder module.

    Parameters
    ----------
        k: int
            number of information bits per codeword.

        n: int
            codeword length.

        num_bits_per_symbol: int
            number of bits per QAM symbol.

        demapping_method: str
            A string defining the demapping method. Can be either "app" or
            "maxlog".

        decoder_type: str
            A string defining the check node update function type of the LDPC
            decoder.

        use_allzero: bool
            A boolean defaults to False. If True, no encoder is used and
            all-zero codewords are sent.

        use_scrambler: bool
            A boolean defaults to False. If True, a scrambler after the encoder
            and a descrambler before the decoder is used, respectively.

        no_est_mismatch: float
            A float defaults to 1.0. Defines the SNR estimation mismatch of the
            demapper such that the effective demapping
            noise variance estimate is the scaled by ``no_est_mismatch``
            version of the true noise_variance

    Input
    -----
        batch_size: int or tf.int
            The batch_size used for the simulation.

        ebno_db: float or tf.float
            A float defining the simulation SNR.

    Output
    ------
        (u, u_hat):
            Tuple:

        u: tf.float32
            A tensor of shape `[batch_size, k] of 0s and 1s containing the
            transmitted information bits.

        u_hat: tf.float32
            A tensor of shape `[batch_size, k] of 0s and 1s containing the
            estimated information bits.
    """

    def __init__(self,
                 k,
                 n,
                 num_bits_per_symbol,
                 demapping_method="app",
                 decoder_type="boxplus",
                 use_allzero=False,
                 use_scrambler=False,
                 no_est_mismatch=1.):
        super().__init__()
        self.k = k
        self.n = n

        self.num_bits_per_symbol = num_bits_per_symbol

        self.use_allzero = use_allzero
        self.use_scrambler = use_scrambler

        # adds noise to SNR estimation at demapper
        self.no_est_mismatch = no_est_mismatch

        # init components
        self.source = BinarySource()

        # initialize mapper and demapper with constellation object
        self.constellation = Constellation("qam",
                                num_bits_per_symbol=self.num_bits_per_symbol)
        self.mapper = Mapper(constellation=self.constellation)
        self.demapper = Demapper(demapping_method,
                                 constellation=self.constellation)

        self.channel = AWGN()

        # LDPC encoder / decoder
        self.encoder = LDPC5GEncoder(self.k, self.n)
        self.decoder = LDPC5GDecoder(self.encoder, cn_update=decoder_type)

        self.scrambler = Scrambler()
        # connect descrambler to scrambler
        self.descrambler = Descrambler(self.scrambler, binary=False)

    @tf.function() # enable graph mode for higher throughputs
    def call(self, batch_size, ebno_db):

        # calculate noise variance
        no = ebnodb2no(ebno_db,
                       num_bits_per_symbol=self.num_bits_per_symbol,
                       coderate=self.k/self.n)

        if self.use_allzero:
            u = tf.zeros([batch_size, self.k]) # only needed for
            c = tf.zeros([batch_size, self.n])
        else:
            u = self.source([batch_size, self.k])
            c = self.encoder(u) # explicitly encode

        # scramble codeword if actively required
        if self.use_scrambler:
            c = self.scrambler(c)

        x = self.mapper(c) # map c to symbols

        y = self.channel(x, no) # transmit over AWGN channel

        no_est = no * self.no_est_mismatch
        llr_ch = self.demapper(y, no_est) # demap

        if self.use_scrambler:
            llr_ch = self.descrambler(llr_ch)

        u_hat = self.decoder(llr_ch) # run LDPC decoder (incl. de-rate-matching)

        return u, u_hat

class TestBICM(unittest.TestCase):

    def test_simple_e2e(self):

        # simulation parameters
        batch_size = int(1e6) # number of symbols to be analyzed
        num_bits_per_symbol = 4 # bits per modulated symbol, i.e., 2^4 = 16-QAM
        ebno_db = 4 # simulation SNR

        # generate 16QAM with Gray labeling
        constellation = Constellation("qam", num_bits_per_symbol)
        constellation.show()

        # init system components
        source = BinarySource()
        channel = AWGN()
        no = ebnodb2no(ebno_db, num_bits_per_symbol, coderate=1.)

        # and generate bins for the histogram
        llr_bins = np.arange(-20,20,0.1)
        constellation = Constellation("qam",
                            num_bits_per_symbol=num_bits_per_symbol)
        mapper = Mapper(constellation=constellation)
        demapper = Demapper("app", constellation=constellation)

        b = source([batch_size, num_bits_per_symbol])
        x = mapper(b)
        y = channel(x, no)
        llr = demapper(y, no)
        llr_b = tf.multiply(llr, (2.*b-1.))
        llr_dist = []
        for i in range(num_bits_per_symbol):

            llr_np = tf.reshape(llr_b[:,i],[-1]).numpy()
            t, _ = np.histogram(llr_np, bins=llr_bins, density=True);
            llr_dist.append(t)

        # calculate bitwise BERs
        b_hat = hard_decisions(llr) # hard decide the LLRs
        errors = tf.cast(tf.not_equal(b, b_hat), tf.float32)
        ber_per_bit = tf.reduce_mean(errors, axis=0)

        # test for consistency
        self.assertFalse(np.isnan(ber_per_bit).any())

        # Run BER sims
        ber_plot = PlotBER()
        num_bits_per_symbol = 2 # QPSK
        num_bp_iter = 20 # number of decoder iterations
        k = 500 # number of information bits per codeword
        n = 1000 # number of codeword bits

        encoder = LDPC5GEncoder(k, n)
        decoder = LDPC5GDecoder(encoder,
                                cn_update="boxplus-phi",
                                num_iter=num_bp_iter)

        # initialize a random interleaver and corresponding deinterleaver
        interleaver = RandomInterleaver()
        deinterleaver = Deinterleaver(interleaver)

        # mapper and demapper
        constellation = Constellation("qam",
            num_bits_per_symbol=num_bits_per_symbol)
        mapper = Mapper(constellation=constellation)
        demapper = Demapper("app", constellation=constellation)

        # define system
        @tf.function()
        def run_ber(batch_size, ebno_db):
            # calculate noise variance
            no = ebnodb2no(ebno_db,
                        num_bits_per_symbol=num_bits_per_symbol,
                        coderate=k/n)
            u = source([batch_size, k])
            c = encoder(u)
            c_int = interleaver(c)
            x = mapper(c_int) # map to symbol (QPSK)
            y = channel(x, no) # transmit over AWGN channel
            llr_ch = demapper(y, no) # demapp
            llr_deint = deinterleaver(llr_ch)
            u_hat = decoder(llr_deint)

            return u, u_hat

        ber_plot.simulate(run_ber,
                          ebno_dbs=np.arange(0, 5, 0.25),
                          legend="Baseline (with encoder)",
                          max_mc_iter=2,
                          num_target_bit_errors=1,
                          batch_size=1,
                          soft_estimates=False,
                          early_stop=True,
                          show_fig=False)

        # check results for consistency
        for ber in ber_plot.ber:
            self.assertFalse(np.isnan(ber).any())

    def test_model_e2e(self):

        # code parameters
        k = 500 # number of information bits per codeword
        n = 1000 # number of codeword bits
        ber_plot = PlotBER("")

        model_allzero = LDPC_QAM_AWGN(k,
                                      n,
                                      num_bits_per_symbol=2,
                                      use_allzero=True,
                                      use_scrambler=False)


        ber_plot.simulate(model_allzero,
                          ebno_dbs=np.arange(0, 5, 0.25),
                          legend="All-zero / QPSK (no encoder)",
                          max_mc_iter=2,
                          num_target_bit_errors=1,
                          batch_size=1,
                          soft_estimates=False,
                          show_fig=False);

        model_baseline_16 = LDPC_QAM_AWGN(k,
                                          n,
                                          num_bits_per_symbol=4,
                                          use_allzero=False,
                                          use_scrambler=False)

        ber_plot.simulate(model_baseline_16,
                          ebno_dbs=np.arange(0, 5, 0.25),
                          legend="Baseline 16-QAM",
                          max_mc_iter=2,
                          num_target_bit_errors=1,
                          batch_size=1,
                          soft_estimates=False,
                          show_fig=False);

        # check results for consistency
        for ber in ber_plot.ber:
            self.assertFalse(np.isnan(ber).any())
