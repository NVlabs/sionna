#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Integration tests for 5G Channel Coding and Rate-Matching: Polar vs. LDPC
Codes"""

import unittest
import numpy as np
import tensorflow as tf
import sionna
from sionna.mapping import Constellation, Mapper, Demapper
from sionna.fec.polar import PolarEncoder, Polar5GEncoder, PolarSCLDecoder, Polar5GDecoder
from sionna.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.fec.polar.utils import generate_5g_ranking, generate_rm_code
from sionna.fec.conv import ConvEncoder, ViterbiDecoder
from sionna.utils import BinarySource, ebnodb2no
from sionna.utils.metrics import  count_block_errors
from sionna.channel import AWGN
from sionna.utils.plotting import PlotBER

class System_Model(tf.keras.Model):
    """System model for channel coding BER simulations.

    This model allows to simulate BERs over an AWGN channel with
    QAM modulation. Arbitrary FEC encoder/decoder layers can be used to
    initialize the model.

    Parameters
    ----------
        k: int
            number of information bits per codeword.

        n: int
            codeword length.

        num_bits_per_symbol: int
            number of bits per QAM symbol.

        encoder: Keras layer
            A Keras layer that encodes information bit tensors.

        decoder: Keras layer
            A Keras layer that decodes llr tensors.

        demapping_method: str
            A string denoting the demapping method. Can be either "app" or
            "maxlog".

        sim_esno: bool
            A boolean defaults to False. If true, no rate-adjustment is done
            for the SNR calculation.

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
                 encoder,
                 decoder,
                 demapping_method="app",
                 sim_esno=False):

        super().__init__()

        # store values internally
        self.k = k
        self.n = n
        self.sim_esno = sim_esno # disable rate-adjustment for SNR calc
        self.num_bits_per_symbol = num_bits_per_symbol

        # init components
        self.source = BinarySource()
        self.constellation = Constellation("qam",
                                num_bits_per_symbol=self.num_bits_per_symbol)
        self.mapper = Mapper(constellation=self.constellation)
        self.demapper = Demapper(demapping_method,
                                 constellation=self.constellation)
        self.channel = AWGN()
        self.encoder = encoder
        self.decoder = decoder

    @tf.function()
    def call(self, batch_size, ebno_db):

        no = ebnodb2no(ebno_db,
                       num_bits_per_symbol=self.num_bits_per_symbol,
                       coderate=self.k/self.n)

        u = self.source([batch_size, self.k]) # generate random data
        c = self.encoder(u) # explicitly encode
        x = self.mapper(c) # map c to symbols x
        y = self.channel([x, no]) # transmit over AWGN channel
        llr_ch = self.demapper([y, no]) # demap y to LLRs
        u_hat = self.decoder(llr_ch) # run FEC decoder (incl. rate-recovery)

        return u, u_hat


class TestFEC(unittest.TestCase):

    def test_short_codes(self):

        # simulation/code parameters
        # these parameters are reduced when compared to the notebook
        max_mc_iter = 2
        k = 64 # number of information bits per codeword
        n = 128 # desired codeword length
        num_bits_per_symbol = 2 # QPSK
        ebno_db = np.arange(0, 5, 0.5) # sim SNR range

        # Create list of encoder/decoder pairs to be analyzed.
        codes_under_test = []

        # 5G LDPC codes with 20 BP iterations
        enc = LDPC5GEncoder(k=k, n=n)
        dec = LDPC5GDecoder(enc, num_iter=20)
        name = "5G LDPC BP-20"
        codes_under_test.append([enc, dec, name])

        # Polar Codes (SC decoding)
        enc = Polar5GEncoder(k=k, n=n)
        dec = Polar5GDecoder(enc, dec_type="SC")
        name = "5G Polar SC"
        codes_under_test.append([enc, dec, name])

        # Polar Codes (SCL decoding) with list size 8.
        # The CRC is automatically added by the layer.
        enc = Polar5GEncoder(k=k, n=n)
        dec = Polar5GDecoder(enc, dec_type="SCL", list_size=8)
        name = "5G Polar SCL-8+CRC UL configuration"
        codes_under_test.append([enc, dec, name])

        # Polar Codes (SCL decoding) with list size 8.
        # The CRC is automatically added by the layer.
        enc = Polar5GEncoder(k=k, n=n, channel_type="downlink")
        dec = Polar5GDecoder(enc, dec_type="SCL", list_size=8)
        name = "5G Polar SCL-8+CRC DL configuration"
        codes_under_test.append([enc, dec, name])

        # RM codes with SCL decoding
        f,_,_,_,_ = generate_rm_code(3,7) # equals k=64 and n=128 code
        enc = PolarEncoder(f, n)
        dec = PolarSCLDecoder(f, n, list_size=8)
        name = "Reed Muller (RM) SCL-8"
        codes_under_test.append([enc, dec, name])

        # Conv. Code with Viterbi decoding can be added here
        enc = ConvEncoder(rate=0.5, constraint_length=8)
        dec = ViterbiDecoder(gen_poly=enc.gen_poly, method="soft_llr")
        name = "Convolutional (constraint length 8)"
        codes_under_test.append([enc, dec, name])

        ber_plot = PlotBER()

        # run ber simulations for each code we have added to the list
        for code in codes_under_test:

            # generate a new model with the given encoder/decoder
            model = System_Model(k=k,
                                 n=n,
                                 num_bits_per_symbol=num_bits_per_symbol,
                                 encoder=code[0],
                                decoder=code[1])

            ber_plot.simulate(model,
                              ebno_dbs=ebno_db,
                              legend=code[2],
                              max_mc_iter=max_mc_iter,
                              num_target_bit_errors=1,
                              batch_size=10,
                              soft_estimates=False,
                              early_stop=True,
                              show_fig=False,
                              add_bler=True,
                              forward_keyboard_interrupt=True);

        # check results for consistency
        for ber in ber_plot.ber:
            self.assertFalse(np.isnan(ber).any())

    def test_different_length_ldpc(self):

        # simulation parameters
        # these parameters are reduced when compared to the notebook
        max_mc_iter = 2
        ns = [128, 512, 2000, 16000]
        rate = 0.5 # fixed coderate
        num_bits_per_symbol = 2 # QPSK
        ebno_db = np.arange(0, 5, 0.25) # sim SNR range

        # init new figure
        ber_plot = PlotBER()

        # create list of encoder/decoder pairs to be analyzed
        codes_under_test = []

        # 5G LDPC codes
        for n in ns:
            k = int(rate*n) # calculate k for given n and rate
            enc = LDPC5GEncoder(k=k, n=n)
            dec = LDPC5GDecoder(enc, num_iter=20)
            name = f"5G LDPC BP-20 (n={n})"
            codes_under_test.append([enc, dec, name, k, n])

        # run ber simulations for each case
        for code in codes_under_test:
            model = System_Model(k=code[3],
                                 n=code[4],
                                 num_bits_per_symbol=num_bits_per_symbol,
                                 encoder=code[0],
                                 decoder=code[1])

            ber_plot.simulate(model, # the function have defined previously
                              ebno_dbs=ebno_db,
                              legend=code[2],
                              max_mc_iter=max_mc_iter,
                              num_target_block_errors=1,
                              batch_size=10,
                              soft_estimates=False,
                              early_stop=True,
                              show_fig=False,
                              forward_keyboard_interrupt=True);

        # check results for consistency
        for ber in ber_plot.ber:
            self.assertFalse(np.isnan(ber).any())

    def test_polar(self):

        batch_size = 2
        k = 32
        n = 64
        # load 5G compliant channel ranking
        frozen_pos, info_pos = generate_5g_ranking(k,n)

        # and RM codes
        r = 3
        m = 7
        frozen_pos, info_pos, n, k, d_min = generate_rm_code(r, m)

        encoder_polar = PolarEncoder(frozen_pos, n)
        source = BinarySource()
        u = source([batch_size, k])
        c = encoder_polar(u)

    def test_find_threshold(self):

        num_bits_per_symbol = 2 # QPSK

        # find the EsNo in dB to achieve target_bler
        def find_threshold(model,
                           batch_size=10,
                           max_batch_iter=1,
                           max_block_errors=1,
                           target_bler=1e-3):
            """Bisection search to find required SNR to reach target SNR."""

            # bisection parameters
            esno_db_min = -15 # smallest possible search SNR
            esno_db_max = 15 # largest possible search SNR
            esno_interval = (esno_db_max-esno_db_min)/4
            esno_db = 2*esno_interval + esno_db_min # current test SNR
            max_iters = 2 # number of iterations for bisection search

            # run bisection
            for i in range(max_iters):
                num_block_error = 0
                num_cws = 0
                for j in range(max_batch_iter):
                    # run model and evaluate BLER
                    u, u_hat = model(tf.constant(batch_size, tf.int32),
                                    tf.constant(esno_db, tf.float32))
                    num_block_error += count_block_errors(u, u_hat)
                    num_cws += batch_size
                    # early stop if target number of block errors is reached
                    if num_block_error>max_block_errors:
                        break
                bler = num_block_error/num_cws
                # increase SNR if BLER was great than target
                # (larger SNR leads to decreases BLER)
                if bler>target_bler:
                    esno_db += esno_interval
                else: # and decrease SNR otherwise
                    esno_db -= esno_interval
                esno_interval = esno_interval/2

            # return final SNR after max_iters
            return esno_db

        # run simulations for multiple code parameters
        ks = np.array([12, 30])
        ns = np.array([28, 134]) # reduced number of evaluations for testing

        # we use EsNo instead of EbNo to have the same results as in [4]
        esno = np.zeros([len(ns), len(ks)])

        for j,n in enumerate(ns):
            for i,k in enumerate(ks):
                if k<n:
                    # initialize new encoder / decoder pair
                    enc = Polar5GEncoder(k=k, n=n)
                    dec = Polar5GDecoder(enc, dec_type="SCL", list_size=8)
                    #build model
                    model = System_Model(k=k,
                                        n=n,
                                        num_bits_per_symbol=num_bits_per_symbol,
                                        encoder=enc,
                                        decoder=dec,
                                        sim_esno=True) # no rate adjustment
                    # and find threshold via bisection search
                    esno[j, i] = find_threshold(model)

