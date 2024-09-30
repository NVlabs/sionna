#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Integration tests for Weighted BP Decoding"""

import unittest
import numpy as np
import tensorflow as tf
from sionna.fec.ldpc import LDPCBPDecoder, LDPC5GEncoder, LDPC5GDecoder
from sionna.utils.metrics import BitwiseMutualInformation
from sionna.fec.utils import GaussianPriorSource, load_parity_check_examples
from sionna.utils import ebnodb2no
from sionna.utils.plotting import PlotBER
from tensorflow.keras.losses import BinaryCrossentropy


class WeightedBP(tf.keras.Model):
    """System model for BER simulations of weighted BP decoding.

    This model uses `GaussianPriorSource` to mimic the LLRs after demapping of
    QPSK symbols transmitted over an AWGN channel.

    Parameters
    ----------
        pcm: ndarray
            The parity-check matrix of the code under investigation.

        num_iter: int
            Number of BP decoding iterations.

        coderate: float
            Coderate of the code.

    Input
    -----
        batch_size: int or tf.int
            The batch_size used for the simulation.

        ebno_db: float or tf.float
            A float defining the simulation SNR.

    Output
    ------
        (u, u_hat, loss):
            Tuple:

        u: tf.float32
            A tensor of shape `[batch_size, k] of 0s and 1s containing the
            transmitted information bits.

        u_hat: tf.float32
            A tensor of shape `[batch_size, k] of 0s and 1s containing the
            estimated information bits.

        loss: tf.float32
            Binary cross-entropy loss between `u` and `u_hat`.
    """
    def __init__(self, pcm, num_iter=5, coderate=1.):
        super().__init__()

        # init components
        self.decoder = LDPCBPDecoder(pcm,
                                     num_iter=1,
                                     stateful=True,
                                     hard_out=False,
                                     cn_type="boxplus",
                                     trainable=True)

        self.llr_source = GaussianPriorSource()
        self._num_iter = num_iter
        self._coderate = coderate
        self._n = pcm.shape[1]

        self._bce = BinaryCrossentropy(from_logits=True)

    def call(self, batch_size, ebno_db):
        noise_var = ebnodb2no(ebno_db,
                              num_bits_per_symbol=2, # QPSK
                              coderate=self._coderate)

        # all-zero CW to calculate loss / BER
        c = tf.zeros([batch_size, self._n])
        llr = self.llr_source([[batch_size, self._n], noise_var])
        loss = 0
        msg_vn = None # no msg_vn for first iteration available
        for i in range(self._num_iter):
            c_hat, msg_vn = self.decoder([llr, msg_vn])
            loss += self._bce(c, c_hat)  # add loss after each iteration

        loss /= self._num_iter # scale loss by number of iterations

        return c, c_hat, loss

class WeightedBP5G(tf.keras.Model):
    """System model for BER simulations of weighted BP decoding for 5G LDPC
    codes.

    This model uses `GaussianPriorSource` to mimic the LLRs after demapping of
    QPSK symbols transmitted over an AWGN channel.

    Parameters
    ----------
        k: int
            Number of information bits per codeword.

        n: int
            Codeword length.

        num_iter: int
            Number of BP decoding iterations.

    Input
    -----
        batch_size: int or tf.int
            The batch_size used for the simulation.

        ebno_db: float or tf.float
            A float defining the simulation SNR.

    Output
    ------
        (u, u_hat, loss):
            Tuple:

        u: tf.float32
            A tensor of shape `[batch_size, k] of 0s and 1s containing the
            transmitted information bits.

        u_hat: tf.float32
            A tensor of shape `[batch_size, k] of 0s and 1s containing the
            estimated information bits.

        loss: tf.float32
            Binary cross-entropy loss between `u` and `u_hat`.
    """
    def __init__(self, k, n, num_iter=20):
        super().__init__()

        # we need to initialize an encoder for the 5G parameters
        self.encoder = LDPC5GEncoder(k, n)
        self.decoder = LDPC5GDecoder(self.encoder,
                                     num_iter=1,
                                     stateful=True,
                                     hard_out=False,
                                     cn_type="boxplus",
                                     trainable=True)

        self.llr_source = GaussianPriorSource()
        self._num_iter = num_iter
        self._coderate = k/n
        self._k = k
        self._n = n
        self._bce = BinaryCrossentropy(from_logits=True)

    def call(self, batch_size, ebno_db):

        noise_var = ebnodb2no(ebno_db,
                              num_bits_per_symbol=2, # QPSK
                              coderate=self._coderate)

        c = tf.zeros([batch_size, self._k]) # decoder only returns info bits
        llr = self.llr_source([[batch_size, self._n], noise_var])
        loss = 0
        msg_vn = None # no msg_vn for first iteration available
        for i in range(self._num_iter):
            c_hat, msg_vn = self.decoder([llr, msg_vn])
            loss += self._bce(c, c_hat)  # add loss after each iteration

        return c, c_hat, loss

class TestWeightedBP(unittest.TestCase):

    def test_simple_training(self):
        pcm_id = 1 # (63,45) BCH code parity check matrix
        pcm, k , n, coderate = load_parity_check_examples(pcm_id=pcm_id)
        num_iter = 10 # set number of decoding iterations
        ebno_dbs = np.array(np.arange(1, 7, 0.5))
        mc_iters = 1 # number of Monte Carlo iterations
        ber_plot = PlotBER()

        # training parameters
        batch_size = 10
        train_iter = 5
        ebno_db = 4.0
        clip_value_grad = 10 # gradient clipping for stable training convergence

        # and initialize the model
        model = WeightedBP(pcm=pcm, num_iter=num_iter, coderate=coderate)
        model.decoder.show_weights()

        # simulate and plot the BER curve of the untrained decoder
        ber_plot.simulate(model,
                          ebno_dbs=ebno_dbs,
                          batch_size=1,
                          num_target_bit_errors=2,
                          legend="Untrained",
                          soft_estimates=True,
                          max_mc_iter=mc_iters,
                          forward_keyboard_interrupt=False);

        # check results for consistency
        for ber in ber_plot.ber:
            self.assertFalse(np.isnan(ber).any(), "ber is nan.")

        # try also different optimizers or different hyperparameters
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

        for it in range(0, train_iter):
            with tf.GradientTape() as tape:
                b, llr, loss = model(batch_size, ebno_db)

            grads = tape.gradient(loss, model.trainable_variables)
            grads = tf.clip_by_value(grads,
                                     -clip_value_grad,
                                     clip_value_grad,
                                     name=None)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # check that gradients are not nan
            self.assertFalse(np.isnan(grads.numpy()).any(), "grads are nan.")


    def test_training_5GLDPC(self):

        # generate model
        num_iter = 10
        k = 400
        n = 800
        ebno_dbs = np.array(np.arange(0, 4, 0.25))
        mc_iters = 1 # number of monte carlo iterations
        ber_plot_5G = PlotBER("")
        # training parameters
        batch_size = 10
        train_iter = 5
        clip_value_grad = 10 # gradient clipping seems to be important
        ebno_db = 1.5 # for training

        model5G = WeightedBP5G(k, n, num_iter=num_iter)

        # simulate the untrained performance
        ber_plot_5G.simulate(model5G,
                             ebno_dbs=ebno_dbs,
                             batch_size=1,
                             num_target_bit_errors=2,
                             legend="Untrained",
                             soft_estimates=True,
                             max_mc_iter=mc_iters);

        # check results for consistency
        for ber in ber_plot_5G.ber:
            self.assertFalse(np.isnan(ber).any())

        # try also different optimizers or different hyperparameters
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

        # and let's go
        for it in range(0, train_iter):
            with tf.GradientTape() as tape:
                b, llr, loss = model5G(batch_size, ebno_db)

            grads = tape.gradient(loss, model5G.trainable_variables)
            grads = tf.clip_by_value(grads,
                                     -clip_value_grad,
                                     clip_value_grad,
                                     name=None)
            optimizer.apply_gradients(zip(grads, model5G.trainable_weights))

            # check that gradients are not nan
            self.assertFalse(np.isnan(grads.numpy()).any())
