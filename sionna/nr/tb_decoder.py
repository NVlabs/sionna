#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Transport block decoding functions for the 5g NR sub-package of Sionna.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from sionna.fec.crc import CRCDecoder
from sionna.fec.scrambling import  Descrambler
from sionna.fec.ldpc import LDPC5GDecoder
from sionna.nr import TBEncoder

class TBDecoder(Layer):
    # pylint: disable=line-too-long
    r"""TBDecoder(encoder, num_bp_iter=20, cn_type="boxplus-phi", output_dtype=tf.float32, **kwargs)
    5G NR transport block (TB) decoder as defined in TS 38.214
    [3GPP38214]_.

    The transport block decoder takes as input a sequence of noisy channel
    observations and reconstructs the corresponding `transport block` of
    information bits. The detailed procedure is described in TS 38.214
    [3GPP38214]_ and TS 38.211 [3GPP38211]_.

    The class inherits from the Keras layer class and can be used as layer in a
    Keras model.

    Parameters
    ----------
        encoder : :class:`~sionna.nr.TBEncoder`
            Associated transport block encoder used for encoding of the signal.

        num_bp_iter : int, 20 (default)
            Number of BP decoder iterations

        cn_type : str, "boxplus-phi" (default) | "boxplus" | "minsum"
            The check node processing function of the LDPC BP decoder.
            One of {`"boxplus"`, `"boxplus-phi"`, `"minsum"`} where
            '"boxplus"' implements the single-parity-check APP decoding rule.
            '"boxplus-phi"' implements the numerical more stable version of
            boxplus [Ryan]_.
            '"minsum"' implements the min-approximation of the CN update rule
            [Ryan]_.

        output_dtype : tf.float32 (default)
            Defines the datatype for internal calculations and the output dtype.

    Input
    -----
        inputs : [...,num_coded_bits], tf.float
            2+D tensor containing channel logits/llr values of the (noisy)
            channel observations.

    Output
    ------
        b_hat : [...,target_tb_size], tf.float
            2+D tensor containing hard decided bit estimates of all information
            bits of the transport block.

        tb_crc_status : [...], tf.bool
            Transport block CRC status indicating if a transport block was
            (most likely) correctly recovered. Note that false positives are
            possible.
    """

    def __init__(self,
                 encoder,
                 num_bp_iter=20,
                 cn_type="boxplus-phi",
                 output_dtype=tf.float32,
                 **kwargs):

        super().__init__(dtype=output_dtype, **kwargs)

        assert output_dtype in (tf.float16, tf.float32, tf.float64), \
                "output_dtype must be (tf.float16, tf.float32, tf.float64)."

        assert isinstance(encoder, TBEncoder), "encoder must be TBEncoder."
        self._tb_encoder = encoder

        self._num_cbs = encoder.num_cbs

        # init BP decoder
        self._decoder = LDPC5GDecoder(encoder=encoder.ldpc_encoder,
                                      num_iter=num_bp_iter,
                                      cn_type=cn_type,
                                      hard_out=True, # TB operates on bit-level
                                      return_infobits=True,
                                      output_dtype=output_dtype)

        # init descrambler
        if encoder.scrambler is not None:
            self._descrambler = Descrambler(encoder.scrambler,
                                            binary=False)
        else:
            self._descrambler = None

        # init CRC Decoder for CB and TB
        self._tb_crc_decoder = CRCDecoder(encoder.tb_crc_encoder)

        if encoder.cb_crc_encoder is not None:
            self._cb_crc_decoder = CRCDecoder(encoder.cb_crc_encoder)
        else:
            self._cb_crc_decoder = None

    #########################################
    # Public methods and properties
    #########################################

    @property
    def tb_size(self):
        """Number of information bits per TB."""
        return self._tb_encoder.tb_size

    # required for
    @property
    def k(self):
        """Number of input information bits. Equals TB size."""
        return self._tb_encoder.tb_size

    @property
    def n(self):
        "Total number of output codeword bits."
        return self._tb_encoder.n

    #########################
    # Keras layer functions
    #########################

    def build(self, input_shapes):
        """Test input shapes for consistency."""

        assert input_shapes[-1]==self.n, \
            f"Invalid input shape. Expected input length is {self.n}."

    def call(self, inputs):
        """Apply transport block decoding."""

        # store shapes
        input_shape = inputs.shape.as_list()
        llr_ch = tf.cast(inputs, tf.float32)

        llr_ch = tf.reshape(llr_ch,
                            (-1, self._tb_encoder.num_tx, self._tb_encoder.n))

        # undo scrambling (only if scrambler was used)
        if self._descrambler is not None:
            llr_scr = self._descrambler(llr_ch)
        else:
            llr_scr = llr_ch

        # undo CB interleaving and puncturing
        num_fillers = self._tb_encoder.ldpc_encoder.n * self._tb_encoder.num_cbs - np.sum(self._tb_encoder.cw_lengths)
        llr_int = tf.concat([llr_scr,
                            tf.zeros([tf.shape(llr_scr)[0], self._tb_encoder.num_tx, num_fillers])], axis=-1)
        llr_int = tf.gather(llr_int, self._tb_encoder.output_perm_inv, axis=-1)

        # undo CB concatenation
        llr_cb = tf.reshape(llr_int,
                        (-1, self._tb_encoder.num_tx, self._num_cbs, self._tb_encoder.ldpc_encoder.n))

        # LDPC decoding
        u_hat_cb = self._decoder(llr_cb)

        # CB CRC removal (if relevant)
        if self._cb_crc_decoder is not None:
            # we are ignoring the CB CRC status for the moment
            # Could be combined with the TB CRC for even better estimates
            u_hat_cb_crc, _ = self._cb_crc_decoder(u_hat_cb)
        else:
            u_hat_cb_crc = u_hat_cb

        # undo CB segmentation
        u_hat_tb = tf.reshape(u_hat_cb_crc,
                (-1, self._tb_encoder.num_tx, self.tb_size+self._tb_encoder.tb_crc_encoder.crc_length))

        # TB CRC removal
        u_hat, tb_crc_status = self._tb_crc_decoder(u_hat_tb)

        # restore input shape
        output_shape = input_shape
        output_shape[0] = -1
        output_shape[-1] = self.tb_size
        u_hat = tf.reshape(u_hat, output_shape)
        # also apply to tb_crc_status
        output_shape[-1] = 1 # but last dim is 1
        tb_crc_status = tf.reshape(tb_crc_status, output_shape)

        # remove if zero-padding was applied
        if self._tb_encoder.k_padding>0:
            u_hat = u_hat[...,:-self._tb_encoder.k_padding]

        # cast to output dtype
        u_hat = tf.cast(u_hat, self.dtype)
        tb_crc_status = tf.squeeze(tf.cast(tb_crc_status, tf.bool), axis=-1)

        return u_hat, tb_crc_status
