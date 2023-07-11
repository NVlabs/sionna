#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Transport block encoding functions for the 5g NR sub-package of Sionna.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from sionna.fec.crc import CRCEncoder
from sionna.fec.scrambling import TB5GScrambler
from sionna.fec.ldpc import LDPC5GEncoder
from sionna.nr.utils import calculate_tb_size

class TBEncoder(Layer):
    # pylint: disable=line-too-long
    r"""TBEncoder(target_tb_size,num_coded_bits,target_coderate,num_bits_per_symbol,num_layers=1,n_rnti=1,n_id=1,channel_type="PUSCH",codeword_index=0,use_scrambler=True,verbose=False,output_dtype=tf.float32,, **kwargs)
    5G NR transport block (TB) encoder as defined in TS 38.214
    [3GPP38214]_ and TS 38.211 [3GPP38211]_

    The transport block (TB) encoder takes as input a `transport block` of
    information bits and generates a sequence of codewords for transmission.
    For this, the information bit sequence is segmented into multiple codewords,
    protected by additional CRC checks and FEC encoded. Further, interleaving
    and scrambling is applied before a codeword concatenation generates the
    final bit sequence. Fig. 1 provides an overview of the TB encoding
    procedure and we refer the interested reader to [3GPP38214]_ and
    [3GPP38211]_ for further details.

    ..  figure:: ../figures/tb_encoding.png

        Fig. 1: Overview TB encoding (CB CRC does not always apply).

    If ``n_rnti`` and ``n_id`` are given as list, the TBEncoder encodes
    `num_tx = len(` ``n_rnti`` `)` parallel input streams with different
    scrambling sequences per user.

    The class inherits from the Keras layer class and can be used as layer in a
    Keras model.

    Parameters
    ----------
        target_tb_size: int
            Target transport block size, i.e., how many information bits are
            encoded into the TB. Note that the effective TB size can be
            slightly different due to quantization. If required, zero padding
            is internally applied.

        num_coded_bits: int
            Number of coded bits after TB encoding.

        target_coderate : float
            Target coderate.

        num_bits_per_symbol: int
            Modulation order, i.e., number of bits per QAM symbol.

        num_layers: int, 1 (default) | [1,...,8]
            Number of transmission layers.

        n_rnti: int or list of ints, 1 (default) | [0,...,65335]
            RNTI identifier provided by higher layer. Defaults to 1 and must be
            in range `[0, 65335]`. Defines a part of the random seed of the
            scrambler. If provided as list, every list entry defines the RNTI
            of an independent input stream.

        n_id: int or list of ints, 1 (default) | [0,...,1023]
            Data scrambling ID :math:`n_\text{ID}` related to cell id and
            provided by higher layer.
            Defaults to 1 and must be in range `[0, 1023]`. If provided as
            list, every list entry defines the scrambling id of an independent
            input stream.

        channel_type: str, "PUSCH" (default) | "PDSCH"
            Can be either "PUSCH" or "PDSCH".

        codeword_index: int, 0 (default) | 1
            Scrambler can be configured for two codeword transmission.
            ``codeword_index`` can be either 0 or 1. Must be 0 for
            ``channel_type`` = "PUSCH".

        use_scrambler: bool, True (default)
            If False, no data scrambling is applied (non standard-compliant).

        verbose: bool, False (default)
            If `True`, additional parameters are printed during initialization.

        dtype: tf.float32 (default)
            Defines the datatype for internal calculations and the output dtype.

    Input
    -----
        inputs: [...,target_tb_size] or [...,num_tx,target_tb_size], tf.float
            2+D tensor containing the information bits to be encoded. If
            ``n_rnti`` and ``n_id`` are a list of size `num_tx`, the input must
            be of shape `[...,num_tx,target_tb_size]`.

    Output
    ------
        : [...,num_coded_bits], tf.float
            2+D tensor containing the sequence of the encoded codeword bits of
            the transport block.

    Note
    ----
    The parameters ``tb_size`` and ``num_coded_bits`` can be derived by the
    :meth:`~sionna.nr.calculate_tb_size` function or
    by accessing the corresponding :class:`~sionna.nr.PUSCHConfig` attributes.
    """

    def __init__(self,
                 target_tb_size,
                 num_coded_bits,
                 target_coderate,
                 num_bits_per_symbol,
                 num_layers=1,
                 n_rnti=1,
                 n_id=1,
                 channel_type="PUSCH",
                 codeword_index=0,
                 use_scrambler=True,
                 verbose=False,
                 output_dtype=tf.float32,
                 **kwargs):

        super().__init__(dtype=output_dtype, **kwargs)

        assert isinstance(use_scrambler, bool), \
                                "use_scrambler must be bool."
        self._use_scrambler = use_scrambler
        assert isinstance(verbose, bool), \
                                "verbose must be bool."
        self._verbose = verbose

        # check input for consistency
        assert channel_type in ("PDSCH", "PUSCH"), \
                                "Unsupported channel_type."
        self._channel_type = channel_type

        assert(target_tb_size%1==0), "target_tb_size must be int."
        self._target_tb_size = int(target_tb_size)

        assert(num_coded_bits%1==0), "num_coded_bits must be int."
        self._num_coded_bits = int(num_coded_bits)

        assert(0.<target_coderate <= 948/1024), \
                    "target_coderate must be in range(0,0.925)."
        self._target_coderate = target_coderate

        assert(num_bits_per_symbol%1==0), "num_bits_per_symbol must be int."
        self._num_bits_per_symbol = int(num_bits_per_symbol)

        assert(num_layers%1==0), "num_layers must be int."
        self._num_layers = int(num_layers)

        if channel_type=="PDSCH":
            assert(codeword_index in (0,1)), "codeword_index must be 0 or 1."
        else:
            assert codeword_index==0, 'codeword_index must be 0 for "PUSCH".'
        self._codeword_index = int(codeword_index)

        if isinstance(n_rnti, (list, tuple)):
            assert isinstance(n_id, (list, tuple)), "n_id must be also a list."
            assert (len(n_rnti)==len(n_id)), \
                                "n_id and n_rnti must be of same length."
            self._n_rnti = n_rnti
            self._n_id = n_id
        else:
            self._n_rnti = [n_rnti]
            self._n_id = [n_id]

        for idx, n in enumerate(self._n_rnti):
            assert(n%1==0), "n_rnti must be int."
            self._n_rnti[idx] = int(n)
        for idx, n in enumerate(self._n_id):
            assert(n%1==0), "n_id must be int."
            self._n_id[idx] = int(n)

        self._num_tx = len(self._n_id)

        tbconfig = calculate_tb_size(target_tb_size=self._target_tb_size,
                                     num_coded_bits=self._num_coded_bits,
                                     target_coderate=self._target_coderate,
                                     modulation_order=self._num_bits_per_symbol,
                                     num_layers=self._num_layers,
                                     verbose=verbose)
        self._tb_size = tbconfig[0]
        self._cb_size = tbconfig[1]
        self._num_cbs = tbconfig[2]
        self._cw_lengths = tbconfig[3]
        self._tb_crc_length = tbconfig[4]
        self._cb_crc_length = tbconfig[5]

        assert self._tb_size <= self._tb_crc_length + np.sum(self._cw_lengths),\
            "Invalid TB parameters."

        # due to quantization, the tb_size can slightly differ from the
        # target tb_size.
        self._k_padding = self._tb_size - self._target_tb_size
        if self._tb_size != self._target_tb_size:
            print(f"Note: actual tb_size={self._tb_size} is slightly "\
                  f"different than requested " \
                  f"target_tb_size={self._target_tb_size} due to "\
                  f"quantization. Internal zero padding will be applied.")

        # calculate effective coderate (incl. CRC)
        self._coderate = self._tb_size / self._num_coded_bits

        # Remark: CRC16 is only used for k<3824 (otherwise CRC24)
        if self._tb_crc_length==16:
            self._tb_crc_encoder = CRCEncoder("CRC16")
        else:
            # CRC24A as defined in 7.2.1
            self._tb_crc_encoder = CRCEncoder("CRC24A")

        # CB CRC only if more than one CB is used
        if self._cb_crc_length==24:
            self._cb_crc_encoder = CRCEncoder("CRC24B")
        else:
            self._cb_crc_encoder = None

        # scrambler can be deactivated (non-standard compliant)
        if self._use_scrambler:
            self._scrambler = TB5GScrambler(n_rnti=self._n_rnti,
                                            n_id=self._n_id,
                                            binary=True,
                                            channel_type=channel_type,
                                            codeword_index=codeword_index,
                                            dtype=tf.float32,)
        else: # required for TBDecoder
            self._scrambler = None

        # ---- Init LDPC encoder ----
        # remark: as the codeword length can be (slightly) different
        # within a TB due to rounding, we initialize the encoder
        # with the max length and apply puncturing if required.
        # Thus, also the output interleaver cannot be applied in the encoder.
        # The procedure is defined in in 5.4.2.1 38.212
        self._encoder = LDPC5GEncoder(self._cb_size,
                                      np.max(self._cw_lengths),
                                      num_bits_per_symbol=1) #deact. interleaver

        # ---- Init interleaver ----
        # remark: explicit interleaver is required as the rate matching from
        # Sec. 5.4.2.1 38.212 could otherwise not be applied here
        perm_seq_short, _ = self._encoder.generate_out_int(
                                            np.min(self._cw_lengths),
                                            num_bits_per_symbol)
        perm_seq_long, _ = self._encoder.generate_out_int(
                                            np.max(self._cw_lengths),
                                            num_bits_per_symbol)

        perm_seq = []
        perm_seq_punc = []

        # define one big interleaver that moves the punctured positions to the
        # end of the TB
        payload_bit_pos = 0 # points to current pos of payload bits

        for l in self._cw_lengths:
            if np.min(self._cw_lengths)==l:
                perm_seq = np.concatenate([perm_seq,
                                           perm_seq_short + payload_bit_pos])
                # move unused bit positions to the end of TB
                # this simplifies the inverse permutation
                r = np.arange(payload_bit_pos+np.min(self._cw_lengths),
                              payload_bit_pos+np.max(self._cw_lengths))
                perm_seq_punc = np.concatenate([perm_seq_punc, r])

                # update pointer
                payload_bit_pos += np.max(self._cw_lengths)
            elif np.max(self._cw_lengths)==l:
                perm_seq = np.concatenate([perm_seq,
                                           perm_seq_long + payload_bit_pos])
                # update pointer
                payload_bit_pos += l
            else:
                raise ValueError("Invalid cw_lengths.")

        # add punctured positions to end of sequence (only relevant for
        # deinterleaving)
        perm_seq = np.concatenate([perm_seq, perm_seq_punc])

        self._output_perm = tf.constant(perm_seq, tf.int32)
        self._output_perm_inv = tf.argsort(perm_seq, axis=-1)

    #########################################
    # Public methods and properties
    #########################################

    @property
    def tb_size(self):
        r"""Effective number of information bits per TB.
        Note that (if required) internal zero padding can be applied to match
        the request exact ``target_tb_size``."""
        return self._tb_size

    @property
    def k(self):
        r"""Number of input information bits. Equals `tb_size` except for zero
        padding of the last positions if the ``target_tb_size`` is quantized."""
        return self._target_tb_size

    @property
    def k_padding(self):
        """Number of zero padded bits at the end of the TB."""
        return self._k_padding

    @property
    def n(self):
        "Total number of output bits."
        return self._num_coded_bits

    @property
    def num_cbs(self):
        "Number code blocks."
        return self._num_cbs

    @property
    def coderate(self):
        """Effective coderate of the TB after rate-matching including overhead
        for the CRC."""
        return self._coderate

    @property
    def ldpc_encoder(self):
        """LDPC encoder used for TB encoding."""
        return self._encoder

    @property
    def scrambler(self):
        """Scrambler used for TB scrambling. `None` if no scrambler is used."""
        return self._scrambler

    @property
    def tb_crc_encoder(self):
        """TB CRC encoder"""
        return self._tb_crc_encoder

    @property
    def cb_crc_encoder(self):
        """CB CRC encoder. `None` if no CB CRC is applied."""
        return self._cb_crc_encoder

    @property
    def num_tx(self):
        """Number of independent streams"""
        return self._num_tx

    @property
    def cw_lengths(self):
        r"""Each list element defines the codeword length of each of the
        codewords after LDPC encoding and rate-matching. The total number of
        coded bits is :math:`\sum` `cw_lengths`."""
        return self._cw_lengths

    @property
    def output_perm_inv(self):
        r"""Inverse interleaver pattern for output bit interleaver."""
        return self._output_perm_inv

    #########################
    # Keras layer functions
    #########################

    def build(self, input_shapes):
        """Test input shapes for consistency."""

        assert input_shapes[-1]==self.k, \
            f"Invalid input shape. Expected TB length is {self.k}."

    def call(self, inputs):
        """Apply transport block encoding procedure."""

        # store shapes
        input_shape = inputs.shape.as_list()
        u = tf.cast(inputs, tf.float32)

        # apply zero padding if tb_size is slightly different to target_tb_size
        if self._k_padding>0:
            s = tf.shape(u)
            s = tf.concat((s[:-1], [self._k_padding]), axis=0)
            u = tf.concat((u, tf.zeros(s, u.dtype)), axis=-1)

        # apply TB CRC
        u_crc = self._tb_crc_encoder(u)

        # CB segmentation
        u_cb = tf.reshape(u_crc,
                          (-1, self._num_tx, self._num_cbs,
                          self._cb_size-self._cb_crc_length))

        # if relevant apply CB CRC
        if self._cb_crc_length==24:
            u_cb_crc = self._cb_crc_encoder(u_cb)
        else:
            u_cb_crc = u_cb # no CRC applied if only one CB exists

        c_cb = self._encoder(u_cb_crc)

        # CB concatenation
        c = tf.reshape(c_cb,
                       (-1, self._num_tx,
                       self._num_cbs*np.max(self._cw_lengths)))

        # apply interleaver (done after CB concatenation)
        c = tf.gather(c, self._output_perm, axis=-1)
        # puncture last bits
        c = c[:, :, :np.sum(self._cw_lengths)]

        # scrambler
        if self._use_scrambler:
            c_scr = self._scrambler(c)
        else: # disable scrambler (non-standard compliant)
            c_scr = c

        # cast to output dtype
        c_scr = tf.cast(c_scr, self.dtype)

        # ensure output shapes
        output_shape = input_shape
        output_shape[0] = -1
        output_shape[-1] = np.sum(self._cw_lengths)
        c_tb = tf.reshape(c_scr, output_shape)

        return c_tb
