#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""PUSCH Receiver for the nr (5G) sub-package of the Sionna library.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
import sionna
from sionna.mimo import StreamManagement
from sionna.ofdm import OFDMDemodulator, LinearDetector
from sionna.utils import insert_dims
from sionna.channel import time_to_ofdm_channel

class PUSCHReceiver(Layer):
    # pylint: disable=line-too-long
    r"""PUSCHReceiver(pusch_transmitter, channel_estimator=None, mimo_detector=None, tb_decoder=None, return_tb_crc_status=False, stream_management=None, input_domain="freq", l_min=None, dtype=tf.complex64, **kwargs)

    This layer implements a full receiver for batches of 5G NR PUSCH slots sent
    by multiple transmitters. Inputs can be in the time or frequency domain.
    Perfect channel state information can be optionally provided.
    Different channel estimatiors, MIMO detectors, and transport decoders
    can be configured.

    The layer combines multiple processing blocks into a single layer
    as shown in the following figure. Blocks with dashed lines are
    optional and depend on the configuration.

    .. figure:: ../figures/pusch_receiver_block_diagram.png
        :scale: 30%
        :align: center

    If the ``input_domain`` equals "time", the inputs :math:`\mathbf{y}` are first
    transformed to resource grids with the :class:`~sionna.ofdm.OFDMDemodulator`.
    Then channel estimation is performed, e.g., with the help of the
    :class:`~sionna.nr.PUSCHLSChannelEstimator`. If ``channel_estimator``
    is chosen to be "perfect", this step is skipped and the input :math:`\mathbf{h}`
    is used instead.
    Next, MIMO detection is carried out with an arbitrary :class:`~sionna.ofdm.OFDMDetector`.
    The resulting LLRs for each layer are then combined to transport blocks
    with the help of the :class:`~sionna.nr.LayerDemapper`.
    Finally, the transport blocks are decoded with the :class:`~sionna.nr.TBDecoder`.

    Parameters
    ----------
    pusch_transmitter : :class:`~sionna.nr.PUSCHTransmitter`
        Transmitter used for the generation of the transmit signals

    channel_estimator : :class:`~sionna.ofdm.BaseChannelEstimator`, "perfect", or `None`
        Channel estimator to be used.
        If `None`, the :class:`~sionna.nr.PUSCHLSChannelEstimator` with
        linear interpolation is used.
        If "perfect", no channel estimation is performed and the channel state information
        ``h`` must be provided as additional input.
        Defaults to `None`.

    mimo_detector : :class:`~sionna.ofdm.OFDMDetector` or `None`
        MIMO Detector to be used.
        If `None`, the :class:`~sionna.ofdm.LinearDetector` with
        LMMSE detection is used.
        Defaults to `None`.

    tb_decoder : :class:`~sionna.nr.TBDecoder` or `None`
        Transport block decoder to be used.
        If `None`, the :class:`~sionna.nr.TBDecoder` with its
        default settings is used.
        Defaults to `None`.

    return_tb_crc_status : bool
        If `True`, the status of the transport block CRC is returned
        as additional output.
        Defaults to `False`.

    stream_management : :class:`~sionna.mimo.StreamManagement` or `None`
        Stream management configuration to be used.
        If `None`, it is assumed that there is a single receiver
        which decodes all streams of all transmitters.
        Defaults to `None`.

    input_domain : str, one of ["freq", "time"]
        Domain of the input signal.
        Defaults to "freq".

    l_min : int or `None`
        Smallest time-lag for the discrete complex baseband channel.
        Only needed if ``input_domain`` equals "time".
        Defaults to `None`.

    dtype : tf.Dtype
        Datatype for internal calculations and the output dtype.
        Defaults to `tf.complex64`.

    Input
    -----
    (y, h, no) :
        Tuple:

    y : [batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex or [batch size, num_rx, num_rx_ant, num_time_samples + l_max - l_min], tf.complex
        Frequency- or time-domain input signal

    h : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, num_subcarriers], tf.complex or [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_samples + l_max - l_min, l_max - l_min + 1], tf.complex
        Perfect channel state information in either frequency or time domain
        (depending on ``input_domain``) to be used for detection.
        Only required if ``channel_estimator`` equals "perfect".

    no : [batch_size, num_rx, num_rx_ant] or only the first n>=0 dims, tf.float
        Variance of the AWGN

    Output
    ------
    b_hat : [batch_size, num_tx, tb_size], tf.float
        Decoded information bits

    tb_crc_status : [batch_size, num_tx], tf.bool
        Transport block CRC status

    Example
    -------
    >>> pusch_config = PUSCHConfig()
    >>> pusch_transmitter = PUSCHTransmitter(pusch_config)
    >>> pusch_receiver = PUSCHReceiver(pusch_transmitter)
    >>> channel = AWGN()
    >>> x, b = pusch_transmitter(16)
    >>> no = 0.1
    >>> y = channel([x, no])
    >>> b_hat = pusch_receiver([x, no])
    >>> compute_ber(b, b_hat)
    <tf.Tensor: shape=(), dtype=float64, numpy=0.0>
    """
    def __init__(self,
                 pusch_transmitter,
                 channel_estimator=None,
                 mimo_detector=None,
                 tb_decoder=None,
                 return_tb_crc_status=False,
                 stream_management=None,
                 input_domain="freq",
                 l_min=None,
                 dtype=tf.complex64,
                 **kwargs):
        assert dtype in [tf.complex64, tf.complex128], \
            "dtype must be tf.complex64 or tf.complex128"
        super().__init__(dtype=dtype, **kwargs)

        assert input_domain in ["time", "freq"], \
            "input_domain must be 'time' or 'freq'"
        self._input_domain = input_domain

        self._return_tb_crc_status = return_tb_crc_status

        self._resource_grid = pusch_transmitter.resource_grid

        # (Optionally) Create OFDMDemodulator
        if self._input_domain=="time":
            assert l_min is not None, \
                "l_min must be provided for input_domain==time"
            self._l_min = l_min
            self._ofdm_demodulator = OFDMDemodulator(
                fft_size=pusch_transmitter._num_subcarriers,
                l_min=self._l_min,
                cyclic_prefix_length=pusch_transmitter._cyclic_prefix_length)

        # Use or create default ChannelEstimator
        self._perfect_csi = False
        self._w = None
        if channel_estimator is None:
            # Default channel estimator
            self._channel_estimator = sionna.nr.PUSCHLSChannelEstimator(
                                self.resource_grid,
                                pusch_transmitter._dmrs_length,
                                pusch_transmitter._dmrs_additional_position,
                                pusch_transmitter._num_cdm_groups_without_data,
                                interpolation_type='lin',
                                dtype=dtype)
        elif channel_estimator=="perfect":
            # Perfect channel estimation
            self._perfect_csi = True
            if pusch_transmitter._precoding=="codebook":
                self._w = pusch_transmitter._precoder._w
                self._w = insert_dims(self._w, 2, 1)
        else:
            # User-provided channel estimator
            self._channel_estimator = channel_estimator

        # Use or create default StreamManagement
        if stream_management is None:
            # Default StreamManagement
            rx_tx_association = np.ones([1, pusch_transmitter._num_tx], bool)
            self._stream_management = StreamManagement(
                                        rx_tx_association,
                                        pusch_transmitter._num_layers)
        else:
            # User-provided StramManagement
            self._stream_management = stream_management

        # Use or create default MIMODetector
        if mimo_detector is None:
            # Default MIMO detector
            self._mimo_detector = LinearDetector("lmmse", "bit", "maxlog",
                                        pusch_transmitter.resource_grid,
                                        self._stream_management,
                                        "qam",
                                        pusch_transmitter._num_bits_per_symbol,
                                        dtype=dtype)
        else:
            # User-provided MIMO detector
            self._mimo_detector = mimo_detector

        # Create LayerDemapper
        self._layer_demapper = sionna.nr.LayerDemapper(
                    pusch_transmitter._layer_mapper,
                    num_bits_per_symbol=pusch_transmitter._num_bits_per_symbol)

        # Use or create default TBDecoder
        if tb_decoder is None:
            # Default TBEncoder
            self._tb_decoder = sionna.nr.TBDecoder(
                                    pusch_transmitter._tb_encoder,
                                    output_dtype=dtype.real_dtype)
        else:
            # User-provided TBEncoder
            self._tb_decoder = tb_decoder

    #########################################
    # Public methods and properties
    #########################################

    @property
    def resource_grid(self):
        """OFDM resource grid underlying the PUSCH transmissions"""
        return self._resource_grid

    def call(self, inputs):
        if self._perfect_csi:
            y, h, no = inputs
        else:
            y, no = inputs

        # (Optional) OFDM Demodulation
        if self._input_domain=="time":
            y = self._ofdm_demodulator(y)

        # Channel estimation
        if self._perfect_csi:

            # Transform time-domain to frequency-domain channel
            if self._input_domain=="time":
                h = time_to_ofdm_channel(h, self.resource_grid, self._l_min)


            if self._w is not None:
                # Reshape h to put channel matrix dimensions last
                # [batch size, num_rx, num_tx, num_ofdm_symbols,...
                #  ...fft_size, num_rx_ant, num_tx_ant]
                h = tf.transpose(h, perm=[0,1,3,5,6,2,4])

                # Multiply by precoding matrices to compute effective channels
                # [batch size, num_rx, num_tx, num_ofdm_symbols,...
                #  ...fft_size, num_rx_ant, num_streams]
                h = tf.matmul(h, self._w)

                # Reshape
                # [batch size, num_rx, num_rx_ant, num_tx, num_streams,...
                #  ...num_ofdm_symbols, fft_size]
                h = tf.transpose(h, perm=[0,1,5,2,6,3,4])
            h_hat = h
            err_var = tf.cast(0, dtype=h_hat.dtype.real_dtype)
        else:
            h_hat,err_var = self._channel_estimator([y, no])

        # MIMO Detection
        llr = self._mimo_detector([y, h_hat, err_var, no])

        # Layer demapping
        llr = self._layer_demapper(llr)

        # TB Decoding
        b_hat, tb_crc_status = self._tb_decoder(llr)

        if self._return_tb_crc_status:
            return b_hat, tb_crc_status
        else:
            return b_hat
