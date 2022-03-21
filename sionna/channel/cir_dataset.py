#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Class for creating a CIR sampler, usuable as a channel model, from a CIR
    generator"""


import tensorflow as tf

from . import ChannelModel

class CIRDataset(ChannelModel):
    # pylint: disable=line-too-long
    r"""CIRDataset(cir_generator, batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps, dtype=tf.complex64)

    Creates a channel model from a dataset that can be used with classes such as
    :class:`~sionna.channel.TimeChannel` and :class:`~sionna.channel.OFDMChannel`.
    The dataset is defined by a `generator <https://wiki.python.org/moin/Generators>`_.

    The batch size is configured when instantiating the dataset or through the :attr:`~sionna.channel.CIRDataset.batch_size` property.
    The number of time steps (`num_time_steps`) and sampling frequency (`sampling_frequency`) can only be set when instantiating the dataset.
    The specified values must be in accordance with the data.

    Example
    --------

    The following code snippet shows how to use this class as a channel model.

    >>> my_generator = MyGenerator(...)
    >>> channel_model = sionna.channel.CIRDataset(my_generator,
    ...                                           batch_size,
    ...                                           num_rx,
    ...                                           num_rx_ant,
    ...                                           num_tx,
    ...                                           num_tx_ant,
    ...                                           num_paths,
    ...                                           num_time_steps+l_tot-1)
    >>> channel = sionna.channel.TimeChannel(channel_model, bandwidth, num_time_steps)

    where ``MyGenerator`` is a generator

    >>> class MyGenerator:
    ...
    ...     def __call__(self):
    ...         ...
    ...         yield a, tau

    that returns complex-valued path coefficients ``a`` with shape `[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]`
    and real-valued path delays ``tau`` (in second) `[batch size, num_rx, num_tx, num_paths]`.

    Parameters
    ----------
    cir_generator : `generator <https://wiki.python.org/moin/Generators>`_
        Generator that returns channel impulse responses ``(a, tau)`` where ``a`` is the tensor of channel coefficients of shape
        `[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]` and dtype ``dtype``, and ``tau`` the tensor of path delays
        of shape  `[batch size, num_rx, num_tx, num_paths]` and dtype ``dtype.real_dtype``.

    batch_size : int
        Batch size

    num_rx : int
        Number of receivers (:math:`N_R`)

    num_rx_ant : int
        Number of antennas per receiver (:math:`N_{RA}`)

    num_tx : int
        Number of transmitters (:math:`N_T`)

    num_tx_ant : int
        Number of antennas per transmitter (:math:`N_{TA}`)

    num_paths : int
        Number of paths (:math:`M`)

    num_time_steps : int
        Number of time steps

    dtype : tf.DType
        Complex datatype to use for internal processing and output.
        Defaults to `tf.complex64`.

    Output
    -------
    a : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps], tf.complex
        Path coefficients

    tau : [batch size, num_rx, num_tx, num_paths], tf.float
        Path delays [s]
    """

    def __init__(self, cir_generator, batch_size, num_rx, num_rx_ant, num_tx,
        num_tx_ant, num_paths, num_time_steps, dtype=tf.complex64):

        self._cir_generator = cir_generator
        self._batch_size = batch_size
        self._num_time_steps = num_time_steps

        # TensorFlow dataset
        output_signature = (tf.TensorSpec(shape=[num_rx,
                                                 num_rx_ant,
                                                 num_tx,
                                                 num_tx_ant,
                                                 num_paths,
                                                 num_time_steps],
                                          dtype=dtype),
                            tf.TensorSpec(shape=[num_rx,
                                                 num_tx,
                                                 num_paths],
                                          dtype=dtype.real_dtype))
        dataset = tf.data.Dataset.from_generator(cir_generator,
                                            output_signature=output_signature)
        dataset = dataset.shuffle(32, reshuffle_each_iteration=True)
        self._dataset = dataset.repeat(None)
        self._batched_dataset = self._dataset.batch(batch_size)
        # Iterator for sampling the dataset
        self._iter = iter(self._batched_dataset)

    @property
    def batch_size(self):
        """Batch size"""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        """Set the batch size"""
        self._batched_dataset = self._dataset.batch(value)
        self._iter = iter(self._batched_dataset)
        self._batch_size = value

    def __call__(self, batch_size=None,
                       num_time_steps=None,
                       sampling_frequency=None):

#         if ( (batch_size is not None)
#                 and tf.not_equal(batch_size, self._batch_size) ):
#             tf.print("Warning: The value of `batch_size` specified when calling \
# the CIRDataset is different from the one configured for the dataset. \
# The value specified when calling is ignored. Use the `batch_size` property \
# of CIRDataset to use a batch size different from the one set when \
# instantiating.")

#         if ( (num_time_steps is not None)
#             and tf.not_equal(num_time_steps, self._num_time_steps) ):
#             tf.print("Warning: The value of `num_time_steps` specified when \
# calling the CIRDataset is different from the one speficied when instantiating \
# the dataset. The value specified when calling is ignored.")

        return next(self._iter)
