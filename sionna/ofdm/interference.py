#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for signal interference generation"""

import sionna
import tensorflow as tf

class OFDMInterferenceSource(tf.keras.layers.Layer):
    # pylint: disable=line-too-long
    r"""OFDMInterferenceSource(density_subcarriers=1.0, sampler="uniform", domain="freq", cyclic_prefix_length=0, dtype=tf.complex64)

    Layer to simulate interference transmitter symbols for OFDM systems in frequency or time domain.
    These can then be sent through a OFDM channel to simulate interference at the receiver.

    The transmit symbols can be sampled form different distributions or constellations, to be configured through the parameter `sampler`.
    
    When `domain` is set to "freq", the interference generated is meant to be used in frequency domain simulations.
    The simulation thus implicitly assumes that the interfering devices are synchronized with the receiver, and send their signal with a cyclic prefix of sufficient length.
    
    When `domain` is set to "time", the interference is generated in time domain. A cyclic prefix may be added to the interference symbols, which can be controlled through the parameter `cyclic_prefix_length`.

    This class supports simulation of narrowband interference through the parameter `density_subcarriers`, which controls the fraction of subcarriers on which the interference takes place.
    The subcarriers on which the interference takes place are randomly selected for each call to the layer.

    Parameters
    ----------
    density_subcarriers: float
        Fraction of subcarriers which are effected by interference. Must be in between 0 and 1.
        The number of subcarriers effected is rounded to the next integer, and the subcarriers are randomly selected for each call to the layer.
    sampler: str, instance of :class:`~sionna.mapping.Constellation`, or callable
        If str, one of ["uniform", "gaussian"].
        If instance of :class:`~sionna.mapping.Constellation`, the constellation is sampled randomly.
        If callable, the callable is used as sampling function. It should have signature ``(shape, var, dtype) -> tf.Tensor``.
        The sampled symbols will always have an expected mean energy of 1.
    domain: str
        Domain in which the interference is generated. One of ["time", "freq"].
    cyclic_prefix_length: int
        Length of the cyclic prefix. Only relevant if `domain` is "time".
    fft_size: int, optional (ignored if `domain` is "freq")
        FFT size. Ignored for frequency domain, as this is separately provided by shape.
        This parameter is relevant in time domain, as a cyclic prefix might be added in, and sparsity over subcarriers mandates the FFT size.
    dtype: tf.complex
        Data type of the interference symbols. Defaults to tf.complex64.

    Input
    -----
    shape: 1D tensor/array/list, int
        List of integers, specifying the shape of the interference symbols to be generated.
        Should consist of `(batch_size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size)` in frequency domain, 
        and `(batch_size, num_tx, num_tx_ant, num_time_samples)` in time domain.
        

    Output
    ------
    x_itf: ``output_shape``, ``dtype``
        Interference symbols in time or frequency domain, depending on the parameter `domain`. ```output_shape``` is ```shape``` if in frequency domain, and `(batch_size, num_tx, num_tx_ant, num_ofdm_symbols * (fft_size + cyclic_prefix_length))` if in time domain.

    """
    def __init__(self,
                 density_subcarriers=1.0,
                 sampler="uniform",
                 domain="freq",
                 cyclic_prefix_length=0,
                 fft_size=None,
                 dtype=tf.complex64,
                 **kwargs):

        super().__init__(trainable=False, dtype=dtype, **kwargs)
        self._density_subcarriers = density_subcarriers
        self._domain = domain
        self._cyclic_prefix_length = cyclic_prefix_length
        self._fft_size = fft_size
        self._dtype_as_dtype = tf.as_dtype(self.dtype)
        # if sampler is string, we use the corresponding function. Otherwise assign the function directly
        self._sample_function = self._sampler_to_callable(sampler, self._dtype_as_dtype)
        if self._domain == "time":
            self._modulator = sionna.ofdm.OFDMModulator(cyclic_prefix_length)
        self._check_settings()

    def _check_settings(self):
        assert self._density_subcarriers >= 0.0 and self._density_subcarriers <= 1.0, "density_subcarriers must be in [0, 1]"
        assert self._domain in ["time", "freq"]
        if self._domain == "time":
            assert self._cyclic_prefix_length >= 0, "cyclic_prefix_length must be non-negative"
            assert self._fft_size is not None, "fft_size must be provided in time domain"
        assert self._dtype_as_dtype.is_complex

    def call(self, inputs):
        if self._domain == "freq":
            self._fft_size = inputs[-1]
            num_ofdm_symbols = inputs[-2]
        else:
            num_ofdm_symbols = tf.math.ceil(tf.cast(inputs[-1], tf.float32) / (self._fft_size + self._cyclic_prefix_length))
        x_itf = self._sample_function(tf.concat([inputs[:3], [num_ofdm_symbols, self._fft_size]], axis=0))
        x_itf = self._make_sparse(x_itf)
        if self._domain == "time":
            x_itf = self._modulator(x_itf)
            x_itf = x_itf[..., :inputs[-1]]
        return x_itf

    def _make_sparse(self, data):
        shape = tf.shape(data)
        num_subcarriers = shape[-1]
        num_nonzero_subcarriers = tf.cast(tf.round(self._density_subcarriers * tf.cast(num_subcarriers, tf.float32)), tf.int32)

        # create sparse masks
        subcarrier_mask = tf.concat([tf.ones([num_nonzero_subcarriers], dtype=self._dtype),
                                     tf.zeros([num_subcarriers - num_nonzero_subcarriers], dtype=self._dtype)], axis=0)
        subcarrier_mask = tf.random.shuffle(subcarrier_mask)

        return data * subcarrier_mask

    def _sampler_to_callable(self, sampler, dtype):
        # pylint: disable=line-too-long
        r"""
        Returns callable which samples from a constellation or a distribution.

        Input
        -----
            sampler : str | Constellation | callable
                String in `["uniform", "gaussian"]`, an instance of :class:`~sionna.mapping.Constellation`, or function with signature ``(shape, dtype) -> tf.Tensor``,
                where elementwise :math:`E[|x|^2] = 1`.
            dtype : tf.Dtype
                Defines the datatype the returned function should return.

        Output
        ------
            callable
                Function with signature ``shape -> tf.Tensor`` which returns a tensor of shape ``shape`` with dtype ``dtype``.
        """
        if isinstance(sampler, sionna.mapping.Constellation):
            assert sampler.dtype == dtype
            sampler.normalize = True
            ret = sionna.utils.SymbolSource('custom', constellation=sampler, dtype=dtype)
        else:
            if isinstance(sampler, str):
                if sampler == "uniform":
                    f = sionna.utils.complex_uniform_disk
                elif sampler == "gaussian":
                    f = sionna.utils.complex_normal
                else:
                    raise ValueError(f"Unknown sampler {sampler}")
            elif callable(sampler):
                f = sampler
            else:
                raise ValueError(f"Unknown sampler {sampler}")
            # pylint: disable=unnecessary-lambda-assignment
            ret = lambda s: f(shape=s, var=1.0, dtype=dtype)
        return ret
