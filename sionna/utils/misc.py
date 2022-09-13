#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Miscellaneous utility functions of the Sionna package."""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.experimental.numpy import log10 as _log10
from tensorflow.experimental.numpy import log2 as _log2
from sionna.utils.metrics import count_errors, count_block_errors
from sionna.mapping import Mapper, Constellation
import time
from sionna import signal


def ebnodb2no(ebno_db, num_bits_per_symbol, coderate, resource_grid=None):
    r"""Compute the noise variance `No` for a given `Eb/No` in dB.

    The function takes into account the number of coded bits per constellation
    symbol, the coderate, as well as possible additional overheads related to
    OFDM transmissions, such as the cyclic prefix and pilots.

    The value of `No` is computed according to the following expression

    .. math::
        N_o = \left(\frac{E_b}{N_o} \frac{r M}{E_s}\right)^{-1}

    where :math:`2^M` is the constellation size, i.e., :math:`M` is the
    average number of coded bits per constellation symbol,
    :math:`E_s=1` is the average energy per constellation per symbol,
    :math:`r\in(0,1]` is the coderate,
    :math:`E_b` is the energy per information bit,
    and :math:`N_o` is the noise power spectral density.
    For OFDM transmissions, :math:`E_s` is scaled
    according to the ratio between the total number of resource elements in
    a resource grid with non-zero energy and the number
    of resource elements used for data transmission. Also the additionally
    transmitted energy during the cyclic prefix is taken into account, as
    well as the number of transmitted streams per transmitter.

    Input
    -----
    ebno_db : float
        The `Eb/No` value in dB.

    num_bits_per_symbol : int
        The number of bits per symbol.

    coderate : float
        The coderate used.

    resource_grid : ResourceGrid
        An (optional) instance of :class:`~sionna.ofdm.ResourceGrid`
        for OFDM transmissions.

    Output
    ------
    : float
        The value of :math:`N_o` in linear scale.
    """

    if tf.is_tensor(ebno_db):
        dtype = ebno_db.dtype
    else:
        dtype = tf.float32

    ebno = tf.math.pow(tf.cast(10., dtype), ebno_db/10.)

    energy_per_symbol = 1
    if resource_grid is not None:
        # Divide energy per symbol by the number of transmitted streams
        energy_per_symbol /= resource_grid.num_streams_per_tx

        # Number of nonzero energy symbols.
        # We do not account for the nulled DC and guard carriers.
        cp_overhead = resource_grid.cyclic_prefix_length \
                      / resource_grid.fft_size
        num_syms = resource_grid.num_ofdm_symbols * (1 + cp_overhead) \
                    * resource_grid.num_effective_subcarriers
        energy_per_symbol *= num_syms / resource_grid.num_data_symbols

    no = 1/(ebno * coderate * tf.cast(num_bits_per_symbol, dtype) \
          / tf.cast(energy_per_symbol, dtype))

    return no


def hard_decisions(llr):
    """Transforms LLRs into hard decisions.

    Positive values are mapped to :math:`1`.
    Nonpositive values are mapped to :math:`0`.

    Input
    -----
    llr : any non-complex tf.DType
        Tensor of LLRs.

    Output
    ------
    : Same shape and dtype as ``llr``
        The hard decisions.
    """
    zero = tf.constant(0, dtype=llr.dtype)

    return tf.cast(tf.math.greater(llr, zero), dtype=llr.dtype)


def log10(x):
    # pylint: disable=C0301
    """TensorFlow implementation of NumPy's `log10` function.

    Simple extension to `tf.experimental.numpy.log10`
    which casts the result to the `dtype` of the input.
    For more details see the `TensorFlow <https://www.tensorflow.org/api_docs/python/tf/experimental/numpy/log10>`_ and `NumPy <https://numpy.org/doc/1.16/reference/generated/numpy.log10.html>`_ documentation.
    """
    return tf.cast(_log10(x), x.dtype)


def log2(x):
    # pylint: disable=C0301
    """TensorFlow implementation of NumPy's `log2` function.

    Simple extension to `tf.experimental.numpy.log2`
    which casts the result to the `dtype` of the input.
    For more details see the `TensorFlow <https://www.tensorflow.org/api_docs/python/tf/experimental/numpy/log2>`_ and `NumPy <https://numpy.org/doc/1.16/reference/generated/numpy.log2.html>`_ documentation.
    """
    return tf.cast(_log2(x), x.dtype)


class BinarySource(Layer):
    """BinarySource(dtype=tf.float32, seed=None, **kwargs)

    Layer generating random binary tensors.

    Parameters
    ----------
    dtype : tf.DType
        Defines the output datatype of the layer.
        Defaults to `tf.float32`.

    seed : int or None
        Set the seed for the random generator used to generate the bits.
        Set to `None` for random initialization of the RNG.

    Input
    -----
    shape : 1D tensor/array/list, int
        The desired shape of the output tensor.

    Output
    ------
    : ``shape``, ``dtype``
        Tensor filled with random binary values.
    """
    def __init__(self, dtype=tf.float32, seed=None, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        self._seed = seed
        if self._seed is not None:
            self._rng = tf.random.Generator.from_seed(self._seed)

    def call(self, inputs):
        if self._seed is not None:
            return tf.cast(self._rng.uniform(inputs, 0, 2, tf.int32),
                           dtype=super().dtype)
        else:
            return tf.cast(tf.random.uniform(inputs, 0, 2, tf.int32),
                           dtype=super().dtype)

class SymbolSource(Layer):
    # pylint: disable=line-too-long
    r"""SymbolSource(constellation_type=None, num_bits_per_symbol=None, constellation=None, return_indices=False, return_bits=False, seed=None, dtype=tf.complex64, **kwargs)

    Layer generating a tensor of arbitrary shape filled with random constellation symbols.
    Optionally, the symbol indices and/or binary representations of the
    constellation symbols can be returned.

    Parameters
    ----------
    constellation_type : One of ["qam", "pam", "custom"], str
        For "custom", an instance of :class:`~sionna.mapping.Constellation`
        must be provided.

    num_bits_per_symbol : int
        The number of bits per constellation symbol.
        Only required for ``constellation_type`` in ["qam", "pam"].

    constellation :  Constellation
        An instance of :class:`~sionna.mapping.Constellation` or
        `None`. In the latter case, ``constellation_type``
        and ``num_bits_per_symbol`` must be provided.

    return_indices : bool
        If enabled, the function also returns the symbol indices.
        Defaults to `False`.

    return_bits : bool
        If enabled, the function also returns the binary symbol
        representations (i.e., bit labels).
        Defaults to `False`.

    seed : int or None
        The seed for the random generator.
        `None` leads to a random initialization of the RNG.
        Defaults to `None`.

    dtype : One of [tf.complex64, tf.complex128], tf.DType
        The output dtype. Defaults to tf.complex64.

    Input
    -----
    shape : 1D tensor/array/list, int
        The desired shape of the output tensor.

    Output
    ------
    symbols : ``shape``, ``dtype``
        Tensor filled with random symbols of the chosen ``constellation_type``.

    symbol_indices : ``shape``, tf.int32
        Tensor filled with the symbol indices.
        Only returned if ``return_indices`` is `True`.

    bits : [``shape``, ``num_bits_per_symbol``], tf.int32
        Tensor filled with the binary symbol representations (i.e., bit labels).
        Only returned if ``return_bits`` is `True`.
    """
    def __init__(self,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 return_indices=False,
                 return_bits=False,
                 seed=None,
                 dtype=tf.complex64,
                 **kwargs
                ):
        super().__init__(dtype=dtype, **kwargs)
        constellation = Constellation.create_or_check_constellation(
            constellation_type,
            num_bits_per_symbol,
            constellation,
            dtype)
        self._num_bits_per_symbol = constellation.num_bits_per_symbol
        self._return_indices = return_indices
        self._return_bits = return_bits
        self._binary_source = BinarySource(seed=seed, dtype=dtype.real_dtype)
        self._mapper = Mapper(constellation=constellation,
                              return_indices=return_indices,
                              dtype=dtype)

    def call(self, inputs):
        shape = tf.concat([inputs, [self._num_bits_per_symbol]], axis=-1)
        b = self._binary_source(tf.cast(shape, tf.int32))
        if self._return_indices:
            x, ind = self._mapper(b)
        else:
            x = self._mapper(b)

        result = tf.squeeze(x, -1)
        if self._return_indices or self._return_bits:
            result = [result]
        if self._return_indices:
            result.append(tf.squeeze(ind, -1))
        if self._return_bits:
            result.append(b)

        return result


class QAMSource(SymbolSource):
    # pylint: disable=line-too-long
    r"""QAMSource(num_bits_per_symbol=None, return_indices=False, return_bits=False, seed=None, dtype=tf.complex64, **kwargs)

    Layer generating a tensor of arbitrary shape filled with random QAM symbols.
    Optionally, the symbol indices and/or binary representations of the
    constellation symbols can be returned.

    Parameters
    ----------
    num_bits_per_symbol : int
        The number of bits per constellation symbol, e.g., 4 for QAM16.

    return_indices : bool
        If enabled, the function also returns the symbol indices.
        Defaults to `False`.

    return_bits : bool
        If enabled, the function also returns the binary symbol
        representations (i.e., bit labels).
        Defaults to `False`.

    seed : int or None
        The seed for the random generator.
        `None` leads to a random initialization of the RNG.
        Defaults to `None`.

    dtype : One of [tf.complex64, tf.complex128], tf.DType
        The output dtype. Defaults to tf.complex64.

    Input
    -----
    shape : 1D tensor/array/list, int
        The desired shape of the output tensor.

    Output
    ------
    symbols : ``shape``, ``dtype``
        Tensor filled with random QAM symbols.

    symbol_indices : ``shape``, tf.int32
        Tensor filled with the symbol indices.
        Only returned if ``return_indices`` is `True`.

    bits : [``shape``, ``num_bits_per_symbol``], tf.int32
        Tensor filled with the binary symbol representations (i.e., bit labels).
        Only returned if ``return_bits`` is `True`.
    """
    def __init__(self,
                 num_bits_per_symbol=None,
                 return_indices=False,
                 return_bits=False,
                 seed=None,
                 dtype=tf.complex64,
                 **kwargs
                ):
        super().__init__(constellation_type="qam",
                         num_bits_per_symbol=num_bits_per_symbol,
                         return_indices=return_indices,
                         return_bits=return_bits,
                         seed=seed,
                         dtype=dtype,
                         **kwargs)

class PAMSource(SymbolSource):
    # pylint: disable=line-too-long
    r"""PAMSource(num_bits_per_symbol=None, return_indices=False, return_bits=False, seed=None, dtype=tf.complex64, **kwargs)

    Layer generating a tensor of arbitrary shape filled with random PAM symbols.
    Optionally, the symbol indices and/or binary representations of the
    constellation symbols can be returned.

    Parameters
    ----------
    num_bits_per_symbol : int
        The number of bits per constellation symbol, e.g., 1 for BPSK.

    return_indices : bool
        If enabled, the function also returns the symbol indices.
        Defaults to `False`.

    return_bits : bool
        If enabled, the function also returns the binary symbol
        representations (i.e., bit labels).
        Defaults to `False`.

    seed : int or None
        The seed for the random generator.
        `None` leads to a random initialization of the RNG.
        Defaults to `None`.

    dtype : One of [tf.complex64, tf.complex128], tf.DType
        The output dtype. Defaults to tf.complex64.

    Input
    -----
    shape : 1D tensor/array/list, int
        The desired shape of the output tensor.

    Output
    ------
    symbols : ``shape``, ``dtype``
        Tensor filled with random PAM symbols.

    symbol_indices : ``shape``, tf.int32
        Tensor filled with the symbol indices.
        Only returned if ``return_indices`` is `True`.

    bits : [``shape``, ``num_bits_per_symbol``], tf.int32
        Tensor filled with the binary symbol representations (i.e., bit labels).
        Only returned if ``return_bits`` is `True`.
    """
    def __init__(self,
                 num_bits_per_symbol=None,
                 return_indices=False,
                 return_bits=False,
                 seed=None,
                 dtype=tf.complex64,
                 **kwargs
                ):
        super().__init__(constellation_type="pam",
                         num_bits_per_symbol=num_bits_per_symbol,
                         return_indices=return_indices,
                         return_bits=return_bits,
                         seed=seed,
                         dtype=dtype,
                         **kwargs)


def sim_ber(mc_fun,
            ebno_dbs,
            batch_size,
            max_mc_iter,
            soft_estimates=False,
            num_target_bit_errors=None,
            num_target_block_errors=None,
            early_stop=True,
            verbose=True,
            forward_keyboard_interrupt=True,
            dtype=tf.complex64):
    """Simulates until target number of errors is reached and returns BER/BLER.

    The simulation continues with the next SNR point if either
    ``num_target_bit_errors`` bit errors or ``num_target_block_errors`` block
    errors is achieved. Further, it continues with the next SNR point after
    ``max_mc_iter`` batches of size ``batch_size`` have been simulated.

    Input
    -----
    mc_fun:
        Function that yields the transmitted bits `b` and the
        receiver's estimate `b_hat` for a given ``batch_size`` and
        ``ebno_db``. If ``soft_estimates`` is True, b_hat is interpreted as
        logit.

    ebno_dbs: tf.float32
        A tensor containing SNR points to be evaluated.

    batch_size: tf.int32
        Batch-size for evaluation.

    max_mc_iter: tf.int32
        Max. number of Monte-Carlo iterations per SNR point.

    soft_estimates: bool
        A boolean, defaults to False. If True, `b_hat``
        is interpreted as logit and an additional hard-decision is applied
        internally.

    num_target_bit_errors: tf.int32
        Defaults to None. Target number of bit errors per SNR point until
        the simulation continues to next SNR point.

    num_target_block_errors: tf.int32
        Defaults to None. Target number of block errors per SNR point
        until the simulation continues

    early_stop: bool
        A boolean defaults to True. If True, the simulation stops after the
        first error-free SNR point (i.e., no error occurred after
        ``max_mc_iter`` Monte-Carlo iterations).

    verbose: bool
        A boolean defaults to True. If True, the current progress will be
        printed.

    forward_keyboard_interrupt: bool
        A boolean defaults to True. If False, KeyboardInterrupts will be
        catched internally and not forwarded (e.g., will not stop outer loops).
        If False, the simulation ends and returns the intermediate simulation
        results.

    dtype: tf.complex64
        Datatype of the model / function to be used (``mc_fun``).

    Output
    ------
    (ber, bler) :
        Tuple:

    ber: tf.float32
        The bit-error rate.

    bler: tf.float32
        The block-error rate.

    Raises
    ------
    AssertionError
        If ``soft_estimates`` is not bool.

    AssertionError
        If ``dtype`` is not `tf.complex`.

    Note
    ----
    This function is implemented based on tensors to allow
    full compatibility with tf.function(). However, to run simulations
    in graph mode, the provided ``mc_fun`` must use the `@tf.function()`
    decorator.

    """

    # utility function to print progress
    def _print_progress(is_final, rt, idx_snr, idx_it, header_text=None):
        """Print summary of current simulation progress.

        Input
        -----
        is_final: bool
            A boolean. If True, the progress is printed into a new line.
        rt: float
            The runtime of the current SNR point in seconds.
        idx_snr: int
            Index of current SNR point.
        idx_it: int
            Current iteration index.
        header_text: list of str
            Elements will be printed instead of current progress, iff not None.
            Can be used to generate table header.
        """
        # set carriage return if not final step
        if is_final:
            end_str = "\n"
        else:
            end_str = "\r"

        # prepare to print table header
        if header_text is not None:
            row_text = header_text
            end_str = "\n"
        else:
            # calculate intermediate ber / bler
            ber_np = (tf.cast(bit_errors[idx_snr], tf.float64)
                        / tf.cast(nb_bits[idx_snr], tf.float64)).numpy()
            ber_np = np.nan_to_num(ber_np) # avoid nan for first point
            bler_np = (tf.cast(block_errors[idx_snr], tf.float64)
                        / tf.cast(nb_blocks[idx_snr], tf.float64)).numpy()
            bler_np = np.nan_to_num(bler_np) # avoid nan for first point

            # load statuslevel
            # print current iter if simulation is still running
            if status[idx_snr]==0:
                status_txt = f"iter: {idx_it:.0f}/{max_mc_iter:.0f}"
            else:
                status_txt = status_levels[int(status[idx_snr])]

            # generate list with all elements to be printed
            row_text = [str(np.round(ebno_dbs[idx_snr].numpy(), 3)),
                        f"{ber_np:.4e}",
                        f"{bler_np:.4e}",
                        np.round(bit_errors[idx_snr].numpy(), 0),
                        np.round(nb_bits[idx_snr].numpy(), 0),
                        np.round(block_errors[idx_snr].numpy(), 0),
                        np.round(nb_blocks[idx_snr].numpy(), 0),
                        np.round(rt, 1),
                        status_txt]

        # pylint: disable=line-too-long, consider-using-f-string
        print("{: >9} |{: >11} |{: >11} |{: >12} |{: >12} |{: >13} |{: >12} |{: >12} |{: >10}".format(*row_text), end=end_str)


     # init table headers
    header_text = ["EbNo [dB]", "BER", "BLER", "bit errors",
                   "num bits", "block errors", "num blocks",
                   "runtime [s]", "status"]

    # replace status by text
    status_levels = ["not simulated", # status=0
            "reached max iter       ", # status=1; spacing for impr. layout
            "no errors - early stop", # status=2
            "reached target bit errors", # status=3
            "reached target block errors"] # status=4

    # check inputs for consistency
    assert isinstance(early_stop, bool), "early_stop must be bool."
    assert isinstance(soft_estimates, bool), "soft_estimates must be bool."
    assert dtype.is_complex, "dtype must be a complex type."
    assert isinstance(verbose, bool), "verbose must be bool."

    ebno_dbs = tf.cast(ebno_dbs, dtype.real_dtype)
    batch_size = tf.cast(batch_size, tf.int32)
    num_points = tf.shape(ebno_dbs)[0]
    bit_errors = tf.Variable(   tf.zeros([num_points], dtype=tf.int64),
                            dtype=tf.int64)
    block_errors = tf.Variable(  tf.zeros([num_points], dtype=tf.int64),
                            dtype=tf.int64)
    nb_bits = tf.Variable(  tf.zeros([num_points], dtype=tf.int64),
                            dtype=tf.int64)
    nb_blocks = tf.Variable(  tf.zeros([num_points], dtype=tf.int64),
                            dtype=tf.int64)

    # track status of simulation (early termination etc.)
    status = np.zeros(num_points)

    # measure runtime per SNR point
    runtime = np.zeros(num_points)

    # ensure num_target_errors is a tensor
    if num_target_bit_errors is not None:
        num_target_bit_errors = tf.cast(num_target_bit_errors, tf.int64)
    if num_target_block_errors is not None:
        num_target_block_errors = tf.cast(num_target_block_errors, tf.int64)

    try:
        # simulate until a target number of errors is reached
        for i in tf.range(num_points):
            runtime[i] = time.perf_counter() # save start time
            iter_count = -1 # for print in verbose mode
            for ii in tf.range(max_mc_iter):

                iter_count += 1

                outputs = mc_fun(batch_size=batch_size, ebno_db=ebno_dbs[i])

                # assume first and second return value is b and b_hat
                # other returns are ignored
                b = outputs[0]
                b_hat = outputs[1]

                if soft_estimates:
                    b_hat = hard_decisions(b_hat)

                # count errors
                bit_e = count_errors(b, b_hat)
                block_e = count_block_errors(b, b_hat)

                # count total number of bits
                bit_n = tf.size(b)
                block_n = tf.size(b[...,-1])

                # update variables
                bit_errors = tf.tensor_scatter_nd_add(  bit_errors, [[i]],
                                                    tf.cast([bit_e], tf.int64))
                block_errors = tf.tensor_scatter_nd_add(  block_errors, [[i]],
                                                tf.cast([block_e], tf.int64))
                nb_bits = tf.tensor_scatter_nd_add( nb_bits, [[i]],
                                                    tf.cast([bit_n], tf.int64))
                nb_blocks = tf.tensor_scatter_nd_add( nb_blocks, [[i]],
                                                tf.cast([block_n], tf.int64))

                # print progress summary
                if verbose:
                    # print summary header during first iteration
                    if i==0 and iter_count==0:
                        _print_progress(is_final=True,
                                        rt=0,
                                        idx_snr=0,
                                        idx_it=0,
                                        header_text=header_text)
                        # print seperator after headline
                        print('-' * 135)

                    # evaluate current runtime
                    rt = time.perf_counter() - runtime[i]
                    # print current progress
                    _print_progress(is_final=False, idx_snr=i, idx_it=ii, rt=rt)


                # bit-error based stopping cond.
                if num_target_bit_errors is not None:
                    if tf.greater_equal(bit_errors[i], num_target_bit_errors):
                        status[i] = 3 # change internal status for summary
                        # stop runtime timer
                        runtime[i] = time.perf_counter() - runtime[i]
                        break # enough errors for SNR point have been simulated

                # block-error based stopping cond.
                if num_target_block_errors is not None:
                    if tf.greater_equal(block_errors[i],
                                        num_target_block_errors):
                        # stop runtime timer
                        runtime[i] = time.perf_counter() - runtime[i]
                        status[i] = 4 # change internal status for summary
                        break # enough errors for SNR point have been simulated

                # max iter have been reached -> continue with next SNR point
                if iter_count==max_mc_iter-1: # all iterations are done
                    # stop runtime timer
                    runtime[i] = time.perf_counter() - runtime[i]
                    status[i] = 1 # change internal status for summary

            # print results again AFTER last iteration / early stop (new status)
            if verbose:
                _print_progress(is_final=True,
                                idx_snr=i,
                                idx_it=iter_count,
                                rt=runtime[i])

            # early stop if no error occurred
            if early_stop: # only if early stop is active
                if block_errors[i]==0:
                    status[i] = 2 # change internal status for summary
                    if verbose:
                        print("\nSimulation stopped as no error occurred " \
                              f"@ EbNo = {ebno_dbs[i].numpy():.1f} dB.\n")
                    break

    # Stop if KeyboardInterrupt is detected and set remaining SNR points to -1
    except KeyboardInterrupt as e:

        # Raise Interrupt again to stop outer loops
        if forward_keyboard_interrupt:
            raise e

        print("\nSimulation stopped by the user " \
              f"@ EbNo = {ebno_dbs[i].numpy()} dB")
        # overwrite remaining BER / BLER positions with -1
        for idx in range(i+1, num_points):
            bit_errors = tf.tensor_scatter_nd_update( bit_errors, [[idx]],
                                                    tf.cast([-1], tf.int64))
            block_errors = tf.tensor_scatter_nd_update( block_errors, [[idx]],
                                                    tf.cast([-1], tf.int64))
            nb_bits = tf.tensor_scatter_nd_update( nb_bits, [[idx]],
                                                    tf.cast([1], tf.int64))
            nb_blocks = tf.tensor_scatter_nd_update( nb_blocks, [[idx]],
                                                    tf.cast([1], tf.int64))

    # calculate BER / BLER
    ber = tf.cast(bit_errors, tf.float64) / tf.cast(nb_bits, tf.float64)
    bler = tf.cast(block_errors, tf.float64) / tf.cast(nb_blocks, tf.float64)

    # replace nans (from early stop)
    ber = tf.where(tf.math.is_nan(ber), tf.zeros_like(ber), ber)
    bler = tf.where(tf.math.is_nan(bler), tf.zeros_like(bler), bler)

    return ber, bler


def complex_normal(shape, var=1.0, dtype=tf.complex64):
    r"""Generates a tensor of complex normal random variables.

    Input
    -----
    shape : tf.shape, or list
        The desired shape.

    var : float
        The total variance., i.e., each complex dimension has
        variance ``var/2``.

    dtype: tf.complex
        The desired dtype. Defaults to `tf.complex64`.

    Output
    ------
    : ``shape``, ``dtype``
        Tensor of complex normal random variables.
    """
    # Half the variance for each dimension
    var_dim = tf.cast(var, dtype.real_dtype)/tf.cast(2, dtype.real_dtype)
    stddev = tf.sqrt(var_dim)

    # Generate complex Gaussian noise with the right variance
    xr = tf.random.normal(shape, stddev=stddev, dtype=dtype.real_dtype)
    xi = tf.random.normal(shape, stddev=stddev, dtype=dtype.real_dtype)
    x = tf.complex(xr, xi)

    return x


###########################################################
# Deprecated aliases that will not be included in the next
# major release
###########################################################

def fft(tensor, axis=-1):
    print(  "Warning: The alias utils.fft will not be included in Sionna 1.0."
            " Please use signal.fft instead.")
    return signal.fft(tensor, axis)


def ifft(tensor, axis=-1):
    print(  "Warning: The alias utils.ifft will not be included in Sionna 1.0."
            " Please use signal.ifft instead.")
    return signal.ifft(tensor, axis)


def empirical_psd(x, show=True, oversampling=1.0, ylim=(-30,3)):
    print(  "Warning: The alias utils.empirical_psd will not be included in"
            " Sionna 1.0. Please use signal.empirical_psd instead.")
    return signal.empirical_psd(x, show, oversampling, ylim)
