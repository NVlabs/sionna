# pylint: disable=line-too-long
#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""Miscellaneous utility functions of Sionna PHY and SYS"""

from abc import ABC, abstractmethod
import time
import numpy as np
import tensorflow as tf
from tensorflow.experimental.numpy import log10 as _log10
from tensorflow.experimental.numpy import log2 as _log2
from scipy.interpolate import RectBivariateSpline, griddata

from sionna.phy import config, dtypes, Block
from sionna.phy.utils.metrics import count_errors, count_block_errors


def complex_normal(shape, var=1.0, precision=None):
    r"""Generates a tensor of complex normal random variables

    Input
    -----
    shape : `tf.shape`, or `list`
        Desired shape

    var : `float`
        Total variance., i.e., each complex dimension has
        variance ``var/2``.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Output
    ------
    : ``shape``, `tf.complex`
        Tensor of complex normal random variables
    """
    if precision is None:
        precision = config.precision
    rdtype = dtypes[precision]['tf']['rdtype']

    # Half the variance for each dimension
    var_dim = tf.cast(var, rdtype)/tf.cast(2, rdtype)
    stddev = tf.sqrt(var_dim)

    # Generate complex Gaussian noise with the right variance
    xr = config.tf_rng.normal(shape, stddev=stddev, dtype=rdtype)
    xi = config.tf_rng.normal(shape, stddev=stddev, dtype=rdtype)
    x = tf.complex(xr, xi)

    return x


def lin_to_db(x,
              precision=None):
    r"""
    Converts the input in linear scale to dB scale

    Input
    -----
    x : `tf.float`
        Input value in linear scale

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Output
    ------
    : `tf.float`
        Input value converted to [dB]
    """
    if precision is None:
        rdtype = config.tf_rdtype
    else:
        rdtype = dtypes[precision]["tf"]["rdtype"]

    ten = tf.cast(10, rdtype)
    x = tf.cast(x, rdtype)
    return ten * log10(x)


def db_to_lin(x,
              precision=None):
    r"""
    Converts the input [dB] to linear scale

    Input
    -----
    x : `tf.float`
        Input value [dB]

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Output
    ------
    : `tf.float`
        Input value converted to linear scale
    """
    if precision is None:
        rdtype = config.tf_rdtype
    else:
        rdtype = dtypes[precision]["tf"]["rdtype"]

    ten = tf.cast(10, rdtype)
    x = tf.cast(x, rdtype)
    return tf.math.pow(ten, x / ten)


def watt_to_dbm(x_w,
                precision=None):
    r"""
    Converts the input [Watt] to dBm

    Input
    -----
    x_w : `tf.float`
        Input value [Watt]

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Output
    ------
    : `tf.float`
        Input value converted to dBm
    """
    if precision is None:
        rdtype = config.tf_rdtype
    else:
        rdtype = dtypes[precision]["tf"]["rdtype"]
    return lin_to_db(x_w, precision=precision) + tf.cast(30., rdtype)


def dbm_to_watt(x_dbm,
                precision=None):
    r"""
    Converts the input [dBm] to Watt

    Input
    -----
    x_dbm : `tf.float`
        Input value [dBm]

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Output
    ------
    : `tf.float`
        Input value converted to Watt
    """
    if precision is None:
        rdtype = config.tf_rdtype
    else:
        rdtype = dtypes[precision]["tf"]["rdtype"]
    return db_to_lin(x_dbm, precision=precision) * tf.cast(.001, rdtype)


def ebnodb2no(ebno_db,
              num_bits_per_symbol,
              coderate,
              resource_grid=None,
              precision=None):
    r"""Computes the noise variance `No` for a given `Eb/No` in dB

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
    ebno_db : `float`
        `Eb/No` value in dB

    num_bits_per_symbol : `int`
        Number of bits per symbol

    coderate : `float`
        Coderate

    resource_grid : `None` (default) | :class:`~sionna.phy.ofdm.ResourceGrid`
        An (optional) resource grid for OFDM transmissions

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Output
    ------
    : `tf.float`
        Value of :math:`N_o` in linear scale
    """

    if precision is None:
        rdtype = config.tf_rdtype
    else:
        rdtype = dtypes[precision]["tf"]["rdtype"]

    ebno_db = tf.cast(ebno_db, rdtype)
    coderate = tf.cast(coderate, rdtype)
    ten = tf.cast(10, rdtype)
    ebno = tf.math.pow(ten, ebno_db/ten)

    energy_per_symbol = 1.
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

    no = 1/(ebno * coderate * tf.cast(num_bits_per_symbol, rdtype)
            / tf.cast(energy_per_symbol, rdtype))

    return no


def hard_decisions(llr):
    """Transforms LLRs into hard decisions

    Positive values are mapped to :math:`1`.
    Nonpositive values are mapped to :math:`0`.

    Input
    -----
    llr : any non-complex tf.DType
        Tensor of LLRs

    Output
    ------
    : Same shape and dtype as ``llr``
        Hard decisions
    """
    zero = tf.constant(0, dtype=llr.dtype)
    return tf.cast(tf.math.greater(llr, zero), dtype=llr.dtype)


def log10(x):
    # pylint: disable=C0301
    """TensorFlow implementation of NumPy's `log10` function

    Simple extension to `tf.experimental.numpy.log10`
    which casts the result to the `dtype` of the input.
    For more details see the `TensorFlow <https://www.tensorflow.org/api_docs/python/tf/experimental/numpy/log10>`__ and `NumPy <https://numpy.org/doc/1.16/reference/generated/numpy.log10.html>`__ documentation.
    """
    return tf.cast(_log10(x), x.dtype)


def log2(x):
    # pylint: disable=C0301
    """TensorFlow implementation of NumPy's `log2` function

    Simple extension to `tf.experimental.numpy.log2`
    which casts the result to the `dtype` of the input.
    For more details see the `TensorFlow <https://www.tensorflow.org/api_docs/python/tf/experimental/numpy/log2>`_ and `NumPy <https://numpy.org/doc/1.16/reference/generated/numpy.log2.html>`_ documentation.
    """
    return tf.cast(_log2(x), x.dtype)


def sample_bernoulli(shape,
                     p,
                     precision=None):
    r"""
    Generates samples from a Bernoulli distribution with probability ``p``

    Input
    --------
    shape : `tf.TensorShape`
        Shape of the tensor to sample

    p : Broadcastable with ``shape``, `tf.float`
        Probability

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Output
    --------
    : Tensor of shape ``shape``, `bool`
        Binary samples
    """
    if precision is None:
        rdtype = config.tf_rdtype
    else:
        rdtype = dtypes[precision]["tf"]["rdtype"]
    z = config.tf_rng.uniform(shape=shape, minval=0.0, maxval=1.0, dtype=rdtype)
    z = tf.math.less(z, p)
    return z


def sim_ber(mc_fun,
            ebno_dbs,
            batch_size,
            max_mc_iter,
            soft_estimates=False,
            num_target_bit_errors=None,
            num_target_block_errors=None,
            target_ber=None,
            target_bler=None,
            early_stop=True,
            graph_mode=None,
            distribute=None,
            verbose=True,
            forward_keyboard_interrupt=True,
            callback=None,
            precision=None):
    # pylint: disable=line-too-long
    """Simulates until target number of errors is reached and returns BER/BLER

    The simulation continues with the next SNR point if either
    ``num_target_bit_errors`` bit errors or ``num_target_block_errors`` block
    errors is achieved. Further, it continues with the next SNR point after
    ``max_mc_iter`` batches of size ``batch_size`` have been simulated.
    Early stopping allows to stop the simulation after the first error-free SNR
    point or after reaching a certain ``target_ber`` or ``target_bler``.

    Input
    -----
    mc_fun: `callable`
        Callable that yields the transmitted bits `b` and the
        receiver's estimate `b_hat` for a given ``batch_size`` and
        ``ebno_db``. If ``soft_estimates`` is True, `b_hat` is interpreted as
        logit.

    ebno_dbs: [n], `tf.float`
        A tensor containing SNR points to be evaluated.

    batch_size: `tf.int`
        Batch-size for evaluation

    max_mc_iter: `tf.int`
        Maximum number of Monte-Carlo iterations per SNR point

    soft_estimates: `bool`, (default `False`)
        If `True`, `b_hat` is interpreted as logit and an additional
        hard-decision is applied internally.

    num_target_bit_errors: `None` (default) | `tf.int32`
        Target number of bit errors per SNR point until
        the simulation continues to next SNR point

    num_target_block_errors: `None` (default) | `tf.int32`
        Target number of block errors per SNR point
        until the simulation continues

    target_ber: `None` (default) | `tf.float32`
        The simulation stops after the first SNR point
        which achieves a lower bit error rate as specified by ``target_ber``.
        This requires ``early_stop`` to be `True`.

    target_bler: `None` (default) | `tf.float32`
        The simulation stops after the first SNR point
        which achieves a lower block error rate as specified by ``target_bler``.
        This requires ``early_stop`` to be `True`.

    early_stop: `None` (default) | `bool`
        If `True`, the simulation stops after the
        first error-free SNR point (i.e., no error occurred after
        ``max_mc_iter`` Monte-Carlo iterations).

    graph_mode: `None` (default) | "graph" | "xla"
        A string describing the execution mode of ``mc_fun``.
        If `None`, ``mc_fun`` is executed as is.

    distribute: `None` (default) | "all" | list of indices | `tf.distribute.strategy`
        Distributes simulation on multiple parallel devices. If `None`,
        multi-device simulations are deactivated. If "all", the workload will
        be automatically distributed across all available GPUs via the
        `tf.distribute.MirroredStrategy`.
        If an explicit list of indices is provided, only the GPUs with the given
        indices will be used. Alternatively, a custom `tf.distribute.strategy`
        can be provided. Note that the same `batch_size` will be
        used for all GPUs in parallel, but the number of Monte-Carlo iterations
        ``max_mc_iter`` will be scaled by the number of devices such that the
        same number of total samples is simulated. However, all stopping
        conditions are still in-place which can cause slight differences in the
        total number of simulated samples.

    verbose: `bool`, (default `True`)
        If `True`, the current progress will be printed.

    forward_keyboard_interrupt: `bool`, (default `True`)
        If `False`, KeyboardInterrupts will be
        catched internally and not forwarded (e.g., will not stop outer loops).
        If `True`, the simulation ends and returns the intermediate simulation
        results.

    callback: `None` (default) | `callable`
        If specified, ``callback`` will be called after each Monte-Carlo step.
        Can be used for logging or advanced early stopping. Input signature of
        ``callback`` must match `callback(mc_iter, snr_idx, ebno_dbs,
        bit_errors, block_errors, nb_bits, nb_blocks)` where ``mc_iter``
        denotes the number of processed batches for the current SNR point,
        ``snr_idx`` is the index of the current SNR point, ``ebno_dbs`` is the
        vector of all SNR points to be evaluated, ``bit_errors`` the vector of
        number of bit errors for each SNR point, ``block_errors`` the vector of
        number of block errors, ``nb_bits`` the vector of number of simulated
        bits, ``nb_blocks`` the vector of number of simulated blocks,
        respectively. If ``callable`` returns `sim_ber.CALLBACK_NEXT_SNR`, early
        stopping is detected and the simulation will continue with the
        next SNR point. If ``callable`` returns
        `sim_ber.CALLBACK_STOP`, the simulation is stopped
        immediately. For `sim_ber.CALLBACK_CONTINUE` continues with
        the simulation.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Output
    ------
    ber: [n], `tf.float`
        Bit-error rate.

    bler: [n], `tf.float`
        Block-error rate

    Note
    ----
    This function is implemented based on tensors to allow
    full compatibility with tf.function(). However, to run simulations
    in graph mode, the provided ``mc_fun`` must use the `@tf.function()`
    decorator.
    """
    if precision is None:
        precision = config.precision
    rdtype = dtypes[precision]['tf']['rdtype']

    # pylint: disable=invalid-name
    STATUS_NA = 0
    STATUS_MAX_IT = 1
    STATUS_NO_ERR = 2
    STATUS_TARGET_BIT = 3
    STATUS_TARGET_BLOCK = 4
    STATUS_TARGET_BER = 5
    STATUS_TARGET_BLER = 6
    STATUS_CB_STOP = 7

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
            # float64 precision is used to avoid rounding errors
            ber_np = (tf.cast(bit_errors[idx_snr], tf.float64)
                      / tf.cast(nb_bits[idx_snr], tf.float64)).numpy()
            ber_np = np.nan_to_num(ber_np)  # avoid nan for first point
            bler_np = (tf.cast(block_errors[idx_snr], tf.float64)
                       / tf.cast(nb_blocks[idx_snr], tf.float64)).numpy()
            bler_np = np.nan_to_num(bler_np)  # avoid nan for first point

            # load statuslevel
            # print current iter if simulation is still running
            if status[idx_snr] == STATUS_NA:
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
        print("{: >9} |{: >11} |{: >11} |{: >12} |{: >12} |{: >13} |{: >12} |{: >12} |{: >10}".format(
            *row_text), end=end_str)

    # distributed execution should not be done in Eager mode
    # XLA mode seems to have difficulties with TF2.13
    @tf.function(jit_compile=False)
    def _run_distributed(strategy, mc_fun, batch_size, ebno_db):
        # use tf.distribute to execute on parallel devices (=replicas)
        outputs_rep = strategy.run(mc_fun,
                                   args=(batch_size, ebno_db))
        # copy replicas back to single device
        b = strategy.gather(outputs_rep[0], axis=0)
        b_hat = strategy.gather(outputs_rep[1], axis=0)
        return b, b_hat

     # init table headers
    header_text = [
        "EbNo [dB]", "BER", "BLER", "bit errors", "num bits",
        "block errors", "num blocks", "runtime [s]", "status"
    ]

    # Dictionary mapping statuses to their labels
    status_levels = {
        STATUS_NA: "not simulated",
        STATUS_MAX_IT: "reached max iterations",
        STATUS_NO_ERR: "no errors - early stop",
        STATUS_TARGET_BIT: "reached target bit errors",
        STATUS_TARGET_BLOCK: "reached target block errors",
        STATUS_TARGET_BER: "reached target BER - early stop",
        STATUS_TARGET_BLER: "reached target BLER - early stop",
        STATUS_CB_STOP: "callback triggered stopping",
    }
    # check inputs for consistency
    if not isinstance(early_stop, bool):
        raise TypeError("early_stop must be bool.")
    if not isinstance(soft_estimates, bool):
        raise TypeError("soft_estimates must be bool.")
    if not isinstance(verbose, bool):
        raise TypeError("verbose must be bool.")

    # target_ber / target_bler only works if early stop is activated
    if target_ber is not None:
        if not early_stop:
            msg = "Warning: early stop is deactivated. target_ber is ignored."
            print(msg)
    else:
        target_ber = -1.  # deactivate early stopping condition
    if target_bler is not None:
        if not early_stop:
            msg = "Warning: early stop is deactivated. target_bler is ignored."
            print(msg)
    else:
        target_bler = -1.  # deactivate early stopping condition

    ###################
    # Graph-Mode & XLA
    ###################

    if graph_mode is None:
        graph_mode = "default"  # applies default graph mode
    if not isinstance(graph_mode, str):
        raise TypeError("graph_mode must be str.")

    if graph_mode == "default":
        pass  # nothing to do
    elif graph_mode == "graph":
        # avoid retracing -> check if mc_fun is already a function
        if not isinstance(mc_fun, tf.types.experimental.GenericFunction):
            mc_fun = tf.function(mc_fun, jit_compile=False,
                                 experimental_follow_type_hints=True)
    elif graph_mode == "xla":
        # avoid retracing -> check if mc_fun is already a function
        if not isinstance(mc_fun, tf.types.experimental.GenericFunction) or \
           not mc_fun.function_spec.jit_compile:
            mc_fun = tf.function(mc_fun, jit_compile=True,
                                 experimental_follow_type_hints=True)
    else:
        raise TypeError("Unknown graph_mode selected.")

    ############
    # Multi-GPU
    ############

    # support multi-device simulations by using the tf.distribute package

    if len(tf.config.list_logical_devices('GPU')) == 0:
        run_multigpu = False
        distribute = None
    if distribute is None:  # disabled per default
        run_multigpu = False
    # use strategy if explicitly provided
    elif isinstance(distribute, tf.distribute.Strategy):
        run_multigpu = True
        strategy = distribute  # distribute is already a tf.distribute.strategy
    else:
        run_multigpu = True
        # use all available gpus
        if distribute == "all":
            gpus = tf.config.list_logical_devices('GPU')
        # mask active GPUs if indices are provided
        elif isinstance(distribute, (tuple, list)):
            gpus_avail = tf.config.list_logical_devices('GPU')
            gpus = [gpus_avail[i] for i in distribute if i < len(gpus_avail)]
        else:
            raise ValueError("Unknown value for distribute.")

        # deactivate logging of tf.device placement
        if verbose:
            print("Setting tf.debugging.set_log_device_placement to False.")
        tf.debugging.set_log_device_placement(False)
        # we reduce to the first device by default
        strategy = tf.distribute.MirroredStrategy(gpus,
                                                  cross_device_ops=tf.distribute.ReductionToOneDevice(
                                                      reduce_to_device=gpus[0].name))

    # reduce max_mc_iter if multi_gpu simulations are activated
    if run_multigpu:
        num_replicas = strategy.num_replicas_in_sync  # pylint: disable=possibly-used-before-assignment
        max_mc_iter = int(np.ceil(max_mc_iter/num_replicas))
        print(f"Distributing simulation across {num_replicas} devices.")
        print(f"Reducing max_mc_iter to {max_mc_iter}")

    ##########################
    # Init internal variables
    ##########################

    # we use int64 and float64 for all bit/error statistics
    # This avoids inaccurate results for long running simulations

    ebno_dbs = tf.cast(ebno_dbs, rdtype)
    batch_size = tf.cast(batch_size, tf.int32)
    num_points = tf.shape(ebno_dbs)[0]
    bit_errors = tf.Variable(tf.zeros([num_points], dtype=tf.int64),
                             dtype=tf.int64)
    block_errors = tf.Variable(tf.zeros([num_points], dtype=tf.int64),
                               dtype=tf.int64)
    nb_bits = tf.Variable(tf.zeros([num_points], dtype=tf.int64),
                          dtype=tf.int64)
    nb_blocks = tf.Variable(tf.zeros([num_points], dtype=tf.int64),
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

    ####################
    # Run MC simulation
    ####################

    try:
        # simulate until a target number of errors is reached
        for i in tf.range(num_points):
            runtime[i] = time.perf_counter()  # save start time
            iter_count = -1  # for print in verbose mode
            for ii in tf.range(max_mc_iter):

                iter_count += 1

                if run_multigpu:  # distributed execution
                    b, b_hat = _run_distributed(strategy,
                                                mc_fun,
                                                batch_size,
                                                ebno_dbs[i])
                else:
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
                block_n = tf.size(b[..., -1])

                # update variables
                bit_errors.scatter_nd_add([[i]], tf.cast([bit_e], tf.int64))
                block_errors.scatter_nd_add([[i]], tf.cast([block_e], tf.int64))
                nb_bits.scatter_nd_add([[i]], tf.cast([bit_n], tf.int64))
                nb_blocks.scatter_nd_add([[i]], tf.cast([block_n], tf.int64))

                cb_state = sim_ber.CALLBACK_CONTINUE
                if callback is not None:
                    cb_state = callback(ii, i, ebno_dbs, bit_errors,
                                        block_errors, nb_bits,
                                        nb_blocks)
                    if cb_state in (sim_ber.CALLBACK_STOP,
                                    sim_ber.CALLBACK_NEXT_SNR):
                        # stop runtime timer
                        runtime[i] = time.perf_counter() - runtime[i]
                        # change internal status for summary
                        status[i] = STATUS_CB_STOP
                        break  # stop for this SNR point have been simulated

                # print progress summary
                if verbose:
                    # print summary header during first iteration
                    if i == 0 and iter_count == 0:
                        _print_progress(is_final=True,
                                        rt=0,
                                        idx_snr=0,
                                        idx_it=0,
                                        header_text=header_text)
                        # print separator after headline
                        print('-' * 135)

                    # evaluate current runtime
                    rt = time.perf_counter() - runtime[i]
                    # print current progress
                    _print_progress(is_final=False, idx_snr=i, idx_it=ii, rt=rt)

                # bit-error based stopping cond.
                if num_target_bit_errors is not None:
                    if tf.greater_equal(bit_errors[i], num_target_bit_errors):
                        # change internal status for summary
                        status[i] = STATUS_TARGET_BIT
                        # stop runtime timer
                        runtime[i] = time.perf_counter() - runtime[i]
                        break  # enough errors for SNR point have been simulated

                # block-error based stopping cond.
                if num_target_block_errors is not None:
                    if tf.greater_equal(block_errors[i],
                                        num_target_block_errors):
                        # stop runtime timer
                        runtime[i] = time.perf_counter() - runtime[i]
                        # change internal status for summary
                        status[i] = STATUS_TARGET_BLOCK
                        break  # enough errors for SNR point have been simulated

                # max iter have been reached -> continue with next SNR point
                if iter_count == max_mc_iter-1:  # all iterations are done
                    # stop runtime timer
                    runtime[i] = time.perf_counter() - runtime[i]
                    # change internal status for summary
                    status[i] = STATUS_MAX_IT

            # print results again AFTER last iteration / early stop (new status)
            if verbose:
                _print_progress(is_final=True,
                                idx_snr=i,
                                idx_it=iter_count,
                                rt=runtime[i])

            # early stop if no error occurred or target_ber/target_bler reached
            if early_stop:  # only if early stop is active
                if block_errors[i] == 0:
                    # change internal status for summary
                    status[i] = STATUS_NO_ERR
                    if verbose:
                        print("\nSimulation stopped as no error occurred "
                              f"@ EbNo = {ebno_dbs[i].numpy():.1f} dB.\n")
                    break

                # check for target_ber / target_bler
                ber_true = bit_errors[i] / nb_bits[i]
                bler_true = block_errors[i] / nb_blocks[i]
                if ber_true < target_ber:
                    # change internal status for summary
                    status[i] = STATUS_TARGET_BER
                    if verbose:
                        print("\nSimulation stopped as target BER is reached"
                              f"@ EbNo = {ebno_dbs[i].numpy():.1f} dB.\n")
                    break
                if bler_true < target_bler:
                    # change internal status for summary
                    status[i] = STATUS_TARGET_BLER
                    if verbose:
                        print("\nSimulation stopped as target BLER is "
                              f"reached @ EbNo = {ebno_dbs[i].numpy():.1f} "
                              "dB.\n")
                    break

            # allow callback to end the entire simulation
            if cb_state is sim_ber.CALLBACK_STOP:
                # stop runtime timer
                # change internal status for summary
                status[i] = STATUS_CB_STOP
                if verbose:
                    print("\nSimulation stopped by callback function "
                          f"@ EbNo = {ebno_dbs[i].numpy():.1f} dB.\n")
                break

    # Stop if KeyboardInterrupt is detected and set remaining SNR points to -1
    except KeyboardInterrupt as e:

        # Raise Interrupt again to stop outer loops
        if forward_keyboard_interrupt:
            raise e

        print("\nSimulation stopped by the user "
              f"@ EbNo = {ebno_dbs[i].numpy()} dB.")
        # overwrite remaining BER / BLER positions with -1
        for idx in range(i+1, num_points):
            bit_errors.scatter_nd_add([[idx]], tf.cast([-1], tf.int64))
            block_errors.scatter_nd_add([[idx]], tf.cast([-1], tf.int64))
            nb_bits.scatter_nd_add([[idx]], tf.cast([1], tf.int64))
            nb_blocks.scatter_nd_add([[idx]], tf.cast([1], tf.int64))

    # calculate BER / BLER
    ber = tf.cast(bit_errors, tf.float64) / tf.cast(nb_bits, tf.float64)
    bler = tf.cast(block_errors, tf.float64) / tf.cast(nb_blocks, tf.float64)

    # replace nans (from early stop)
    ber = tf.where(tf.math.is_nan(ber), tf.zeros_like(ber), ber)
    bler = tf.where(tf.math.is_nan(bler), tf.zeros_like(bler), bler)

    # cast output to target dtype
    ber = tf.cast(ber, rdtype)
    bler = tf.cast(bler, rdtype)

    return ber, bler


sim_ber.CALLBACK_CONTINUE = None
sim_ber.CALLBACK_STOP = 2
sim_ber.CALLBACK_NEXT_SNR = 1


def to_list(x):
    """
    Converts the input to a list

    Input
    -----

    x : `list` | `float` | `int` | `str` | `None`
        Input, to be converted to a list

    Output
    ------
    : `list`
        Input converted to a list
    """
    if x is not None:
        if isinstance(x, str) | \
                (not hasattr(x, '__len__')):
            x = [x]
        else:
            x = list(x)
    return x


def dict_keys_to_int(x):
    r"""
    Converts the string keys of an input dictionary to integers whenever
    possible

    Input
    -----

    x : `dict`
        Input dictionary

    Output
    ------

    : `dict`
        Dictionary with integer keys

    Example
    -------
    .. code-block:: Python

        from sionna.phy.utils import dict_keys_to_int

        dict_in = {'1': {'2': [45, '3']}, '4.3': 6, 'd': [5, '87']}
        print(dict_keys_to_int(dict_in))
        # {1: {'2': [45, '3']}, '4.3': 6, 'd': [5, '87']})
    """
    if isinstance(x, dict):
        x_new = {}
        for k, v in x.items():
            try:
                k_new = int(k)
            except ValueError:
                k_new = k
            x_new[k_new] = v
        return x_new
    else:
        return x


def scalar_to_shaped_tensor(inp, dtype, shape):
    r"""
    Converts a scalar input to a tensor of specified shape, or validates and casts 
    an existing input tensor. If the input is a scalar, creates a
    tensor of the specified shape filled with that value. Otherwise, verifies
    the input tensor matches the required shape and casts it to the specified dtype.

    Input
    -----
    inp : `int` | `float` | `bool` | `tf.Tensor`
        Input value. If scalar (int, float, bool, or shapeless tensor), it will be
        used to fill a new tensor. If a shaped tensor, its shape must match the
        specified shape.

    dtype : tf.dtype
        Desired data type of the output tensor

    shape : `list`
        Required shape of the output tensor

    Output
    ------
    : `tf.Tensor`
        A tensor of shape `shape` and type `dtype`. Either filled with the scalar
        input value or the input tensor cast to the specified dtype.
    """
    if isinstance(inp, (int, float, bool)) or (len(inp.shape) == 0):
        return tf.cast(tf.fill(shape, inp), dtype)
    else:
        tf.debugging.assert_shapes([(inp, shape)],
                                   message='Inconsistent shape')
        return tf.cast(inp, dtype)


class DeepUpdateDict(dict):
    r"""
    Class inheriting from `dict` enabling nested merging of the dictionary with
    a new one
    """

    def _deep_update(self, dict_orig, delta, stop_at_keys=()):
        for key in delta:
            if (key not in dict_orig) or \
                (not isinstance(delta[key], dict)) or \
                (not isinstance(dict_orig[key], dict)) or \
                    (key in to_list(stop_at_keys)):
                dict_orig[key] = delta[key]
            else:
                self._deep_update(dict_orig[key], delta[key],
                                  stop_at_keys=stop_at_keys)

    def deep_update(self, delta, stop_at_keys=()):
        r"""
        Merges ``self`` with the input ``delta`` in nested fasion. 
        In case of conflict, the values of the new dictionary prevail.
        The two dictionary are merged at intermediate keys 
        ``stop_at_keys``, if provided.

        Input
        -----

        delta : `dict`
            Dictionary to be merged with ``self``

        stop_at_keys : `tuple`
            Tuple of keys at which the subtree of ``delta`` replaces the
            corresponding subtree of ``self``

        Example
        -------
        .. code-block:: Python

            from sionna.phy.utils import DeepUpdateDict

            # Merge without conflicts
            dict1 = DeepUpdateDict(
                {'a': 1,
                 'b':
                  {'b1': 10,
                   'b2': 20}})
            dict_delta1 = {'c': -2,
                        'b':
                        {'b3': 30}}
            dict1.deep_update(dict_delta1)
            print(dict1)
            # {'a': 1, 'b': {'b1': 10, 'b2': 20, 'b3': 30}, 'c': -2}

            # Compare against the classic "update" method, which is not nested
            dict1 = DeepUpdateDict(
                {'a': 1,
                 'b':
                  {'b1': 10,
                   'b2': 20}})
            dict1.update(dict_delta1)
            print(dict1)
            # {'a': 1, 'b': {'b3': 30}, 'c': -2}

            # Handle key conflicts
            dict2 = DeepUpdateDict(
                {'a': 1,
                 'b':
                  {'b1': 10,
                   'b2': 20}})
            dict_delta2 = {'a': -2,
                        'b':
                        {'b1': {'f': 3, 'g': 4}}}
            dict2.deep_update(dict_delta2)
            print(dict2)
            # {'a': -2, 'b': {'b1': {'f': 3, 'g': 4}, 'b2': 20}}

            # Merge at intermediate keys
            dict2 = DeepUpdateDict(
                {'a': 1,
                 'b':
                  {'b1': 10,
                   'b2': 20}})
            dict2.deep_update(dict_delta2, stop_at_keys='b')
            print(dict2)
            # {'a': -2, 'b': {'b1': {'f': 3, 'g': 4}}}
        """
        self._deep_update(self, delta, stop_at_keys=stop_at_keys)


class Interpolate(ABC):
    r"""
    Class template for interpolating data defined on unstructured or rectangular
    grids. Used in :class:`~sionna.sys.PHYAbstraction` for 
    BLER and SNR interpolation.
    """

    @abstractmethod
    def unstruct(self,
                 z,
                 x,
                 y,
                 x_interp,
                 y_interp,
                 **kwargs):
        r"""
        Interpolates unstructured data

        Input
        -----

        z : [N], `array`
            Co-domain sample values. Informally, ``z`` = f (
            ``x`` , ``y`` )

        x : [N], `array`
            First coordinate of the domain sample values 

        y : [N], `array`
            Second coordinate of the domain sample values 

        x_interp : [L], `array`
            Interpolation grid for the first (`x`) coordinate. Typically, :math:`L
            \gg N` 

        y_interp : [J], `array`
            Interpolation grid for the second (`y`) coordinate. Typically, :math:`J
            \gg N`

        griddata_method : `linear` | `nearest`, `cubic`
            Interpolation method. See Scipy's `interpolate.griddata` for more details

        Output
        ------
        z_interp : [L, J], `np.array`
            Interpolated data
        """
        pass

    @abstractmethod
    def struct(self,
               z,
               x,
               y,
               x_interp,
               y_interp,
               **kwargs):
        r"""
        Interpolates data structured in rectangular grids

        Input
        -----

        z : [N, M], `array`
            Co-domain sample values. Informally, ``z`` = f (
            ``x`` , ``y`` )

        x : [N], `array`
            First coordinate of the domain sample values 

        y : [M], `array`
            Second coordinate of the domain sample values 

        x_interp : [L], `array`
            Interpolation grid for the first (`x`) coordinate. Typically, :math:`L
            \gg N` 

        y_interp : [J], `array`
            Interpolation grid for the second (`y`) coordinate. Typically, :math:`J
            \gg M`

        kwargs :
            Additional interpolation parameters

        Output
        ------
        z_interp : [L, J], `np.array`
            Interpolated data
        """
        pass


class SplineGriddataInterpolation(Interpolate):
    r"""
    Interpolates data defined on rectangular or unstructured grids via
    Scipy's `interpolate.RectBivariateSpline` and `interpolate.griddata`, respectively.
    It inherits from :class:`~sionna.phy.utils.Interpolate`
    """

    def struct(self,
               z,
               x,
               y,
               x_interp,
               y_interp,
               spline_degree=1,
               **kwargs):
        r"""
        Perform spline interpolation via Scipy's
        `interpolate.RectBivariateSpline`

        Input
        -----

        z : [N, M], `array`
            Co-domain sample values. Informally, ``z`` = f (
            ``x`` , ``y`` ).

        x : [N], `array`
            First coordinate of the domain sample values 

        y : [M], `array`
            Second coordinate of the domain sample values 

        x_interp : [L], `array`
            Interpolation grid for the first (`x`) coordinate. Typically, :math:`L
            \gg N`.

        y_interp : [J], `array`
            Interpolation grid for the second (`y`) coordinate. Typically, :math:`J
            \gg M`.

        spline_degree : `int` (default: 1)
            Spline interpolation degree

        Output
        ------
        z_interp : [L, J], `np.array`
            Interpolated data
        """
        if len(x) <= spline_degree:
            raise ValueError('Too few points for interpolation')

        # Compute log10(mat), replacing zeros with a "low" value to avoid inf
        log_mat = np.zeros(z.shape)
        mat_is0 = z == 0
        if mat_is0.sum() > 0:
            log_mat_not0 = np.log10(z[~mat_is0])
            min_log_mat_not0 = min(log_mat_not0)
            log_mat[~mat_is0] = log_mat_not0
            log_mat[mat_is0] = min(log_mat_not0) - 2
        else:
            log_mat = np.log10(z)
            min_log_mat_not0 = -np.inf

        # Spline interpolation on log10(BLER) to improve
        # numerical precision
        fun_interp = RectBivariateSpline(
            x,
            y,
            log_mat,
            kx=spline_degree,
            ky=spline_degree)

        log_mat_interp = fun_interp(
            x_interp,
            y_interp)

        # Retrieve the BLER
        mat_interp = np.power(10, log_mat_interp)
        # Replace "low" value and retrieve zeros
        mat_interp[mat_interp < 10**min_log_mat_not0] = 0

        return mat_interp

    def unstruct(self,
                 z,
                 x,
                 y,
                 x_interp,
                 y_interp,
                 griddata_method='linear',
                 **kwargs):
        r"""
        Interpolates unstructured data via Scipy's `interpolate.griddata`

        Input
        -----

        z : [N], `array`
            Co-domain sample values. Informally, ``z`` = f (
            ``x`` , ``y`` ).

        x : [N], `array`
            First coordinate of the domain sample values 

        y : [N], `array`
            Second coordinate of the domain sample values 

        x_interp : [L], `array`
            Interpolation grid for the first (`x`) coordinate. Typically, :math:`L
            \gg N`.

        y_interp : [J], `array`
            Interpolation grid for the second (`y`) coordinate. Typically, :math:`J
            \gg N`.

        griddata_method : "linear" | "nearest" | "cubic"
            Interpolation method. See Scipy's `interpolate.griddata` for more details.

        Output
        ------
        z_interp : [L, J], `np.array`
            Interpolated data
        """
        y_grid, x_grid = np.meshgrid(y_interp, x_interp)
        # Interpolate on irregular grid data
        z_interp = griddata(list(zip(y, x)),
                            z,
                            (y_grid, x_grid),
                            method=griddata_method)
        return z_interp


class MCSDecoder(Block):
    r"""
    Class template for mapping a Modulation and Coding Scheme (MCS) index to the
    corresponding modulation order, i.e., number of bits per symbol, and
    coderate.
    
    Input
    -----

    mcs_index : [...], `tf.int32`
        MCS index

    mcs_table_index : [...], `tf.int32`
        MCS table index. Different tables contain different mappings.

    mcs_category : [...], `tf.int32`
        Table category which may correspond, e.g., to uplink or
        downlink transmission

    check_index_validity : `bool` (default: `True`)
        If `True`, an ValueError is thrown is the input mcs indices are not
        valid for the given configuration

    Output
    ------

    modulation_order : [...], `tf.int32`
        Modulation order corresponding to the input MCS index

    coderate : [...], `tf.float`
        Coderate corresponding to the input MCS index
    """

    def call(self,
             mcs_index,
             mcs_table_index,
             mcs_category,
             check_index_validity=True,
             **kwargs):
        pass


class TransportBlock(Block):
    r"""
    Class template for computing the number and size (measured in n. bits) of code
    blocks within a transport block, given the modulation order, coderate and
    the total number of coded bits of a transport block. Used in
    :class:`~sionna.sys.PHYAbstraction`.

    Input 
    -----

    modulation_order : [...], `tf.int32`
        Modulation order, i.e., number of bits per symbol

    target_rate : [...], `tf.float32`
        Target coderate

    num_coded_bits : [...], `tf.float32`
        Total number of coded bits across all codewords

    Output
    ------

    cb_size : [...], `tf.int32`
        Code block (CB) size, i.e., number of information bits per code block

    num_cb : [...], `tf.int32`
        Number of code blocks that the transport block is segmented into
    """
    @abstractmethod
    def call(self,
             modulation_order,
             target_coderate,
             num_coded_bits,
             **kwargs):
        pass


class SingleLinkChannel(Block):
    """ Class template for simulating a single-link, i.e., single-carrier and
    single-stream, channels. Used for generating BLER tables in
    :meth:`~sionna.sys.PHYAbstraction.new_bler_table`.

    Parameters
    ----------
    num_bits_per_symbol : `int`
        Number of bits per symbol, i.e., modulation order

    num_info_bits : `int`
        Number of information bits per code block

    target_coderate : `float`
        Target code rate, i.e., the target ratio between the information and the
        coded bits within a block

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.
    
    Input
    -----
    batch_size : int
        Size of the simulation batches

    ebno_db : float
        `Eb/No` value in dB

    Output
    ------
    bits : [``batch_size``, ``num_info_bits``], `int`
        Transmitted bits

    bits_hat : [``batch_size``, ``num_info_bits``], `int`
        Decoded bits
    """
    def __init__(self,
                 num_bits_per_symbol,
                 num_info_bits,
                 target_coderate,
                 precision=None):

        super().__init__(precision=precision)
        self._num_bits_per_symbol = None
        self._num_info_bits = None
        self._target_coderate = None
        self._num_coded_bits = None

        if num_bits_per_symbol is not None:
            self.num_bits_per_symbol = num_bits_per_symbol
        if target_coderate is not None:
            self.target_coderate = target_coderate
        if num_info_bits is not None:
            self.num_info_bits = num_info_bits

    @property
    def num_bits_per_symbol(self):
        """
        `int`: Get/set the modulation order
        """
        return self._num_bits_per_symbol

    @num_bits_per_symbol.setter
    def num_bits_per_symbol(self, value):
        tf.debugging.assert_positive(
            value,
            message="num_bits_per_symbol must be a positive integer")
        self._num_bits_per_symbol = int(value)
        self.set_num_coded_bits()

    @property
    def num_info_bits(self):
        """
        `int` : Get/set the number of information bits per code block
        """
        return self._num_info_bits

    @num_info_bits.setter
    def num_info_bits(self, value):
        tf.debugging.assert_positive(
            value,
            message="num_info_bits must be a positive integer")
        self._num_info_bits = int(value)
        self.set_num_coded_bits()

    @property
    def target_coderate(self):
        """
        `float` : Get/set the target coderate
        """
        return self._target_coderate

    @target_coderate.setter
    def target_coderate(self, value):
        tf.debugging.assert_equal(
            0 <= value <= 1,
            True,
            message="target_coderate must be within [0;1]")
        self._target_coderate = value
        self.set_num_coded_bits()

    @property
    def num_coded_bits(self):
        """
        `int` (read-only) : Number of coded bits in a code block
        """
        return self._num_coded_bits

    def set_num_coded_bits(self):
        """
        Compute the number of coded bits per code block
        """
        if (self.num_info_bits is not None) and \
            (self.target_coderate is not None) and \
                (self.num_bits_per_symbol is not None):

            num_coded_bits = self.num_info_bits / self.target_coderate
            # Ensure that num_coded_bits is a multiple of num_bits_per_symbol
            self._num_coded_bits = \
                np.ceil(num_coded_bits / self.num_bits_per_symbol) * \
                    self.num_bits_per_symbol

    @abstractmethod
    def call(self,
             batch_size,
             ebno_db,
             **kwargs):
        raise NotImplementedError('Not implemented')
