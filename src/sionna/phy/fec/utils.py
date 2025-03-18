#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""Utility functions and blocks for the FEC package."""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import warnings
from importlib_resources import files, as_file
from sionna.phy import Block
from sionna.phy.fec.ldpc import codes
from sionna.phy.utils import log2
from sionna.phy import config

class GaussianPriorSource(Block):
    r"""Generates synthetic Log-Likelihood Ratios (LLRs) for Gaussian channels.

    Generates synthetic Log-Likelihood Ratios (LLRs) as if an all-zero codeword
    was transmitted over a Binary Additive White Gaussian Noise (Bi-AWGN)
    channel. The LLRs are generated based on either the noise variance ``no``
    or mutual information. If mutual information is used, it represents the
    information associated with a binary random variable observed through an
    AWGN channel.

    .. image:: ../figures/GaussianPriorSource.png

    The generated LLRs follow a Gaussian distribution with parameters:

    .. math::
        \sigma_{\text{llr}}^2 = \frac{4}{\sigma_\text{ch}^2}

    .. math::
        \mu_{\text{llr}} = \frac{\sigma_\text{llr}^2}{2}

    where :math:`\sigma_\text{ch}^2` is the noise variance specified by ``no``.

    If the mutual information is provided as input, the J-function as described
    in [Brannstrom]_ is used to relate the mutual information to the
    corresponding LLR distribution.

    Parameters
    ----------
    precision : `None` (default) | 'single' | 'double'
        Precision used for internal calculations and outputs.
        If set to `None`, :py:attr:`~sionna.phy.config.precision` is used.

    Input
    -----
    output_shape : tf.int
        Integer tensor or Python list defining the shape of the generated LLR
        tensor.

    no : None (default) | tf.float
        Scalar defining the noise variance for the synthetic AWGN channel.

    mi : None (default) | tf.float
        Scalar defining the mutual information for the synthetic AWGN channel.
        Only used of ``no`` is None.

    Output
    ------
    tf.Tensor of dtype ``dtype`` (defaults to `tf.float32`)
        Tensor with shape defined by ``output_shape``.
    """

    def __init__(self, *, precision=None, **kwargs):

        super().__init__(precision=precision, **kwargs)

    def call(self, output_shape, no=None, mi=None):
        """Generate Gaussian distributed fake LLRs as if the all-zero codeword
        was transmitted over an Bi-AWGN channel.

        Args:
        output_shape : tf.int
            Integer tensor or Python list defining the shape of the generated
            LLR tensor.

        no : None (default) | tf.float
            Scalar defining the noise variance for the synthetic AWGN channel.

        mi : None (default) | tf.float
            Scalar defining the mutual information for the synthetic AWGN
            channel. Only used of ``no`` is None.

        Returns:
            1+D Tensor (``dtype``): Shape as defined by ``output_shape``.
        """

        if no is None:
            if mi is None:
                raise ValueError("Either no or mi must be provided.")
            # clip Ia to range (0,1)
            mi = tf.cast(mi, self.rdtype)
            mi = tf.maximum(mi, 1e-7)
            mi = tf.minimum(mi, 1.)
            mu_llr = j_fun_inv(mi)
            sigma_llr = tf.math.sqrt(2*mu_llr)
        else:
            # noise_var must be positive
            no = tf.cast(no, self.rdtype)
            no = tf.maximum(no, 1e-7)
            sigma_llr = tf.math.sqrt(4 / no)
            mu_llr = sigma_llr**2  / 2

        # generate LLRs with Gaussian approximation (BPSK, all-zero cw)
        # Use negative mean as we generate logits with definition p(b=1)/p(b=0)

        llr = config.tf_rng.normal(output_shape,
                               mean=-1.*mu_llr,
                               stddev=sigma_llr,
                               dtype=self.rdtype)
        return llr

def llr2mi(llr, s=None, reduce_dims=True):
    # pylint: disable=line-too-long
    r"""Approximates the mutual information based on Log-Likelihood Ratios
    (LLRs).

    This function approximates the mutual information for a given set of ``llr``
    values, assuming an `all-zero codeword` transmission as derived in
    [Hagenauer]_:

    .. math::

        I \approx 1 - \sum \operatorname{log_2} \left( 1 + \operatorname{e}^{-\text{llr}} \right)

    The approximation relies on the `symmetry condition`:

    .. math::

        p(\text{llr}|x=0) = p(\text{llr}|x=1) \cdot \operatorname{exp}(\text{llr})

    For cases where the transmitted codeword is not all-zero, this method
    requires knowledge of the original bit sequence ``s`` to adjust the LLR
    signs accordingly, simulating an all-zero codeword transmission.

    Note that the LLRs are defined as :math:`\frac{p(x=1)}{p(x=0)}`, which
    reverses the sign compared to the solution in [Hagenauer]_.

    Parameters
    ----------
    llr : tf.float
        Tensor of arbitrary shape containing LLR values.

    s : None | tf.float
        Tensor of the same shape as ``llr`` representing the signs of the
        transmitted sequence (assuming BPSK), with values of +/-1.

    reduce_dims : `bool`, (default `True`)
        If `True`,  reduces all dimensions and returns a scalar.
        If False, averages only over the last dimension.

    Returns
    -------
    mi : tf.float
        If ``reduce_dims`` is `True`,  returns a scalar tensor. Otherwise, returns
        a tensor with the same shape as ``llr`` except for the last dimension,
        which is removed. Contains the approximated mutual information.
    """

    if llr.dtype not in (tf.float16, tf.bfloat16, tf.float32, tf.float64):
        raise TypeError("Dtype of llr must be a real-valued float.")

    if s is not None:
        # ensure compatible types
        s = tf.cast(s, llr.dtype)
        # scramble sign as if all-zero cw was transmitted
        llr_zero = tf.multiply(s, llr)
    else:
        llr_zero = llr

    # clip for numerical stability
    llr_zero = tf.clip_by_value(llr_zero, -100., 100.)

    x = log2(1. + tf.exp(1.* llr_zero))
    if reduce_dims:
        x = 1. - tf.reduce_mean(x)
    else:
        x = 1. - tf.reduce_mean(x, axis=-1)
    return x

def j_fun(mu):
     # pylint: disable=line-too-long
    r"""Computes the `J-function`

    The `J-function` relates mutual information to the mean of
    Gaussian-distributed Log-Likelihood Ratios (LLRs) using the Gaussian
    approximation. This function implements the approximation proposed in
    [Brannstrom]_:

    .. math::

        J(\mu) \approx \left( 1 - 2^{H_\text{1}(2\mu)^{H_\text{2}}}\right)^{H_\text{3}}

    where :math:`\mu` represents the mean of the LLR distribution, and the
    constants are defined as :math:`H_\text{1}=0.3073`,
    :math:`H_\text{2}=0.8935`, and :math:`H_\text{3}=1.1064`.

    Input values are clipped to [1e-10, 1000] for numerical stability.
    The output is clipped to a maximum LLR of 20.

    Parameters
    ----------
    mu : tf.float32
        Tensor of arbitrary shape, representing the mean of the LLR values.

    Returns
    -------
    tf.float32
        Tensor of the same shape and dtype as ``mu``, containing the calculated
        mutual information values.

    """

    # input must be positive for numerical stability
    mu = tf.maximum(mu, 1e-10)
    mu = tf.minimum(mu, 1000)

    h1 = 0.3073
    h2 = 0.8935
    h3 = 1.1064
    mi = (1-2**(-h1*(2*mu)**h2))**h3
    return mi

def j_fun_inv(mi):
    # pylint: disable=line-too-long
    r"""Computes the inverse of the `J-function`

    The `J-function` relates mutual information to the mean of
    Gaussian-distributed Log-Likelihood Ratios (LLRs) using the Gaussian
    approximation. This function computes the inverse `J-function` based on the
    approximation proposed in [Brannstrom]_:

    .. math::

        J(\mu) \approx \left( 1 - 2^{H_\text{1}(2\mu)^{H_\text{2}}}\right)^{H_\text{3}}

    where :math:`\mu` is the mean of the LLR distribution, and constants are
    defined as :math:`H_\text{1}=0.3073`, :math:`H_\text{2}=0.8935`, and
    :math:`H_\text{3}=1.1064`.

    Input values are clipped to [1e-10, 1] for numerical stability.
    The output is clipped to a maximum LLR of 20.

    Parameters
    ----------
    mi : tf.float32
        Tensor of arbitrary shape, representing mutual information values.

    Returns
    -------
    tf.float32
        Tensor of the same shape and dtype as ``mi``, containing the computed
        mean values of the LLR distribution.
    """

    # input must be positive for numerical stability
    mi = tf.maximum(mi, 1e-10) # ensure that I>0
    mi = tf.minimum(mi, 1.) # ensure that I=<1

    h1 = 0.3073
    h2 = 0.8935
    h3 = 1.1064
    mu = 0.5*((-1/h1) * log2((1-mi**(1/h3))))**(1/(h2))
    return tf.minimum(mu, 20) # clip the output to mu_max=20

def plot_trajectory(plot, mi_v, mi_c, ebno=None):
    """Plots the trajectory of an EXIT-chart.

    This utility function plots the trajectory of mutual information values in
    an EXIT-chart, based on variable and check node mutual information values.

    Parameters
    ----------
    plot : matplotlib.figure.Figure
        A handle to a matplotlib figure where the trajectory will be plotted.

    mi_v : numpy.ndarray
        Array of floats representing the variable node mutual information
        values.

    mi_c : numpy.ndarray
        Array of floats representing the check node mutual information values.

    ebno : float
        The Eb/No value in dB, used for the legend entry.

    """

    assert (len(mi_v)==len(mi_c)), "mi_v and mi_c must have same length."

    # number of decoding iterations to plot
    iters = np.shape(mi_v)[0] - 1

    x = np.zeros([2*iters])
    y = np.zeros([2*iters])

    #  iterate between VN and CN MI value
    y[1] = mi_v[0]
    for i in range(1, iters):
        x[2*i] = mi_c[i-1]
        y[2*i] = mi_v[i-1]
        x[2*i+1] = mi_c[i-1]
        y[2*i+1] = mi_v[i]

    label_str = "Actual trajectory"

    if ebno is not None:
        label_str += f" @ {ebno} dB"

    # plot trajectory
    plot.plot(x, y, "-", linewidth=3, color="g", label=label_str)

    # and show the legend
    plot.legend(fontsize=18)

def plot_exit_chart(mi_a=None, mi_ev=None, mi_ec=None, title="EXIT-Chart"):
    """Plots an EXIT-chart based on mutual information curves [tenBrinkEXIT]_.

    This utility function generates an EXIT-chart plot. If all inputs are
    `None`, an empty EXIT chart is created; otherwise, mutual information
    curves are plotted.

    Parameters
    ----------
    mi_a : numpy.ndarray, optional
        Array of floats representing the a priori mutual information.

    mi_v : numpy.ndarray, optional
        Array of floats representing the variable node mutual information.

    mi_c : numpy.ndarray, optional
        Array of floats representing the check node mutual information.

    title : str
        Title of the EXIT chart.

    Returns
    -------
    matplotlib.figure.Figure
        A handle to the generated matplotlib figure.
    """

    assert isinstance(title, str), "title must be a string."

    if not (mi_ev is None and mi_ec is None):
        if mi_a is None:
            raise ValueError("mi_a cannot be None if mi_e is provided.")

    if mi_ev is not None:
        assert (len(mi_a)==len(mi_ev)), "mi_a and mi_ev must have same length."
    if mi_ec is not None:
        assert (len(mi_a)==len(mi_ec)), "mi_a and mi_ec must have same length."

    plt.figure(figsize=(10,10))
    plt.title(title, fontsize=25)
    plt.xlabel("$I_{a}^v$, $I_{e}^c$", fontsize=25)
    plt.ylabel("$I_{e}^v$, $I_{a}^c$", fontsize=25)
    plt.grid(visible=True, which='major')

    # for MI, the x,y limits are always (0,1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # and plot EXIT curves
    if mi_ec is not None:
        plt.plot(mi_ec, mi_a, "r", linewidth=3, label="Check node")
        plt.legend()
    if mi_ev is not None:
        plt.plot(mi_a, mi_ev, "b", linewidth=3, label="Variable node")
        plt.legend()
    return plt

def get_exit_analytic(pcm, ebno_db):
    """Calculates analytic EXIT curves for a given parity-check matrix.

    This function extracts the degree profile from the provided parity-check
    matrix ``pcm`` and calculates the EXIT (Extrinsic Information Transfer)
    curves for variable nodes (VN) and check nodes (CN) decoders. Note that this
    approach relies on asymptotic analysis, which requires a sufficiently large
    codeword length for accurate predictions.

    It assumes transmission over an AWGN channel with BPSK modulation at an
    SNR specified by ``ebno_db``. For more details on the equations, see
    [tenBrink]_ and [tenBrinkEXIT]_.

    Parameters
    ----------
    pcm : numpy.ndarray
        The parity-check matrix.

    ebno_db : float
        Channel SNR in dB.

    Returns
    -------
    mi_a : numpy.ndarray
        Array of floats containing the a priori mutual information.

    mi_ev : numpy.ndarray
        Array of floats containing the extrinsic mutual information of the
        variable node decoder for each ``mi_a`` value.

    mi_ec : numpy.ndarray
        Array of floats containing the extrinsic mutual information of the
        check node decoder for each ``mi_a`` value.

    Notes
    -----
    This function assumes random, unstructured parity-check matrices. Thus,
    applying it to parity-check matrices with specific structures or constraints
    may result in inaccurate EXIT predictions. Additionally, this function is
    based on asymptotic properties and performs best with large parity-check
    matrices. For more information, refer to [tenBrink]_.
    """

    # calc coderate
    n = pcm.shape[1]
    k = n - pcm.shape[0]
    coderate = k/n

    # calc mean and noise_var of Gaussian distributed LLRs for given channel SNR
    ebno = 10**(ebno_db/10)
    snr = ebno*coderate
    noise_var = 1/(2*snr)

    # For BiAWGN channels the LLRs follow a Gaussian distr. as given below [1]
    sigma_llr = np.sqrt(4 / noise_var)
    mu_llr = sigma_llr**2  / 2

    # calculate max node degree
    # "+1" as the array indices later directly denote the node degrees and we
    # have to account the array start at position 0 (i.e., we need one more
    # element)
    c_max = int(np.max(np.sum(pcm, axis=1)) + 1 )
    v_max = int(np.max(np.sum(pcm, axis=0)) + 1 )

    # calculate degree profile (node perspective)
    c = np.histogram(np.sum(pcm, axis=1),
                     bins=c_max,
                     range=(0, c_max),
                     density=False)[0]

    v = np.histogram(np.sum(pcm, axis=0),
                     bins=v_max,
                     range=(0, v_max),
                     density=False)[0]

    # calculate degrees from edge perspective
    r = np.zeros([c_max])
    for i in range(1,c_max):
        r[i] = (i-1)*c[i]
    r = r / np.sum(r)
    l = np.zeros([v_max])
    for i in range(1,v_max):
        l[i] = (i-1)*v[i]
    l = l / np.sum(l)

    mi_a = np.arange(0.002, 0.998, 0.001) # quantize Ia with 0.01 resolution

    # Exit function of check node update
    mi_ec = np.zeros_like(mi_a)
    for i in range(1, c_max):
        mi_ec += r[i] * j_fun((i-1.) * j_fun_inv(1 - mi_a).numpy()).numpy()
    mi_ec = 1 - mi_ec

    # Exit function of variable node update
    mi_ev = np.zeros_like(mi_a)
    for i in range(1, v_max):
        mi_ev += l[i] * j_fun(mu_llr + (i-1.) * j_fun_inv(mi_a)).numpy()

    return mi_a, mi_ev, mi_ec

def load_parity_check_examples(pcm_id, verbose=False):
    # pylint: disable=line-too-long
    """Loads parity-check matrices of built-in example codes.

    This utility function loads predefined example codes, including Hamming,
    BCH, and LDPC codes. The following codes are available:

    - ``pcm_id`` =0 : `(7,4)` Hamming code with `k=4` information bits and `n=7` codeword length.

    - ``pcm_id`` =1 : `(63,45)` BCH code with `k=45` information bits and `n=63` codeword length.

    - ``pcm_id`` =2 : `(127,106)` BCH code with `k=106` information bits and `n=127` codeword length.

    - ``pcm_id`` =3 : Random LDPC code with variable node degree 3 and check node degree 6, with `k=50` information bits and `n=100` codeword length.

    - ``pcm_id`` =4 : 802.11n LDPC code with `k=324` information bits and `n=648` codeword length.

    Parameters
    ----------
    pcm_id : int
        An integer identifying the code matrix to load.

    verbose : `bool`, (default `False`)
        If `True`,  prints the code parameters.

    Returns
    -------
    pcm : numpy.ndarray
        Array containing the parity-check matrix (values are `0` and `1`).

    k : int
        Number of information bits.

    n : int
        Number of codeword bits.

    coderate : float
        Code rate, assuming full rank of the parity-check matrix.

    """

    source = files(codes).joinpath("example_codes.npy")
    with as_file(source) as code:
        pcms = np.load(code, allow_pickle=True)

    pcm = np.array(pcms[pcm_id]) # load parity-check matrix
    n = int(pcm.shape[1]) # number of codeword bits (codeword length)
    k = int(n - pcm.shape[0]) # number of information bits k per codeword
    coderate = k / n

    if verbose:
        print(f"\nn: {n}, k: {k}, coderate: {coderate:.3f}")
    return pcm, k, n, coderate

def bin2int(arr):
    """Converts a binary array to its integer representation.

    This function converts an iterable binary array to its equivalent integer.
    For example, `[1, 0, 1]` is converted to `5`.

    Parameters
    ----------
    arr : iterable of int or float
        An iterable that contains binary values (0's and 1's).

    Returns
    -------
    int
        The integer representation of the binary array.
    """

    if len(arr) == 0: return None
    return int(''.join([str(x) for x in arr]), 2)

def bin2int_tf(arr):
    """
    Converts a binary tensor to an integer tensor.

    Interprets the binary representation across the last dimension of ``arr``,
    from most significant to least significant bit. For example, an input
    of `[0, 1, 1]` is converted to `3`.

    Parameters
    ----------
    arr : tf.Tensor
        Tensor of integers or floats containing binary values (0's and 1's)
        along the last dimension.

    Returns
    -------
    tf.Tensor
        Tensor with the integer representation of ``arr``.
    """

    len_ = tf.shape(arr)[-1]
    shifts = tf.range(len_-1,-1,-1)

    # (2**len_-1)*arr[0] +... 2*arr[len_-2] + 1*arr[len_-1]
    op = tf.reduce_sum(tf.bitwise.left_shift(arr, shifts), axis=-1)

    return op

def int2bin(num, length):
    # pylint: disable=line-too-long
    """
    Converts an integer to a binary list of specified length.

    This function converts an integer ``num`` to a list of 0's and 1's, with
    the binary representation padded to a length of ``length``. Both ``num``
    and ``length`` must be non-negative.

    For example:

    - If ``num = 5`` and ``length = 4``, the output is `[0, 1, 0, 1]`.

    - If ``num = 12`` and ``length = 3``, the output is `[1, 0, 0]` (truncated to length).

    Parameters
    ----------
    num : int
        The integer to be converted into binary representation.

    length : int
        The desired length of the binary output list.

    Returns
    -------
    list of int
        A list of 0's and 1's representing the binary form of ``num``, padded
        or truncated to a length of ``length``.
    """
    assert num >= 0,  "Input integer should be non-negative"
    assert length >= 0,  "length should be non-negative"

    bin_ = format(num, f'0{length}b')
    binary_vals = [int(x) for x in bin_[-length:]] if length else []
    return binary_vals

def int2bin_tf(ints, length):
    """
    Converts an integer tensor to a binary tensor with specified bit length.

    This function converts each integer in the input tensor ``ints`` to a binary
    representation, with an additional dimension of size ``length`` added at the
    end to represent the binary bits. The ``length`` parameter must be
    non-negative.

    Parameters
    ----------
    ints : tf.Tensor
        Tensor of arbitrary shape `[..., k]` containing integers to be
        converted into binary representation.

    length : int
        An integer specifying the bit length of the binary representation for
        each integer.

    Returns
    -------
    tf.Tensor
        A tensor of the same shape as ``ints`` with an additional dimension of
        size ``length`` at the end, i.e., shape `[..., k, length]`.
        This tensor contains the binary representation of each integer in
        ``ints``.
    """
    assert length >= 0

    shifts = tf.range(length-1, -1, delta=-1)
    bits = tf.math.floormod(
        tf.bitwise.right_shift(tf.expand_dims(ints, -1), shifts), 2)
    return bits

def alist2mat(alist, verbose=True):
    # pylint: disable=line-too-long
    r"""Converts an `alist` [MacKay]_ code definition to a NumPy parity-check matrix.

    This function converts an `alist` format representation of a code's
    parity-check matrix to a NumPy array. Many example codes in `alist`
    format can be found in [UniKL]_.

    About the `alist` format (see [MacKay]_ for details):

        - Row 1: Defines the parity-check matrix dimensions `m x n`.
        - Row 2: Contains two integers, `max_CN_degree` and `max_VN_degree`.
        - Row 3: Lists the degrees of all `n` variable nodes (columns).
        - Row 4: Lists the degrees of all `m` check nodes (rows).
        - Next `n` rows: Non-zero entries of each column, zero-padded as needed.
        - Following `m` rows: Non-zero entries of each row, zero-padded as needed.

    Parameters
    ----------
    alist : list
        Nested list in `alist` format [MacKay]_ representing the parity-check matrix.

    verbose : `bool`, (default `True`)
        If `True`,  prints the code parameters.

    Returns
    -------
    pcm : numpy.ndarray
        NumPy array of shape `[n - k, n]` representing the parity-check
        matrix.

    k : int
        Number of information bits.

    n : int
        Number of codeword bits.

    coderate : float
        Code rate of the code.

    Notes
    -----
    Use :class:`~sionna.phy.fec.utils.load_alist` to import an `alist` from a text
    file.

    Example
    -------
    The following code snippet imports an `alist` from a file called
    ``filename``:

    .. code-block:: python

        al = load_alist(path=filename)
        pcm, k, n, coderate = alist2mat(al)
    """

    assert len(alist)>4, "Invalid alist format."

    n = alist[0][0]
    m = alist[0][1]
    v_max = alist[1][0]
    c_max = alist[1][1]
    k = n - m
    coderate = k / n

    vn_profile = alist[2]
    cn_profile = alist[3]

    # plausibility checks
    assert np.sum(vn_profile)==np.sum(cn_profile), "Invalid alist format."
    assert np.max(vn_profile)==v_max, "Invalid alist format."
    assert np.max(cn_profile)==c_max, "Invalid alist format."

    if len(alist)==len(vn_profile)+4:
        print("Note: .alist does not contain (redundant) CN perspective.")
        print("Recovering parity-check matrix from VN only.")
        print("Please verify the correctness of the results manually.")
        vn_only = True
    else:
        assert len(alist)==len(vn_profile) + len(cn_profile) + 4, \
                                                "Invalid alist format."
        vn_only = False

    pcm = np.zeros((m,n))
    num_edges = 0 # count number of edges

    for idx_v in range(n):
        for idx_i in range(vn_profile[idx_v]):
            # first 4 rows of alist contain meta information
            idx_c = alist[4+idx_v][idx_i]-1 # "-1" as this is python
            pcm[idx_c, idx_v] = 1
            num_edges += 1 # count number of edges (=each non-zero entry)

    # validate results from CN perspective
    if not vn_only:
        for idx_c in range(m):
            for idx_i in range(cn_profile[idx_c]):
                # first 4 rows of alist contain meta information
                # following n rows contained VN perspective
                idx_v = alist[4+n+idx_c][idx_i]-1 # "-1" as this is python
                assert pcm[idx_c, idx_v]==1 # entry must already exist

    if verbose:
        print("Number of variable nodes (columns): ", n)
        print("Number of check nodes (rows): ", m)
        print("Number of information bits per cw: ", k)
        print("Number edges: ", num_edges)
        print("Max. VN degree: ", v_max)
        print("Max. CN degree: ", c_max)
        print("VN degree: ", vn_profile)
        print("CN degree: ", cn_profile)

    return pcm, k, n, coderate

def load_alist(path):
    """Reads an `alist` file and returns a nested list describing a code's
    parity-check matrix.

    This function reads a file in `alist` format [MacKay]_ and returns a nested
    list representing the parity-check matrix. Numerous example codes in
    `alist` format are available in [UniKL]_.

    Parameters
    ----------
    path : str
        Path to the `alist` file to be loaded.

    Returns
    -------
    list
        A nested list containing the imported `alist` data representing the
        parity-check matrix.
    """

    alist = []
    with open(path, "r") as reader: # pylint: disable=unspecified-encoding
        # read list line by line (different length)
        for line in reader:
            l = []
            # append all entries
            for word in line.split():
                l.append(int(word))
            if l: # ignore empty lines
                alist.append(l)

    return alist

def make_systematic(mat, is_pcm=False):
    r"""Converts a binary matrix to its systematic form.

    This function transforms a binary matrix into systematic form, where the
    first `k` columns (or last `k` columns if ``is_pcm`` is True) form an
    identity matrix.

    Parameters
    ----------
    mat : numpy.ndarray
        Binary matrix of shape `[k, n]` to be transformed to systematic form.

    is_pcm : `bool`, (default `False`)
        If `True`,  ``mat`` is treated as a parity-check matrix,
        and the identity part will be placed in the last `k` columns.

    Returns
    -------
    mat_sys : numpy.ndarray
        Binary matrix in systematic form, where the first `k` columns (or last
        `k` columns if ``is_pcm`` is True) form the identity matrix.

    column_swaps : list of tuple of int
        A list of integer tuples representing the column swaps performed to
        achieve the systematic form, in order of execution.

    Notes
    -----
    This function may swap columns of the input matrix to achieve systematic
    form. As a result, the output matrix represents a permuted version of the
    code, defined by the ``column_swaps`` list. To revert to the original
    column order, apply the inverse permutation in reverse order of the swaps.

    If ``is_pcm`` is `True`,  indicating a parity-check matrix, the identity
    matrix portion will be arranged in the last `k` columns.
    """

    m = mat.shape[0]
    n = mat.shape[1]

    assert m<=n, "Invalid matrix dimensions."

    # check for all-zero columns (=unchecked nodes)
    if is_pcm:
        c_node_deg = np.sum(mat, axis=0)
        if np.any(c_node_deg==0):
            warnings.warn("All-zero column in parity-check matrix detected. " \
                "It seems as if the code contains unprotected nodes.")

    mat = np.copy(mat)
    column_swaps = [] # store all column swaps

    # convert to bool for faster arithmetics
    mat = mat.astype(bool)

    # bring in upper triangular form
    for idx_c in range(m):
        success = mat[idx_c, idx_c]
        if not success: # skip if leading "1" already occurred
            # step 1: find next leading "1"
            for idx_r in range(idx_c+1,m):
                # skip if entry is "0"
                if mat[idx_r, idx_c]:
                    mat[[idx_c, idx_r]] = mat[[idx_r, idx_c]] # swap rows
                    success = True
                    break

        # Could not find "1"-entry for column idx_c
        # => swap with columns from non-sys part
        # The task is to find a column with index idx_cc that has a "1" at
        # row idx_c
        if not success:
            for idx_cc in range(idx_c+1, n):
                if mat[idx_c, idx_cc]:
                    # swap columns
                    mat[:,[idx_c, idx_cc]] = mat[:,[idx_cc, idx_c]]
                    column_swaps.append([idx_c, idx_cc])
                    success = True
                    break

        if not success:
            raise ValueError("Could not succeed; mat is not full rank?")

        # we can now assume a leading "1" at row idx_c
        for idx_r in range(idx_c+1, m):
            if mat[idx_r, idx_c]:
                mat[idx_r,:] ^= mat[idx_c,:] # bin. add of row idx_c to idx_r

    # remove upper triangle part in inverse order
    for idx_c in range(m-1, -1, -1):
        for idx_r in range(idx_c-1, -1, -1):
            if mat[idx_r, idx_c]:
                mat[idx_r,:] ^= mat[idx_c,:] # bin. add of row idx_c to idx_r

    # verify results
    assert np.array_equal(mat[:,:m], np.eye(m)), \
                            "Internal error, could not find systematic matrix."

    # bring identity part to end of matrix if parity-check matrix is provided
    if is_pcm:
        # individual column swaps instead of copying entire block
        # this simplifies the tracking of column swaps.
        for i in range(n-1, (n-1)-m, -1):
            j = i - (n-m)
            mat[:,[i, j]] = mat[:,[j, i]]
            column_swaps.append([i, j])

    # return integer array
    mat = mat.astype(int)
    return mat, column_swaps

def gm2pcm(gm, verify_results=True):
    r"""Generates the parity-check matrix for a given generator matrix.

    This function converts the generator matrix ``gm`` (denoted as
    :math:`\mathbf{G}`) to systematic form and uses the following relationship
    to compute the parity-check matrix :math:`\mathbf{H}` over GF(2):

    .. math::

        \mathbf{G} = [\mathbf{I} |  \mathbf{M}]
        \Rightarrow \mathbf{H} = [\mathbf{M}^T | \mathbf{I}]. \tag{1}

    This is derived from the requirement for an all-zero syndrome, such that:

    .. math::

        \mathbf{H} \mathbf{c}^T = \mathbf{H} * (\mathbf{u} * \mathbf{G})^T =
        \mathbf{H} * \mathbf{G}^T * \mathbf{u}^T = \mathbf{0},

    where :math:`\mathbf{c}` represents an arbitrary codeword and
    :math:`\mathbf{u}` the corresponding information bits.

    This leads to:

    .. math::

        \mathbf{G} * \mathbf{H}^T = \mathbf{0}. \tag{2}

    It can be seen that (1) satisfies (2), as in GF(2):

    .. math::

        [\mathbf{I} |  \mathbf{M}] * [\mathbf{M}^T | \mathbf{I}]^T
         = \mathbf{M} + \mathbf{M} = \mathbf{0}.

    Parameters
    ----------
    gm : numpy.ndarray
        Binary generator matrix of shape `[k, n]`.

    verify_results : `bool`, (default `True`)
        If `True`, verifies that the generated parity-check
        matrix is orthogonal to the generator matrix in GF(2).

    Returns
    -------
    numpy.ndarray
        Binary parity-check matrix of shape `[n - k, n]`.

    Notes
    -----
    This function requires ``gm`` to have full rank. An error is raised if
    ``gm`` does not meet this requirement.
    """

    k = gm.shape[0]
    n = gm.shape[1]

    assert k<n, "Invalid matrix dimensions."

    # bring gm in systematic form
    gm_sys, c_swaps = make_systematic(gm, is_pcm=False)

    m_mat = np.transpose(np.copy(gm_sys[:,-(n-k):]))
    i_mat = np.eye(n-k)

    pcm = np.concatenate((m_mat, i_mat), axis=1)

    # undo column swaps
    for l in c_swaps[::-1]: # reverse ordering when going through list
        pcm[:,[l[0], l[1]]] = pcm[:,[l[1], l[0]]] # swap columns

    if verify_results:
        assert verify_gm_pcm(gm=gm, pcm=pcm), \
            "Resulting parity-check matrix does not match to generator matrix."

    return pcm

def pcm2gm(pcm, verify_results=True):
    r"""Generates the generator matrix for a given parity-check matrix.

    This function converts the parity-check matrix ``pcm`` (denoted as
    :math:`\mathbf{H}`) to systematic form and uses the following relationship
    to compute the generator matrix :math:`\mathbf{G}` over GF(2):

    .. math::

        \mathbf{G} = [\mathbf{I} | \mathbf{M}]
        \Rightarrow \mathbf{H} = [\mathbf{M}^T | \mathbf{I}]. \tag{1}

    This derivation is based on the requirement for an all-zero syndrome:

    .. math::

        \mathbf{H} \mathbf{c}^T = \mathbf{H} * (\mathbf{u} * \mathbf{G})^T =
        \mathbf{H} * \mathbf{G}^T * \mathbf{u}^T = \mathbf{0},

    where :math:`\mathbf{c}` represents an arbitrary codeword and
    :math:`\mathbf{u}` the corresponding information bits.

    This leads to:

    .. math::

        \mathbf{G} * \mathbf{H}^T = \mathbf{0}. \tag{2}

    It can be shown that (1) satisfies (2), as in GF(2):

    .. math::

        [\mathbf{I} | \mathbf{M}] * [\mathbf{M}^T | \mathbf{I}]^T
         = \mathbf{M} + \mathbf{M} = \mathbf{0}.

    Parameters
    ----------
    pcm : numpy.ndarray
        Binary parity-check matrix of shape `[n - k, n]`.

    verify_results : `bool`, (default `True`)
        If `True`,  verifies that the generated generator matrix
        is orthogonal to the parity-check matrix in GF(2).

    Returns
    -------
    numpy.ndarray
        Binary generator matrix of shape `[k, n]`.

    Notes
    -----
    This function requires ``pcm`` to have full rank. An error is raised if
    ``pcm`` does not meet this requirement.
    """

    n = pcm.shape[1]
    k = n - pcm.shape[0]

    assert k<n, "Invalid matrix dimensions."

    # bring pcm in systematic form
    pcm_sys, c_swaps = make_systematic(pcm, is_pcm=True)

    m_mat = np.transpose(np.copy(pcm_sys[:,:k]))
    i_mat = np.eye(k)
    gm = np.concatenate((i_mat, m_mat), axis=1)

    # undo column swaps
    for l in c_swaps[::-1]: # reverse ordering when going through list
        gm[:,[l[0], l[1]]] = gm[:,[l[1], l[0]]] # swap columns

    if verify_results:
        assert verify_gm_pcm(gm=gm, pcm=pcm), \
            "Resulting parity-check matrix does not match to generator matrix."
    return gm

def verify_gm_pcm(gm, pcm):
    r"""Verifies that the generator matrix :math:`\mathbf{G}` (``gm``) and
    parity-check matrix :math:`\mathbf{H}` (``pcm``) are orthogonal in GF(2).

    For a valid code with an all-zero syndrome, the following condition must
    hold:

    .. math::

        \mathbf{H} \mathbf{c}^T = \mathbf{H} * (\mathbf{u} * \mathbf{G})^T =
        \mathbf{H} * \mathbf{G}^T * \mathbf{u}^T = \mathbf{0},

    where :math:`\mathbf{c}` represents an arbitrary codeword and
    :math:`\mathbf{u}` the corresponding information bits.

    Since :math:`\mathbf{u}` can be arbitrary, this leads to the condition:

    .. math::
        \mathbf{H} * \mathbf{G}^T = \mathbf{0}.

    Parameters
    ----------
    gm : numpy.ndarray
        Binary generator matrix of shape `[k, n]`.

    pcm : numpy.ndarray
        Binary parity-check matrix of shape `[n - k, n]`.

    Returns
    -------
    bool
        `True` if ``gm`` and ``pcm`` define a valid pair of orthogonal
        parity-check and generator matrices in GF(2).
    """

    # check for valid dimensions
    k = gm.shape[0]
    n = gm.shape[1]

    n_pcm = pcm.shape[1]
    k_pcm = n_pcm - pcm.shape[0]

    assert k==k_pcm, "Inconsistent shape of gm and pcm."
    assert n==n_pcm, "Inconsistent shape of gm and pcm."

    # check that both matrices are binary
    assert ((gm==0) | (gm==1)).all(), "gm is not binary."
    assert ((pcm==0) | (pcm==1)).all(), "pcm is not binary."

    # check for zero syndrome
    s = np.mod(np.matmul(pcm, np.transpose(gm)), 2) # mod2 to account for GF(2)
    return np.sum(s)==0 # Check for Non-zero syndrome of H*G'

def generate_reg_ldpc(v, c, n, allow_flex_len=True, verbose=True):
    r"""Generates a random regular (v, c) LDPC code.

    This function generates a random Low-Density Parity-Check (LDPC)
    parity-check matrix of length ``n`` where each variable node (VN) has
    degree ``v`` and each check node (CN) has degree ``c``. Note that the
    generated LDPC code is not optimized to avoid short cycles, which may
    result in a non-negligible error floor. For encoding, the :class:`~sionna.
    fec.utils.LinearEncoder` block can be used, but the construction does not
    guarantee that the parity-check matrix (``pcm``) has full rank.

    Parameters
    ----------
    v : int
        Desired degree of each variable node (VN).

    c : int
        Desired degree of each check node (CN).

    n : int
        Desired codeword length.

    allow_flex_len : `bool`, (default `True`)
        If `True`,  the resulting codeword length may be
        slightly increased to meet the degree requirements.

    verbose : `bool`, (default `True`)
        If `True`,  prints code parameters.

    Returns
    -------
    pcm : numpy.ndarray
        Parity-check matrix of shape `[n - k, n]`.

    k : int
        Number of information bits per codeword.

    n : int
        Number of codeword bits.

    coderate : float
        Code rate of the LDPC code.

    Notes
    -----
    This algorithm is designed only for regular node degrees. To achieve
    state-of-the-art bit-error-rate performance, optimizing irregular degree
    profiles is usually necessary (see [tenBrink]_).
    """

    # check input values for consistency
    assert isinstance(allow_flex_len, bool), \
                                    'allow_flex_len must be bool.'

    # allow slight change in n to keep num edges
    # from CN and VN perspective an integer
    if allow_flex_len:
        for n_mod in range(n, n+2*c):
            if np.mod((v/c) * n_mod, 1.)==0:
                n = n_mod
                if verbose:
                    print("Setting n to: ", n)
                break

    # calculate number of nodes
    coderate = 1 - (v/c)
    n_v = n
    n_c = int((v/c) * n)
    k = n_v - n_c

    # generate sockets
    v_socks = np.tile(np.arange(n_v),v)
    c_socks = np.tile(np.arange(n_c),c)
    if verbose:
        print("Number of edges (VN perspective): ", len(v_socks))
        print("Number of edges (CN perspective): ", len(c_socks))
    assert len(v_socks) == len(c_socks), "Number of edges from VN and CN " \
        "perspective does not match. Consider to (slightly) change n."

    # apply random permutations
    config.np_rng.shuffle(v_socks)
    config.np_rng.shuffle(c_socks)

    # and generate matrix
    pcm = np.zeros([n_c, n_v])

    idx = 0
    shuffle_max = 200 # stop if no success
    shuffle_counter = 0
    cont = True
    while cont:
        # if edge is available, take it
        if pcm[c_socks[idx],v_socks[idx]]==0:
            pcm[c_socks[idx],v_socks[idx]] = 1
            idx +=1 # and go to next socket
            shuffle_counter = 0 # reset counter
            if idx==len(v_socks):
                cont = False
        else: # shuffle sockets
            shuffle_counter+=1
            if shuffle_counter<shuffle_max:
                config.np_rng.shuffle(v_socks[idx:])
                config.np_rng.shuffle(c_socks[idx:])
            else:
                print("Stopping - no solution found!")
                cont=False

    v_deg = np.sum(pcm, axis=0)
    c_deg = np.sum(pcm, axis=1)

    assert((v_deg==v).all()), "VN degree not always v."
    assert((c_deg==c).all()), "CN degree not always c."

    if verbose:
        print(f"Generated regular ({v},{c}) LDPC code of length n={n}")
        print(f"Code rate is r={coderate:.3f}.")
        plt.spy(pcm)

    return pcm, k, n, coderate


def int_mod_2(x):
    r"""Modulo 2 operation and implicit rounding for floating point inputs

    Performs more efficient modulo-2 operation for integer inputs.
    Uses `tf.math.floormod` for floating inputs and applies implicit rounding
    for floating point inputs.

    Parameters
    ----------
    x : tf.Tensor
        Tensor to which the modulo 2 operation is applied.

    Returns
    -------
    x_mod: tf.Tensor
        Binary Tensor containing the result of the modulo 2 operation, with the
        same shape as ``x``.
    """

    if x.dtype in (tf.int8, tf.int32, tf.int64):
        x_mod = tf.bitwise.bitwise_and(x, tf.constant(1, x.dtype))
    else:
        # round to next integer
        x_ = tf.math.abs(tf.math.round(x))
        # tf.math.mod seems deprecated
        x_mod = tf.math.floormod(x_, 2)
    return x_mod

