#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""Blocks for (de)mapping, constellation class, and utility functions"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sionna.phy.block import Block, Object
from sionna.phy.config import config, dtypes
from sionna.phy.utils import expand_to_rank, flatten_last_dims,\
                             hard_decisions, split_dim

def pam_gray(b):
    # pylint: disable=line-too-long
    r"""Maps a vector of bits to a PAM constellation points with Gray labeling

    This recursive function maps a binary vector to Gray-labelled PAM
    constellation points. It can be used to generated QAM constellations.
    The constellation is not normalized.

    Input
    -----
    b : [n], `np.array`
        Tensor with with binary entries

    Output
    ------
    : `signed int`
        PAM constellation point taking values in
        :math:`\{\pm 1,\pm 3,\dots,\pm (2^n-1)\}`.

    Note
    ----
    This algorithm is a recursive implementation of the expressions found in
    Section 5.1 of [3GPPTS38211]_. It is used in the 5G standard.
    """ # pylint: disable=C0301

    if len(b)>1:
        return (1-2*b[0])*(2**len(b[1:]) - pam_gray(b[1:]))
    return 1-2*b[0]

def qam(num_bits_per_symbol, normalize=True, precision=None):
    r"""Generates a QAM constellation

    This function generates a complex-valued vector, where each element is
    a constellation point of an M-ary QAM constellation. The bit
    label of the ``n`` th point is given by the length-``num_bits_per_symbol``
    binary represenation of ``n``.

    Input
    -----
    num_bits_per_symbol : `int`
        Number of bits per constellation point.
        Must be a multiple of two, e.g., 2, 4, 6, 8, etc.

    normalize: `bool`, (default `True`)
        If `True`, the constellation is normalized to have unit power.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Output
    ------
    : [2**num_bits_per_symbol], np.complex64
        QAM constellation points

    Note
    ----
    The bit label of the nth constellation point is given by the binary
    representation of its position within the array and can be obtained
    through ``np.binary_repr(n, num_bits_per_symbol)``.


    The normalization factor of a QAM constellation is given in
    closed-form as:

    .. math::
        \sqrt{\frac{1}{2^{n-2}}\sum_{i=1}^{2^{n-1}}(2i-1)^2}

    where :math:`n= \text{num_bits_per_symbol}/2` is the number of bits
    per dimension.

    This algorithm is a recursive implementation of the expressions found in
    Section 5.1 of [3GPPTS38211]_. It is used in the 5G standard.
    """ # pylint: disable=C0301

    try:
        assert num_bits_per_symbol % 2 == 0 # is even
        assert num_bits_per_symbol >0 # is larger than zero
    except AssertionError as error:
        raise ValueError("num_bits_per_symbol must be a multiple of 2") \
        from error
    assert isinstance(normalize, bool), "normalize must be boolean"

    if precision is None:
        rdtype = config.np_rdtype
        cdtype = config.np_cdtype
    else:
        rdtype = dtypes[precision]["np"]["rdtype"]
        cdtype = dtypes[precision]["np"]["cdtype"]

    # Build constellation by iterating through all points
    c = np.zeros([2**num_bits_per_symbol], dtype=cdtype)
    for i in range(0, 2**num_bits_per_symbol):
        b = np.array(list(np.binary_repr(i,num_bits_per_symbol)),
                     dtype=np.int32)
        c[i] = pam_gray(b[0::2]) + 1j*pam_gray(b[1::2]) # PAM in each dimension

    if normalize: # Normalize to unit energy
        n = int(num_bits_per_symbol/2)
        qam_var = 1/(2**(n-2))*np.sum(np.linspace(1,2**n-1, 2**(n-1),
                                                  dtype=rdtype)**2)
        c /= np.sqrt(qam_var)
    return c

def pam(num_bits_per_symbol, normalize=True, precision=None):
    r"""Generates a PAM constellation

    This function generates a real-valued vector, where each element is
    a constellation point of an M-ary PAM constellation. The bit
    label of the ``n`` th point is given by the length-``num_bits_per_symbol``
    binary represenation of ``n``.

    Input
    -----
    num_bits_per_symbol : `int`
        Number of bits per constellation point.
        Must be positive.

    normalize: `bool`, (default `True`)
        If `True`, the constellation is normalized to have unit power.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Output
    ------
    : [2**num_bits_per_symbol], `np.float32`
        PAM constellation symbols

    Note
    ----
    The bit label of the nth constellation point is given by the binary
    representation of its position within the array and can be obtained
    through ``np.binary_repr(n, num_bits_per_symbol)``.


    The normalization factor of a PAM constellation is given in
    closed-form as:

    .. math::
        \sqrt{\frac{1}{2^{n-1}}\sum_{i=1}^{2^{n-1}}(2i-1)^2}

    where :math:`n= \text{num_bits_per_symbol}` is the number of bits
    per symbol.

    This algorithm is a recursive implementation of the expressions found in
    Section 5.1 of [3GPPTS38211]_. It is used in the 5G standard.
    """ # pylint: disable=C0301

    try:
        assert num_bits_per_symbol >0 # is larger than zero
    except AssertionError as error:
        raise ValueError("num_bits_per_symbol must be positive") \
        from error
    assert isinstance(normalize, bool), "normalize must be boolean"

    if precision is None:
        rdtype = config.np_rdtype
        cdtype = config.np_cdtype
    else:
        rdtype = dtypes[precision]["np"]["rdtype"]
        cdtype = dtypes[precision]["np"]["cdtype"]

    # Build constellation by iterating through all points
    c = np.zeros([2**num_bits_per_symbol], dtype=cdtype)
    for i in range(0, 2**num_bits_per_symbol):
        b = np.array(list(np.binary_repr(i,num_bits_per_symbol)),
                     dtype=np.int32)
        c[i] = pam_gray(b)

    if normalize: # Normalize to unit energy
        n = int(num_bits_per_symbol)
        pam_var = 1/(2**(n-1))*np.sum(np.linspace(1,2**n-1, 2**(n-1),
                                                  dtype=rdtype)**2)
        c /= np.sqrt(pam_var)
    return c

class Constellation(Block):
    # pylint: disable=line-too-long
    r"""
    Constellation that can be used by a (de-)mapper

    This class defines a constellation, i.e., a complex-valued vector of
    constellation points. The binary
    representation of the index of an element of this vector corresponds
    to the bit label of the constellation point. This implicit bit
    labeling is used by the :class:`~sionna.phy.mapping.Mapper` and
    :class:`~sionna.phy.mapping.Demapper`.

    Parameters
    ----------
    constellation_type : "qam" | "pam" | "custom"
        For "custom", the constellation ``points`` must be provided.

    num_bits_per_symbol : int
        Number of bits per constellation symbol, e.g., 4 for QAM16.

    points : `None` (default) | [2**num_bits_per_symbol], `array_like`
        Custom constellation points

    normalize : `bool`, (default `False`)
        If `True`, the constellation is normalized to have unit power.
        Only applies to custom constellations.

    center : `bool`, (default `False`)
        If `True`, the constellation is ensured to have zero mean.
        Only applies to custom constellations.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    : `None`

    Output
    ------
    : [2**num_bits_per_symbol], `tf.complex`
        (Possibly) centered and normalized constellation points
    """
    # pylint: enable=C0301
    def __init__(self,
                 constellation_type,
                 num_bits_per_symbol,
                 points=None,
                 normalize=False,
                 center=False,
                 precision=None,
                 **kwargs):
        super().__init__(precision=precision, **kwargs)

        if constellation_type not in ("qam", "pam", "custom"):
            raise ValueError("Wrong `constellation_type` {constellation_type}")
        self._constellation_type = constellation_type

        if num_bits_per_symbol is None:
            raise ValueError("No value for `num_bits_per_symbol`")
        n = num_bits_per_symbol
        if (n <= 0) or (n%1 != 0):
            raise ValueError("`num_bits_per_symbol` must be a" +
                             " positive integer")
        if self.constellation_type == "qam":
            if n%2 != 0:
                raise ValueError("`num_bits_per_symbol` must" +
                                 "be a positive integer multiple of 2")
        self._num_bits_per_symbol = int(n)
        self._num_points = 2**self._num_bits_per_symbol

        self.normalize = normalize
        self.center = center

        if (points is not None) and (constellation_type != "custom"):
            raise ValueError("`points` can only be provided for" +
                             " `constellation_type`='custom'")
        elif (points is None) and (constellation_type == "custom"):
            raise ValueError("You must provide a value for `points`")

        self._points = None
        if self.constellation_type == "qam":
            points = qam(self.num_bits_per_symbol,
                         normalize=True,
                         precision=precision)
        elif self.constellation_type == "pam":
            points = pam(self.num_bits_per_symbol,
                         normalize=True,
                         precision=precision)
        self.points = points

    @property
    def constellation_type(self):
        """
        "qam" | "pam" | "custom" : Constellation type"""
        return self._constellation_type

    @property
    def num_bits_per_symbol(self):
        """
        `int` : Number of bits per symbol"""
        return self._num_bits_per_symbol

    @property
    def num_points(self):
        """
        `int` : Number of constellation points"""
        return self._num_points

    @property
    def normalize(self):
        """
        `bool` : Get/set if the constellation is normalized"""
        return self._normalize

    @normalize.setter
    def normalize(self, value):
        assert isinstance(value, bool), "`normalize` must be boolean"
        self._normalize = value

    @property
    def center(self):
        """
        `bool` : Get/set if the constellation is centered"""
        return self._center

    @center.setter
    def center(self, value):
        assert isinstance(value, bool), "`center` must be boolean"
        self._center = value

    @property
    def points(self):
        # pylint: disable=line-too-long
        """
        [2**num_bits_per_symbol], `tf.complex` :  Get/set constellation points"""
        return self._points

    @points.setter
    def points(self, v):
        if not (self._points is None) and \
           (self.constellation_type != "custom"):
            msg = "`points` can only be modified for custom constellations"
            raise ValueError(msg)

        if tf.executing_eagerly():
            if not tf.shape(v)==[2**self.num_bits_per_symbol]:
                err_msg = "`points` must have shape [2**num_bits_per_symbol]"
                raise ValueError(err_msg)

        if isinstance(v, tf.Variable):
            if v.dtype != self.cdtype:
                msg = f"`points` must have dtype={self.cdtype}"
                raise TypeError(msg)
            else:
                self._points = v
        else:
            self._points = tf.cast(v, self.cdtype)

    def call(self):
        x = self.points
        if self.constellation_type == "custom":
            if self._center:
                x = x - tf.reduce_mean(x)
            if self.normalize:
                energy = tf.reduce_mean(tf.square(tf.abs(x)))
                energy_sqrt = tf.complex(tf.sqrt(energy),
                                        tf.constant(0.,
                                        dtype=self.rdtype))
                x = x / energy_sqrt
        return x

    def show(self, labels=True, figsize=(7,7)):
        """Generate a scatter-plot of the constellation

        Input
        -----
        labels : `bool`, (default `True`)
            If `True`, the bit labels will be drawn next to each constellation
            point.

        figsize : Two-element Tuple, `float`, (default `(7,7)`)
            Width and height in inches

        Output
        ------
        : matplotlib.figure.Figure
            Handle to matplot figure object
        """
        p = self().numpy()
        maxval = np.max(np.abs(p))*1.05
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        plt.xlim(-maxval, maxval)
        plt.ylim(-maxval, maxval)
        plt.scatter(np.real(p), np.imag(p))
        ax.set_aspect("equal", adjustable="box")
        plt.xlabel("Real Part")
        plt.ylabel("Imaginary Part")
        plt.grid(True, which="both", axis="both")
        plt.title("Constellation Plot")
        if labels is True:
            for j, p in enumerate(p):
                plt.annotate(
                    np.binary_repr(j, self.num_bits_per_symbol),
                    (np.real(p), np.imag(p))
                )
        return fig

    @staticmethod
    def check_or_create(*,
                        constellation_type=None,
                        num_bits_per_symbol=None,
                        constellation=None,
                        precision=None):
        """Either creates a new constellation or checks an existing one"""
        if isinstance(constellation, Constellation):
            return constellation
        elif constellation_type in ["qam", "pam"]:
            return Constellation(constellation_type,
                                 num_bits_per_symbol,
                                 precision=precision)
        else:
            raise ValueError("You must provide a valid `constellation`")

class Mapper(Block):
    # pylint: disable=line-too-long
    r"""
    Maps binary tensors to points of a constellation

    This class defines a block that maps a tensor of binary values
    to a tensor of points from a provided constellation.

    Parameters
    ----------
    constellation_type : "qam" | "pam" | "custom"
        For "custom", an instance of :class:`~sionna.phy.mapping.Constellation`
        must be provided.

    num_bits_per_symbol : `int`
        The number of bits per constellation symbol, e.g., 4 for QAM16.
        Only required for ``constellation_type`` in ["qam", "pam"].

    constellation :  `None` (default) | :class:`~sionna.phy.mapping.Constellation`
        If no constellation is provided, ``constellation_type``
        and ``num_bits_per_symbol`` must be provided.

    return_indices : bool, (default `False`)
        If enabled, symbol indices are additionally returned.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    : [..., n], `tf.float` or `tf.int`
        Tensor with with binary entries

    Output
    ------
    : [...,n/Constellation.num_bits_per_symbol], `tf.complex`
        Mapped constellation symbols

    : [...,n/Constellation.num_bits_per_symbol], `tf.int32`
        Symbol indices corresponding to the constellation symbols.
        Only returned if ``return_indices`` is set to True.

    Note
    ----
    The last input dimension must be an integer multiple of the
    number of bits per constellation symbol.
    """
    def __init__(self,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 return_indices=False,
                 precision=None,
                 **kwargs
                ):
        super().__init__(precision=precision, **kwargs)
        self._constellation = Constellation.check_or_create(
                                constellation_type=constellation_type,
                                num_bits_per_symbol=num_bits_per_symbol,
                                constellation=constellation,
                                precision=precision)
        self._return_indices = return_indices
        n = self.constellation.num_bits_per_symbol
        self._bit_positions = tf.cast(tf.range(n-1, -1, -1), dtype=tf.int32)

    @property
    def constellation(self):
        """
        :class:`~sionna.phy.mapping.Constellation` : Constellation used by the
        Mapper
        """
        return self._constellation

    def call(self, bits):

         # Convert to int32
        bits = tf.cast(bits, dtype=tf.int32)

        # Reshape last dimensions to the desired format
        n1 = int(bits.shape[-1]/self.constellation.num_bits_per_symbol)
        new_shape = [n1 , self.constellation.num_bits_per_symbol]
        bits = split_dim(bits, new_shape, axis=tf.rank(bits)-1)

        # Use bitwise left shift to compute powers of two
        shifted_bits = tf.bitwise.left_shift(bits, self._bit_positions)

        # Compute the integer representation using bitwise operations
        int_rep = tf.reduce_sum(shifted_bits, axis=-1)

        # Map integers to constellation symbols
        x = tf.gather(self._constellation(), int_rep, axis=0)

        if self._return_indices:
            return x, int_rep
        else:
            return x

class Demapper(Block):
    # pylint: disable=line-too-long
    r"""
    Computes log-likelihood ratios (LLRs) or hard-decisions on bits
    for a tensor of received symbols

    Prior knowledge on the bits can be optionally provided.

    This class defines a block implementing different demapping
    functions. All demapping functions are fully differentiable when soft-decisions
    are computed.

    Parameters
    ----------
    demapping_method : "app" | "maxlog"
        Demapping method

    constellation_type : "qam" | "pam" | "custom"
        For "custom", an instance of :class:`~sionna.phy.mapping.Constellation`
        must be provided.

    num_bits_per_symbol : `int`
        Number of bits per constellation symbol, e.g., 4 for QAM16.
        Only required for ``constellation_type`` in ["qam", "pam"].

    constellation : `None` (default) | :class:`~sionna.phy.mapping.Constellation`
        If no constellation is provided, ``constellation_type``
        and ``num_bits_per_symbol`` must be provided.

    hard_out : bool, (default `False`)
        If `True`, the demapper provides hard-decided bits instead of soft-values.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    y : [...,n], `tf.complex`
        Received symbols

    no : Scalar or [...,n], `tf.float`
        The noise variance estimate. It can be provided either as scalar
        for the entire input batch or as a tensor that is "broadcastable" to
        ``y``.

    prior : `None` (default) | [num_bits_per_symbol] or [...,num_bits_per_symbol], `tf.float`
        Prior for every bit as LLRs.
        It can be provided either as a tensor of shape `[num_bits_per_symbol]` for the
        entire input batch, or as a tensor that is "broadcastable"
        to `[..., n, num_bits_per_symbol]`.

    Output
    ------
    : [...,n*num_bits_per_symbol], `tf.float`
        LLRs or hard-decisions for every bit

    Note
    ----
    With the "app" demapping method, the LLR for the :math:`i\text{th}` bit
    is computed according to

    .. math::
        LLR(i) = \ln\left(\frac{\Pr\left(b_i=1\lvert y,\mathbf{p}\right)}{\Pr\left(b_i=0\lvert y,\mathbf{p}\right)}\right) =\ln\left(\frac{
                \sum_{c\in\mathcal{C}_{i,1}} \Pr\left(c\lvert\mathbf{p}\right)
                \exp\left(-\frac{1}{N_o}\left|y-c\right|^2\right)
                }{
                \sum_{c\in\mathcal{C}_{i,0}} \Pr\left(c\lvert\mathbf{p}\right)
                \exp\left(-\frac{1}{N_o}\left|y-c\right|^2\right)
                }\right)

    where :math:`\mathcal{C}_{i,1}` and :math:`\mathcal{C}_{i,0}` are the
    sets of constellation points for which the :math:`i\text{th}` bit is
    equal to 1 and 0, respectively. :math:`\mathbf{p} = \left[p_0,\dots,p_{K-1}\right]`
    is the vector of LLRs that serves as prior knowledge on the :math:`K` bits that are mapped to
    a constellation point and is set to :math:`\mathbf{0}` if no prior knowledge is assumed to be available,
    and :math:`\Pr(c\lvert\mathbf{p})` is the prior probability on the constellation symbol :math:`c`:

    .. math::
        \Pr\left(c\lvert\mathbf{p}\right) = \prod_{k=0}^{K-1} \text{sigmoid}\left(p_k \ell(c)_k\right)

    where :math:`\ell(c)_k` is the :math:`k^{th}` bit label of :math:`c`, where 0 is
    replaced by -1.
    The definition of the LLR has been
    chosen such that it is equivalent with that of logits. This is
    different from many textbooks in communications, where the LLR is
    defined as :math:`LLR(i) = \ln\left(\frac{\Pr\left(b_i=0\lvert y\right)}{\Pr\left(b_i=1\lvert y\right)}\right)`.

    With the "maxlog" demapping method, LLRs for the :math:`i\text{th}` bit
    are approximated like

    .. math::
        \begin{align}
            LLR(i) &\approx\ln\left(\frac{
                \max_{c\in\mathcal{C}_{i,1}} \Pr\left(c\lvert\mathbf{p}\right)
                    \exp\left(-\frac{1}{N_o}\left|y-c\right|^2\right)
                }{
                \max_{c\in\mathcal{C}_{i,0}} \Pr\left(c\lvert\mathbf{p}\right)
                    \exp\left(-\frac{1}{N_o}\left|y-c\right|^2\right)
                }\right)\\
                &= \max_{c\in\mathcal{C}_{i,0}}
                    \left(\ln\left(\Pr\left(c\lvert\mathbf{p}\right)\right)-\frac{|y-c|^2}{N_o}\right) -
                 \max_{c\in\mathcal{C}_{i,1}}\left( \ln\left(\Pr\left(c\lvert\mathbf{p}\right)\right) - \frac{|y-c|^2}{N_o}\right)
                .
        \end{align}
    """
    def __init__(self,
                 demapping_method,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 hard_out=False,
                 precision=None,
                 **kwargs):
        super().__init__(precision=precision, **kwargs)

        # Create constellation object
        self._constellation = Constellation.check_or_create(
                                constellation_type=constellation_type,
                                num_bits_per_symbol=num_bits_per_symbol,
                                constellation=constellation,
                                precision=precision)

        num_bits_per_symbol = self._constellation.num_bits_per_symbol

        self._logits2llrs = SymbolLogits2LLRs(demapping_method,
                                              num_bits_per_symbol,
                                              hard_out=hard_out,
                                              precision=precision,
                                              **kwargs)

        self._no_threshold = tf.cast(np.finfo(self.rdtype.as_numpy_dtype).tiny,
                                     self.rdtype)

    @property
    def constellation(self):
        """
        :class:`~sionna.phy.mapping.Constellation` : Constellation used by the
            Demapper
        """
        return self._constellation

    def call(self, y, no, prior=None):

        # Reshape constellation points to [1,...1,num_points]
        points_shape = [1]*y.shape.rank + self.constellation.points.shape
        points = tf.reshape(self.constellation.points, points_shape)

        # Compute squared distances from y to all points
        # shape [...,n,num_points]
        squared_dist = tf.pow(tf.abs(tf.expand_dims(y, axis=-1) - points), 2)

        # Add a dummy dimension for broadcasting. This is not needed when no
        # is a scalar, but also does not do any harm.
        no = tf.expand_dims(no, axis=-1)
        # Deal with zero or very small values.
        no = tf.math.maximum(no, self._no_threshold)

        # Compute exponents
        exponents = -squared_dist/no

        llr = self._logits2llrs(exponents, prior)

        # Reshape LLRs to [...,n*num_bits_per_symbol]
        out_shape = tf.concat([tf.shape(y)[:-1],
                               [y.shape[-1] * \
                                self.constellation.num_bits_per_symbol]], 0)
        llr_reshaped = tf.reshape(llr, out_shape)

        return llr_reshaped

class SymbolDemapper(Block):
    # pylint: disable=line-too-long
    r"""
    Computes normalized log-probabilities (logits) or hard-decisions on symbols
    for a tensor of received symbols

    Prior knowldge on the transmitted constellation points can be optionnaly provided.
    The demapping function is fully differentiable when soft-values are
    computed.

    Parameters
    ----------
    constellation_type : "qam" | "pam" | "custom"
        For "custom", an instance of :class:`~sionna.phy.mapping.Constellation`
        must be provided.

    num_bits_per_symbol : `int`
        Number of bits per constellation symbol, e.g., 4 for QAM16.
        Only required for ``constellation_type`` in ["qam", "pam"].

    constellation : `None` (default) | :class:`~sionna.phy.mapping.Constellation`
        If no constellation is provided, ``constellation_type``
        and ``num_bits_per_symbol`` must be provided.

    hard_out : bool, (default `False`)
        If `True`, the demapper provides hard-decided symbols instead of soft-values.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    y : [...,n], `tf.complex`
        Received symbols

    no : Scalar or [...,n], `tf.float`
        Noise variance estimate. It can be provided either as scalar
        for the entire input batch or as a tensor that is "broadcastable" to
        ``y``.

    prior : `None` (default) | [num_points] or [...,num_points], `tf.float`
        Prior for every symbol as log-probabilities (logits).
        It can be provided either as a tensor of shape `[num_points]` for the
        entire input batch, or as a tensor that is "broadcastable"
        to `[..., n, num_points]`.

    Output
    ------
    : [...,n, num_points] or [...,n], `tf.float` or `tf.int32`
        A tensor of shape `[...,n, num_points]` of logits for every constellation
        point if `hard_out` is set to `False`.
        Otherwise, a tensor of shape `[...,n]` of hard-decisions on the symbols.

    Note
    ----
    The normalized log-probability for the constellation point :math:`c` is computed according to

    .. math::
        \ln\left(\Pr\left(c \lvert y,\mathbf{p}\right)\right) = \ln\left( \frac{\exp\left(-\frac{|y-c|^2}{N_0} + p_c \right)}{\sum_{c'\in\mathcal{C}} \exp\left(-\frac{|y-c'|^2}{N_0} + p_{c'} \right)} \right)

    where :math:`\mathcal{C}` is the set of constellation points used for modulation,
    and :math:`\mathbf{p} = \left\{p_c \lvert c \in \mathcal{C}\right\}` the prior information on constellation points given as log-probabilities
    and which is set to :math:`\mathbf{0}` if no prior information on the constellation points is assumed to be available.
    """
    def __init__(self,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 hard_out=False,
                 precision=None,
                 **kwargs):
        super().__init__(precision=precision, **kwargs)
        self._hard_out = hard_out

        # Create constellation object
        self._constellation = Constellation.check_or_create(
                                constellation_type=constellation_type,
                                num_bits_per_symbol=num_bits_per_symbol,
                                constellation=constellation,
                                precision=precision)

    def call(self, y, no, prior=None):
        points = expand_to_rank(self._constellation.points,
                                             tf.rank(y)+1, axis=0)
        y = tf.expand_dims(y, axis=-1)
        d = tf.abs(y-points)

        no = expand_to_rank(no, tf.rank(d), axis=-1)
        exp = -d**2 / no

        if prior is not None:
            prior = expand_to_rank(prior, tf.rank(exp), axis=0)
            exp = exp + prior

        if self._hard_out:
            return tf.argmax(exp, axis=-1, output_type=tf.int32)
        else:
            return tf.nn.log_softmax(exp, axis=-1)

class SymbolLogits2LLRs(Block):
    # pylint: disable=line-too-long
    r"""
    Computes log-likelihood ratios (LLRs) or hard-decisions on bits
    from a tensor of logits (i.e., unnormalized log-probabilities) on constellation points.
    Prior knowledge on the bits can be optionally provided

    Parameters
    ----------
    method : "app" | "maxlog"
        Method used for computing the LLRs

    num_bits_per_symbol : `int`
        Number of bits per constellation symbol, e.g., 4 for QAM16.

    hard_out : `bool`, (default `False`)
        If `True`, the layer provides hard-decided bits instead of soft-values.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    logits : [...,n, num_points], `tf.float`
        Logits on constellation points

    prior : `None` (default) | [num_bits_per_symbol] or [...n, num_bits_per_symbol], `tf.float`
        Prior for every bit as LLRs.
        It can be provided either as a tensor of shape `[num_bits_per_symbol]`
        for the entire input batch, or as a tensor that is "broadcastable"
        to `[..., n, num_bits_per_symbol]`.

    Output
    ------
    : [...,n, num_bits_per_symbol], `tf.float`
        LLRs or hard-decisions for every bit

    Note
    ----
    With the "app" method, the LLR for the :math:`i\text{th}` bit
    is computed according to

    .. math::
        LLR(i) = \ln\left(\frac{\Pr\left(b_i=1\lvert \mathbf{z},\mathbf{p}\right)}{\Pr\left(b_i=0\lvert \mathbf{z},\mathbf{p}\right)}\right) =\ln\left(\frac{
                \sum_{c\in\mathcal{C}_{i,1}} \Pr\left(c\lvert\mathbf{p}\right)
                e^{z_c}
                }{
                \sum_{c\in\mathcal{C}_{i,0}} \Pr\left(c\lvert\mathbf{p}\right)
                e^{z_c}
                }\right)

    where :math:`\mathcal{C}_{i,1}` and :math:`\mathcal{C}_{i,0}` are the
    sets of :math:`2^K` constellation points for which the :math:`i\text{th}` bit is
    equal to 1 and 0, respectively. :math:`\mathbf{z} = \left[z_{c_0},\dots,z_{c_{2^K-1}}\right]` is the vector of logits on the constellation points, :math:`\mathbf{p} = \left[p_0,\dots,p_{K-1}\right]`
    is the vector of LLRs that serves as prior knowledge on the :math:`K` bits that are mapped to
    a constellation point and is set to :math:`\mathbf{0}` if no prior knowledge is assumed to be available,
    and :math:`\Pr(c\lvert\mathbf{p})` is the prior probability on the constellation symbol :math:`c`:

    .. math::
        \Pr\left(c\lvert\mathbf{p}\right) = \prod_{k=0}^{K-1} \Pr\left(b_k = \ell(c)_k \lvert\mathbf{p} \right)
        = \prod_{k=0}^{K-1} \text{sigmoid}\left(p_k \ell(c)_k\right)

    where :math:`\ell(c)_k` is the :math:`k^{th}` bit label of :math:`c`, where 0 is
    replaced by -1.
    The definition of the LLR has been
    chosen such that it is equivalent with that of logits. This is
    different from many textbooks in communications, where the LLR is
    defined as :math:`LLR(i) = \ln\left(\frac{\Pr\left(b_i=0\lvert y\right)}{\Pr\left(b_i=1\lvert y\right)}\right)`.

    With the "maxlog" method, LLRs for the :math:`i\text{th}` bit
    are approximated like

    .. math::
        \begin{align}
            LLR(i) &\approx\ln\left(\frac{
                \max_{c\in\mathcal{C}_{i,1}} \Pr\left(c\lvert\mathbf{p}\right)
                    e^{z_c}
                }{
                \max_{c\in\mathcal{C}_{i,0}} \Pr\left(c\lvert\mathbf{p}\right)
                    e^{z_c}
                }\right)
                .
        \end{align}
    """
    def __init__(self,
                 method,
                 num_bits_per_symbol,
                 *,
                 hard_out=False,
                 precision=None,
                 **kwargs):
        super().__init__(precision=precision, **kwargs)
        assert method in ("app","maxlog"), "Unknown demapping method"
        self._method = method
        self._hard_out = hard_out
        self._num_bits_per_symbol = num_bits_per_symbol
        num_points = int(2**num_bits_per_symbol)

        # Array composed of binary representations of all symbols indices
        a = np.zeros([num_points, num_bits_per_symbol])
        for i in range(0, num_points):
            a[i,:] = np.array(list(np.binary_repr(i, num_bits_per_symbol)),
                              dtype=np.int32)

        # Compute symbol indices for which the bits are 0 or 1
        c0 = np.zeros([int(num_points/2), num_bits_per_symbol])
        c1 = np.zeros([int(num_points/2), num_bits_per_symbol])
        for i in range(num_bits_per_symbol-1,-1,-1):
            c0[:,i] = np.where(a[:,i]==0)[0]
            c1[:,i] = np.where(a[:,i]==1)[0]
        self._c0 = tf.constant(c0, dtype=tf.int32) # Symbols with ith bit=0
        self._c1 = tf.constant(c1, dtype=tf.int32) # Symbols with ith bit=1

        # Array of labels from {-1, 1} of all symbols
        # [num_points, num_bits_per_symbol]
        a = 2*a-1
        self._a = tf.constant(a, dtype=self.rdtype)

        # Determine the reduce function for LLR computation
        if self._method == "app":
            self._reduce = tf.reduce_logsumexp
        else:
            self._reduce = tf.reduce_max

    @property
    def num_bits_per_symbol(self):
        """
        `int` : Number of bits per symbol
        """
        return self._num_bits_per_symbol

    def call(self, logits, prior=None):
        # Compute exponents
        exponents = logits

        # Gather exponents for all bits
        # shape [...,n,num_points/2,num_bits_per_symbol]
        exp0 = tf.gather(exponents, self._c0, axis=-1, batch_dims=0)
        exp1 = tf.gather(exponents, self._c1, axis=-1, batch_dims=0)

        # Process the prior information
        if prior is not None:
            # Expanding `prior` such that it is broadcastable with
            # shape [..., n or 1, 1, num_bits_per_symbol]
            prior = expand_to_rank(prior, tf.rank(logits), axis=0)
            prior = tf.expand_dims(prior, axis=-2)

            # Expand the symbol labeling to be broadcastable with prior
            # shape [..., 1, num_points, num_bits_per_symbol]
            a = expand_to_rank(self._a, tf.rank(prior), axis=0)

            # Compute the prior probabilities on symbols exponents
            # shape [..., n or 1, num_points]
            exp_ps = tf.reduce_sum(tf.math.log_sigmoid(a*prior), axis=-1)

            # Gather prior probability symbol for all bits
            # shape [..., n or 1, num_points/2, num_bits_per_symbol]
            exp_ps0 = tf.gather(exp_ps, self._c0, axis=-1)
            exp_ps1 = tf.gather(exp_ps, self._c1, axis=-1)

        # Compute LLRs using the definition log( Pr(b=1)/Pr(b=0) )
        # shape [..., n, num_bits_per_symbol]
        if prior is not None:
            llr = self._reduce(exp_ps1 + exp1, axis=-2)\
                    - self._reduce(exp_ps0 + exp0, axis=-2)
        else:
            llr = self._reduce(exp1, axis=-2) - self._reduce(exp0, axis=-2)

        if self._hard_out:
            return hard_decisions(llr)
        else:
            return llr

class LLRs2SymbolLogits(Block):
    # pylint: disable=line-too-long
    r"""
    Computes logits (i.e., unnormalized log-probabilities) or hard decisions
    on constellation points from a tensor of log-likelihood ratios (LLRs) on bits

    Parameters
    ----------
    num_bits_per_symbol : `int`
        Number of bits per constellation symbol, e.g., 4 for QAM16.

    hard_out : `bool`, (default `False`)
        If `True`, the layer provides hard-decided constellation points instead of soft-values.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    llrs : [..., n, num_bits_per_symbol], `tf.float`
        LLRs for every bit

    Output
    ------
    : [...,n, num_points], `tf.float` or [..., n], `tf.int32`
        Logits or hard-decisions on constellation points

    Note
    ----
    The logit for the constellation :math:`c` point
    is computed according to

    .. math::
        \begin{align}
            \log{\left(\Pr\left(c\lvert LLRs \right)\right)}
                &= \log{\left(\prod_{k=0}^{K-1} \Pr\left(b_k = \ell(c)_k \lvert LLRs \right)\right)}\\
                &= \log{\left(\prod_{k=0}^{K-1} \text{sigmoid}\left(LLR(k) \ell(c)_k\right)\right)}\\
                &= \sum_{k=0}^{K-1} \log{\left(\text{sigmoid}\left(LLR(k) \ell(c)_k\right)\right)}
        \end{align}

    where :math:`\ell(c)_k` is the :math:`k^{th}` bit label of :math:`c`, where 0 is
    replaced by -1.
    The definition of the LLR has been
    chosen such that it is equivalent with that of logits. This is
    different from many textbooks in communications, where the LLR is
    defined as :math:`LLR(i) = \ln\left(\frac{\Pr\left(b_i=0\lvert y\right)}{\Pr\left(b_i=1\lvert y\right)}\right)`.
    """

    def __init__(self,
                 num_bits_per_symbol,
                 hard_out=False,
                 precision=None,
                 **kwargs):
        super().__init__(precision=precision, **kwargs)

        self._hard_out = hard_out
        self._num_bits_per_symbol = num_bits_per_symbol
        num_points = int(2**num_bits_per_symbol)

        # Array composed of binary representations of all symbols indices
        a = np.zeros([num_points, num_bits_per_symbol])
        for i in range(0, num_points):
            a[i,:] = np.array(list(np.binary_repr(i, num_bits_per_symbol)),
                              dtype=np.int32)

        # Array of labels from {-1, 1} of all symbols
        # [num_points, num_bits_per_symbol]
        a = 2*a-1
        self._a = tf.constant(a, dtype=self.rdtype)

    @property
    def num_bits_per_symbol(self):
        return self._num_bits_per_symbol

    def call(self, llrs):

        # Expand the symbol labeling to be broadcastable with prior
        # shape [1, ..., 1, num_points, num_bits_per_symbol]
        a = expand_to_rank(self._a, tf.rank(llrs), axis=0)

        # Compute the prior probabilities on symbols exponents
        # shape [..., 1, num_points]
        llrs = tf.expand_dims(llrs, axis=-2)
        logits = tf.reduce_sum(tf.math.log_sigmoid(a*llrs), axis=-1)

        if self._hard_out:
            return tf.argmax(logits, axis=-1, output_type=tf.int32)
        else:
            return logits

class SymbolLogits2Moments(Block):
    # pylint: disable=line-too-long
    r"""
    Computes the mean and variance of a constellation from logits (unnormalized log-probabilities) on the
    constellation points

    More precisely, given a constellation :math:`\mathcal{C} = \left[ c_0,\dots,c_{N-1} \right]` of size :math:`N`, this layer computes the mean and variance
    according to

    .. math::
        \begin{align}
            \mu &= \sum_{n = 0}^{N-1} c_n \Pr \left(c_n \lvert \mathbf{\ell} \right)\\
            \nu &= \sum_{n = 0}^{N-1} \left( c_n - \mu \right)^2 \Pr \left(c_n \lvert \mathbf{\ell} \right)
        \end{align}


    where :math:`\mathbf{\ell} = \left[ \ell_0, \dots, \ell_{N-1} \right]` are the logits, and

    .. math::
        \Pr \left(c_n \lvert \mathbf{\ell} \right) = \frac{\exp \left( \ell_n \right)}{\sum_{i=0}^{N-1} \exp \left( \ell_i \right) }.

    Parameters
    ----------
    constellation_type : "qam" | "pam" | "custom"
        For "custom", an instance of :class:`~sionna.phy.mapping.Constellation`
        must be provided.

    num_bits_per_symbol : `int`
        The number of bits per constellation symbol, e.g., 4 for QAM16.
        Only required for ``constellation_type`` in ["qam", "pam"].

    constellation :  `None` (default) | :class:`~sionna.phy.mapping.Constellation`
        If no constellation is provided, ``constellation_type``
        and ``num_bits_per_symbol`` must be provided.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    logits : [...,n, num_points], `tf.float`
        Logits on constellation points

    Output
    ------
    mean : [...,n], `tf.float`
        Mean of the constellation

    var : [...,n], `tf.float`
        Variance of the constellation
    """
    def __init__(self,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 precision=None,
                 **kwargs):
        super().__init__(precision=precision, **kwargs)

        # Create constellation object
        self._constellation = Constellation.check_or_create(
                                constellation_type=constellation_type,
                                num_bits_per_symbol=num_bits_per_symbol,
                                constellation=constellation,
                                precision=precision)

    def call(self, logits):
        p = tf.math.softmax(logits, axis=-1)
        p_c = tf.complex(p, tf.cast(0.0, self.rdtype))
        points = self._constellation()
        points = expand_to_rank(points, tf.rank(p), axis=0)

        mean = tf.reduce_sum(p_c*points, axis=-1, keepdims=True)
        var = tf.reduce_sum(p*tf.square(tf.abs(points - mean)), axis=-1)
        mean = tf.squeeze(mean, axis=-1)

        return mean, var

class SymbolInds2Bits(Block):
    # pylint: disable=line-too-long
    r"""
    Transforms symbol indices to their binary representations

    Parameters
    ----------
    num_bits_per_symbol : `int`
        Number of bits per constellation symbol

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    : Tensor, `tf.int`
        Symbol indices

    Output
    ------
    : input.shape + [num_bits_per_symbol], `tf.float`
        Binary representation of symbol indices
    """
    def __init__(self,
               num_bits_per_symbol,
               precision=None,
               **kwargs):
        super().__init__(precision=precision, **kwargs)
        num_symbols = 2**num_bits_per_symbol
        b = np.zeros([num_symbols, num_bits_per_symbol])
        for i in range(0, num_symbols):
            b[i,:] = np.array(list(np.binary_repr(i, num_bits_per_symbol)),
                              dtype=np.int32)
        self._bit_labels = tf.constant(b, self.rdtype)

    def call(self, symbol_ind):
        return tf.gather(self._bit_labels, symbol_ind)

class QAM2PAM(Object):
    r"""Transforms QAM symbol indices to PAM symbol indices

    For indices in a QAM constellation, computes the corresponding indices
    for the two PAM constellations corresponding the real and imaginary
    components of the QAM constellation.

    Parameters
    ----------
    num_bits_per_symbol : `int`
        Number of bits per QAM constellation symbol, e.g., 4 for QAM16.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    ind_qam : Tensor, `tf.int32`
        Indices in the QAM constellation

    Output
    -------
    ind_pam1 : Tensor, `tf.int32`
        Indices for the first component of the corresponding PAM modulation

    ind_pam2 : Tensor, `tf.int32`
        Indices for the first component of the corresponding PAM modulation
    """
    def __init__(self,
                 num_bits_per_symbol,
                 precision=None,
                 **kwargs):
        super().__init__(precision=precision, **kwargs)
        base = [2**i for i in range(num_bits_per_symbol//2-1, -1, -1)]
        base = np.array(base)
        pam1_ind = np.zeros([2**num_bits_per_symbol], dtype=np.int32)
        pam2_ind = np.zeros([2**num_bits_per_symbol], dtype=np.int32)
        for i in range(0, 2**num_bits_per_symbol):
            b = np.array(list(np.binary_repr(i,num_bits_per_symbol)),
                         dtype=np.int32)
            pam1_ind[i] = np.sum(b[0::2]*base)
            pam2_ind[i] = np.sum(b[1::2]*base)
        self._pam1_ind = tf.constant(pam1_ind, dtype=tf.int32)
        self._pam2_ind = tf.constant(pam2_ind, dtype=tf.int32)

    def __call__(self, ind_qam):
        ind_pam1 = tf.gather(self._pam1_ind, ind_qam, axis=0)
        ind_pam2 = tf.gather(self._pam2_ind, ind_qam, axis=0)

        return ind_pam1, ind_pam2

class PAM2QAM(Object):
    r"""Transforms PAM symbol indices/logits to QAM symbol indices/logits

    For two PAM constellation symbol indices or logits, corresponding to
    the real and imaginary components of a QAM constellation,
    compute the QAM symbol index or logits.

    Parameters
    ----------
    num_bits_per_symbol : `int`
        Number of bits per QAM constellation symbol, e.g., 4 for QAM16

    hard_in_out : `bool`, (default `True`)
        Determines if inputs and outputs are indices or logits over
        constellation symbols.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    pam1 : Tensor, `tf.int32`, or [...,2**(num_bits_per_symbol/2)], `tf.float`
        Indices or logits for the first PAM constellation

    pam2 : Tensor, `tf.int32`, or [...,2**(num_bits_per_symbol/2)], `tf.float`
        Indices or logits for the second PAM constellation

    Output
    -------
    qam : Tensor, `tf.int32`, or [...,2**num_bits_per_symbol], `tf.float`
        Indices or logits for the corresponding QAM constellation
    """
    def __init__(self,
                 num_bits_per_symbol,
                 hard_in_out=True,
                 precision=None,
                 **kwargs):
        super().__init__(precision=precision, **kwargs)
        num_pam_symbols = 2**(num_bits_per_symbol//2)
        base = np.array([2**i for i in range(num_bits_per_symbol-1, -1, -1)])

        # Create an array of QAM symbol indices, index by two PAM indices
        ind = np.zeros([num_pam_symbols, num_pam_symbols], np.int32)
        for i in range(0, num_pam_symbols):
            for j in range(0, num_pam_symbols):
                b1 = np.array(list(np.binary_repr(i,num_bits_per_symbol//2)),
                              dtype=np.int32)
                b2 = np.array(list(np.binary_repr(j,num_bits_per_symbol//2)),
                              dtype=np.int32)
                b = np.zeros([num_bits_per_symbol], np.int32)
                b[0::2] = b1
                b[1::2] = b2
                ind[i, j] = np.sum(b*base)
        self._qam_ind = tf.constant(ind, dtype=tf.int32)
        self._hard_in_out = hard_in_out

    def __call__(self, pam1, pam2):

        # PAM indices to QAM indices
        if self._hard_in_out:
            shape = tf.shape(pam1)
            ind_pam1 = tf.reshape(pam1, [-1, 1])
            ind_pam2 = tf.reshape(pam2, [-1, 1])
            ind_pam = tf.concat([ind_pam1, ind_pam2], axis=-1)
            ind_qam = tf.gather_nd(self._qam_ind, ind_pam)
            ind_qam = tf.reshape(ind_qam, shape)
            return ind_qam

        # PAM logits to QAM logits
        else:
            # Compute all combination of sums of logits
            logits_mat = tf.expand_dims(pam1, -1) + tf.expand_dims(pam2, -2)

            # Flatten to a vector
            logits = flatten_last_dims(logits_mat)

            # Gather symbols in the correct order
            gather_ind = tf.reshape(self._qam_ind, [-1])
            logits = tf.gather(logits, gather_ind, axis=-1)
            return logits

class BinarySource(Block):
    """
    Generates a random binary tensor

    Parameters
    ----------
    seed : `None` (default) | `int`
        Set the seed for the random generator used to generate the bits.
        If set to `None`, :attr:`~sionna.phy.config.Config.tf_rng` is used.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    shape : 1D tensor/array/list, `int`
        Desired shape of the output tensor

    Output
    ------
    : ``shape``, `tf.float`
        Tensor filled with random binary values
    """
    def __init__(self, precision=None, seed=None, **kwargs):
        super().__init__(precision=precision, **kwargs)
        self._seed = seed
        if self._seed is not None:
            self._rng = tf.random.Generator.from_seed(self._seed)
        else:
            self._rng = config.tf_rng

    def call(self, inputs):
        return tf.cast(self._rng.uniform(inputs, 0, 2, tf.int32),
                       dtype=self.rdtype)

class SymbolSource(Block):
    # pylint: disable=line-too-long
    r"""
    Generates a tensor of random constellation symbols

    Optionally, the symbol indices and/or binary representations of the
    constellation symbols can be returned.

    Parameters
    ----------
    constellation_type : `str`, "qam" | "pam" | "custom"]
        For "custom", an instance of :class:`~sionna.phy.mapping.Constellation`
        must be provided.

    num_bits_per_symbol : `int`
        Number of bits per constellation symbol, e.g., 4 for QAM16.
        Only required for ``constellation_type`` in ["qam", "pam"].

    constellation : `None` (default) | :class:`~sionna.phy.mapping.Constellation`
        If no constellation is provided, ``constellation_type``
        and ``num_bits_per_symbol`` must be provided.

    return_indices : `bool`, (default `False`)
        If enabled, the function also returns the symbol indices.

    return_bits : `bool`, (default `False`)
        If enabled, the function also returns the binary symbol
        representations (i.e., bit labels).

    seed : `None` (default) | `int`
        Set the seed for the random generator used to generate the bits.
        If set to `None`, :attr:`~sionna.phy.config.Config.tf_rng` is used.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    shape : 1D tensor/array/list, `int`
        Desired shape of the output tensor

    Output
    ------
    symbols : ``shape``, `tf.complex`
        Tensor filled with random symbols of the chosen ``constellation_type``.

    symbol_indices : ``shape``, `tf.int32`
        Tensor filled with the symbol indices.
        Only returned if ``return_indices`` is `True`.

    bits : [``shape``, ``num_bits_per_symbol``], `tf.float`
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
                 precision=None,
                 **kwargs
                ):
        super().__init__(precision=precision, **kwargs)
        constellation = Constellation.check_or_create(
                                constellation_type=constellation_type,
                                num_bits_per_symbol=num_bits_per_symbol,
                                constellation=constellation,
                                precision=precision)
        self._num_bits_per_symbol = constellation.num_bits_per_symbol
        self._return_indices = return_indices
        self._return_bits = return_bits
        self._binary_source = BinarySource(seed=seed, precision=precision)
        self._mapper = Mapper(constellation=constellation,
                              return_indices=return_indices,
                              precision=precision)

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
    r"""
    Generates a tensor of random QAM symbols

    Optionally, the symbol indices and/or binary representations of the
    constellation symbols can be returned.

    Parameters
    ----------
    num_bits_per_symbol : `int`
        Number of bits per constellation symbol, e.g., 4 for QAM16.
        Only required for ``constellation_type`` in ["qam", "pam"].

    return_indices : `bool`, (default `False`)
        If enabled, the function also returns the symbol indices.

    return_bits : `bool`, (default `False`)
        If enabled, the function also returns the binary symbol
        representations (i.e., bit labels).

    seed : `None` (default) | `int`
        Set the seed for the random generator used to generate the bits.
        If set to `None`, :attr:`~sionna.phy.config.Config.tf_rng` is used.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    shape : 1D tensor/array/list, `int`
        Desired shape of the output tensor

    Output
    ------
    symbols : ``shape``, `tf.complex`
        Tensor filled with random QAM symbols

    symbol_indices : ``shape``, `tf.int32`
        Tensor filled with the symbol indices.
        Only returned if ``return_indices`` is `True`.

    bits : [``shape``, ``num_bits_per_symbol``], `tf.float`
        Tensor filled with the binary symbol representations (i.e., bit labels).
        Only returned if ``return_bits`` is `True`.
    """
    def __init__(self,
                 num_bits_per_symbol=None,
                 return_indices=False,
                 return_bits=False,
                 seed=None,
                 precision=None,
                 **kwargs
                ):
        super().__init__(constellation_type="qam",
                         num_bits_per_symbol=num_bits_per_symbol,
                         return_indices=return_indices,
                         return_bits=return_bits,
                         seed=seed,
                         precision=precision,
                         **kwargs)

class PAMSource(SymbolSource):
    # pylint: disable=line-too-long
    r"""
    Generates a tensor of random PAM symbols

    Optionally, the symbol indices and/or binary representations of the
    constellation symbols can be returned.

    Parameters
    ----------
    num_bits_per_symbol : `int`
        Number of bits per constellation symbol, e.g., 4 for QAM16.
        Only required for ``constellation_type`` in ["qam", "pam"].

    return_indices : `bool`, (default `False`)
        If enabled, the function also returns the symbol indices.

    return_bits : `bool`, (default `False`)
        If enabled, the function also returns the binary symbol
        representations (i.e., bit labels).
        Defaults to `False`.

    seed : `None` (default) | `int`
        Set the seed for the random generator used to generate the bits.
        If set to `None`, :attr:`~sionna.phy.config.Config.tf_rng` is used.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----
    shape : 1D tensor/array/list, `int`
        Desired shape of the output tensor

    Output
    ------
    symbols : ``shape``, `tf.complex`
        Tensor filled with random PAM symbols

    symbol_indices : ``shape``, `tf.int32`
        Tensor filled with the symbol indices.
        Only returned if ``return_indices`` is `True`.

    bits : [``shape``, ``num_bits_per_symbol``], `tf.float`
        Tensor filled with the binary symbol representations (i.e., bit labels).
        Only returned if ``return_bits`` is `True`.
    """
    def __init__(self,
                 num_bits_per_symbol=None,
                 return_indices=False,
                 return_bits=False,
                 seed=None,
                 precision=None,
                 **kwargs
                ):
        super().__init__(constellation_type="pam",
                         num_bits_per_symbol=num_bits_per_symbol,
                         return_indices=return_indices,
                         return_bits=return_bits,
                         seed=seed,
                         precision=precision,
                         **kwargs)
