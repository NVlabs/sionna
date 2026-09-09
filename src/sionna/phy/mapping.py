#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Blocks for (de)mapping, constellation class, and utility functions"""

from numbers import Integral
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sionna._validation import check_instance, check_one_of
from sionna.phy.block import Block
from sionna.phy.object import Object
from sionna.phy.config import config, dtypes, Precision
from sionna.phy.utils import (
    expand_to_rank,
    flatten_last_dims,
    hard_decisions,
    split_dim,
    randint,
)

__all__ = [
    "pam_gray",
    "qam",
    "pam",
    "Constellation",
    "Mapper",
    "Demapper",
    "SymbolDemapper",
    "SymbolLogits2LLRs",
    "LLRs2SymbolLogits",
    "SymbolLogits2Moments",
    "SymbolInds2Bits",
    "QAM2PAM",
    "PAM2QAM",
    "BinarySource",
    "SymbolSource",
    "QAMSource",
    "PAMSource",
]


def _compute_binary_labels(num_bits: int) -> np.ndarray:
    """Compute binary representation of indices 0 to 2**num_bits - 1.

    :param num_bits: Number of bits per symbol.

    :output labels: `np.ndarray` of shape `[2**num_bits, num_bits]`.
        Binary labels.
    """
    num_points = 2**num_bits
    indices = np.arange(num_points)[:, None]
    bit_positions = np.arange(num_bits - 1, -1, -1)
    return (indices >> bit_positions) & 1


def _check_square_qam_bit_width(num_bits_per_symbol: Any) -> int:
    """Validate a QAM bit width that must split evenly over both dimensions.

    :param num_bits_per_symbol: Number of bits per QAM symbol.

    :output num_bits_per_symbol: `int`.
        The validated bit width.
    """
    if isinstance(num_bits_per_symbol, bool) or not isinstance(
        num_bits_per_symbol, Integral
    ):
        raise TypeError("`num_bits_per_symbol` must be an integer")
    num_bits_per_symbol = int(num_bits_per_symbol)
    if num_bits_per_symbol <= 0 or num_bits_per_symbol % 2 != 0:
        raise ValueError("`num_bits_per_symbol` must be a positive even integer")
    return num_bits_per_symbol


def pam_gray(b: np.ndarray) -> int:
    r"""Maps a vector of bits to a PAM constellation point with Gray labeling.

    This recursive function maps a binary vector to Gray-labelled PAM
    constellation points. It can be used to generate QAM constellations.
    The constellation is not normalized.

    :param b: Array with binary entries of shape `[n]`.

    :output point: `int`.
        PAM constellation point taking values in
        :math:`\{\pm 1,\pm 3,\dots,\pm (2^n-1)\}`.

    .. rubric:: Notes

    This algorithm is a recursive implementation of the expressions found in
    Section 5.1 of :cite:p:`3GPPTS38211`. It is used in the 5G standard.

    .. rubric:: Examples

    .. code-block:: python

        import numpy as np
        from sionna.phy.mapping import pam_gray

        b = np.array([1, 0])
        print(pam_gray(b))
        # -1
    """
    if len(b) > 1:
        return (1 - 2 * b[0]) * (2 ** len(b[1:]) - pam_gray(b[1:]))
    return 1 - 2 * b[0]


def qam(
    num_bits_per_symbol: int,
    normalize: bool = True,
    precision: Optional[Precision] = None,
) -> np.ndarray:
    r"""Generates a QAM constellation.

    This function generates a complex-valued vector, where each element is
    a constellation point of an M-ary QAM constellation. The bit
    label of the ``n`` th point is given by the length-``num_bits_per_symbol``
    binary representation of ``n``.

    :param num_bits_per_symbol: Number of bits per constellation point.
        Must be a multiple of two, e.g., 2, 4, 6, 8, etc.
    :param normalize: If `True`, the constellation is normalized to have
        unit power. Defaults to `True`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.

    :output c: `np.ndarray`, shape `[2**num_bits_per_symbol]`.
        QAM constellation points.

    .. rubric:: Notes

    The bit label of the nth constellation point is given by the binary
    representation of its position within the array and can be obtained
    through ``np.binary_repr(n, num_bits_per_symbol)``.

    The normalization factor of a QAM constellation is given in
    closed-form as:

    .. math::
        \sqrt{\frac{1}{2^{n-2}}\sum_{i=1}^{2^{n-1}}(2i-1)^2}

    where :math:`n= \text{num\_bits\_per\_symbol}/2` is the number of bits
    per dimension.

    This algorithm is a recursive implementation of the expressions found in
    Section 5.1 of :cite:p:`3GPPTS38211`. It is used in the 5G standard.

    .. rubric:: Examples

    .. code-block:: python

        from sionna.phy.mapping import qam

        # Generate 16-QAM constellation
        constellation = qam(4)
        print(constellation.shape)
        # (16,)
    """
    if not (num_bits_per_symbol > 0 and num_bits_per_symbol % 2 == 0):
        raise ValueError("num_bits_per_symbol must be a positive multiple of 2")
    check_instance(
        normalize,
        bool,
        name="normalize",
        message="normalize must be boolean",
    )

    if precision is None:
        rdtype = config.np_dtype
        cdtype = config.np_cdtype
    else:
        rdtype = dtypes[precision]["np"]["dtype"]
        cdtype = dtypes[precision]["np"]["cdtype"]

    # Build constellation by iterating through all points
    c = np.zeros([2**num_bits_per_symbol], dtype=cdtype)
    for i in range(0, 2**num_bits_per_symbol):
        b = np.array(list(np.binary_repr(i, num_bits_per_symbol)), dtype=np.int32)
        c[i] = pam_gray(b[0::2]) + 1j * pam_gray(b[1::2])  # PAM in each dimension

    if normalize:  # Normalize to unit energy
        n = int(num_bits_per_symbol / 2)
        qam_var = (
            1
            / (2 ** (n - 2))
            * np.sum(np.linspace(1, 2**n - 1, 2 ** (n - 1), dtype=rdtype) ** 2)
        )
        c /= np.sqrt(qam_var)
    return c


def pam(
    num_bits_per_symbol: int,
    normalize: bool = True,
    precision: Optional[Precision] = None,
) -> np.ndarray:
    r"""Generates a PAM constellation.

    This function generates a real-valued vector, where each element is
    a constellation point of an M-ary PAM constellation. The bit
    label of the ``n`` th point is given by the length-``num_bits_per_symbol``
    binary representation of ``n``.

    :param num_bits_per_symbol: Number of bits per constellation point.
        Must be positive.
    :param normalize: If `True`, the constellation is normalized to have
        unit power. Defaults to `True`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.

    :output c: `np.ndarray`, shape `[2**num_bits_per_symbol]`.
        PAM constellation symbols.

    .. rubric:: Notes

    The bit label of the nth constellation point is given by the binary
    representation of its position within the array and can be obtained
    through ``np.binary_repr(n, num_bits_per_symbol)``.

    The normalization factor of a PAM constellation is given in
    closed-form as:

    .. math::
        \sqrt{\frac{1}{2^{n-1}}\sum_{i=1}^{2^{n-1}}(2i-1)^2}

    where :math:`n= \text{num\_bits\_per\_symbol}` is the number of bits
    per symbol.

    This algorithm is a recursive implementation of the expressions found in
    Section 5.1 of :cite:p:`3GPPTS38211`. It is used in the 5G standard.

    .. rubric:: Examples

    .. code-block:: python

        from sionna.phy.mapping import pam

        # Generate 4-PAM constellation
        constellation = pam(2)
        print(constellation.shape)
        # (4,)
    """
    if not num_bits_per_symbol > 0:
        raise ValueError("num_bits_per_symbol must be positive")
    check_instance(
        normalize,
        bool,
        name="normalize",
        message="normalize must be boolean",
    )

    if precision is None:
        rdtype = config.np_dtype
        cdtype = config.np_cdtype
    else:
        rdtype = dtypes[precision]["np"]["dtype"]
        cdtype = dtypes[precision]["np"]["cdtype"]

    # Build constellation by iterating through all points
    c = np.zeros([2**num_bits_per_symbol], dtype=cdtype)
    for i in range(0, 2**num_bits_per_symbol):
        b = np.array(list(np.binary_repr(i, num_bits_per_symbol)), dtype=np.int32)
        c[i] = pam_gray(b)

    if normalize:  # Normalize to unit energy
        n = int(num_bits_per_symbol)
        pam_var = (
            1
            / (2 ** (n - 1))
            * np.sum(np.linspace(1, 2**n - 1, 2 ** (n - 1), dtype=rdtype) ** 2)
        )
        c /= np.sqrt(pam_var)
    return c


class Constellation(Block):
    r"""
    Constellation that can be used by a (de-)mapper.

    This class defines a constellation, i.e., a complex-valued vector of
    constellation points. The binary representation of the index of an
    element of this vector corresponds to the bit label of the constellation
    point. This implicit bit labeling is used by the
    :class:`~sionna.phy.mapping.Mapper` and :class:`~sionna.phy.mapping.Demapper`.

    :param constellation_type: Type of constellation. One of "qam", "pam", or "custom".
        For "custom", the constellation ``points`` must be provided.
    :param num_bits_per_symbol: Number of bits per constellation symbol,
        e.g., 4 for QAM16.
    :param points: Custom constellation points of shape `[2**num_bits_per_symbol]`.
        Only used when ``constellation_type`` is "custom". Defaults to `None`.
    :param normalize: If `True`, the constellation is normalized to have unit power.
        Only applies to custom constellations. Defaults to `False`.
    :param center: If `True`, the constellation is ensured to have zero mean.
        Only applies to custom constellations. Defaults to `False`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for tensor operations. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :output points: [2\*\*num_bits_per_symbol], `torch.complex`.
        (Possibly) centered and normalized constellation points.

    .. rubric:: Examples

    .. code-block:: python

        from sionna.phy.mapping import Constellation

        # Create a 16-QAM constellation
        const = Constellation("qam", 4)
        points = const()
        print(points.shape)
        # torch.Size([16])

        # Visualize the constellation
        const.show()
    """

    def __init__(
        self,
        constellation_type: str,
        num_bits_per_symbol: int,
        points: Optional[Union[np.ndarray, torch.Tensor]] = None,
        normalize: bool = False,
        center: bool = False,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(precision=precision, device=device, **kwargs)

        check_one_of(
            constellation_type,
            ("qam", "pam", "custom"),
            name="constellation_type",
            message=f"Wrong `constellation_type` {constellation_type}",
        )
        self._constellation_type = constellation_type

        if num_bits_per_symbol is None:
            raise ValueError("No value for `num_bits_per_symbol`")
        n = num_bits_per_symbol
        if (n <= 0) or (n % 1 != 0):
            raise ValueError("`num_bits_per_symbol` must be a positive integer")
        if self.constellation_type == "qam":
            if n % 2 != 0:
                raise ValueError(
                    "`num_bits_per_symbol` must be a positive integer multiple of 2"
                )
        self._num_bits_per_symbol = int(n)
        self._num_points = 2**self._num_bits_per_symbol

        self.normalize = normalize
        self.center = center

        if (points is not None) and (constellation_type != "custom"):
            raise ValueError(
                "`points` can only be provided for `constellation_type`='custom'"
            )
        elif (points is None) and (constellation_type == "custom"):
            raise ValueError("You must provide a value for `points`")

        # Initialize _points placeholder (will be set by property setter)
        self._points = None
        if self.constellation_type == "qam":
            points = qam(self.num_bits_per_symbol, normalize=True, precision=precision)
        elif self.constellation_type == "pam":
            points = pam(self.num_bits_per_symbol, normalize=True, precision=precision)
        self.points = points

    @property
    def constellation_type(self) -> str:
        """Constellation type ("qam", "pam", or "custom")"""
        return self._constellation_type

    @property
    def num_bits_per_symbol(self) -> int:
        """Number of bits per symbol"""
        return self._num_bits_per_symbol

    @property
    def num_points(self) -> int:
        """Number of constellation points"""
        return self._num_points

    @property
    def normalize(self) -> bool:
        """Get/set if the constellation is normalized"""
        return self._normalize

    @normalize.setter
    def normalize(self, value: bool) -> None:
        check_instance(
            value,
            bool,
            name="normalize",
            message="`normalize` must be boolean",
        )
        self._normalize = value

    @property
    def center(self) -> bool:
        """Get/set if the constellation is centered"""
        return self._center

    @center.setter
    def center(self, value: bool) -> None:
        check_instance(
            value,
            bool,
            name="center",
            message="`center` must be boolean",
        )
        self._center = value

    @property
    def points(self) -> torch.Tensor:
        """Get/set constellation points of shape `[2**num_bits_per_symbol]`"""
        return self._points

    @points.setter
    def points(self, v: Union[np.ndarray, torch.Tensor]) -> None:
        if (self._points is not None) and (self.constellation_type != "custom"):
            msg = "`points` can only be modified for custom constellations"
            raise ValueError(msg)

        if isinstance(v, torch.Tensor):
            if v.shape != torch.Size([2**self.num_bits_per_symbol]):
                err_msg = "`points` must have shape [2**num_bits_per_symbol]"
                raise ValueError(err_msg)
            # Use .to() to preserve gradient flow for trainable constellations
            if v.dtype != self.cdtype or v.device != torch.device(self.device):
                v = v.to(dtype=self.cdtype, device=self.device)
            self._points = v
        else:
            # Convert numpy array to tensor
            if np.shape(v) != (2**self.num_bits_per_symbol,):
                err_msg = "`points` must have shape [2**num_bits_per_symbol]"
                raise ValueError(err_msg)
            self._points = torch.tensor(v, dtype=self.cdtype, device=self.device)

    def call(self) -> torch.Tensor:
        x = self.points
        if self.constellation_type == "custom":
            if self._center:
                x = x - x.mean()
            if self.normalize:
                energy = x.abs().square().mean()
                x = x / energy.sqrt()
        return x

    def show(
        self, labels: bool = True, figsize: Tuple[float, float] = (7, 7)
    ) -> plt.Figure:
        """Generate a scatter-plot of the constellation.

        :param labels: If `True`, the bit labels will be drawn next to each
            constellation point. Defaults to `True`.
        :param figsize: Width and height in inches. Defaults to `(7, 7)`.

        :output fig: `plt.Figure`.
            The matplotlib figure object.
        """
        p = self().detach().cpu().numpy()
        maxval = np.max(np.abs(p)) * 1.05
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
            for j, point in enumerate(p):
                plt.annotate(
                    np.binary_repr(j, self.num_bits_per_symbol),
                    (np.real(point), np.imag(point)),
                )
        return fig

    @staticmethod
    def check_or_create(
        *,
        constellation_type: Optional[str] = None,
        num_bits_per_symbol: Optional[int] = None,
        constellation: Optional["Constellation"] = None,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
    ) -> "Constellation":
        """Either creates a new constellation or checks an existing one."""
        if isinstance(constellation, Constellation):
            return constellation
        elif constellation_type in ["qam", "pam"]:
            return Constellation(
                constellation_type,
                num_bits_per_symbol,
                precision=precision,
                device=device,
            )
        else:
            raise ValueError("You must provide a valid `constellation`")


class Mapper(Block):
    r"""
    Maps binary tensors to points of a constellation.

    This class defines a block that maps a tensor of binary values
    to a tensor of points from a provided constellation.

    :param constellation_type: Type of constellation. One of "qam", "pam", or "custom".
        For "custom", an instance of :class:`~sionna.phy.mapping.Constellation`
        must be provided.
    :param num_bits_per_symbol: The number of bits per constellation symbol,
        e.g., 4 for QAM16. Only required for ``constellation_type`` in ["qam", "pam"].
    :param constellation: If no constellation is provided, ``constellation_type``
        and ``num_bits_per_symbol`` must be provided. Defaults to `None`.
    :param return_indices: If enabled, symbol indices are additionally returned.
        Defaults to `False`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for tensor operations. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input bits: [..., n], `torch.float` or `torch.int`.
        Tensor with binary entries.

    :output x: [..., n/Constellation.num_bits_per_symbol], `torch.complex`.
        Mapped constellation symbols.
    :output ind: [..., n/Constellation.num_bits_per_symbol], `torch.int32`.
        Symbol indices corresponding to the constellation symbols.
        Only returned if ``return_indices`` is set to `True`.

    .. rubric:: Notes

    The last input dimension must be an integer multiple of the
    number of bits per constellation symbol.

    To avoid synchronizing accelerator devices in this performance-critical
    path, input values are not checked at runtime. Every entry must be exactly
    zero or one; passing non-binary values has undefined behavior.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.mapping import Mapper

        mapper = Mapper("qam", 4)  # 16-QAM
        bits = torch.randint(0, 2, (10, 100))
        symbols = mapper(bits)
        print(symbols.shape)
        # torch.Size([10, 25])
    """

    def __init__(
        self,
        constellation_type: Optional[str] = None,
        num_bits_per_symbol: Optional[int] = None,
        constellation: Optional[Constellation] = None,
        return_indices: bool = False,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(precision=precision, device=device, **kwargs)
        self._constellation = Constellation.check_or_create(
            constellation_type=constellation_type,
            num_bits_per_symbol=num_bits_per_symbol,
            constellation=constellation,
            precision=precision,
            device=device,
        )
        self._return_indices = return_indices
        n = self.constellation.num_bits_per_symbol
        self._bit_positions = torch.arange(
            n - 1, -1, -1, dtype=torch.int32, device=self.device
        )

    @property
    def constellation(self) -> Constellation:
        """Constellation used by the Mapper"""
        return self._constellation

    def call(
        self, bits: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        num_bits_per_symbol = self.constellation.num_bits_per_symbol
        if bits.dim() == 0 or bits.shape[-1] % num_bits_per_symbol != 0:
            raise ValueError(
                "The last dimension of `bits` must be a multiple of "
                "`num_bits_per_symbol`."
            )

        # Convert to int32
        bits = bits.to(dtype=torch.int32)

        # Reshape last dimensions to the desired format
        n1 = bits.shape[-1] // num_bits_per_symbol
        new_shape = [n1, num_bits_per_symbol]
        bits = split_dim(bits, new_shape, axis=bits.dim() - 1)

        # Use bitwise left shift to compute powers of two
        shifted_bits = torch.bitwise_left_shift(bits, self._bit_positions)

        # Compute the integer representation using bitwise operations
        int_rep = shifted_bits.sum(dim=-1).to(torch.int32)

        # Map integers to constellation symbols
        x = self._constellation()[int_rep]

        if self._return_indices:
            return x, int_rep
        else:
            return x


class Demapper(Block):
    r"""
    Computes log-likelihood ratios (LLRs) or hard-decisions on bits
    for a tensor of received symbols.

    Prior knowledge on the bits can be optionally provided.

    This class defines a block implementing different demapping
    functions. All demapping functions are fully differentiable when soft-decisions
    are computed.

    :param demapping_method: Demapping method. One of "app" or "maxlog".
    :param constellation_type: Type of constellation. One of "qam", "pam", or "custom".
        For "custom", an instance of :class:`~sionna.phy.mapping.Constellation`
        must be provided.
    :param num_bits_per_symbol: Number of bits per constellation symbol,
        e.g., 4 for QAM16. Only required for ``constellation_type`` in ["qam", "pam"].
    :param constellation: If no constellation is provided, ``constellation_type``
        and ``num_bits_per_symbol`` must be provided. Defaults to `None`.
    :param hard_out: If `True`, the demapper provides hard-decided bits instead of
        soft-values. Defaults to `False`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for tensor operations. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input y: [..., n], `torch.complex`. Received symbols.
    :input no: Scalar or [..., n], `torch.float`. The noise variance estimate. It can be
        provided either as scalar for the entire input batch or as a tensor
        that is "broadcastable" to ``y``.
    :input prior: `None` (default) or [num_bits_per_symbol] or [..., num_bits_per_symbol],
        `torch.float`. Prior for every bit as LLRs. It can be provided either as
        a tensor of shape `[num_bits_per_symbol]` for the entire input batch, or
        as a tensor that is "broadcastable" to `[..., n, num_bits_per_symbol]`.

    :output llr: [..., n\*num_bits_per_symbol], `torch.float`. LLRs or hard-decisions
        for every bit.

    .. rubric:: Notes

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
        \begin{aligned}
            LLR(i) &\approx\ln\left(\frac{
                \max_{c\in\mathcal{C}_{i,1}} \Pr\left(c\lvert\mathbf{p}\right)
                    \exp\left(-\frac{1}{N_o}\left|y-c\right|^2\right)
                }{
                \max_{c\in\mathcal{C}_{i,0}} \Pr\left(c\lvert\mathbf{p}\right)
                    \exp\left(-\frac{1}{N_o}\left|y-c\right|^2\right)
                }\right)\\
                &= \max_{c\in\mathcal{C}_{i,1}}
                    \left(\ln\left(\Pr\left(c\lvert\mathbf{p}\right)\right)-\frac{|y-c|^2}{N_o}\right) -
                 \max_{c\in\mathcal{C}_{i,0}}\left( \ln\left(\Pr\left(c\lvert\mathbf{p}\right)\right) - \frac{|y-c|^2}{N_o}\right)
                .
        \end{aligned}

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.mapping import Mapper, Demapper

        mapper = Mapper("qam", 4)
        demapper = Demapper("app", "qam", 4)

        bits = torch.randint(0, 2, (10, 100), dtype=torch.float32)
        symbols = mapper(bits)

        # Add noise
        noise = 0.1 * torch.randn_like(symbols)
        y = symbols + noise

        # Compute LLRs
        llr = demapper(y, no=0.01)
        print(llr.shape)
        # torch.Size([10, 100])
    """

    def __init__(
        self,
        demapping_method: str,
        constellation_type: Optional[str] = None,
        num_bits_per_symbol: Optional[int] = None,
        constellation: Optional[Constellation] = None,
        hard_out: bool = False,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(precision=precision, device=device, **kwargs)

        # Create constellation object
        self._constellation = Constellation.check_or_create(
            constellation_type=constellation_type,
            num_bits_per_symbol=num_bits_per_symbol,
            constellation=constellation,
            precision=precision,
            device=device,
        )

        num_bits_per_symbol = self._constellation.num_bits_per_symbol

        self._logits2llrs = SymbolLogits2LLRs(
            demapping_method,
            num_bits_per_symbol,
            hard_out=hard_out,
            precision=precision,
            device=device,
            **kwargs,
        )

        # A Python scalar keeps the clamp tied to `no`'s dtype and device.
        self._no_threshold = float(
            np.finfo(dtypes[self.precision]["np"]["dtype"]).tiny
        )

    @property
    def constellation(self) -> Constellation:
        """Constellation used by the Demapper"""
        return self._constellation

    def call(
        self,
        y: torch.Tensor,
        no: torch.Tensor,
        prior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Reshape constellation points to [1,...1,num_points]
        points_shape = [1] * y.dim() + list(self.constellation.points.shape)
        points = self.constellation.points.reshape(points_shape)

        # Compute squared distances from y to all points
        # shape [...,n,num_points]
        squared_dist = (y.unsqueeze(-1) - points).abs().square()

        # Add a dummy dimension for broadcasting. This is not needed when no
        # is a scalar, but also does not do any harm.
        no = no.unsqueeze(-1)
        # Deal with zero or very small values.
        no = torch.clamp_min(no, self._no_threshold)

        # Compute exponents
        exponents = -squared_dist / no

        llr = self._logits2llrs(exponents, prior)

        # Reshape LLRs to [...,n*num_bits_per_symbol]
        out_shape = list(y.shape[:-1]) + [
            y.shape[-1] * self.constellation.num_bits_per_symbol
        ]
        llr_reshaped = llr.reshape(out_shape)

        return llr_reshaped


class SymbolDemapper(Block):
    r"""
    Computes normalized log-probabilities (logits) or hard-decisions on symbols
    for a tensor of received symbols.

    Prior knowledge on the transmitted constellation points can be optionally provided.
    The demapping function is fully differentiable when soft-values are
    computed.

    :param constellation_type: Type of constellation. One of "qam", "pam", or "custom".
        For "custom", an instance of :class:`~sionna.phy.mapping.Constellation`
        must be provided.
    :param num_bits_per_symbol: Number of bits per constellation symbol,
        e.g., 4 for QAM16. Only required for ``constellation_type`` in ["qam", "pam"].
    :param constellation: If no constellation is provided, ``constellation_type``
        and ``num_bits_per_symbol`` must be provided. Defaults to `None`.
    :param hard_out: If `True`, the demapper provides hard-decided symbols instead of
        soft-values. Defaults to `False`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for tensor operations. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input y: [..., n], `torch.complex`.
        Received symbols.
    :input no: Scalar or [..., n], `torch.float`.
        Noise variance estimate. It can be provided either as scalar
        for the entire input batch or as a tensor that is "broadcastable" to
        ``y``.
    :input prior: `None` (default) | [num_points] or [..., num_points], `torch.float`.
        Prior for every symbol as log-probabilities (logits).
        It can be provided either as a tensor of shape `[num_points]` for the
        entire input batch, or as a tensor that is "broadcastable"
        to `[..., n, num_points]`.

    :output logits: [..., n, num_points] or [..., n], `torch.float` or `torch.int32`.
        A tensor of shape `[..., n, num_points]` of logits for every constellation
        point if ``hard_out`` is set to `False`.
        Otherwise, a tensor of shape `[..., n]` of hard-decisions on the symbols.

    .. rubric:: Notes

    The normalized log-probability for the constellation point :math:`c` is computed according to

    .. math::
        \ln\left(\Pr\left(c \lvert y,\mathbf{p}\right)\right) = \ln\left( \frac{\exp\left(-\frac{|y-c|^2}{N_0} + p_c \right)}{\sum_{c'\in\mathcal{C}} \exp\left(-\frac{|y-c'|^2}{N_0} + p_{c'} \right)} \right)

    where :math:`\mathcal{C}` is the set of constellation points used for modulation,
    and :math:`\mathbf{p} = \left\{p_c \lvert c \in \mathcal{C}\right\}` the prior information on constellation points given as log-probabilities
    and which is set to :math:`\mathbf{0}` if no prior information on the constellation points is assumed to be available.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.mapping import Mapper, SymbolDemapper

        mapper = Mapper("qam", 4)
        demapper = SymbolDemapper("qam", 4)

        bits = torch.randint(0, 2, (10, 100), dtype=torch.float32)
        symbols = mapper(bits)

        # Add noise
        noise = 0.1 * torch.randn_like(symbols)
        y = symbols + noise

        # Compute symbol logits
        logits = demapper(y, no=0.01)
        print(logits.shape)
        # torch.Size([10, 25, 16])
    """

    def __init__(
        self,
        constellation_type: Optional[str] = None,
        num_bits_per_symbol: Optional[int] = None,
        constellation: Optional[Constellation] = None,
        hard_out: bool = False,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(precision=precision, device=device, **kwargs)
        self._hard_out = hard_out

        # Create constellation object
        self._constellation = Constellation.check_or_create(
            constellation_type=constellation_type,
            num_bits_per_symbol=num_bits_per_symbol,
            constellation=constellation,
            precision=precision,
            device=device,
        )

        # A Python scalar keeps the clamp tied to `no`'s dtype and device.
        self._no_threshold = float(
            np.finfo(dtypes[self.precision]["np"]["dtype"]).tiny
        )

    def call(
        self,
        y: torch.Tensor,
        no: torch.Tensor,
        prior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        points = expand_to_rank(self._constellation.points, y.dim() + 1, axis=0)
        y = y.unsqueeze(-1)
        squared_dist = (y - points).abs().square()

        no = expand_to_rank(no, squared_dist.dim(), axis=-1)
        no = torch.clamp_min(no, self._no_threshold)
        exp = -squared_dist / no

        if prior is not None:
            prior = expand_to_rank(prior, exp.dim(), axis=0)
            exp = exp + prior

        if self._hard_out:
            return exp.argmax(dim=-1).to(torch.int32)
        else:
            return F.log_softmax(exp, dim=-1)


class SymbolLogits2LLRs(Block):
    r"""
    Computes log-likelihood ratios (LLRs) or hard-decisions on bits
    from a tensor of logits (i.e., unnormalized log-probabilities) on constellation points.

    Prior knowledge on the bits can be optionally provided.

    :param method: Method used for computing the LLRs. One of "app" or "maxlog".
    :param num_bits_per_symbol: Number of bits per constellation symbol,
        e.g., 4 for QAM16.
    :param hard_out: If `True`, the layer provides hard-decided bits instead of
        soft-values. Defaults to `False`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for tensor operations. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input logits: [..., n, num_points], `torch.float`.
        Logits on constellation points.
    :input prior: `None` (default) | [num_bits_per_symbol] or [..., n, num_bits_per_symbol], `torch.float`.
        Prior for every bit as LLRs.
        It can be provided either as a tensor of shape `[num_bits_per_symbol]`
        for the entire input batch, or as a tensor that is "broadcastable"
        to `[..., n, num_bits_per_symbol]`.

    :output llr: [..., n, num_bits_per_symbol], `torch.float`.
        LLRs or hard-decisions for every bit.

    .. rubric:: Notes

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
    sets of :math:`2^{K-1}` constellation points for which the :math:`i\text{th}` bit is
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
        \begin{aligned}
            LLR(i) &\approx\ln\left(\frac{
                \max_{c\in\mathcal{C}_{i,1}} \Pr\left(c\lvert\mathbf{p}\right)
                    e^{z_c}
                }{
                \max_{c\in\mathcal{C}_{i,0}} \Pr\left(c\lvert\mathbf{p}\right)
                    e^{z_c}
                }\right)
                .
        \end{aligned}

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.mapping import SymbolLogits2LLRs

        converter = SymbolLogits2LLRs("app", 4)  # 16-QAM
        logits = torch.randn(10, 25, 16)  # 10 batches, 25 symbols, 16 constellation points
        llr = converter(logits)
        print(llr.shape)
        # torch.Size([10, 25, 4])
    """

    def __init__(
        self,
        method: str,
        num_bits_per_symbol: int,
        *,
        hard_out: bool = False,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(precision=precision, device=device, **kwargs)
        check_one_of(
            method,
            ("app", "maxlog"),
            name="method",
            message="method must be one of: 'app', 'maxlog'",
        )
        self._method = method
        self._hard_out = hard_out
        self._num_bits_per_symbol = num_bits_per_symbol

        # Binary representation of all symbol indices [num_points, num_bits_per_symbol]
        a = _compute_binary_labels(num_bits_per_symbol)

        # Compute symbol indices for which each bit is 0 or 1
        # [num_points/2, num_bits_per_symbol]
        c0 = np.array([np.where(a[:, i] == 0)[0] for i in range(num_bits_per_symbol)]).T
        c1 = np.array([np.where(a[:, i] == 1)[0] for i in range(num_bits_per_symbol)]).T
        self._c0 = torch.tensor(c0, dtype=torch.int64, device=self.device)
        self._c1 = torch.tensor(c1, dtype=torch.int64, device=self.device)

        # Array of labels from {-1, 1} of all symbols
        # [num_points, num_bits_per_symbol]
        a = 2 * a - 1
        self._a = torch.tensor(a, dtype=self.dtype, device=self.device)

    @property
    def num_bits_per_symbol(self) -> int:
        """Number of bits per symbol"""
        return self._num_bits_per_symbol

    def call(
        self, logits: torch.Tensor, prior: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Compute exponents
        exponents = logits

        # Gather exponents for all bits
        # shape [...,n,num_points/2,num_bits_per_symbol]
        # We need to gather along the last dimension using the indices in _c0 and _c1
        # Transpose to make gathering easier, then transpose back
        exp0 = exponents[..., self._c0]
        exp1 = exponents[..., self._c1]

        # Process the prior information
        if prior is not None:
            # Expanding `prior` such that it is broadcastable with
            # shape [..., n or 1, 1, num_bits_per_symbol]
            prior = expand_to_rank(prior, logits.dim(), axis=0)
            prior = prior.unsqueeze(-2)

            # Expand the symbol labeling to be broadcastable with prior
            # shape [..., 1, num_points, num_bits_per_symbol]
            a = expand_to_rank(self._a, prior.dim(), axis=0)

            # Compute the prior probabilities on symbols exponents
            # shape [..., n or 1, num_points]
            exp_ps = F.logsigmoid(a * prior).sum(dim=-1)

            # Gather prior probability symbol for all bits
            # shape [..., n or 1, num_points/2, num_bits_per_symbol]
            exp_ps0 = exp_ps[..., self._c0]
            exp_ps1 = exp_ps[..., self._c1]

        # Compute LLRs using the definition log( Pr(b=1)/Pr(b=0) )
        # shape [..., n, num_bits_per_symbol]
        if self._method == "app":
            if prior is not None:
                llr = torch.logsumexp(exp_ps1 + exp1, dim=-2) - torch.logsumexp(
                    exp_ps0 + exp0, dim=-2
                )
            else:
                llr = torch.logsumexp(exp1, dim=-2) - torch.logsumexp(exp0, dim=-2)
        else:  # maxlog
            if prior is not None:
                llr = (exp_ps1 + exp1).max(dim=-2).values - (exp_ps0 + exp0).max(
                    dim=-2
                ).values
            else:
                llr = exp1.max(dim=-2).values - exp0.max(dim=-2).values

        if self._hard_out:
            return hard_decisions(llr)
        else:
            return llr


class LLRs2SymbolLogits(Block):
    r"""
    Computes logits (i.e., unnormalized log-probabilities) or hard decisions
    on constellation points from a tensor of log-likelihood ratios (LLRs) on bits.

    :param num_bits_per_symbol: Number of bits per constellation symbol,
        e.g., 4 for QAM16.
    :param hard_out: If `True`, the layer provides hard-decided constellation points
        instead of soft-values. Defaults to `False`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for tensor operations. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input llrs: [..., n, num_bits_per_symbol], `torch.float`.
        LLRs for every bit.

    :output logits: [..., n, num_points], `torch.float` or [..., n], `torch.int32`.
        Logits or hard-decisions on constellation points.

    .. rubric:: Notes

    The logit for the constellation point :math:`c`
    is computed according to

    .. math::
        \begin{aligned}
            \log{\left(\Pr\left(c\lvert LLRs \right)\right)}
                &= \log{\left(\prod_{k=0}^{K-1} \Pr\left(b_k = \ell(c)_k \lvert LLRs \right)\right)}\\
                &= \log{\left(\prod_{k=0}^{K-1} \text{sigmoid}\left(LLR(k) \ell(c)_k\right)\right)}\\
                &= \sum_{k=0}^{K-1} \log{\left(\text{sigmoid}\left(LLR(k) \ell(c)_k\right)\right)}
        \end{aligned}

    where :math:`\ell(c)_k` is the :math:`k^{th}` bit label of :math:`c`, where 0 is
    replaced by -1.
    The definition of the LLR has been
    chosen such that it is equivalent with that of logits. This is
    different from many textbooks in communications, where the LLR is
    defined as :math:`LLR(i) = \ln\left(\frac{\Pr\left(b_i=0\lvert y\right)}{\Pr\left(b_i=1\lvert y\right)}\right)`.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.mapping import LLRs2SymbolLogits

        converter = LLRs2SymbolLogits(4)  # 16-QAM
        llr = torch.randn(10, 25, 4)  # 10 batches, 25 symbols, 4 bits per symbol
        logits = converter(llr)
        print(logits.shape)
        # torch.Size([10, 25, 16])
    """

    def __init__(
        self,
        num_bits_per_symbol: int,
        hard_out: bool = False,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(precision=precision, device=device, **kwargs)

        self._hard_out = hard_out
        self._num_bits_per_symbol = num_bits_per_symbol

        # Binary labels from {-1, 1} of all symbols [num_points, num_bits_per_symbol]
        a = 2 * _compute_binary_labels(num_bits_per_symbol) - 1
        self._a = torch.tensor(a, dtype=self.dtype, device=self.device)

    @property
    def num_bits_per_symbol(self) -> int:
        """Number of bits per symbol"""
        return self._num_bits_per_symbol

    def call(self, llrs: torch.Tensor) -> torch.Tensor:
        # Expand the symbol labeling to be broadcastable with prior
        # shape [1, ..., 1, num_points, num_bits_per_symbol]
        a = expand_to_rank(self._a, llrs.dim(), axis=0)

        # Compute the prior probabilities on symbols exponents
        # shape [..., 1, num_points]
        llrs = llrs.unsqueeze(-2)
        logits = F.logsigmoid(a * llrs).sum(dim=-1)

        if self._hard_out:
            return logits.argmax(dim=-1).to(torch.int32)
        else:
            return logits


class SymbolLogits2Moments(Block):
    r"""
    Computes the mean and variance of a constellation from logits (unnormalized log-probabilities) on the
    constellation points.

    More precisely, given a constellation :math:`\mathcal{C} = \left[ c_0,\dots,c_{N-1} \right]` of size :math:`N`, this layer computes the mean and variance
    according to

    .. math::
        \begin{aligned}
            \mu &= \sum_{n = 0}^{N-1} c_n \Pr \left(c_n \lvert \mathbf{\ell} \right)\\
            \nu &= \sum_{n = 0}^{N-1} \left| c_n - \mu \right|^2 \Pr \left(c_n \lvert \mathbf{\ell} \right)
        \end{aligned}

    where :math:`\mathbf{\ell} = \left[ \ell_0, \dots, \ell_{N-1} \right]` are the logits, and

    .. math::
        \Pr \left(c_n \lvert \mathbf{\ell} \right) = \frac{\exp \left( \ell_n \right)}{\sum_{i=0}^{N-1} \exp \left( \ell_i \right) }.

    :param constellation_type: Type of constellation. One of "qam", "pam", or "custom".
        For "custom", an instance of :class:`~sionna.phy.mapping.Constellation`
        must be provided.
    :param num_bits_per_symbol: The number of bits per constellation symbol,
        e.g., 4 for QAM16. Only required for ``constellation_type`` in ["qam", "pam"].
    :param constellation: If no constellation is provided, ``constellation_type``
        and ``num_bits_per_symbol`` must be provided. Defaults to `None`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for tensor operations. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input logits: [..., n, num_points], `torch.float`.
        Logits on constellation points.

    :output mean: [..., n], `torch.complex`.
        Mean of the constellation.
    :output var: [..., n], `torch.float`.
        Variance of the constellation.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.mapping import SymbolLogits2Moments

        converter = SymbolLogits2Moments("qam", 4)
        logits = torch.randn(10, 25, 16)  # 10 batches, 25 symbols, 16 constellation points
        mean, var = converter(logits)
        print(mean.shape, var.shape)
        # torch.Size([10, 25]) torch.Size([10, 25])
    """

    def __init__(
        self,
        constellation_type: Optional[str] = None,
        num_bits_per_symbol: Optional[int] = None,
        constellation: Optional[Constellation] = None,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(precision=precision, device=device, **kwargs)

        # Create constellation object
        self._constellation = Constellation.check_or_create(
            constellation_type=constellation_type,
            num_bits_per_symbol=num_bits_per_symbol,
            constellation=constellation,
            precision=precision,
            device=device,
        )

    def call(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        p = F.softmax(logits, dim=-1)
        points = self._constellation()
        points = expand_to_rank(points, p.dim(), axis=0)

        # Compute weighted mean (p is real, points is complex)
        mean = (p * points).sum(dim=-1, keepdim=True)
        var = (p * (points - mean).abs().square()).sum(dim=-1)
        mean = mean.squeeze(-1)

        return mean, var


class SymbolInds2Bits(Block):
    r"""
    Transforms symbol indices to their binary representations.

    :param num_bits_per_symbol: Number of bits per constellation symbol.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for tensor operations. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input symbol_ind: `torch.Tensor`, `torch.int`.
        Symbol indices.

    :output bits: input.shape + [num_bits_per_symbol], `torch.float`.
        Binary representation of symbol indices.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.mapping import SymbolInds2Bits

        converter = SymbolInds2Bits(4)  # 16-QAM
        indices = torch.tensor([0, 5, 10, 15])
        bits = converter(indices)
        print(bits.shape)
        # torch.Size([4, 4])
    """

    def __init__(
        self,
        num_bits_per_symbol: int,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(precision=precision, device=device, **kwargs)
        # Binary representation of all symbol indices [num_symbols, num_bits_per_symbol]
        b = _compute_binary_labels(num_bits_per_symbol)
        self._bit_labels = torch.tensor(b, dtype=self.dtype, device=self.device)

    def call(self, symbol_ind: torch.Tensor) -> torch.Tensor:
        return self._bit_labels[symbol_ind]


class QAM2PAM(Object):
    r"""Transforms QAM symbol indices to PAM symbol indices.

    For indices in a QAM constellation, computes the corresponding indices
    for the two PAM constellations corresponding to the real and imaginary
    components of the QAM constellation.

    :param num_bits_per_symbol: Number of bits per QAM constellation symbol,
        e.g., 4 for QAM16.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for tensor operations. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input ind_qam: `torch.Tensor`, `torch.int32`.
        Indices in the QAM constellation.

    :output ind_pam1: `torch.Tensor`, `torch.int32`.
        Indices for the first component of the corresponding PAM modulation.
    :output ind_pam2: `torch.Tensor`, `torch.int32`.
        Indices for the second component of the corresponding PAM modulation.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.mapping import QAM2PAM

        converter = QAM2PAM(4)  # 16-QAM
        ind_qam = torch.tensor([0, 5, 10, 15])
        ind_pam1, ind_pam2 = converter(ind_qam)
        print(ind_pam1, ind_pam2)
    """

    def __init__(
        self,
        num_bits_per_symbol: int,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(precision=precision, device=device, **kwargs)
        num_bits_per_symbol = _check_square_qam_bit_width(num_bits_per_symbol)
        half_bits = num_bits_per_symbol // 2

        # Binary representation of all QAM symbol indices
        bits = _compute_binary_labels(num_bits_per_symbol)

        # Extract even/odd bit positions and compute PAM indices
        base = np.array([2**i for i in range(half_bits - 1, -1, -1)])
        pam1_ind = (bits[:, 0::2] * base).sum(axis=1)
        pam2_ind = (bits[:, 1::2] * base).sum(axis=1)

        self._pam1_ind = torch.tensor(pam1_ind, dtype=torch.int64, device=self.device)
        self._pam2_ind = torch.tensor(pam2_ind, dtype=torch.int64, device=self.device)

    def __call__(self, ind_qam: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ind_pam1 = self._pam1_ind[ind_qam]
        ind_pam2 = self._pam2_ind[ind_qam]
        return ind_pam1, ind_pam2


class PAM2QAM(Object):
    r"""Transforms PAM symbol indices/logits to QAM symbol indices/logits.

    For two PAM constellation symbol indices or logits, corresponding to
    the real and imaginary components of a QAM constellation,
    compute the QAM symbol index or logits.

    :param num_bits_per_symbol: Number of bits per QAM constellation symbol,
        e.g., 4 for QAM16.
    :param hard_in_out: Determines if inputs and outputs are indices or logits
        over constellation symbols. Defaults to `True`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for tensor operations. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input pam1: `torch.Tensor`, `torch.int32`, or [..., 2\*\*(num_bits_per_symbol/2)], `torch.float`.
        Indices or logits for the first PAM constellation.
    :input pam2: `torch.Tensor`, `torch.int32`, or [..., 2\*\*(num_bits_per_symbol/2)], `torch.float`.
        Indices or logits for the second PAM constellation.

    :output qam: `torch.Tensor`, `torch.int32`, or [..., 2\*\*num_bits_per_symbol], `torch.float`.
        Indices or logits for the corresponding QAM constellation.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.mapping import PAM2QAM

        converter = PAM2QAM(4)  # 16-QAM
        ind_pam1 = torch.tensor([0, 1, 2, 3])
        ind_pam2 = torch.tensor([0, 1, 2, 3])
        ind_qam = converter(ind_pam1, ind_pam2)
        print(ind_qam)
    """

    def __init__(
        self,
        num_bits_per_symbol: int,
        hard_in_out: bool = True,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(precision=precision, device=device, **kwargs)
        num_bits_per_symbol = _check_square_qam_bit_width(num_bits_per_symbol)
        half_bits = num_bits_per_symbol // 2
        num_pam_symbols = 2**half_bits

        # Binary representation for PAM indices [num_pam, half_bits]
        pam_bits = _compute_binary_labels(half_bits)

        # Create interleaved bits for all (i,j) combinations
        # Shape: [num_pam, num_pam, num_bits_per_symbol]
        b1 = pam_bits[:, None, :]  # [num_pam, 1, half_bits]
        b2 = pam_bits[None, :, :]  # [1, num_pam, half_bits]

        # Interleave: even positions from b1, odd positions from b2
        interleaved = np.zeros(
            [num_pam_symbols, num_pam_symbols, num_bits_per_symbol], np.int32
        )
        interleaved[:, :, 0::2] = b1
        interleaved[:, :, 1::2] = b2

        # Compute QAM indices
        qam_base = np.array([2**k for k in range(num_bits_per_symbol - 1, -1, -1)])
        ind = (interleaved * qam_base).sum(axis=-1)

        self._qam_ind = torch.tensor(ind, dtype=torch.int64, device=self.device)
        self._hard_in_out = hard_in_out

    def __call__(self, pam1: torch.Tensor, pam2: torch.Tensor) -> torch.Tensor:
        # PAM indices to QAM indices
        if self._hard_in_out:
            shape = pam1.shape
            ind_pam1 = pam1.reshape(-1, 1)
            ind_pam2 = pam2.reshape(-1, 1)
            ind_pam = torch.cat([ind_pam1, ind_pam2], dim=-1)
            # Use advanced indexing
            ind_qam = self._qam_ind[ind_pam[:, 0], ind_pam[:, 1]]
            ind_qam = ind_qam.reshape(shape)
            return ind_qam

        # PAM logits to QAM logits
        else:
            # Compute all combination of sums of logits
            logits_mat = pam1.unsqueeze(-1) + pam2.unsqueeze(-2)

            # Flatten to a vector
            logits = flatten_last_dims(logits_mat)

            # Gather symbols in the correct order
            gather_ind = self._qam_ind.reshape(-1)
            logits = logits[..., gather_ind]
            return logits


class BinarySource(Block):
    """
    Generates a random binary tensor.

    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for tensor operations. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input shape: 1D tensor/array/list, `int`.
        Desired shape of the output tensor.

    :output bits: ``shape``, `torch.float`.
        Tensor filled with random binary values.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.mapping import BinarySource

        source = BinarySource()
        bits = source([10, 100])
        print(bits.shape)
        # torch.Size([10, 100])
    """

    def __init__(
        self,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(precision=precision, device=device, **kwargs)

    def call(
        self, inputs: Union[List[int], Tuple[int, ...], torch.Size]
    ) -> torch.Tensor:
        # Uses smart randint that switches to global RNG in compiled mode for graph fusion
        return randint(
            0,
            2,
            list(inputs),
            dtype=self.dtype,
            device=self.device,
            generator=self.torch_rng,
        )


class SymbolSource(Block):
    r"""
    Generates a tensor of random constellation symbols.

    Optionally, the symbol indices and/or binary representations of the
    constellation symbols can be returned.

    :param constellation_type: Type of constellation. One of "qam", "pam", or "custom".
        For "custom", an instance of :class:`~sionna.phy.mapping.Constellation`
        must be provided.
    :param num_bits_per_symbol: Number of bits per constellation symbol,
        e.g., 4 for QAM16. Only required for ``constellation_type`` in ["qam", "pam"].
    :param constellation: If no constellation is provided, ``constellation_type``
        and ``num_bits_per_symbol`` must be provided. Defaults to `None`.
    :param return_indices: If enabled, the function also returns the symbol indices.
        Defaults to `False`.
    :param return_bits: If enabled, the function also returns the binary symbol
        representations (i.e., bit labels). Defaults to `False`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for tensor operations. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input shape: 1D tensor/array/list, `int`.
        Desired shape of the output tensor. Must not be empty.

    :output symbols: ``shape``, `torch.complex`.
        Tensor filled with random symbols of the chosen ``constellation_type``.
    :output symbol_indices: ``shape``, `torch.int32`.
        Tensor filled with the symbol indices.
        Only returned if ``return_indices`` is `True`.
    :output bits: [``shape[:-1]``, ``shape[-1] * num_bits_per_symbol``], `torch.float`.
        Tensor filled with the binary symbol representations (i.e., bit labels).
        Only returned if ``return_bits`` is `True`.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.mapping import SymbolSource

        source = SymbolSource("qam", 4)
        symbols = source([10, 100])
        print(symbols.shape)
        # torch.Size([10, 100])
    """

    def __init__(
        self,
        constellation_type: Optional[str] = None,
        num_bits_per_symbol: Optional[int] = None,
        constellation: Optional[Constellation] = None,
        return_indices: bool = False,
        return_bits: bool = False,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(precision=precision, device=device, **kwargs)
        constellation = Constellation.check_or_create(
            constellation_type=constellation_type,
            num_bits_per_symbol=num_bits_per_symbol,
            constellation=constellation,
            precision=precision,
            device=device,
        )
        self._num_bits_per_symbol = constellation.num_bits_per_symbol
        self._return_indices = return_indices
        self._return_bits = return_bits
        self._binary_source = BinarySource(precision=precision, device=device)
        self._mapper = Mapper(
            constellation=constellation,
            return_indices=return_indices,
            precision=precision,
            device=device,
        )

    def call(
        self, inputs: Union[List[int], Tuple[int, ...], torch.Size]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        shape = list(inputs)
        if not shape:
            raise ValueError("`shape` must have at least one dimension")

        # Must not scale the last entry in place: for a tensor-valued
        # ``inputs``, the entries alias the caller's tensor.
        bits_shape = shape[:-1] + [shape[-1] * self._num_bits_per_symbol]

        b = self._binary_source(bits_shape)
        if self._return_indices:
            x, ind = self._mapper(b)
        else:
            x = self._mapper(b)

        result = x
        if self._return_indices or self._return_bits:
            result = [result]
        if self._return_indices:
            result.append(ind)
        if self._return_bits:
            result.append(b)

        return result


class QAMSource(SymbolSource):
    r"""
    Generates a tensor of random QAM symbols.

    Optionally, the symbol indices and/or binary representations of the
    constellation symbols can be returned.

    :param num_bits_per_symbol: Number of bits per constellation symbol,
        e.g., 4 for QAM16.
    :param return_indices: If enabled, the function also returns the symbol indices.
        Defaults to `False`.
    :param return_bits: If enabled, the function also returns the binary symbol
        representations (i.e., bit labels). Defaults to `False`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for tensor operations. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input shape: 1D tensor/array/list, `int`.
        Desired shape of the output tensor. Must not be empty.

    :output symbols: ``shape``, `torch.complex`.
        Tensor filled with random QAM symbols.
    :output symbol_indices: ``shape``, `torch.int32`.
        Tensor filled with the symbol indices.
        Only returned if ``return_indices`` is `True`.
    :output bits: [``shape[:-1]``, ``shape[-1] * num_bits_per_symbol``], `torch.float`.
        Tensor filled with the binary symbol representations (i.e., bit labels).
        Only returned if ``return_bits`` is `True`.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.mapping import QAMSource

        source = QAMSource(4)  # 16-QAM
        symbols = source([10, 100])
        print(symbols.shape)
        # torch.Size([10, 100])
    """

    def __init__(
        self,
        num_bits_per_symbol: Optional[int] = None,
        return_indices: bool = False,
        return_bits: bool = False,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            constellation_type="qam",
            num_bits_per_symbol=num_bits_per_symbol,
            return_indices=return_indices,
            return_bits=return_bits,
            precision=precision,
            device=device,
            **kwargs,
        )


class PAMSource(SymbolSource):
    r"""
    Generates a tensor of random PAM symbols.

    Optionally, the symbol indices and/or binary representations of the
    constellation symbols can be returned.

    :param num_bits_per_symbol: Number of bits per constellation symbol,
        e.g., 2 for 4-PAM.
    :param return_indices: If enabled, the function also returns the symbol indices.
        Defaults to `False`.
    :param return_bits: If enabled, the function also returns the binary symbol
        representations (i.e., bit labels). Defaults to `False`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for tensor operations. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input shape: 1D tensor/array/list, `int`.
        Desired shape of the output tensor. Must not be empty.

    :output symbols: ``shape``, `torch.complex`.
        Tensor filled with random PAM symbols.
    :output symbol_indices: ``shape``, `torch.int32`.
        Tensor filled with the symbol indices.
        Only returned if ``return_indices`` is `True`.
    :output bits: [``shape[:-1]``, ``shape[-1] * num_bits_per_symbol``], `torch.float`.
        Tensor filled with the binary symbol representations (i.e., bit labels).
        Only returned if ``return_bits`` is `True`.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.mapping import PAMSource

        source = PAMSource(2)  # 4-PAM
        symbols = source([10, 100])
        print(symbols.shape)
        # torch.Size([10, 100])
    """

    def __init__(
        self,
        num_bits_per_symbol: Optional[int] = None,
        return_indices: bool = False,
        return_bits: bool = False,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            constellation_type="pam",
            num_bits_per_symbol=num_bits_per_symbol,
            return_indices=return_indices,
            return_bits=return_bits,
            precision=precision,
            device=device,
            **kwargs,
        )
