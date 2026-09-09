#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Utility functions for 5G NR physical layer processing."""

from typing import Optional, Tuple, Union
import numpy as np
import torch

from sionna._validation import (
    check_tensor_all,
    check_tensor_range,
    check_tensor_values_in,
)
from sionna.phy import config
from sionna.phy.utils import (
    MCSDecoder,
    TransportBlock,
    SingleLinkChannel,
    ebnodb2no,
)
from sionna.phy.channel import AWGN
from sionna.phy.mapping import Mapper, Demapper, Constellation, BinarySource


__all__ = [
    "generate_prng_seq",
    "decode_mcs_index",
    "calculate_num_coded_bits",
    "calculate_codeword_bits",
    "calculate_tb_size",
    "MCSDecoderNR",
    "TransportBlockNR",
    "CodedAWGNChannelNR",
]


def generate_prng_seq(length: int, c_init: int) -> np.ndarray:
    r"""Implements pseudo-random sequence generator as defined in Sec. 5.2.1
    in :cite:p:`3GPPTS38211` based on a length-31 Gold sequence.

    :param length: Desired output sequence length.
    :param c_init: Initialization sequence of the PRNG. Must be in the range
        of 0 to :math:`2^{31}-1`, matching the length-31 Gold-sequence state.
        TS 38.211 initializers typically already apply
        :math:`c_{\text{init}} \bmod 2^{31}`.

    :output seq: [``length``], `ndarray` of 0s and 1s.
        Containing the scrambling sequence.

    .. rubric:: Notes

    The initialization sequence ``c_init`` is application specific and is
    usually provided by higher layer protocols.

    .. rubric:: Examples

    .. code-block:: python

        from sionna.phy.nr.utils import generate_prng_seq

        seq = generate_prng_seq(100, 12345)
        print(seq.shape)
        # (100,)
    """
    # Check inputs for consistency
    if length % 1 != 0:
        raise ValueError("length must be a positive integer.")
    length = int(length)
    if length <= 0:
        raise ValueError("length must be a positive integer.")

    if c_init % 1 != 0:
        raise ValueError("c_init must be integer.")
    c_init = int(c_init)
    if c_init >= 2**31:
        raise ValueError("c_init must be in [0, 2^31-1].")
    if c_init < 0:
        raise ValueError("c_init must be in [0, 2^31-1].")

    # Internal parameters
    n_seq = 31  # Length of gold sequence
    n_c = 1600  # Defined in 5.2.1 in 38.211

    # Init sequences
    c = np.zeros(length)
    x1 = np.zeros(length + n_c + n_seq)
    x2 = np.zeros(length + n_c + n_seq)

    # int2bin
    bin_ = format(c_init, f"0{n_seq}b")
    c_init_arr = [int(x) for x in bin_[-n_seq:]] if n_seq else []
    c_init_arr = np.flip(c_init_arr)  # Reverse order

    # Init x1 and x2
    x1[0] = 1
    x2[0:n_seq] = c_init_arr

    # Run the generator
    for idx in range(length + n_c):
        x1[idx + 31] = np.mod(x1[idx + 3] + x1[idx], 2)
        x2[idx + 31] = np.mod(x2[idx + 3] + x2[idx + 2] + x2[idx + 1] + x2[idx], 2)

    # Update output sequence
    for idx in range(length):
        c[idx] = np.mod(x1[idx + n_c] + x2[idx + n_c], 2)

    return c


def decode_mcs_index(
    mcs_index: Union[int, torch.Tensor],
    table_index: Union[int, torch.Tensor] = 1,
    is_pusch: Union[bool, torch.Tensor] = True,
    transform_precoding: Union[bool, torch.Tensor] = False,
    pi2bpsk: Union[bool, torch.Tensor] = False,
    check_index_validity: bool = True,
    verbose: bool = False,
    device: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Returns the modulation order and target coderate for a given MCS index.

    Implements MCS tables as defined in :cite:p:`3GPPTS38214` for PUSCH and PDSCH.

    :param mcs_index: MCS index (denoted as :math:`I_{MCS}` in
        :cite:p:`3GPPTS38214`). Accepted values are ``{0,1,...28}``.
    :param table_index: MCS table index from :cite:p:`3GPPTS38214`. Accepted values
        are ``{1,2,3,4}``.
    :param is_pusch: Specifies whether the 5G NR physical channel is of
        "PUSCH" type. If `False`, then the "PDSCH" channel is considered.
    :param transform_precoding: Specifies whether the MCS tables described in
        Sec. 6.1.4.1 of :cite:p:`3GPPTS38214` are applied. Only relevant for "PUSCH".
    :param pi2bpsk: Specifies whether the higher-layer parameter `tp-pi2BPSK`
        described in Sec. 6.1.4.1 of :cite:p:`3GPPTS38214` is applied. Only relevant
        for "PUSCH".
    :param check_index_validity: If `True`, a ValueError is raised if the
        input MCS indices are not valid for the given configuration.
    :param verbose: If `True`, additional information is printed.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :output modulation_order: [...], `torch.int32`.
        Modulation order, i.e., number of bits per symbol,
        associated with the input MCS index.

    :output target_rate: [...], `torch.float32`.
        Target coderate associated with the input MCS index.

    .. rubric:: Examples

    .. code-block:: python

        from sionna.phy.nr.utils import decode_mcs_index
        import torch

        # Scalar input
        mod_order, rate = decode_mcs_index(14, table_index=1)
        print(f"Modulation order: {mod_order.item()}, Target rate: {rate.item():.3f}")

        # Tensor input
        mcs_indices = torch.tensor([10, 14, 20])
        mod_orders, rates = decode_mcs_index(mcs_indices, table_index=1)
    """
    from sionna.phy.utils import scalar_to_shaped_tensor

    if device is None:
        device = config.device

    # Reject non-integer MCS indices before the int32 cast truncates them.
    # Cover Python and NumPy scalars; torch tensors are handled below.
    if isinstance(mcs_index, torch.Tensor):
        if mcs_index.dtype.is_floating_point:
            check_tensor_all(
                mcs_index == torch.round(mcs_index),
                name="mcs_index",
                message="MCS index must contain integer values",
            )
        mcs_index = mcs_index.to(dtype=torch.int32, device=device)
    else:
        if isinstance(mcs_index, (float, np.floating)):
            if not float(mcs_index).is_integer():
                raise ValueError("MCS index must be an integer")
            mcs_index = int(mcs_index)
        elif isinstance(mcs_index, (bool, np.bool_)):
            raise ValueError("MCS index must be an integer")
        elif isinstance(mcs_index, (int, np.integer)):
            mcs_index = int(mcs_index)
        else:
            raise TypeError(
                "mcs_index must be an int, float, NumPy scalar, or torch.Tensor"
            )
        mcs_index = torch.tensor(mcs_index, dtype=torch.int32, device=device)

    shape = list(mcs_index.shape)

    # Cast and reshape inputs to match mcs_index shape
    table_index = scalar_to_shaped_tensor(table_index, torch.int32, shape, device)
    is_pusch = scalar_to_shaped_tensor(is_pusch, torch.bool, shape, device)
    transform_precoding = scalar_to_shaped_tensor(
        transform_precoding, torch.bool, shape, device
    )
    pi2bpsk = scalar_to_shaped_tensor(pi2bpsk, torch.bool, shape, device)

    if check_index_validity:
        check_tensor_all(
            mcs_index >= 0,
            name="mcs_index",
            message="MCS index cannot be negative",
        )
        check_tensor_all(
            mcs_index <= 28,
            name="mcs_index",
            message="MCS index cannot be higher than 28",
        )
        valid_tables = (table_index >= 1) & (table_index <= 4)
        check_tensor_all(
            valid_tables,
            name="table_index",
            message="table_index must contain values in [1, 2, 3, 4]",
        )

    # Modulation orders lookup table
    # [2, 4, 29]: [channel_type, table_index, mcs_index]
    mod_orders = torch.tensor(
        [
            [  # PUSCH with transform_precoding
                # Table 1 (q=1)
                [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 6,
                 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, -1],
                # Table 2 (q=1)
                [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4,
                 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, -1],
                # Table 3 (dummy)
                [-1] * 29,
                # Table 4 (dummy)
                [-1] * 29
            ],
            [  # PDSCH | transform_precoding is False
                # Table 1
                [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 6,
                 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                # Table 2
                [2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6,
                 6, 6, 6, 8, 8, 8, 8, 8, 8, 8, 8, -1],
                # Table 3
                [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4,
                 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6],
                # Table 4
                [2, 2, 2, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 8, 8, 8,
                 8, 8, 8, 8, 8, 10, 10, 10, 10, -1, -1]
            ]
        ],
        dtype=torch.int32,
        device=device,
    )

    # Target rates lookup table (x1024)
    target_rates = torch.tensor(
        [
            [  # PUSCH with transform_precoding
                # Table 1 (q=1)
                [240, 314, 193, 251, 308, 379, 449, 526, 602,
                 679, 340, 378, 434, 490, 553, 616, 658, 466, 517,
                 567, 616, 666, 719, 772, 822, 873, 910, 948, -1],
                # Table 2 (q=1)
                [60, 80, 100, 128, 156, 198, 120, 157,
                 193, 251, 308, 379, 449, 526, 602, 679, 378, 434,
                 490, 553, 616, 658, 699, 772, 567, 616, 666, 772, -1],
                # Table 3 (dummy)
                [-1] * 29,
                # Table 4 (dummy)
                [-1] * 29
            ],
            [  # PDSCH | transform_precoding is False
                # Table 1
                [120, 157, 193, 251, 308, 379, 449, 526, 602, 679,
                 340, 378, 434, 490, 553, 616, 658, 438, 466, 517,
                 567, 616, 666, 719, 772, 822, 873, 910, 948],
                # Table 2
                [120, 193, 308, 449, 602, 378, 434, 490, 553, 616,
                 658, 466, 517, 567, 616, 666, 719, 772, 822, 873,
                 682.5, 711, 754, 797, 841, 885, 916.5, 948, -1],
                # Table 3
                [30, 40, 50, 64, 78, 99, 120, 157, 193, 251, 308,
                 379, 449, 526, 602, 340, 378, 434, 490, 553, 616,
                 438, 466, 517, 567, 616, 666, 719, 772],
                # Table 4
                [120, 193, 449, 378, 490, 616, 466, 517, 567, 616,
                 666, 719, 772, 822, 873, 682.5, 711, 754, 797, 841,
                 885, 916.5, 948, 805.5, 853, 900.5, 948, -1, -1]
            ]
        ],
        dtype=torch.float32,
        device=device,
    )

    # Compute channel type index: 0 for PUSCH with transform_precoding,
    # 1 for PDSCH or no transform_precoding
    channel_type_idx = (~is_pusch | ~transform_precoding).to(torch.int32)

    # Flatten inputs for indexing, then reshape back
    orig_shape = mcs_index.shape
    mcs_flat = mcs_index.flatten()
    table_flat = (table_index - 1).flatten()
    channel_flat = channel_type_idx.flatten()

    # Use advanced indexing for batched lookup
    mod_orders_sel = mod_orders[channel_flat, table_flat, mcs_flat]
    target_rates_sel = target_rates[channel_flat, table_flat, mcs_flat]

    # Reshape back to original shape
    if len(orig_shape) > 0:
        mod_orders_sel = mod_orders_sel.reshape(orig_shape)
        target_rates_sel = target_rates_sel.reshape(orig_shape)

    if check_index_validity:
        check_tensor_all(
            mod_orders_sel >= 0,
            name="mcs_index",
            message=(
                "Invalid MCS index for the selected table and configuration"
            ),
        )

    #######################
    # Account for pi2BPSK #
    #######################
    q = torch.where(pi2bpsk, 1, 2)

    # Condition: channel_type == 0 AND ((table_index == 1 AND mcs < 2) OR
    #                                   (table_index == 2 AND mcs < 6))
    needs_q_correction = (
        (channel_type_idx == 0) &
        (((table_index == 1) & (mcs_index < 2)) |
         ((table_index == 2) & (mcs_index < 6)))
    )

    # Apply correction where needed
    mod_orders_sel = torch.where(
        needs_q_correction,
        mod_orders_sel * q,
        mod_orders_sel
    )
    target_rates_sel = torch.where(
        needs_q_correction,
        target_rates_sel / q.to(target_rates_sel.dtype),
        target_rates_sel
    )

    # Convert target rate from x1024 to actual rate
    target_rates_sel = target_rates_sel / 1024.0

    if verbose:
        print(f"Modulation order: {mod_orders_sel}")
        print(f"Target code rate: {target_rates_sel}")

    return mod_orders_sel, target_rates_sel


def _num_coded_bits_from_grid(
    modulation_order: int,
    num_prbs: int,
    num_ofdm_symbols: int,
    num_dmrs_per_prb: int,
    num_layers: int = 1,
    num_ov: int = 0,
    tb_scaling: float = 1.0,
    apply_tbs_re_limit: bool = True,
) -> int:
    """Shared RE/bit budget for TBS and rate-matching helpers."""
    if num_ofdm_symbols < 1 or num_ofdm_symbols > 14:
        raise ValueError("num_ofdm_symbols must be in [1, 14]")
    if num_prbs < 1 or num_prbs > 275:
        raise ValueError("num_prbs must be in [1, 275]")
    if tb_scaling not in (0.25, 0.5, 1.0):
        raise ValueError("tb_scaling must be 0.25, 0.5, or 1.0")

    n_re_per_prb = 12 * num_ofdm_symbols - num_dmrs_per_prb - num_ov
    # TS 38.214 caps N_RE at 156 only for transport-block size derivation
    if apply_tbs_re_limit:
        n_re_per_prb = min(156, n_re_per_prb)

    return int(
        tb_scaling * n_re_per_prb * num_prbs * modulation_order * num_layers
    )


def calculate_num_coded_bits(
    modulation_order: int,
    num_prbs: int,
    num_ofdm_symbols: int,
    num_dmrs_per_prb: int,
    num_layers: int = 1,
    num_ov: int = 0,
    tb_scaling: float = 1.0,
) -> int:
    r"""Computes the coded-bit budget used as the TBS intermediate in
    TS 38.214 Sec. 5.1.3.2 / 6.1.4.2.

    Resource elements per PRB are capped at 156, as required for transport-
    block size derivation. For the uncapped rate-matching codeword budget
    :math:`G`, use :func:`~sionna.phy.nr.utils.calculate_codeword_bits`
    instead (see also :attr:`~sionna.phy.nr.PUSCHConfig.num_coded_bits`).

    :param modulation_order: Modulation order, i.e., number of bits per QAM
        symbol.
    :param num_prbs: Total number of allocated PRBs per OFDM symbol, where
        1 PRB equals 12 subcarriers. Must not exceed 275.
    :param num_ofdm_symbols: Number of OFDM symbols allocated for
        transmission. Cannot be larger than 14.
    :param num_dmrs_per_prb: Number of DMRS (i.e., pilot) symbols per PRB
        that are `not` used for data transmission, across all
        ``num_ofdm_symbols`` OFDM symbols.
    :param num_layers: Number of MIMO layers.
    :param num_ov: Number of unused resource elements due to additional
        overhead as specified by higher layer.
    :param tb_scaling: TB scaling factor for PDSCH as defined in TS 38.214
        Tab. 5.1.3.2-2. Must contain values in {0.25, 0.5, 1.0}.

    :output num_coded_bits: `int`.
        Capped coded-bit budget used when deriving the transport block size.

    .. rubric:: Examples

    .. code-block:: python

        from sionna.phy.nr.utils import calculate_num_coded_bits

        num_bits = calculate_num_coded_bits(4, 50, 14, 12, 2)
        print(num_bits)
    """
    return _num_coded_bits_from_grid(
        modulation_order,
        num_prbs,
        num_ofdm_symbols,
        num_dmrs_per_prb,
        num_layers,
        num_ov,
        tb_scaling,
        apply_tbs_re_limit=True,
    )


def calculate_codeword_bits(
    modulation_order: int,
    num_prbs: int,
    num_ofdm_symbols: int,
    num_dmrs_per_prb: int,
    num_layers: int = 1,
    num_ov: int = 0,
    tb_scaling: float = 1.0,
) -> int:
    r"""Computes the uncapped rate-matching codeword bit budget :math:`G`.

    Unlike :func:`~sionna.phy.nr.utils.calculate_num_coded_bits`, this helper
    does **not** apply the TS 38.214 156-RE-per-PRB cap. That cap applies only
    when deriving the transport block size; the actual number of coded bits
    available for rate matching can be larger. This matches
    :attr:`~sionna.phy.nr.PUSCHConfig.num_coded_bits`.

    :param modulation_order: Modulation order, i.e., number of bits per QAM
        symbol.
    :param num_prbs: Total number of allocated PRBs per OFDM symbol, where
        1 PRB equals 12 subcarriers. Must not exceed 275.
    :param num_ofdm_symbols: Number of OFDM symbols allocated for
        transmission. Cannot be larger than 14.
    :param num_dmrs_per_prb: Number of DMRS (i.e., pilot) symbols per PRB
        that are `not` used for data transmission, across all
        ``num_ofdm_symbols`` OFDM symbols.
    :param num_layers: Number of MIMO layers.
    :param num_ov: Number of unused resource elements due to additional
        overhead as specified by higher layer.
    :param tb_scaling: TB scaling factor for PDSCH as defined in TS 38.214
        Tab. 5.1.3.2-2. Must contain values in {0.25, 0.5, 1.0}.

    :output num_coded_bits: `int`.
        Uncapped number of coded bits available for rate matching.

    .. rubric:: Examples

    .. code-block:: python

        from sionna.phy.nr.utils import calculate_codeword_bits

        g = calculate_codeword_bits(4, 1, 14, 0, 1)
        print(g)  # 672 (= 4 * 168), not capped to 156 REs
    """
    return _num_coded_bits_from_grid(
        modulation_order,
        num_prbs,
        num_ofdm_symbols,
        num_dmrs_per_prb,
        num_layers,
        num_ov,
        tb_scaling,
        apply_tbs_re_limit=False,
    )


def calculate_tb_size(
    modulation_order: Union[int, torch.Tensor],
    target_coderate: Union[float, torch.Tensor],
    target_tb_size: Optional[Union[int, float, torch.Tensor]] = None,
    num_coded_bits: Optional[Union[int, torch.Tensor]] = None,
    num_prbs: Optional[Union[int, torch.Tensor]] = None,
    num_ofdm_symbols: Optional[Union[int, torch.Tensor]] = None,
    num_dmrs_per_prb: Optional[Union[int, torch.Tensor]] = None,
    num_layers: Union[int, torch.Tensor] = 1,
    num_ov: Union[int, torch.Tensor] = 0,
    tb_scaling: Union[float, torch.Tensor] = 1.0,
    return_cw_length: bool = True,
    verbose: bool = False,
    device: Optional[str] = None,
) -> Tuple:
    r"""Calculates the transport block (TB) size for given system parameters.

    This function follows the procedure defined in TS 38.214 Sec.
    5.1.3.2 and Sec. 6.1.4.2 :cite:p:`3GPPTS38214`.

    :param modulation_order: Modulation order, i.e., number of bits per QAM
        symbol.
    :param target_coderate: Target coderate.
    :param target_tb_size: Target transport block size, i.e., number of
        information bits that can be encoded into a slot for the given slot
        configuration.
    :param num_coded_bits: Number of coded bits that can be fit into a given
        slot. If provided, ``num_prbs``, ``num_ofdm_symbols`` and
        ``num_dmrs_per_prb`` are ignored.
    :param num_prbs: Total number of allocated PRBs per OFDM symbol, where
        1 PRB equals 12 subcarriers. Must not exceed 275.
    :param num_ofdm_symbols: Number of OFDM symbols allocated for
        transmission. Cannot be larger than 14.
    :param num_dmrs_per_prb: Number of DMRS (i.e., pilot) symbols per PRB
        that are `not` used for data transmission, across all
        ``num_ofdm_symbols`` OFDM symbols.
    :param num_layers: Number of MIMO layers.
    :param num_ov: Number of unused resource elements due to additional
        overhead as specified by higher layer.
    :param tb_scaling: TB scaling factor for PDSCH as defined in TS 38.214
        Tab. 5.1.3.2-2.
    :param return_cw_length: If `True`, the function returns ``tb_size``,
        ``cb_size``, ``num_cb``, ``tb_crc_length``, ``cb_crc_length``,
        ``cw_length``. Otherwise, it does not return ``cw_length`` to reduce
        computation time.
    :param verbose: If `True`, additional information is printed.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :output tb_size: [...], `torch.int32`.
        Transport block (TB) size, i.e., how many information bits can be
        encoded into a slot for the given slot configuration.

    :output cb_size: [...], `torch.int32`.
        Code block (CB) size, i.e., the number of information bits per
        codeword, including the TB/CB CRC parity bits.

    :output num_cb: [...], `torch.int32`.
        Number of CBs that the TB is segmented into.

    :output tb_crc_length: [...], `torch.int32`.
        Length of the TB CRC.

    :output cb_crc_length: [...], `torch.int32`.
        Length of each CB CRC.

    :output cw_length: [..., N], `torch.int32`.
        Codeword length of each of the ``num_cbs`` codewords after LDPC
        encoding and rate-matching.
        Note that zeros are appended along the last axis to obtain a dense
        tensor. The total number of coded bits, ``num_coded_bits``, is the
        sum of ``cw_length`` across its last axis.
        Only returned if ``return_cw_length`` is `True`.

    .. rubric:: Examples

    .. code-block:: python

        from sionna.phy.nr.utils import calculate_tb_size

        tb_size, cb_size, num_cb, tb_crc, cb_crc, cw_len = calculate_tb_size(
            modulation_order=4,
            target_coderate=0.5,
            num_coded_bits=4800,
            num_layers=1
        )
        print(f"TB size: {tb_size}, CB size: {cb_size}, Num CBs: {num_cb}")
    """
    from sionna.phy.utils import scalar_to_shaped_tensor

    if device is None:
        device = config.device

    # Convert modulation_order to tensor
    if isinstance(modulation_order, (int, float)):
        modulation_order = torch.tensor(modulation_order, dtype=torch.int32, device=device)
    else:
        modulation_order = modulation_order.to(dtype=torch.int32, device=device)

    shape = list(modulation_order.shape)

    # Convert target_coderate to tensor
    if isinstance(target_coderate, (int, float)):
        target_coderate = torch.tensor(target_coderate, dtype=torch.float32, device=device)
        if shape:
            target_coderate = target_coderate.expand(shape)
    else:
        target_coderate = target_coderate.to(dtype=torch.float32, device=device)

    # Broadcast scalars
    num_layers = scalar_to_shaped_tensor(num_layers, torch.int32, shape, device)
    tb_scaling = scalar_to_shaped_tensor(tb_scaling, torch.float32, shape, device)

    # ---------------#
    # N. coded bits #
    # ---------------#
    if num_coded_bits is not None:
        if isinstance(num_coded_bits, (int, float)):
            num_coded_bits = torch.tensor(num_coded_bits, dtype=torch.int32, device=device)
            if shape:
                num_coded_bits = num_coded_bits.expand(shape)
        else:
            num_coded_bits = num_coded_bits.to(dtype=torch.int32, device=device)
    else:
        if (
            num_prbs is None
            or num_ofdm_symbols is None
            or num_dmrs_per_prb is None
        ):
            raise ValueError(
                "If num_coded_bits is None, then num_prbs, num_ofdm_symbols, "
                "and num_dmrs_per_prb must be specified"
            )
        # Vectorized TBS coded-bit budget (applies the 156-RE-per-PRB cap)
        num_prbs_t = scalar_to_shaped_tensor(num_prbs, torch.int32, shape, device)
        num_ofdm_symbols_t = scalar_to_shaped_tensor(
            num_ofdm_symbols, torch.int32, shape, device
        )
        num_dmrs_per_prb_t = scalar_to_shaped_tensor(
            num_dmrs_per_prb, torch.int32, shape, device
        )
        num_ov_t = scalar_to_shaped_tensor(num_ov, torch.int32, shape, device)

        check_tensor_range(
            num_ofdm_symbols_t,
            name="num_ofdm_symbols",
            minimum=1,
            maximum=14,
            message="num_ofdm_symbols must be in [1, 14]",
        )
        check_tensor_range(
            num_prbs_t,
            name="num_prbs",
            minimum=1,
            maximum=275,
            message="num_prbs must be in [1, 275]",
        )
        check_tensor_values_in(
            tb_scaling,
            (0.25, 0.5, 1.0),
            name="tb_scaling",
            message="tb_scaling must be 0.25, 0.5, or 1.0",
        )

        n_re_per_prb = 12 * num_ofdm_symbols_t - num_dmrs_per_prb_t - num_ov_t
        n_re_cap = torch.tensor(156, dtype=n_re_per_prb.dtype, device=device)
        n_re_per_prb = torch.minimum(n_re_per_prb, n_re_cap)
        num_coded_bits = (
            tb_scaling
            * n_re_per_prb.to(torch.float32)
            * num_prbs_t.to(torch.float32)
            * modulation_order.to(torch.float32)
            * num_layers.to(torch.float32)
        ).to(torch.int32)

    # --------------#
    # Target TB size #
    # --------------#
    if target_tb_size is not None:
        if isinstance(target_tb_size, (int, float)):
            target_tb_size = torch.tensor(target_tb_size, dtype=torch.float32, device=device)
            if shape:
                target_tb_size = target_tb_size.expand(shape)
        else:
            target_tb_size = target_tb_size.to(dtype=torch.float32, device=device)
    else:
        target_tb_size = target_coderate * num_coded_bits.to(torch.float32)

    # -----------------------------#
    # Quantized n. information bits #
    # -----------------------------#
    # For target_tb_size <= 3824
    log2_n_info = torch.log2(torch.clamp(target_tb_size, min=1.0))
    n_small = torch.clamp(torch.floor(log2_n_info) - 6, min=3.0)
    n_info_q_small = torch.clamp(
        2**n_small * torch.floor(target_tb_size / 2**n_small),
        min=24.0
    )

    # For target_tb_size > 3824
    log2_n_info_minus_24 = torch.log2(torch.clamp(target_tb_size - 24, min=1.0))
    n_large = torch.floor(log2_n_info_minus_24) - 5.0
    n_info_q_large = torch.clamp(
        2**n_large * torch.round((target_tb_size - 24) / 2**n_large),
        min=3840.0
    )

    n_info_q = torch.where(target_tb_size <= 3824, n_info_q_small, n_info_q_large)

    # -----------------#
    # N. of code blocks #
    # -----------------#
    # Case 1: n_info_q <= 3824
    num_cb_case1 = torch.ones_like(n_info_q, dtype=torch.int32)

    # Case 2: target_coderate <= 0.25
    num_cb_case2 = torch.ceil((n_info_q + 24) / 3816).to(torch.int32)

    # Case 3: n_info_q > 8424
    num_cb_case3 = torch.ceil((n_info_q + 24) / 8424).to(torch.int32)

    # Case 4: else
    num_cb_case4 = torch.ones_like(n_info_q, dtype=torch.int32)

    num_cb = torch.where(
        n_info_q <= 3824,
        num_cb_case1,
        torch.where(
            target_coderate <= 0.25,
            num_cb_case2,
            torch.where(n_info_q > 8424, num_cb_case3, num_cb_case4)
        )
    )

    # ----------------------#
    # TB size (n. info bits) #
    # ----------------------#
    # Table 5.1.3.2-1 from 38.214
    tab51321 = torch.tensor([
        -1, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128,
        136, 144, 152, 160, 168, 176, 184, 192, 208, 224, 240, 256,
        272, 288, 304, 320, 336, 352, 368, 384, 408, 432, 456, 480,
        504, 528, 552, 576, 608, 640, 672, 704, 736, 768, 808, 848,
        888, 928, 984, 1032, 1064, 1128, 1160, 1192, 1224, 1256,
        1288, 1320, 1352, 1416, 1480, 1544, 1608, 1672, 1736, 1800,
        1864, 1928, 2024, 2088, 2152, 2216, 2280, 2408, 2472, 2536,
        2600, 2664, 2728, 2792, 2856, 2976, 3104, 3240, 3368, 3496,
        3624, 3752, 3824], dtype=torch.int32, device=device)

    # For n_info_q <= 3824: find smallest TB size >= n_info_q using searchsorted
    idx = torch.searchsorted(tab51321, n_info_q.to(torch.int32))
    idx = torch.clamp(idx, max=len(tab51321) - 1)
    tb_size_small = tab51321[idx]

    # For n_info_q > 3824: Step 5 of 38.214 5.1.3.2
    tb_size_large = (8 * num_cb * torch.ceil((n_info_q + 24) / (8 * num_cb.to(torch.float32))) - 24).to(torch.int32)

    tb_size = torch.where(n_info_q <= 3824, tb_size_small, tb_size_large)

    # ----------------#
    # TB/CB CRC length #
    # ----------------#
    tb_crc_length = torch.where(tb_size > 3824,
                                 torch.tensor(24, dtype=torch.int32, device=device),
                                 torch.tensor(16, dtype=torch.int32, device=device))

    cb_crc_length = torch.where(num_cb > 1,
                                 torch.tensor(24, dtype=torch.int32, device=device),
                                 torch.tensor(0, dtype=torch.int32, device=device))

    # -------#
    # CB size #
    # -------#
    cb_size = (tb_size + tb_crc_length) // num_cb + cb_crc_length

    if verbose:
        print(f"Modulation order: {modulation_order}")
        print(f"Target coderate: {target_coderate}")
        print(f"Number of layers: {num_layers}")
        print("------------------")
        print(f"Info bits per TB: {tb_size}")
        print(f"TB CRC length: {tb_crc_length}")
        print(f"Total number of coded TB bits: {num_coded_bits}")
        print("------------------")
        print(f"Info bits per CB: {cb_size}")
        print(f"Number of CBs: {num_cb}")
        print(f"CB CRC length: {cb_crc_length}")

    if not return_cw_length:
        return tb_size, cb_size, num_cb, tb_crc_length, cb_crc_length

    # ---------------------------#
    # Codeword length for each CB #
    # ---------------------------#
    num_last_blocks = (num_coded_bits // (num_layers * modulation_order)) % num_cb
    cw_length_last_blocks = (num_layers * modulation_order *
                              torch.ceil(num_coded_bits.to(torch.float32) /
                                        (num_layers * modulation_order * num_cb).to(torch.float32))).to(torch.int32)

    num_first_blocks = num_cb - num_last_blocks
    cw_length_first_blocks = (num_layers * modulation_order *
                               torch.floor(num_coded_bits.to(torch.float32) /
                                          (num_layers * modulation_order * num_cb).to(torch.float32))).to(torch.int32)

    # For tensor outputs, we return the max num_cb and pad with zeros
    # Flatten for construction
    orig_shape = tb_size.shape
    num_last_flat = num_last_blocks.flatten()
    cw_last_flat = cw_length_last_blocks.flatten()
    num_first_flat = num_first_blocks.flatten()
    cw_first_flat = cw_length_first_blocks.flatten()
    num_cb_flat = num_cb.flatten()

    # Build codeword lengths: for each element, first blocks have cw_first, last have cw_last
    max_num_cb = num_cb_flat.max().item()
    batch_size = num_cb_flat.numel()

    # Create range tensor for comparison
    r = torch.arange(max_num_cb, device=device).unsqueeze(0)  # [1, max_num_cb]

    # Construct cw_length tensor
    cw_length = torch.where(
        r < num_first_flat.unsqueeze(1),
        cw_first_flat.unsqueeze(1),
        torch.where(
            r < num_cb_flat.unsqueeze(1),
            cw_last_flat.unsqueeze(1),
            torch.zeros(1, dtype=torch.int32, device=device)
        )
    )

    # Reshape to original shape + [max_num_cb]
    if len(orig_shape) > 0:
        cw_length = cw_length.reshape(list(orig_shape) + [max_num_cb])
    else:
        cw_length = cw_length.squeeze(0)

    if verbose:
        print(f"Output codeword lengths: {cw_length}")

    return tb_size, cb_size, num_cb, tb_crc_length, cb_crc_length, cw_length


class MCSDecoderNR(MCSDecoder):
    r"""Maps a Modulation and Coding Scheme (MCS) index to the
    corresponding modulation order, i.e., number of bits per symbol, and
    coderate for 5G-NR networks. Wraps
    :func:`~sionna.phy.nr.utils.decode_mcs_index` and inherits
    from :class:`~sionna.phy.utils.MCSDecoder`.

    :input mcs_index: [...], `torch.int32`.
        MCS index.

    :input mcs_table_index: [...], `torch.int32`.
        MCS table index. Different tables contain different mappings.

    :input mcs_category: [...], `torch.int32`.
        `0` for PUSCH, `1` for PDSCH channel.

    :input check_index_validity: `bool`.
        If `True`, a ValueError is raised if the input MCS indices are not
        valid for the given configuration. Defaults to `True`.

    :input transform_precoding: [...], `torch.bool` | `bool`.
        Specifies whether the MCS tables described in
        Sec. 6.1.4.1 of :cite:p:`3GPPTS38214` are applied.
        Only relevant for "PUSCH". Defaults to `False`.

    :input pi2bpsk: [...], `torch.bool` | `bool`.
        Specifies whether the higher-layer parameter `tp-pi2BPSK`
        described in Sec. 6.1.4.1 of :cite:p:`3GPPTS38214` is applied.
        Only relevant for "PUSCH". Defaults to `False`.

    :input verbose: `bool`.
        If `True`, additional information is printed. Defaults to `False`.

    :output modulation_order: [...], `torch.int32`.
        Modulation order corresponding to the input MCS index.

    :output target_coderate: [...], `torch.float32`.
        Target coderate corresponding to the input MCS index.

    .. rubric:: Examples

    .. code-block:: python

        from sionna.phy.nr.utils import MCSDecoderNR
        import torch

        decoder = MCSDecoderNR()
        # Scalar input
        mod_order, rate = decoder(mcs_index=14, mcs_table_index=1, mcs_category=0)
        print(f"Modulation order: {mod_order.item()}, Target rate: {rate.item():.3f}")

        # Tensor input
        mcs_indices = torch.tensor([10, 14, 20])
        mod_orders, rates = decoder(mcs_index=mcs_indices,
                                    mcs_table_index=1, mcs_category=0)
    """

    def call(
        self,
        mcs_index: Union[int, torch.Tensor],
        mcs_table_index: Union[int, torch.Tensor],
        mcs_category: Union[int, torch.Tensor],
        check_index_validity: bool = True,
        transform_precoding: Union[bool, torch.Tensor] = False,
        pi2bpsk: Union[bool, torch.Tensor] = False,
        verbose: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process MCS index to return modulation order and coderate."""
        # Convert mcs_category to is_pusch: 0 -> True (PUSCH), 1 -> False (PDSCH)
        if isinstance(mcs_category, torch.Tensor):
            check_tensor_values_in(
                mcs_category,
                (0, 1),
                name="mcs_category",
                message="mcs_category must contain values in {0, 1}",
            )
            is_pusch = mcs_category == 0
        else:
            if mcs_category not in (0, 1):
                raise ValueError(
                    "mcs_category must be 0 (PUSCH) or 1 (PDSCH)"
                )
            is_pusch = mcs_category == 0

        modulation_order, target_coderate = decode_mcs_index(
            mcs_index,
            table_index=mcs_table_index,
            is_pusch=is_pusch,
            transform_precoding=transform_precoding,
            pi2bpsk=pi2bpsk,
            check_index_validity=check_index_validity,
            verbose=verbose,
            device=self.device,
        )
        return modulation_order, target_coderate


class TransportBlockNR(TransportBlock):
    r"""Computes the number and size (measured in number of bits) of code
    blocks within a 5G-NR compliant transport block, given the modulation
    order, coderate and the total number of coded bits of a transport block.
    Used in :class:`~sionna.sys.PHYAbstraction`. Inherits from
    :class:`~sionna.phy.utils.TransportBlock` and wraps
    :func:`~sionna.phy.nr.utils.calculate_tb_size`.

    :input modulation_order: [...], `torch.int32`.
        Modulation order, i.e., number of bits per symbol,
        associated with the input MCS index.

    :input target_rate: [...], `torch.float32`.
        Target coderate.

    :input num_coded_bits: [...], `torch.int32`.
        Total number of coded bits across all codewords.

    :output cb_size: [...], `torch.int32`.
        Code block (CB) size after 3GPP segmentation, including CRC bits that
        are part of the FEC code-block payload. This is not the transport-block
        information-bit count; see :meth:`transport_block_size`.

    :output num_cb: [...], `torch.int32`.
        Number of code blocks that the transport block is segmented into.

    .. rubric:: Examples

    .. code-block:: python

        from sionna.phy.nr.utils import TransportBlockNR
        import torch

        tb = TransportBlockNR()
        # Scalar input
        cb_size, num_cb = tb(modulation_order=4, target_coderate=0.5,
                             num_coded_bits=4800)
        print(f"CB size: {cb_size}, Num CBs: {num_cb}")

        # Tensor input
        mod_orders = torch.tensor([4, 6, 4])
        rates = torch.tensor([0.5, 0.5, 0.75])
        coded_bits = torch.tensor([4800, 7200, 3600])
        cb_sizes, num_cbs = tb(mod_orders, rates, coded_bits)
    """

    def transport_block_size(
        self,
        modulation_order: Union[int, torch.Tensor],
        target_coderate: Union[float, torch.Tensor],
        num_coded_bits: Union[int, torch.Tensor],
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return transport-block info bits, code-block size, and block count.

        ``cb_size`` follows 3GPP segmentation and includes CRC bits that are
        part of the FEC code-block payload. ``tb_size`` is the transport-block
        information-bit count used for system-level throughput accounting.
        """
        _ = kwargs
        tb_size, cb_size, num_cb, *_ = calculate_tb_size(
            modulation_order,
            target_coderate,
            num_coded_bits=num_coded_bits,
            tb_scaling=1.0,
            return_cw_length=False,
            verbose=False,
            device=self.device,
        )
        return tb_size, cb_size, num_cb

    def call(
        self,
        modulation_order: Union[int, torch.Tensor],
        target_coderate: Union[float, torch.Tensor],
        num_coded_bits: Union[int, torch.Tensor],
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute code block size and count."""
        _, cb_size, num_cb = self.transport_block_size(
            modulation_order, target_coderate, num_coded_bits, **kwargs
        )
        return cb_size, num_cb


class CodedAWGNChannelNR(SingleLinkChannel):
    r"""Simulates a 5G-NR compliant single-link coded AWGN channel.
    Inherits from :class:`~sionna.phy.utils.SingleLinkChannel`.

    :param num_bits_per_symbol: Number of bits per symbol, i.e., modulation
        order.
    :param num_info_bits: Number of information bits per code block.
    :param target_coderate: Target code rate, i.e., the target ratio between
        the information and the coded bits within a block.
    :param num_iter_decoder: Number of decoder iterations. See
        :class:`~sionna.phy.fec.ldpc.decoding.LDPC5GDecoder` for more
        details.
    :param cn_update_decoder: Check node update rule. See
        :class:`~sionna.phy.fec.ldpc.decoding.LDPC5GDecoder` for more
        details.
    :param precision: Precision for internal calculations.
        If `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation.
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.
    :param kwargs: Additional keyword arguments for
        :class:`~sionna.phy.fec.ldpc.decoding.LDPC5GDecoder`.

    :input batch_size: `int`.
        Size of the simulation batches.

    :input ebno_db: `float`.
        Eb/No value in dB.

    :output bits: [``batch_size``, ``num_info_bits``], `torch.int32`.
        Transmitted bits.

    :output bits_hat: [``batch_size``, ``num_info_bits``], `torch.int32`.
        Decoded bits.

    .. rubric:: Examples

    .. code-block:: python

        from sionna.phy.nr.utils import CodedAWGNChannelNR

        channel = CodedAWGNChannelNR(
            num_bits_per_symbol=4,
            num_info_bits=1024,
            target_coderate=0.5
        )
        bits, bits_hat = channel(batch_size=100, ebno_db=5.0)
    """

    def __init__(
        self,
        num_bits_per_symbol: Optional[int] = None,
        num_info_bits: Optional[int] = None,
        target_coderate: Optional[float] = None,
        num_iter_decoder: int = 20,
        cn_update_decoder: str = "boxplus-phi",
        precision=None,
        device=None,
        **kwargs,
    ):
        super().__init__(
            num_bits_per_symbol,
            num_info_bits,
            target_coderate,
            precision=precision,
            device=device,
        )
        self._num_iter_decoder = num_iter_decoder
        self._cn_update_decoder = cn_update_decoder
        self._kwargs = kwargs

    def call(
        self,
        batch_size: int,
        ebno_db: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simulate the 5G-NR coded AWGN channel."""
        # Import here to avoid circular imports
        from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder

        # Set the QAM constellation
        constellation = Constellation(
            "qam",
            self.num_bits_per_symbol,
            precision=self.precision,
            device=self.device,
        )

        # Set the Mapper/Demapper
        mapper = Mapper(
            constellation=constellation,
            precision=self.precision,
            device=self.device,
        )
        demapper = Demapper(
            "app",
            constellation=constellation,
            precision=self.precision,
            device=self.device,
        )

        binary_source = BinarySource(
            precision=self.precision,
            device=self.device,
        )
        awgn_channel = AWGN(
            precision=self.precision,
            device=self.device,
        )

        # 5G code block encoder
        encoder = LDPC5GEncoder(
            self.num_info_bits,
            int(self.num_coded_bits),
            num_bits_per_symbol=self.num_bits_per_symbol,
            precision=self.precision,
            device=self.device,
        )

        # 5G code block decoder
        decoder = LDPC5GDecoder(
            encoder,
            hard_out=True,
            num_iter=self._num_iter_decoder,
            cn_update=self._cn_update_decoder,
            precision=self.precision,
            device=self.device,
            **self._kwargs,
        )

        # Noise power
        no = ebnodb2no(
            ebno_db,
            num_bits_per_symbol=self.num_bits_per_symbol,
            coderate=self.target_coderate,
        )

        # Generate random information bits
        bits = binary_source([batch_size, self.num_info_bits])

        # Encode bits
        codewords = encoder(bits)

        # Map coded bits to complex symbols
        x = mapper(codewords)

        # Pass through an AWGN channel
        y = awgn_channel(x, no)

        # Compute log-likelihood ratio (LLR)
        llr = demapper(y, no)

        # Decode transmitted bits
        bits_hat = decoder(llr)

        return bits, bits_hat

