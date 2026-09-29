#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Utility functions for sionna.phy.nr tests."""

import numpy as np


def decode_mcs_index_numpy(
    mcs_index: int,
    table_index: int = 1,
    channel_type: str = "PUSCH",
    transform_precoding: bool = False,
    pi2bpsk: bool = False,
    verbose: bool = False,
) -> tuple:
    r"""NumPy reference implementation for MCS index decoding.

    Implements MCS tables as defined in [3GPPTS38214]_ for PUSCH and PDSCH.

    :param mcs_index: MCS index [0, 28].
    :param table_index: MCS table index (1, 2, 3, or 4).
    :param channel_type: Channel type ("PUSCH" or "PDSCH").
    :param transform_precoding: Apply transform precoding tables.
    :param pi2bpsk: Apply pi/2-BPSK modulation.
    :param verbose: Print additional information.
    :return: Tuple of (modulation_order, target_rate).
    """
    if not isinstance(mcs_index, int) or mcs_index < 0:
        raise ValueError("mcs_index must be a non-negative int")
    if not isinstance(table_index, int) or table_index < 1:
        raise ValueError("table_index must be an int >= 1")
    if channel_type not in ("PDSCH", "PUSCH"):
        raise ValueError("channel_type must be 'PDSCH' or 'PUSCH'")

    if verbose:
        print(f"Selected MCS index {mcs_index} for {channel_type} channel "
              f"and Table index {table_index}.")

    if channel_type == "PDSCH" or not transform_precoding:
        if table_index == 1:
            if verbose:
                print("Applying Table 5.1.3.1-1 from TS 38.214.")
            if mcs_index >= 29:
                raise ValueError("mcs_index not supported")
            mod_orders = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 6,
                          6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
            target_rates = [120, 157, 193, 251, 308, 379, 449, 526, 602, 679,
                            340, 378, 434, 490, 553, 616, 658, 438, 466, 517,
                            567, 616, 666, 719, 772, 822, 873, 910, 948]

        elif table_index == 2:
            if verbose:
                print("Applying Table 5.1.3.1-2 from TS 38.214.")
            if mcs_index >= 28:
                raise ValueError("mcs_index not supported")
            mod_orders = [2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6,
                          6, 6, 8, 8, 8, 8, 8, 8, 8, 8]
            target_rates = [120, 193, 308, 449, 602, 378, 434, 490, 553, 616,
                            658, 466, 517, 567, 616, 666, 719, 772, 822, 873,
                            682.5, 711, 754, 797, 841, 885, 916.5, 948]

        elif table_index == 3:
            if verbose:
                print("Applying Table 5.1.3.1-3 from TS 38.214.")
            if mcs_index >= 29:
                raise ValueError("mcs_index not supported")
            mod_orders = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4,
                          4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6]
            target_rates = [30, 40, 50, 64, 78, 99, 120, 157, 193, 251, 308,
                            379, 449, 526, 602, 340, 378, 434, 490, 553, 616,
                            438, 466, 517, 567, 616, 666, 719, 772]

        elif table_index == 4:
            if verbose:
                print("Applying Table 5.1.3.1-4 from TS 38.214.")
            if mcs_index >= 27:
                raise ValueError("mcs_index not supported")
            mod_orders = [2, 2, 2, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 8, 8, 8,
                          8, 8, 8, 8, 8, 10, 10, 10, 10]
            target_rates = [120, 193, 449, 378, 490, 616, 466, 517, 567, 616,
                            666, 719, 772, 822, 873, 682.5, 711, 754, 797, 841,
                            885, 916.5, 948, 805.5, 853, 900.5, 948]
        else:
            raise ValueError("Unsupported table_index")

    elif channel_type == "PUSCH":
        if table_index == 1:
            if verbose:
                print("Applying Table 6.1.4.1-1 from TS 38.214.")
            if mcs_index >= 28:
                raise ValueError("mcs_index not supported")
            q = 1 if pi2bpsk else 2
            if verbose and pi2bpsk:
                print("Assuming pi2BPSK modulation.")

            mod_orders = [q, q, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 6,
                          6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
            target_rates = [240/q, 314/q, 193, 251, 308, 379, 449, 526, 602,
                            679, 340, 378, 434, 490, 553, 616, 658, 466, 517,
                            567, 616, 666, 719, 772, 822, 873, 910, 948]

        elif table_index == 2:
            if verbose:
                print("Applying Table 6.1.4.1-2 from TS 38.214.")
            if mcs_index >= 28:
                raise ValueError("mcs_index not supported")
            q = 1 if pi2bpsk else 2
            if verbose and pi2bpsk:
                print("Assuming pi2BPSK modulation.")

            mod_orders = [q, q, q, q, q, q, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4,
                          4, 4, 4, 4, 4, 4, 6, 6, 6, 6]
            target_rates = [60/q, 80/q, 100/q, 128/q, 156/q, 198/q, 120, 157,
                            193, 251, 308, 379, 449, 526, 602, 679, 378, 434,
                            490, 553, 616, 658, 699, 772, 567, 616, 666, 772]
        else:
            raise ValueError("Unsupported table_index")
    else:
        raise ValueError("Unsupported channel_type")

    mod_order = mod_orders[mcs_index]
    target_rate = target_rates[mcs_index] / 1024

    if verbose:
        print("Modulation order:", mod_order)
        print("Target code rate:", target_rate)

    return mod_order, target_rate


def calculate_tb_size_numpy(
    modulation_order: int,
    target_coderate: float,
    target_tb_size: int = None,
    num_coded_bits: int = None,
    num_prbs: int = None,
    num_ofdm_symbols: int = None,
    num_dmrs_per_prb: int = None,
    num_layers: int = 1,
    num_ov: int = 0,
    tb_scaling: float = 1.0,
    verbose: bool = True,
) -> tuple:
    r"""NumPy reference implementation for TB size calculation.

    Follows the procedure in TS 38.214 Sec. 5.1.3.2 and 6.1.4.2 [3GPPTS38214]_.

    :param modulation_order: Modulation order (bits per symbol).
    :param target_coderate: Target coderate.
    :param target_tb_size: Target TB size. If provided, resource parameters ignored.
    :param num_coded_bits: Number of coded bits.
    :param num_prbs: Number of PRBs per OFDM symbol.
    :param num_ofdm_symbols: Number of OFDM symbols (1-14).
    :param num_dmrs_per_prb: Number of DMRS symbols per PRB.
    :param num_layers: Number of MIMO layers.
    :param num_ov: Overhead resource elements.
    :param tb_scaling: TB scaling factor (0.25, 0.5, or 1.0).
    :param verbose: Print additional information.
    :return: Tuple of (tb_size, cb_size, num_cbs, tb_crc_length, cb_crc_length, cw_lengths).
    """
    if target_tb_size is not None:
        if num_coded_bits is None:
            raise ValueError("num_coded_bits required when target_tb_size provided")
        num_coded_bits = int(num_coded_bits)
        if num_coded_bits % num_layers != 0:
            raise ValueError("num_coded_bits must be multiple of num_layers")
        if num_coded_bits % modulation_order != 0:
            raise ValueError("num_coded_bits must be multiple of modulation_order")
        n_info = int(target_tb_size)
        if target_tb_size >= num_coded_bits:
            raise ValueError("target_tb_size must be less than num_coded_bits")
    else:
        if num_ofdm_symbols not in range(1, 15):
            raise ValueError("num_ofdm_symbols must be in [1, 14]")
        if num_prbs not in range(1, 276):
            raise ValueError("num_prbs must be in [1, 275]")
        if tb_scaling not in (0.25, 0.5, 1.0):
            raise ValueError("tb_scaling must be 0.25, 0.5, or 1.0")

        n_re_per_prb = 12 * num_ofdm_symbols - num_dmrs_per_prb - num_ov
        n_re_per_prb = min(156, n_re_per_prb)
        n_re = n_re_per_prb * num_prbs
        num_coded_bits = int(tb_scaling * n_re * num_layers * modulation_order)
        n_info = target_coderate * num_coded_bits

    # Quantize info bits
    if n_info <= 3824:
        n = max(3, np.floor(np.log2(n_info)) - 6)
        n_info_q = max(24, 2**n * np.floor(n_info / 2**n))
    else:
        n = np.floor(np.log2(n_info - 24)) - 5
        # Ties are rounded up, whereas np.round rounds half to even
        n_info_q = max(3840, 2**n * np.floor((n_info - 24) / 2**n + 0.5))

    if n_info_q <= 3824:
        c = 1
        tab51321 = [24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128,
                    136, 144, 152, 160, 168, 176, 184, 192, 208, 224, 240, 256,
                    272, 288, 304, 320, 336, 352, 368, 384, 408, 432, 456, 480,
                    504, 528, 552, 576, 608, 640, 672, 704, 736, 768, 808, 848,
                    888, 928, 984, 1032, 1064, 1128, 1160, 1192, 1224, 1256,
                    1288, 1320, 1352, 1416, 1480, 1544, 1608, 1672, 1736, 1800,
                    1864, 1928, 2024, 2088, 2152, 2216, 2280, 2408, 2472, 2536,
                    2600, 2664, 2728, 2792, 2856, 2976, 3104, 3240, 3368, 3496,
                    3624, 3752, 3824]

        for tbs in tab51321:
            if tbs >= n_info_q:
                break
    else:
        if target_coderate <= 0.25:
            c = np.ceil((n_info_q + 24) / 3816)
            tbs = 8 * c * np.ceil((n_info_q + 24) / (8 * c)) - 24
        elif n_info_q > 8424:
            c = np.ceil((n_info_q + 24) / 8424)
            tbs = 8 * c * np.ceil((n_info_q + 24) / (8 * c)) - 24
        else:
            c = 1
            tbs = 8 * np.ceil((n_info_q + 24) / 8) - 24

    # CRC lengths
    tb_crc_length = 24 if tbs > 3824 else 16
    cb_crc_length = 24 if c > 1 else 0

    cb_size = int((tbs + tb_crc_length) / c + cb_crc_length)
    num_cbs = int(c)
    tb_size = int(tbs)

    # Codeword lengths
    cw_length = []
    for j in range(num_cbs):
        threshold = num_cbs - np.mod(
            num_coded_bits / (num_layers * modulation_order), num_cbs) - 1
        if j <= threshold:
            length = num_layers * modulation_order * np.floor(
                num_coded_bits / (num_layers * modulation_order * num_cbs))
        else:
            length = num_layers * modulation_order * np.ceil(
                num_coded_bits / (num_layers * modulation_order * num_cbs))
        cw_length.append(int(length))

    if num_coded_bits != np.sum(cw_length):
        raise ValueError("Internal error: invalid codeword lengths")

    effective_rate = tb_size / num_coded_bits

    if verbose:
        print(f"Modulation order: {modulation_order}")
        if target_coderate is not None:
            print(f"Target coderate: {target_coderate:.3f}")
        print(f"Effective coderate: {effective_rate:.3f}")
        print(f"Number of layers: {num_layers}")
        print("------------------")
        print(f"Info bits per TB: {tb_size}")
        print(f"TB CRC length: {tb_crc_length}")
        print(f"Total number of coded TB bits: {num_coded_bits}")
        print("------------------")
        print(f"Info bits per CB: {cb_size}")
        print(f"Number of CBs: {num_cbs}")
        print(f"CB CRC length: {cb_crc_length}")
        print(f"Output CB lengths: {cw_length}")

    return tb_size, cb_size, num_cbs, tb_crc_length, cb_crc_length, cw_length

