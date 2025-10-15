#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
""" Utility functions for sionna.phy.nr.utils tests"""

import numpy as np
import tensorflow as tf
from sionna.phy import config


def decode_mcs_index_numpy(mcs_index,
                           table_index=1,
                           channel_type="PUSCH",
                           transform_precoding=False,
                           pi2bpsk=False,
                           verbose=False):
    # pylint: disable=line-too-long
    r"""Numpy version of :func:`~sionna.phy.nr.utils.decode_mcs_index`.
    Selects modulation and coding scheme (MCS) as specified in TS 38.214 [3GPP38214]_.

    Implements MCS tables as defined in [3GPP38214]_ for PUSCH and PDSCH.

    Parameters
    ----------
    mcs_index : int| [0,...,28]
        MCS index (denoted as :math:`I_{MCS}` in [3GPP38214]_).

    table_index : int, 1 (default) | 2 | 3 | 4
        Indicates which MCS table from [3GPP38214]_ to use. Starts with index "1".

    channel_type : str, "PUSCH" (default) | "PDSCH"
        5G NR physical channel type. Valid choices are "PDSCH" and "PUSCH".

    transform_precoding : bool, False (default)
        If True, the MCS tables as described in Sec. 6.1.4.1
        in [3GPP38214]_ are applied. Only relevant for "PUSCH".

    pi2bpsk : bool, False (default)
        If True, the higher-layer parameter `tp-pi2BPSK` as
        described in Sec. 6.1.4.1 in [3GPP38214]_ is applied. Only relevant
        for "PUSCH".

    verbose : bool, False (default)
        If True, additional information will be printed.

    Returns
    -------
    (modulation_order, target_rate) :
            Tuple:

    modulation_order : int
        Modulation order, i.e., number of bits per symbol.

    target_rate : float
        Target coderate.
    """

    # check inputs
    assert isinstance(mcs_index, int), "mcs_index must be int."
    assert (mcs_index>=0), "mcs_index cannot be negative."
    assert isinstance(table_index, int), "table_index must be int."
    assert (table_index>0), "table_index starts with 1."
    assert isinstance(channel_type, str), "channel_type must be str."
    assert (channel_type in ("PDSCH", "PUSCH")), \
                        "channel_type must be either `PDSCH` or `PUSCH`."
    assert isinstance(transform_precoding, bool), \
                                    "transform_precoding must be bool."
    assert isinstance(pi2bpsk, bool), "pi2bpsk must be bool."
    assert isinstance(verbose, bool), "verbose must be bool."

    if verbose:
        print(f"Selected MCS index {mcs_index} for {channel_type} channel " \
              f"and Table index {table_index}.")

    # without pre-coding the Tables from 5.1.3.1 are used
    if channel_type=="PDSCH" or transform_precoding is False:

        if table_index==1: # Table 5.1.3.1-1 in 38.214
            if verbose:
                print("Applying Table 5.1.3.1-1 from TS 38.214.")

            assert mcs_index<29, "mcs_index not supported."
            mod_orders = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 6,
                          6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
            target_rates = [120, 157, 193, 251, 308, 379, 449, 526, 602, 679,
                            340, 378, 434, 490, 553, 616, 658, 438, 466, 517,
                            567, 616, 666, 719, 772, 822, 873, 910, 948]

        elif table_index==2: # Table 5.1.3.1-2 in 38.214
            if verbose:
                print("Applying Table 5.1.3.1-2 from TS 38.214.")

            assert mcs_index<28, "mcs_index not supported."
            mod_orders = [2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6,
                          6, 6, 8, 8, 8, 8, 8, 8, 8, 8]
            target_rates = [120, 193, 308, 449, 602, 378, 434, 490, 553, 616,
                            658, 466, 517, 567, 616, 666, 719, 772, 822, 873,
                            682.5, 711, 754, 797, 841, 885, 916.5, 948]

        elif table_index==3: # Table 5.1.3.1-3 in 38.214
            if verbose:
                print("Applying Table 5.1.3.1-3 from TS 38.214.")

            assert mcs_index<29, "mcs_index not supported."
            mod_orders = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4,
                          4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6]
            target_rates = [30, 40, 50, 64, 78, 99, 120, 157, 193, 251, 308,
                            379, 449, 526, 602, 340, 378, 434, 490, 553, 616,
                            438, 466, 517, 567, 616, 666, 719, 772]

        elif table_index==4: # Table 5.1.3.1-4 in 38.214
            if verbose:
                print("Applying Table 5.1.3.1-4 from TS 38.214.")

            assert mcs_index<27, "mcs_index not supported."
            mod_orders = [2, 2, 2, 4, 4, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 8, 8, 8,
                          8, 8, 8, 8, 8, 10, 10, 10, 10]
            target_rates = [120, 193, 449, 378, 490, 616, 466, 517, 567, 616,
                            666, 719, 772, 822, 873, 682.5, 711, 754, 797, 841,
                            885, 916.5, 948, 805.5, 853, 900.5, 948]
        else:
            raise ValueError("Unsupported table_index.")

    elif channel_type=="PUSCH": # only if pre-coding is true

        if table_index==1: # Table 6.1.4.1-1 in 38.214
            if verbose:
                print("Applying Table 6.1.4.1-1 from TS 38.214.")

            assert mcs_index<28, "mcs_index not supported."
            # higher layer parameter as defined in 6.1.4.1
            if pi2bpsk:
                if verbose:
                    print("Assuming pi2BPSK modulation.")
                q=1
            else:
                q=2

            mod_orders = [q, q, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 6,
                          6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
            target_rates = [240/q, 314/q, 193, 251, 308, 379, 449, 526, 602,
                            679, 340, 378, 434, 490, 553, 616, 658, 466, 517,
                            567, 616, 666, 719, 772, 822, 873, 910, 948]

        elif table_index==2: # Table 6.1.4.1-2 in 38.214
            if verbose:
                print("Applying Table 6.1.4.1-2 from TS 38.214.")

            assert mcs_index<28, "mcs_index not supported."
            # higher layer parameter as defined in 6.1.4.1
            if pi2bpsk:
                if verbose:
                    print("Assuming pi2BPSK modulation.")
                q=1
            else:
                q=2
            mod_orders = [q, q, q, q, q, q, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4,
                          4, 4, 4, 4, 4, 4, 6, 6, 6, 6]
            target_rates = [60/q, 80/q, 100/q, 128/q, 156/q, 198/q, 120, 157,
                            193, 251, 308, 379, 449, 526, 602, 679, 378, 434,
                            490, 553, 616, 658, 699, 772, 567, 616, 666, 772]
        else:
            raise ValueError("Unsupported table_index.")
    else:
        raise ValueError("Unsupported channel_type.")

    mod_order = mod_orders[mcs_index]
    target_rate = target_rates[mcs_index] / 1024 # rate is given as r*1024

    if verbose:
        print("Modulation order: ", mod_order)
        print("Target code rate: ", target_rate)

    return mod_order, target_rate


def calculate_tb_size_numpy(modulation_order,
                            target_coderate,
                            target_tb_size=None,
                            num_coded_bits=None,
                            num_prbs=None,
                            num_ofdm_symbols=None,
                            num_dmrs_per_prb=None,
                            num_layers=1,
                            num_ov=0,
                            tb_scaling=1.,
                            verbose=True):
    # pylint: disable=line-too-long
    r""" Numpy version of :func:`~sionna.phy.nr.utils.calculate_tb_size`.
    Calculates transport block (TB) size for given system parameters.

    This function follows the basic procedure as defined in TS 38.214 Sec.
    5.1.3.2 and Sec. 6.1.4.2 [3GPP38214]_.

    Parameters
    ----------
    modulation_order : int
        Modulation order, i.e., number of bits per QAM symbol.

    target_coderate : float
        Target coderate.

    target_tb_size: None (default) | int
        Target transport block size, i.e., how many information bits can be
        encoded into a slot for the given slot configuration. If provided,
        ``num_prbs``, ``num_ofdm_symbols`` and ``num_dmrs_per_prb`` will be
        ignored.

    num_coded_bits: None (default) | int
        How many coded bits can be fit into a given slot. If provided,
        ``num_prbs``, ``num_ofdm_symbols`` and ``num_dmrs_per_prb`` will be
        ignored.

    num_prbs : None (default) | int
        Total number of allocated PRBs per OFDM symbol where 1 PRB equals 12
        subcarriers.

    num_ofdm_symbols : None (default) | int
        Number of OFDM symbols allocated for transmission. Cannot be larger
        than 14.

    num_dmrs_per_prb : None (default) | int
        Number of DMRS (i.e., pilot) symbols per PRB that are NOT used for data
        transmission. Sum over all ``num_ofdm_symbols`` OFDM symbols.

    num_layers: int, 1 (default)
        Number of MIMO layers.

    num_ov : int, 0 (default)
        Number of unused resource elements due to additional
        overhead as specified by higher layer.

    tb_scaling: float, 0.25 | 0.5 | 1 (default)
        TB scaling factor for PDSCH as defined in TS 38.214 Tab. 5.1.3.2-2.
        Valid choices are 0.25, 0.5 and 1.0.

    verbose : bool, False (default)
        If True, additional information will be printed.

    Returns
    -------

    tb_size : int
        Transport block size, i.e., how many information bits can be encoded
        into a slot for the given slot configuration.

    cb_size : int
        Code block (CB) size. Determines the number of
        information bits (including TB/CB CRC parity bits) per codeword.

    num_cbs : int
        Number of code blocks. Determines into how many CBs the TB is segmented.

    tb_crc_length : int
        Length of the TB CRC.

    cb_crc_length : int
        Length of each CB CRC.

    cw_lengths : list of ints
        Each list element defines the codeword length of each of the ``num_cbs``
        codewords after LDPC encoding and rate-matching. The total number of
        coded bits is :math:`\sum` ``cw_lengths``.

    Note
    ----
    Due to rounding, ``cw_lengths`` (=length of each codeword after encoding),
    can be slightly different within a transport block. Thus,
    ``cw_lengths`` is given as a list of ints where each list elements denotes
    the number of codeword bits of the corresponding codeword after
    rate-matching.
    """

    # supports two modi:
    # a) target_tb_size and num_coded_bits given
    # b) available res in slot given

    # mode a)
    if target_tb_size is not None:

        if num_coded_bits is None:
            raise ValueError("num_coded_bits cannot be None if " \
                             "target_tb_size is provided.")
        assert num_coded_bits%1==0, "num_coded_bits must be int."
        num_coded_bits = int(num_coded_bits)

        assert num_coded_bits%num_layers==0, \
            "num_coded_bits must be a multiple of num_layers."

        assert num_coded_bits%modulation_order==0, \
            "num_coded_bits must be a multiple of modulation_order."

        assert target_tb_size%1==0, "target_tb_size must be int."
        n_info = int(target_tb_size)
        n_info_q = n_info # not quantized for user specified target_tb_size

        assert target_tb_size<num_coded_bits, \
            "Invalid transport block parameters. target_tb_size must be less " \
            "than the requested num_coded_bits excluding the overhead for the "\
            "TB CRC."

    else:
        if num_coded_bits is not None:
            print("num_coded_bits will be ignored if target_tb_size " \
                  "is None.")

        assert num_ofdm_symbols in range(1, 15),\
                "num_ofdm_symbols must be in the range from 1 to 14."
        assert num_prbs in range(1, 276),\
                "num_prbs must be in the range from 1 to 275."

        assert tb_scaling in (0.25, 0.5, 1.), \
                            "tb_scaling must be in (0.25,0.5,1.)."

        # compute n. of Resource Elements (REs), i.e., data symbols, per PRB
        n_re_per_prb = 12 * num_ofdm_symbols - num_dmrs_per_prb - num_ov
        # the max. number of REs per PRB is limited to 156 in 38.214
        n_re_per_prb = min(156, n_re_per_prb)

        # number of allocated REs
        n_re = n_re_per_prb * num_prbs

        # number of coded bits that fit into the given slot configuration
        num_coded_bits = int(tb_scaling * n_re * num_layers * modulation_order)

        # include tb_scaling as defined in Tab. 5.1.3.2-2 38.214
        n_info = target_coderate * num_coded_bits

    # apply quantization of info bit
    if n_info <= 3824:
        # step3 in 38.214 5.1.3.2
        n = max(3, np.floor(np.log2(n_info)) - 6)
        n_info_q = max(24, 2**n * np.floor(n_info/2**n))
    else:
        # step 4 in 38.212 5.3.1.2
        n = np.floor(np.log2(n_info-24)) - 5
        # "ties in the round function are broken towards next largest
        # integer"
        n_info_q = max(3840, 2**n * np.round((n_info-24)/2**n))

    if n_info_q <= 3824:
        c=1
        # go to step 3 in 38.214 5.1.3.2

        # already applied in the previous step
        # n = max(3, np.floor(np.log2(n_info)) - 6)
        # n_info_q = max(24, 2**n * np.floor(n_info/2**n))

        # explicit lengths given in Tab 5.1.3.2-1
        tab51321 = [24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128,
                    136, 144, 152, 160, 168, 176, 184, 192, 208, 224, 240, 256,
                    272, 288, 304, 320, 336, 352, 368, 384, 408, 432, 456, 480,
                    504, 528, 552, 576, 608, 640, 672, 704, 736, 768, 808, 848,
                    888, 928, 984, 1032, 1064, 1128, 1160, 1192, 1224, 1256,
                    1288, 1320, 1352, 1416, 1480, 1544, 1608, 1672, 1736, 1800,
                    1864, 1928, 2024, 2088, 2152, 2216, 2280, 2408, 2472, 2536,
                    2600, 2664, 2728, 2792, 2856, 2976, 3104, 3240, 3368, 3496,
                    3624, 3752, 3824]

        # find closest TBS that is not less n_info
        for tbs in tab51321:
            if tbs>=n_info_q:
                break
    else:
        # go to step 4 in 38.212 5.3.1.2

        # already applied in the previous step
        # n = np.floor(np.log2(n_info-24)) - 5
        # "ties in the round function are broken towards next largest integer"
        # n_info_q = max(3840, 2**n * np.round((n_info-24)/2**n))

        if target_coderate<=1/4:
            c = np.ceil((n_info_q + 24) / 3816)
            tbs = 8 * c * np.ceil((n_info_q + 24) / (8 * c)) - 24
        else:
            if n_info_q > 8424:
                c = np.ceil((n_info_q + 24) / 8424)
                tbs = 8 * c * np.ceil((n_info_q + 24) / (8*c)) - 24
            else:
                c = 1
                tbs = 8 * np.ceil((n_info_q + 24) / 8) - 24

    # TB CRC see 6.2.1 in 38.212
    if tbs>3824:
        tb_crc_length = 24
    else:
        tb_crc_length = 16

    # if tbs > max CB length, CRC-24 is added; see 5.2.2 in 38.212
    if c>1: # if multiple CBs exists, additional CRC is applied
        cb_crc_length = 24
    else:
        cb_crc_length = 0

    cb_size = (tbs + tb_crc_length)/c + cb_crc_length # bits per CW
    # internal sanity check
    assert (cb_size%1==0), "cb_size not an integer."

    # c is the number of code blocks
    num_cbs = int(c)
    cb_size = int(cb_size)
    tb_size = int(tbs)

    # cb_length as specified in 5.4.2.1 38.212
    # remark: the length can be different for multiple cws due to rounding
    # thus a list of lengths is generated
    cw_length = []

    for j in range(num_cbs):
        # first blocks are floored
        if j <= num_cbs \
              - np.mod(num_coded_bits/(num_layers*modulation_order),num_cbs)-1:
            l = num_layers * modulation_order \
              * np.floor(num_coded_bits / (num_layers*modulation_order*num_cbs))
            cw_length += [int(l)]
        else: # last blocks are ceiled
            l = num_layers * modulation_order \
              * np.ceil(num_coded_bits / (num_layers*modulation_order*num_cbs))
            cw_length += [int(l)]
    # sanity check that total length matches to total number of cws
    assert num_coded_bits==np.sum(cw_length), \
                        "Internal error: invalid codeword lengths."

    effective_rate = tb_size / num_coded_bits

    if verbose:
        print("Modulation order:", modulation_order)
        if target_coderate is not None:
            print(f"Target coderate: {target_coderate:.3f}")
        print(f"Effective coderate: {effective_rate:.3f}")
        print("Number of layers:", num_layers)
        print("------------------")
        print("Info bits per TB: ", tb_size)
        print("TB CRC length: ", tb_crc_length)
        print("Total number of coded TB bits:", num_coded_bits)
        print("------------------")
        print("Info bits per CB:", cb_size)
        print("Number of CBs:", num_cbs)
        print("CB CRC length: ", cb_crc_length)
        print("Output CB lengths:", cw_length)

    return tb_size, cb_size, num_cbs, tb_crc_length, cb_crc_length, cw_length

