#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""Utility functions for the NR (5G) module of Sionna PHY"""

import numpy as np
import tensorflow as tf
from sionna.phy.utils import tensor_values_are_in_set, insert_dims
from sionna.phy import dtypes, config
from sionna.phy.utils import TransportBlock, SingleLinkChannel, \
    ebnodb2no, MCSDecoder, scalar_to_shaped_tensor
from sionna.phy.channel import AWGN
from sionna.phy.mapping import Mapper, Demapper, Constellation, BinarySource


def generate_prng_seq(length, c_init):
    r"""Implements pseudo-random sequence generator as defined in Sec. 5.2.1
    in [3GPP38211]_ based on a length-31 Gold sequence.

    Parameters
    ----------
    length: `int`
        Desired output sequence length

    c_init: `int`
        Initialization sequence of the PRNG. Must be in the range of 0 to
        :math:`2^{32}-1`.

    Output
    ------
    :[``length``], `ndarray` of 0s and 1s
        Containing the scrambling sequence

    Note
    ----
    The initialization sequence ``c_init`` is application specific and is
    usually provided be higher layer protocols.
    """

    # check inputs for consistency
    assert(length%1==0), "length must be a positive integer."
    length = int(length)
    assert(length>0), "length must be a positive integer."

    assert(c_init%1==0), "c_init must be integer."
    c_init = int(c_init)
    assert(c_init<2**32), "c_init must be in [0, 2^32-1]."
    assert(c_init>=0), "c_init must be in [0, 2^32-1]."

    # internal parameters
    n_seq = 31 # length of gold sequence
    n_c = 1600 # defined in 5.2.1 in 38.211

    # init sequences
    c = np.zeros(length)
    x1 = np.zeros(length + n_c + n_seq)
    x2 = np.zeros(length + n_c + n_seq)

    #int2bin
    bin_ = format(c_init, f'0{n_seq}b')
    c_init = [int(x) for x in bin_[-n_seq:]] if n_seq else []
    c_init = np.flip(c_init) # reverse order

    # init x1 and x2
    x1[0] = 1
    x2[0:n_seq] = c_init

    # and run the generator
    for idx in range(length + n_c):
        x1[idx+31] = np.mod(x1[idx+3] + x1[idx], 2)
        x2[idx+31] = np.mod(x2[idx+3] + x2[idx+2] + x2[idx+1] + x2[idx], 2)

    # update output sequence
    for idx in range(length):
        c[idx] = np.mod(x1[idx+n_c] + x2[idx+n_c], 2)

    return c


def decode_mcs_index(mcs_index,
                     table_index=1,
                     is_pusch=True,
                     transform_precoding=False,
                     pi2bpsk=False,
                     check_index_validity=True,
                     verbose=False):
    # pylint: disable=line-too-long
    r"""Returns the modulation order and target coderate for a given MCS index

    Implements MCS tables as defined in [3GPP38214]_ for PUSCH and PDSCH.

    Input
    -----

    mcs_index : [...], `tf.int32` | `int`
        MCS indices (denoted as :math:`I_{MCS}` in
        [3GPP38214]_). Accepted values are `{0,1,...28}`

    table_index : [...], `tf.int32` | `int` (default: 1)
        MCS table indices from [3GPP38214]_ to be used.
        Accepted values are `{1,2,3,4}`.

    is_pusch : [...], `tf.bool` | `bool` (default: `True`)
        Specifies whether the 5G NR physical channel is of 'PUSCH' type. If
        `False`, then the 'PDSCH' channel is considered.

    transform_precoding : [...], `tf.bool` | `bool` (default: `False`)
        Specifies whether the MCS tables described in
        Sec. 6.1.4.1 of [3GPP38214]_ are applied.
        Only relevant for "PUSCH".

    pi2bpsk : [...], `tf.bool` | `bool` | `None` (default)
        Specifies whether the higher-layer parameter `tp-pi2BPSK`
        described in Sec. 6.1.4.1 of [3GPP38214]_ is applied. Only relevant for
        "PUSCH".

    check_index_validity : `bool` (default: `True`)
        If `True`, an ValueError is thrown is the input MCS indices are not
        valid for the given configuration.

    verbose : `bool` (default: `False`)
        If `True`, then additional information is printed.

    Output
    ------

    modulation_order : [...], `tf.int32`
        Modulation order, i.e., number of bits per symbol,
        associated with the input MCS index

    target_rate : [...], `tf.float32`
        Target coderate associated with the input MCS index
    """
    mcs_index = tf.cast(mcs_index, tf.int32)
    shape = mcs_index.shape

    # Cast and reshape inputs
    table_index = scalar_to_shaped_tensor(table_index, tf.int32, shape)
    is_pusch = scalar_to_shaped_tensor(is_pusch, tf.bool, shape)
    transform_precoding = scalar_to_shaped_tensor(transform_precoding,
                                                  tf.bool,
                                                  shape)
    pi2bpsk = scalar_to_shaped_tensor(pi2bpsk, tf.bool, shape)

    # Check input consistency
    tf.debugging.assert_shapes([(table_index, shape),
                                (is_pusch, shape),
                                (transform_precoding, shape),
                                (pi2bpsk, shape)],
                               message="inconsistent input shapes")
    tf.debugging.assert_greater_equal(mcs_index, 0,
                                      message='MCS index cannot be negative')
    tf.debugging.assert_less_equal(
        mcs_index, 28,
        message='MCS index cannot be higher than 28')
    tf.debugging.assert_equal(
        tensor_values_are_in_set(table_index, [1, 2, 3, 4]),
        True,
        message="table_index must contain " +
        "values in [1,2,3,4]")
    tf.debugging.assert_type(is_pusch, tf.bool,
                             message="is_pusch.dtype must be bool")
    tf.debugging.assert_type(transform_precoding, tf.bool,
                             message="transform_precoding.dtype must be bool")
    tf.debugging.assert_type(pi2bpsk, tf.bool,
                             message="pi2bpsk.dtype must be bool")

    if verbose:
        print(f"Selected MCS index {mcs_index.numpy()} for "
              f"{tf.where(is_pusch, 'PUSCH', 'PDSCH')} channel "
              f"and Table index {table_index.numpy()}.")

    # modulation orders
    # [2, 4, 29]: [is_pusch, table_index, mcs_index]
    mod_orders = tf.convert_to_tensor([
        [  # PUSCH
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
    ])

    # [2, 4, 29]: [is_pusch, table_index, mcs_index]
    target_rates = tf.convert_to_tensor([
        [  # PUSCH
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
    ])

    channel_type_idx = tf.cast(tf.logical_not(is_pusch) |
                               tf.logical_not(transform_precoding), tf.int32)

    # [shape, 3]
    idx = tf.stack([channel_type_idx,
                    table_index-1,
                    mcs_index],
                   axis=-1)

    # Select modulation orders and target rates from input
    # [shape]
    mod_orders_sel = tf.gather_nd(mod_orders, idx)
    target_rates_sel = tf.gather_nd(target_rates, idx)

    # Check that the selected indices are valid
    if check_index_validity:
        tf.debugging.assert_non_negative(mod_orders_sel,
                                         message='Invalid MCS index')

    #######################
    # Account for pi2BPSK #
    #######################
    # [shape]
    q = tf.where(pi2bpsk, 1, 2)

    # input indices at which modulation and target rates depend on q
    if len(shape)==0:
        if (channel_type_idx == 0) & (
            ((table_index == 1) & (mcs_index < 2)) |
            ((table_index == 2) & (mcs_index < 6))):

            mod_orders_sel = mod_orders_sel * tf.cast(q, mod_orders_sel.dtype)
            target_rates_sel = target_rates_sel / tf.cast(q, target_rates.dtype)
    else:
        # [_, len(shape)]
        idx_q = tf.where(
            (channel_type_idx == 0) &
            (
                ((table_index == 1) & (mcs_index < 2)) |
                ((table_index == 2) & (mcs_index < 6))
            ))

        # correct mod_orders via q
        mod_orders_sel_q = tf.gather_nd(
            mod_orders_sel, idx_q) * tf.gather_nd(q, idx_q)
        mod_orders_sel = tf.tensor_scatter_nd_update(mod_orders_sel,
                                                     idx_q,
                                                     mod_orders_sel_q)

        # correct target_rates via q
        target_rates_sel_q = tf.gather_nd(target_rates_sel, idx_q) / \
            tf.cast(tf.gather_nd(q, idx_q), target_rates.dtype)
        target_rates_sel = tf.tensor_scatter_nd_update(
            target_rates_sel,
            idx_q,
            target_rates_sel_q)

    target_rates_sel = target_rates_sel / 1024
    if verbose:
        print(f"Modulation order: {mod_orders_sel.numpy()}")
        print(f"Target code rate: {target_rates_sel.numpy()}")

    return mod_orders_sel, target_rates_sel


class MCSDecoderNR(MCSDecoder):
    r"""
    Maps a Modulation and Coding Scheme (MCS) index to the
    corresponding modulation order, i.e., number of bits per symbol, and
    coderate for 5G-NR networks. Wraps
    :func:`~sionna.phy.nr.utils.decode_mcs_index` and inherits
    from :class:`~sionna.phy.utils.MCSDecoder`.

    Input
    -----

    mcs_index : [...], `tf.int32`
        MCS index

    mcs_table_index : [...], `tf.int32`
        MCS table index. Different tables contain different mappings.

    mcs_category : [...], `tf.int32`
        `0` for PUSCH, `1` for PDSCH channel

    check_index_validity : `bool` (default: `True`)
        If `True`, an ValueError is thrown is the input mcs indices are not
        valid for the given configuration.

    transform_precoding : [...], `tf.bool` | `bool` (default: `False`)
        Specifies whether the MCS tables described in
        Sec. 6.1.4.1 of [3GPP38214]_ are applied.
        Only relevant for "PUSCH".

    pi2bpsk : [...], `tf.bool` | `bool` | `None` (default)
        Specifies whether the higher-layer parameter `tp-pi2BPSK`
        described in Sec. 6.1.4.1 of [3GPP38214]_ is applied.
        Only relevant for "PUSCH".

    verbose : `bool` (default: `False`)
        If `True`, then additional information is printed.

    Output
    ------

    modulation_order : [...], `tf.int32`
        Modulation order corresponding to the input MCS index

    target_coderate : [...], `tf.float`
        Target coderate corresponding to the input MCS index
    """

    def call(self,
             mcs_index,
             mcs_table_index,
             mcs_category,*,
             check_index_validity=True,
             transform_precoding=True,
             pi2bpsk=False,
             verbose=False,
             **kwargs):
        modulation_order, target_coderate = \
            decode_mcs_index(mcs_index,
                             table_index=mcs_table_index,
                             is_pusch=tf.cast(1 - mcs_category, tf.bool),
                             transform_precoding=transform_precoding,
                             pi2bpsk=pi2bpsk,
                             check_index_validity=check_index_validity,
                             verbose=verbose)
        return modulation_order, target_coderate


def calculate_num_coded_bits(modulation_order,
                             num_prbs,
                             num_ofdm_symbols,
                             num_dmrs_per_prb,
                             num_layers,
                             num_ov,
                             tb_scaling,
                             precision=None):
    r"""
    Computes the number of coded bits that fit in a slot for the given resource
    grid structure

    Input
    -----

    modulation_order : [...], `tf.int32`
        Modulation order, i.e., number of bits per QAM symbol

    num_prbs : [...], `tf.int32` | `int`
        Total number of allocated PRBs per OFDM symbol, where 1 PRB equals 12
        subcarriers. Must not exceed 275.

    num_ofdm_symbols : [...], `tf.int32`
        Number of OFDM symbols allocated for transmission. Cannot be larger
        than 14.

    num_dmrs_per_prb : [...], `tf.int32`
        Number of DMRS (i.e., pilot) symbols per PRB that are `not` used for
        data transmission, across all ``num_ofdm_symbols`` OFDM symbols.

    num_layers: [...], `tf.int32`
        Number of MIMO layers.

    num_ov : [...], `tf.int32`
        Number of unused resource elements due to additional
        overhead as specified by higher layer.

    tb_scaling: [...], `tf.float`
        TB scaling factor for PDSCH as defined in TS 38.214 Tab. 5.1.3.2-2. Must
        contain values in {0.25, 0.5, 1.0}.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`, then :attr:`~sionna.phy.config.Config.precision`
        is used.

    Output
    ------

    num_coded_bits: [...], `tf.int` | `int` | `None` (default)
        Number of coded bits can be fit into a given slot for the fiven
        configuration.

    """
    # Cast inputs
    if precision is None:
        rdtype = config.tf_rdtype
    else:
        rdtype = dtypes[precision]["tf"]["rdtype"]
    num_prbs = tf.cast(num_prbs, tf.int32)
    modulation_order = tf.cast(modulation_order, tf.int32)
    num_layers = tf.cast(num_layers, tf.int32)
    num_ofdm_symbols = tf.cast(num_ofdm_symbols, tf.int32)
    num_dmrs_per_prb = tf.cast(num_dmrs_per_prb, tf.int32)
    num_ov = tf.cast(num_ov, tf.int32)
    tb_scaling = tf.cast(tb_scaling, rdtype)

    # Validate inputs
    tf.debugging.assert_greater_equal(
        num_ofdm_symbols, 1,
        message="num_ofdm_symbols must be at least 1.")
    tf.debugging.assert_less_equal(
        num_ofdm_symbols, 14,
        message="num_ofdm_symbols must be at most 14.")
    tf.debugging.assert_greater_equal(
        num_prbs, 1,
        message="num_prbs must be at least 1.")
    tf.debugging.assert_less_equal(
        num_prbs, 275,
        message="num_prbs must be at most 275.")
    tf.debugging.assert_equal(
        tensor_values_are_in_set(tb_scaling, [0.25, 0.5, 1.0]),
        True,
        message="tb_scaling must be 0.25, 0.5, or 1.0.")

    # Compute n. Resource Elements (RE) per PRB
    n_re_per_prb = 12 * num_ofdm_symbols - num_dmrs_per_prb - num_ov
    # The max. number of REs per PRB is limited to 156 in 38.214
    n_re_per_prb = tf.minimum(156, n_re_per_prb)

    # Compute n. coded bits
    num_coded_bits = tb_scaling * tf.cast(
        n_re_per_prb * num_prbs * modulation_order * num_layers,
        rdtype)
    num_coded_bits = tf.cast(num_coded_bits, tf.int32)

    return num_coded_bits


def calculate_tb_size(modulation_order,
                      target_coderate,
                      target_tb_size=None,
                      num_coded_bits=None,
                      num_prbs=None,
                      num_ofdm_symbols=None,
                      num_dmrs_per_prb=None,
                      num_layers=1,
                      num_ov=0,
                      tb_scaling=1.0,
                      return_cw_length=True,
                      verbose=False,
                      precision=None):
    # pylint: disable=line-too-long
    r"""Calculates the transport block (TB) size for given system parameters

    This function follows the procedure defined in TS 38.214 Sec.
    5.1.3.2 and Sec. 6.1.4.2 [3GPP38214]_

    Input
    -----

    modulation_order : [...], `tf.int` | `int`
        Modulation order, i.e., number of bits per QAM symbol.

    target_coderate : [...], `tf.float` | `float`
        Target coderate.

    target_tb_size: [...], `tf.float` | `float` | `None` (default)
        Target transport block size, i.e., number of information bits that can
        be encoded into a slot for the given slot configuration.

    num_coded_bits: [...], `tf.int` | `int` | `None` (default)
        Number of coded bits can be fit into a given slot. If provided,
        ``num_prbs``, ``num_ofdm_symbols`` and ``num_dmrs_per_prb`` are
        ignored.

    num_prbs : [...], `tf.int` | `int` | `None` (default)
        Total number of allocated PRBs per OFDM symbol, where 1 PRB equals 12
        subcarriers. Must not exceed 275.

    num_ofdm_symbols : [...], `tf.int` | `int` | `None` (default)
        Number of OFDM symbols allocated for transmission. Cannot be larger
        than 14.

    num_dmrs_per_prb : [...], `tf.int` | `int` | `None` (default)
        Number of DMRS (i.e., pilot) symbols per PRB that are `not` used for data
        transmission, across all ``num_ofdm_symbols`` OFDM symbols.

    num_layers: [...], `tf.int` | `int` (default: 1)
        Number of MIMO layers.

    num_ov : [...], `tf.int` | `int` | `None` (default)
        Number of unused resource elements due to additional
        overhead as specified by higher layer.

    tb_scaling: [...], `tf.float` | {0.25, 0.5, 1.0} (default: 1.0)
        TB scaling factor for PDSCH as defined in TS 38.214 Tab. 5.1.3.2-2.

    return_cw_length : `bool` (default: `True`)
        If `True`, then the function returns ``tb_size``, ``cb_size``,
        ``num_cb``, ``tb_crc_length``, ``cb_crc_length``, ``cw_length``.
        Else, it does not return ``cw_length`` to reduce computation time.

    verbose : `bool`, (default: `False`)
        If `True`, then additional information is printed.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`, then :attr:`~sionna.phy.config.Config.precision` is used

    Output
    ------

    tb_size : [...], `tf.int32`
        Transport block (TB) size, i.e., how many information bits can be encoded
        into a slot for the given slot configuration

    cb_size : [...], `tf.int32`
        Code block (CB) size, i.e., the number of information bits per codeword,
        including the TB/CB CRC parity bits

    num_cb : [...], `tf.int32`
        Number of CBs that the TB is segmented into

    tb_crc_length : [...], `tf.int32`
        Length of the TB CRC

    cb_crc_length : [...], `tf.int32`
        Length of each CB CRC

    cw_length : [..., N], `tf.int32`
        Codeword length of each of the ``num_cbs`` codewords after LDPC encoding
        and rate-matching.
        Note that zeros are appended along the last axis to obtain a dense tensor.
        The total number of coded bits, ``num_coded_bits``, is the sum of
        ``cw_length`` across its last axis.
        Only returned if ``return_cw_length`` is `True`
    """
    if precision is None:
        rdtype = config.tf_rdtype
    else:
        rdtype = dtypes[precision]["tf"]["rdtype"]

    # Cast inputs and assign default values
    modulation_order = tf.cast(modulation_order, tf.int32)
    target_coderate = tf.cast(target_coderate, rdtype)
    shape = modulation_order.shape
    num_layers = scalar_to_shaped_tensor(num_layers, tf.int32, shape)
    tb_scaling = scalar_to_shaped_tensor(tb_scaling, rdtype, shape)

    #---------------#
    # N. coded bits #
    #---------------#
    if num_coded_bits is not None:
        num_coded_bits = tf.cast(num_coded_bits, dtype=tf.int32)

        tf.debugging.assert_equal(
            num_coded_bits % modulation_order, 0,
            message="num_coded_bits must be a multiple of modulation_order.")
    else:
        tf.debugging.assert_equal(
            (num_prbs is not None) and
            (num_ofdm_symbols is not None) and
            (num_dmrs_per_prb is not None),
            True,
            message="If num_coded_bits is None then "
            "num_prbs, num_ofdm_symbols, num_dmrs_per_prb must be specified.")

        # Compute num_coded_bits from n. allocated resources
        num_prbs = tf.cast(num_prbs, tf.int32)
        num_ofdm_symbols = tf.cast(num_ofdm_symbols, tf.int32)
        num_dmrs_per_prb = tf.cast(num_dmrs_per_prb, tf.int32)
        num_ov = scalar_to_shaped_tensor(num_ov, tf.int32, shape)

        num_coded_bits = calculate_num_coded_bits(modulation_order,
                                                  num_prbs,
                                                  num_ofdm_symbols,
                                                  num_dmrs_per_prb,
                                                  num_layers,
                                                  num_ov,
                                                  tb_scaling,
                                                  precision=precision)

    tf.debugging.assert_equal(
            num_coded_bits % num_layers, 0,
            message="num_coded_bits must be a multiple of num_layers.")

    # -------------- #
    # Target TB size #
    # -------------- #
    if target_tb_size is not None:
        # input num_prbs, num_ofdm_symbols and num_dmrs_per_prb are ignored

        target_tb_size = tf.cast(target_tb_size, dtype=rdtype)

        # Validate inputs
        tf.debugging.assert_less(
            target_tb_size, tf.cast(num_coded_bits, target_tb_size.dtype),
            message="target_tb_size must be less than num_coded_bits.")

    else:
        # Compute n info bits (Target TB size)
        target_tb_size = target_coderate * tf.cast(num_coded_bits, rdtype)

    # quantize target_tb_size
    def n_info_q_if_target_tbs_greater_3824():
        # Compute quantized n. info bits if target TB size > 3824
        # Step 4 of 38.214 5.3.1.2
        log2_n_info_minus_24 = tf.math.log(
            target_tb_size - tf.cast(24, target_tb_size.dtype)) \
        / tf.math.log(tf.cast(2.0, target_tb_size.dtype))
        n = tf.math.floor(log2_n_info_minus_24) - 5.
        n_info_q = tf.math.maximum(
            tf.cast(3840.0, rdtype),
            tf.cast(2**n * tf.math.round((target_tb_size - 24) / 2**n), rdtype))
        return n_info_q

    def n_info_q_if_target_tbs_smaller_3824():
        # Compute quantized n. info bits if target TB size <= 3824
        log2_n_info = tf.math.log(target_tb_size) \
            / tf.cast(tf.math.log(2.0), target_tb_size.dtype)
        n = tf.math.maximum(tf.cast(3.0, rdtype),
                            tf.cast(tf.math.floor(log2_n_info) - 6, rdtype))
        n_info_q = tf.math.maximum(
            tf.cast(24.0, rdtype),
            tf.cast(2**n * tf.math.floor(target_tb_size / 2**n), rdtype))
        return n_info_q

    # ----------------------------- #
    # Quantized n. information bits #
    # ----------------------------- #
    n_info_q = tf.where(target_tb_size <= 3824,
                        n_info_q_if_target_tbs_smaller_3824(),
                        n_info_q_if_target_tbs_greater_3824())
    # ------------------- #
    # Auxiliary functions #
    # ------------------- #

    def tbs_if_target_tbs_higher_3824():
        # Compute TB size if target_tb_size>3824
        tbs = 8 * num_cb * tf.math.ceil((n_info_q + 24) / (8 * num_cb)) - 24
        return tf.cast(tbs, tf.int32)

    def tbs_if_target_tbs_smaller_3824():
        # Compute TB size if target TB size <= 3824
        # Step 3 of 38.214 5.1.3.2
        tab51321 = tf.constant(
            [-1, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128,
                136, 144, 152, 160, 168, 176, 184, 192, 208, 224, 240, 256,
                272, 288, 304, 320, 336, 352, 368, 384, 408, 432, 456, 480,
                504, 528, 552, 576, 608, 640, 672, 704, 736, 768, 808, 848,
                888, 928, 984, 1032, 1064, 1128, 1160, 1192, 1224, 1256,
                1288, 1320, 1352, 1416, 1480, 1544, 1608, 1672, 1736, 1800,
                1864, 1928, 2024, 2088, 2152, 2216, 2280, 2408, 2472, 2536,
                2600, 2664, 2728, 2792, 2856, 2976, 3104, 3240, 3368, 3496,
                3624, 3752, 3824], dtype=rdtype)

        # Find the smallest TB size >= n_info
        tab51321_expand = insert_dims(tab51321, len(n_info_q.shape), axis=0)
        n_info_q_expand = tf.expand_dims(n_info_q, axis=-1)
        is_greater_than_n_info_q = tf.greater_equal(tab51321_expand,
                                                    n_info_q_expand)
        tbs_ind = tf.argmax(
            tf.cumsum(1 - 2 * tf.cast(is_greater_than_n_info_q, rdtype),
                      axis=-1), axis=-1)
        tbs_ind = tf.minimum(tbs_ind + 1, tab51321.shape[0] - 1)
        tbs = tf.gather(tab51321, tbs_ind)
        return tf.cast(tbs, tf.int32)

    # ----------------- #
    # N. of code blocks #
    # ----------------- #
    num_cb = tf.where(
        n_info_q <= 3824,
            # if target_tb_size <= 3824
            tf.cast(tf.fill(shape, 1.0), rdtype),
            # else:
            tf.where(target_coderate <= 1/4,
                # if target_coderate <= 1/4:
                    tf.cast(tf.math.ceil((n_info_q + 24) / 3816), rdtype),
                # else:
                    tf.where(n_info_q > 8424,
                    # if target_tb_size > 8424:
                        tf.cast(tf.math.ceil((n_info_q + 24) / 8424), rdtype),
                    # else:
                        tf.cast(tf.fill(shape, 1.0), rdtype))))

    # ---------------------- #
    # TB size (n. info bits) #
    # ---------------------- #
    tb_size = tf.where(n_info_q <= 3824,
                       tbs_if_target_tbs_smaller_3824(),
                       tbs_if_target_tbs_higher_3824())

    num_cb = tf.cast(num_cb, tf.int32)

    # ---------------- #
    # TB/CB CRC length #
    # ---------------- #
    # TB CRC see 6.2.1 in 38.212
    tb_crc_length = tf.where(tb_size > 3824, tf.fill(
        tb_size.shape, 24), tf.fill(tb_size.shape, 16))

    # if tbs > max CB length, then CRC-24 is added; see 5.2.2 in 38.212
    cb_crc_length = tf.where(num_cb > 1, tf.fill(
        tb_size.shape, 24), tf.fill(tb_size.shape, 0))

    # ------- #
    # CB size #
    # ------- #
    cb_size = tf.cast((tb_size + tb_crc_length) /
                      num_cb, tf.int32) + cb_crc_length

    if verbose:
        print(f"Modulation order: {modulation_order.numpy()}")
        if target_coderate is not None:
            print(f"Target coderate: {target_coderate.numpy():.3f}")
        effective_rate = tb_size / num_coded_bits
        print(f"Effective coderate: {effective_rate.numpy():.3f}")
        print(f"Number of layers: {num_layers.numpy()}")
        print("------------------")
        print(f"Info bits per TB: {tb_size.numpy()}")
        print(f"TB CRC length: {tb_crc_length.numpy()}")
        print(f"Total number of coded TB bits: {num_coded_bits.numpy()}")
        print("------------------")
        print(f"Info bits per CB: {cb_size.numpy()}")
        print(f"Number of CBs: {num_cb.numpy()}")
        print(f"CB CRC length: {cb_crc_length.numpy()}")

    if not return_cw_length:
        return tb_size, cb_size, num_cb, tb_crc_length, cb_crc_length

    # --------------------------- #
    # Codeword length for each CB #
    # --------------------------- #
    # The last "num_last_blocks[...]" blocks have a codeword length of
    # "cw_length_last_blocks[...]"
    num_last_blocks = tf.math.mod(
        tf.cast(num_coded_bits / (num_layers * modulation_order),
                tf.int32), num_cb)
    cw_length_last_blocks = num_layers * modulation_order * \
        tf.cast(tf.math.ceil(num_coded_bits /
                (num_layers*modulation_order*num_cb)), tf.int32)

    # The first "num_first_blocks[...]" blocks have a codeword length of
    # "cw_length_first_blocks[...]"
    num_first_blocks = num_cb - num_last_blocks
    cw_length_first_blocks = num_layers * modulation_order * \
        tf.cast(tf.math.floor(num_coded_bits /
                (num_layers*modulation_order*num_cb)), tf.int32)

    # Flatten
    num_last_blocks = tf.reshape(num_last_blocks, [-1])
    cw_length_last_blocks = tf.reshape(cw_length_last_blocks, [-1])
    num_first_blocks = tf.reshape(num_first_blocks, [-1])
    cw_length_first_blocks = tf.reshape(cw_length_first_blocks, [-1])

    def populate_tensor(rep1, val1, rep2, val2):
        # Construct a tensor whose i-th row is [[val1[i]]*rep1[i],
        # [val2[i]]*rep2[i], 0, ..., 0]
        num_cols = tf.reduce_max(rep1 + rep2)
        r = tf.range(num_cols)[tf.newaxis, ...]
        return tf.where(r<rep1[..., tf.newaxis], val1[..., tf.newaxis],
                tf.where(r<(rep1+rep2)[..., tf.newaxis], val2[..., tf.newaxis], 0))

    # Compute codeword lengths
    cw_length = populate_tensor(num_first_blocks, cw_length_first_blocks,
                                num_last_blocks, cw_length_last_blocks)

    # Reshape
    cw_length = tf.reshape(cw_length,
                        tf.concat([shape, [-1]], axis=0))

    if verbose:
        print(f"Output codeword lengths: {cw_length.numpy()}")

    return tb_size, cb_size, num_cb, tb_crc_length, cb_crc_length, cw_length


class TransportBlockNR(TransportBlock):
    r"""
    Computes the number and size (measured in n. bits) of code
    blocks within a 5G-NR compliant transport block, given the modulation order,
    coderate and the total number of coded bits of a transport block.
    Used in :class:`~sionna.sys.PHYAbstraction`. Inherits from
    :class:`sionna.phy.utils.TransportBlock` and wraps
    :func:`~sionna.phy.nr.utils.calculate_tb_size`

    Input
    -----

    modulation_order : [...], `tf.int32`
        Modulation order, i.e., number of bits per symbol,
        associated with the input MCS index.

    target_rate : [...], `tf.float32`
        Target coderate.

    num_coded_bits : [...], `tf.float32`
        Total number of coded bits across all codewords.

    Output
    ------

    cb_size : [...], `tf.int32`
        Code block (CB) size, i.e., the number of information bits
        per code block.

    num_cb : [...], `tf.int32`
        Number of code blocks that the transport block is segmented into.
    """
    def call(self,
             modulation_order,
             target_coderate,
             num_coded_bits,
             **kwargs):
        _, cb_size, num_cb, *_ = \
            calculate_tb_size(modulation_order,
                              target_coderate,
                              num_coded_bits=num_coded_bits,
                              tb_scaling=1.,
                              return_cw_length=False,
                              verbose=False)
        return cb_size, num_cb


class CodedAWGNChannelNR(SingleLinkChannel):
    # pylint: disable=line-too-long
    r"""Simulates a 5G-NR compliant single-link coded AWGN channel.
    Inherits from :class:`~sionna.phy.utils.SingleLinkChannel`

    Parameters
    ----------
    num_bits_per_symbol : `int` | `None` (default)
        Number of bits per symbol, i.e., modulation order.

    num_info_bits : `int` | `None` (default)
        Number of information bits per code block.

    target_coderate : `float` | `None` (default)
        Target code rate, i.e., the target ratio between the information and the
        coded bits within a block

    num_iter_decoder: `int` (default: 20)
        Number of decoder iterations. See
        :class:`~sionna.phy.fec.ldpc.decoding.LDPC5GDecoder`
        for more details.

    cn_update: "boxplus-phi" (default) | "boxplus" | "minsum" | "offset-minsum" | "identity" | callable
        Check node update rule. See
        :class:`~sionna.phy.fec.ldpc.decoding.LDPC5GDecoder` for more details.

    kwargs :
        Additional keyword arguments for
        :class:`~sionna.phy.fec.ldpc.decoding.LDPC5GDecoder`.

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
                 num_bits_per_symbol=None,
                 num_info_bits=None,
                 target_coderate=None,
                 num_iter_decoder=20,
                 cn_update_decoder="boxplus-phi",
                 precision=None,
                 **kwargs):

        super().__init__(num_bits_per_symbol,
                         num_info_bits,
                         target_coderate,
                         precision=precision)
        self._num_iter_decoder = num_iter_decoder
        self._cn_update_decoder = cn_update_decoder
        self._kwargs = kwargs

    def call(self,
             batch_size,
             ebno_db):
        # pylint: disable=import-outside-toplevel
        from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder

        # Set the QAM constellation
        self.constellation = Constellation("qam",
                                           self.num_bits_per_symbol)

        # Set the Mapper/Demapper
        self.mapper = Mapper(constellation=self.constellation)
        self.demapper = Demapper("app",
                                 constellation=self.constellation)

        self.binary_source = BinarySource()
        self.awgn_channel = AWGN()

        # 5G code block encoder
        self.encoder = LDPC5GEncoder(
            self.num_info_bits,
            self.num_coded_bits,
            num_bits_per_symbol=self.num_bits_per_symbol)

        # 5G code block decoder
        self.decoder = LDPC5GDecoder(
            self.encoder,
            hard_out=True,
            num_iter=self._num_iter_decoder,
            cn_update=self._cn_update_decoder,
            **self._kwargs)

        # Noise power
        no = ebnodb2no(ebno_db,
                       num_bits_per_symbol=self.num_bits_per_symbol,
                       coderate=self.target_coderate)

        # Generate random information bits
        bits = self.binary_source([batch_size, self.num_info_bits])

        # Encode bits
        codewords = self.encoder(bits)

        # Map coded bits to complex symbols
        x = self.mapper(codewords)

        # Pass through an AWGN channel
        y = self.awgn_channel(x, no)

        # Compute log-likelihooh ratio (LLR)
        llr = self.demapper(y, no)

        # Decode transmitted bits
        bits_hat = self.decoder(llr)

        return bits, bits_hat
