# pylint: disable=line-too-long
#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""
Physical layer abstraction for Sionna SYS
"""

import json
import time
import os
import logging
import warnings
import datetime
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sionna.phy import Block, config
from sionna.phy.utils import gather_from_batched_indices, \
    to_list, dict_keys_to_int, DeepUpdateDict, sim_ber, \
    Interpolate, MCSDecoder, TransportBlock, SingleLinkChannel, \
    scalar_to_shaped_tensor, SplineGriddataInterpolation, \
    lin_to_db
from sionna.sys import EffectiveSINR, EESM
from sionna.phy.nr.utils import MCSDecoderNR, TransportBlockNR, \
    CodedAWGNChannelNR


class PHYAbstraction(Block):
    # pylint: disable=line-too-long
    r"""
    Class for physical layer abstraction 
    
    For a given
    signal-to-interference-plus-noise-ratio (SINR) provided on a per-stream
    basis, and for a given modulation order, coderate,
    and number of coded bits specified for each user, it produces the
    corresponding number of successfully decoded bits, 
    HARQ feedback, effective SINR, block error rate (BLER), and transport BLER
    (TBLER).

    At object instantiation, precomputed BLER tables are loaded and interpolated
    on a fine (SINR, code block size) grid for each modulation and coding scheme
    (MCS) index.

    When the object is called, the post-equalization SINR is first converted to
    an effective SINR. Then, the 
    effective SINR is used to retrieve the BLER from pre-computed and
    interpolated tables. Finally, the BLER determines the TBLER, which
    represents the probability that at least one code block is incorrectly
    received.

    Parameters
    ----------

    interp_fun : instance of :class:`~sionna.phy.utils.Interpolate` | `None` (default)
        Function for interpolating data defined on rectangular or unstructured
        grids, used for BLER and SINR interpolation.
        If `None`, it is set to an instance of
        :class:`~sionna.phy.utils.SplineGriddataInterpolation`.

    mcs_decoder_fun : instance of :class:`~sionna.phy.utils.MCSDecoder` | `None` (default)
        Function mapping MCS indices to modulation order and coderate.
        If `None`, it is set to an instance of
        :class:`~sionna.phy.nr.utils.MCSDecoderNR`.

    transport_block_fun : instance of :class:`~sionna.phy.utils.TransportBlock` | `None` (default)
        Function computing the number and size (measured in bits) of code
        blocks within a transport block.
        If `None`, it is set to an instance of
        :class:`~sionna.phy.nr.utils.TransportBlockNR`.

    sinr_effective_fun : instance of :class:`~sionna.sys.EffectiveSINR` | `None` (default)
        Function computing the effective SINR.
        If `None`, it is set to an instance of
        :class:`~sionna.sys.EESM`.

    load_bler_tables_from : `str` | list of `str` (default: "default")
        Name of file(s) containing pre-computed SINR-to-BLER tables for different
        categories, tables indices, MCS indices, SINR and code block sizes. If
        "default", then the pre-computed tables stored in
        "phy/abstraction/bler_tables/" folder are loaded.

    snr_db_interp_min_max_delta : [3], `tuple` (default: (-5, 30.01, .1))
        Tuple of (`min`, `max`, `delta`)
        values [dB] defining the list of SINR [dB] values at which the BLER is
        interpolated, as `min, min+delta, min+2*delta,...,` up until `max`

    cbs_interp_min_max_delta : [3], `tuple` (default: (24, 8448, 100))
        Tuple of (`min`, `max`, `delta`)
        values defining the list of code block size values at which the BLER and
        SINR are interpolated, as `min, min+delta, min+2*delta,...,max`

    bler_interp_delta : `float` (default: 0.01)
        Spacing of the BLER grid at which SINR is interpolated

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    kwargs :
        Additional inputs for ``bler_snr_interp_fun``, ``mcs_decoder_fun``,
        ``transport_block_fun`` 

    Input
    -----

    mcs_index : [..., num_ut], `tf.int32`
        MCS index for each user

    sinr : [..., num_ofdm_symbols, num_subcarriers, num_ut, num_streams_per_ut], `tf.float` | `None` (default)
        Post-equalization SINR in linear scale for each OFDM symbol, subcarrier,
        user and stream. 
        If `None`, then ``sinr_eff`` and ``num_allocated_re`` are both required.

    sinr_eff : [..., num_ut], `tf.float` | `None` (default)
        Effective SINR in linear scale for each user. 
        If `None`, then ``sinr`` is required.

    num_allocated_re : [..., num_ut], `tf.int32` | `None` (default)
        Number of allocated resources in a slot, computed across OFDM symbols,
        subcarriers and streams, for each user.
        If `None`, then ``sinr`` is required.

    mcs_table_index : [..., num_ut], `tf.int32` | `int` (default: 1)ß
        MCS table index. For further details, refer to the 
        :ref:`mcs_table_cat_note`.

    mcs_category : [..., num_ut], `tf.int32` | `int` (default: 0)
        MCS table category. For further details, refer to the 
        :ref:`mcs_table_cat_note`.

    check_mcs_index_validity : `bool` (default: `True`)
        If `True`, an ValueError is thrown is the input MCS indices are not
        valid for the given configuration

    Output
    ------

    num_decoded_bits : [..., num_ut], `tf.int32`
        Number of successfully decoded bits for each user

    harq_feedback : [..., num_ut], -1 | 0 | 1
        If 0 (1, resp.), then a NACK (ACK, resp.) is received. If -1, feedback
        is missing since the user is not scheduled for transmission.

    sinr_eff : [..., num_ut], `tf.float`
        Effective SINR in linear scale for each user

    tbler : [..., num_ut], `tf.float`
        Transport block error rate (BLER) for each user

    bler : [..., num_ut], `tf.float`
        Block error rate (BLER) for each user

    Note
    ----

    In this class, the terms SNR (signal-to-noise ratio) and SINR
    (signal-to-interference-plus-noise ratio) can be used interchangeably.
    This is because the equivalent AWGN model used for BLER mapping does not
    explicitly account for interference. 

    Example
    -------
    .. code-block:: Python

        import numpy as np
        from sionna.sys import PHYAbstraction, EESM
        from sionna.phy.nr.utils import MCSDecoderNR, TransportBlockNR
        from sionna.phy.utils import SplineGriddataInterpolation

        # Instantiate the class for BLER and SINR interpolation
        bler_snr_interp_fun = SplineGriddataInterpolation()
        # Instantiate the class for mapping MCS to modulation order and coderate
        # in 5G NR
        mcs_decoder_fun = MCSDecoderNR()
        # Instantiate the class for computing the number and size of code blocks
        # within a transport block in 5G NR
        transport_block_fun = TransportBlockNR()
        # Instantiate the class for computing the effective SINR
        sinr_effective_fun = EESM()

        # By instantiating a PHYAbstraction object, precomputed BLER tables are
        # loaded and interpolated on a fine (SINR, code block size) grid for each MCS
        phy_abs = PHYAbstraction(
            bler_snr_interp_fun=bler_snr_interp_fun,
            mcs_decoder_fun=mcs_decoder_fun,
            transport_block_fun=transport_block_fun,
            sinr_effective_fun=sinr_effective_fun)

        # Plot a BLER table
        phy_abs.plot(plot_subset={'category': {0: {'index': {1: {'MCS': 14}}}}},
                        show=True);

    .. figure:: ../figures/category0_table1_mcs14.png
        :align: center
        :width: 70%

    .. code-block:: Python

        # One can also compute new BLER tables
        # SINR values and code block sizes @ new simulations are performed
        snr_dbs = np.linspace(-5, 25, 5)
        cb_sizes = np.arange(24, 8448, 1000)
        # MCS values @ new simulations are performed
        sim_set = {'category': {
            0:
            {'index': {
                1: {'MCS': [15]}
            }}}}

        # Compute new tables
        new_table = phy_abs.new_bler_table(
            snr_dbs,
            cb_sizes,
            sim_set,
            max_mc_iter=15,
            batch_size=10,
            verbose=True)
    """
    def __init__(self,
                 interp_fun=None,
                 mcs_decoder_fun=None,
                 transport_block_fun=None,
                 sinr_effective_fun=None,
                 load_bler_tables_from='default',
                 snr_db_interp_min_max_delta=(-5, 30.01, .1),
                 cbs_interp_min_max_delta=(24, 8448, 100),
                 bler_interp_delta=0.01,
                 precision=None,
                 **kwargs):

        super().__init__(precision=precision)

        # Set default values
        if interp_fun is None:
            interp_fun = SplineGriddataInterpolation()
        if mcs_decoder_fun is None:
            mcs_decoder_fun = MCSDecoderNR(precision=precision)
        if transport_block_fun is None:
            transport_block_fun = TransportBlockNR(precision=precision)
        if sinr_effective_fun is None:
            sinr_effective_fun = EESM(precision=precision)

        # Check inputs
        if not isinstance(interp_fun, Interpolate):
            raise ValueError("interp_fun must be an instance of "
                             "sionna.phy.utils.Interpolate")
        if not isinstance(mcs_decoder_fun, MCSDecoder):
            raise ValueError("mcs_decoder_fun must be an instance of "
                             "sionna.phy.utils.MCSDecoder")
        if not isinstance(transport_block_fun, TransportBlock):
            raise ValueError("transport_block_fun must be a subclass of "
                             "sionna.phy.utils.TransportBlock")
        if not isinstance(sinr_effective_fun, EffectiveSINR):
            raise ValueError("eff_sinr_fun must be a subclass of "
                             "sionna.phy.utils.EffectiveSINR")

        # -------------- #
        # Initialization #
        # -------------- #
        self._kwargs = kwargs
        self._bler_table = None
        self._bler_table_interp = None
        self._snr_table_interp = None

        # ------------- #
        # Instantiation #
        # ------------- #
        # Function interpolating (CBS, SNR) -> BLER
        self._bler_interp_fun = interp_fun.struct
        # Function interpolating (CBS, BLER) -> SNR
        self._snr_interp_fun = interp_fun.unstruct
        # Function mapping MCS index to modulation order and coderate
        self._mcs_decoder_fun = mcs_decoder_fun
        # Function computing number and size of code blocks
        self._transport_block_fun = transport_block_fun
        # Function computing the effective SINR
        self._sinr_effective_fun = sinr_effective_fun

        # Interpolation grid
        # List of CBS values at which the BLER is interpolated
        self._cbs_interp = None
        # List of SNR values at which the BLER is interpolated
        self._snr_dbs_interp = None
        # List of BLER values at which the SNR is interpolated
        self._blers_interp = None

        if load_bler_tables_from == 'default':
            filenames = ['bler_tables/PUSCH_table1.json',
                         'bler_tables/PUSCH_table2.json',
                         'bler_tables/PDSCH_table1.json',
                         'bler_tables/PDSCH_table2.json',
                         'bler_tables/PDSCH_table3.json',
                         'bler_tables/PDSCH_table4.json']
            self.bler_table_filenames = [os.path.join(
                os.path.dirname(os.path.abspath(__file__)), f)
                for f in filenames]
        else:
            self.bler_table_filenames = load_bler_tables_from

        # SNR/BLER interpolation grid
        self.snr_db_interp_min_max_delta = snr_db_interp_min_max_delta
        self.cbs_interp_min_max_delta = cbs_interp_min_max_delta
        self.bler_interp_delta = bler_interp_delta

    @staticmethod
    def load_table(filename):
        r"""
        Loads a table stored in JSON file.

        Input
        -----
        filename : `str`
            Name of the JSON file containing the table

        Output
        ------
        : `dict`
            table loaded from file
        """
        with open(filename, "r", encoding="utf-8") as f:
            table = json.load(f, object_hook=dict_keys_to_int)
        return table

    # ----------- #
    # BLER tables #
    # ----------- #
    @property
    def bler_table_filenames(self):
        r"""
        `str` | list of `str` : Get/set the absolute path name of the files
        containing BLER tables 
        """
        return self._bler_table_filenames

    @bler_table_filenames.setter
    def bler_table_filenames(self, value):
        self._bler_table_filenames = to_list(value)
        # Load the table
        self._bler_table = DeepUpdateDict({'category': {}})
        for f in self.bler_table_filenames:
            try:
                with open(f, "r", encoding="utf-8") as f:
                    bler_subtable = json.load(f, object_hook=dict_keys_to_int)
                    # Merge with the existing one
                    self._bler_table.deep_update(
                        bler_subtable,
                        stop_at_keys=('CBS', 'SNR_db'))
            except FileNotFoundError:
                warnings.warn(f"BLER table file '{f}' does not exist. "
                              "Skipping...")
        if self._bler_table == {}:
            warnings.warn("No BLER table found. You can generate them via "
                          "PHYAbstraction.compute_bler_table method.")
        # Check table validity
        self.validate_bler_table()

    def _get_batch_size_interp_mat(self):
        # Compute the batch size of interpolation tensors
        categories = list(self.bler_table['category'].keys())
        max_table_idx_list, max_mcs_list = [], []
        for ch in categories:
            table_idx_list = list(self.bler_table['category']
                                  [ch]['index'].keys())
            max_table_idx_list.append(max(table_idx_list))
            for table_idx in table_idx_list:
                mcs_list = list(self.bler_table['category'][ch]
                                ['index'][table_idx]['MCS'].keys())
                max_mcs_list.append(max(mcs_list))
        if len(categories) > 0 and len(max_table_idx_list) > 0 and \
                len(max_mcs_list) > 0:
            return [max(categories) + 1,
                    max(max_table_idx_list),
                    max(max_mcs_list) + 1]
        return [0, 0, 0]

    @property
    def bler_table(self):
        r"""
        `dict` (read-only) : Collection of tables containing BLER
        values for different values of SNR, MCS table, MCS index and CB size.
        ``bler_table['category'][cat]['index'][mcs_table_index]['MCS'][mcs]['CBS'][cb_size]``
        contains the lists of BLER values. 
        ``bler_table['category'][cat]['index'][mcs_table_index]['MCS'][mcs]['SNR_db']``
        contains the list of SNR values. 
        ``bler_table['category'][cat]['index'][mcs_table_index]['MCS'][mcs]['EbN0_db']``
        contains the list of :math:`E_b/N_0` values
        """
        return self._bler_table

    # ------------------- #
    # Interpolated tables #
    # ------------------- #
    @property
    def bler_table_interp(self):
        r"""
        [n_categories,n_tables,n_mcs,n_cbs_index,n_snr], `tf.float` (read-only): Tensor 
        containing BLER values 
        interpolated across SINR and CBS values, for different categories and
        MCS table indices. The first axis accounts for
        the category, e.g., 'PDSCH' or 'PUSCH' in 5G-NR, the second axis corresponds to
        the 38.214 MCS table index while the third axis carries the MCS index. 
        """
        return self._bler_table_interp

    @property
    def snr_table_interp(self):
        r"""
        [n_categories,n_tables,n_mcs,n_cbs_index,n_bler], `tf.float` (read-only) : Tensor
        containing SINR values interpolated across BLER and CBS values, for
        different categories and MCS table indices. 
        The first axis accounts for
        the category, e.g., 'PDSCH' or 'PUSCH' in 5G-NR, the second axis corresponds to
        the 38.214 MCS table index and the third axis accounts for the MCS
        index. 
        """
        return self._snr_table_interp

    # ------------------ #
    # Interpolation grid #
    # ------------------ #
    @property
    def snr_db_interp_min_max_delta(self):
        r"""
        [3], `tuple` : Get/set the tuple of (`min`, `max`, `delta`)
        values [dB] defining the list of SINR values at which the BLER is
        interpolated, as `min, min+delta, min+2*delta,...,` up until `max`
        """
        return self._snr_db_interp_min_max_delta

    @snr_db_interp_min_max_delta.setter
    def snr_db_interp_min_max_delta(self,
                                    value):
        if hasattr(value, '__len__') and len(value) == 3:
            self._snr_db_interp_min_max_delta = value
        else:
            raise ValueError("snr_db_interp_min_max_delta must have length 3")
        self._snr_dbs_interp = np.arange(self._snr_db_interp_min_max_delta[0],
                                         self._snr_db_interp_min_max_delta[1],
                                         self._snr_db_interp_min_max_delta[2])

        if (self.bler_table is not None) and \
                (self._cbs_interp is not None):
            # Interpolate BLER
            self._interpolate_bler()

    @property
    def cbs_interp_min_max_delta(self):
        r"""
        [3], `tuple` : Get/set the tuple of (`min`, `max`, `delta`)
        values defining the list of code block size values at which the BLER and
        SINR are interpolated, as `min, min+delta, min+2*delta,...,` up until `max`.
        """
        return self._cbs_interp_min_max_delta

    @cbs_interp_min_max_delta.setter
    def cbs_interp_min_max_delta(self,
                                 value):
        if hasattr(value, '__len__') and len(value) == 3:
            self._cbs_interp_min_max_delta = value
        else:
            raise ValueError("cbs_interp_min_max_delta must have length 3")
        self._cbs_interp = np.arange(self._cbs_interp_min_max_delta[0],
                                     self._cbs_interp_min_max_delta[1],
                                     self._cbs_interp_min_max_delta[2])
        if self.bler_table is not None:
            if self._blers_interp is not None:
                # Interpolate SNR
                self._interpolate_snr()
            if self._snr_dbs_interp is not None:
                # Interpolate BLER
                self._interpolate_bler()

    @property
    def bler_interp_delta(self):
        r"""
        `float`: Get/set the spacing of the BLER grid at which SINR is
        interpolated 
        """
        return self._bler_interp_delta

    @bler_interp_delta.setter
    def bler_interp_delta(self,
                          value):
        self._bler_interp_delta = value
        self._blers_interp = np.arange(0, 1, self._bler_interp_delta)
        if (self.bler_table is not None) and \
                (self._cbs_interp is not None):
            # Interpolate BLER
            self._interpolate_snr()

    def get_idx_from_grid(self,
                          val,
                          which):
        r"""
        Retrieves the index of a SINR of CBS value in the interpolation grid.

        Input
        -----

        val : [...], `tf.float`
            Values to be quantized

        which : "snr | "cbs"
            Whether the values are SNR (equivalent to SINR) or CBS

        Output
        ------
        idx : [...], `tf.int32`
            Index of the values in the interpolation grid
        """
        tf.debugging.assert_equal(
            which in ['snr', 'cbs'],
            True,
            message="which must be 'snr' or 'cbs'")
        if which == 'snr':
            len_grid = len(self._snr_dbs_interp)
            min_max_delta = self.snr_db_interp_min_max_delta
        else:
            len_grid = len(self._cbs_interp)
            min_max_delta = self.cbs_interp_min_max_delta
        min_grid = min_max_delta[0]
        delta_grid = min_max_delta[2]
        idx = tf.cast(tf.round((val - min_grid) / delta_grid),
                      tf.int32)
        idx = tf.minimum(idx, len_grid - 1)
        idx = tf.maximum(idx, 0)
        return idx

    # ------------- #
    # Retrieve BLER #
    # ------------- #
    def get_bler(self,
                 mcs_index,
                 mcs_table_index,
                 mcs_category,
                 cb_size,
                 snr_eff):
        r"""
        Retrieves from interpolated tables the BLER corresponding to a certain
        table index, MCS, CB size, and SINR values provided as input. 
        If the corresponding interpolated table is not available, it returns
        `Inf`. 

        Input
        -----

        mcs_index : [...], `tf.int32`
            MCS index for each user

        mcs_table_index : [...], `tf.int.32` | `int`
            MCS table index for each user. For further details, refer to the
            :ref:`mcs_table_cat_note`. 

        mcs_category : [...], `tf.int32`
            MCS table category for each user. For further details, refer to the
            :ref:`mcs_table_cat_note`. 

        cb_size : [...], `tf.int32`
            Code block size for each user

        snr_eff : [...], `tf.float`
            Effective SINR for each user

        Output
        ------
        bler : [...], `tf.float`
            BLER corresponding to the input channel type, table index, MCS, CB
            size and SINR, retrieved from internal interpolation tables

        """
        # Cast inputs to appropriate type and shape
        snr_eff = tf.cast(snr_eff, self.rdtype)
        shape = snr_eff.shape
        mcs_category = scalar_to_shaped_tensor(mcs_category,
                                               tf.int32,
                                               shape)
        mcs_index = scalar_to_shaped_tensor(mcs_index,
                                            tf.int32,
                                            shape)
        mcs_table_index = scalar_to_shaped_tensor(mcs_table_index,
                                                  tf.int32,
                                                  shape)
        cb_size = scalar_to_shaped_tensor(cb_size,
                                          tf.int32,
                                          shape)

        # Convert SNR to dB
        snr_eff_db = lin_to_db(snr_eff, precision=self.precision)

        # Quantize the SNR [dB] and CBS to the corresponding interpolation index
        snr_db_idx = self.get_idx_from_grid(snr_eff_db, 'snr')
        cbs_idx = self.get_idx_from_grid(cb_size, 'cbs')

        # Stack indices to extract BLER from interpolated table
        idx = tf.stack([mcs_category,
                        mcs_table_index - 1,
                        mcs_index,
                        cbs_idx,
                        snr_db_idx], axis=-1)

        # Compute BLER
        bler = gather_from_batched_indices(self.bler_table_interp,
                                           idx)
        bler = tf.cast(bler, self.rdtype)

        return bler

    def call(self,
             mcs_index,
             sinr=None,
             sinr_eff=None,
             num_allocated_re=None,
             mcs_table_index=1,
             mcs_category=0,
             check_mcs_index_validity=True,
             **kwargs):

        tf.debugging.assert_equal(
            (sinr is not None) ^
            ((sinr_eff is not None) and (num_allocated_re is not None)),
            True,
            message="Either 'sinr' or "
            "('sinr_eff','num_allocated_re') is required as input")

        if sinr is not None:
            # Total number of allocated streams across all resource elements
            # [..., num_ut]
            num_allocated_re = tf.reduce_sum(
                tf.cast(sinr > 0, tf.int32),
                axis=[-4, -3, -1])

            # Effective SINR
            # [..., num_ut]
            sinr_eff = self._sinr_effective_fun(
                sinr,
                mcs_index=mcs_index,
                mcs_table_index=mcs_table_index,
                mcs_category=mcs_category,
                per_stream=False,
                **kwargs)
        else:
            sinr_eff = tf.cast(sinr_eff, self.rdtype)
            num_allocated_re = tf.cast(num_allocated_re, tf.int32)

        # Whether a user is scheduled
        # [..., num_ut]
        ut_is_scheduled = num_allocated_re > 0

        # Convert MCS index to modulation order and coderate
        # [..., num_ut]
        modulation_order, target_coderate = self._mcs_decoder_fun(
            mcs_index,
            mcs_table_index,
            mcs_category,
            check_index_validity=check_mcs_index_validity,
            **kwargs)

        # Compute the number of coded bits
        num_coded_bits = modulation_order * num_allocated_re

        # Compute n. and size of Code Blocks (CBs) in a Transport Block
        # [..., num_ut]
        cb_size, num_cb = self._transport_block_fun(
            modulation_order,
            target_coderate,
            num_coded_bits,
            **kwargs)

        # Retrieve the BLER from the stored tables
        # [..., num_ut]
        bler = self.get_bler(mcs_index,
                             mcs_table_index,
                             mcs_category,
                             cb_size,
                             sinr_eff)

        # Compute TBLER = Pr(at least a CB is incorrectly received)
        # [..., num_ut]
        one = tf.cast(1, bler.dtype)
        tbler = one - tf.math.pow(one - bler,
                                  tf.cast(num_cb, bler.dtype))

        # Set BLER=-1 and TBLER=-1 for non-scheduled UTs
        bler = tf.where(ut_is_scheduled,
                        bler,
                        -1)
        tbler = tf.where(ut_is_scheduled,
                         tbler,
                         -1)

        # HARQ feedback
        rnd = config.tf_rng.uniform(
            tbler.shape, minval=0, maxval=1, dtype=self.rdtype)
        harq_feedback = tf.where(rnd < tbler,
                                 tf.cast(0, tf.int32),
                                 tf.cast(1, tf.int32))

        # Successfully decoded bits for each user
        # [..., num_ut]
        num_decoded_bits = harq_feedback * num_cb * cb_size
        num_decoded_bits = tf.where(ut_is_scheduled,
                                    num_decoded_bits,
                                    tf.cast(0, tf.int32))

        # Assign HARQ=-1 for the non-scheduled UTs
        harq_feedback = tf.where(ut_is_scheduled,
                                 harq_feedback,
                                 tf.cast(-1, tf.int32))

        return num_decoded_bits, harq_feedback, sinr_eff, tbler, bler

    # -------------------- #
    # Interpolation method #
    # -------------------- #
    def _interpolate_bler(self):
        """
        Interpolates the BLER over a fine (CBS, SINR) grid
        """
        if self.bler_table is None:
            raise ValueError('BLER table is not provided; ' +
                             'Interpolation cannot be performed')
        if self._cbs_interp is None:
            raise ValueError('CBS interpolation grid is not provided; ' +
                             'Interpolation cannot be performed')

        interp_batch_size = self._get_batch_size_interp_mat()

        # [num_category, num_table_idx, num_mcs_index, num_cbs_interp, num_snr_interp]
        self._bler_table_interp = np.full(
            interp_batch_size +
            [len(self._cbs_interp), len(self._snr_dbs_interp)], np.inf)

        for category in self.bler_table['category']:

            for table_idx in self.bler_table['category'][category]['index']:

                table_mcs = self.bler_table['category'][category]['index'][table_idx]['MCS']

                for mcs in table_mcs:

                    cbs_vec = list(table_mcs[mcs]['CBS'].keys())
                    snr_vec = table_mcs[mcs]['SNR_db']

                    # Collect BLER values
                    bler_val = np.zeros((len(cbs_vec), len(snr_vec)))
                    for idx_cbs, cbs in enumerate(cbs_vec):
                        bler_val[idx_cbs, :] = \
                            table_mcs[mcs]['CBS'][cbs]['BLER']

                    # Interpolate BLER
                    try:
                        bler_interp = self._bler_interp_fun(
                            bler_val,
                            cbs_vec,
                            snr_vec,
                            self._cbs_interp,
                            self._snr_dbs_interp,
                            **self._kwargs)
                    except ValueError as e:
                        warnings.warn(
                            f"SINR-to-BLER interpolation failed for "
                            f"category {category}, "
                            f"index {table_idx}, MCS {mcs}.\nError: {e}")
                        continue

                    # Ensure BLER is within 0 and 1
                    bler_interp = np.minimum(bler_interp, 1)
                    bler_interp = np.maximum(bler_interp, 0)

                    # Store it
                    self._bler_table_interp[
                        category, table_idx-1, mcs, ::] = bler_interp

        # Convert to tensor
        self._bler_table_interp = \
            tf.convert_to_tensor(self._bler_table_interp,
                                 dtype=self.rdtype)

    def _interpolate_snr(self):
        """
        Interpolates the SINR table over a fine (CBS, BLER) grid
        """
        if self.bler_table is None:
            raise ValueError('BLER table is not provided; ' +
                             'Interpolation cannot be performed')
        if self._blers_interp is None:
            raise ValueError('BLER interpolation grid is not provided; ' +
                             'Interpolation cannot be performed')

        interp_batch_size = self._get_batch_size_interp_mat()
        self._snr_table_interp = np.full(interp_batch_size +
                                         [len(self._cbs_interp), len(self._blers_interp)], np.inf)

        for category in self.bler_table['category']:

            for table_index in self.bler_table['category'][category]['index']:

                table_mcs = self.bler_table['category'][category]['index'][table_index]['MCS']

                for mcs in table_mcs:
                    # Collect values of SNR, CBS and BLER in arrays
                    snr_vec = table_mcs[mcs]['SNR_db']
                    cbs_vec = list(table_mcs[mcs]['CBS'].keys())
                    snr_vec_tile = np.tile(snr_vec,
                                           len(cbs_vec))
                    cbs_vec_rep = np.repeat(cbs_vec,
                                            len(snr_vec))
                    bler_vec = [bler for cbs in cbs_vec for bler in
                                table_mcs[mcs]['CBS'][cbs]['BLER']]
                    try:
                        # Interpolate the SNR as a function of CBS and BLER
                        snr_interp = self._snr_interp_fun(snr_vec_tile,
                                                          cbs_vec_rep,
                                                          bler_vec,
                                                          self._cbs_interp,
                                                          self._blers_interp,
                                                          **self._kwargs)
                    except ValueError as e:
                        warnings.warn(
                            f"BLER-to-SINR interpolation failed for "
                            f"category {category}, "
                            f"index {table_index}, MCS {mcs}.\n"
                            f"Error message: {e}")
                        continue
                    self._snr_table_interp[
                        category, table_index-1, mcs, ::] = snr_interp

        # Convert to tensor
        self._snr_table_interp = \
            tf.convert_to_tensor(self._snr_table_interp,
                                 dtype=self.rdtype)

    def validate_bler_table(self):
        r"""
        Validates the dictionary structure of ``self.bler_table``

        Output
        ------

        : `bool` | `ValueError`
            Returns `True` if ``self.bler_table`` has a valid structure.
            Else, a `ValueError` is raised
        """
        if not isinstance(self.bler_table, dict):
            raise ValueError('Must be a dictionary')

        if np.any(np.array(list(self.bler_table['category'].keys())) < 0):
            raise ValueError("Categories must nonegative integers")

        for _, bler_table_tmp in self.bler_table['category'].items():

            if set(bler_table_tmp.keys()) != set(['index']):
                raise ValueError("Key must be 'index'")

            if np.any(np.array(list(bler_table_tmp['index'].keys())) < 1):
                raise ValueError("Table indices must be positive integers")

            for table_index in bler_table_tmp['index']:

                if set(bler_table_tmp['index']
                       [table_index].keys()) != set(['MCS']):
                    raise ValueError("Key must be 'MCS'")

                if np.any(np.array(list(bler_table_tmp['index']
                                        [table_index]['MCS'].keys())) < 0):
                    raise ValueError("MCS indices must be nonnegative integers")

                for mcs in bler_table_tmp['index'][table_index]['MCS']:

                    if (set(bler_table_tmp['index']
                        [table_index]['MCS']
                            [mcs].keys()) != set(['CBS', 'SNR_db'])):
                        raise ValueError("Keys must be ['CBS', 'SNR_db']")

                    for cbs in bler_table_tmp['index'][table_index]['MCS'][mcs]['CBS']:

                        if (set(bler_table_tmp['index'][table_index]
                            ['MCS'][mcs]['CBS'][cbs].keys()) !=
                                set(['BLER'])):
                            raise ValueError("Keys must be 'BLER'")
        return True

    def plot(self,
             plot_subset='all',
             show=True,
             save_path=None):
        r"""
        Visualizes and/or saves to file the SINR-to-BLER tables

        Input
        -----

        plot_subset : `dict` | "all"
            Dictionary containing the list of MCS indices to consider, stored at
            ``plot_subset['category'][category]['index'][mcs_table_index]['MCS']``.
            If "all", then plots are produced for all available BLER tables.

        show : `bool` (default: `True`)
            If `True`, then plots are visualized

        save_path : `str` | `None` (default)
            Folder path where BLER plots are saved. If `None`, then plots are
            not saved

        Output
        ------
        fignames : `list`
            List of names of files containing BLER plots
        """
        if self.bler_table is None:
            raise ValueError("Plots cannot be produced as "
                             "self.bler_table has not been loaded or computed")

        # Create folder if it does not exist
        if save_path is not None:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                logging.info('\nCreated folder %s\n', save_path)

        fignames = []
        if plot_subset == 'all':
            plot_subset = self.bler_table

        for category, plot_subset_cat in plot_subset['category'].items():
            for table_index, plot_subset_cat_tab in plot_subset_cat['index'].items():
                for mcs in to_list(plot_subset_cat_tab['MCS']):
                    try:
                        num_bits_per_symbol, coderate = self._mcs_decoder_fun(
                            mcs,
                            table_index,
                            category,
                            **self._kwargs)
                    except tf.errors.InvalidArgumentError as e:
                        print(f'Invalid (category={category}, '
                              f'index={table_index}, '
                              f'MCS={mcs}) combination. \n'
                              f'Error message: {e}\n'
                              f'Skipping...\n')
                        continue

                    num_bits_per_symbol = num_bits_per_symbol.numpy()
                    coderate = coderate.numpy()

                    # SNR value at the channel capacity for the spectral
                    # efficiency associated to the current MCS
                    snr_shannon = 2**(num_bits_per_symbol * coderate) - 1
                    snr_shannon_db = 10 * np.log10(snr_shannon)

                    try:
                        snr_dbs = self.bler_table['category'][category]['index'][table_index]['MCS'][mcs]['SNR_db']

                        fig, ax = plt.subplots()
                        for cbs in self.bler_table['category'][category]['index'][table_index]['MCS'][mcs]['CBS']:

                            bler = self.bler_table['category'][category]['index'][table_index]['MCS'][mcs]['CBS'][cbs]['BLER']
                            ax.semilogy(snr_dbs,
                                        bler,
                                        label=f'code block size={cbs}')

                        ax.plot([snr_shannon_db]*2, ax.get_ylim(), '--k',
                                label='SNR @capacity')
                        ax.set_title(f'MCS index {mcs} (table category {category}, '
                                     f'index {table_index})')
                        ax.legend()
                        ax.grid(True)
                        ax.set_xlabel('SNR [dB]')
                        ax.set_ylabel('BLER')

                        # Save to file
                        if save_path is not None:
                            figname = f'category{category}_table' + \
                                f'{table_index}_mcs{mcs}.png'
                            figname = os.path.join(save_path, figname)
                            fig.savefig(figname)
                            fignames.append(figname)

                        if show:
                            plt.show()
                        plt.close(fig)
                    except KeyError as e:
                        print(f'\nBLER for (category={category}, index=' +
                              f'{table_index}, MCS={mcs}) not available ' +
                              f'for plotting. Error message: {e}.' +
                              '\nSkipping...')

        return fignames

    def new_bler_table(self,
                       snr_dbs,
                       cb_sizes,
                       sim_set,
                       channel=None,
                       filename=None,
                       write_mode='w',
                       batch_size=1000,
                       max_mc_iter=100,
                       target_bler=None,
                       graph_mode="graph",
                       early_stop=True,
                       filename_log=None,
                       verbose=True):
        # pylint: disable=line-too-long
        r"""
        Computes static tables mapping SNR values of an AWGN channel to the
        corresponding block error rate (BLER) via
        Monte-Carlo simulations for different MCS indices, code block sizes and
        channel types. 
        Note that the newly computed table is merged with the internal
        ``self.bler_table``.

        The simulation continues with the next SNR point after
        ``max_mc_iter`` batches of size ``batch_size`` have been simulated.
        Early stopping allows to stop the simulation after the first error-free SNR
        point or after reaching a certain ``target_ber`` or ``target_bler``. 
        For more details, please see :func:`~sionna.phy.utils.misc.sim_ber`.

        Input
        -----

        snr_dbs : `list` | `float`
            List of SNR [dB] value(s) at which the BLER is computed

        cb_sizes : `list` | `int`
            List of code block (CB) size(s) at which the BLER is computed

        sim_set : dict
            Dictionary contains the list of the MCS indices at which the BLER is
            computed via simulation. The dictionary structure is of the kind: 
            ``sim_set['category'][category]['index'][mcs_table_index]['MCS'][mcs_list]``.

        channel : instance of :class:`~sionna.phy.utils.SingleLinkChannel` | `None`
            Object for simulating single-link i.e., single-carrier and single-stream,
            channels. If `None`, it is set to an instance of
            :class:`~sionna.phy.nr.utils.CodedAWGNChannelNR`.

        filename : `str` | `None` (default)
            Name of JSON file where the BLER tables are saved. 
            If `None`, results are not saved.

        write_mode : 'w' (default) | 'a'
            If 'w', then ``bler_table_filename`` is rewritten.
            If 'a', then the produced results are appended to
            ``bler_table_filename``.

        batch_size : `int` (default: 2000)
            Batch size for Monte-Carlo BLER simulations

        max_mc_iter : `int` (default: 100)
            Maximum number of Monte-Carlo iterations per SNR point

        target_bler: `None` (default) | `tf.float32`
            The simulation stops after the first SNR point
            which achieves a lower block error rate as specified by ``target_bler``.
            This requires ``early_stop`` to be `True`.

        graph_mode: `None` | "graph" (default) | "xla"
            Execution mode of ``EquivalentChannel`` call method.
            If `None`, then ``EquivalentChannel`` is executed as is.

        num_iter_decoder: `int` (default: 20)
            Number of decoder iterations. See
            :class:`~sionna.phy.fec.ldpc.decoding.LDPC5GDecoder` for more details.

        cn_update: "boxplus-phi" (default) | "boxplus" | "minsum" | "offset-minsum" | "identity" | callable
            Check node update rule. See
            :class:`~sionna.phy.fec.ldpc.decoding.LDPC5GDecoder` for more details.

        filename_log : `str` | `None` (default)
            Name of logging file. 
            If `None`, logs are not produced.

        verbose : `bool` (default: `True`)
            If `True`, the simulation progress is visualized, as well as
            the names of files of results and figures

        Output
        ------

        new_table : `dict`
            Newly computed BLER table
        """

        def _log(msg,
                 level='info'):
            """
            Logging and printing simulation progress
            """
            if verbose:
                print(msg)
            if filename_log is not None:
                if level == 'info':
                    logging.info(msg)
                elif level == 'warning':
                    logging.warning(msg)
                elif level == 'error':
                    logging.error(msg)
                else:
                    raise ValueError("unrecognized 'level' input")

        if channel is None:
            channel = CodedAWGNChannelNR(precision=self.precision)

        # Check input validity
        if not isinstance(channel, SingleLinkChannel):
            raise ValueError("'channel' must be an instance of "
                             "sionna.phy.utils.SingleLinkChannel")

        if filename_log is not None:
            # Logging settings
            logging.basicConfig(
                filename=filename_log,
                filemode='w',  # 'w' for overwrite or 'a' for append
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s')

        if write_mode not in ['a', 'w']:
            raise ValueError("'write_mode' must be either 'a' " +
                             "for appending to file or 'w' for rewriting")

        snr_dbs = to_list(snr_dbs)
        cb_sizes = to_list(cb_sizes)

        if (filename is not None) and \
                os.path.isfile(filename) and (write_mode == 'a'):
            # Append to file
            new_table = self.load_table(filename)
        else:
            # (Re)write to file
            new_table = {'category': {}}

        # Total number of points to simulate
        n_sims_tot = len(cb_sizes) * sum(
            len(sim_set['category'][cat]['index'][ti]['MCS'])
            for cat in sim_set['category']
            for ti in sim_set['category'][cat]['index'])

        _log('\nBLER simulations started. ' +
             '\nTotal # (category, index, MCS, SINR) ' +
             f'points to simulate: {n_sims_tot}\n')

        # ---------------- #
        # BLER simulations #
        # ---------------- #
        n_sims_done = 0
        time_start = time.time()

        for category, sim_set_cat in sim_set['category'].items():
            if category not in new_table.keys():
                new_table['category'][category] = {'index': {}}

            for table_index, sim_set_tab in sim_set_cat['index'].items():

                if table_index not in new_table['category'][category]['index'].keys():
                    new_table['category'][category]['index'][table_index] = \
                        {'MCS': {}}

                for mcs in sim_set_tab['MCS']:

                    # Compute modulation order and code-rate associated with the
                    # MCS index, table index and channel type
                    try:
                        num_bits_per_symbol, coderate = self._mcs_decoder_fun(
                            mcs,
                            table_index,
                            category,
                            **self._kwargs)
                        num_bits_per_symbol = num_bits_per_symbol.numpy()
                        coderate = coderate.numpy()
                    except tf.errors.InvalidArgumentError as e:
                        _log(f'Invalid (category={category}, '
                             f'index={table_index}, '
                             f'MCS={mcs}) combination. \n'
                             f'Error message: {e}\n'
                             f'Skipping...\n',
                             level='warning')
                        continue

                    # Eb/N0, where Eb=energy per information (uncoded) bit
                    ebno_dbs = [x - 10 * np.log10(num_bits_per_symbol * coderate)
                                for x in snr_dbs]

                    # N. successful simulations for the MCS
                    n_sims_mcs = 0

                    for cbs in cb_sizes:
                        _log(f'\nSimulating category={category}, '
                             f'index={table_index}, '
                             f'CBS={cbs}, MCS={mcs}...\n')

                        try:
                            # Instantiate the AWGN coded channel
                            channel.num_bits_per_symbol = num_bits_per_symbol
                            channel.num_info_bits = cbs
                            channel.target_coderate = coderate

                            # Compute BLER via Monte-Carlo simulations
                            _, bler = sim_ber(
                                channel,
                                ebno_dbs,
                                batch_size,
                                max_mc_iter,
                                soft_estimates=False,
                                early_stop=early_stop,
                                target_bler=target_bler,
                                graph_mode=graph_mode,
                                forward_keyboard_interrupt=True,
                                verbose=verbose,
                                precision=self.precision)

                            n_sims_mcs += 1
                            if n_sims_mcs == 1:
                                # Initialize dictionary
                                new_table['category'][category]['index'][table_index]['MCS'][mcs] = {'CBS': {},
                                                                                                     'SNR_db': snr_dbs}

                            # record results in bler_table
                            new_table['category'][category]['index'][table_index]['MCS'][mcs]['CBS'][cbs] = {
                                'BLER': bler.numpy().tolist()}

                            # Write to JSON file
                            if filename is not None:
                                with open(filename, "w",
                                          encoding="utf-8") as file:
                                    json.dump(new_table, file, indent=6)
                                    _log(f'\nResults written in file '
                                         f'{os.path.abspath(filename)}\n')

                        except (ValueError, tf.errors.InvalidArgumentError) as e:
                            _log(f'\nBER/BLER simulations failed for '
                                 f'(category={category}, '
                                 f'index={table_index}, '
                                 f'CBS={cbs}, MCS={mcs})\n'
                                 f'Error message: {e}\n'
                                 f'Skipping...\n',
                                 level='error')

                        n_sims_done += 1

                        # Compute simulation progress and
                        # remaining execution time
                        progress = n_sims_done / n_sims_tot * 100
                        time_s = (time.time() - time_start) * \
                            (100-progress) / progress
                        time_s = datetime.timedelta(seconds=time_s)
                        time_hms = str(time_s).split(".", maxsplit=1)[0]
                        _log(f'\nPROGRESS: {progress:.2f}%\n'
                             f'Estimated remaining time: '
                             f'{time_hms} [h:m:s]\n')

        _log('\nSimulations completed!\n')

        # Merge the newly computed table with self.bler_table
        self._bler_table.deep_update(new_table,
                                     stop_at_keys=('CBS', 'SNR_db'))
        self.validate_bler_table()

        # Append file name to the internal list of loaded BLER files
        if filename is not None:
            self._bler_table_filenames.append(filename)

        return new_table
