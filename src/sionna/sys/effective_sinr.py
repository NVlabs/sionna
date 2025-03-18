# pylint: disable=line-too-long
#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""
Effective SINR computation for Sionna SYS
"""

from abc import abstractmethod
import warnings
import os
import json
import numpy as np
import tensorflow as tf
from sionna.phy.utils import expand_to_rank, DeepUpdateDict, dict_keys_to_int, \
    to_list, scalar_to_shaped_tensor, gather_from_batched_indices, db_to_lin
from sionna.phy import Block


class EffectiveSINR(Block):
    r"""
    Class template for computing the effective SINR from input SINR values
    across multiple subcarriers and streams

    Input
    -----

    sinr : [..., num_ofdm_symbols, num_subcarriers, num_ut, num_streams_per_ut], `tf.float`
        Post-equalization SINR in linear scale for different OFDM symbols,
        subcarriers, users and streams.
        If one entry is zero, the corresponding stream is considered as not
        utilized. 

    mcs_index : [..., num_ut], `tf.int32` (default: `None`)
        Modulation and coding scheme (MCS) index for each user

    mcs_table_index : [..., num_ut], `tf.int32` (default: `None`)
        MCS table index for each user. For further details, refer to the 
        :ref:`mcs_table_cat_note`.

    mcs_category : [..., num_ut], `tf.int32` (default: `None`)
        MCS table category for each user. For further details, refer to the 
        :ref:`mcs_table_cat_note`.

    per_stream : `bool` (default: `False`)
        If `True`, the effective SINR is computed on a per-user and
        per-stream basis and is aggregated across different subcarriers.
        If `False`, the effective SINR is computed on a per-user basis and
        is aggregated across streams and subcarriers.

    kwargs : `dict`
        Additional input parameters

    Output
    ------

    sinr_eff: ([..., num_ut, num_streams_per_ut] | [..., num_ut]), `tf.float`
        Effective SINR in linear scale for each user and associated stream. 
        If ``per_stream`` is `True`, then ``sinr_eff`` has shape `[..., num_ut,
        num_streams_per_rx]`, and ``sinr_eff[..., u, s]`` is the effective SINR
        for stream `s` of user `u` across all subcarriers.
        If ``per_stream`` is `False`, then ``sinr_eff`` has shape `[..., num_ut]`,
        and ``sinr_eff[..., u]`` is the effective SINR for user `u` across
        all streams and subcarriers.
    """

    def calibrate(self):
        r"""
        Optional method for calibrating the Effective SINR model
        """
        pass

    @abstractmethod
    def call(self,
             sinr,
             mcs_index=None,
             mcs_table_index=None,
             mcs_category=None,
             per_stream=False,
             **kwargs):
        pass


class EESM(EffectiveSINR):
    # pylint: disable=line-too-long
    r"""Computes the effective SINR from input SINR values
    across multiple subcarriers and streams via the exponential effective SINR
    mapping (EESM) method 

    Let :math:`\mathrm{SINR}_{u,c,s}>0` be the SINR experienced by user :math:`u`
    on subcarrier :math:`c=1,\dots,C`, and stream :math:`s=1,\dots,S_c`. 
    If ``per_stream`` is `False`, it computes the effective SINR aggregated
    across all utilized streams and subcarriers for each user :math:`u`:

    .. math::
        \mathrm{SINR}^{\mathrm{eff}}_u = -\beta_u \log \left( \frac{1}{CS} 
        \sum_{c=1}^{C} \sum_{s=1}^{S_c} e^{-\frac{\mathrm{SINR}_{u,c,s}}{\beta_u}} \right),
        \quad \forall\, u

    where :math:`\beta>0` is a parameter depending on the Modulation and Coding
    Scheme (MCS) of user :math:`u`.

    If ``per_stream`` is `True`, it computes the effective SINR aggregated
    across subcarriers, for each user :math:`u` and associated stream :math:`s`:

        .. math::
            \mathrm{SINR}^{\mathrm{eff}}_{u,s} = -\beta_u \log \left( \frac{1}{C} 
            \sum_{c=1}^{C} e^{-\frac{\mathrm{SINR}_{u,c,s}}{\beta_u}} \right),
            \quad \forall\, u,s.

    Parameters
    ----------

    load_beta_table_from : `str`
        File name from which the tables containing the values of :math:`\beta`
        parameters are loaded

    sinr_eff_min_db : `float` (default: -30)
        Minimum effective SINR value [dB]. Useful to avoid numerical errors

    sinr_eff_max_db : `float` (default: 50)
        Maximum effective SINR value [dB]. Useful to avoid numerical errors

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----

    sinr : [..., num_ofdm_symbols, num_subcarriers, num_ut, num_streams_per_ut], `tf.float`
        Post-equalization SINR in linear scale for different OFDM symbols,
        subcarriers, users and streams.
        If one entry is zero, the corresponding stream is considered as not
        utilized. 

    mcs_index : [..., num_ut], `tf.int32`
        Modulation and coding scheme (MCS) index for each user 

    mcs_table_index : [..., num_ut], `tf.int32` (default: 1)
        MCS table index for each user. For further details, refer to the 
        :ref:`mcs_table_cat_note`.

    mcs_category : [..., num_ut], `tf.int32` (default: `None`)
        MCS table category for each user. For further details, refer to the 
        :ref:`mcs_table_cat_note`.

    per_stream : `bool` (default: `False`)
        If `True`, then the effective SINR is computed on a per-user and
        per-stream basis and is aggregated across different subcarriers.
        If `False`, then the effective SINR is computed on a per-user basis and
        is aggregated across streams and subcarriers.

    Output
    ------

    sinr_eff: ([..., num_ut, num_streams_per_ut] | [..., num_ut]), `tf.float`
        Effective SINR in linear scale for each user and associated stream. 
        If ``per_stream`` is `True`, then ``sinr_eff`` has shape `[..., num_ut,
        num_streams_per_rx]`, and ``sinr_eff[..., u, s]`` is the effective SINR
        for stream `s` of user `u` across all subcarriers.
        If ``per_stream`` is `False`, then ``sinr_eff`` has shape `[..., num_ut]`,
        and ``sinr_eff[..., u]`` is the effective SINR for user `u` across
        all streams and subcarriers.

    Note
    ----

    If the input SINR is zero for a specific stream, the stream is
    considered unused and does not contribute to the effective SINR computation. 

    Example
    -------
    .. code-block:: Python

        from sionna.phy import config
        from sionna.sys import EESM
        from sionna.phy.utils import db_to_lin

        batch_size = 10
        num_ofdm_symbols = 12
        num_subcarriers = 32
        num_ut = 15
        num_streams_per_ut = 2

        # Generate random MCS indices
        mcs_index = config.tf_rng.uniform([batch_size, num_ut],
                                          minval=0, maxval=27, dtype=tf.int32)

        # Instantiate the EESM object
        eesm = EESM()

        # Generate random SINR values
        sinr_db = config.tf_rng.uniform([batch_size,
                                         num_ofdm_symbols,
                                         num_subcarriers,
                                         num_ut,
                                         num_streams_per_ut],
                                         minval=-5, maxval=30)
        sinr = db_to_lin(sinr_db)

        # Compute the effective SINR for each receiver
        # [batch_size, num_rx]
        sinr_eff = eesm(sinr, mcs_index, mcs_table_index=1, per_stream=False)
        print(sinr_eff.shape)
        # (10, 15)

        # Compute the per-stream effective SINR for each receiver
        # [batch_size, num_rx, num_streams_per_rx]
        sinr_eff_per_stream = eesm(sinr, mcs_index, mcs_table_index=2, per_stream=True)
        print(sinr_eff_per_stream.shape)
        # (10, 15, 2)
    """

    def __init__(self,
                 load_beta_table_from='default',
                 sinr_eff_min_db=-30,
                 sinr_eff_max_db=30,
                 precision=None):

        super().__init__(precision=precision)
        self._sinr_eff_min = db_to_lin(sinr_eff_min_db, precision=precision)
        self._sinr_eff_max = db_to_lin(sinr_eff_max_db, precision=precision)
        self._beta_table = None
        self._beta_tensor = None
        if load_beta_table_from == 'default':
            self.beta_table_filenames = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'esm_params/eesm_beta_table.json')
        else:
            self.beta_table_filenames = load_beta_table_from

    @property
    def beta_table(self):
        r"""
        `dict` (read-only) : Maps MCS indices
         to the corresponding parameters, commonly called :math:`\beta`,
         calibrating the Exponential Effective SINR Map (EESM) method. It has
         the form  ``beta_table['index'][mcs_table_index][mcs]`` 
        """
        return self._beta_table

    @property
    def beta_tensor(self):
        r"""
        [n_tables, n_mcs], `tf.float` (read-only) : Tensor corresponding to
        ``self.beta_table``
        """
        return self._beta_tensor

    @property
    def beta_table_filenames(self):
        r"""
        `str` | list of `str` : Get/set the absolute path name of the JSON
        file containing the mapping between MCS and EESM beta parameters, stored
        in ``beta`` 
        """
        return self._beta_table_filenames

    @beta_table_filenames.setter
    def beta_table_filenames(self, value):
        self._beta_table_filenames = to_list(value)
        # Load the table
        self._beta_table = DeepUpdateDict({})
        for f in self.beta_table_filenames:
            try:
                with open(f, "r", encoding="utf-8") as f:
                    subtable = json.load(f, object_hook=dict_keys_to_int)
                    # Merge with the existing one
                    self._beta_table.deep_update(subtable)
            except FileNotFoundError:
                warnings.warn(f"EESM beta parameters file '{f}' does not exist. "
                              "Skipping...")

        if self._beta_table == {}:
            raise ValueError("No EESM beta parameter table found.")

        # Check table validity
        self.validate_beta_table()

        # Build the corresponding tensor
        table_idx_vec = self._beta_table['index'].keys()
        n_mcs_vec = []
        for table_idx in table_idx_vec:
            n_mcs_vec.append(len(self._beta_table['index'][table_idx]))
        beta_tensor = np.zeros([max(table_idx_vec),
                                max(n_mcs_vec)])
        for table_idx in table_idx_vec:
            mcs_vec = self._beta_table['index'][table_idx]
            beta_tensor[table_idx-1, :len(mcs_vec)] = mcs_vec
        self._beta_tensor = tf.convert_to_tensor(beta_tensor,
                                                 self.rdtype)

    def validate_beta_table(self):
        r"""
        Validates the EESM beta parameter dictionary ``self.beta_table``

        Output
        ------

        : `bool` | `ValueError`
            Returns `True` if ``self.beta_table`` has a valid structure.
            Else, a `ValueError` is raised
        """
        if not isinstance(self.beta_table, dict):
            raise ValueError('Must be a dictionary')
        if not set(self.beta_table.keys()) >= set(['index']):
            raise ValueError("Key must be 'index'")
        for table_index in self.beta_table['index']:
            if not isinstance(self.beta_table['index']
                              [table_index], list):
                raise ValueError("self.beta_table['index']"
                                 f"[{table_index}] must be a list")
        return True

    def call(self,
             sinr,
             mcs_index,
             mcs_table_index=1,
             mcs_category=None,
             per_stream=False,
             **kwargs):

        # Cast and reshape inputs
        sinr = tf.cast(sinr, self.rdtype)
        num_ut = sinr.shape[-2]
        num_batch_dim = len(sinr.shape) - 4
        batch_dim = sinr.shape[:num_batch_dim]
        mcs_index = scalar_to_shaped_tensor(mcs_index,
                                            tf.int32,
                                            batch_dim + [num_ut])
        mcs_table_index = scalar_to_shaped_tensor(mcs_table_index,
                                                  tf.int32,
                                                  batch_dim + [num_ut])

        # Transpose SINR from / to:
        # [..., num_ofdm_symbols, num_subcarriers, num_ut, num_streams_per_ut]
        # [..., num_ut, num_streams_per_ut, num_ofdm_symbols, num_subcarriers]
        sinr = tf.transpose(sinr, list(range(num_batch_dim)) +
                            [num_batch_dim + 2,
                             num_batch_dim + 3,
                             num_batch_dim,
                             num_batch_dim + 1])

        # Axis over which SINR is aggregated
        axis = [-2, -1] if per_stream else [-3, -2, -1]

        # If per_stream is True: n. used subcarriers per stream and rx
        # [..., num_ut, num_streams_per_ut]
        # If per_stream is False: n. used subcarriers and layers per rx
        # [..., num_ut]
        num_used_res = tf.reduce_sum(tf.cast(sinr > 0, self.rdtype),
                                     axis=axis)

        # Ensure MCS is non-negative (for non-scheduled UTs)
        mcs_index = tf.maximum(mcs_index, tf.cast(0, tf.int32))

        # Gather beta
        # Stack indices to extract BLER from interpolated table
        idx = tf.stack([mcs_table_index - 1, mcs_index], axis=-1)
        # [..., num_ut]
        beta = gather_from_batched_indices(self.beta_tensor, idx)
        beta = tf.cast(beta, sinr.dtype)

        # [..., num_ut, 1, 1, 1]
        beta_expand1 = expand_to_rank(beta, tf.rank(sinr), axis=-1)

        # Exponentiate SINR
        # [..., num_ut, num_streams_per_ut, num_ofdm_symbols, num_subcarriers]
        sinr_exp = tf.where(sinr > 0, tf.math.exp(-sinr / beta_expand1), 0)

        # Log + average across resources
        sinr_eff = tf.math.log(tf.reduce_sum(
            sinr_exp, axis=axis) / num_used_res)
        beta_expand2 = expand_to_rank(beta, tf.rank(sinr_eff), axis=-1)
        # If per_stream is True: [..., num_ut, num_streams_per_ut]
        # If per_stream is False: [..., num_ut]
        sinr_eff = - beta_expand2 * sinr_eff

        # Assign a null SINR to users with no assigned resources
        sinr_eff = tf.where(num_used_res > 0, sinr_eff, 0)

        # Project sinr_eff within [self._sinr_eff_min, self._sinr_eff_max]
        sinr_eff = tf.where(sinr_eff < self._sinr_eff_min,
                            self._sinr_eff_min,
                            sinr_eff)
        sinr_eff = tf.where(sinr_eff > self._sinr_eff_max,
                            self._sinr_eff_max,
                            sinr_eff)

        return sinr_eff
