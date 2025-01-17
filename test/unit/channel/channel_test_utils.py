#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
from re import L
import tensorflow as tf
import numpy as np
from functools import wraps
from sionna.channel.utils import sample_bernoulli
from sionna import config

####################################################################
# Utility functions
####################################################################

def generate_random_loc(bs, n, x_range, y_range, z_range, share_loc=False,
                        dtype=tf.float32):
    r"""
    Generate random locations.

    Input
    ------
    bs : int
        Batch size

    n : int
        Number of locations per batch example

    x_range : (float, float)
        Pair of floats, giving the min and max values for the X component

    y_range : (float, float)
        Pair of floats, giving the min and max values for the Y component

    z_range : (float, float)
        Pair of floats, giving the min and max values for the Z component

    share_loc : bool
        If set to `True`, then all batch examples share the same locations.
        Defaults to `False`.

    dtype : tf.dtype
        Real datatype. Defaults to tf.float32.

    Output
    -------
    : [bs, n, 3], tf.float
        Tensor of coordinates uniformly distributed in ``x_range``, ``y_range``,
        and ``z_range``.
    """

    if share_loc:
        loc_x = config.tf_rng.uniform([1, n], x_range[0], x_range[1], dtype)
        loc_y = config.tf_rng.uniform([1, n], y_range[0], y_range[1], dtype)
        loc_z = config.tf_rng.uniform([1, n], z_range[0], z_range[1], dtype)
        loc = tf.stack([loc_x, loc_y, loc_z], axis=2)
        loc = tf.tile(loc, [bs, 1, 1])
    else:
        loc_x = config.tf_rng.uniform([bs, n], x_range[0], x_range[1], dtype)
        loc_y = config.tf_rng.uniform([bs, n], y_range[0], y_range[1], dtype)
        loc_z = config.tf_rng.uniform([bs, n], z_range[0], z_range[1], dtype)
        loc = tf.stack([loc_x, loc_y, loc_z], axis=2)

    return loc

def generate_random_bool(bs, n, p, share_state=False):
    r"""
    Sample tensor of random boolean values, following a Bernoulli distribution
    with probability ``p``.

    Input
    ------
    bs : int
        Batch size

    n : int
        Number of bools per batch example

    p : float
        Probability of a randomly generated boolean to be `True`.

    share_state : bool
        If set to `True`, then all batch examples share the same bool array.
        Defaults to `False`.

    Output
    -------
    : [bs, n], tf.bool
        Tensor of boolean following a Bernoulli probability ``p``.
    """

    if share_state:
        bool_tensor = sample_bernoulli([1, n], p, tf.float32)
        bool_tensor = tf.tile(bool_tensor, [bs, 1])
    else:
        bool_tensor = sample_bernoulli([bs, n], p, tf.float32)
    bool_tensor = tf.cast(bool_tensor, tf.bool)
    return bool_tensor

#########################################################################
# Decorator for making testing all models easier
#########################################################################

def channel_test_on_models(models, submodels):
    def channel_test_on_models_decorator(func):
        @wraps(func)
        def wrapped_function(self, *args, **kwargs):
            for model in models:
                for submodel in submodels:
                    func(self, model, submodel, *args, **kwargs)
        return wrapped_function
    return channel_test_on_models_decorator

#########################################################################
# Channel parameters, extracted from 3GPP TR38.901 specification
#########################################################################


######## Mean and standard deviation

def log10DS(model, submodel, fc):
    r"""
    Return the mean and standard deviation of log10(DS) [s]

    Input:
    ------
    model : str
        One of 'rma', 'umi', or 'uma'

    submodel : str
        One of 'los', 'nlos', 'o2i'

    fc : float
        Carrier frequency [Hz]

    Output
    ------
    log10_mu : float
        Mean in the log10 domain

    log10_std : float
        Standard deviation in the log10 domain
    """

    fc = fc/1e9
    if model == 'rma':
        if submodel == 'los' : return (-7.49, 0.55)
        elif submodel == 'nlos' : return (-7.43, 0.48)
        elif submodel == 'o2i' : return (-7.47, 0.24)
    elif model == 'umi':
        if fc < 2. : fc = 2.
        if submodel == 'los' : return (-0.24*np.log10(1.+fc) - 7.14, 0.38)
        elif submodel == 'nlos' : return (-0.24*np.log10(1+ fc) - 6.83,
                                          0.16*np.log10(1.+fc) + 0.28)
        elif submodel == 'o2i' : return (-6.62, 0.32)
    elif model == 'uma':
        if fc < 6. : fc = 6.
        if submodel == 'los' : return (-6.955 - 0.0963*np.log10(fc), 0.66)
        elif submodel == 'nlos' : return (-6.28 - 0.204*np.log10(fc), 0.39)
        elif submodel == 'o2i' : return (-6.62, 0.32)

def log10ASD(model, submodel, fc):
    r"""
    Return the mean and standard deviation of log10(ASD)[deg]

    Input:
    ------
    model : str
        One of 'rma', 'umi', or 'uma'

    submodel : str
        One of 'los', 'nlos', 'o2i'

    fc : float
        Carrier frequency [Hz]

    Output
    ------
    log10_mu : float
        Mean in the log10 domain

    log10_std : float
        Standard deviation in the log10 domain
    """

    fc = fc/1e9
    if model == 'rma':
        if submodel == 'los' : return (0.90, 0.38)
        elif submodel == 'nlos' : return (0.95, 0.45)
        elif submodel == 'o2i' : return (0.67, 0.18)
    elif model == 'umi':
        if fc < 2. : fc = 2.
        if submodel == 'los' : return (-0.05*np.log10(1.+fc) + 1.21, 0.41)
        elif submodel == 'nlos' : return (-0.23*np.log10(1+ fc) + 1.53,
                                          0.11*np.log10(1.+fc) + 0.33)
        elif submodel == 'o2i' : return (1.25, 0.42)
    elif model == 'uma':
        if fc < 6. : fc = 6.
        if submodel == 'los' : return (1.06 + 0.1114*np.log10(fc), 0.28)
        elif submodel == 'nlos' : return (1.5 - 0.1144*np.log10(fc), 0.28)
        elif submodel == 'o2i' : return (1.25, 0.42)

def log10ASA(model, submodel, fc):
    r"""
    Return the mean and standard deviation of log10(ASA) [deg]

    Input:
    ------
    model : str
        One of 'rma', 'umi', or 'uma'

    submodel : str
        One of 'los', 'nlos', 'o2i'

    fc : float
        Carrier frequency [Hz]

    Output
    ------
    log10_mu : float
        Mean in the log10 domain

    log10_std : float
        Standard deviation in the log10 domain
    """

    fc = fc/1e9
    if model == 'rma':
        if submodel == 'los' : return (1.52, 0.24)
        elif submodel == 'nlos' : return (1.52, 0.13)
        elif submodel == 'o2i' : return (1.66, 0.21)
    elif model == 'umi':
        if fc < 2. : fc = 2.
        if submodel == 'los' : return (-0.08*np.log10(1+fc) + 1.73,
                                       0.014*np.log10(1+fc) + 0.28)
        elif submodel == 'nlos' : return (-0.08*np.log10(1+fc) + 1.81,
                                          0.05*np.log10(1+fc) + 0.3)
        elif submodel == 'o2i' : return (1.76, 0.16)
    elif model == 'uma':
        if fc < 6. : fc = 6.
        if submodel == 'los' : return (1.81, 0.20)
        elif submodel == 'nlos' : return (2.08 - 0.27*np.log10(fc), 0.11)
        elif submodel == 'o2i' : return (1.76, 0.16)

def log10ZSA(model, submodel, fc):
    r"""
    Return the mean and standard deviation of log10(ZSA) [deg]

    Input:
    ------
    model : str
        One of 'rma', 'umi', or 'uma'

    submodel : str
        One of 'los', 'nlos', 'o2i'

    fc : float
        Carrier frequency [Hz]

    Output
    ------
    log10_mu : float
        Mean in the log10 domain

    log10_std : float
        Standard deviation in the log10 domain
    """

    fc = fc/1e9
    if model == 'rma':
        if submodel == 'los' : return (0.47, 0.40)
        elif submodel == 'nlos' : return (0.58, 0.37)
        elif submodel == 'o2i' : return (0.93, 0.22)
    elif model == 'umi':
        if fc < 2. : fc = 2.
        if submodel == 'los' : return (-0.1*np.log10(1+fc) + 0.73,
                                       -0.04*np.log10(1+fc) + 0.34)
        elif submodel == 'nlos' : return (-0.04*np.log10(1+fc) + 0.92,
                                          -0.07*np.log10(1+fc) + 0.41)
        elif submodel == 'o2i' : return (1.01, 0.43)
    elif model == 'uma':
        if fc < 6. : fc = 6.
        if submodel == 'los' : return (0.95, 0.16)
        elif submodel == 'nlos' : return (-0.3236*np.log10(fc) + 1.512, 0.16)
        elif submodel == 'o2i' : return (1.01, 0.43)

def log10SF_dB(model, submodel, d_2d, fc, h_bs, h_ut):
    r"""
    Return the mean and standard deviation of log10(SF) [dB]

    Input:
    ------
    model : str
        One of 'rma', 'umi', or 'uma'

    submodel : str
        One of 'los', 'nlos', 'o2i'

    d_2d: float
        2D distance [m]

    fc : float
        Carrier frequency [Hz]

    h_bs: float
        BS height [m]

    h_ut : float
        UT height [m]

    Output
    ------
    log10_mu : float
        Mean in the log10 domain

    log10_std : float
        Standard deviation in the log10 domain
    """

    if model == 'rma':
        d_bp = 2.*np.pi*fc/3e8*h_bs*h_ut
        if submodel == 'los' :
            if d_2d < d_bp:
                return (0.0, 4.0)
            else:
                return (0.0, 6.0)
        elif submodel == 'nlos' : return (0.0, 8.0)
        elif submodel == 'o2i' : return (0.0, 8.0)
    elif model == 'umi':
        if submodel == 'los' : return (0.0, 4.0)
        elif submodel == 'nlos' : return (0.0, 7.82)
        elif submodel == 'o2i' : return (0.0, 7.0)
    elif model == 'uma':
        if submodel == 'los' : return (0.0, 4.0)
        elif submodel == 'nlos' : return (0.0, 6.0)
        elif submodel == 'o2i' : return (0.0, 7.0)

def log10K_dB(model, submodel):
    r"""
    Return the mean and standard deviation of log10(K) [dB]

    Input:
    ------
    model : str
        One of 'rma', 'umi', or 'uma'

    submodel : str
        One of 'los', 'nlos', 'o2i'

    Output
    ------
    log10_mu : float
        Mean in the log10 domain

    log10_std : float
        Standard deviation in the log10 domain
    """

    if model == 'rma':
        if submodel == 'los' : return (7.0, 4.0)
        elif submodel == 'nlos' : return None
        elif submodel == 'o2i' : return None
    elif model == 'umi':
        if submodel == 'los' : return (9.0, 5.0)
        elif submodel == 'nlos' : return None
        elif submodel == 'o2i' : return None
    elif model == 'uma':
        if submodel == 'los' : return (9.0, 3.5)
        elif submodel == 'nlos' : return None
        elif submodel == 'o2i' : return None

def log10ZSD(model, submodel, d_2d, fc, h_bs, h_ut):
    r"""
    Return the mean and standard deviation of log10(ZSD) [deg]

    Input:
    ------
    model : str
        One of 'rma', 'umi', or 'uma'

    submodel : str
        One of 'los', 'nlos', 'o2i'

    d_2d: float
        2D distance [m]

    fc : float
        Carrier frequency [Hz]

    h_bs: float
        BS height [m]

    h_ut : float
        UT height [m]

    Output
    ------
    log10_mu : float
        Mean in the log10 domain

    log10_std : float
        Standard deviation in the log10 domain
    """

    fc = fc / 1e9
    if model == 'rma':
        if submodel == 'los' : return (np.maximum(-1.0,
                                    -0.17*d_2d/1000.-0.01*(h_ut-1.5)+0.22),
                                       0.34)
        elif submodel == 'nlos' : return (np.maximum(-1.0,
                                    -0.19*d_2d/1000.-0.01*(h_ut-1.5)+0.28),
                                       0.30)
        elif submodel == 'o2i' : return (np.maximum(-1.0,
                                    -0.19*d_2d/1000.-0.01*(h_ut-1.5)+0.28),
                                       0.30)
    elif model == 'umi':
        if submodel == 'los' : return (np.maximum(-0.21,
                                -14.8*d_2d/1000.+0.01*np.abs(h_ut-h_bs)+0.83),
                                       0.35)
        elif submodel == 'nlos' : return (np.maximum(-0.5,
                        -3.1*d_2d/1000.+0.01*np.maximum(h_ut-h_bs, 0.0)+0.2),
                                       0.35)
        elif submodel == 'o2i' : return (np.maximum(-0.5,
                        -3.1*d_2d/1000.+0.01*np.maximum(h_ut-h_bs, 0.0)+0.2),
                                       0.35)
    elif model == 'uma':
        if fc < 6. : fc = 6.
        if submodel == 'los' : return (np.maximum(-0.5,
                        -2.1*d_2d/1000.-0.01*(h_ut-1.5)+0.75),
                                       0.40)
        elif submodel == 'nlos' : return (np.maximum(-0.5,
                        -2.1*d_2d/1000.-0.01*(h_ut-1.5)+0.9),
                                       0.49)
        elif submodel == 'o2i' : return (np.maximum(-0.5,
                        -2.1*d_2d/1000.-0.01*(h_ut-1.5)+0.9),
                                       0.49)


######## LSP cross-correlations
# Order: DS, ASD, ASA, SF, (K if LoS), ZSA, ZSD

def cross_corr(model, submodel):
    r"""
    Return the LSP cross-correlation matrix for the model ``model`` and the
    submodel ``submodel``.

    Input
    ------
    model : str
        One of 'rma', 'umi', or 'uma'

    submodel : str
        One of 'los', 'nlos', 'o2i'

    Output
    -------
    : [7,7] or [6,6], float
        LSP cross-correlation  matrix of size 7x7 for LoS, and 6x6 for NLoS and
        O2I as K-factor is not required.
    """
    if model == 'rma':
        if submodel == 'los':
            return np.array([ [1.0, 0.0, 0.0, -0.5, 0.0, 0.27, -0.05],
                            [0.0, 1.0, 0.0, 0.0, 0.0, -0.14, 0.73],
                            [0.0, 0.0, 1.0, 0.0, 0.0, 0.24, -0.20],
                            [-0.5, 0.0, 0.0, 1.0, 0.0, -0.17, 0.01],
                            [0.0, 0.0, 0.0, 0.0, 1.0, -0.02, 0.0],
                            [0.27, -0.14, 0.24, -0.17, -0.02, 1.0, -0.07],
                            [-0.05, 0.73, -0.20, 0.01, 0.0, -0.07, 1.0]])
        elif submodel == 'nlos':
            return np.array([[1.0, -0.4, 0.0, -0.5, -0.4, -0.10],
                            [-0.4, 1.0, 0.0, 0.6, -0.27, 0.42],
                            [0.0, 0.0, 1.0, 0.0, 0.26, -0.18],
                            [-0.5, 0.6, 0.0, 1.0, -0.25, -0.04],
                            [-0.4, -0.27, 0.26, -0.25, 1.0, -0.27],
                            [-0.10, 0.42, -0.18, -0.04, -0.27, 1.0]])
        elif submodel == 'o2i':
            return np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, -0.7, 0.0, 0.47, 0.66],
                             [0.0, -0.7, 1.0, 0.0, -0.22, -0.55],
                             [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.47, -0.22, 0.0, 1.0, 0.0],
                             [0.0, 0.66, -0.55, 0.0, 0.0, 1.0]])
    elif model == 'umi':
        if submodel == 'los':
            return np.array([[1.0, 0.5, 0.8, -0.4, -0.7, 0.2, 0.0],
                            [0.5, 1.0, 0.4, -0.5, -0.2, 0.3, 0.5],
                            [0.8, 0.4, 1.0, -0.4, -0.3, 0.0, 0.0],
                            [-0.4, -0.5, -0.4, 1.0, 0.5, 0.0, 0.0],
                            [-0.7, -0.2, -0.3, 0.5, 1.0, 0.0, 0.0],
                            [0.2, 0.3, 0.0, 0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0]])
        elif submodel == 'nlos':
            return np.array([[1.0, 0.0, 0.4, -0.7, 0.0, -0.5],
                             [0.0, 1.0, 0.0, 0.0, 0.5, 0.5],
                             [0.4, 0.0, 1.0, -0.4, 0.2, 0.0],
                             [-0.7, 0.0, -0.4, 1.0, 0.0, 0.0],
                             [0.0, 0.5, 0.2, 0.0, 1.0, 0.0],
                             [-0.5, 0.5, 0.0, 0.0, 0.0, 1.0]])
        elif submodel == 'o2i':
            return np.array([[1.0, 0.4, 0.4, -0.5, -0.2, -0.6],
                            [0.4, 1.0, 0.0, 0.2, 0.0, -0.2],
                            [0.4, 0.0, 1.0, 0.0, 0.5, 0.0],
                            [-0.5, 0.2, 0.0, 1.0, 0.0, 0.0],
                            [-0.2, 0.0, 0.5, 0.0, 1.0, 0.5],
                            [-0.6, -0.2, 0.0, 0.0, 0.5, 1.0]])
    elif model == 'uma':
        if submodel == 'los':
            return np.array([[1.0, 0.4, 0.8, -0.4, -0.4, 0.0, -0.2],
                            [0.4, 1.0, 0.0, -0.5, 0.0, 0.0, 0.5],
                            [0.8, 0.0, 1.0, -0.5, -0.2, 0.4, -0.3],
                            [-0.4, -0.5, -0.5, 1.0, 0.0, -0.8, 0.0],
                            [-0.4, 0.0, -0.2, 0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.4, -0.8, 0.0, 1.0, 0.0],
                            [-0.2, 0.5, -0.3, 0.0, 0.0, 0.0, 1.0]])
        elif submodel == 'nlos':
            return np.array([[1.0, 0.4, 0.6, -0.4, 0.0, -0.5],
                            [0.4, 1.0, 0.4, -0.6, -0.1, 0.5],
                            [0.6, 0.4, 1.0, 0.0, 0.0, 0.0],
                            [-0.4, -0.6, 0.0, 1.0, -0.4, 0.0],
                            [0.0, -0.1, 0.0, -0.4, 1.0, 0.0],
                            [-0.5, 0.5, 0.0, 0.0, 0.0, 1.0]])
        elif submodel == 'o2i':
            return np.array([[1.0, 0.4, 0.4, -0.5, -0.2, -0.6],
                            [0.4, 1.0, 0.0, 0.2, 0.0, -0.2],
                            [0.4, 0.0, 1.0, 0.0, 0.5, 0.0],
                            [-0.5, 0.2, 0.0, 1.0, 0.0, 0.0],
                            [-0.2, 0.0, 0.5, 0.0, 1.0, 0.5],
                            [-0.6, -0.2, 0.0, 0.0, 0.5, 1.0]])

######## LSP spatial correlations
# Order: DS, ASD, ASA, SF, (K if LoS), ZSA, ZSD

def corr_dist_ds(model, submodel):
    r"""
    Return the correlation distance of LSP DS for ``model`` and ``submodel``.

    Input
    ------
    model : str
        One of 'rma', 'umi', or 'uma'

    submodel : str
        One of 'los', 'nlos', 'o2i'

    Output
    -------
    : float
        Correlation distance
    """
    if model == 'umi':
        if submodel == 'los' : return 7
        elif submodel == 'nlos' : return 10
        elif submodel == 'o2i' : return 10
    elif model == 'uma':
        if submodel == 'los' : return 30
        elif submodel == 'nlos' : return 40
        elif submodel == 'o2i' : return 10
    elif model == 'rma':
        if submodel == 'los' : return 50
        elif submodel == 'nlos' : return 36
        elif submodel == 'o2i' : return 36

def corr_dist_asd(model, submodel):
    r"""
    Return the correlation distance of LSP ASD for ``model`` and ``submodel``.

    Input
    ------
    model : str
        One of 'rma', 'umi', or 'uma'

    submodel : str
        One of 'los', 'nlos', 'o2i'

    Output
    -------
    : float
        Correlation distance
    """
    if model == 'umi':
        if submodel == 'los' : return 8
        elif submodel == 'nlos' : return 10
        elif submodel == 'o2i' : return 11
    elif model == 'uma':
        if submodel == 'los' : return 18
        elif submodel == 'nlos' : return 50
        elif submodel == 'o2i' : return 11
    elif model == 'rma':
        if submodel == 'los' : return 25
        elif submodel == 'nlos' : return 30
        elif submodel == 'o2i' : return 30

def corr_dist_asa(model, submodel):
    r"""
    Return the correlation distance of LSP ASA for ``model`` and ``submodel``.

    Input
    ------
    model : str
        One of 'rma', 'umi', or 'uma'

    submodel : str
        One of 'los', 'nlos', 'o2i'

    Output
    -------
    : float
        Correlation distance
    """
    if model == 'umi':
        if submodel == 'los' : return 8
        elif submodel == 'nlos' : return 9
        elif submodel == 'o2i' : return 17
    elif model == 'uma':
        if submodel == 'los' : return 15
        elif submodel == 'nlos' : return 50
        elif submodel == 'o2i' : return 11
    elif model == 'rma':
        if submodel == 'los' : return 35
        elif submodel == 'nlos' : return 40
        elif submodel == 'o2i' : return 40

def corr_dist_sf(model, submodel):
    r"""
    Return the correlation distance of LSP SF for ``model`` and ``submodel``.

    Input
    ------
    model : str
        One of 'rma', 'umi', or 'uma'

    submodel : str
        One of 'los', 'nlos', 'o2i'

    Output
    -------
    : float
        Correlation distance
    """
    if model == 'umi':
        if submodel == 'los' : return 10
        elif submodel == 'nlos' : return 13
        elif submodel == 'o2i' : return 7
    elif model == 'uma':
        if submodel == 'los' : return 37
        elif submodel == 'nlos' : return 50
        elif submodel == 'o2i' : return 7
    elif model == 'rma':
        if submodel == 'los' : return 37
        elif submodel == 'nlos' : return 120
        elif submodel == 'o2i' : return 120

def corr_dist_k(model, submodel):
    r"""
    Return the correlation distance of LSP K for ``model`` and ``submodel``.

    Input
    ------
    model : str
        One of 'rma', 'umi', or 'uma'

    submodel : str
        One of 'los', 'nlos', 'o2i'

    Output
    -------
    : float
        Correlation distance
    """
    if model == 'umi':
        if submodel == 'los' : return 15
    elif model == 'uma':
        if submodel == 'los' : return 12
    elif model == 'rma':
        if submodel == 'los' : return 40

def corr_dist_zsa(model, submodel):
    r"""
    Return the correlation distance of LSP ZSA for ``model`` and ``submodel``.

    Input
    ------
    model : str
        One of 'rma', 'umi', or 'uma'

    submodel : str
        One of 'los', 'nlos', 'o2i'

    Output
    -------
    : float
        Correlation distance
    """
    if model == 'umi':
        if submodel == 'los' : return 12
        elif submodel == 'nlos' : return 10
        elif submodel == 'o2i' : return 25
    elif model == 'uma':
        if submodel == 'los' : return 15
        elif submodel == 'nlos' : return 50
        elif submodel == 'o2i' : return 25
    elif model == 'rma':
        if submodel == 'los' : return 15
        elif submodel == 'nlos' : return 50
        elif submodel == 'o2i' : return 50

def corr_dist_zsd(model, submodel):
    r"""
    Return the correlation distance of LSP ZSD for ``model`` and ``submodel``.

    Input
    ------
    model : str
        One of 'rma', 'umi', or 'uma'

    submodel : str
        One of 'los', 'nlos', 'o2i'

    Output
    -------
    : float
        Correlation distance
    """
    if model == 'umi':
        if submodel == 'los' : return 12
        elif submodel == 'nlos' : return 10
        elif submodel == 'o2i' : return 25
    elif model == 'uma':
        if submodel == 'los' : return 15
        elif submodel == 'nlos' : return 50
        elif submodel == 'o2i' : return 25
    elif model == 'rma':
        if submodel == 'los' : return 15
        elif submodel == 'nlos' : return 50
        elif submodel == 'o2i' : return 50

####### Pathlosses

def rma_los_pathloss(d_2d, d_3d, fc, h_bs, h_ut, h, w):
    r"""
    Return the LoS pathloss for RMa.

    Input
    ------
    d_2d : float
        2D distance [m]

    d_3d : float
        3D distance [m]

    fc : float
        Frequency carruer [Hz]

    h_bs : float
        BS height [m]

    h_ut : float
        UT height [m]

    h : float
        Average building height

    w : float
        Average street width

    Output
    -------
    : float
        Pathloss [dB]
    """
    dbp = 2*np.pi*h_bs*h_ut*fc/299792458.
    def pl1(d_):
        p = (20.0*np.log10(40*np.pi*d_*fc/3e9)
               + np.minimum(0.03*np.power(h,1.72),10.0)*np.log10(d_)
               - np.minimum(0.044*np.power(h,1.72),14.77)+0.002*np.log10(h)*d_)
        return p
    if d_2d < dbp:
        return pl1(d_3d)
    return pl1(dbp) + 40.0*np.log10(d_3d/dbp)

def rma_nlos_pathloss(d_2d, d_3d, fc, h_bs, h_ut, h, w):
    r"""
    Return the NLoS pathloss for RMa.

    Input
    ------
    d_2d : float
        2D distance [m]

    d_3d : float
        3D distance [m]

    fc : float
        Frequency carruer [Hz]

    h_bs : float
        BS height [m]

    h_ut : float
        UT height [m]

    h : float
        Average building height

    w : float
        Average street width

    Output
    -------
    : float
        Pathloss [dB]
    """
    pl3 = (161.04 - 7.1*np.log10(w) + 7.5*np.log10(h)
           - (24.37-3.5*np.square(h/h_bs))*np.log10(h_bs)
              +(43.42-3.1*np.log10(h_bs))*(np.log10(d_3d)-3)
              +20*np.log10(fc/1e9)-(3.2*np.square(np.log10(11.75*h_ut))-4.97))
    return np.maximum(rma_los_pathloss(d_2d, d_3d, fc, h_bs, h_ut, h, w), pl3)

def rma_o2i_pathloss(d_2d, d_3d, fc, h_bs, h_ut, h, w):
    r"""
    Return the O2I average pathloss for RMa.

    Input
    ------
    d_2d : float
        2D distance [m]

    d_3d : float
        3D distance [m]

    fc : float
        Frequency carruer [Hz]

    h_bs : float
        BS height [m]

    h_ut : float
        UT height [m]

    h : float
        Average building height

    w : float
        Average street width

    Output
    -------
    : float
        Pathloss [dB]
    """
    pltw = 5.0-10.0*np.log10(0.3*np.power(10.0, (-2.-0.2*fc/1e9)/10.0)\
        +0.7*np.power(10.0, (-5.-4.*fc/1e9)/10.0))
    return rma_nlos_pathloss(d_2d, d_3d, fc, h_bs, h_ut, h, w) + pltw + 0.5*5.0

def umi_los_pathloss(d_2d, d_3d, fc, h_bs, h_ut):
    r"""
    Return the LoS pathloss for UMi.

    Input
    ------
    d_2d : float
        2D distance [m]

    d_3d : float
        3D distance [m]

    fc : float
        Frequency carruer [Hz]

    h_bs : float
        BS height [m]

    h_ut : float
        UT height [m]

    Output
    -------
    : float
        Pathloss [dB]
    """
    dbp = 4*(h_bs-1.0)*(h_ut-1.0)*fc/299792458.

    pl1 = 32.4 + 21.0*np.log10(d_3d) + 20.0*np.log10(fc/1e9)
    pl2 = 32.4 + 40.0*np.log10(d_3d) + 20.0*np.log10(fc/1e9)\
        - 9.5*np.log10(np.square(dbp) + np.square(h_bs - h_ut))
    pl_los = np.where(np.less(d_2d, dbp), pl1, pl2)
    return pl_los

def umi_nlos_pathloss(d_2d, d_3d, fc, h_bs, h_ut):
    r"""
    Return the NLoS pathloss for UMi.

    Input
    ------
    d_2d : float
        2D distance [m]

    d_3d : float
        3D distance [m]

    fc : float
        Frequency carruer [Hz]

    h_bs : float
        BS height [m]

    h_ut : float
        UT height [m]

    Output
    -------
    : float
        Pathloss [dB]
    """
    pl3 = 35.3*np.log10(d_3d) + 22.4 + 21.3*np.log10(fc/1e9) - 0.3*(h_ut-1.5)
    return np.maximum(umi_los_pathloss(d_2d, d_3d, fc, h_bs, h_ut), pl3)

def umi_o2i_pathloss(d_2d, d_3d, fc, h_bs, h_ut, o2i_model):
    r"""
    Return the O2I average pathloss for UMi.

    Input
    ------
    d_2d : float
        2D distance [m]

    d_3d : float
        3D distance [m]

    fc : float
        Frequency carruer [Hz]

    h_bs : float
        BS height [m]

    h_ut : float
        UT height [m]

    o2i_model : str
        O2I loss model. Must be 'low' or 'high'

    Output
    -------
    : float
        Pathloss [dB]
    """
    if o2i_model == 'low':
        pltw = 5.0-10.0*np.log10(0.3*np.power(10.0, (-2.-0.2*fc/1e9)/10.0)\
            + 0.7*np.power(10.0, (-5.-4.*fc/1e9)/10.0))
        return umi_nlos_pathloss(d_2d, d_3d, fc, h_bs, h_ut) + pltw + 0.5*12.5
    else:
        pltw = 5.0-10.0*np.log10(0.7*np.power(10.0, (-23.-0.3*fc/1e9)/10.0)\
            + 0.3*np.power(10.0, (-5.-4.*fc/1e9)/10.0))
        return umi_nlos_pathloss(d_2d, d_3d, fc, h_bs, h_ut) + pltw + 0.5*12.5

def uma_los_pathloss(d_2d, d_3d, fc, h_bs, h_ut):
    r"""
    Return the LoS average pathloss for UMa.

    Input
    ------
    d_2d : float
        2D distance [m]

    d_3d : float
        3D distance [m]

    fc : float
        Frequency carruer [Hz]

    h_bs : float
        BS height [m]

    h_ut : float
        UT height [m]

    Output
    -------
    : float
        Pathloss [dB]
    """
    dbp = 4*(h_bs-1.0)*(h_ut-1.0)*fc/299792458.

    pl1 = 28. + 22.0*np.log10(d_3d) + 20.0*np.log10(fc/1e9)
    pl2 = 28 + 40.0*np.log10(d_3d) + 20.0*np.log10(fc/1e9) -\
        9*np.log10(np.square(dbp) + np.square(h_bs - h_ut))
    pl_los = np.where(np.less(d_2d, dbp), pl1, pl2)
    return pl_los

def uma_nlos_pathloss(d_2d, d_3d, fc, h_bs, h_ut):
    r"""
    Return the NLoS average pathloss for UMa.

    Input
    ------
    d_2d : float
        2D distance [m]

    d_3d : float
        3D distance [m]

    fc : float
        Frequency carruer [Hz]

    h_bs : float
        BS height [m]

    h_ut : float
        UT height [m]

    o2i_model : str
        O2I loss model. Must be 'low' or 'high'

    Output
    -------
    : float
        Pathloss [dB]
    """
    pl3 = 13.54 + 39.08*np.log10(d_3d) + 20*np.log10(fc/1e9) - 0.6*(h_ut-1.5)
    return np.maximum(uma_los_pathloss(d_2d, d_3d, fc, h_bs, h_ut), pl3)

def uma_o2i_pathloss(d_2d, d_3d, fc, h_bs, h_ut, o2i_model):
    r"""
    Return the O2I average pathloss for UMa.

    Input
    ------
    d_2d : float
        2D distance [m]

    d_3d : float
        3D distance [m]

    fc : float
        Frequency carruer [Hz]

    h_bs : float
        BS height [m]

    h_ut : float
        UT height [m]

    o2i_model : str
        O2I loss model. Must be 'low' or 'high'

    Output
    -------
    : float
        Pathloss [dB]
    """
    if o2i_model == 'low':
        pltw = 5.0-10.0*np.log10(0.3*np.power(10.0, (-2.-0.2*fc/1e9)/10.0)\
            + 0.7*np.power(10.0, (-5.-4.*fc/1e9)/10.0))
        return uma_nlos_pathloss(d_2d, d_3d, fc, h_bs, h_ut) + pltw + 0.5*12.5
    else:
        pltw = 5.0-10.0*np.log10(0.7*np.power(10.0, (-23.-0.3*fc/1e9)/10.0)\
            + 0.3*np.power(10.0, (-5.-4.*fc/1e9)/10.0))
        return uma_nlos_pathloss(d_2d, d_3d, fc, h_bs, h_ut) + pltw + 0.5*12.5

def pathloss(model, submodel, *args):
    r"""
    Convenience function for computing the pathloss from the model and submodel
    name

    Input
    ------
    model : str
        One of 'rma', 'umi', or 'uma'

    submodel : str
        One of 'los', 'nlos', 'o2i'

    args : List of parameters
        Additional list of parameters for computing the pathloss. Model
        specific.

    Output
    -------
    : float
        Pathloss [dB]
    """
    if model == 'rma':
        if submodel == 'los':
            return rma_los_pathloss(*args)
        elif submodel == 'nlos':
            return rma_nlos_pathloss(*args)
        elif submodel == 'o2i':
            return rma_o2i_pathloss(*args)
    elif model == 'umi':
        if submodel == 'los':
            return umi_los_pathloss(*args)
        elif submodel == 'nlos':
            return umi_nlos_pathloss(*args)
        elif submodel == 'o2i':
            return umi_o2i_pathloss(*args)
    elif model == 'uma':
        if submodel == 'los':
            return uma_los_pathloss(*args)
        elif submodel == 'nlos':
            return uma_nlos_pathloss(*args)
        elif submodel == 'o2i':
            return uma_o2i_pathloss(*args)

def pathloss_std(model, submodel, o2i_model=None):
    r"""
    Pathloss standard deviation [dB]

    Input
    ------
    model : str
        One of 'rma', 'umi', or 'uma'

    submodel : str
        One of 'los', 'nlos', 'o2i'

    o2i_model : str
        O2I loss model. One of 'low' or 'high'. Only required for O2I with UMi
        and UMa.
        Let to `None` if not required. Defaults to `None`.

    Output
    -------
    : float
        Pathloss standard deviation [dB]
    """
    if model == 'rma':
        if submodel == 'los':
            return 0.0
        elif submodel == 'nlos':
            return 0.0
        elif submodel == 'o2i':
            return np.sqrt((4.4**2)+0.25/12*(10**2))
    elif model == 'umi':
        if submodel == 'los':
            return 0.0
        elif submodel == 'nlos':
            return 0.0
        elif submodel == 'o2i':
            if o2i_model == 'low':
                return np.sqrt((4.4**2)+0.25/12*(25**2))
            elif o2i_model == 'high':
                return np.sqrt((6.5**2)+0.25/12*(25**2))
    elif model == 'uma':
        if submodel == 'los':
            return 0.0
        elif submodel == 'nlos':
            return 0.0
        elif submodel == 'o2i':
            if o2i_model == 'low':
                return np.sqrt((4.4**2)+0.25/12*(25**2))
            elif o2i_model == 'high':
                return np.sqrt((6.5**2)+0.25/12*(25**2))

############# ZoD offset

def zod_offset(model, submodel, fc, d_2d, h_ut):
    """
    Return the ZOD offset

    Input
    ------
    model : str
        One of 'rma', 'umi', or 'uma'

    submodel : str
        One of 'los', 'nlos', 'o2i'

    fc : float
        Carrier frequency [Hz]

    d_2d : any shape, float
        2D distances [m]

    h_ut : float
        UT height [m]

    Output
    -------
    : same shape as d_2d
        ZOD offsets
    """
    if model == 'umi':
        if submodel == 'los':
            return 0.0
        elif submodel == 'nlos':
            return -np.power(10, -1.5*np.log10(np.maximum(10, d_2d))+3.3)
        elif submodel == 'o2i':
            return -np.power(10, -1.5*np.log10(np.maximum(10, d_2d))+3.3)
    elif model == 'uma':
        fc = fc/1e9
        if fc < 6. : fc = 6.
        a = 0.208*np.log10(fc) - 0.782
        b = 25.
        c = -0.13*np.log10(fc) + 2.03
        e = 7.66*np.log10(fc) - 5.96
        if submodel == 'los':
            return 0.0
        elif submodel == 'nlos':
            return e - np.power(10, a*np.log10(np.maximum(b, d_2d)) + c
                                - 0.07*(h_ut -1.5))
        elif submodel == 'o2i':
            return e - np.power(10, a*np.log10(np.maximum(b, d_2d)) + c
                                - 0.07*(h_ut -1.5))
    elif model == 'rma':
        if submodel == 'los':
            return 0.0
        elif submodel == 'nlos':
            return np.arctan((35. - 3.5)/d_2d) - np.arctan((35 - 1.5)/d_2d)
        elif submodel == 'o2i':
            return np.arctan((35 - 3.5)/d_2d) - np.arctan((35 - 1.5)/d_2d)

###### LoS probability

def los_probability(model, d_2d_out, h_ut):
    """Return LoS probability

    Input
    ------
    model : str
        Model. Should be one of 'umi', 'uma', or 'rma'

    d_2d_out : any shape, float
        Outdoor distance

    h_ut : float
        UT heigh

    Output
    -------
    : same shape as ``d_2d_out``, float
        Probability of LoS
    """
    if model == 'rma':
        return np.where(d_2d_out < 10.0, 1.0, np.exp(-(d_2d_out-10)/1e3))
    elif model == 'umi':
        p = (18./d_2d_out)+np.exp(-d_2d_out/36.)*(1.-18./d_2d_out)
        return np.where(d_2d_out < 18.0, 1.0, p)
    elif model == 'uma':
        c = np.where(h_ut < 13.0, 0.0, np.power(np.abs(h_ut-13.)/10., 1.5))
        p1 = (18./d_2d_out)+np.exp(-d_2d_out/63.)*(1.-18./d_2d_out)
        p2 = 1.+c*5/4*np.power(d_2d_out/1e2,3)*np.exp(-d_2d_out/150)
        return np.where(d_2d_out<18., 1., p1*p2)

#########################################################################
# Procedures to generate rays, extracted from 3GPP TR38.901 specification
#########################################################################

r_tau = {'rma' : {  'los'  : 3.8,
                    'nlos' : 1.7,
                    'o2i'  : 1.7},
         'umi' : {  'los'  : 3.0,
                    'nlos' : 2.1,
                    'o2i'  : 2.2},
         'uma' : {  'los'  : 2.5,
                    'nlos' : 2.3,
                    'o2i'  : 2.2}
}

zeta = {'umi'   : { 'los'  : 3.0,
                    'nlos' : 3.0,
                    'o2i'  : 4.0},
        'uma'   : { 'los'  : 3.0,
                    'nlos' : 3.0,
                    'o2i'  : 4.0},
        'rma'   : { 'los'  : 3.0,
                    'nlos' : 3.0,
                    'o2i'  : 3.0}
}

C_phi = {   4   : 0.779,
            5   : 0.860,
            8   : 1.018,
            10  : 1.090,
            11  : 1.123,
            12  : 1.146,
            14  : 1.190,
            15  : 1.211,
            16  : 1.226,
            19  : 1.273,
            20  : 1.289,
            25  : 1.358
}

C_theta = { 8   : 0.889,
            10  : 0.957,
            11  : 1.031,
            12  : 1.104,
            15  : 1.1088,
            19  : 1.184,
            20  : 1.178,
            25  : 1.282
}

c_asa = {   'umi'   :   {   'los'   :   17,
                            'nlos'  :   22,
                            'o2i'   :   8},
            'uma'   :   {   'los'   :   11,
                            'nlos'  :   15,
                            'o2i'   :   8},
            'rma'   :   {   'los'   :   3,
                            'nlos'  :   3,
                            'o2i'   :   3}
}

c_asd = {   'umi'   :   {   'los'   :   3,
                            'nlos'  :   10,
                            'o2i'   :   5},
            'uma'   :   {   'los'   :   5,
                            'nlos'  :   2,
                            'o2i'   :   5},
            'rma'   :   {   'los'   :   2,
                            'nlos'  :   2,
                            'o2i'   :   2}
}

c_zsa = {   'umi'   :   {   'los'   :   7,
                            'nlos'  :   7,
                            'o2i'   :   3},
            'uma'   :   {   'los'   :   7,
                            'nlos'  :   7,
                            'o2i'   :   3},
            'rma'   :   {   'los'   :   3,
                            'nlos'  :   3,
                            'o2i'   :   3}
}

xpr_mu = {  'umi'   :   {   'los'   :   9.0,
                            'nlos'  :   8.0,
                            'o2i'   :   9.0},
            'uma'   :   {   'los'   :   8.0,
                            'nlos'  :   7.0,
                            'o2i'   :   9.0},
            'rma'   :   {   'los'   :   12.0,
                            'nlos'  :   7.0,
                            'o2i'   :   7.0}
}

xpr_std = {  'umi'   :  {   'los'   :   3.0,
                            'nlos'  :   3.0,
                            'o2i'   :   5.0},
            'uma'   :   {   'los'   :   4.0,
                            'nlos'  :   3.0,
                            'o2i'   :   5.0},
            'rma'   :   {   'los'   :   4.0,
                            'nlos'  :   3.0,
                            'o2i'   :   3.0}
}

def delays(model, submodel, batch_size, num_clusters, ds, k):
    """Reference implementation: Delays"""
    x = config.np_rng.uniform(size=[batch_size, num_clusters], low=1e-6,
                            high=1.0)
    x = -r_tau[model][submodel]*ds*np.log(x)
    x = np.sort(x - np.min(x, axis=1, keepdims=True), axis=1)
    if submodel == 'los':
        k = 10.*np.log10(k)
        c = 0.7705 - 0.0433*k + 0.0002*np.square(k) + 0.000017*np.power(k,3)
        tau = x / c
    else:
        tau = x
    return x, tau

def powers(model, submodel, batch_size, num_clusters, unscaled_tau, ds, k):
    """Reference implementation: Powers"""
    z = config.np_rng.normal(size=[batch_size, num_clusters], loc=0.0,
                            scale=zeta[model][submodel])
    rt = r_tau[model][submodel]
    p = np.exp(-unscaled_tau*(rt-1.)/(rt*ds))*np.power(10.0, -z/10.0)
    p = p / np.sum(p, axis=1, keepdims=True)
    if submodel == 'los':
        p_angles = p*(1./(k+1.))
        p_angles[:,:1] = p_angles[:,:1] + k/(k+1.)
    else:
        p_angles = p
    return p, p_angles

def aoa(model, submodel, batch_size, num_clusters, asa, p, los_aoa, k=None):
    """Reference implementation: AoA"""
    a = 2*(asa/1.4)
    c = C_phi[num_clusters]
    if submodel == 'los':
        k = 10.0*np.log10(k)
        c = c*(1.1035-0.028*k-0.002*np.square(k)+0.0001*np.power(k,3))
    aoa_prime = a*np.sqrt(-np.log(p/np.max(p, axis=1, keepdims=True)))/c
    x = config.np_rng.integers(0, 2, size=[batch_size, num_clusters])
    x = 2*x-1
    y = config.np_rng.normal(size=[batch_size, num_clusters], loc=0.0,
                            scale=asa/7.0)
    aoa = x*aoa_prime + y
    if submodel == 'los':
        aoa = aoa - (aoa[:,:1] - los_aoa[:,:,0])
    else:
        aoa = aoa + los_aoa[:,:,0]
    aoa = np.expand_dims(aoa, axis=-1)
    alpha_m = np.array([[[ 0.0447, -0.0447,
                            0.1413, -0.1413,
                            0.2492, -0.2492,
                            0.3715, -0.3715,
                            0.5129, -0.5129,
                            0.6797, -0.6797,
                            0.8844, -0.8844,
                            1.1481, -1.1481,
                            1.5195, -1.5195,
                            2.1551, -2.1551]]])
    aoa = aoa + alpha_m*c_asa[model][submodel]

    # Wrap to (-180,180)
    aoa = np.mod(aoa, 360.)
    aoa = np.where(aoa < 180.0, aoa, aoa-360.)

    # Deg -> Rad
    aoa = aoa*np.pi/180.

    return aoa

def aod(model, submodel, batch_size, num_clusters, asd, p, los_aod, k=None):
    """Reference implementation: AoD"""
    a = 2*(asd/1.4)
    c = C_phi[num_clusters]
    if submodel == 'los':
        k = 10.0*np.log10(k)
        c = c*(1.1035-0.028*k-0.002*np.square(k)+0.0001*np.power(k,3))
    aod_prime = a*np.sqrt(-np.log(p/np.max(p, axis=1, keepdims=True)))/c
    x = config.np_rng.integers(0, 2, size=[batch_size, num_clusters])
    x = 2*x-1
    y = config.np_rng.normal(size=[batch_size, num_clusters], loc=0.0,
                            scale=asd/7.0)
    aod = x*aod_prime + y
    if submodel == 'los':
        aod = aod - (aod[:,:1] - los_aod[:,:,0])
    else:
        aod = aod + los_aod[:,:,0]
    aod = np.expand_dims(aod, axis=-1)
    alpha_m = np.array([[[ 0.0447, -0.0447,
                            0.1413, -0.1413,
                            0.2492, -0.2492,
                            0.3715, -0.3715,
                            0.5129, -0.5129,
                            0.6797, -0.6797,
                            0.8844, -0.8844,
                            1.1481, -1.1481,
                            1.5195, -1.5195,
                            2.1551, -2.1551]]])
    aod = aod + alpha_m*c_asd[model][submodel]

    # Wrap to (-180,180)
    aod = np.mod(aod, 360.)
    aod = np.where(aod < 180.0, aod, aod-360.)

    # deg -> rad
    aod = aod*np.pi/180.

    return aod

def zoa(model, submodel, batch_size, num_clusters, zsa, p, los_zoa, k=None):
    """Reference implementation: ZoA"""
    c = C_theta[num_clusters]
    if submodel == 'los':
        k = 10.0*np.log10(k)
        c = c*(1.3086+0.0339*k-0.0077*np.square(k)+0.0002*np.power(k,3))
    zoa_prime = -zsa*np.log(p/np.max(p, axis=1, keepdims=True))/c
    x = config.np_rng.integers(0, 2, size=[batch_size, num_clusters])
    x = 2*x-1
    y = config.np_rng.normal(size=[batch_size, num_clusters], loc=0.0,
                            scale=zsa/7.0)
    zoa = x*zoa_prime + y
    if submodel == 'los':
        zoa = zoa - (zoa[:,:1] - los_zoa[:,:,0])
    elif submodel == 'nlos':
        zoa = zoa + los_zoa[:,:,0]
    elif submodel == 'o2i':
        zoa = zoa + 90.0
    zoa = np.expand_dims(zoa, axis=-1)
    alpha_m = np.array([[[ 0.0447, -0.0447,
                            0.1413, -0.1413,
                            0.2492, -0.2492,
                            0.3715, -0.3715,
                            0.5129, -0.5129,
                            0.6797, -0.6797,
                            0.8844, -0.8844,
                            1.1481, -1.1481,
                            1.5195, -1.5195,
                            2.1551, -2.1551]]])
    zoa = zoa + alpha_m*c_zsa[model][submodel]

    # Wrap to (0,180)
    zoa = np.mod(zoa, 360.)
    zoa = np.where(zoa < 180.0, zoa, 360.-zoa)

    # deg-> rad
    zoa = zoa*np.pi/180.

    return zoa

def zod(model, submodel, batch_size, num_clusters, zsd, p, los_zod, offset,
        mu_log_zod, k=None):
    """Reference implementation: ZoD"""
    c = C_theta[num_clusters]
    if submodel == 'los':
        k = 10.0*np.log10(k)
        c = c*(1.3086+0.0339*k-0.0077*np.square(k)+0.0002*np.power(k,3))
    zod_prime = -zsd*np.log(p/np.max(p, axis=1, keepdims=True))/c
    x = config.np_rng.integers(0, 2, size=[batch_size, num_clusters])
    x = 2*x-1
    y = config.np_rng.normal(size=[batch_size, num_clusters], loc=0.0,
                            scale=zsd/7.0)
    zod = x*zod_prime + y
    if submodel == 'los':
        zod = zod - (zod[:,:1] - los_zod[:,:,0])
    else:
        zod = zod + los_zod[:,:,0] + offset
    zod = np.expand_dims(zod, axis=-1)
    alpha_m = np.array([[[ 0.0447, -0.0447,
                           0.1413, -0.1413,
                           0.2492, -0.2492,
                           0.3715, -0.3715,
                           0.5129, -0.5129,
                           0.6797, -0.6797,
                           0.8844, -0.8844,
                           1.1481, -1.1481,
                           1.5195, -1.5195,
                           2.1551, -2.1551]]])
    mu_zsd = np.power(10.0, mu_log_zod)
    mu_zsd = np.expand_dims(np.expand_dims(mu_zsd, axis=1), axis=2)
    zod = zod + alpha_m*(3/8)*mu_zsd

    # Wrap to (0,180)
    zod = np.mod(zod, 360.)
    zod = np.where(zod < 180.0, zod, 360.-zod)

    # deg -> rad
    zod = zod*np.pi/180.

    return zod

def xpr(model, submodel, batch_size, num_clusters):
    """Reference implementation: XPR"""
    x = config.np_rng.normal(loc=xpr_mu[model][submodel],
                            scale=xpr_std[model][submodel],
                            size=[batch_size, num_clusters, 20])
    return np.power(10., x/10.)

#########################################################################
# TDL power delay profiles and Rician K-factor
#########################################################################

TDL_POWERS = {
    'A' : np.array([-13.4,
                    0.0,
                    -2.2,
                    -4.0,
                    -6.0,
                    -8.2,
                    -9.9,
                    -10.5,
                    -7.5,
                    -15.9,
                    -6.6,
                    -16.7,
                    -12.4,
                    -15.2,
                    -10.8,
                    -11.3,
                    -12.7,
                    -16.2,
                    -18.3,
                    -18.9,
                    -16.6,
                    -19.9,
                    -29.7]),
    'B' : np.array([0.0,
                    -2.2,
                    -4.0,
                    -3.2,
                    -9.8,
                    -1.2,
                    -3.4,
                    -5.2,
                    -7.6,
                    -3.0,
                    -8.9,
                    -9.0,
                    -4.8,
                    -5.7,
                    -7.5,
                    -1.9,
                    -7.6,
                    -12.2,
                    -9.8,
                    -11.4,
                    -14.9,
                    -9.2,
                    -11.3]),
    'C' : np.array([-4.4,
                    -1.2,
                    -3.5,
                    -5.2,
                    -2.5,
                    0.0,
                    -2.2,
                    -3.9,
                    -7.4,
                    -7.1,
                    -10.7,
                    -11.1,
                    -5.1,
                    -6.8,
                    -8.7,
                    -13.2,
                    -13.9,
                    -13.9,
                    -15.8,
                    -17.1,
                    -16,
                    -15.7,
                    -21.6,
                    -22.8]),
    'D' : np.array([0.0,
                    -18.8,
                    -21,
                    -22.8,
                    -17.9,
                    -20.1,
                    -21.9,
                    -22.9,
                    -27.8,
                    -23.6,
                    -24.8,
                    -30.0,
                    -27.7]),
    'E' : np.array([0.0,
                    -15.8,
                    -18.1,
                    -19.8,
                    -22.9,
                    -22.4,
                    -18.6,
                    -20.8,
                    -22.6,
                    -22.3,
                    -25.6,
                    -20.2,
                    -29.8,
                    -29.2]),
    'A30' : np.array([-15.5,
                      0.0,
                      -5.1,
                      -5.1,
                      -9.6,
                      -8.2,
                      -13.1,
                      -11.5,
                      -11.0,
                      -16.2,
                      -16.6,
                      -26.2]),
    'B100' : np.array([0.0,
                       -2.2,
                       -0.6,
                       -0.6,
                       -0.3,
                       -1.2,
                       -5.9,
                       -2.2,
                       -0.8,
                       -6.3,
                       -7.5,
                       -7.1]),
    'C300' : np.array([-6.9,
                       0.0,
                       -7.7,
                       -2.5,
                       -2.4,
                       -9.9,
                       -8.0,
                       -6.6,
                       -7.1,
                       -13.0,
                       -14.2,
                       -16.0])
}

TDL_DELAYS = {
    'A' : np.array([0.0000,
                    0.3819,
                    0.4025,
                    0.5868,
                    0.4610,
                    0.5375,
                    0.6708,
                    0.5750,
                    0.7618,
                    1.5375,
                    1.8978,
                    2.2242,
                    2.1718,
                    2.4942,
                    2.5119,
                    3.0582,
                    4.0810,
                    4.4579,
                    4.5695,
                    4.7966,
                    5.0066,
                    5.3043,
                    9.6586]),
    'B' : np.array([0.0000,
                    0.1072,
                    0.2155,
                    0.2095,
                    0.2870,
                    0.2986,
                    0.3752,
                    0.5055,
                    0.3681,
                    0.3697,
                    0.5700,
                    0.5283,
                    1.1021,
                    1.2756,
                    1.5474,
                    1.7842,
                    2.0169,
                    2.8294,
                    3.0219,
                    3.6187,
                    4.1067,
                    4.2790,
                    4.7834]),

    'C' : np.array([0,
                    0.2099,
                    0.2219,
                    0.2329,
                    0.2176,
                    0.6366,
                    0.6448,
                    0.6560,
                    0.6584,
                    0.7935,
                    0.8213,
                    0.9336,
                    1.2285,
                    1.3083,
                    2.1704,
                    2.7105,
                    4.2589,
                    4.6003,
                    5.4902,
                    5.6077,
                    6.3065,
                    6.6374,
                    7.0427,
                    8.6523]),
    'D' : np.array([0.0,
                    0.035,
                    0.612,
                    1.363,
                    1.405,
                    1.804,
                    2.596,
                    1.775,
                    4.042,
                    7.937,
                    9.424,
                    9.708,
                    12.525]),
    'E' : np.array([0.0,
                    0.5133,
                    0.5440,
                    0.5630,
                    0.5440,
                    0.7112,
                    1.9092,
                    1.9293,
                    1.9589,
                    2.6426,
                    3.7136,
                    5.4524,
                    12.0034,
                    20.6519]),
    'A30' : np.array([0.0,
                      10.0,
                      15.0,
                      20.0,
                      25.0,
                      50.0,
                      65.0,
                      75.0,
                      105.0,
                      135.0,
                      150.0,
                      190.0]),
    'B100' : np.array([0.0,
                       10.0,
                       20.0,
                       30.0,
                       35.0,
                       45.0,
                       55.0,
                       120.0,
                       170.0,
                       245.0,
                       330.0,
                       480.0]),
    'C300' : np.array([0.0,
                       65.0,
                       70.0,
                       190.0,
                       195.0,
                       200.0,
                       240.0,
                       325.0,
                       520.0,
                       1045.0,
                       1510.0,
                       2595.0])
}

TDL_RICIAN_K = {'A' : None,
                'B' : None,
                'C' : None,
                'D' : 13.3,
                'E' : 22.0
}

#########################################################################
# CDL power delay profiles and Rician K-factor
#########################################################################

CDL_POWERS = {
    'A' : np.array([-13.4,
                    0.0,
                    -2.2,
                    -4.0,
                    -6.0,
                    -8.2,
                    -9.9,
                    -10.5,
                    -7.5,
                    -15.9,
                    -6.6,
                    -16.7,
                    -12.4,
                    -15.2,
                    -10.8,
                    -11.3,
                    -12.7,
                    -16.2,
                    -18.3,
                    -18.9,
                    -16.6,
                    -19.9,
                    -29.7]),

    'B' : np.array([0.0,
                    -2.2,
                    -4.0,
                    -3.2,
                    -9.8,
                    -1.2,
                    -3.4,
                    -5.2,
                    -7.6,
                    -3.0,
                    -8.9,
                    -9.0,
                    -4.8,
                    -5.7,
                    -7.5,
                    -1.9,
                    -7.6,
                    -12.2,
                    -9.8,
                    -11.4,
                    -14.9,
                    -9.2,
                    -11.3]),

    'C' : np.array([-4.4,
                    -1.2,
                    -3.5,
                    -5.2,
                    -2.5,
                    0.0,
                    -2.2,
                    -3.9,
                    -7.4,
                    -7.1,
                    -10.7,
                    -11.1,
                    -5.1,
                    -6.8,
                    -8.7,
                    -13.2,
                    -13.9,
                    -13.9,
                    -15.8,
                    -17.1,
                    -16.0,
                    -15.7,
                    -21.6,
                    -22.8]),

    'D' : np.array([0.0,
                    -18.8,
                    -21.0,
                    -22.8,
                    -17.9,
                    -20.1,
                    -21.9,
                    -22.9,
                    -27.8,
                    -23.6,
                    -24.8,
                    -30.0,
                    -27.7]),

    'E' : np.array([0.0,
                    -15.8,
                    -18.1,
                    -19.8,
                    -22.9,
                    -22.4,
                    -18.6,
                    -20.8,
                    -22.6,
                    -22.3,
                    -25.6,
                    -20.2,
                    -29.8,
                    -29.2]),
}

CDL_DELAYS = {
    'A' : np.array([0.0,
                    0.3819,
                    0.4025,
                    0.5868,
                    0.4610,
                    0.5375,
                    0.6708,
                    0.5750,
                    0.7618,
                    1.5375,
                    1.8978,
                    2.2242,
                    2.1718,
                    2.4942,
                    2.5119,
                    3.0582,
                    4.0810,
                    4.4579,
                    4.5695,
                    4.7966,
                    5.0066,
                    5.3043,
                    9.6586]),

    'B' : np.array([0.0,
                    0.1072,
                    0.2155,
                    0.2095,
                    0.2870,
                    0.2986,
                    0.3752,
                    0.5055,
                    0.3681,
                    0.3697,
                    0.5700,
                    0.5283,
                    1.1021,
                    1.2756,
                    1.5474,
                    1.7842,
                    2.0169,
                    2.8294,
                    3.0219,
                    3.6187,
                    4.1067,
                    4.2790,
                    4.7834]),

    'C' : np.array([0.0,
                    0.2099,
                    0.2219,
                    0.2329,
                    0.2176,
                    0.6366,
                    0.6448,
                    0.6560,
                    0.6584,
                    0.7935,
                    0.8213,
                    0.9336,
                    1.2285,
                    1.3083,
                    2.1704,
                    2.7105,
                    4.2589,
                    4.6003,
                    5.4902,
                    5.6077,
                    6.3065,
                    6.6374,
                    7.0427,
                    8.6523]),

    'D' : np.array([0.0,
                    0.035,
                    0.612,
                    1.363,
                    1.405,
                    1.804,
                    2.596,
                    1.775,
                    4.042,
                    7.937,
                    9.424,
                    9.708,
                    12.525]),

    'E' : np.array([0.0,
                    0.5133,
                    0.5440,
                    0.5630,
                    0.5440,
                    0.7112,
                    1.9092,
                    1.9293,
                    1.9589,
                    2.6426,
                    3.7136,
                    5.4524,
                    12.0034,
                    20.6419])
}

CDL_AOD = {
    'A' : np.array([-178.1,
                    -4.2,
                    -4.2,
                    -4.2,
                    90.2,
                    90.2,
                    90.2,
                    121.5,
                    -81.7,
                    158.4,
                    -83.0,
                    134.8,
                    -153.0,
                    -172.0,
                    -129.9,
                    -136.0,
                    165.4,
                    148.4,
                    132.7,
                    -118.6,
                    -154.1,
                    126.5,
                    -56.2]),

    'B' : np.array([9.3,
                    9.3,
                    9.3,
                    -34.1,
                    -65.4,
                    -11.4,
                    -11.4,
                    -11.4,
                    -67.2,
                    52.5,
                    -72.0,
                    74.3,
                    -52.2,
                    -50.5,
                    61.4,
                    30.6,
                    -72.5,
                    -90.6,
                    -77.6,
                    -82.6,
                    -103.6,
                    75.6,
                    -77.6]),

    'C' : np.array([-46.6,
                    -22.8,
                    -22.8,
                    -22.8,
                    -40.7,
                    0.3,
                    0.3,
                    0.3,
                    73.1,
                    -64.5,
                    80.2,
                    -97.1,
                    -55.3,
                    -64.3,
                    -78.5,
                    102.7,
                    99.2,
                    88.8,
                    -101.9,
                    92.2,
                    93.3,
                    106.6,
                    119.5,
                    -123.8]),

    'D' : np.array([0.0,
                    89.2,
                    89.2,
                    89.2,
                    13.0,
                    13.0,
                    13.0,
                    34.6,
                    -64.5,
                    -32.9,
                    52.6,
                    -132.1,
                    77.2]),

    'E' : np.array([0.0,
                    57.5,
                    57.5,
                    57.5,
                    -20.1,
                    16.2,
                    9.3,
                    9.3,
                    9.3,
                    19.0,
                    32.7,
                    0.5,
                    55.9,
                    57.6])
}

CDL_AOA = {
    'A' : np.array([51.3,
                    -152.7,
                    -152.7,
                    -152.7,
                    76.6,
                    76.6,
                    76.6,
                    -1.8,
                    -41.9,
                    94.2,
                    51.9,
                    -115.9,
                    26.6,
                    76.6,
                    -7.0,
                    -23.0,
                    -47.2,
                    110.4,
                    144.5,
                    155.3,
                    102.0,
                    -151.8,
                    55.2]),

    'B' : np.array([-173.3,
                    -173.3,
                    -173.3,
                    125.5,
                    -88.0,
                    155.1,
                    155.1,
                    155.1,
                    -89.8,
                    132.1,
                    -83.6,
                    95.3,
                    103.7,
                    -87.8,
                    -92.5,
                    -139.1,
                    -90.6,
                    58.6,
                    -79.0,
                    65.8,
                    52.7,
                    88.7,
                    -60.4]),

    'C' : np.array([-101.0,
                    120.0,
                    120.0,
                    120.0,
                    -127.5,
                    170.4,
                    170.4,
                    170.4,
                    55.4,
                    66.5,
                    -48.1,
                    46.9,
                    68.1,
                    -68.7,
                    81.5,
                    30.7,
                    -16.4,
                    3.8,
                    -13.7,
                    9.7,
                    5.6,
                    0.7,
                    -21.9,
                    33.6]),

    'D' : np.array([-180.0,
                    89.2,
                    89.2,
                    89.2,
                    163.0,
                    163.0,
                    163.0,
                    -137.0,
                    74.5,
                    127.7,
                    -119.6,
                    -9.1,
                    -83.8]),

    'E' : np.array([-180.0,
                    18.2,
                    18.2,
                    18.2,
                    101.8,
                    112.9,
                    -155.5,
                    -155.5,
                    -155.5,
                    -143.3,
                    -94.7,
                    147.0,
                    -36.2,
                    -26.0])
}

CDL_ZOD = {
    'A' : np.array([50.2,
                    93.2,
                    93.2,
                    93.2,
                    122.0,
                    122.0,
                    122.0,
                    150.2,
                    55.2,
                    26.4,
                    126.4,
                    171.6,
                    151.4,
                    157.2,
                    47.2,
                    40.4,
                    43.3,
                    161.8,
                    10.8,
                    16.7,
                    171.7,
                    22.7,
                    144.9]),

    'B' : np.array([105.8,
                    105.8,
                    105.8,
                    115.3,
                    119.3,
                    103.2,
                    103.2,
                    103.2,
                    118.2,
                    102.0,
                    100.4,
                    98.3,
                    103.4,
                    102.5,
                    101.4,
                    103.0,
                    100.0,
                    115.2,
                    100.5,
                    119.6,
                    118.7,
                    117.8,
                    115.7]),

    'C' : np.array([97.2,
                    98.6,
                    98.6,
                    98.6,
                    100.6,
                    99.2,
                    99.2,
                    99.2,
                    105.2,
                    95.3,
                    106.1,
                    93.5,
                    103.7,
                    104.2,
                    93.0,
                    104.2,
                    94.9,
                    93.1,
                    92.2,
                    106.7,
                    93.0,
                    92.9,
                    105.2,
                    107.8]),

    'D' : np.array([98.5,
                    85.5,
                    85.5,
                    85.5,
                    97.5,
                    97.5,
                    97.5,
                    98.5,
                    88.4,
                    91.3,
                    103.8,
                    80.3,
                    86.5]),

    'E' : np.array([99.6,
                    104.2,
                    104.2,
                    104.2,
                    99.4,
                    100.8,
                    98.8,
                    98.8,
                    98.8,
                    100.8,
                    96.4,
                    98.9,
                    95.6,
                    104.6])
}

CDL_ZOA = {
    'A' : np.array([125.4,
                    91.3,
                    91.3,
                    91.3,
                    94.0,
                    94.0,
                    94.0,
                    47.1,
                    56.0,
                    30.1,
                    58.8,
                    26.0,
                    49.2,
                    143.1,
                    117.4,
                    122.7,
                    123.2,
                    32.6,
                    27.2,
                    15.2,
                    146.0,
                    150.7,
                    156.1]),

    'B' : np.array([78.9,
                    78.9,
                    78.9,
                    63.3,
                    59.9,
                    67.5,
                    67.5,
                    67.5,
                    82.6,
                    66.3,
                    61.6,
                    58.0,
                    78.2,
                    82.0,
                    62.4,
                    78.0,
                    60.9,
                    82.9,
                    60.8,
                    57.3,
                    59.9,
                    60.1,
                    62.3]),

    'C' : np.array([87.6,
                    72.1,
                    72.1,
                    72.1,
                    70.1,
                    75.3,
                    75.3,
                    75.3,
                    67.4,
                    63.8,
                    71.4,
                    60.5,
                    90.6,
                    60.1,
                    61.0,
                    100.7,
                    62.3,
                    66.7,
                    52.9,
                    61.8,
                    51.9,
                    61.7,
                    58.0,
                    57.0]),

    'D' : np.array([81.5,
                    86.9,
                    86.9,
                    86.9,
                    79.4,
                    79.4,
                    79.4,
                    78.2,
                    73.6,
                    78.3,
                    87.0,
                    70.6,
                    72.9]),

    'E' : np.array([80.4,
                    80.4,
                    80.4,
                    80.4,
                    80.8,
                    86.3,
                    82.7,
                    82.7,
                    82.7,
                    82.9,
                    88.0,
                    81.0,
                    88.6,
                    78.3])
}

CDL_XPR = { 'A' : np.power(10.0, 10.0/10.0),
            'B' : np.power(10.0, 8.0/10.0),
            'C' : np.power(10.0, 7.0/10.0),
            'D' : np.power(10.0, 11.0/10.0),
            'E' : np.power(10.0, 8.0/10.0)}

def cdl_aod(model):
    C_AOD = {   'A' : 5.0,
                'B' : 10.0,
                'C' : 2.0,
                'D' : 5.0,
                'E' : 5.0   }
    aod = np.expand_dims(CDL_AOD[model], axis=-1)
    alpha_m = np.array([[ 0.0447, -0.0447,
                           0.1413, -0.1413,
                           0.2492, -0.2492,
                           0.3715, -0.3715,
                           0.5129, -0.5129,
                           0.6797, -0.6797,
                           0.8844, -0.8844,
                           1.1481, -1.1481,
                           1.5195, -1.5195,
                           2.1551, -2.1551]])
    c = C_AOD[model]
    aod = aod + c*alpha_m
    aod = aod*np.pi/180.
    return aod

def cdl_aod(model):
    C_AOD = {   'A' : 5.0,
                'B' : 10.0,
                'C' : 2.0,
                'D' : 5.0,
                'E' : 5.0   }
    aod = np.expand_dims(CDL_AOD[model], axis=-1)
    alpha_m = np.array([[ 0.0447, -0.0447,
                           0.1413, -0.1413,
                           0.2492, -0.2492,
                           0.3715, -0.3715,
                           0.5129, -0.5129,
                           0.6797, -0.6797,
                           0.8844, -0.8844,
                           1.1481, -1.1481,
                           1.5195, -1.5195,
                           2.1551, -2.1551]])
    c = C_AOD[model]
    aod = aod + c*alpha_m
    aod = aod*np.pi/180.
    return aod

def cdl_aoa(model):
    C_AOA = {   'A' : 11.0,
                'B' : 22.0,
                'C' : 15.0,
                'D' : 8.0,
                'E' : 11.0   }
    aoa = np.expand_dims(CDL_AOA[model], axis=-1)
    alpha_m = np.array([[ 0.0447, -0.0447,
                           0.1413, -0.1413,
                           0.2492, -0.2492,
                           0.3715, -0.3715,
                           0.5129, -0.5129,
                           0.6797, -0.6797,
                           0.8844, -0.8844,
                           1.1481, -1.1481,
                           1.5195, -1.5195,
                           2.1551, -2.1551]])
    c = C_AOA[model]
    aoa = aoa + c*alpha_m
    aoa = aoa*np.pi/180.
    return aoa

def cdl_zod(model):
    C_ZOD = {   'A' : 3.0,
                'B' : 3.0,
                'C' : 3.0,
                'D' : 3.0,
                'E' : 3.0   }
    zod = np.expand_dims(CDL_ZOD[model], axis=-1)
    alpha_m = np.array([[ 0.0447, -0.0447,
                           0.1413, -0.1413,
                           0.2492, -0.2492,
                           0.3715, -0.3715,
                           0.5129, -0.5129,
                           0.6797, -0.6797,
                           0.8844, -0.8844,
                           1.1481, -1.1481,
                           1.5195, -1.5195,
                           2.1551, -2.1551]])
    c = C_ZOD[model]
    zod = zod + c*alpha_m
    zod = zod*np.pi/180.
    return zod

def cdl_zoa(model):
    C_ZOA = {   'A' : 3.0,
                'B' : 7.0,
                'C' : 7.0,
                'D' : 3.0,
                'E' : 7.0   }
    zoa = np.expand_dims(CDL_ZOA[model], axis=-1)
    alpha_m = np.array([[ 0.0447, -0.0447,
                           0.1413, -0.1413,
                           0.2492, -0.2492,
                           0.3715, -0.3715,
                           0.5129, -0.5129,
                           0.6797, -0.6797,
                           0.8844, -0.8844,
                           1.1481, -1.1481,
                           1.5195, -1.5195,
                           2.1551, -2.1551]])
    c = C_ZOA[model]
    zoa = zoa + c*alpha_m
    zoa = zoa*np.pi/180.
    return zoa
