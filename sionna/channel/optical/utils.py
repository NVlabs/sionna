# $Id: utils.py 0 13.06.2022 13:53$
# Author: Tim Alexander Uhlemann <uhlemann@ieee.org>
# Copyright:

"""
This module defines the following classes:

Exception classes:

Functions:

 - 'conv1d', Complex convolution in 1-D
 - 'generate_time_frequency', Generates time and frequency vector


How To Use This Module
======================
(See the individual classes, methods, and attributes for details.)
"""

# Standard library imports

# Third party imports
import tensorflow as tf

# Local application imports


def time_frequency_vector(n, dt, dtype=tf.float32):
    # pylint: disable=line-too-long
    r"""
    Compute the time and frequency vector with sample duration ``dt`` in
    normalized time unit unit_time and ``n`` samples, i.e.,

    >>> t = tf.cast(tf.linspace(-n_min, n_max, n), dtype) * dt
    >>> f = tf.cast(tf.linspace(-n_min, n_max, n), dtype) * df

    Input
    ------
        n : int
            Number of samples (1)

        dt : float
            Sample duration (unit_time)

        dtype : tf.DType
            Datatype to use for internal processing and output.
            If a complex datatype is provided, the corresponding precision of
            real components is used.
            Defaults to `tf.float32`.

    Output
    ------
        t : [n], tf.float
            Time vector

        f : [n], tf.float
            Frequency vector
    """

    # Time vector
    n_min = tf.cast(tf.math.ceil((n - 1) / 2), dtype=tf.int32)
    n_max = n - n_min - 1
    t = tf.cast(tf.linspace(-n_min, n_max, n), dtype) * dt

    # Frequency vector
    df = 1.0/dt/tf.cast(n, dtype)
    f = tf.cast(tf.linspace(-n_min, n_max, n), dtype) * df

    return t, f
