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


def conv1d(x, h, padding="SAME"):
    x = tf.expand_dims(x, axis=-1)
    x_real = tf.math.real(x)
    x_imag = tf.math.imag(x)
    x = tf.concat([x_real, x_imag], axis=2)
    x = tf.transpose(x, [0, 2, 1])

    h = tf.reverse(h, [-1])
    h = tf.stack([tf.math.real(h), tf.math.imag(h)])
    h = tf.transpose(h)
    h = tf.reshape(h, [-1, 2])
    h = tf.stack([h, tf.reverse(h, [-1])], axis=2)
    h = tf.reshape(h, [-1, 4])
    h1, h2, h3, h4 = tf.split(h, 4, axis=1)
    h = tf.stack([h1, h2, -h3, h4], axis=1)
    h = tf.reshape(h, [-1, 2, 2])

    x = tf.nn.conv1d(x, h, stride=1, padding=padding, data_format='NCW')

    x = tf.transpose(x, [0, 2, 1])
    x_real, x_imag = tf.split(x, 2, axis=2)
    x = tf.complex(x_real, x_imag)
    x = tf.reshape(x, [1, -1])

    return x


def generate_time_frequency(N, dt, dtype=tf.float32):
    # Time vector
    N_min = tf.cast(tf.math.ceil((N-1)/2), dtype=tf.int32)
    N_max = N-N_min-1
    t = tf.cast(tf.linspace(-N_min, N_max, N), dtype) * dt

    # Frequency vector
    df = 1.0/dt/tf.cast(N, dtype)
    f = tf.cast(tf.linspace(-N_min, N_max, N), dtype) * df

    return t, f


pi = 3.141592653589793
h = 6.62607015 * 10 ** (-34)
