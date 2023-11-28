#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Implements classes and methods related to antenna arrays
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from .antenna import Antenna
from sionna.constants import SPEED_OF_LIGHT
from . import scene
from .utils import rotate

class AntennaArray():
    # pylint: disable=line-too-long
    r"""
    Class implementing an antenna array

    An antenna array is composed of identical antennas that are placed
    at different positions. The ``positions`` parameter can be assigned
    to a TensorFlow variable or tensor.

    .. code-block:: Python

        array = AntennaArray(antenna=Antenna("tr38901", "V"),
                             positions=tf.Variable([[0,0,0], [0, 1, 1]]))

    Parameters
    ----------
    antenna : :class:`~sionna.rt.Antenna`
        Antenna instance

    positions : [array_size, 3], array_like
        Array of relative positions :math:`(x,y,z)` [m] of each
        antenna (dual-polarized antennas are counted as a single antenna
        and share the same position).
        The absolute position of the antennas is obtained by
        adding the position of the :class:`~sionna.rt.Transmitter`
        or :class:`~sionna.rt.Receiver` using it.

    dtype : tf.complex64 or tf.complex128
        Data type used for all computations.
        Defaults to `tf.complex64`.
    """
    def __init__(self, antenna, positions, dtype=tf.complex64):
        super().__init__()

        if dtype not in (tf.complex64, tf.complex128):
            raise ValueError("`dtype` must be tf.complex64 or tf.complex128`")
        self._rdtype = dtype.real_dtype
        self.antenna = antenna
        self.positions = positions

    @property
    def antenna(self):
        """
        :class:`~sionna.rt.Antenna` : Get/set the antenna
        """
        return self._antenna

    @antenna.setter
    def antenna(self, antenna):
        if not isinstance(antenna, Antenna):
            raise TypeError("``antenna`` must be an instance of Antenna.")
        self._antenna = antenna

    @property
    def positions(self):
        """
        [array_size, 3], `tf.float` : Get/set  array of relative positions
        :math:`(x,y,z)` [m] of each antenna (dual-polarized antennas are
        counted as a single antenna and share the same position).
        """
        return self._positions

    @positions.setter
    def positions(self, positions):
        if isinstance(positions, tf.Variable):
            if positions.dtype != self._rdtype:
                raise TypeError(f"`positions` must have dtype={self._rdtype}")
            else:
                self._positions = positions
        else:
            self._positions = tf.cast(positions, self._rdtype)

    @property
    def num_ant(self):
        """
        int (read-only) : Number of linearly polarized antennas in the array.
            Dual-polarized antennas are counted as two linearly polarized
            antennas.
        """
        return self._positions.shape[0]*len(self._antenna.patterns)

    @property
    def array_size(self):
        """
        int (read-only) : Number of antennas in the array.
            Dual-polarized antennas are counted as a single antenna.
        """
        return self._positions.shape[0]

    def rotated_positions(self, orientation):
        r"""
        Get the antenna positions rotated according to ``orientation``

        Input
        ------
        orientation : [3], tf.float
            Orientation :math:`(\alpha, \beta, \gamma)` [rad] specified
            through three angles corresponding to a 3D rotation
            as defined in :eq:`rotation`.

        Output
        -------
        : [array_size, 3]
            Rotated positions
        """
        # [array_size, 3]
        rot_p = rotate(self.positions, orientation)
        return rot_p

class PlanarArray(AntennaArray):
    # pylint: disable=line-too-long
    r"""
    Class implementing a planar antenna array

    The antennas are regularly spaced, located in the y-z plane, and
    numbered column-first from the top-left to bottom-right corner.

    Parameters
    ----------
    num_rows : int
        Number of rows

    num_cols : int
        Number of columns

    vertical_spacing : float
        Vertical antenna spacing [multiples of wavelength].

    horizontal_spacing : float
        Horizontal antenna spacing [multiples of wavelength].

    pattern : str, callable, or length-2 sequence of callables
        Antenna pattern. Either one of
        ["iso", "dipole", "hw_dipole", "tr38901"],
        or a callable, or a length-2 sequence of callables defining
        antenna patterns. In the latter case, the antennas are dual
        polarized and each callable defines the antenna pattern
        in one of the two orthogonal polarization directions.
        An antenna pattern is a callable that takes as inputs vectors of
        zenith and azimuth angles of the same length and returns for each
        pair the corresponding zenith and azimuth patterns. See :eq:`C` for
        more detail.

    polarization : str or None
        Type of polarization. For single polarization, must be "V" (vertical)
        or "H" (horizontal). For dual polarization, must be "VH" or "cross".
        Only needed if ``pattern`` is a string.

    polarization_model: int, one of [1,2]
        Polarization model to be used. Options `1` and `2`
        refer to :func:`~sionna.rt.antenna.polarization_model_1`
        and :func:`~sionna.rt.antenna.polarization_model_2`,
        respectively.
        Defaults to `2`.

    dtype : tf.complex64 or tf.complex128
        Datatype used for all computations.
        Defaults to `tf.complex64`.

    Example
    -------
    .. code-block:: Python

        array = PlanarArray(8,4, 0.5, 0.5, "tr38901", "VH")
        array.show()

    .. figure:: ../figures/antenna_array.png
        :align: center
        :scale: 100%
    """
    def __init__(self,
                 num_rows,
                 num_cols,
                 vertical_spacing,
                 horizontal_spacing,
                 pattern,
                 polarization=None,
                 polarization_model=2,
                 dtype=tf.complex64):

        if dtype not in (tf.complex64, tf.complex128):
            raise ValueError("`dtype` must be tf.complex64 or tf.complex128`")

        # Create list of antennas
        array_size = num_rows*num_cols
        antenna = Antenna(pattern, polarization, polarization_model, dtype)

        # Compute antenna positions
        frequency = scene.Scene().frequency
        wavelength = SPEED_OF_LIGHT/frequency
        d_v = vertical_spacing*wavelength
        d_h = horizontal_spacing*wavelength
        positions =  np.zeros([array_size, 3])

        for i in range(num_rows):
            for j in range(num_cols):
                positions[i + j*num_rows] = [0,
                                             j*d_h,
                                             -i*d_v]

        # Center the panel around the origin
        offset = [0,
                  -(num_cols-1)*d_h/2,
                  (num_rows-1)*d_v/2]
        positions += offset
        super().__init__(antenna, positions, dtype)

    def show(self):
        r"""show()

        Visualizes the antenna array

        Antennas are depicted by markers that are annotated with the antenna
        number. The marker is not related to the polarization of an antenna.

        Output
        ------
        : :class:`matplotlib.pyplot.Figure`
            Figure depicting the antenna array
        """
        fig = plt.figure()
        plt.plot(self.positions[:,1], self.positions[:,2],
                 marker=MarkerStyle("+").get_marker(), markeredgecolor='red',
                 markerfacecolor='red', markersize="10", linestyle="None",
                 markeredgewidth="1")
        for i, p in enumerate(self.positions):
            fig.axes[0].annotate(i+1, (p[1], p[2]))
        plt.xlabel("y (m)")
        plt.ylabel("z (m)")
        plt.title("Planar Array Layout")
        return fig
