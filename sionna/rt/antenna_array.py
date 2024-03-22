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
from .utils import rotate, theta_hat

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
                 trainable_positions=False, # NOTE: this argument was deprecated
                 trainable_azimuth=False,
                 trainable_elevation=False,
                 trainable_beamweights=False,
                 dtype=tf.complex64):

        if dtype not in (tf.complex64, tf.complex128):
            raise ValueError("`dtype` must be tf.complex64 or tf.complex128`")

        self._dtype = dtype
        self._rdtype = dtype.real_dtype

        # Create list of antennas
        self.num_cols = num_cols
        self.num_rows = num_rows
        array_size = self.num_rows * self.num_cols
        antenna = Antenna(pattern, polarization, polarization_model, dtype)

        # Compute antenna positions
        self.frequency = scene.Scene().frequency
        self.wavelength = SPEED_OF_LIGHT / self.frequency
        self.vertical_spacing = vertical_spacing
        self.d_v = self.vertical_spacing * self.wavelength
        self.horizontal_spacing = horizontal_spacing
        self.d_h = self.horizontal_spacing * self.wavelength
        self.positions_xyz = np.zeros([array_size, 3])

        for i in range(self.num_rows):
            for j in range(self.num_cols):
                self.positions_xyz[i + j * self.num_rows] = [0,
                                                     j * self.d_h,
                                                     -i * self.d_v]

        # Center the panel around the origin
        offset = [0,
                  -(self.num_cols - 1) * self.d_h * self.vertical_spacing,
                  (self.num_rows - 1) * self.d_v * self.horizontal_spacing]
        self.positions_xyz += offset
        self.antenna = antenna

        self.azimuth = 0.
        self.elevation = 0.

        self.beamweights = tf.zeros((array_size,), dtype=self._dtype) # init beamweights
        self.precoding_weights(azimuth_deg=self.azimuth, elevation_deg=self.elevation) # this will also set self.beamweights

        self.trainable_beamweights = trainable_beamweights
        # trainable weights and trainable_azimuth/elevation should be mutually exclusive

        self.trainable_azimuth = trainable_azimuth
        self.trainable_elevation = trainable_elevation

        assert not (self.trainable_beamweights and (self.trainable_azimuth or self.trainable_elevation)), \
            "Either beamweights or azimuth/elevation angles can be trainable at the same time."


        super().__init__(self.antenna, self.positions_xyz, dtype)   # NOTE: removed trainable_positions argument

    @property
    def trainable_beamweights(self):
        """
        bool : Get/set if the antenna beamweights are trainable
            variables or not.
            Defaults to `False`.
        """
        return self._trainable_beamweights

    @trainable_beamweights.setter
    def trainable_beamweights(self, value):
        if not isinstance(value, bool):
            raise TypeError("`trainable_beamweights` must be bool")
        # pylint: disable=protected-access
        self._beamweights._trainable = value
        self._trainable_beamweights = value

    @property
    def beamweights(self):
        """
        [array_size,], `tf.complex64` : Get/set  array of complex weights used for beamsteering.
        """
        return self._beamweights

    @beamweights.setter
    def beamweights(self, beamweights):
        beamweights = tf.cast(beamweights, self._dtype)
        if not (tf.rank(beamweights) == 1):
            msg = "Each element of ``beamweights`` must must be a single complex value"
            raise ValueError(msg)
        if not hasattr(self, "_beamweights"):
            self._beamweights = tf.Variable(beamweights, constraint=lambda x: x / tf.linalg.norm(x))
        else:
            self._beamweights.assign(beamweights)

    @property
    def trainable_azimuth(self):
        """
        bool : Get/set if the antenna azimuth are trainable
            variables or not.
            Defaults to `False`.
        """
        return self._trainable_azimuth

    @trainable_azimuth.setter
    def trainable_azimuth(self, value):
        if not isinstance(value, bool):
            raise TypeError("`trainable_azimuth` must be bool")
        # pylint: disable=protected-access
        self._azimuth._trainable = value
        self._trainable_azimuth = value

    @property
    def azimuth(self):
        """
        [1,], `tf.float32` : Get/set  array of azimuth angle used for beamsteering.
        """
        return self._azimuth

    @azimuth.setter
    def azimuth(self, azimuth):
        azimuth = tf.cast(azimuth, self._rdtype)
        if not (tf.rank(azimuth) == 0):
            msg = "Each element of ``azimuth`` must must be a single real value"
            raise ValueError(msg)
        if not hasattr(self, "_azimuth"):
            self._azimuth = tf.Variable(azimuth, name="Azimuth", constraint=lambda x: tf.clip_by_value(x, clip_value_min=-180, clip_value_max=180))
        else:
            self._azimuth.assign(azimuth)

    @property
    def trainable_elevation(self):
        """
        bool : Get/set if the antenna elevation are trainable
            variables or not.
            Defaults to `False`.
        """
        return self._trainable_elevation

    @trainable_elevation.setter
    def trainable_elevation(self, value):
        if not isinstance(value, bool):
            raise TypeError("`trainable_elevation` must be bool")
        # pylint: disable=protected-access
        self._elevation._trainable = value
        self._trainable_elevation = value

    @property
    def elevation(self):
        """
        [1,], `tf.float32` : Get/set  array of elevation angle used for beamsteering.
        """
        return self._elevation

    @elevation.setter
    def elevation(self, elevation):
        elevation = tf.cast(elevation, self._rdtype)
        if not (tf.rank(elevation) == 0):
            msg = "Each element of ``elevation`` must must be a single real value"
            raise ValueError(msg)
        if not hasattr(self, "_elevation"):
            self._elevation = tf.Variable(elevation, name="Elevation", constraint=lambda x: tf.clip_by_value(x, clip_value_min=-180, clip_value_max=180))
        else:
            self._elevation.assign(elevation)

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

    def precoding_weights(self, azimuth_deg, elevation_deg):
        r"""
        Compute precoding weights to perform analog beam-steering.
        For 1D arrays, elevation_deg should default to 0.
        This function currently supports either 1D (on y-axis) arrays or 2D (on yz-axis)

        :param azimuth_deg: Azimuth angle. Expressed in degrees, aligned on the xy-axis
                (positive rotation is counter clockwise from positive x to positive y).
        :param elevation_deg: Elevation angles. Expressed in degrees, aligned on the z-axis
                (positive rotation goes from positive xy-plane to positive z)
        :return: normalized 1D vector of phases (elements are enumerated and listed in row-first order).
                    NOTE: current marking of antenna elements in self.show() function consider elements
                    in column-first order instead.
        """

        if self.num_rows == 1:
            elevation_deg = 0   # force elevation_deg=0 when array is ULA (1D)

        spacing_y = self.horizontal_spacing
        spacing_z = self.vertical_spacing
        azimuth_rad = azimuth_deg * tf.convert_to_tensor(np.pi) / 180.
        elevation_rad = elevation_deg * tf.convert_to_tensor(np.pi) / 180.
        yz_component = []
        for n in range(self.num_cols): # columns refer to antenna elements position on the y-axis
            for l in range(self.num_rows):  # rows refer to antenna elements position on the z-axis
                # compute spherical unit vector for given elevation and azimuth
                uv = theta_hat(elevation_rad, azimuth_rad)
                yz_component.append(
                    (n * spacing_y * uv[1] + l * spacing_z * uv[2])
                )
        # Calculate the total phase for each element in the array
        yz_component = tf.convert_to_tensor(yz_component)
        phase = -2 * np.pi * (yz_component)  # assuming spacing is all the same in all directions
        precoding_weights_ura = tf.exp(tf.complex(0.0, phase))
        precoding_weights_ura = precoding_weights_ura / tf.linalg.norm(precoding_weights_ura)

        self.beamweights = precoding_weights_ura

        return precoding_weights_ura
