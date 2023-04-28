#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
A scene contains everything that is needed for rendering and radio propagation
simulation. This includes the scene geometry, materials, transmitters,
receivers, as well as cameras.
"""

import os
from importlib_resources import files

import matplotlib
import matplotlib.pyplot as plt
import mitsuba as mi
import tensorflow as tf

from .antenna_array import AntennaArray
from .camera import Camera
from sionna.constants import SPEED_OF_LIGHT
from .itu_materials import instantiate_itu_materials
from .oriented_object import OrientedObject
from .radio_material import RadioMaterial
from .receiver import Receiver
from .scene_object import SceneObject
from .solver import Solver
from .solver_cm import SolverCoverageMap
from .transmitter import Transmitter
from .previewer import InteractiveDisplay
from .renderer import render, coverage_map_color_mapping
from .utils import expand_to_rank
from . import scenes


class Scene:
    # pylint: disable=line-too-long
    r"""
    Scene()

    The scene contains everything that is needed for radio propagation simulation
    and rendering.

    A scene is a collection of multiple instances of :class:`~sionna.rt.SceneObject` which define
    the geometry and materials of the objects in the scene.
    The scene also includes transmitters (:class:`~sionna.rt.Transmitter`) and receivers (:class:`~sionna.rt.Receiver`)
    for which propagation :class:`~sionna.rt.Paths` or  channel impulse responses (CIRs) can be computed,
    as well as cameras (:class:`~sionna.rt.Camera`) for rendering.

    The only way to instantiate a scene is by calling :meth:`~sionna.rt.Scene,.load_scene`.
    Note that only a single scene can be loaded at a time.

    Example scenes can be loaded as follows:

    .. code-block:: Python

        scene = load_scene(sionna.rt.scene.munich)
        scene.preview()

    .. figure:: ../figures/scene_preview.png
        :align: center
    """

    # Default frequency
    DEFAULT_FREQUENCY = 3.5e9 # Hz

    # This object is a singleton, as it is assumed that only one scene can be
    # loaded at a time.
    _instance = None
    def __new__(cls, *args, **kwargs): # pylint: disable=unused-argument
        if cls._instance is None:
            instance = object.__new__(cls)

            # Creates fields if required.
            # This is done only the first time the instance is created.

            # Transmitters
            instance._transmitters = {}
            # Receivers
            instance._receivers = {}
            # Cameras
            instance._cameras = {}
            # Transmitter antenna array
            instance._tx_array = None
            # Receiver antenna array
            instance._rx_array = None
            # Radio materials
            instance._radio_materials = {}
            # Scene objects
            instance._scene_objects = {}
            # By default, the antenna arrays is applied synthetically
            instance._synthetic_array = True
            # Holds a reference to the interactive preview widget
            instance._preview_widget = None

            # Set the unique instance of the scene
            cls._instance = instance

        return cls._instance

    def __init__(self,
                 env_filename = None,
                 dtype = tf.complex64):

        # If a filename is provided, loads the scene from it.
        # The previous scene is overwritten.
        if env_filename:

            if dtype not in (tf.complex64, tf.complex128):
                msg = "`dtype` must be tf.complex64 or tf.complex64`"
                raise ValueError(msg)
            self._dtype = dtype
            self._rdtype = dtype.real_dtype

            # Clear it all
            self._clear()

            # Set the frequency to the default value
            self.frequency = Scene.DEFAULT_FREQUENCY

            # Populate with ITU materials
            instantiate_itu_materials(self._dtype)

            # Load the scene
            # Keep track of the Mitsuba scene
            if env_filename == "__empty__":
                # Set an empty scene
                self._scene = mi.load_dict({"type": "scene",
                                            "integrator": {
                                                "type": "path",
                                            }})
            else:
                self._scene = mi.load_file(env_filename)

            # Instantiate the solver
            self._solver = Solver(self, dtype=dtype)

            # Solver for coverage map
            self._solver_cm = SolverCoverageMap(self, solver=self._solver,
                                                dtype=dtype)

            # Load the cameras
            self._load_cameras()

            # Load the scene objects
            self._load_scene_objects()

    @property
    def cameras(self):
        """
        `dict` (read-only), { "name", :class:`~sionna.rt.Camera`} : Dictionary
                    of cameras in the scene
        """
        return dict(self._cameras)

    @property
    def frequency(self):
        """
        float : Get/set the carrier frequency [Hz]

            Setting the frequency updates the parameters of frequency-dependent
            radio materials. Defaults to 3.5e9.
        """
        return self._frequency

    @frequency.setter
    def frequency(self, f):
        if f <= 0.0:
            raise ValueError("Frequency must be positive")
        self._frequency = tf.cast(f, self._dtype.real_dtype)
        # Wavelength [m]
        self._wavelength = tf.cast(SPEED_OF_LIGHT/f,
                                    self._dtype.real_dtype)

        # Update radio materials
        for mat in self.radio_materials.values():
            mat.frequency_update(f)

    @property
    def wavelength(self):
        """
        float (read-only) :  Wavelength [m]
        """
        return self._wavelength

    @property
    def synthetic_array(self):
        """
        bool : Get/set if the antenna arrays are applied synthetically.
            Defaults to `False`.
        """
        return self._synthetic_array

    @synthetic_array.setter
    def synthetic_array(self, value):
        if not isinstance(value, bool):
            raise TypeError("'synthetic_array' must be boolean")
        self._synthetic_array = value

    @property
    def transmitters(self):
        # pylint: disable=line-too-long
        """
        `dict` (read-only), { "name", :class:`~sionna.rt.Transmitter`} : Dictionary
            of transmitters in the scene
        """
        return dict(self._transmitters)

    @property
    def receivers(self):
        """
        `dict` (read-only), { "name", :class:`~sionna.rt.Receiver`} : Dictionary
             of receivers in the scene
        """
        return dict(self._receivers)

    @property
    def radio_materials(self):
        # pylint: disable=line-too-long
        """
        `dict` (read-only), { "name", :class:`~sionna.rt.RadioMaterial`} : Dictionary
            of radio materials
        """
        return dict(self._radio_materials)

    @property
    def objects(self):
        # pylint: disable=line-too-long
        """
        `dict` (read-only), { "name", :class:`~sionna.rt.SceneObject`} : Dictionary
            of scene objects
        """
        return dict(self._scene_objects)

    @property
    def tx_array(self):
        """
        :class:`~sionna.rt.AntennaArray` : Get/set the antenna array used by
            all transmitters in the scene. Defaults to `None`.
        """
        return self._tx_array

    @tx_array.setter
    def tx_array(self, array):
        if not isinstance(array, AntennaArray):
            raise TypeError("`array` must be an instance of ``AntennaArray``")
        self._tx_array = array

    @property
    def rx_array(self):
        """
        :class:`~sionna.rt.AntennaArray` : Get/set the antenna array used by
            all receivers in the scene. Defaults to `None`.
        """
        return self._rx_array

    @rx_array.setter
    def rx_array(self, array):
        if not isinstance(array, AntennaArray):
            raise TypeError("`array` must be an instance of ``AntennaArray``")
        self._rx_array = array

    @property
    def size(self):
        """
        [3], tf.float : Get the size of the sceme, i.e., the size of the
        axis-aligned minimum bounding box for the scene
        """
        size = tf.cast(self._scene.bbox().max - self._scene.bbox().min,
                       self._rdtype)
        return size

    @property
    def center(self):
        """
        [3], tf.float : Get the center of the scene
        """
        size = tf.cast((self._scene.bbox().max + self._scene.bbox().min)*0.5,
                       self._rdtype)
        return size

    def get(self, name):
        # pylint: disable=line-too-long
        """
        Returns a scene object, transmitter, receiver, camera, or radio material

        Input
        -----
        name : str
            Name of the item to retrieve

        Output
        ------
        item : :class:`~sionna.rt.SceneObject` | :class:`~sionna.rt.RadioMaterial` | :class:`~sionna.rt.Transmitter` | :class:`~sionna.rt.Receiver` | :class:`~sionna.rt.Camera` | `None`
            Retrieved item. Returns `None` if no corresponding item was found in the scene.
        """
        if name in self._transmitters:
            return self._transmitters[name]
        if name in self._receivers:
            return self._receivers[name]
        if name in self._radio_materials:
            return self._radio_materials[name]
        if name in self._scene_objects:
            return self._scene_objects[name]
        if name in self._cameras:
            return self._cameras[name]
        return None

    def add(self, item):
        """
        Adds a transmitter, receiver, radio material, or camera to the scene.

        If a different item with the same name as ``item`` is already part of the scene,
        an error is raised.

        Input
        ------
        item : :class:`~sionna.rt.Transmitter` | :class:`~sionna.rt.Receiver` | :class:`~sionna.rt.RadioMaterial` | :class:`~sionna.rt.Camera`
            Item to add to the scene
        """
        if ( (not isinstance(item, OrientedObject))
         and (not isinstance(item, RadioMaterial)) ):
            err_msg = "The input must be a Transmitter, Receiver, Camera, or"\
                      " RadioMaterial"
            raise ValueError(err_msg)

        name = item.name
        s_item = self.get(name)
        if s_item is not None:
            if  s_item is not item:
                # In the case of RadioMaterial, the current item with same
                # name could just be a placeholder
                if isinstance(s_item, RadioMaterial)\
                    and s_item.is_placeholder:
                    s_item.relative_permittivity = item.relative_permittivity
                    s_item.conductivity = item.conductivity
                    s_item.frequency_update_callback =\
                        item.frequency_update_callback
                    s_item.trainable = item.trainable
                    s_item.is_placeholder = False
                else:
                    msg = f"Name '{name}' is already used by another item of"\
                           " the scene"
                    raise ValueError(msg)
            else:
                # This item was already added.
                return

        if isinstance(item, Transmitter):
            self._transmitters[name] = item
            item.scene = self
        elif isinstance(item, Receiver):
            self._receivers[name] = item
            item.scene = self
        elif isinstance(item, RadioMaterial):
            self._radio_materials[name] = item
        elif isinstance(item, Camera):
            self._cameras[name] = item
            item.scene = self

    def remove(self, name):
        # pylint: disable=line-too-long
        """
        Removes a transmitter, receiver, camera, or radio material from the
        scene.

        In the case of a radio material, it must not be used by any object of
        the scene.

        Input
        -----
        name : str
            Name of the item to remove
        """
        if not isinstance(name, str):
            raise ValueError("The input should be a string")
        item = self.get(name)

        if item is None:
            pass

        elif isinstance(item, Transmitter):
            del self._transmitters[name]

        elif isinstance(item, Receiver):
            del self._receivers[name]

        elif isinstance(item, Camera):
            del self._cameras[name]

        elif isinstance(item, RadioMaterial):
            if item.is_used:
                msg = f"The radio material '{name}' is used by at least one"\
                        " object"
                raise ValueError(msg)
            del self._radio_materials[name]

        else:
            msg = "Only Transmitters, Receivers, Cameras, or RadioMaterials"\
                  " can be removed"
            raise TypeError(msg)

    def compute_paths(self, max_depth=3, method="stochastic",
                      num_samples=int(1e6), seed=None):
        # pylint: disable=line-too-long
        r"""compute_paths(self, max_depth=3, method="stochastic", num_samples=int(1e6), seed=None)
        Computes propagation paths and their associated transfer matrices

        This function computes propagation paths between the antennas of
        all transmitters and receivers in the current scene.
        For each propagation path :math:`i`, the corresponding transfer matrix
        :math:`\mathbf{T}_i` and delay :math:`\tau_i`, as well as the
        angles of departure :math:`(\theta_{\text{T},i}, \varphi_{\text{T},i})`
        and arrival :math:`(\theta_{\text{R},i}, \varphi_{\text{R},i})` are returned.
        For more detail, see :eq:`H_final`.
        Antenna patterns are not applied and only specular reflections are considered.

        If the scene is configured to use synthetic arrays
        (:attr:`~sionna.rt.Scene.synthetic_array` is `True`), transmitters and receivers
        are modelled as if they had a single antenna located at their
        :attr:`~sionna.rt.Transmitter.position`. The channel responses for each
        individual antenna of the arrays are then computed "synthetically"
        in the :class:`Paths2CIR` layer. This reduces the complexity significantly
        for large arrays.

        Example
        -------
        .. code-block:: Python

            import sionna
            from sionna.rt import load_scene, Camera, Transmitter, Receiver, PlanarArray

            # Load example scene
            scene = load_scene(sionna.rt.scene.munich)

            # Configure antenna array for all transmitters
            scene.tx_array = PlanarArray(num_rows=8,
                                      num_cols=2,
                                      vertical_spacing=0.7,
                                      horizontal_spacing=0.5,
                                      pattern="tr38901",
                                      polarization="VH")

            # Configure antenna array for all receivers
            scene.rx_array = PlanarArray(num_rows=1,
                                      num_cols=1,
                                      vertical_spacing=0.5,
                                      horizontal_spacing=0.5,
                                      pattern="dipole",
                                      polarization="cross")

            # Create transmitter
            tx = Transmitter(name="tx",
                          position=[8.5,21,27],
                          orientation=[0,0,0])
            scene.add(tx)

            # Create a receiver
            rx = Receiver(name="rx",
                       position=[45,90,1.5],
                       orientation=[0,0,0])
            scene.add(rx)

            # TX points towards RX
            tx.look_at(rx)

            # Compute paths
            paths = scene.compute_paths()

            # Open preview showing paths
            scene.preview(paths=paths, resolution=[1000,600]) 

        .. figure:: ../figures/paths_preview.png
            :align: center

        Input
        ------
        max_depth : int
            Maximum depth (i.e., number of bounces) allowed for tracing the
            paths. Defaults to 3.

        method : str ("exhaustive"|"stochastic")
            Ray tracing method to be used.
            The "exhaustive" method uses a
            brute-force approach and considers all possible chains of
            primitives up to a maximum depth. It can therefore only be used
            for very small scenes.
            The "stochastic" method uses a shoot-and-bounce approach to find
            candidate chains of primitives. This method can be applied to very
            large scenes. However, there is no guarantee that all possible
            paths are found.
            Defaults to "stochastic".

        num_samples : int
            Number of random rays to trace in order to generate candidates with
            the stochastic method.
            This number is splitted equally over the different transmitters
            (when using synthetic arrays) or transmit antennas (when not using
            synthetic arrays).
            This parameter is ignored when using the exhaustive method.
            Defaults to 1e6.

        seed: int | `None`
            Seed for the random number generator used for sampling rays with the
            stochastic method. Using the same seed will always result in the
            same paths.
            If set to `None`, then a random seed is used.
            This parameter is ignored when using the exhaustive method.
            Defaults to `None`.

        Output
        ------
        :paths : :class:`~sionna.rt.Paths`
            Simulated paths
        """

        # Check that all is set to compute paths
        self._ready_for_paths_computation()

        # Rotated position of the TX and RX antenna elements in the device
        # local frames.
        # [num_tx, num_tx_ant, 3]
        tx_rel_ant_pos = [self.tx_array.rotated_positions(tx.orientation)
                            for tx in self.transmitters.values()]
        tx_rel_ant_pos = tf.stack(tx_rel_ant_pos, axis=0)
        # [num_rx, num_rx_ant, 3]
        rx_rel_ant_pos = [self.rx_array.rotated_positions(rx.orientation)
                            for rx in self.receivers.values()]
        rx_rel_ant_pos = tf.stack(rx_rel_ant_pos, axis=0)

        # Number of receive antennas (not counting for dual polarization)
        num_tx_ant = tx_rel_ant_pos.shape[1]
        # Number of transmit antennas (not counting for dual polarization)
        num_rx_ant = rx_rel_ant_pos.shape[1]

        # Transmitters and receivers positions
        # [num_tx, 3]
        tx_pos = [tx.position for tx in self.transmitters.values()]
        tx_pos = tf.stack(tx_pos, axis=0)
        # [num_rx, 3]
        rx_pos = [rx.position for rx in self.receivers.values()]
        rx_pos = tf.stack(rx_pos, axis=0)

        # Positions of the endpoints
        if self._synthetic_array:
            # [num_tx, 3]
            sources = tx_pos
            # [num_rx, 3]
            targets = rx_pos
        else:
            # [num_tx, num_tx_ant, 3]
            sources = tf.expand_dims(tx_pos, axis=1) + tx_rel_ant_pos
            # [num_tx*num_tx_ant, 3]
            sources = tf.reshape(sources, [-1, 3])
            # [num_rx, num_rx_ant, 3]
            targets = tf.expand_dims(rx_pos, axis=1) + rx_rel_ant_pos
            # [num_rx*num_rx_ant, 3]
            targets = tf.reshape(targets, [-1, 3])

        # Draw a random seed if required
        if seed is None:
            seed = tf.random.uniform(shape=(), maxval=0x7FFFFFFF,
                                     dtype=tf.int32)

        # Compute the paths using the solver
        # Returns: A Paths object
        paths = self._solver(max_depth, sources, targets, method=method,
                             num_samples=num_samples, seed=seed)

        # If not using synthetic array, then the paths for the different
        # antenna elements were generated by the solver and only reshaping is
        # needed.
        # Otherwise, the phase shifts due to the antenna array need to be
        # applied.
        num_rx = len(self.receivers)
        num_tx = len(self.transmitters)
        max_num_paths = tf.shape(paths.mat_t)[2]
        batch_dims = [num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths]
        if not self._synthetic_array:
            # [num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths, 2, 2]
            paths.mat_t = tf.reshape(paths.mat_t, batch_dims + [2,2])
            # [num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths]
            paths.tau = tf.reshape(paths.tau, batch_dims)
            # [num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths]
            paths.theta_t = tf.reshape(paths.theta_t, batch_dims)
            # [num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths]
            paths.phi_t = tf.reshape(paths.phi_t, batch_dims)
            # [num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths]
            paths.theta_r = tf.reshape(paths.theta_r, batch_dims)
            # [num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths]
            paths.phi_r = tf.reshape(paths.phi_r, batch_dims)

        # Add dummy-dimension for batch_size
        # [1, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths, 2, 2]
        paths.mat_t = tf.expand_dims(paths.mat_t, axis=0)
        # [1, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths]
        paths.tau = tf.expand_dims(paths.tau, axis=0)
        # [1, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths]
        paths.theta_t = tf.expand_dims(paths.theta_t, axis=0)
        # [1, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths]
        paths.phi_t = tf.expand_dims(paths.phi_t, axis=0)
        # [1, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths]
        paths.theta_r = tf.expand_dims(paths.theta_r, axis=0)
        # [1, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths]
        paths.phi_r = tf.expand_dims(paths.phi_r, axis=0)

        return paths

    def coverage_map(self,
                     rx_orientation=(0.,0.,0.),
                     max_depth=3,
                     cm_center=None,
                     cm_orientation=None,
                     cm_size=None,
                     cm_cell_size=(10.,10.),
                     combining_vec=None,
                     precoding_vec=None,
                     num_samples=int(2e6),
                     seed=None):
        # pylint: disable=line-too-long
        r"""coverage_map(rx_orientation=(0.,0.,0.), max_depth=3, cm_center=None, cm_orientation=None, cm_size=None, cm_cell_size=(10.,10.), combining_vec=None, precoding_vec=None, num_samples=int(2e6), seed=None)

        This function computes a coverage map for every transmitter in the scene.

        For a given transmitter, a coverage map associates every point on a surface
        with the sum of the squared amplitudes of the path coefficients :math:`a_i` :eq:`H_final`
        that a receiver with orientation ``rx_orientation`` would observe at this point.
        The receiver is assumed to use the antenna array
        ``scene.rx_array``. If transmitter and/or receiver have multiple antennas, transmit precoding
        and receive combining are applied which are defined by ``precoding_vec`` and
        ``combining_vec``, respectively.

        The surface defining the coverage map is a rectangle centered at
        ``cm_center``, with orientation ``cm_orientation``, and with size
        ``cm_size``. An orientation of (0,0,0) corresponds to
        a coverage map parallel to the XY plane, with surface normal pointing towards
        the :math:`+z` axis. By default, the coverage map
        is parallel to the XY plane, covers all of the scene, and has
        an elevation of :math:`z = 1.5\text{m}`.
        The coverage map is divided into rectangular cells of size
        ``cm_cell_size``. This parameter therefore controls the granularity of the map.

        The output of this function is a real-valued matrix of size ``[num_cells_y, num_cells_x]``,
        for every transmitter, with elements

        .. math::
            b_{y,x} = \sum_i \lvert h_{y,x,i} \rvert^2

        where the sum is over the paths that intersect the cell :math:`(y,x)`,
        :math:`h_{y,x,i}` is the complex-valued channel coefficient
        corresponding to the :math:`i^{\text{th}}` path that intersects the
        cell :math:`(y,x)`, and

        .. math::
            \texttt{num_cells_x} = \bigg\lceil\frac{\texttt{cm_size[0]}}{\texttt{cm_cell_size[0]}} \bigg\rceil\\
            \texttt{num_cells_y} = \bigg\lceil \frac{\texttt{cm_size[1]}}{\texttt{cm_cell_size[1]}} \bigg\rceil.

        The :math:`(y,x)` indices are omitted in the following for clarity.
        Paths are generated by randomly and uniformly sampling ``num_samples``
        rays from the transmitters and by simulating their propagation for up
        to ``max_depth`` interactions with scene objects.
        If ``max_depth`` is set to 0, only the line-of-sight path is considered.

        For every ray :math:`i` intersecting the coverage map cell :math:`(y,x)`, the
        transfer matrix, :math:`\mathbf{T}_i`, and the angles of departure (AoDs)
        :math:`(\theta_{\text{T},i}, \varphi_{\text{T},i})`
        and arrival (AoAs) :math:`(\theta_{\text{R},i}, \varphi_{\text{R},i})`
        are computed. See :eq:`H_final` for further details.

        Given the AoDs :math:`(\theta_{\text{T},i}, \varphi_{\text{T},i})`,
        the orientation of the transmitter :math:`(\alpha_{\text{T}}, \beta_{\text{T}}, \gamma_{\text{T}})`,
        as well as the antenna pattern of the transmitting antenna :math:`\mathbf{C}'_\text{T}(\theta', \varphi')`
        in its local coordinate system (LCS), one can first compute the AoDs :math:`(\theta'_{\text{T},i}, \varphi'_{\text{T},i})` in the
        local coordinate system (LCS) of the transmitter according to :eq:`theta_phi_prime` and then compute
        the antenna response in the global coordinate system (GCS) :math:`\mathbf{C}_\text{T}(\theta_{\text{T},i}, \varphi_{\text{T},i})` as in :eq:`F_prime_2_F`,
        using the rotation matrix :math:`\mathbf{R}=\mathbf{R}(\alpha_{\text{T}}, \beta_{\text{T}}, \gamma_{\text{T}})`.

        The antenna response of the receiver in the GCS :math:`\mathbf{C}_\text{R}(\theta_{\text{R},i}, \varphi_{\text{R},i})` can be computed in an analog fashion
        using the AoAs :math:`(\theta_{\text{R},i}, \varphi_{\text{R},i})`, the orientation
        of the receiver :math:`(\alpha_{\text{R}}, \beta_{\text{R}}, \gamma_{\text{R}})`, and
        the antenna pattern of the receiving antenna :math:`\mathbf{C}'_\text{R}(\theta', \varphi')`
        in its LCS. The receivers of the scene are not used for the computation of the coverage map,
        and the receiver orientation is specified by the ``rx_orientation`` input.

        One then obtains the complex-valued path coefficient :math:`a_i` as :eq:`H_final`:

        .. math::

            a_i = \frac{\lambda}{4\pi} \mathbf{C}_\text{R}(\theta_{\text{R},i}, \varphi_{\text{R},i})^{\mathsf{H}}\mathbf{T}_{i} \mathbf{C}_\text{T}(\theta_{\text{T},i}, \varphi_{\text{T},i}).

        For dual-polarized antennas, a separate channel coefficient for each
        polarization direction is computed by applying the corresponding antenna pattern.

        A "synthetic" array is simulated by adding additional phase shifts that depend on the
        antenna position relative to the position of the transmitter (receiver) as well as the AoDs (AoAs).
        For the :math:`m^\text{th}` transmit antenna and :math:`n^\text{th}` receive antenna, let
        us denote by :math:`\mathbf{d}_{\text{T},m}` and :math:`\mathbf{d}_{\text{R},n}` the relative positions (with respect to
        the positions of the transmitter/receiver) of the pair of antennas
        for which the channel impulse response shall be computed. These can be accessed through the antenna array's property
        :attr:`~sionna.rt.AntennaArray.positions`. Using a plane-wave assumption, the resulting phase shifts
        from these displacements can be computed as

        .. math::

            p_{\text{T}, i,m} &= \frac{2\pi}{\lambda}\hat{\mathbf{r}}(\theta_{\text{T},i}, \varphi_{\text{T},i})^\mathsf{T} \mathbf{d}_{\text{T},m}\\
            p_{\text{R}, i,n} &= \frac{2\pi}{\lambda}\hat{\mathbf{r}}(\theta_{\text{R},i}, \varphi_{\text{R},i})^\mathsf{T} \mathbf{d}_{\text{R},n}.

        The final expression for the path coefficient is

        .. math::

            h_{i,n,m} =  a_i e^{j(p_{\text{T}, i,m} + p_{\text{R}, i,n})}

        for every transmit antenna :math:`m` and receive antenna :math:`n`.
        These coefficients form the complex-valued channel matrix, :math:`\mathbf{H}_i`,
        of size :math:`\texttt{num_rx_ant} \times \texttt{num_tx_ant}`.

        Finally, the coefficient of the equivalent SISO channel is

        .. math::
            h_i =  \mathbf{c}^{\mathsf{H}} \mathbf{H}_i \mathbf{p}

        where :math:`\mathbf{c}` and :math:`\mathbf{p}` are the combining and
        precoding vectors (``combining_vec`` and ``precoding_vec``),
        respectively.

        The coverage map entry :math:`b_{y,x}` is then computed as the
        sum of the squared amplitudes of the channel coefficients
        :math:`h_i` of all paths intersecting the cell :math:`(y,x)`
        of the coverage map.

        Example
        -------
        .. code-block:: Python

            import sionna
            from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver
            scene = load_scene(sionna.rt.scene.munich)

            # Configure antenna array for all transmitters
            scene.tx_array = PlanarArray(num_rows=8,
                                      num_cols=2,
                                      vertical_spacing=0.7,
                                      horizontal_spacing=0.5,
                                      pattern="tr38901",
                                      polarization="VH")

            # Configure antenna array for all receivers
            scene.rx_array = PlanarArray(num_rows=1,
                                      num_cols=1,
                                      vertical_spacing=0.5,
                                      horizontal_spacing=0.5,
                                      pattern="dipole",
                                      polarization="cross")
            # Add a transmitters
            tx = Transmitter(name="tx",
                          position=[8.5,21,27],
                          orientation=[0,0,0])
            scene.add(tx)

            # Compute coverage map
            cm = scene.coverage_map(max_depth=8)

            # Visualize coverage in preview
            scene.preview(coverage_map=cm, resolution=[1000, 600])

        .. figure:: ../figures/coverage_map_preview.png
            :align: center

        

        Input
        ------
        rx_orientation : [3], float
            Orientation of the receiver :math:`(\alpha, \beta, \gamma)`
            specified through three angles corresponding to a 3D rotation
            as defined in :eq:`rotation`. Defaults to :math:`(0,0,0)`.

        max_depth : int
            Maximum depth (i.e., number of bounces) allowed for tracing the
            paths. Defaults to 3.

        cm_center : [3], float | `None`
            Center of the coverage map :math:`(x,y,z)` as three-dimensional
            vector. If set to `None`, the coverage map is centered on the
            center of the scene, except for the elevation :math:`z` that is set
            to 1.5m. Otherwise, ``cm_orientation`` and ``cm_scale`` must also
            not be `None`. Defaults to `None`.

        cm_orientation : [3], float | `None`
            Orientation of the coverage map :math:`(\alpha, \beta, \gamma)`
            specified through three angles corresponding to a 3D rotation
            as defined in :eq:`rotation`.
            An orientation of :math:`(0,0,0)` or `None` corresponds to a
            coverage map that is parallel to the XY plane.
            If not set to `None`, then ``cm_center`` and ``cm_scale`` must also
            not be `None`.
            Defaults to `None`.

        cm_size : [2], float | `None`
            Size of the coverage map [m].
            If set to `None`, then the size of the coverage map is set such that
            it covers the entire scene.
            Otherwise, ``cm_center`` and ``cm_orientation`` must also not be
            `None`. Defaults to `None`.

        cm_cell_size : [2], float
            Size of a cell of the coverage map [m].
            Defaults to :math:`(10,10)`.

        combining_vec : [num_rx_ant], complex | None
            Combining vector.
            If set to `None`, then defaults to
            :math:`\frac{1}{\sqrt{\text{num_rx_ant}}} [1,\dots,1]^{\mathsf{T}}`.

        precoding_vec : [num_tx_ant], complex | None
            Precoding vector.
            If set to `None`, then defaults to
            :math:`\frac{1}{\sqrt{\text{num_tx_ant}}} [1,\dots,1]^{\mathsf{T}}`.

        num_samples : int
            Number of random rays to trace.
            This number is split equally over the different transmitters.
            Defaults to 2e6.

        seed: int | `None`
            Seed for the random number generator used for sampling rays with the
            stochastic method. Using the same seed will always result in the
            same paths. If set to `None`, then a random seed is used.
            Defaults to `None`.

        Output
        ------
        :cm : :class:`~sionna.rt.CoverageMap`
            The coverage maps
        """

        # Check that all is set to compute the coverage map
        self._ready_for_coverage_map()

        # Check the properties of the rectangle defining the coverage map
        if ((cm_center is None)
            and (cm_size is None)
            and (cm_orientation is None)):
            # Default value for center: Center of the scene
            # Default value for the scale: Just enough to cover all the scene
            # with axis-aligned edges of the rectangle
            # [min_x, min_y, min_z]
            scene_min = self._scene.bbox().min
            scene_min = tf.cast(scene_min, self._rdtype)
            # In case of empty scene, bbox min is -inf
            scene_min = tf.where(tf.math.is_inf(scene_min),
                                 -tf.ones_like(scene_min),
                                 scene_min)
            # [max_x, max_y, max_z]
            scene_max = self._scene.bbox().max
            scene_max = tf.cast(scene_max, self._rdtype)
            # In case of empty scene, bbox min is inf
            scene_max = tf.where(tf.math.is_inf(scene_max),
                                 tf.ones_like(scene_max),
                                 scene_max)
            cm_center = tf.cast([(scene_min[0] + scene_max[0])*0.5,
                                 (scene_min[1] + scene_max[1])*0.5,
                                 1.5], dtype=self._rdtype)
            cm_size = tf.cast([(scene_max[0] - scene_min[0]),
                               (scene_max[1] - scene_min[1])],
                                dtype=self._rdtype)
            # Set the orientation to default value
            cm_orientation = tf.zeros([3], dtype=self._rdtype)
        elif ((cm_center is None)
              or (cm_size is None)
              or (cm_orientation is None)):
            raise ValueError("If one of `cm_center`, `cm_orientation`,"\
                             " or `cm_size` is not None, then all of them"\
                             " must not be None")
        else:
            cm_center = tf.cast(cm_center, self._rdtype)
            cm_orientation = tf.cast(cm_orientation, self._rdtype)
            cm_size = tf.cast(cm_size, self._rdtype)

        # Check and initialize the combining and precoding vector
        if combining_vec is None:
            combining_vec = tf.ones([self.rx_array.num_ant], self._dtype)
            combining_vec /= tf.sqrt(tf.cast(self.rx_array.num_ant,
                                             self._dtype))
        else:
            combining_vec = tf.cast(combining_vec, self._dtype)
        if precoding_vec is None:
            num_tx = len(self.transmitters)
            precoding_vec = tf.ones([num_tx, self.tx_array.num_ant],
                                    self._dtype)
            precoding_vec /= tf.sqrt(tf.cast(self.tx_array.num_ant,
                                             self._dtype))
        else:
            precoding_vec = tf.cast(precoding_vec, self._dtype)
            precoding_vec = expand_to_rank(precoding_vec, 2, 0)

        # [3]
        rx_orientation = tf.cast(rx_orientation, self._rdtype)

        # Draw a random seed if required
        if seed is None:
            seed = tf.random.uniform(shape=(), maxval=0x7FFFFFFF,
                                     dtype=tf.int32)

        # Compute the coverage map using the solver
        # [num_sources, num_cells_x, num_cells_y]
        output = self._solver_cm(max_depth=max_depth,
                                 rx_orientation=rx_orientation,
                                 cm_center=cm_center,
                                 cm_orientation=cm_orientation,
                                 cm_size=cm_size,
                                 cm_cell_size=cm_cell_size,
                                 combining_vec=combining_vec,
                                 precoding_vec=precoding_vec,
                                 num_samples=num_samples,
                                 seed=seed)

        return output

    def preview(self, paths=None, show_paths=True, show_devices=True, show_orientations=False,
                coverage_map=None, cm_tx=0,
                cm_vmin=None, cm_vmax=None,
                resolution=(655, 500), fov=45, background='#ffffff'):
        # pylint: disable=line-too-long
        r"""preview(paths=None, show_paths=True, show_devices=True, coverage_map=None, cm_tx=0, cm_vmin=None, cm_vmax=None, resolution=(655, 500), fov=45, background='#ffffff')

        In an interactive notebook environment, opens an interactive 3D
        viewer of the scene.

        The returned value of this method must be the last line of
        the cell so that it is displayed. For example:

        .. code-block:: Python

            fig = scene.preview()
            # ...
            fig

        Or simply:

        .. code-block:: Python

            scene.preview()

        Color coding:

        * Green: Receiver
        * Blue: Transmitter

        Controls:

        * Mouse left: Rotate
        * Scroll wheel: Zoom
        * Mouse right: Move

        Input
        -----
        paths : :class:`~sionna.rt.Paths` | `None`
            Simulated paths generated by
            :meth:`~sionna.rt.Scene.compute_paths()` or `None`.
            If `None`, only the scene is rendered.
            Defaults to `None`.

        show_paths : bool
            If `paths` is not `None`, shows the paths.
            Defaults to `True`.

        show_devices : bool
            If `paths` is not `None`, shows the radio devices.
            Defaults to `True`.

        show_orientations : bool
            If `show_devices` is `True`, shows the transmitters' orientations.
            Defaults to `False`.

        coverage_map : :class:`~sionna.rt.CoverageMap` | `None`
            An optional coverage map to overlay in the scene for visualization.
            Defaults to `None`.

        cm_tx : int | str
            When `coverage_map` is specified, controls which of the transmitters
            to display the coverage map for. Either the transmitter's name
            or index can be given.
            Defaults to `0`.

        cm_vmin, cm_vmax: floot | None
            For coverage map visualization, defines the range of path gains that
            the colormap covers.
            If set to None, then covers the complete range.
            Defaults to `None`.

        resolution : [2], int
            Size of the viewer figure.
            Defaults to `[655, 500]`.

        fov : float
            Field of view, in degrees.
            Defaults to 45°.

        background : str
            Background color in hex format prefixed by '#'.
            Defaults to '#ffffff' (white).

        """
        if (self._preview_widget is not None) and (resolution is not None):
            assert isinstance(resolution, (tuple, list)) and len(resolution) == 2
            if tuple(resolution) != self._preview_widget.resolution():
                # User requested a different rendering resolution, create
                # a new viewer from scratch to match it.
                self._preview_widget = None

        # Cache the render widget so that we don't need to re-create it
        # every time
        fig = self._preview_widget
        needs_reset = fig is not None
        if needs_reset:
            fig.reset()
        else:
            fig = InteractiveDisplay(scene=self,
                                     resolution=resolution,
                                     fov=fov,
                                     background=background)
            self._preview_widget = fig

        # Show paths and devices, if required
        if show_paths and (paths is not None):
            fig.plot_paths(paths)
        if show_devices:
            fig.plot_radio_devices(show_orientations=show_orientations)
        if coverage_map is not None:
            fig.plot_coverage_map(
                coverage_map, tx=cm_tx, db_scale=True,
                vmin=cm_vmin, vmax=cm_vmax)

        # Update the camera state
        if not needs_reset:
            fig.center_view()

        return fig

    def render(self, camera, paths=None, show_paths=True, show_devices=True,
               coverage_map=None, cm_tx=0,
               cm_vmin=None, cm_vmax=None, cm_show_color_bar=True,
               num_samples=512, resolution=(655, 500), fov=45):
        # pylint: disable=line-too-long
        r"""render(camera, paths=None, show_paths=True, show_devices=True, coverage_map=None, cm_tx=0, cm_vmin=None, cm_vmax=None, cm_show_color_bar=True, num_samples=512, resolution=(655, 500), fov=45)

        Renders the scene from the viewpoint of a camera or the interactive
        viewer

        Input
        ------
        camera : str | :class:`~sionna.rt.Camera`
            The name or instance of a :class:`~sionna.rt.Camera`.
            If an interactive viewer was opened with
            :meth:`~sionna.rt.Scene.preview()`, set to `"preview"` to use its
            viewpoint.

        paths : :class:`~sionna.rt.Paths` | `None`
            Simulated paths generated by
            :meth:`~sionna.rt.Scene.compute_paths()` or `None`.
            If `None`, only the scene is rendered.
            Defaults to `None`.

        show_paths : bool
            If `paths` is not `None`, shows the paths.
            Defaults to `True`.

        show_devices : bool
            If `paths` is not `None`, shows the radio devices.
            Defaults to `True`.

        coverage_map : :class:`~sionna.rt.CoverageMap` | `None`
            An optional coverage map to overlay in the scene for visualization.
            Defaults to `None`.

        cm_tx : int | str
            When `coverage_map` is specified, controls which of the transmitters
            to display the coverage map for. Either the transmitter's name
            or index can be given.
            Defaults to `0`.

        cm_vmin, cm_vmax: floot | None
            For coverage map visualization, defines the range of path gains that
            the colormap covers.
            If set to None, then covers the complete range.
            Defaults to `None`.

        cm_show_color_bar: bool
            For coverage map visualization, show the color bar describing the
            color mapping used next to the rendering.
            Defaults to `True`.

        num_samples : int
            Number of rays thrown per pixel.
            Defaults to 512.

        resolution : [2], int
            Size of the rendered figure.
            Defaults to `[655, 500]`.

        fov : float
            Field of view, in degrees.
            Defaults to 45°.

        Output
        -------
        : :class:`~matplotlib.pyplot.Figure`
            Rendered image
        """

        image = render(scene=self,
                       camera=camera,
                       paths=paths,
                       show_paths=show_paths,
                       show_devices=show_devices,
                       coverage_map=coverage_map,
                       cm_tx=cm_tx,
                       cm_db_scale=True,
                       cm_vmin=cm_vmin,
                       cm_vmax=cm_vmax,
                       num_samples=num_samples,
                       resolution=resolution,
                       fov=fov)

        to_show = image.convert(component_format=mi.Struct.Type.UInt8,
                                srgb_gamma=True)

        show_color_bar = (coverage_map is not None) and cm_show_color_bar

        if show_color_bar:
            aspect = image.width()*1.06 / image.height()
            fig, ax = plt.subplots(1, 2,
                                   gridspec_kw={'width_ratios': [0.97, 0.03]},
                                   figsize=(aspect * 6, 6))
            im_ax = ax[0]
        else:
            aspect = image.width() / image.height()
            fig, ax = plt.subplots(1, 1, figsize=(aspect * 6, 6))
            im_ax = ax

        im_ax.imshow(to_show)

        if show_color_bar:
            _, normalizer, color_map = coverage_map_color_mapping(
                coverage_map[cm_tx, :, :].numpy(), db_scale=True,
                vmin=cm_vmin, vmax=cm_vmax)
            mappable = matplotlib.cm.ScalarMappable(
                norm=normalizer, cmap=color_map)

            cax = ax[1]
            cax.set_title('dB')
            fig.colorbar(mappable, cax=cax)

        # Remove axes and margins
        im_ax.axis('off')
        fig.tight_layout()
        return fig

    def render_to_file(self, camera, filename, paths=None, show_paths=True, show_devices=True,
                       coverage_map=None, cm_tx=0, cm_db_scale=True,
                       cm_vmin=None, cm_vmax=None,
                       num_samples=512, resolution=(655, 500), fov=45):
        # pylint: disable=line-too-long
        r"""render_to_file(camera, filename, paths=None, show_paths=True, show_devices=True, coverage_map=None, cm_tx=0, cm_db_scale=True, cm_vmin=None, cm_vmax=None, num_samples=512, resolution=(655, 500), fov=45)

        Renders the scene from the viewpoint of a camera or the interactive
        viewer, and saves the resulting image

        Input
        ------
        camera : str | :class:`~sionna.rt.Camera`
            The name or instance of a :class:`~sionna.rt.Camera`.
            If an interactive viewer was opened with
            :meth:`~sionna.rt.Scene.preview()`, set to `"preview"` to use its
            viewpoint.

        filename : str
            Filename for saving the rendered image, e.g., "my_scene.png"

        paths : :class:`~sionna.rt.Paths` | `None`
            Simulated paths generated by
            :meth:`~sionna.rt.Scene.compute_paths()` or `None`.
            If `None`, only the scene is rendered.
            Defaults to `None`.

        show_paths : bool
            If `paths` is not `None`, shows the paths.
            Defaults to `True`.

        show_devices : bool
            If `paths` is not `None`, shows the radio devices.
            Defaults to `True`.

        coverage_map : :class:`~sionna.rt.CoverageMap` | `None`
            An optional coverage map to overlay in the scene for visualization.
            Defaults to `None`.

        cm_tx : int | str
            When `coverage_map` is specified, controls which of the transmitters
            to display the coverage map for. Either the transmitter's name
            or index can be given.
            Defaults to `0`.

        cm_db_scale: bool
            Use logarithmic scale for coverage map visualization, i.e. the
            coverage values are mapped with:
            :math:`y = 10 \cdot \log_{10}(x)`.
            Defaults to `True`.

        cm_vmin, cm_vmax: floot | None
            For coverage map visualization, defines the range of path gains that
            the colormap covers.
            If set to None, then covers the complete range.
            Defaults to `None`.

        num_samples : int
            Number of rays thrown per pixel.
            Defaults to 512.

        resolution : [2], int
            Size of the rendered figure.
            Defaults to `[655, 500]`.

        fov : float
            Field of view, in degrees.
            Defaults to 45°.

        """
        image = render(scene=self,
                       camera=camera,
                       paths=paths,
                       show_paths=show_paths,
                       show_devices=show_devices,
                       coverage_map=coverage_map,
                       cm_tx=cm_tx,
                       cm_db_scale=cm_db_scale,
                       cm_vmin=cm_vmin,
                       cm_vmax=cm_vmax,
                       num_samples=num_samples,
                       resolution=resolution,
                       fov=fov)

        ext = os.path.splitext(filename)[1].lower()
        if ext in ('.jpg', '.jpeg', '.ppm',):
            image = image.convert(component_format=mi.Struct.Type.UInt8,
                                  pixel_format=mi.Bitmap.PixelFormat.RGB,
                                  srgb_gamma=True)
        elif ext in ('.png', '.tga' '.bmp'):
            image = image.convert(component_format=mi.Struct.Type.UInt8,
                                  srgb_gamma=True)
        image.write(filename)

    ##############################################
    # Internal methods.
    # Should not be appear in the user
    # documentation
    ##############################################

    @property
    def mi_scene(self):
        """
        :class:`~mitsuba.Scene` : Get the Mitsuba scene
        """
        return self._scene

    @property
    def preview_widget(self):
        """
        :class:`~sionna.rt.InteractiveDisplay` : Get the preview widget
        """
        return self._preview_widget

    def _clear(self):
        r"""
        Clear everything.
        Should be called when a new scene is loaded.
        """

        self._transmitters.clear()
        self._receivers.clear()
        self._cameras.clear()
        self._radio_materials.clear()
        self._scene_objects.clear()
        self._tx_array = None
        self._rx_array = None
        self._preview_widget = None

    def _ready_for_paths_computation(self):
        r"""
        Check that all is set for paths computation.
        If not, raises an exception with the appropriate error message.
        """
        if not self._rx_array:
            raise ValueError("Receiver array not set.")

        if not self._tx_array:
            raise ValueError("Transmitter array not set.")

        if len(self._transmitters) == 0:
            raise ValueError("No transmitter defined.")

        if len(self._receivers) == 0:
            raise ValueError("No receiver defined.")

        # Check that all scene objects have a radio material
        for obj in self.objects.values():
            mat = obj.radio_material
            if mat is None:
                msg = f"Scene object {obj.name} has no material set."
                raise ValueError(msg)
            else:
                # Check that the material is well-defined
                if not mat.well_defined:
                    msg = f"Material '{mat.name}' is used by the object "\
                           f" '{obj.name}' but is not well-defined."
                    raise ValueError(msg)
                # Check that the material is not a placeholder
                if mat.is_placeholder:
                    msg = f"Material '{mat.name}' is used by the object "\
                           f" '{obj.name}' but not defined."
                    raise ValueError(msg)

    def _ready_for_coverage_map(self):
        r"""
        Check that all is set for coverage map.
        If not, raises an exception with the appropriate error message.
        """
        if not self._rx_array:
            raise ValueError("Receiver array not set.")

        if not self._tx_array:
            raise ValueError("Transmitter array not set.")

        if len(self._transmitters) == 0:
            raise ValueError("No transmitter defined.")

        # Check that all scene objects have a radio material
        for obj in self.objects.values():
            mat = obj.radio_material
            if mat is None:
                msg = f"Scene object {obj.name} has no material set."
                raise ValueError(msg)
            else:
                # Check that the material is well-defined
                if not mat.well_defined:
                    msg = f"Material '{mat.name}' is used by the object "\
                           f" '{obj.name}' but is not well-defined."
                    raise ValueError(msg)
                # Check that the material is not a placeholder
                if mat.is_placeholder:
                    msg = f"Material '{mat.name}' is used by the object "\
                           f" '{obj.name}' but not defined."
                    raise ValueError(msg)

    def _load_cameras(self):
        """
        Load the camera(s) available in the scene
        """
        for i, mi_cam in enumerate(self._scene.sensors()):
            # Extract the transformation paramters
            transform = mi.traverse(mi_cam)['to_world']
            position = Camera.world_to_position(transform)
            orientation = Camera.world_to_angles(transform)

            # Create the camera
            name = f"scene-cam-{i}"
            new_cam = Camera(name=name,
                             position=position,
                             orientation=orientation)
            new_cam.scene = self

            self._cameras[name] = new_cam

    def _load_scene_objects(self):
        """
        Load the scene objects availabe in the scene
        """
        # Parse all shapes in the scene
        scene = self._scene
        for s in scene.shapes():
            # Only meshes are handled
            if not isinstance(s, mi.Mesh):
                raise TypeError('Only triangle meshes are supported')

            # Setup the material
            mat_name = s.bsdf().id()
            if mat_name.startswith("mat-"):
                mat_name = mat_name[4:]
            mat = self.get(mat_name)
            if (mat is not None) and (not isinstance(mat, RadioMaterial)):
                raise ValueError(f"Name'{name}' already used by another item")
            elif mat is None:
                # If the radio material does not exist, then a placeholder is
                # used.
                mat = RadioMaterial(mat_name)
                mat.is_placeholder = True
                self._radio_materials[mat_name] = mat

            # Instantiate the scene objects
            name = s.id()
            if name.startswith('mesh-'):
                name = name[5:]
            if self._is_name_used(name):
                raise ValueError(f"Name'{name}' already used by another item")
            obj = SceneObject(name)
            obj.scene = self
            obj.radio_material = mat_name

            self._scene_objects[name] = obj

    def _is_name_used(self, name):
        """
        Returns `True` if ``name`` is used by a scene object, a transmitter,
        or a receiver.
        """
        used = ((name in self._transmitters)
             or (name in self._receivers)
             or (name in self._radio_materials)
             or (name in self._scene_objects))
        return used


def load_scene(filename=None, dtype=tf.complex64):
    # pylint: disable=line-too-long
    r"""
    Load a scene from file

    Note that only one scene can be loaded at a time.

    Input
    -----
    filename : str
        Name of a valid scene file. Sionna uses the simple XML-based format
        from `Mitsuba 3 <https://mitsuba.readthedocs.io/en/stable/src/key_topics/scene_format.html>`_.
        Defaults to `None` for which an empty scene is created.

    dtype : tf.complex
        Dtype used for all internal computations and outputs.
        Defaults to `tf.complex64`.

    Output
    ------
    scene : :class:`~sionna.rt.Scene`
        Reference to the current scene
    """
    # Create empty scene using the reserved filename "__empty__"
    if filename is None:
        filename = "__empty__"
    return Scene(filename, dtype=dtype)

#
# Module variables for example scene files
#
floor_wall = str(files(scenes).joinpath("floor_wall/floor_wall.xml"))
# pylint: disable=C0301
"""
Example scene containing a ground plane and a vertical wall
(`Blender file <https://drive.google.com/file/d/1djXBj3VYLT4_bQpmp4vR6o6agGmv_p1F/view?usp=share_link>`__).

.. figure:: ../figures/floor_wall.png
   :align: center
"""

# pylint: disable=C0301
simple_street_canyon = str(files(scenes).joinpath("simple_street_canyon/simple_street_canyon.xml"))
"""
Example scene containing a few rectangular building blocks and a ground plane
(`Blender file <https://drive.google.com/file/d/1_1nsLtSC8cy1QfRHAN_JetT3rPP21tNb/view?usp=share_link>`__).

.. figure:: ../figures/street_canyon.png
   :align: center
"""

etoile = str(files(scenes).joinpath("etoile/etoile.xml"))
# pylint: disable=C0301
"""
Example scene containing the area around the Arc de Triomphe in Paris
(`Blender file <https://drive.google.com/file/d/1bamQ67lLGZHTfNmcVajQDmq2oiSY8FEn/view?usp=share_link>`__).
The scene was created with data downloaded from `OpenStreetMap <https://www.openstreetmap.org>`_ and
the help of `Blender <https://www.blender.org>`_ and the `Blender-OSM <https://github.com/vvoovv/blender-osm>`_
and `Mitsuba Blender <https://github.com/mitsuba-renderer/mitsuba-blender>`_ add-ons.
The data is licensed under the `Open Data Commons Open Database License (ODbL) <https://openstreetmap.org/copyright>`_.

.. figure:: ../figures/etoile.png
   :align: center
"""

munich = str(files(scenes).joinpath("munich/munich.xml"))
# pylint: disable=C0301
"""
Example scene containing the area around the Frauenkirche in Munich
(`Blender file <https://drive.google.com/file/d/15WrvMGrPWsoVKYvDG6Ab7btq-ktTCGR1/view?usp=share_link>`__).
The scene was created with data downloaded from `OpenStreetMap <https://www.openstreetmap.org>`_ and
the help of `Blender <https://www.blender.org>`_ and the `Blender-OSM <https://github.com/vvoovv/blender-osm>`_
and `Mitsuba Blender <https://github.com/mitsuba-renderer/mitsuba-blender>`_ add-ons.
The data is licensed under the `Open Data Commons Open Database License (ODbL) <https://openstreetmap.org/copyright>`_.

.. figure:: ../figures/munich.png
   :align: center
"""
