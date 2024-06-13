#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import drjit as dr

from .antenna_array import AntennaArray
from .camera import Camera
from .radio_device import RadioDevice
from sionna.constants import SPEED_OF_LIGHT, PI
from .itu_materials import instantiate_itu_materials
from .radio_material import RadioMaterial
from .receiver import Receiver
from .ris import RIS
from .scene_object import SceneObject
from .solver_paths import SolverPaths, PathsTmpData
from .solver_cm import SolverCoverageMap
from .transmitter import Transmitter
from .previewer import InteractiveDisplay
from .renderer import render, coverage_map_color_mapping
from .utils import expand_to_rank
from .paths import Paths
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
    for which propagation :class:`~sionna.rt.Paths`, channel impulse responses (CIRs) or coverage maps (:class:`~sionna.rt.CoverageMap`) can be computed,
    as well as cameras (:class:`~sionna.rt.Camera`) for rendering.

    The only way to instantiate a scene is by calling :meth:`~sionna.rt.Scene,.load_scene()`.
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
            # Reconfigurable intelligent surfaces
            instance._ris = {}
            # Reconfigurable intelligent surfaces
            instance._ris = {}
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

            # By default, no callable is used for radio materials
            cls._instance._radio_material_callable = None

            # By default, no callable is used for scattering patterns
            cls._instance._scattering_pattern_callable = None

        return cls._instance

    def __init__(self, env_filename = None, dtype = tf.complex64):

        # If a filename is provided, loads the scene from it.
        # The previous scene is overwritten.
        if env_filename:

            if dtype not in (tf.complex64, tf.complex128):
                msg = "`dtype` must be tf.complex64 or tf.complex128`"
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
            self._scene_params = mi.traverse(self._scene)

            # Instantiate the solver
            self._solver_paths = SolverPaths(self, dtype=dtype)

            # Solver for coverage map
            self._solver_cm = SolverCoverageMap(self, solver=self._solver_paths,
                                                dtype=dtype)

            # Load the cameras
            self._load_cameras()

            # Load the scene objects
            self._load_scene_objects()

            # By default, no callable is used for radio materials
            self.radio_material_callable = None

            # By default, no callable is used for scattering patterns
            self._scattering_pattern_callable = None

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
            mat.frequency_update()

    @property
    def wavelength(self):
        """
        float (read-only) :  Wavelength [m]
        """
        return self._wavelength

    @property
    def wavenumber(self):
        r"""
        float (read-only) :  Wavenumber :math:`k=2\pi/\lambda` [m^-1]
        """
        return tf.cast(2*PI, self._dtype.real_dtype)/self._wavelength

    @property
    def synthetic_array(self):
        """
        bool : Get/set if the antenna arrays are applied synthetically.
            Defaults to `True`.
        """
        return self._synthetic_array

    @synthetic_array.setter
    def synthetic_array(self, value):
        if not isinstance(value, bool):
            raise TypeError("'synthetic_array' must be boolean")
        self._synthetic_array = value

    @property
    def dtype(self):
        r"""
        `tf.complex64 | tf.complex128` : Datatype used in tensors
        """
        return self._dtype

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
    def ris(self):
        """
        `dict` (read-only), { "name", :class:`~sionna.rt.RIS`} : Dictionary
             of reconfigurable intelligent surfaces (RIS) in the scene
        """
        return dict(self._ris)

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
        [3], tf.float : Get the size of the scene, i.e., the size of the
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
        item : :class:`~sionna.rt.SceneObject` | :class:`~sionna.rt.RadioMaterial` | :class:`~sionna.rt.Transmitter` | :class:`~sionna.rt.Receiver` | :class:`~sionna.rt.RIS` | :class:`~sionna.rt.Camera` | `None`
            Retrieved item. Returns `None` if no corresponding item was found in the scene.
        """
        if name in self._transmitters:
            return self._transmitters[name]
        if name in self._receivers:
            return self._receivers[name]
        if name in self._ris:
            return self._ris[name]
        if name in self._ris:
            return self._ris[name]
        if name in self._radio_materials:
            return self._radio_materials[name]
        if name in self._scene_objects:
            return self._scene_objects[name]
        if name in self._cameras:
            return self._cameras[name]
        return None

    def add(self, item):
        # pylint: disable=line-too-long
        # pylint: disable=line-too-long
        """
        Adds a transmitter, receiver, RIS, radio material, or camera to the scene.

        If a different item with the same name as ``item`` is already part of the scene,
        an error is raised.

        Input
        ------
        item : :class:`~sionna.rt.Transmitter` | :class:`~sionna.rt.Receiver` | :class:`~sionna.rt.RIS` | :class:`~sionna.rt.RadioMaterial` | :class:`~sionna.rt.Camera`
            Item to add to the scene
        """
        if ( (not isinstance(item, Camera))
         and (not isinstance(item, RadioDevice))
         and (not isinstance(item, RadioMaterial)) ):
            err_msg = "The input must be a Transmitter, Receiver, RIS, Camera, or"\
                      " RadioMaterial"
            raise ValueError(err_msg)

        name = item.name
        s_item = self.get(name)
        if s_item is not None:
            if  s_item is not item:
                # In the case of RadioMaterial, the current item with same
                # name could just be a placeholder
                if (isinstance(s_item, RadioMaterial)
                    and isinstance(item, RadioMaterial)
                    and s_item.is_placeholder):
                    s_item.assign(item)
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
        elif isinstance(item, RIS):
            self._ris[name] = item
            # Manually assign object_id to each RIS
            if len(self.objects)>0:
                max_id = max(obj.object_id for obj in self.objects.values())
            else:
                max_id=0
            max_id += len(self._ris)
            item.object_id = max_id
            # Set scene propety and radio material
            item.scene = self
            item.radio_material = "itu_metal"
        elif isinstance(item, RadioMaterial):
            self._radio_materials[name] = item
            item.frequency_update()
        elif isinstance(item, Camera):
            self._cameras[name] = item
            item.scene = self

    def remove(self, name):
        # pylint: disable=line-too-long
        """
        Removes a transmitter, receiver, RIS, camera, or radio material from the
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

        elif isinstance(item, RIS):
            del self._ris[name]

        elif isinstance(item, RIS):
            del self._ris[name]

        elif isinstance(item, Camera):
            del self._cameras[name]

        elif isinstance(item, RadioMaterial):
            if item.is_used:
                msg = f"The radio material '{name}' is used by at least one"\
                        " object"
                raise ValueError(msg)
            del self._radio_materials[name]

        else:
            msg = "Only Transmitters, Receivers, RIS, Cameras, or RadioMaterials"\
                  " can be removed"
            raise TypeError(msg)


    def trace_paths(self, max_depth=3, method="fibonacci", num_samples=int(1e6),
                    los=True, reflection=True, diffraction=False,
                    scattering=False, ris=True, scat_keep_prob=0.001,
                    edge_diffraction=False, check_scene=True):
        # pylint: disable=line-too-long
        r"""
        Computes the trajectories of the paths by shooting rays

        The EM fields corresponding to the traced paths are not computed.
        They can be computed using :meth:`~sionna.rt.Scene.compute_fields()`:

        .. code-block:: Python

            traced_paths = scene.trace_paths()
            paths = scene.compute_fields(*traced_paths)

        Path tracing is independent of the radio materials, antenna patterns,
        and radio device orientations.
        Therefore, a set of traced paths could be reused for different values
        of these quantities, e.g., to calibrate the ray tracer.
        This can enable significant resource savings as path tracing is
        typically significantly more resource-intensive than field computation.

        Note that :meth:`~sionna.rt.Scene.compute_paths()` does both path tracing and
        field computation.

        Input
        ------
        max_depth : int
            Maximum depth (i.e., number of interaction with objects in the scene)
            allowed for tracing the paths.
            Defaults to 3.

        method : str ("exhaustive"|"fibonacci")
            Method to be used to list candidate paths.
            The "exhaustive" method tests all possible combination of primitives as
            paths. This method is not compatible with scattering.
            The "fibonacci" method uses a shoot-and-bounce approach to find
            candidate chains of primitives. Initial ray directions are arranged
            in a Fibonacci lattice on the unit sphere. This method can be
            applied to very large scenes. However, there is no guarantee that
            all possible paths are found.
            Defaults to "fibonacci".

        num_samples: int
            Number of random rays to trace in order to generate candidates.
            A large sample count may exhaust GPU memory.
            Defaults to 1e6. Only needed if ``method`` is "fibonacci".

        los : bool
            If set to `True`, then the LoS paths are computed.
            Defaults to `True`.

        reflection : bool
            If set to `True`, then the reflected paths are computed.
            Defaults to `True`.

        diffraction : bool
            If set to `True`, then the diffracted paths are computed.
            Defaults to `False`.

        scattering : bool
            If set to `True`, then the scattered paths are computed.
            Only works with the Fibonacci method.
            Defaults to `False`.

        ris : bool
            If set to `True`, then the paths involving RIS are computed.
            Defaults to `True`.

        ris : bool
            If set to `True`, then the paths involving RIS are computed.
            Defaults to `True`.

        scat_keep_prob : float
            Probability with which to keep scattered paths.
            This is helpful to reduce the number of scattered paths computed,
            which might be prohibitively high in some setup.
            Must be in the range (0,1).
            Defaults to 0.001.

        edge_diffraction : bool
            If set to `False`, only diffraction on wedges, i.e., edges that
            connect two primitives, is considered.
            Defaults to `False`.

        check_scene : bool
            If set to `True`, checks that the scene is well configured before
            computing the paths. This can add a significant overhead.
            Defaults to `True`.

        Output
        -------
        spec_paths : :class:`~sionna.rt.Paths`
            Computed specular paths

        diff_paths : :class:`~sionna.rt.Paths`
            Computed diffracted paths

        scat_paths : :class:`~sionna.rt.Paths`
            Computed scattered paths

        ris_paths : :class:`~sionna.rt.Paths`
            Computed paths involving RIS

        ris_paths : :class:`~sionna.rt.Paths`
            Computed paths involving RIS

        spec_paths_tmp : :class:`~sionna.rt.PathsTmpData`
            Additional data required to compute the EM fields of the specular
            paths

        diff_paths_tmp : :class:`~sionna.rt.PathsTmpData`
            Additional data required to compute the EM fields of the diffracted
            paths

        scat_paths_tmp : :class:`~sionna.rt.PathsTmpData`
            Additional data required to compute the EM fields of the scattered
            paths

        ris_paths_tmp : :class:`~sionna.rt.PathsTmpData`
            Additional data required to compute the EM fields of the paths
            involving RIS

        ris_paths_tmp : :class:`~sionna.rt.PathsTmpData`
            Additional data required to compute the EM fields of the paths
            involving RIS
        """

        if scat_keep_prob < 0. or scat_keep_prob > 1.:
            msg = "The parameter `scat_keep_prob` must be in the range (0,1)"
            raise ValueError(msg)

        # Check that all is set to compute paths
        if check_scene:
            self._check_scene(False)

        # Trace the paths
        paths = self._solver_paths.trace_paths(max_depth,
                                               method=method,
                                               num_samples=num_samples,
                                               los=los, reflection=reflection,
                                               diffraction=diffraction,
                                               scattering=scattering,
                                               ris=ris,
                                               scat_keep_prob=scat_keep_prob,
                                               edge_diffraction=edge_diffraction)

        return paths

    def compute_fields(self, spec_paths, diff_paths, scat_paths, ris_paths,
                       spec_paths_tmp, diff_paths_tmp, scat_paths_tmp,
                       ris_paths_tmp, check_scene=True, scat_random_phases=True,
                       testing=False):
        r"""compute_fields(self, spec_paths, diff_paths, scat_paths, spec_paths_tmp, diff_paths_tmp, scat_paths_tmp, check_scene=True, scat_random_phases=True)
        Computes the EM fields corresponding to traced paths

        Paths can be traced using :meth:`~sionna.rt.Scene.trace_paths()`.
        This method can then be used to finalize the paths calculation by
        computing the corresponding fields:

        .. code-block:: Python

            traced_paths = scene.trace_paths()
            paths = scene.compute_fields(*traced_paths)

        Paths tracing is independent from the radio materials, antenna patterns,
        and radio devices orientations.
        Therefore, a set of traced paths could be reused for different values
        of these quantities, e.g., to calibrate the ray tracer.
        This can enable significant resource savings as paths tracing is
        typically significantly more resource-intensive than field computation.

        Note that :meth:`~sionna.rt.Scene.compute_paths()` does both tracing and
        field computation.

        Input
        ------
        spec_paths : :class:`~sionna.rt.Paths`
            Specular paths

        diff_paths : :class:`~sionna.rt.Paths`
            Diffracted paths

        scat_paths : :class:`~sionna.rt.Paths`
            Scattered paths

        ris_paths : :class:`~sionna.rt.Paths`
            Computed paths involving RIS

        ris_paths : :class:`~sionna.rt.Paths`
            Computed paths involving RIS

        spec_paths_tmp : :class:`~sionna.rt.PathsTmpData`
            Additional data required to compute the EM fields of the specular
            paths

        diff_paths_tmp : :class:`~sionna.rt.PathsTmpData`
            Additional data required to compute the EM fields of the diffracted
            paths

        scat_paths_tmp : :class:`~sionna.rt.PathsTmpData`
            Additional data required to compute the EM fields of the scattered
            paths

        ris_paths_tmp : :class:`~sionna.rt.PathsTmpData`
            Additional data required to compute the EM fields of the paths
            involving RIS

        ris_paths_tmp : :class:`~sionna.rt.PathsTmpData`
            Additional data required to compute the EM fields of the paths
            involving RIS

        check_scene : bool
            If set to `True`, checks that the scene is well configured before
            computing the paths. This can add a significant overhead.
            Defaults to `True`.

        scat_random_phases : bool
            If set to `True` and if scattering is enabled, random uniform phase
            shifts are added to the scattered paths.
            Defaults to `True`.

        Output
        -------
        paths : :class:`~sionna.rt.Paths`
            Computed paths
        """

        # Check that all is set to compute paths
        if check_scene:
            self._check_scene(False)

        # Compute the fields and merge the paths
        output = self._solver_paths.compute_fields(spec_paths, diff_paths,
            scat_paths, ris_paths, spec_paths_tmp, diff_paths_tmp,
            scat_paths_tmp, ris_paths_tmp,
            scat_random_phases, testing)
        sources, targets, paths_as_dict = output[:3]
        paths = Paths(sources, targets, self)
        paths.from_dict(paths_as_dict)

        # If the hidden input flag testing is True, additional data
        # is returned which is required for some unit tests
        if testing:
            spec_tmp_as_dict, diff_tmp_as_dict, scat_tmp_as_dict = output[3:]
            spec_tmp = PathsTmpData(sources, targets, self._dtype)
            spec_tmp.from_dict(spec_tmp_as_dict)
            diff_tmp = PathsTmpData(sources, targets, self._dtype)
            diff_tmp.from_dict(diff_tmp_as_dict)
            scat_tmp = PathsTmpData(sources, targets, self._dtype)
            scat_tmp.from_dict(scat_tmp_as_dict)
            paths.spec_tmp = spec_tmp
            paths.diff_tmp = diff_tmp
            paths.scat_tmp = scat_tmp

        # Finalize paths computation
        paths.finalize()

        return paths

    def compute_paths(self, max_depth=3, method="fibonacci",
                      num_samples=int(1e6), los=True, reflection=True,
                      diffraction=False, scattering=False, ris=True,
                      scat_keep_prob=0.001, edge_diffraction=False,
                      check_scene=True, scat_random_phases=True,
                      testing=False):
        # pylint: disable=line-too-long
        r"""
        Computes propagation paths

        This function computes propagation paths between the antennas of
        all transmitters and receivers in the current scene.
        For each propagation path :math:`i`, the corresponding channel coefficient
        :math:`a_i` and delay :math:`\tau_i`, as well as the
        angles of departure :math:`(\theta_{\text{T},i}, \varphi_{\text{T},i})`
        and arrival :math:`(\theta_{\text{R},i}, \varphi_{\text{R},i})` are returned.
        For more detail, see :eq:`H_final`.
        Different propagation phenomena, such as line-of-sight, reflection, diffraction,
        and diffuse scattering can be individually enabled/disabled.

        If the scene is configured to use synthetic arrays
        (:attr:`~sionna.rt.Scene.synthetic_array` is `True`), transmitters and receivers
        are modelled as if they had a single antenna located at their
        :attr:`~sionna.rt.Transmitter.position`. The channel responses for each
        individual antenna of the arrays are then computed "synthetically" by applying
        appropriate phase shifts. This reduces the complexity significantly
        for large arrays. Time evolution of the channel coefficients can be simulated with
        the help of the function :meth:`~sionna.rt.Paths.apply_doppler` of the returned
        :class:`~sionna.rt.Paths` object.

        The path computation consists of two main steps as shown in the below figure.

        .. figure:: ../figures/compute_paths.svg
            :align: center

        For a configured :class:`~sionna.rt.Scene`, the function first traces geometric propagation paths
        using :meth:`~sionna.rt.Scene.trace_paths`. This step is independent of the
        :class:`~sionna.rt.RadioMaterial` of the scene objects as well as the transmitters' and receivers'
        antenna :attr:`~sionna.rt.Antenna.patterns` and  :attr:`~sionna.rt.Transmitter.orientation`,
        but depends on the selected propagation
        phenomena, such as reflection, scattering, and diffraction. The traced paths
        are then converted to EM fields by the function :meth:`~sionna.rt.Scene.compute_fields`.
        The resulting :class:`~sionna.rt.Paths` object can be used to compute channel
        impulse responses via :meth:`~sionna.rt.Paths.cir`. The advantage of separating path tracing
        and field computation is that one can study the impact of different radio materials
        by executing :meth:`~sionna.rt.Scene.compute_fields` multiple times without
        re-tracing the propagation paths. This can for example speed-up the calibration of scene parameters
        by several orders of magnitude.

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

        method : str ("exhaustive"|"fibonacci")
            Ray tracing method to be used.
            The "exhaustive" method tests all possible combinations of primitives.
            This method is not compatible with scattering.
            The "fibonacci" method uses a shoot-and-bounce approach to find
            candidate chains of primitives. Initial ray directions are chosen
            according to a Fibonacci lattice on the unit sphere. This method can be
            applied to very large scenes. However, there is no guarantee that
            all possible paths are found.
            Defaults to "fibonacci".

        num_samples : int
            Number of rays to trace in order to generate candidates with
            the "fibonacci" method.
            This number is split equally among the different transmitters
            (when using synthetic arrays) or transmit antennas (when not using
            synthetic arrays).
            This parameter is ignored when using the exhaustive method.
            Tracing more rays can lead to better precision
            at the cost of increased memory requirements.
            Defaults to 1e6.

        los : bool
            If set to `True`, then the LoS paths are computed.
            Defaults to `True`.

        reflection : bool
            If set to `True`, then the reflected paths are computed.
            Defaults to `True`.

        diffraction : bool
            If set to `True`, then the diffracted paths are computed.
            Defaults to `False`.

        scattering : bool
            If set to `True`, then the scattered paths are computed.
            If set to `True`, then the scattered paths are computed.
            Only works with the Fibonacci method.
            Defaults to `False`.

        ris : bool
            If set to `True`, then paths involving RIS are computed.
            Defaults to `True`.

        scat_keep_prob : float
            Probability with which a scattered path is kept.
            This is helpful to reduce the number of computed scattered
            paths, which might be prohibitively high in some scenes.
            Must be in the range (0,1). Defaults to 0.001.

        edge_diffraction : bool
            If set to `False`, only diffraction on wedges, i.e., edges that
            connect two primitives, is considered.
            Defaults to `False`.

        check_scene : bool
            If set to `True`, checks that the scene is well configured before
            computing the paths. This can add a significant overhead.
            Defaults to `True`.

        scat_random_phases : bool
            If set to `True` and if scattering is enabled, random uniform phase
            shifts are added to the scattered paths.
            Defaults to `True`.

        testing : bool
            If set to `True`, then additional data is returned for testing.
            Defaults to `False`.

        Output
        ------
        :paths : :class:`~sionna.rt.Paths`
            Simulated paths
        """

        # Trace the paths
        traced_paths = self.trace_paths(max_depth, method, num_samples, los,
            reflection, diffraction, scattering, ris, scat_keep_prob,
            edge_diffraction, check_scene)

        # Compute the fields and merge the paths
        # Check scene is not done twice
        paths = self.compute_fields(*traced_paths, False, scat_random_phases,
                                    testing)

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
                     los=True,
                     reflection=True,
                     diffraction=False,
                     scattering=False,
                     ris=True,
                     edge_diffraction=False,
                     check_scene=True):
        # pylint: disable=line-too-long
        r"""
        This function computes a coverage map for every transmitter in the scene.

        For a given transmitter, a coverage map is a rectangular surface with
        arbitrary orientation subdivded
        into rectangular cells of size :math:`\lvert C \rvert = \texttt{cm_cell_size[0]} \times  \texttt{cm_cell_size[1]}`.
        The parameter ``cm_cell_size`` therefore controls the granularity of the map.
        The coverage map associates with every cell :math:`(i,j)` the quantity

        .. math::
            :label: cm_def

            b_{i,j} = \frac{1}{\lvert C \rvert} \int_{C_{i,j}} \lvert h(s) \rvert^2 ds

        where :math:`\lvert h(s) \rvert^2` is the squared amplitude
        of the path coefficients :math:`a_i` at position :math:`s=(x,y)`,
        the integral is over the cell :math:`C_{i,j}`, and
        :math:`ds` is the infinitesimal small surface element
        :math:`ds=dx \cdot dy`.
        The dimension indexed by :math:`i` (:math:`j`) corresponds to the :math:`y\, (x)`-axis of the
        coverage map in its local coordinate system.

        For specularly and diffusely reflected paths, :eq:`cm_def` can be rewritten as an integral over the directions
        of departure of the rays from the transmitter, by substituting :math:`s`
        with the corresponding direction :math:`\omega`:

        .. math::
            b_{i,j} = \frac{1}{\lvert C \rvert} \int_{\Omega} \lvert h\left(s(\omega) \right) \rvert^2 \frac{r(\omega)^2}{\lvert \cos{\alpha(\omega)} \rvert} \mathbb{1}_{\left\{ s(\omega) \in C_{i,j} \right\}} d\omega

        where the integration is over the unit sphere :math:`\Omega`, :math:`r(\omega)` is the length of
        the path with direction of departure :math:`\omega`, :math:`s(\omega)` is the point
        where the path with direction of departure :math:`\omega` intersects the coverage map,
        :math:`\alpha(\omega)` is the angle between the coverage map normal and the direction of arrival
        of the path with direction of departure :math:`\omega`,
        and :math:`\mathbb{1}_{\left\{ s(\omega) \in C_{i,j} \right\}}` is the function that takes as value
        one if :math:`s(\omega) \in C_{i,j}` and zero otherwise.
        Note that :math:`ds = \frac{r(\omega)^2 d\omega}{\lvert \cos{\alpha(\omega)} \rvert}`.

        The previous integral is approximated through Monte Carlo sampling by shooting :math:`N` rays
        with directions :math:`\omega_n` arranged as a Fibonacci lattice on the unit sphere around the transmitter,
        and bouncing the rays on the intersected objects until the maximum depth (``max_depth``) is reached or
        the ray bounces out of the scene.
        At every intersection with an object of the scene, a new ray is shot from the intersection which corresponds to either
        specular reflection or diffuse scattering, following a Bernoulli distribution with parameter the
        squared scattering coefficient.
        When diffuse scattering is selected, the direction of the scattered ray is uniformly sampled on the half-sphere.
        The resulting Monte Carlo estimate is:

        .. math::
            :label: cm_mc_ref

            \hat{b}_{i,j}^{\text{(ref)}} = \frac{4\pi}{N\lvert C \rvert} \sum_{n=1}^N \lvert h\left(s(\omega_n)\right)  \rvert^2 \frac{r(\omega_n)^2}{\lvert \cos{\alpha(\omega_n)} \rvert} \mathbb{1}_{\left\{ s(\omega_n) \in C_{i,j} \right\}}.

        For the diffracted paths, :eq:`cm_def` can be rewritten for any wedge
        with length :math:`L` and opening angle :math:`\Phi` as an integral over the wedge and its opening angle,
        by substituting :math:`s` with the position on the wedge :math:`\ell \in [1,L]` and the angle :math:`\phi \in [0, \Phi]`:

        .. math::
            b_{i,j} = \frac{1}{\lvert C \rvert} \int_{\ell} \int_{\phi} \lvert h\left(s(\ell,\phi) \right) \rvert^2 \mathbb{1}_{\left\{ s(\ell,\phi) \in C_{i,j} \right\}} \left\lVert \frac{\partial r}{\partial \ell} \times \frac{\partial r}{\partial \phi} \right\rVert d\ell d\phi

        where the integral is over the wedge length :math:`L` and opening angle :math:`\Phi`, and
        :math:`r\left( \ell, \phi \right)` is the reparametrization with respected to :math:`(\ell, \phi)` of the
        intersection between the diffraction cone at :math:`\ell` and the rectangle defining the coverage map (see, e.g., [SurfaceIntegral]_).
        The previous integral is approximated through Monte Carlo sampling by shooting :math:`N'` rays from equally spaced
        locations :math:`\ell_n` along the wedge with directions :math:`\phi_n` sampled uniformly from :math:`(0, \Phi)`:

        .. math::
            :label: cm_mc_diff

            \hat{b}_{i,j}^{\text{(diff)}} = \frac{L\Phi}{N'\lvert C \rvert} \sum_{n=1}^{N'} \lvert h\left(s(\ell_n,\phi_n)\right) \rvert^2 \mathbb{1}_{\left\{ s(\ell_n,\phi_n) \in C_{i,j} \right\}} \left\lVert \left(\frac{\partial r}{\partial \ell}\right)_n \times \left(\frac{\partial r}{\partial \phi}\right)_n \right\rVert.

        The output of this function is therefore a real-valued matrix of size ``[num_cells_y, num_cells_x]``,
        for every transmitter, with elements equal to the sum of the contributions of the reflected and scattered paths
        :eq:`cm_mc_ref` and diffracted paths :eq:`cm_mc_diff` for all the wedges, and where

        .. math::
            \texttt{num_cells_x} = \bigg\lceil\frac{\texttt{cm_size[0]}}{\texttt{cm_cell_size[0]}} \bigg\rceil\\
            \texttt{num_cells_y} = \bigg\lceil \frac{\texttt{cm_size[1]}}{\texttt{cm_cell_size[1]}} \bigg\rceil.

        The surface defining the coverage map is a rectangle centered at
        ``cm_center``, with orientation ``cm_orientation``, and with size
        ``cm_size``. An orientation of (0,0,0) corresponds to
        a coverage map parallel to the XY plane, with surface normal pointing towards
        the :math:`+z` axis. By default, the coverage map
        is parallel to the XY plane, covers all of the scene, and has
        an elevation of :math:`z = 1.5\text{m}`.
        The receiver is assumed to use the antenna array
        ``scene.rx_array``. If transmitter and/or receiver have multiple antennas, transmit precoding
        and receive combining are applied which are defined by ``precoding_vec`` and
        ``combining_vec``, respectively.

        The :math:`(i,j)` indices are omitted in the following for clarity.
        For reflection and scattering, paths are generated by shooting ``num_samples`` rays from the
        transmitters with directions arranged in a Fibonacci lattice on the unit
        sphere and by simulating their propagation for up to ``max_depth`` interactions with
        scene objects.
        If ``max_depth`` is set to 0 and if ``los`` is set to `True`,
        only the line-of-sight path is considered.
        For diffraction, paths are generated by shooting ``num_samples`` rays from equally
        spaced locations along the wedges in line-of-sight with the transmitter, with
        directions uniformly sampled on the diffraction cone.

        For every ray :math:`n` intersecting the coverage map cell :math:`(i,j)`, the
        channel coefficients, :math:`a_n`, and the angles of departure (AoDs)
        :math:`(\theta_{\text{T},n}, \varphi_{\text{T},n})`
        and arrival (AoAs) :math:`(\theta_{\text{R},n}, \varphi_{\text{R},n})`
        are computed. See the `Primer on Electromagnetics <../em_primer.html>`_ for more details.

        A "synthetic" array is simulated by adding additional phase shifts that depend on the
        antenna position relative to the position of the transmitter (receiver) as well as the AoDs (AoAs).
        For the :math:`k^\text{th}` transmit antenna and :math:`\ell^\text{th}` receive antenna, let
        us denote by :math:`\mathbf{d}_{\text{T},k}` and :math:`\mathbf{d}_{\text{R},\ell}` the relative positions (with respect to
        the positions of the transmitter/receiver) of the pair of antennas
        for which the channel impulse response shall be computed. These can be accessed through the antenna array's property
        :attr:`~sionna.rt.AntennaArray.positions`. Using a plane-wave assumption, the resulting phase shifts
        from these displacements can be computed as

        .. math::

            p_{\text{T}, n,k} &= \frac{2\pi}{\lambda}\hat{\mathbf{r}}(\theta_{\text{T},n}, \varphi_{\text{T},n})^\mathsf{T} \mathbf{d}_{\text{T},k}\\
            p_{\text{R}, n,\ell} &= \frac{2\pi}{\lambda}\hat{\mathbf{r}}(\theta_{\text{R},n}, \varphi_{\text{R},n})^\mathsf{T} \mathbf{d}_{\text{R},\ell}.

        The final expression for the path coefficient is

        .. math::

            h_{n,k,\ell} =  a_n e^{j(p_{\text{T}, i,k} + p_{\text{R}, i,\ell})}

        for every transmit antenna :math:`k` and receive antenna :math:`\ell`.
        These coefficients form the complex-valued channel matrix, :math:`\mathbf{H}_n`,
        of size :math:`\texttt{num_rx_ant} \times \texttt{num_tx_ant}`.

        Finally, the coefficient of the equivalent SISO channel is

        .. math::
            h_n =  \mathbf{c}^{\mathsf{H}} \mathbf{H}_n \mathbf{p}

        where :math:`\mathbf{c}` and :math:`\mathbf{p}` are the combining and
        precoding vectors (``combining_vec`` and ``precoding_vec``),
        respectively.

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
                        position=[8.5,21,30],
                        orientation=[0,0,0])
            scene.add(tx)
            tx.look_at([40,80,1.5])

            # Compute coverage map
            cm = scene.coverage_map(cm_cell_size=[1.,1.],
                                num_samples=int(10e6))

            # Visualize coverage in preview
            scene.preview(coverage_map=cm,
                        resolution=[1000, 600])

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
            If set to `None`, then no combining is applied, and
            the energy received by all antennas is summed.

        precoding_vec : [num_tx_ant], complex | None
            Precoding vector.
            If set to `None`, then defaults to
            :math:`\frac{1}{\sqrt{\text{num_tx_ant}}} [1,\dots,1]^{\mathsf{T}}`.

        num_samples : int
            Number of random rays to trace.
            For the reflected paths, this number is split equally over the different transmitters.
            For the diffracted paths, it is split over the wedges in line-of-sight with the
            transmitters such that the number of rays allocated
            to a wedge is proportional to its length.
            Defaults to 2e6.

        los : bool
            If set to `True`, then the LoS paths are computed.
            Defaults to `True`.

        reflection : bool
            If set to `True`, then the reflected paths are computed.
            Defaults to `True`.

        diffraction : bool
            If set to `True`, then the diffracted paths are computed.
            Defaults to `False`.

        scattering : bool
            If set to `True`, then the scattered paths are computed.
            Defaults to `False`.

        ris : bool
            If set to `True`, then paths involving RIS are computed.
            Defaults to `True`.

        edge_diffraction : bool
            If set to `False`, only diffraction on wedges, i.e., edges that
            connect two primitives, is considered.
            Defaults to `False`.

        check_scene : bool
            If set to `True`, checks that the scene is well configured before
            computing the coverage map. This can add a significant overhead.
            Defaults to `True`.

        Output
        ------
        :cm : :class:`~sionna.rt.CoverageMap`
            The coverage maps
        """

        # Check that all is set to compute the coverage map
        if check_scene:
            self._check_scene(True)

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
        if combining_vec is not None:
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
                                 los=los,
                                 reflection=reflection,
                                 diffraction=diffraction,
                                 scattering=scattering,
                                 ris=ris,
                                 edge_diffraction=edge_diffraction)

        return output

    def preview(self, paths=None, show_paths=True, show_devices=True,
                show_orientations=False,
                coverage_map=None, cm_tx=0, cm_db_scale=True,
                cm_vmin=None, cm_vmax=None,
                resolution=(655, 500), fov=45, background='#ffffff',
                clip_at=None, clip_plane_orientation=(0, 0, -1)):
        # pylint: disable=line-too-long
        r"""preview(paths=None, show_paths=True, show_devices=True, coverage_map=None, cm_tx=0, cm_vmin=None, cm_vmax=None, resolution=(655, 500), fov=45, background='#ffffff', clip_at=None, clip_plane_orientation=(0, 0, -1))

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

        Default color coding:

        * Green: Receiver
        * Blue: Transmitter
        * Red: Reconfigurable Intelligent Surface (RIS)

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
            If set to `True`, shows the radio devices.
            Defaults to `True`.

        show_orientations : bool
            If `show_devices` is `True`, shows the radio devices orientations.
            Defaults to `False`.

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
            These parameters should be provided in dB if ``cm_db_scale`` is
            set to `True`, or in linear scale otherwise.
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

        clip_at : float
            If not `None`, the scene preview will be clipped (cut) by a plane
            with normal orientation ``clip_plane_orientation`` and offset ``clip_at``.
            That means that everything *behind* the plane becomes invisible.
            This allows visualizing the interior of meshes, such as buildings.
            Defaults to `None`.

        clip_plane_orientation : tuple[float, float, float]
            Normal vector of the clipping plane.
            Defaults to (0,0,-1).
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
            fig.plot_ris()
        if coverage_map is not None:
            fig.plot_coverage_map(
                coverage_map, tx=cm_tx, db_scale=cm_db_scale,
                vmin=cm_vmin, vmax=cm_vmax)

        # Clipping
        fig.set_clipping_plane(offset=clip_at, orientation=clip_plane_orientation)

        # Update the camera state
        if not needs_reset:
            fig.center_view()

        return fig

    def render(self, camera, paths=None, show_paths=True, show_devices=True,
               coverage_map=None, cm_tx=0, cm_db_scale=True,
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

        cm_db_scale: bool
            Use logarithmic scale for coverage map visualization, i.e. the
            coverage values are mapped with:
            :math:`y = 10 \cdot \log_{10}(x)`.
            Defaults to `True`.

        cm_vmin, cm_vmax: float | None
            For coverage map visualization, defines the range of path gains that
            the colormap covers.
            These parameters should be provided in dB if ``cm_db_scale`` is
            set to `True`, or in linear scale otherwise.
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
                       cm_db_scale=cm_db_scale,
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
                coverage_map[cm_tx, :, :].numpy(), db_scale=cm_db_scale,
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

        cm_vmin, cm_vmax: float | None
            For coverage map visualization, defines the range of path gains that
            the colormap covers.
            These parameters should be provided in dB if ``cm_db_scale`` is
            set to `True`, or in linear scale otherwise.
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

    @property
    def radio_material_callable(self):
        # pylint: disable=line-too-long
        r"""
        Get/set a callable that computes the radio material properties at the
        points of intersection between the rays and the scene objects.

        If set, then the :class:`~sionna.rt.RadioMaterial` of the objects are
        not used and the callable is invoked instead to obtain the
        electromagnetic properties required to simulate the propagation of radio
        waves.

        If not set, i.e., `None` (default), then the
        :class:`~sionna.rt.RadioMaterial` of objects are used to simulate the
        propagation of radio waves in the scene.

        This callable is invoked on batches of intersection points.
        It takes as input the following tensors:

        * ``object_id`` (`[batch_dims]`, `int`) : Integers uniquely identifying the intersected objects
        * ``points`` (`[batch_dims, 3]`, `float`) : Positions of the intersection points

        The callable must output a tuple/list of the following tensors:

        * ``complex_relative_permittivity`` (`[batch_dims]`, `complex`) : Complex relative permittivities :math:`\eta` :eq:`eta`
        * ``scattering_coefficient`` (`[batch_dims]`, `float`) : Scattering coefficients :math:`S\in[0,1]` :eq:`scattering_coefficient`
        * ``xpd_coefficient`` (`[batch_dims]`, `float`) : Cross-polarization discrimination coefficients :math:`K_x\in[0,1]` :eq:`xpd`. Only relevant for the scattered field.

        **Note:** The number of batch dimensions is not necessarily equal to one.
        """
        return self._radio_material_callable

    @radio_material_callable.setter
    def radio_material_callable(self, rm_callable):
        self._radio_material_callable = rm_callable

    @property
    def scattering_pattern_callable(self):
        # pylint: disable=line-too-long
        r"""
        Get/set a callable that computes the scattering pattern at the
        points of intersection between the rays and the scene objects.

        If set, then the :attr:`~sionna.rt.RadioMaterial.scattering_pattern` of
        the radio materials of the objects are not used and the callable is invoked
        instead to evaluate the scattering pattern required to simulate the
        propagation of diffusely reflected radio waves.

        If not set, i.e., `None` (default), then the
        :attr:`~sionna.rt.RadioMaterial.scattering_pattern` of the objects'
        radio materials are used to simulate the propagation of diffusely
        reflected radio waves in the scene.

        This callable is invoked on batches of intersection points.
        It takes as input the following tensors:

        * ``object_id`` (`[batch_dims]`, `int`) : Integers uniquely identifying the intersected objects
        * ``points`` (`[batch_dims, 3]`, `float`) : Positions of the intersection points
        * ``k_i`` (`[batch_dims, 3]`, `float`) : Unitary vector corresponding to the direction of incidence in the scene's global coordinate system
        * ``k_s`` (`[batch_dims, 3]`, `float`) : Unitary vector corresponding to the direction of the diffuse reflection in the scene's global coordinate system
        * ``n`` (`[batch_dims, 3]`, `float`) : Unitary vector corresponding to the normal to the surface at the intersection point

        The callable must output the following tensor:

        * ``f_s`` (`[batch_dims]`, `float`) : The scattering pattern evaluated for the previous inputs

        **Note:** The number of batch dimensions is not necessarily equal to one.
        """
        return self._scattering_pattern_callable

    @scattering_pattern_callable.setter
    def scattering_pattern_callable(self, sp_callable):
        self._scattering_pattern_callable = sp_callable

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
    def mi_scene_params(self):
        """
        :class:`~mitsuba.SceneParameters` : Get the Mitsuba scene parameters
        """
        return self._scene_params

    @property
    def solver_paths(self):
        """
        :class:`~sionna.rt.SolverPaths` : Get the paths solver
        """
        return self._solver_paths

    @property
    def solver_cm(self):
        """
        :class:`~sionna.rt.SolverCoverageMap` : Get the coverage map solver
        """
        return self._solver_cm

    @property
    def preview_widget(self):
        """
        :class:`~sionna.rt.InteractiveDisplay` : Get the preview widget
        """
        return self._preview_widget

    def scene_geometry_updated(self):
        """
        Callback to trigger when the scene geometry is updated
        """
        # Update the scene geometry in the preview
        if self._preview_widget:
            self._preview_widget.redraw_scene_geometry()

    def _clear(self):
        r"""
        Clear everything.
        Should be called when a new scene is loaded.
        """

        self._transmitters.clear()
        self._receivers.clear()
        self._ris.clear()
        self._ris.clear()
        self._cameras.clear()
        self._radio_materials.clear()
        self._scene_objects.clear()
        self._tx_array = None
        self._rx_array = None
        self._preview_widget = None

    def _check_scene(self, coverage_map):
        r"""
        Check that all is set for paths or coverage map computation.
        If not, raises an exception with the appropriate error message.

        Input
        ------
        coverage_map : bool
            If set to `True`, then checks the scene in preparation for coverage
            map computation. Otherwise, checks the scene in preparation for
            paths computation.
        """
        if not self._rx_array:
            raise ValueError("Receiver array not set.")

        if not self._tx_array:
            raise ValueError("Transmitter array not set.")

        if len(self._transmitters) == 0:
            raise ValueError("No transmitter defined.")

        # Instantiation of receivers is not needed to compute a coverage map
        if not coverage_map:
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
        Load the scene objects available in the scene
        """
        # Parse all shapes in the scene
        scene = self._scene
        objects_id = dr.reinterpret_array_v(mi.UInt32,
                                            scene.shapes_dr()).tf()
        for obj_id,s in zip(objects_id,scene.shapes()):
            obj_id = int(obj_id.numpy())
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
            obj = SceneObject(name, object_id=obj_id, mi_shape=s, dtype=self._dtype)
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

.. figure:: ../figures/floor_wall.png
   :align: center
"""

# pylint: disable=C0301
simple_street_canyon = str(files(scenes).joinpath("simple_street_canyon/simple_street_canyon.xml"))
"""
Example scene containing a few rectangular building blocks and a ground plane

.. figure:: ../figures/street_canyon.png
   :align: center
"""

# pylint: disable=C0301
simple_street_canyon_with_cars = str(files(scenes).joinpath("simple_street_canyon_with_cars/simple_street_canyon_with_cars.xml"))
"""
Example scene containing a few rectangular building blocks and a ground plane as well as some cars

.. figure:: ../figures/street_canyon_with_cars.png
   :align: center
"""

etoile = str(files(scenes).joinpath("etoile/etoile.xml"))
# pylint: disable=C0301
"""
Example scene containing the area around the Arc de Triomphe in Paris
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
The scene was created with data downloaded from `OpenStreetMap <https://www.openstreetmap.org>`_ and
the help of `Blender <https://www.blender.org>`_ and the `Blender-OSM <https://github.com/vvoovv/blender-osm>`_
and `Mitsuba Blender <https://github.com/mitsuba-renderer/mitsuba-blender>`_ add-ons.
The data is licensed under the `Open Data Commons Open Database License (ODbL) <https://openstreetmap.org/copyright>`_.

.. figure:: ../figures/munich.png
   :align: center
"""

simple_wedge = str(files(scenes).joinpath("simple_wedge/simple_wedge.xml"))
# pylint: disable=C0301
r"""
Example scene containing a wedge with a :math:`90^{\circ}` opening angle

.. figure:: ../figures/simple_wedge.png
   :align: center
"""

simple_reflector = str(files(scenes).joinpath("simple_reflector/simple_reflector.xml"))
# pylint: disable=C0301
r"""
Example scene containing a metallic square

.. figure:: ../figures/simple_reflector.png
   :align: center
"""

double_reflector = str(files(scenes).joinpath("double_reflector/double_reflector.xml"))
# pylint: disable=C0301
r"""
Example scene containing two metallic squares

.. figure:: ../figures/double_reflector.png
   :align: center
"""

triple_reflector = str(files(scenes).joinpath("triple_reflector/triple_reflector.xml"))
# pylint: disable=C0301
r"""
Example scene containing three metallic rectangles

.. figure:: ../figures/triple_reflector.png
   :align: center
"""

box = str(files(scenes).joinpath("box/box.xml"))
# pylint: disable=C0301
r"""
Example scene containing a metallic box

.. figure:: ../figures/box.png
   :align: center
"""
