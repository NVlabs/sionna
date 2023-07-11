#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Dataclass that stores paths
"""

import tensorflow as tf
import os

from sionna.utils.tensors import expand_to_rank, insert_dims
from sionna.constants import PI
from .utils import dot, r_hat

class Paths:
    # pylint: disable=line-too-long
    r"""
    Paths()

    Stores the simulated propagation paths

    Paths are generated for the loaded scene using
    :meth:`~sionna.rt.Scene.compute_paths`. Please refer to the
    documentation of this function for further details.
    These paths can then be used to compute channel impulse responses:

    .. code-block:: Python

        paths = scene.compute_paths()
        a, tau = paths.cir()

    where ``scene`` is the :class:`~sionna.rt.Scene` loaded using
    :func:`~sionna.rt.load_scene`.
    """

    # Input
    # ------
    # a : [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths, num_time_steps], tf.complex
    #     Channel coefficients :math:`a_i` as defined in :eq:`T_tilde`.
    #     If there are less than `max_num_path` valid paths between a
    #     transmit and receive antenna, the irrelevant elements are
    #     filled with zeros.

    # tau : [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], tf.float
    #     Propagation delay of each path [s].
    #     If :attr:`~sionna.rt.Scene.synthetic_array` is `True`, the shape of this tensor
    #     is `[1, num_rx, num_tx, max_num_paths]` as the delays for the
    #     individual antenna elements are assumed to be equal.
    #     If there are less than `max_num_path` valid paths between a
    #     transmit and receive antenna, the irrelevant elements are
    #     filled with -1.

    # theta_t : [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], tf.float
    #     Zenith  angles of departure :math:`\theta_{\text{T},i}` [rad].
    #     If :attr:`~sionna.rt.Scene.synthetic_array` is `True`, the shape of this tensor
    #     is `[1, num_rx, num_tx, max_num_paths]` as the angles for the
    #     individual antenna elements are assumed to be equal.

    # phi_t : [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], tf.float
    #     Azimuth angles of departure :math:`\varphi_{\text{T},i}` [rad].
    #     See description of ``theta_t``.

    # theta_r : [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], tf.float
    #     Zenith angles of arrival :math:`\theta_{\text{R},i}` [rad].
    #     See description of ``theta_t``.

    # phi_r : [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], tf.float
    #     Azimuth angles of arrival :math:`\varphi_{\text{T},i}` [rad].
    #     See description of ``theta_t``.

    # types : [batch_size, max_num_paths], tf.int
    #     Type of path:

    #     - 0 : LoS
    #     - 1 : Reflected
    #     - 2 : Diffracted
    #     - 3 : Scattered

    # Types of paths
    LOS = 0
    SPECULAR = 1
    DIFFRACTED = 2
    SCATTERED = 3

    def __init__(self,
                 sources,
                 targets,
                 scene,
                 types=None):

        dtype = scene.dtype
        num_sources = sources.shape[0]
        num_targets = targets.shape[0]
        rdtype = dtype.real_dtype

        self._a = tf.zeros([num_targets, num_sources, 0, 2, 2], dtype)
        self._tau = tf.zeros([num_targets, num_sources, 0], rdtype)
        self._theta_t = tf.zeros([num_targets, num_sources, 0], rdtype)
        self._theta_r = tf.zeros([num_targets, num_sources, 0], rdtype)
        self._phi_t = tf.zeros([num_targets, num_sources, 0], rdtype)
        self._phi_r = tf.zeros([num_targets, num_sources, 0], rdtype)
        self._mask = tf.fill([num_targets, num_sources, 0], False)
        self._vertices = tf.zeros([0, num_targets, num_sources, 0, 3], rdtype)
        self._objects = tf.fill([0, num_targets, num_sources, 0], -1)
        if types is None:
            self._types = tf.fill([0], -1)
        else:
            self._types = types

        self._sources = sources
        self._targets = targets
        self._scene = scene

        # Is the direction reversed?
        self._reverse_direction = False
        # Normalize paths delays?
        self._normalize_delays = False

    def export(self, filename):
        r"""
        export(filename)

        Saves the paths as an OBJ file for visualisation, e.g., in Blender

        Input
        ------
        filename : str
            Path and name of the file
        """
        vertices = self.vertices
        objects = self.objects
        sources = self.sources
        targets = self.targets
        mask = self.mask

        # Content of the obj file
        r = ''
        offset = 0
        for rx in range(vertices.shape[1]):
            tgt = targets[rx].numpy()
            for tx in range(vertices.shape[2]):
                src = sources[tx].numpy()
                for p in range(vertices.shape[3]):

                    # If the path is masked, skip it
                    if not mask[rx,tx,p]:
                        continue

                    # Add a comment to describe this path
                    r += f'# Path {p} from tx {tx} to rx {rx}' + os.linesep
                    # Vertices and intersected objects
                    vs = vertices[:,rx,tx,p].numpy()
                    objs = objects[:,rx,tx,p].numpy()

                    depth = 0
                    # First vertex is the source
                    r += f"v {src[0]:.8f} {src[1]:.8f} {src[2]:.8f}"+os.linesep
                    # Add intersection points
                    for v,o in zip(vs,objs):
                        # Skip if no intersection
                        if o == -1:
                            continue
                        r += f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}" + os.linesep
                        depth += 1
                    r += f"v {tgt[0]:.8f} {tgt[1]:.8f} {tgt[2]:.8f}"+os.linesep

                    # Add the connections
                    for i in range(1, depth+2):
                        v0 = i + offset
                        v1 = i + offset + 1
                        r += f"l {v0} {v1}" + os.linesep

                    # Prepare for the next path
                    r += os.linesep
                    offset += depth+2

        # Save the file
        # pylint: disable=unspecified-encoding
        with open(filename, 'w') as f:
            f.write(r)

    @property
    def a(self):
        # pylint: disable=line-too-long
        """
        [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths, num_time_steps], tf.complex : Channel coefficients
        """
        return self._a

    @a.setter
    def a(self, v):
        self._a = v

    @property
    def tau(self):
        # pylint: disable=line-too-long
        """
        [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], tf.float : Propagation delay of each path [s]
        """
        return self._tau

    @tau.setter
    def tau(self, v):
        self._tau = v

    @property
    def theta_t(self):
        # pylint: disable=line-too-long
        """
        [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], tf.float : Zenith  angles of departure [rad]
        """
        return self._theta_t

    @theta_t.setter
    def theta_t(self, v):
        self._theta_t = v

    @property
    def phi_t(self):
        # pylint: disable=line-too-long
        """
        [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], tf.float : Azimuth angles of departure [rad]
        """
        return self._phi_t

    @phi_t.setter
    def phi_t(self, v):
        self._phi_t = v

    @property
    def theta_r(self):
        # pylint: disable=line-too-long
        """
        [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], tf.float : Zenith angles of arrival [rad]
        """
        return self._theta_r

    @theta_r.setter
    def theta_r(self, v):
        self._theta_r = v

    @property
    def phi_r(self):
        # pylint: disable=line-too-long
        """
        [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], tf.float : Azimuth angles of arrival [rad]
        """
        return self._phi_r

    @phi_r.setter
    def phi_r(self, v):
        self._phi_r = v

    @property
    def types(self):
        """
        [batch_size, max_num_paths], tf.int : Type of the paths:

        - 0 : LoS
        - 1 : Reflected
        - 2 : Diffracted
        - 3 : Scattered
        """
        return self._types

    @types.setter
    def types(self, v):
        self._types = v

    @property
    def sources(self):
        # pylint: disable=line-too-long
        """
        [num_sources, 3], tf.float : Sources from which rays (paths) are emitted
        """
        return self._sources

    @sources.setter
    def sources(self, v):
        self._sources = v

    @property
    def targets(self):
        # pylint: disable=line-too-long
        """
        [num_targets, 3], tf.float : Targets at which rays (paths) are received
        """
        return self._targets

    @targets.setter
    def targets(self, v):
        self._targets = v

    @property
    def normalize_delays(self):
        """
        bool : Set to `True` to normalize path delays such that the first path
        between any pair of antennas of a transmitter and receiver arrives at
        ``tau = 0``. Defaults to `True`.
        """
        return self._normalize_delays

    @normalize_delays.setter
    def normalize_delays(self, v):
        if v == self._normalize_delays:
            return

        if ~v and self._normalize_delays:
            self.tau += self._min_tau
        else:
            self.tau -= self._min_tau
        self._normalize_delays = v

    def apply_doppler(self, sampling_frequency, num_time_steps,
                      tx_velocities=(0.,0.,0.), rx_velocities=(0.,0.,0.)):
        # pylint: disable=line-too-long
        r"""
        Apply Doppler shifts corresponding to input transmitters and receivers
        velocities.

        This function replaces the last dimension of the tensor storing the
        paths coefficients ``a``, which stores the the temporal evolution of
        the channel, with a dimension of size ``num_time_steps`` computed
        according to the input velocities.

        When this function is called multiple times, it overwrites the previous
        time steps dimension.

        Input
        ------
        sampling_frequency : float
            Frequency [Hz] at which the channel impulse response is sampled

        num_time_steps : int
            Number of time steps.

        tx_velocities : [batch_size, num_tx, 3] or broadcastable, tf.float | `None`
            Velocity vectors :math:`(v_\text{x}, v_\text{y}, v_\text{z})` of all
            transmitters [m/s].
            Defaults to `[0,0,0]`.

        rx_velocities : [batch_size, num_tx, 3] or broadcastable, tf.float | `None`
            Velocity vectors :math:`(v_\text{x}, v_\text{y}, v_\text{z})` of all
            receivers [m/s].
            Defaults to `[0,0,0]`.
        """

        dtype = self._scene.dtype
        rdtype = dtype.real_dtype
        zeror = tf.zeros((), rdtype)
        two_pi = tf.cast(2.*PI, rdtype)

        tx_velocities = tf.cast(tx_velocities, rdtype)
        tx_velocities = expand_to_rank(tx_velocities, 3, 0)
        if tx_velocities.shape[2] != 3:
            raise ValueError("Last dimension of `tx_velocities` must equal 3")

        if rx_velocities is None:
            rx_velocities = [0.,0.,0.]
        rx_velocities = tf.cast(rx_velocities, rdtype)
        rx_velocities = expand_to_rank(rx_velocities, 3, 0)
        if rx_velocities.shape[2] != 3:
            raise ValueError("Last dimension of `rx_velocities` must equal 3")

        sampling_frequency = tf.cast(sampling_frequency, rdtype)
        if sampling_frequency <= 0.0:
            raise ValueError("The sampling frequency must be positive")

        num_time_steps = tf.cast(num_time_steps, tf.int32)
        if num_time_steps <= 0:
            msg = "The number of time samples must a positive integer"
            raise ValueError(msg)

        # Drop previous time step dimension, if any
        if tf.rank(self.a) == 7:
            self.a = self.a[...,0]

        # [batch_size, num_rx, num_tx, max_num_paths, 3]
        # or
        # [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths, 3]
        k_t = r_hat(self.theta_t, self.phi_t)
        k_r = r_hat(self.theta_r, self.phi_r)

        if self._scene.synthetic_array:
            # [batch_size, num_rx, 1, num_tx, 1, max_num_paths, 3]
            k_t = tf.expand_dims(tf.expand_dims(k_t, axis=2), axis=4)
            # [batch_size, num_rx, 1, num_tx, 1, max_num_paths, 3]
            k_r = tf.expand_dims(tf.expand_dims(k_r, axis=2), axis=4)

        # Expand rank of the speed vector for broadcasting with k_r
        # [batch_dim, 1, 1, num_tx, 1, 1, 3]
        tx_velocities = insert_dims(insert_dims(tx_velocities, 2,1), 2,4)
        # [batch_dim, num_rx, 1, 1, 1, 1, 3]
        rx_velocities = insert_dims(rx_velocities, 4, 1)

        # Generate time steps
        # [num_time_steps]
        ts = tf.range(num_time_steps, dtype=rdtype)
        ts = ts / sampling_frequency

        # Compute the Doppler shift
        # [batch_dim, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths]
        # or
        # [batch_dim, num_rx, 1, num_tx, 1, max_num_paths]
        tx_ds = two_pi*dot(tx_velocities, k_t)/self._scene.wavelength
        rx_ds = two_pi*dot(rx_velocities, k_r)/self._scene.wavelength
        ds = tx_ds + rx_ds
        # Expand for the time sample dimension
        # [batch_dim, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths, 1]
        # or
        # [batch_dim, num_rx, 1, num_tx, 1, max_num_paths, 1]
        ds = tf.expand_dims(ds, axis=-1)
        # Expand time steps for broadcasting
        # [1, 1, 1, 1, 1, 1, num_time_steps]
        ts = expand_to_rank(ts, tf.rank(ds), 0)
        # [batch_dim, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths,
        #   num_time_steps]
        # or
        # [batch_dim, num_rx, 1, num_tx, 1, max_num_paths, num_time_steps]
        ds = ds*ts
        exp_ds = tf.exp(tf.complex(zeror, ds))

        # Apply Doppler shift
        # Expand with time dimension
        # [batch_dim, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths, 1]
        a = tf.expand_dims(self.a, axis=-1)
        if self._scene.synthetic_array:
            # Broadcast is not supported by TF for such high rank tensors.
            # We therefore do it manually
            # [batch_dim, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths,
            #   num_time_steps]
            a = tf.tile(a, [1, 1, 1, 1, 1, 1, exp_ds.shape[6]])
        # [batch_dim, num_rx,  num_rx_ant, num_tx, num_tx_ant, max_num_paths,
        #   num_time_steps]
        a = a*exp_ds

        self.a = a

    @property
    def reverse_direction(self):
        r"""
        bool : If set to `True`, swaps receivers and transmitters
        """
        return self._reverse_direction

    @reverse_direction.setter
    def reverse_direction(self, v):

        if v == self._reverse_direction:
            return

        if tf.rank(self.a) == 6:
            self.a = tf.transpose(self.a, perm=[0,3,4,1,2,5])
        else:
            self.a = tf.transpose(self.a, perm=[0,3,4,1,2,5,6])

        if self._scene.synthetic_array:
            self.tau = tf.transpose(self.tau, perm=[0,2,1,3])
            self.theta_t = tf.transpose(self.theta_t, perm=[0,2,1,3])
            self.phi_t = tf.transpose(self.phi_t, perm=[0,2,1,3])
            self.theta_r = tf.transpose(self.theta_r, perm=[0,2,1,3])
            self.phi_r = tf.transpose(self.phi_r, perm=[0,2,1,3])
        else:
            self.tau = tf.transpose(self.tau, perm=[0,3,4,1,2,5])
            self.theta_t = tf.transpose(self.theta_t, perm=[0,3,4,1,2,5])
            self.phi_t = tf.transpose(self.phi_t, perm=[0,3,4,1,2,5])
            self.theta_r = tf.transpose(self.theta_r, perm=[0,3,4,1,2,5])
            self.phi_r = tf.transpose(self.phi_r, perm=[0,3,4,1,2,5])

        self._reverse_direction = v

    def cir(self, los=True, reflection=True, diffraction=True, scattering=True):
        # pylint: disable=line-too-long
        r"""
        Returns the channel impulse response ``(a, tau)`` which can be used
        for link simulations by other Sionna components.

        In the case of non-synthetic arrays, the delay for each transmitter-receiver
        pair is determined by the smallest observed delay among all corresponding
        transmit-receive antenna pairs. To accommodate varying
        propagation delays among these antenna pairs, the channel coefficients,
        returned by this function, are updated as follows:

        .. math::
            \tilde{a}_{k,\ell,m,n} = a_{k,\ell,m,n} e^{j 2 \pi \left( \tau_{k,\ell,m,n} - \tau_{k,\ell,\text{min}} \right) f}

        where :math:`(k,\ell)` denotes the transmitter-receiver pair,
        :math:`(m,n)` denotes the transmit-receive antenna pair, :math:`a_{k,\ell,m,n}`
        is the path coefficient (:attr:`~sionna.rt.Paths.a`),
        :math:`\tau_{k,\ell,m,n}` is the path delay (:attr:`~sionna.rt.Paths.tau`),
        :math:`f` is the carrier frequency, and

        .. math::

            \tau_{k, \ell, \text{min}} = \min_{m,n} \tau_{k, \ell, m,n}.

        Note: For the paths of a given type to be returned (LoS, reflection, etc.), they
        need to have been previously computed by :meth:`~sionna.rt.Scene.compute_paths` by
        setting the corresponding flag to `True`.

        Input
        ------
        los : bool
            If set to `False`, LoS paths are not returned.
            Defaults to `True`.

        reflection : bool
            If set to `False`, specular paths are not returned.
            Defaults to `True`.

        diffraction : bool
            If set to `False`, diffracted paths are not returned.
            Defaults to `True`.

        scattering : bool
            If set to `False`, scattered paths are not returned.
            Defaults to `True`.

        Output
        -------
        a : [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths, num_time_steps], tf.complex
            Paths coefficients

        tau : [batch_size, num_rx, num_tx, max_num_paths], tf.float
            Paths delays
        """

        two_pi = tf.cast(2.*PI, self._scene.dtype.real_dtype)

        # Select only the desired effects
        types = self.types[0]
        # [max_num_paths]
        selection_mask = tf.fill(tf.shape(types), False)
        if los:
            selection_mask = tf.logical_or(selection_mask,
                                           types == Paths.LOS)
        if reflection:
            selection_mask = tf.logical_or(selection_mask,
                                           types == Paths.SPECULAR)
        if diffraction:
            selection_mask = tf.logical_or(selection_mask,
                                           types == Paths.DIFFRACTED)
        if scattering:
            selection_mask = tf.logical_or(selection_mask,
                                           types == Paths.SCATTERED)

        # Extract selected paths
        a = tf.gather(self.a, tf.where(selection_mask)[:,0], axis=-2)
        tau = tf.gather(self.tau, tf.where(selection_mask)[:,0], axis=-1)

        # If not using synthetic array, apply the phase shifts due to the
        # difference in the time-of-arrivals between the different paths.
        if not self._scene.synthetic_array:
            # [batch_size, num_rx, 1, num_tx, 1, max_num_paths]
            tau_min = tf.reduce_min(tau, axis=(2,4), keepdims=True)
            # [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant,
            #   max_num_paths]
            delta_tau = tau - tau_min
            delta_phase = two_pi*delta_tau*self._scene.frequency
            # [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant,
            #   max_num_paths, 1]
            delta_phase = tf.expand_dims(delta_phase, axis=-1)
            # Apply the phase shift
            # [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant,
            #   max_num_paths, num_time_steps]
            a = a*tf.exp(tf.complex(tf.zeros_like(delta_phase), delta_phase))
            # Keep only the min delays
            # [batch_size, num_rx, num_tx, max_num_paths, num_time_steps]
            tau = tf.squeeze(tau_min, axis=(2,4))

        return a,tau

    #######################################################
    # Internal methods and properties
    #######################################################

    @property
    def mask(self):
        # pylint: disable=line-too-long
        """
        [num_targets, num_sources, max_num_paths], tf.bool : Mask indicating if a path is valid
        """
        return self._mask

    @mask.setter
    def mask(self, v):
        self._mask = v

    @property
    def vertices(self):
        # pylint: disable=line-too-long
        """
        [max_depth, num_targets, num_sources, max_num_paths, 3], tf.float : Positions of intersection points.
        """
        return self._vertices

    @vertices.setter
    def vertices(self, v):
        self._vertices = v

    @property
    def objects(self):
        # pylint: disable=line-too-long
        """
        [max_depth, num_targets, num_sources, max_num_paths], tf.int : Indices of the intersected scene objects
        or wedges. Paths with depth lower than ``max_depth`` are padded with `-1`.
        """
        return self._objects

    @objects.setter
    def objects(self, v):
        self._objects = v

    def merge(self, more_paths):
        r"""
        Merge ``more_paths`` with the current paths and returns the so-obtained
        instance. `self` is not updated.

        Input
        -----
        more_paths : :class:`~sionna.rt.Paths`
            First set of paths to merge
        """

        dtype = self._scene.dtype

        # The paths to merge must have the same number of sources and targets
        assert more_paths.targets.shape[0] == self.targets.shape[0],\
            "Paths to merge must have same number of targets"
        assert more_paths.sources.shape[0] == self.sources.shape[0],\
            "Paths to merge must have same number of targets"

        # Pad the paths with the lowest depth
        padding = self.vertices.shape[0] - more_paths.vertices.shape[0]
        if padding > 0:
            more_paths.vertices = tf.pad(more_paths.vertices,
                                         [[0,padding],[0,0],[0,0],[0,0],[0,0]],
                                constant_values=tf.zeros((), dtype.real_dtype))
            more_paths.objects = tf.pad(more_paths.objects,
                                        [[0,padding],[0,0],[0,0],[0,0]],
                                        constant_values=-1)
        elif padding < 0:
            padding = -padding
            self.vertices = tf.pad(self.vertices,
                                   [[0,padding],[0,0],[0,0],[0,0],[0,0]],
                            constant_values=tf.zeros((), dtype.real_dtype))
            self.objects = tf.pad(self.objects,
                                  [[0,padding],[0,0],[0,0],[0,0]],
                                  constant_values=-1)

        # Merge types
        if tf.rank(self.types) == 0:
            merged_types = tf.fill(tf.shape(self.vertices)[3],
                                   self.types)
        else:
            merged_types = self.types
        if tf.rank(more_paths.types) == 0:
            more_paths.types = tf.fill(tf.shape(more_paths.vertices)[3],
                                       more_paths.types)
        else:
            more_paths.types = more_paths.types

        self.types = tf.concat([merged_types, more_paths.types], axis=0)

        # Concatenate all
        self.a = tf.concat([self.a, more_paths.a], axis=2)
        self.tau = tf.concat([self.tau, more_paths.tau], axis=2)
        self.theta_t = tf.concat([self.theta_t, more_paths.theta_t], axis=2)
        self.phi_t = tf.concat([self.phi_t, more_paths.phi_t], axis=2)
        self.theta_r = tf.concat([self.theta_r, more_paths.theta_r], axis=2)
        self.phi_r = tf.concat([self.phi_r, more_paths.phi_r], axis=2)
        self.mask = tf.concat([self.mask, more_paths.mask], axis=2)
        self.vertices = tf.concat([self.vertices, more_paths.vertices], axis=3)
        self.objects = tf.concat([self.objects, more_paths.objects], axis=3)

        return self

    def finalize(self):
        """
        This function must be call to finalize the creation of the paths.
        This function:

        - Flags the LoS paths

        - Computes the smallest delay for delay normalization
        """

        self.set_los_path_type()

        tau = self.tau
        if self._scene.synthetic_array:
            min_tau = tf.reduce_min(tf.abs(tau), axis=2, keepdims=True)
        else:
            min_tau = tf.reduce_min(tf.abs(tau), axis=[1,3,4], keepdims=True)
        self._min_tau = min_tau

        # Add dummy-dimension for batch_size
        # [1, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths]
        self.a = tf.expand_dims(self.a, axis=0)
        # [1, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths]
        self.tau = tf.expand_dims(self.tau, axis=0)
        # [1, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths]
        self.theta_t = tf.expand_dims(self.theta_t, axis=0)
        # [1, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths]
        self.phi_t = tf.expand_dims(self.phi_t, axis=0)
        # [1, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths]
        self.theta_r = tf.expand_dims(self.theta_r, axis=0)
        # [1, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths]
        self.phi_r = tf.expand_dims(self.phi_r, axis=0)
        # [1, max_num_paths]
        self.types = tf.expand_dims(self.types, axis=0)

        # Add the time steps dimension
        # [1, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths, 1]
        self.a = tf.expand_dims(self.a, axis=-1)

        # Normalize delays
        self.normalize_delays = True

    def set_los_path_type(self):
        """
        Flags paths that do not hit any objects to as LoS ones.
        """

        if self.objects.shape[3] > 0:
            # [num_targets, num_sources, num_paths]
            los_path = tf.reduce_all(self.objects == -1, axis=0)
            # [num_targets, num_sources, num_paths]
            los_path = tf.logical_and(los_path, self.mask)
            # [num_paths]
            los_path = tf.reduce_any(los_path, axis=(0,1))
            # [[1]]
            los_path_index = tf.where(los_path)
            assert los_path_index.shape[0] < 2, "Only one LoS path can exist"
            if los_path_index.shape[0] > 0:
                self.types = tf.tensor_scatter_nd_update(self.types,
                                                         los_path_index,
                                                         [Paths.LOS])
