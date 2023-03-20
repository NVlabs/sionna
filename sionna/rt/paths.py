#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Dataclass that stores paths
"""

import os

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
        p2c = Paths2CIR(sampling_frequency=1e6, scene=scene)
        a, tau = p2c(paths.as_tuple())

    where ``scene`` is the :class:`~sionna.rt.Scene` loaded using
    :func:`~sionna.rt.load_scene`.

    Input
    ------
    mat_t : [batch_size, num_rx, rx_array_size, num_tx, tx_array_size, max_num_paths, 2, 2] or [batch_size, num_rx, num_tx, max_num_paths, 2, 2], tf.complex
        Transfer matrices :math:`\mathbf{T}_i` as defined in :eq:`T_tilde`.
        These are 2x2 complex-valued matrices modeling the transformation
        of the vertically and horizontally polarized field components along
        the paths.
        If :attr:`~sionna.rt.Scene.synthetic_array` is `True`, the shape of this tensor
        is `[1, num_rx, num_tx, max_num_paths, 2, 2]` as the responses for the
        individual antenna elements are synthetically computed by
        the :class:`Paths2CIR` layer.
        If there are less than `max_num_path` valid paths between a
        transmit and receive antenna, the irrelevant elements are
        filled with zeros.

    tau : [batch_size, num_rx, rx_array_size, num_tx, tx_array_size, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], tf.float
        Propagation delay of each path [s].
        If :attr:`~sionna.rt.Scene.synthetic_array` is `True`, the shape of this tensor
        is `[1, num_rx, num_tx, max_num_paths]` as the delays for the
        individual antenna elements are assumed to be equal.
        If there are less than `max_num_path` valid paths between a
        transmit and receive antenna, the irrelevant elements are
        filled with -1.

    theta_t : [batch_size, num_rx, rx_array_size, num_tx, tx_array_size, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], tf.float
        Zenith  angles of departure :math:`\theta_{\text{T},i}` [rad].
        If :attr:`~sionna.rt.Scene.synthetic_array` is `True`, the shape of this tensor
        is `[1, num_rx, num_tx, max_num_paths]` as the angles for the
        individual antenna elements are assumed to be equal.

    phi_t : [batch_size, num_rx, rx_array_size, num_tx, tx_array_size, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], tf.float
        Azimuth angles of departure :math:`\varphi_{\text{T},i}` [rad].
        See description of ``theta_t``.

    theta_r : [batch_size, num_rx, rx_array_size, num_tx, tx_array_size, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], tf.float
        Zenith angles of arrival :math:`\theta_{\text{R},i}` [rad].
        See description of ``theta_t``.

    phi_r : [batch_size, num_rx, rx_array_size, num_tx, tx_array_size, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], tf.float
        Azimuth angles of arrival :math:`\varphi_{\text{T},i}` [rad].
        See description of ``theta_t``.
    """

    def __init__(self,
                 mat_t,
                 tau,
                 theta_t,
                 phi_t,
                 theta_r,
                 phi_r,
                 sources=None,
                 targets=None,
                 mask=None,
                 vertices=None,
                 normals=None,
                 objects=None):

        self._mat_t = mat_t
        self._tau = tau
        self._theta_t = theta_t
        self._theta_r = theta_r
        self._phi_t = phi_t
        self._phi_r = phi_r
        self._mask = mask
        self._sources = sources
        self._targets = targets
        self._vertices = vertices
        self._normals = normals
        self._objects = objects

    def export(self, filename):
        r"""
        export(filaneme)

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
    def mat_t(self):
        # pylint: disable=line-too-long
        """
        [batch_size, num_rx, rx_array_size, num_tx, tx_array_size, max_num_paths, 2, 2] or [batch_size, num_rx, num_tx, max_num_paths, 2, 2], tf.complex : Get/set the transfer
        matrices
        """
        return self._mat_t

    @mat_t.setter
    def mat_t(self, v):
        self._mat_t = v

    @property
    def tau(self):
        # pylint: disable=line-too-long
        """
        [batch_size, num_rx, rx_array_size, num_tx, tx_array_size, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], tf.float : Get/set the propagation delay of each
        path [s]
        """
        return self._tau

    @tau.setter
    def tau(self, v):
        self._tau = v

    @property
    def theta_t(self):
        # pylint: disable=line-too-long
        """
        [batch_size, num_rx, rx_array_size, num_tx, tx_array_size, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], tf.float : Get/set the zenith  angles of departure [rad]
        """
        return self._theta_t

    @theta_t.setter
    def theta_t(self, v):
        self._theta_t = v

    @property
    def phi_t(self):
        # pylint: disable=line-too-long
        """
        [batch_size, num_rx, rx_array_size, num_tx, tx_array_size, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], tf.float : Get/set the azimuth angles of departure [rad]
        """
        return self._phi_t

    @phi_t.setter
    def phi_t(self, v):
        self._phi_t = v

    @property
    def theta_r(self):
        # pylint: disable=line-too-long
        """
        [batch_size, num_rx, rx_array_size, num_tx, tx_array_size, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], tf.float : Get/set the zenith angles of arrival [rad]
        """
        return self._theta_r

    @theta_r.setter
    def theta_r(self, v):
        self._theta_r = v

    @property
    def phi_r(self):
        # pylint: disable=line-too-long
        """
        [batch_size, num_rx, rx_array_size, num_tx, tx_array_size, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], tf.float : Get/set the azimuth angles of arrival [rad]
        """
        return self._phi_r

    @phi_r.setter
    def phi_r(self, v):
        self._phi_r = v

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
        max_depth, num_targets, num_sources, max_num_paths, 3], tf.float : Positions of intersection points.
        Paths with depth lower than ``max_depth`` are padded with zeros.
        """
        return self._vertices

    @vertices.setter
    def vertices(self, v):
        self._vertices = v

    @property
    def normals(self):
        # pylint: disable=line-too-long
        """
        [max_depth, num_targets, num_sources, max_num_paths, 3], tf.float : Normals to the primitives at the intersection points. Paths with depth lower than ``max_depth`` are padded with zeros.
        """
        return self._normals

    @normals.setter
    def normals(self, v):
        self._normals = v

    @property
    def objects(self):
        return self._objects

    @objects.setter
    def objects(self, v):
        """
        [max_depth, num_targets, num_sources, max_num_paths], tf.int : Indices of the intersected scene objects.
        Paths with depth lower than ``max_depth`` are padded with `-1`.
        """
        self._objects = v

    def as_tuple(self):
        # pylint: disable=line-too-long
        """
        Returns the fields as a tuple.
        The returned tuple can be used as input to :class:`~Paths2CIR`.

        Output
        -------
        (mat_t, tau, theta_t, phi_t, thera_r, phi_r) :
            Tuple:

        mat_t : [batch_size, num_rx, rx_array_size, num_tx, tx_array_size, max_num_paths, 2, 2] or [batch_size, num_rx, num_tx, max_num_paths, 2, 2], tf.complex
            Transfer matrices

        tau : [batch_size, num_rx, rx_array_size, num_tx, tx_array_size, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], tf.float
            Propagation delay of each path [s]

        theta_t : [batch_size, num_rx, rx_array_size, num_tx, tx_array_size, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], tf.float
            Zenith  angles of departure [rad]

        phi_t : [batch_size, num_rx, rx_array_size, num_tx, tx_array_size, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], tf.float
            Azimuth  angles of departure [rad]

        theta_r : [batch_size, num_rx, rx_array_size, num_tx, tx_array_size, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], tf.float
            Zenith  angles of arrival [rad]

        phi_r : [batch_size, num_rx, rx_array_size, num_tx, tx_array_size, max_num_paths] or [batch_size, num_rx, num_tx, max_num_paths], tf.float
            Azimuth  angles of arrival [rad]
        """
        return (self.mat_t, self.tau, self.theta_t, self.phi_t, self.theta_r,
                self.phi_r)
