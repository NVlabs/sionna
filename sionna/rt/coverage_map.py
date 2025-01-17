#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Class that stores coverage map
"""
import matplotlib as mpl
from matplotlib.colors import from_levels_and_colors
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sionna.utils import expand_to_rank, insert_dims, log10
from .utils import rotation_matrix, mitsuba_rectangle_to_world, watt_to_dbm
import warnings


class CoverageMap:
    # pylint: disable=line-too-long
    r"""
    CoverageMap()

    Stores the simulated coverage maps

    A coverage map is generated for the loaded scene for all transmitters using
    :meth:`~sionna.rt.Scene.coverage_map`. Please refer to the documentation of this function
    for further details.

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
        cm = scene.coverage_map(max_depth=8)

        # Show coverage map
        cm.show();

    .. figure:: ../figures/coverage_map_show.png
        :align: center
    """

    def __init__(self,
                 center,
                 orientation,
                 size,
                 cell_size,
                 path_gain,
                 scene,
                 dtype=tf.complex64):

        self._rdtype = dtype.real_dtype

        if (tf.rank(center) != 1) or (tf.shape(center)[0] != 3):
            msg = "`center` must be shaped as [x,y,z] (rank=1 and shape=[3])"
            raise ValueError(msg)

        if (tf.rank(orientation) != 1) or (tf.shape(orientation)[0] != 3):
            msg = "`orientation` must be shaped as [a,b,c]"\
                  " (rank=1 and shape=[3])"
            raise ValueError(msg)

        if (tf.rank(size) != 1) or (tf.shape(size)[0] != 2):
            msg = "`size` must be shaped as [w,h]"\
                  " (rank=1 and shape=[2])"
            raise ValueError(msg)

        if (tf.rank(cell_size) != 1) or (tf.shape(cell_size)[0] != 2):
            msg = "`cell_size` must be shaped as [w,h]"\
                  " (rank=1 and shape=[2])"
            raise ValueError(msg)

        num_cells_x = tf.cast(tf.math.ceil(size[0]/cell_size[0]), tf.int32)
        num_cells_y = tf.cast(tf.math.ceil(size[1]/cell_size[1]), tf.int32)

        if (tf.rank(path_gain) != 3)\
                or (tf.shape(path_gain)[1] != num_cells_y)\
                or (tf.shape(path_gain)[2] != num_cells_x):
            msg = "`path_gain` must have shape"\
                  " [num_tx, num_cells_y, num_cells_x]"
            raise ValueError(msg)

        self._center = tf.cast(center, self._rdtype)
        self._orientation = tf.cast(orientation, self._rdtype)
        self._size = tf.cast(size, self._rdtype)
        self._cell_size = tf.cast(cell_size, self._rdtype)
        self._path_gain = tf.cast(path_gain, self._rdtype)
        #self._path_gain = tf.where(tf.math.is_nan(self._path_gain),
        #                           0, self._path_gain)
        self._transmitters = scene.transmitters
        self._scene = scene

        # Dict mapping names to index for transmitters
        self._tx_name_2_ind = {}
        for tx_ind, tx_name in enumerate(self._transmitters):
            self._tx_name_2_ind[tx_name] = tx_ind

        ###############################################################
        # Position of the center of the cells in the world
        # coordinate system
        ###############################################################
        # [num_cells_x]
        x_positions = tf.range(num_cells_x, dtype=self._rdtype)
        x_positions = (x_positions + 0.5)*self._cell_size[0]
        # [num_cells_x, num_cells_y]
        x_positions = tf.expand_dims(x_positions, axis=1)
        x_positions = tf.tile(x_positions, [1, num_cells_y])
        # [num_cells_y]
        y_positions = tf.range(num_cells_y, dtype=self._rdtype)
        y_positions = (y_positions + 0.5)*self._cell_size[1]
        # [num_cells_x, num_cells_y]
        y_positions = tf.expand_dims(y_positions, axis=0)
        y_positions = tf.tile(y_positions, [num_cells_x, 1])
        # [num_cells_x, num_cells_y, 2]
        cell_pos = tf.stack([x_positions, y_positions], axis=-1)
        # Move to global coordinate system
        # [1, 1, 2]
        size = expand_to_rank(self._size, tf.rank(cell_pos), 0)
        # [num_cells_x, num_cells_y, 2]
        cell_pos = cell_pos - size*0.5
        # [num_cells_x, num_cells_y, 3]
        cell_pos = tf.concat([cell_pos,
                              tf.zeros([num_cells_x, num_cells_y, 1],
                                       dtype=self._rdtype)],
                             axis=-1)
        # [3, 3]
        rot_cm_2_gcs = rotation_matrix(self._orientation)
        # [1, 1, 3, 3]
        rot_cm_2_gcs_ = expand_to_rank(rot_cm_2_gcs, tf.rank(cell_pos)+1,
                                       axis=0)
        # [num_cells_x, num_cells_y, 3]
        cell_pos = tf.linalg.matvec(rot_cm_2_gcs_, cell_pos)
        # [num_cells_x, num_cells_y, 3]
        cell_pos = cell_pos + self._center
        # [num_cells_y, num_cells_x, 3]
        cell_pos = tf.transpose(cell_pos, [1, 0, 2])
        self._cell_pos = cell_pos

        ######################################################################
        # Position of the transmitters, receivers, and RIS in the coverage map
        ######################################################################
        # [num_tx/num_rx/num_ris, 3]
        tx_pos = [tx.position for tx in scene.transmitters.values()]
        tx_pos = tf.stack(tx_pos, axis=0)

        rx_pos = [rx.position for rx in scene.receivers.values()]
        rx_pos = tf.stack(rx_pos, axis=0)
        if len(rx_pos) == 0:
            rx_pos = tf.zeros([0, 3], dtype=self._rdtype)

        ris_pos = [ris.position for ris in scene.ris.values()]
        ris_pos = tf.stack(ris_pos, axis=0)
        if len(ris_pos) == 0:
            ris_pos = tf.zeros([0, 3], dtype=self._rdtype)

        # [num_tx/num_rx/num_ris, 3]
        center_ = tf.expand_dims(self._center, axis=0)
        tx_pos = tx_pos - center_
        rx_pos = rx_pos - center_
        ris_pos = ris_pos - center_

        # [3, 3]
        rot_gcs_2_cm = tf.transpose(rot_cm_2_gcs)
        # [1, 3, 3]
        rot_gcs_2_cm_ = tf.expand_dims(rot_gcs_2_cm, axis=0)
        # Positions in the coverage map system
        # [num_tx/num_rx/num_ris, 3]
        tx_pos = tf.linalg.matvec(rot_gcs_2_cm_, tx_pos)
        rx_pos = tf.linalg.matvec(rot_gcs_2_cm_, rx_pos)
        ris_pos = tf.linalg.matvec(rot_gcs_2_cm_, ris_pos)

        # Keep only x and y
        # [num_tx/num_rx/num_ris, 2]
        tx_pos = tx_pos[:, :2]
        rx_pos = rx_pos[:, :2]
        ris_pos = ris_pos[:, :2]

        # Quantizing, using the bottom left corner as origin
        # [num_tx/num_rx/num_ris, 2]
        tx_pos = self._pos_to_idx_cell(tx_pos)
        rx_pos = self._pos_to_idx_cell(rx_pos)
        ris_pos = self._pos_to_idx_cell(ris_pos)

        self._tx_pos = tx_pos
        self._rx_pos = rx_pos
        self._ris_pos = ris_pos

    @property
    def center(self):
        """
        [3], tf.float : Center of the coverage map in the 
            global coordinate system
        """
        return self._center

    @property
    def orientation(self):
        r"""
        [3], tf.float : Orientation of the coverage map
            :math:`(\alpha, \beta, \gamma)`
            specified through three angles corresponding to a 3D rotation
            as defined in :eq:`rotation`.
            An orientation of :math:`(0,0,0)` corresponds to a
            coverage map that is parallel to the XY plane.
        """
        return self._orientation

    @property
    def size(self):
        """
        [2], tf.float : Size of the coverage map
        """
        return self._size

    @property
    def cell_size(self):
        """
        [2], tf.float : Resolution of the coverage map, i.e., width
            (in the local X direction) and height (in the local Y direction) in
            of the cells of the coverage map
        """
        return self._cell_size

    @property
    def cell_centers(self):
        """
        [num_cells_y, num_cells_x, 3], tf.float : Positions of the
        centers of the cells in the global coordinate system
        """
        return self._cell_pos

    @property
    def num_cells_x(self):
        """
        int : Number of cells along the local X-axis
        """
        return self.path_gain.shape[2]

    @property
    def num_cells_y(self):
        """
        int : Number of cells along the local Y-axis
        """
        return self.path_gain.shape[1]

    @property
    def num_tx(self):
        """
        int : Number of transmitters
        """
        return self.path_gain.shape[0]

    @property
    def tx_pos(self):
        """
        [num_tx, 2], int : (column, row) cell index position of each transmitter
        """
        return self._tx_pos

    @property
    def rx_pos(self):
        """
        [num_rx, 2], int : (column, row) cell index position of each receiver
        """
        return self._rx_pos

    @property
    def ris_pos(self):
        """
        [num_ris, 2], int : (column, row) cell index position of each RIS
        """
        return self._ris_pos

    @property
    def path_gain(self):
        """
        [num_tx, num_cells_y, num_cells_x], tf.float : Path gains across the
        coverage map from all transmitters
        """
        return self._path_gain

    @property
    def rss(self):
        """
        [num_tx, num_cells_y, num_cells_x], tf.float : Received signal strength
        (RSS) across the coverage map from all transmitters
        """
        tx_powers = [tx.power for tx in self._scene.transmitters.values()]
        tx_powers = tf.convert_to_tensor(tx_powers)
        return tx_powers[:, tf.newaxis, tf.newaxis] * self.path_gain

    @property
    def sinr(self):
        """
        [num_tx, num_cells_y, num_cells_x], tf.float : SINR
        across the coverage map from all transmitters
        """
        # Total received power from all transmitters
        # [num_tx, num_cells_y, num_cells_x]
        total_pow = tf.reduce_sum(self.rss, axis=0)

        # Interference for each transmitter
        interference = total_pow[tf.newaxis] - self.rss

        # Thermal noise
        noise = self._scene.thermal_noise_power

        # SINR
        return self.rss / (interference + noise)

    def _pos_to_idx_cell(self, pos):
        """
        Convert local position [m] in the coverage map to cell index

        Input
        -----
        pos : [num_pos, 2], tf.float
            Local positions within the coverage map

        Output
        ------
        [num_pos, 2], tf.int32 : Cell index corresponding to each position
        """
        idx_cell = pos + self._size * 0.5
        idx_cell = tf.cast(tf.math.floor(idx_cell / self._cell_size), tf.int32)
        return idx_cell

    def cell_to_tx(self, metric):
        r""" Computes cell-to-transmitter association. Each cell 
        is associated with the transmitter providing the highest
        metric, such as path gain, received signal strength (RSS), or
        SINR.

        Input
        -------
        metric : str, one of ["path_gain", "rss", "sinr"]
            Metric to be used

        Output
        -------
        cell_to_tx : [num_cells_y, num_cells_x], tf.int64
            Cell-to-transmitter association
        """
        # Get tensor for desired metric
        if metric not in ["path_gain", "rss", "sinr"]:
            raise ValueError("Invalid metric")
        cm = getattr(self, metric)
        # Assign each cell to the transmitter guaranteeing the highest metric
        # [num_cells_y, num_cells_x]:
        cell_to_tx = tf.math.argmax(cm, axis=0)

        # No transmitter assignment for the cells with no coverage
        mask = tf.equal(tf.reduce_max(cm, axis=0), 0)
        cell_to_tx = tf.where(
            mask, tf.constant(-1, dtype=cell_to_tx.dtype), cell_to_tx)

        return cell_to_tx

    def cdf(self, metric="path_gain", tx=None):
        r"""Computes and visualizes the CDF of a metric of the coverage map

        Input
        -----
        metric : str, one of ["path_gain", "rss", "sinr"]
            Metric to be shown. Defaults to "path_gain".

        tx : int | str | None
            Index or name of the transmitter for which to show the coverage
            map. If `None`, the maximum value over all transmitters for each
            cell is shown.
            Defaults to `None`.

        Output
        ------
        : :class:`~matplotlib.pyplot.Figure`
            Figure showing the CDF

        x : tf.float, [num_cells_x * num_cells_y]
            Data points for the chosen metric

        cdf : tf.float, [num_cells_x * num_cells_y]
            Cummulative probabilities for the data points
        """
        if metric not in ["path_gain", "rss", "sinr"]:
            raise ValueError("Invalid metric")

        if isinstance(tx, int):
            if tx >= self.num_tx:
                raise ValueError("Invalid transmitter index")
        elif isinstance(tx, str):
            if tx in self._tx_name_2_ind:
                tx = self._tx_name_2_ind[tx]
            else:
                raise ValueError(f"Unknown transmitter with name '{tx}'")
        elif tx is None:
            pass
        else:
            msg = "Invalid type for `tx`: Must be a string, int, or None"
            raise ValueError(msg)

        x = getattr(self, metric)
        if tx is not None:
            x = x[tx]
        else:
            x = tf.reduce_max(x, axis=0)
        x = tf.reshape(x, [-1])
        x = 10 * log10(x)
        # Add 30dB for RSS to acount for dBm
        if metric=="rss":
            x += 30
        x = tf.sort(x)
        cdf = tf.range(1, tf.size(x) + 1, dtype=tf.float32) \
              / tf.cast(tf.size(x), tf.float32)
        fig, _ = plt.subplots()
        plt.plot(x.numpy(), cdf.numpy())
        plt.grid(True, which="both")
        plt.ylabel("Cummulative probability")

        # Set x-label and title
        if metric=="path_gain":
            xlabel = "Path gain [dB]"
            title = "Path gain"
        elif metric=="rss":
            xlabel = "Received signal strength (RSS) [dBm]"
            title = "RSS"
        else:
            xlabel = "Signal-to-interference-plus-noise ratio (SINR) [dB]"
            title = "SINR"
        if (tx is None) & (self.num_tx > 1):
            title = 'Highest ' + title + ' across all TXs'
        elif tx is not None:
            title = title + f' for TX {tx}'

        plt.xlabel(xlabel)
        plt.title(title)

        return fig, x, cdf


    def show(self,
             metric="path_gain",
             tx=None,
             vmin=None,
             vmax=None,
             show_tx=True,
             show_rx=False,
             show_ris=False):
        r"""Visualizes a coverage map

        The position of the transmitter is indicated by a red "+" marker.
        The positions of the receivers are indicated by blue "x" markers.
        The positions of the RIS are indicated by black "*" markers.

        Input
        -----
        metric : str, one of ["path_gain", "rss", "sinr"]
            Metric to be shown. Defaults to "path_gain".

        tx : int | str | None
            Index or name of the transmitter for which to show the coverage
            map. If `None`, the maximum value over all transmitters for each
            cell is shown.
            Defaults to `None`.

        vmin,vmax : float | `None`
            Define the range of values [dB] that the colormap covers.
            If set to `None`, the complete range is shown.
            Defaults to `None`.

        show_tx : bool
            If set to `True`, then the position of the transmitters are shown.
            Defaults to `True`.

        show_rx : bool
            If set to `True`, then the position of the receivers are shown.
            Defaults to `False`.

        show_ris : bool
            If set to `True`, then the position of the RIS are shown.
            Defaults to `False`.

        Output
        ------
        : :class:`~matplotlib.pyplot.Figure`
            Figure showing the coverage map
        """

        if metric not in ["path_gain", "rss", "sinr"]:
            raise ValueError("Invalid metric")

        if isinstance(tx, int):
            if tx >= self.num_tx:
                raise ValueError("Invalid transmitter index")
        elif isinstance(tx, str):
            if tx in self._tx_name_2_ind:
                tx = self._tx_name_2_ind[tx]
            else:
                raise ValueError(f"Unknown transmitter with name '{tx}'")
        elif tx is None:
            pass
        else:
            msg = "Invalid type for `tx`: Must be a string, int, or None"
            raise ValueError(msg)

        # Select metric for a specific transmitter or compute max
        cm = getattr(self, metric)
        if tx is not None:
            cm = cm[tx]
        else:
            cm = tf.reduce_max(cm, axis=0)

        # Convert to dB-scale
        if metric in ["path_gain", "sinr"]:
            with warnings.catch_warnings(record=True) as _:
                # Convert the path gain to dB
                cm = 10.*np.log10(cm.numpy())
        else:
            with warnings.catch_warnings(record=True) as _:
                # Convert the signal strengmth to dBm
                cm = watt_to_dbm(cm).numpy()

        # Visualization the coverage map
        fig_cm = plt.figure()
        plt.imshow(cm, origin='lower', vmin=vmin, vmax=vmax)

        # Set label
        if metric == "path_gain":
            label = "Path gain [dB]"
            title = "Path gain"
        elif metric == "rss":
            label = "Received signal strength (RSS) [dBm]"
            title = 'RSS'
        else:
            label = "Signal-to-interference-plus-noise ratio (SINR) [dB]"
            title = 'SINR'
        if (tx is None) & (self.num_tx > 1):
            title = 'Highest ' + title + ' across all TXs'
        elif tx is not None:
            title = title + f' for TX {tx}'
        plt.colorbar(label=label)
        plt.xlabel('Cell index (X-axis)')
        plt.ylabel('Cell index (Y-axis)')
        plt.title(title)

        # Show transmitter, receiver, RIS positions
        if show_tx:
            if tx is not None:
                tx_pos = self._tx_pos[tx]
                fig_cm.axes[0].scatter(*tx_pos, marker='P', c='r')
            else:
                for tx_pos in self._tx_pos:
                    fig_cm.axes[0].scatter(*tx_pos, marker='P', c='r')

        if show_rx:
            for rx_pos in self._rx_pos:
                fig_cm.axes[0].scatter(*rx_pos, marker='x', c='b')

        if show_ris:
            for ris_pos in self._ris_pos:
                fig_cm.axes[0].scatter(*ris_pos, marker='*', c='k')

        return fig_cm

    def show_association(self,
                         metric="path_gain",
                         show_tx=True,
                         show_rx=False,
                         show_ris=False):
        r"""Visualizes cell-to-tx association for a given metric

        The position of the transmitter is indicated by a red "+" marker.
        The positions of the receivers are indicated by blue "x" markers.
        The positions of the RIS are indicated by black "*" markers.

        Input
        -----
        metric : str, one of ["path_gain", "rss", "sinr"]
            Metric based on which the cell-to-tx association
            is computed.
            Defaults to "path_gain".

        show_tx : bool
            If set to `True`, then the position of the transmitters are shown.
            Defaults to `True`.

        show_rx : bool
            If set to `True`, then the position of the receivers are shown.
            Defaults to `False`.

        show_ris : bool
            If set to `True`, then the position of the RIS are shown.
            Defaults to `False`.

        Output
        ------
        : :class:`~matplotlib.pyplot.Figure`
            Figure showing the cell-to-transmitter association
        """
        if metric not in ["path_gain", "rss", "sinr"]:
            raise ValueError("Invalid metric")

        # Create the colormap and normalization
        colors = mpl.colormaps['Dark2'].colors[:self.num_tx]
        cmap, norm = from_levels_and_colors(
            list(range(self.num_tx+1)), colors)
        fig_tx = plt.figure()
        plt.imshow(self.cell_to_tx(metric).numpy(),
                    origin='lower', cmap=cmap, norm=norm)
        plt.xlabel('Cell index (X-axis)')
        plt.ylabel('Cell index (Y-axis)')
        plt.title('Cell-to-TX association')
        cbar = plt.colorbar(label="TX")
        cbar.ax.get_yaxis().set_ticks([])
        for tx_ in range(self.num_tx):
            cbar.ax.text(.5, tx_ + .5, str(tx_), ha='center', va='center')

        # Visualizing transmitter, receiver, RIS positions
        if show_tx:
            for tx_pos in self._tx_pos:
                fig_tx.axes[0].scatter(*tx_pos, marker='P', c='r')

        if show_rx:
            for rx_pos in self._rx_pos:
                fig_tx.axes[0].scatter(*rx_pos, marker='x', c='b')

        if show_ris:
            for ris_pos in self._ris_pos:
                fig_tx.axes[0].scatter(*ris_pos, marker='*', c='k')

        return fig_tx


    def sample_positions(self,
                         num_pos,
                         metric="path_gain",
                         min_val_db=None,
                         max_val_db=None,
                         min_dist=None,
                         max_dist=None,
                         tx_association=True,
                         center_pos=False):
        # pylint: disable=line-too-long
        r"""Sample random user positions in a scene based on a coverage map

        For a given coverage map, ``num_pos`` random positions are sampled
        around each transmitter,
        such that the selected metric, e.g., SINR, is larger
        than ``min_val_db`` and/or smaller than ``max_val_db``.
        Similarly, ``min_dist`` and ``max_dist`` define the minimum and maximum
        distance of the random positions to the transmitter under consideration.
        By activating the flag ``tx_association``, only positions are sampled
        for which the selected metric is the highest across all transmitters.
        This is useful if one wants to ensure, e.g., that the sampled positions for
        each transmitter provide the highest SINR or RSS.

        Note that due to the quantization of the coverage map into cells it is
        not guaranteed that all above parameters are exactly fulfilled for a
        returned position. This stems from the fact that every
        individual cell of the coverage map describes the expected *average*
        behavior of the surface within this cell. For instance, it may happen
        that half of the selected cell is shadowed and, thus, no path to the
        transmitter exists but the average path gain is still larger than the
        given threshold. Please enable the flag ``center_pos`` to sample only
        positions from the cell centers.

        .. figure:: ../figures/cm_user_sampling.png
            :align: center

        The above figure shows an example for random positions between 220m and
        250m from the transmitter and a maximum path gain of -100 dB.
        Keep in mind that the transmitter can have a different height than the
        coverage map which also contributes to this distance.
        For example if the transmitter is located 20m above the surface of the
        coverage map and a ``min_dist`` of 20m is selected, also positions
        directly below the transmitter are sampled.

        Input
        -----
        num_pos: int
            Number of returned random positions for ech transmitter

        metric : str, one of ["path_gain", "rss", "sinr"]
            Metric to be considered for sampling positions. Defaults to
            "path_gain".

        min_val_db: float | None
            Minimum value for the selected metric ([dB] for path gain and SINR;
            [dBm] for RSS). 
            Positions are only sampled from cells where the selected metric is
            larger than or equal to this value. 
            Ignored if `None`.
            Defaults to `None`.

        max_val_db: float | None
            Maximum value for the selected metric ([dB] for path gain and SINR;
            [dBm] for RSS). 
            Positions are only sampled from cells where the selected metric is
            smaller than or equal to this value. 
            Ignored if `None`.
            Defaults to `None`.

        min_dist: float | None
            Minimum distance [m] from transmitter for all random positions.
            Ignored if `None`.
            Defaults to `None`.

        max_dist: float | None
            Maximum distance [m] from transmitter for all random positions.
            Ignored if `None`.
            Defaults to `None`.

        tx_association : bool
            If `True`, only positions associated with a transmitter are chosen,
            i.e., positions where the chosen metric is the highest among all
            all transmitters. Else, a user located in a sampled position for a
            specific transmitter may perceive a higher metric from another TX.
            Defaults to `True`.

        center_pos: bool
            If `True`, all returned positions are sampled from the cell center
            (i.e., the grid of the coverage map). Otherwise, the positions are
            randomly drawn from the surface of the cell.
            Defaults to `False`.

        Output
        ------
        : [num_tx, num_pos, 3], tf.float
            Random positions :math:`(x,y,z)` [m] that are in cells fulfilling the
            configured constraints

        : [num_tx, num_pos, 2], tf.float
            Cell indices corresponding to the random positions
        """

        if metric not in ["path_gain", "rss", "sinr"]:
            raise ValueError("Invalid metric")

        # allow float values for batch_size
        if not isinstance(num_pos, (int, float)) or not num_pos % 1 == 0:
            raise ValueError("num_pos must be int.")
        # cast batch_size to int
        num_pos = int(num_pos)

        if min_val_db is None:
            min_val_db = -1. * np.infty
        min_val_db = tf.constant(min_val_db, self._rdtype)

        if max_val_db is None:
            max_val_db = np.infty
        max_val_db = tf.constant(max_val_db, self._rdtype)

        if min_val_db > max_val_db:
            raise ValueError("min_val_d cannot be larger than max_val_db.")

        if min_dist is None:
            min_dist = 0.
        min_dist = tf.constant(min_dist, self._rdtype)

        if max_dist is None:
            max_dist = np.infty
        max_dist = tf.constant(max_dist, self._rdtype)

        if min_dist > max_dist:
            raise ValueError("min_dist cannot be larger than max_dist.")

        # Select metric to be used
        cm = getattr(self, metric)

        # Convert to dB-scale
        if metric in ["path_gain", "sinr"]:
            with warnings.catch_warnings(record=True) as _:
                # Convert the path gain to dB
                cm = 10.*np.log10(cm.numpy())
        else:
            with warnings.catch_warnings(record=True) as _:
                # Convert the signal strengmth to dBm
                cm = watt_to_dbm(cm).numpy()

        # [num_tx, 3]: tx_pos_xyz[i, :] contains the i-th tx (x,y,z) coordinate
        # positions
        tx_pos_xyz = tf.stack([tx.position for tx
                               in self._scene.transmitters.values()])

        # Compute distance from each tx to all cells
        # [num_tx, num_cells_y. num_cells_x]
        cell_distance_from_tx = tf.math.reduce_euclidean_norm(
            self.cell_centers[tf.newaxis] -
            insert_dims(tx_pos_xyz, 2, axis=1), axis=-1)

        # [num_tx, num_cells_y. num_cells_x]
        distance_mask = tf.logical_and(cell_distance_from_tx >= min_dist,
                                       cell_distance_from_tx <= max_dist)

        # Get cells for which metric criterion is valid
        # [num_tx, num_cells_y. num_cells_x]
        cm_mask = tf.logical_and(cm >= min_val_db,
                                 cm <= max_val_db)

        # Get cells for which the tx association is valid
        tx_ids = insert_dims(tf.range(self.num_tx, dtype=tf.int64), 2, 1)
        association_mask = tx_ids == self.cell_to_tx(metric)[tf.newaxis]

        # Compute combined mask
        mask = distance_mask & cm_mask
        if tx_association:
            mask = mask & association_mask
        mask = tf.cast(mask, tf.int64)

        sampled_cell_ids = []
        sampled_cell_pos = []
        for i, m in enumerate(mask):
            valid_ids = tf.where(m)
            num_valid_ids = len(valid_ids)
            if num_valid_ids == 0:
                msg = f"No valid cells for transmitter {i} to sample from."
                raise RuntimeError(msg)
            cell_ids = tf.random.uniform(shape=[num_pos],
                                         minval=0, maxval=num_valid_ids,
                                         dtype=tf.int64)
            sampled_ids = tf.gather(valid_ids, cell_ids, axis=0)
            sampled_cell_ids.append(sampled_ids)
            sampled_pos = tf.gather_nd(self.cell_centers,
                                       sampled_ids)
            sampled_cell_pos.append(sampled_pos)
        sampled_cell_ids = tf.stack(sampled_cell_ids, axis=0)
        # swap cell indexes to produce (column, row) index pairs
        sampled_cell_ids = tf.gather(sampled_cell_ids, [1, 0], axis=-1)

        sampled_cell_pos = tf.stack(sampled_cell_pos, axis=0)

        # Add random offset within cell-size, if positions should not be
        # centered
        if not center_pos:
            # cell can be rotated
            dir_x = tf.expand_dims(0.5*(self.cell_centers[0, 0] -
                                        self.cell_centers[1, 0]), axis=0)
            dir_y = tf.expand_dims(0.5*(self.cell_centers[0, 0] -
                                        self.cell_centers[0, 1]), axis=0)
            rand_x = tf.random.uniform((num_pos, 1),
                                       minval=-1.,
                                       maxval=1.,
                                       dtype=self._rdtype)
            rand_y = tf.random.uniform((num_pos, 1),
                                       minval=-1.,
                                       maxval=1.,
                                       dtype=self._rdtype)

            sampled_cell_pos += rand_x * dir_x + rand_y * dir_y

        return sampled_cell_pos, sampled_cell_ids

    def to_world(self):
        r"""
        Returns the `to_world` transformation that maps a default Mitsuba
        rectangle to the rectangle that defines the coverage map surface

        Output
        -------
        to_world : :class:`mitsuba.ScalarTransform4f`
            Rectangle to world transformation
        """
        return mitsuba_rectangle_to_world(self._center, self._orientation,
                                          self._size)
