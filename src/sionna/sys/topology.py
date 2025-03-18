# pylint: disable=too-many-lines, line-too-long, too-many-arguments
# pylint: disable=too-many-locals, too-many-instance-attributes, too-many-positional-arguments
#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""
Multicell topology generation for Sionna SYS
"""

import tensorflow as tf
import matplotlib.pyplot as plt

from sionna.phy.utils import insert_dims, scalar_to_shaped_tensor, \
    flatten_dims, sample_bernoulli
from sionna.phy import PI, config, dtypes, Block, Object
from sionna.phy.channel.utils import random_ut_properties, \
    set_3gpp_scenario_parameters


def get_num_hex_in_grid(num_rings):
    r"""
    Computes the number of hexagons in a spiral hexagonal grid with a given
    number of rings :math:`N`. It equals :math:`1+3N(N+1)`

    Input
    -----
    num_rings : `int`
        Number of rings of the hexagonal spiral grid

    Output
    ------
    : `int`
        Number of hexagons in the spiral hexagonal grid

    """
    return 1 + 3 * num_rings * (num_rings + 1)


def convert_hex_coord(coord,
                      conversion_type,
                      hex_radius=None,
                      precision=None):
    # pylint: disable=line-too-long
    """
    Converts the center coordinates of a hexagon within a grid between any two
    of the types {"offset", "axial", "euclid"}

    Input
    -----
    coord : [..., 2], `tf.int` | `tf.float`
        Coordinates of the center of a hexagon contained in a
        hexagonal grid

    conversion_type : 'offset2euclid' | 'euclid2offset' | 'euclid2axial' | 'offset2axial' | 'axial2offset' | 'axial2euclid' 
        Type of coordinate conversion

    hex_radius : [...], `tf.float` | `None` (default)
        Hexagon radius, i.e., distance between its center and any of
        its corners. It must be specified if ``convert_type`` is `'offset2euclid'`.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Output
    ------
    : [2], `tf.float` | `tf.int`
        Output coordinates
    """
    def inter_center_distance():
        # Inter-center distance between two horizontally adjacent hexagons
        dist_x = hex_radius * tf.cast(1.5, rdtype)
        # Inter-center distance between two vertically adjacent hexagons
        dist_y = hex_radius * tf.cast(tf.sqrt(3.), rdtype)
        return dist_x, dist_y

    if precision is None:
        rdtype = config.tf_rdtype
    else:
        rdtype = dtypes[precision]["tf"]["rdtype"]

    tf.debugging.assert_equal(
        (conversion_type in ["offset2euclid", "euclid2offset",
                             "euclid2axial", "offset2axial",
                             "axial2offset", "axial2euclid"]),
        True,
        message="Invalid convert_type input value. "
                "Must be one of 'offset2euclid', 'euclid2offset', "
                "'euclid2axial', 'offset2axial', "
                "'axial2offset', 'axial2euclid'")

    if conversion_type[:6] == 'euclid':
        coord = tf.cast(coord, rdtype)
    else:
        coord = tf.cast(coord, tf.int32)

    if hex_radius is not None:
        hex_radius = scalar_to_shaped_tensor(hex_radius,
                                             rdtype,
                                             coord.shape[:-1])

    if conversion_type == 'offset2euclid':
        tf.debugging.assert_equal(
            hex_radius is None,
            False,
            message="if convert_type is 'offset2euclid', "
            "then hex_radius must be specified")
        col, row = coord[..., 0], coord[..., 1]
        dist_x, dist_y = inter_center_distance()
        # Euclidean coordinates
        col = tf.cast(col, rdtype)
        row = tf.cast(row, rdtype)
        x = col * dist_x
        y = row * dist_y + (col % 2) * dist_y/2
        coord_out = tf.concat([tf.expand_dims(x, axis=-1),
                               tf.expand_dims(y, axis=-1)], axis=-1)

    elif conversion_type == 'euclid2offset':
        tf.debugging.assert_equal(
            hex_radius is None,
            False,
            message="If convert_type is 'offset2euclid', "
                    "then hex_radius must be specified")
        x, y = coord[..., 0], coord[..., 1]
        dist_x, dist_y = inter_center_distance()
        col = x / dist_x
        row = (y - (col % 2) * dist_y/2) / dist_y
        col, row = tf.cast(col, tf.int32), tf.cast(row, tf.int32)
        coord_out = tf.concat([tf.expand_dims(col, axis=-1),
                               tf.expand_dims(row, axis=-1)], axis=-1)

    elif conversion_type == 'euclid2axial':
        if hex_radius is None:
            raise ValueError("if convert_type=='offset2euclid', "
                             "then hex_radius must be specified")
        coord_offset = convert_hex_coord(coord,
                                         conversion_type='euclid2offset',
                                         hex_radius=hex_radius)
        coord_out = convert_hex_coord(coord_offset,
                                      conversion_type='offset2axial')

    elif conversion_type == 'offset2axial':
        col, row = coord[..., 0], coord[..., 1]
        q = tf.cast(col, tf.int32)
        r = row - tf.cast((col - (col % 2)) / 2, tf.int32)
        coord_out = tf.concat([tf.expand_dims(q, axis=-1),
                               tf.expand_dims(r, axis=-1)], axis=-1)

    elif conversion_type == 'axial2offset':
        q, r = coord[..., 0], coord[..., 1]
        col = tf.cast(q, tf.int32)
        row = r + tf.cast((q - (q % 2)) / 2, tf.int32)
        coord_out = tf.concat([tf.expand_dims(col, axis=-1),
                               tf.expand_dims(row, axis=-1)], axis=-1)

    else:  # convert_type == 'axial2euclid':
        coord_offset = convert_hex_coord(coord,
                                         conversion_type='axial2offset')
        coord_out = convert_hex_coord(coord_offset,
                                      conversion_type='offset2euclid',
                                      hex_radius=hex_radius,
                                      precision=precision)
    return coord_out


class Hexagon(Object):
    """
    Class defining a hexagon placed in a hexagonal grid

    Input
    -----
    radius : `float`
        Hexagon radius, defined as the distance between the hexagon center and
        any of its corners

    coord : [2], `list` | `tuple`
        Coordinates of the hexagon center within the grid. If ``coord_type`` is
        `euclid`, the unit of measurement is meters [m].

    coord_type : 'offset' (default) | 'axial' | 'euclid'
        Coordinate type of ``coord``
    """
    def __init__(self,
                 radius,
                 coord,
                 coord_type='offset',
                 precision=None):
        super().__init__(precision=precision)

        self._coord_offset = None
        self.radius = radius

        if coord_type not in ['offset', 'axial', 'euclid']:
            raise ValueError('Invalid input value for coord_type')

        if coord_type == 'offset':
            self.coord_offset = coord
        elif coord_type == 'axial':
            self.coord_axial = coord
        else:  # coord_type == 'euclid'
            self.coord_euclid = coord

        self._neighbor_axial_directions = \
            tf.convert_to_tensor([[1, 0],
                                  [1, -1],
                                  [0, -1],
                                  [-1, 0],
                                  [-1, 1],
                                  [0, 1]], tf.int32)

    @property
    def coord_offset(self):
        """
        [2], `tf.int32` : Offset coordinates of the hexagon within a grid. The
        first (second, respectively) coordinate defines the horizontal
        (vertical, resp.) offset with respect to the grid center.

        .. figure:: ../figures/offset_coord.png
            :align: center
        """
        return self._coord_offset

    @coord_offset.setter
    def coord_offset(self, value):
        self._coord_offset = tf.cast(value, tf.int32)

        # compute axial coordinates
        self._coord_axial = convert_hex_coord(self.coord_offset,
                                              conversion_type='offset2axial')

        # compute center
        self._coord_euclid = convert_hex_coord(self.coord_offset,
                                               conversion_type='offset2euclid',
                                               hex_radius=self.radius,
                                               precision=self.precision)

    @property
    def coord_axial(self):
        r"""
        [2], `tf.int32` : Axial coordinates of the hexagon within a grid

        .. figure:: ../figures/axial_coord.png
            :align: center

        The basis of axial coordinates are 2D vectors
        :math:`\mathbf{b}^{(1)}=\left(\frac{3}{2}r,\frac{\sqrt{3}}{2}r \right)`,
        :math:`\mathbf{b}^{(2)}=\left(0, \sqrt{3}r \right)`. Thus, the
        relationship between axial coordinates :math:`\mathbf{a}=(a_1,a_2)` and
        their corresponding Euclidean ones :math:`\mathbf{x}=(x_1,x_2)` is the
        following: 

        .. math::
            \mathbf{x} = a_1 \mathbf{b}^{(1)} + a_2 \mathbf{b}^{(2)}

        .. figure:: ../figures/axial_coord_basis.png
            :align: center
            :width: 70%

        """
        return self._coord_axial

    @coord_axial.setter
    def coord_axial(self, value):
        self._coord_axial = tf.cast(value, tf.int32)

        # compute offset coordinates
        self._coord_offset = convert_hex_coord(self.coord_axial,
                                               conversion_type='axial2offset')

        # compute center
        self._coord_euclid = convert_hex_coord(self.coord_offset,
                                               conversion_type='offset2euclid',
                                               hex_radius=self.radius,
                                               precision=self.precision)

    @property
    def coord_euclid(self):
        """
        [2], `tf.float` : Euclidean coordinates of the hexagon within a grid

        .. figure:: ../figures/euclid_coord.png
            :align: center
        """
        return self._coord_euclid

    @coord_euclid.setter
    def coord_euclid(self, value):

        # compute offset coordinates
        self._coord_offset = convert_hex_coord(value,
                                               conversion_type='euclid2offset',
                                               hex_radius=self.radius)

        # convert back to Euclidean coordinates, should the input not belong to
        # the grid
        self._coord_euclid = convert_hex_coord(self.coord_offset,
                                               conversion_type='offset2euclid',
                                               hex_radius=self.radius,
                                               precision=self.precision)

        # compute axial coordinates
        self._coord_axial = convert_hex_coord(self.coord_offset,
                                              conversion_type='offset2axial')

    @property
    def radius(self):
        """
        `tf.float` : Hexagon radius, defined as the distance between its center and
        any of its corners
        """
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = tf.cast(value, self.rdtype)
        if self.coord_offset is not None:
            # update Euclidean coordinates
            self._coord_euclid = convert_hex_coord(self.coord_offset,
                                                   conversion_type='offset2euclid',
                                                   hex_radius=self.radius,
                                                   precision=self.precision)

    def corners(self):
        """
        Computes the Euclidean coordinates of the 6 corners of the hexagon

        Output
        ------
        : [6, 2], `float`
            Euclidean coordinates of the 6 corners of the hexagon
        """
        corners = tf.stack(
            [self.radius * tf.math.cos(tf.cast(tf.range(6),
                                               self.rdtype) * PI/3),
             self.radius * tf.math.sin(tf.cast(tf.range(6),
                                               self.rdtype) * PI/3)],
            axis=1)
        return tf.expand_dims(self.coord_euclid, axis=0) + corners

    def neighbor(self,
                 axial_direction_idx):
        """
        Returns the neighboring hexagon over the specified axial
        direction

        Input
        -----
        axial_direction_idx : `int`
            Index determining the neighbor relative axial direction with respect
            to the current hexagon. Must be one of {0,...,5}.

        Output
        ------
        : :class:`~sionna.system.Hexagon`
            Neighboring hexagon, in the axial relative direction
        """
        neighbor_coord_axial = \
            [self.coord_axial[0] +
             self._neighbor_axial_directions[axial_direction_idx][0],
             self.coord_axial[1] +
             self._neighbor_axial_directions[axial_direction_idx][1]]
        return Hexagon(radius=self.radius,
                       coord=neighbor_coord_axial,
                       coord_type='axial',
                       precision=self.precision)

    def coord_dict(self):
        """ Returns the hexagon coordinates in the form of dictionary

        Output
        ------
        : `dict`
            Dictionary containing the three hexagon coordinates,
            with key 'euclid', 'offset', 'axial'
        """
        return {'euclid': self.coord_euclid,
                'offset': self.coord_offset,
                'axial': self.coord_axial}

# pylint: disable=arguments-differ


class HexGrid(Block):
    """
    Creates a hexagonal spiral grid of cells, drops users uniformly at random in
    it uniformly at random and computes wraparound distances and base station
    positions

    Cell sectors are numbered as follows:

    .. figure:: ../figures/multicell_sectors.png
        :align: center
        :width: 80%

    To eliminate border effects that would cause users at the edge of the grid
    to experience reduced interference, the wraparound principle artificially
    translates each base station to its closest corresponding "mirror" image in
    a neighboring hexagon for each user. 

    .. figure:: ../figures/wraparound.png
        :align: center

    Parameters
    ----------
 
    num_rings : `int`
        Number of spiral rings in the grid
    
    cell_radius : `float` | `None` (default)
        Radius of each hexagonal cell in the grid, defined as the distance
        between the cell center and any of its corners. Either ``isd`` or
        ``cell_radius`` must be specified.

    cell_height : `float` (default: 0.)
        Cell height [m]

    isd : `float` | `None` (default)
        Inter-site distance. Either ``isd`` or ``cell_radius`` must be
        specified.

    center_loc : [2], `list` | `tuple` (default: (0,0))
        Coordinates of the grid center

    center_loc_type : 'offset' (default) | 'axial' | 'euclid'
        Coordinate type of ``center_coord``

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----

    batch_size : `int`
        Batch size

    num_ut_per_sector : `int`
        Number of users to sample per sector and per batch

    min_bs_ut_dist : `float`
        Minimum distance between a base station (BS) and a user [m]

    max_bs_ut_dist : `float` (default: `None`)
        Maximum distance between a base station (BS) and a user [m]. If `None`, it
        is not considered.

    ut_height : `float` (default: 0)
        User height, i.e., distance between the user and the X-Y plane [m]

    Output
    ------

    ut_loc : [batch_size, num_cells, num_sectors=3, num_ut_per_sector, 3], ``tf.float``
        Location of users, dropped uniformly at random within each sector

    mirror_cell_per_ut_loc : [batch_size, num_cells, num_sectors=3, num_ut_per_sector, num_cells, 3], `tf.float`
        Coordinates of the artificial mirror cell centers, located
        at Euclidean distance ``wraparound_dist`` from each user

    wraparound_dist : [batch_size, num_cells, num_sectors=3, num_ut_per_sector, num_cells], `tf.float`
        Wraparound distance in the X-Y plane between each user
        and the cell centers

    Example
    -------
    .. code-block:: python

        from sionna.sys import HexGrid

        # Create a hexagonal grid with a specified radius and number of rings
        grid = HexGrid(cell_radius=1,
                       cell_height=10,
                       num_rings=1,
                       center_loc=(0,0))

        # Cell center locations
        print(grid.cell_loc.numpy())
        # [[ 0.         0.        10.       ]
        #  [-1.5        0.8660254 10.       ]
        #  [ 0.         1.7320508 10.       ]
        #  [ 1.5        0.8660254 10.       ]
        #  [ 1.5       -0.8660254 10.       ]
        #  [ 0.        -1.7320508 10.       ]
        #  [-1.5       -0.8660254 10.       ]]

    """

    def __init__(self,
                 num_rings,
                 cell_radius=None,
                 cell_height=0.,
                 isd=None,
                 center_loc=(0, 0),
                 center_loc_type='offset',
                 precision=None):
        super().__init__(precision=precision)
        if ((cell_radius is None) and (isd is None)) or \
                ((cell_radius is not None) and (isd is not None)):
            raise ValueError("Exactly one of {'cell_radius', 'isd'} "
                             "must be provided as input")
        self._grid = {}
        self._num_rings = None
        self._cell_radius = None
        self._isd = None
        self._cell_height = None
        self._mirror_cell_loc = None
        self._mirror_displacements_offset = None
        self._mirror_displacements_euclid = None
        self._center_loc_type = center_loc_type
        self.center_loc = center_loc
        self.cell_height = cell_height
        if cell_radius is not None:
            self.cell_radius = cell_radius
        if isd is not None:
            self.isd = isd
        self.num_rings = num_rings

    @property
    def grid(self):
        """
        `dict` : Collection of :class:`~sionna.sys.topology.Hexagon` objects
        corresponding to the cells in the grid
        """
        return self._grid

    @property
    def cell_loc(self):
        """
        [num_cells, 3], `float` : Euclidean coordinates of the cell centers [m]
        """
        cell_loc = tf.convert_to_tensor([cell.coord_euclid
                                         for _, cell in self.grid.items()],
                                        dtype=self.rdtype)
        cell_height = tf.fill([cell_loc.shape[0], 1], self.cell_height)
        return tf.concat([cell_loc, cell_height], axis=-1)

    @property
    def center_loc(self):
        """
        [2], `int` | `float` : Grid center coordinates in the X-Y plane, of type
        ``center_loc_type``
        """
        return self._center_loc

    @center_loc.setter
    def center_loc(self, value):
        dtype = self.rdtype if self._center_loc_type == 'euclid' else tf.int32
        self._center_loc = tf.cast(value, dtype)
        if (self.num_rings is not None) & (self.cell_radius is not None):
            self._compute_grid()

    @property
    def num_rings(self):
        """
        `int` : Number of rings of the spiral grid
        """
        return self._num_rings

    @num_rings.setter
    def num_rings(self, value):
        tf.debugging.assert_greater(
            value, 0, message='The number of rings must be positive')
        self._num_rings = value
        if self.cell_radius is not None:
            self._compute_grid()
            self._get_mirror_displacements()
            self._get_mirror_cell_loc()

    @property
    def num_cells(self):
        """
        `int` : Number of cells in the grid
        """
        return len(self.grid)

    @property
    def cell_radius(self):
        """
        `float` : Radius of any hexagonal cell in the grid [m]
        """
        return self._cell_radius

    @cell_radius.setter
    def cell_radius(self, value):
        tf.debugging.assert_positive(
            value,
            message='The call radius must be positive')
        self._cell_radius = tf.cast(value, self.rdtype)
        self._isd = self.cell_radius * tf.cast(tf.math.sqrt(3.), self.rdtype)
        for _, cell in self.grid.items():
            cell.radius = self.cell_radius
        if self._num_rings is not None:
            self._get_mirror_displacements()
            self._get_mirror_cell_loc()

    @property
    def isd(self):
        """
        `float` : Inter-site Euclidean distance [m]
        """
        return self._isd

    @isd.setter
    def isd(self, value):
        tf.debugging.assert_positive(
            value,
            message='The inter-site distance must be positive')
        self._isd = tf.cast(value, self.rdtype)
        self._cell_radius = self.isd / tf.cast(tf.math.sqrt(3.), self.rdtype)
        for _, cell in self.grid.items():
            cell.radius = self.cell_radius
        if self._num_rings is not None:
            self._get_mirror_displacements()
            self._get_mirror_cell_loc()

    @property
    def cell_height(self):
        r"""
        `float` : Cell height [m]
        """
        return self._cell_height

    @cell_height.setter
    def cell_height(self, value):
        tf.debugging.assert_non_negative(
            value,
            message='The cell height must be non-negative')
        self._cell_height = tf.cast(value, self.rdtype)

    @property
    def mirror_cell_loc(self):
        """
        [num_cells, num_mirror_grids+1=7, 3], tf.float : Euclidean (x,y,z) coordinates
        (axis=2) of the 6 mirror + base cells (axis=1) for each base cell (axis=0)
        """
        return self._mirror_cell_loc

    def _get_mirror_cell_loc(self):
        # For each cell (axis=0), returns the coordinates (axis=2) of the
        # corresponding mirror cells (axis=1)
        # [num_cells, num_mirror_grids+1=7, 3]

        # [7, 3]
        mirror_displacements_euclid_3d = tf.concat(
            [self._mirror_displacements_euclid,
             tf.zeros([7, 1], dtype=self.rdtype)],
            axis=-1)
        # [num_cells, 1, 3] + [1, 7, 3]
        self._mirror_cell_loc = tf.expand_dims(self.cell_loc, axis=1) + \
            tf.expand_dims(mirror_displacements_euclid_3d, axis=0)

    def _get_mirror_displacements(self):
        """
        self._mirror_displacements_offset : [7, 2], `tf.int32` 
            2D displacement between the grid center and the mirror grid centers,
            in offset coordinates

        self._mirror_displacements_euclid : [7, 2], `tf.int32`
            2D displacement between the grid center and the mirror grid centers,
            in Euclidean coordinates 
        """
        # [7, 2]
        self._mirror_displacements_offset = tf.convert_to_tensor(
            [[0, 0],
             [2 * self.num_rings + 1, 0],
             [self.num_rings,
             int(3*self.num_rings/2 + 1 - .5*(self.num_rings & 1))],
             [- self.num_rings - 1,
             int(3*self.num_rings/2 + .5*(self.num_rings & 1))],
             [-(2*self.num_rings + 1), -1],
             [- self.num_rings,
             -int(3*self.num_rings/2 + .5*(self.num_rings & 1) + 1)],
             [self.num_rings + 1,
             -int(3*self.num_rings/2 + 1 - .5*(self.num_rings & 1))]],
            dtype=tf.int32)

        # [7, 2]
        self._mirror_displacements_euclid = convert_hex_coord(
            self._mirror_displacements_offset,
            conversion_type='offset2euclid',
            hex_radius=self.cell_radius,
            precision=self.precision)

    def call(self,
             batch_size,
             num_ut_per_sector,
             min_bs_ut_dist,
             max_bs_ut_dist=None,
             min_ut_height=0.,
             max_ut_height=0.):
        # pylint: disable=line-too-long

        min_ut_height = tf.cast(min_ut_height, self.rdtype)
        max_ut_height = tf.cast(max_ut_height, self.rdtype)
        tf.debugging.assert_greater_equal(
            max_ut_height,
            min_ut_height,
            message="max_ut_height must be >= mix_ut_height")

        # Cast to rdtype
        min_bs_ut_dist = tf.cast(min_bs_ut_dist, self.rdtype)
        if max_bs_ut_dist is None:
            max_bs_ut_dist = self.cell_radius
        else:
            max_bs_ut_dist = tf.cast(max_bs_ut_dist, self.rdtype)
        tf.debugging.assert_less_equal(
            min_bs_ut_dist, max_bs_ut_dist,
            message="min_bs_ut_dist must not exceed max_bs_ut_dist")

        # Minimum cell-UT vertical distance
        if (max_ut_height >= self.cell_height) and \
                (min_ut_height <= self.cell_height):
            cell_ut_min_dist_z = tf.cast(0, self.rdtype)
        else:
            cell_ut_min_dist_z = tf.minimum(
                tf.abs(self.cell_height - min_ut_height),
                tf.abs(self.cell_height - max_ut_height))

        # Maximum cell-UT vertical distance
        cell_ut_max_dist_z = tf.maximum(
            tf.abs(self.cell_height - min_ut_height),
            tf.abs(self.cell_height - max_ut_height))

        # Force minimum BS-UT distance >= their height difference
        min_bs_ut_dist = tf.maximum(min_bs_ut_dist, cell_ut_min_dist_z)

        # Minimum squared distance between BS and UT on the X-Y plane
        r_min2 = min_bs_ut_dist**2 - cell_ut_min_dist_z**2

        # Maximum squared distance between BS and UT on the X-Y plane
        r_max2 = max_bs_ut_dist**2 - cell_ut_max_dist_z**2

        # Check the consistency of input parameters
        tf.debugging.assert_less_equal(
            tf.sqrt(r_min2), tf.cast(self.isd / 2, self.rdtype),
            message='The minimum BS-UT distance cannot be larger ' +
                    'than half the inter-site distance')

        # --------#
        # UT drop #
        # --------#
        # Broadcast to [1, num_cells, 1, 1, 3]
        cell_loc_bcast = insert_dims(self.cell_loc, num_dims=1, axis=0)
        cell_loc_bcast = insert_dims(cell_loc_bcast, num_dims=2, axis=2)
        cell_loc_bcast = tf.cast(cell_loc_bcast, self.rdtype)

        # Random angles within half a sector, between [-pi/6; pi/6]
        # [batch_size, num_cells, 3, num_ut_per_sector]
        alpha_half = config.tf_rng.uniform(shape=[batch_size,
                                                  self.num_cells,
                                                  3,  # n. sectors
                                                  num_ut_per_sector],
                                           minval=-PI/6.,
                                           maxval=PI/6.,
                                           dtype=self.rdtype)

        # Maximum distance (on the X-Y plane) from BS to a point in
        # the sector, at each angle in alpha_half
        r_max = tf.cast(self.isd, self.rdtype) / (2*tf.math.cos(alpha_half))
        r_max = tf.minimum(r_max, tf.sqrt(r_max2))

        # To ensure the UT distribution to be uniformly distributed across the
        # sector, we sample positions such that their *squared* distance from
        # the BS is uniformly distributed within (r_min**2, r_max**2)
        distance2 = config.tf_rng.uniform(shape=[batch_size,
                                                 self.num_cells,
                                                 3,
                                                 num_ut_per_sector],
                                          minval=r_min2,
                                          maxval=r_max**2,
                                          dtype=self.rdtype)
        distance = tf.sqrt(distance2)

        # Randomly assign the UTs to one of the two halves of the sector
        side = sample_bernoulli([batch_size, self.num_cells, 3, num_ut_per_sector],
                                tf.cast(0.5, self.rdtype),
                                precision=self.precision)
        side = tf.cast(side, self.rdtype)
        side = 2. * side + 1.
        alpha = alpha_half + side * PI/6.

        # Add an offset to angles alpha depending on the sector they belong to
        alpha_offset = tf.cast([0, 2*PI/3, 4*PI/3], self.rdtype)
        # [1, 1, 3, 1]
        alpha_offset = insert_dims(alpha_offset, num_dims=2, axis=0)
        alpha_offset = insert_dims(alpha_offset, num_dims=1, axis=-1)
        alpha = alpha + alpha_offset

        # Compute UT locations on the X-Y plane
        # [batch_size, num_cells, 3, num_ut_per_sector, 2]
        ut_loc = tf.stack([distance * tf.math.cos(alpha),
                           distance * tf.math.sin(alpha)], axis=-1)
        ut_loc = ut_loc + cell_loc_bcast[..., :2]

        # Add 3rd dimension
        # [batch_size, num_cells, 3, num_ut_per_sector, 3]
        ut_loc_z = config.tf_rng.uniform(shape=ut_loc.shape[:-1] + [1],
                                         minval=min_ut_height,
                                         maxval=max_ut_height,
                                         dtype=self.rdtype)
        ut_loc = tf.concat([ut_loc,
                            ut_loc_z], axis=-1)

        # ------------#
        # Wraparound #
        # ------------#
        # [..., 1, 1, 3]
        ut_loc_bcast = insert_dims(ut_loc,
                                   num_dims=2,
                                   axis=4)

        # [..., num_cells, num_mirror_grids+1=7, 3]
        mirror_loc_bcast = insert_dims(self.mirror_cell_loc,
                                       num_dims=4,
                                       axis=0)
        mirror_loc_bcast = tf.tile(mirror_loc_bcast,
                                   multiples=[batch_size,
                                              self.num_cells,
                                              3,
                                              num_ut_per_sector,
                                              1, 1, 1])

        # Distance between each point and the 6 mirror + 1 base cells
        # [..., num_cells, num_mirror_grids+1=7]
        ut_mirror_cells_dist = tf.norm(
            ut_loc_bcast - tf.cast(mirror_loc_bcast, self.rdtype),
            ord='euclidean',
            axis=-1)

        # Wraparound distance: min across 6 mirror + 1 base cells
        # [..., num_cells]
        wraparound_dist = tf.reduce_min(ut_mirror_cells_dist,
                                        axis=-1)

        # The closest among 6 mirror + 1 base cells for each (UT, base cell)
        # [..., num_cells]
        wraparound_mirror_idx = tf.argmin(ut_mirror_cells_dist,
                                          axis=-1)

        # Coordinates of the cell at wraparound distance for each (UT,
        # base cell)
        # [..., num_cells, 3]
        mirror_cell_per_ut_loc = tf.gather(mirror_loc_bcast,
                                           wraparound_mirror_idx,
                                           axis=-2,
                                           batch_dims=5)
        return ut_loc, mirror_cell_per_ut_loc, wraparound_dist

    def _compute_grid(self):
        """
        Compute the spiral grid of hexagonal cells
        """
        self._grid = {}
        # add the central hexagon
        self._grid[0] = Hexagon(self.cell_radius,
                                coord=self.center_loc,
                                coord_type=self._center_loc_type,
                                precision=self.precision)
        # grid center (axial coordinates)
        grid_center_axial = self.grid[0].coord_axial

        # spiral over concentric circles of radius ring_radius
        hex_key = 1
        for ring_radius in range(1, self.num_rings+1):
            hex_curr = Hexagon(self.cell_radius,
                               coord=(-ring_radius + grid_center_axial[0],
                                      ring_radius + grid_center_axial[1]),
                               coord_type='axial',
                               precision=self.precision)
            # loop over 6 corners
            for ii in range(6):
                # add 'ring_radius' hexagons in the ii-th direction
                for _ in range(ring_radius):
                    # do not add twice the first hexagon
                    self._grid[hex_key] = hex_curr
                    hex_curr = hex_curr.neighbor(axial_direction_idx=ii)
                    hex_key += 1

    def show(self,
             show_mirrors=False,
             show_coord=False,
             show_coord_type='euclid',
             show_sectors=False,
             coord_fontsize=8,
             fig=None,
             color='b',
             label='base'):
        """
        Visualizes the base hexagonal grid and, if specified, the
        mirror grids too

        Note that a mirror grid is a replica of the base grid, repeated
        around its boundaries to enable wraparound.

        Input
        -----

        show_mirrors : `bool`
            If `True`, then the mirror grids are visualized

        show_coord : `bool`
            If `True`, then the hexagon coordinates are visualized

        show_coord_type : `str`
            Type of coordinates to be visualized. Must be one of {'offset',
            'axial', 'euclid'}. Only effective if `show_coord` is `True`

        show_sectors : `bool`
            If `True`, then the three sectors within each hexagon are visualized

        coord_fontsize : `int`
            Coordinate fontsize. Only effective if `show_coord` is `True`

        fig : `matplotlib.figure.Figure` | `None` (default)
            Existing figure handle on which the grid is overlayed.
            If `None`, then a new figure is created

        color : `str` (default: 'b')
            Matplotlib line color

        Output
        ------
        fig : `matplotlib.figure.Figure`
            Figure handle

        """
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.gca()

        if show_mirrors:
            for rr in range(6):
                # Mirror spiral grid
                grid_mirror = HexGrid(
                    cell_radius=self.cell_radius,
                    num_rings=self.num_rings,
                    center_loc=self.center_loc[:2] +
                    self._mirror_displacements_offset[rr+1][:2],
                    center_loc_type='offset',
                    precision=self.precision)
                # Plot mirror grid
                fig = grid_mirror.show(color='r',
                                       fig=fig,
                                       show_mirrors=False,
                                       show_coord=show_coord,
                                       show_coord_type=show_coord_type,
                                       label='mirror' if rr == 0 else None)

        for cell_idx, cell in self.grid.items():
            # Visualize hexagon edges
            corners = cell.corners()
            ax.plot([corners[-1][0]] + [c[0] for c in corners],
                    [corners[-1][1]] + [c[1] for c in corners],
                    color=color)

            # Visualize sectors
            if show_sectors:
                for sector, ii in enumerate([0, 2, 4]):
                    ax.plot([cell.coord_euclid[0], corners[ii][0]],
                            [cell.coord_euclid[1], corners[ii][1]],
                            linestyle='--',
                            color=color)
                    ax.annotate(str(sector + 1),
                                xy=((cell.coord_euclid[0] + corners[ii+1][0]) / 2,
                                    (cell.coord_euclid[1] + corners[ii+1][1]) / 2),
                                horizontalalignment='center',
                                verticalalignment='center')

            # Visualize hexagon coordinates
            if show_coord:
                if show_coord_type == 'euclid':
                    text = f'({cell.coord_dict()[show_coord_type][0]:.1f},' + \
                        f'{cell.coord_dict()[show_coord_type][1]:.1f})'
                else:
                    text = f'({cell.coord_dict()[show_coord_type][0]},' + \
                        f'{cell.coord_dict()[show_coord_type][1]})'
                ax.annotate(text,
                            xy=(cell.coord_euclid[0], cell.coord_euclid[1]),
                            horizontalalignment='center',
                            verticalalignment='center',
                            fontsize=coord_fontsize)
            else:
                ax.plot(*cell.coord_euclid,
                        marker='.',
                        color=color,
                        label=(label + ' cell')
                        if ((label is not None) and (cell_idx == 0))
                        else None)
        ax.set_aspect('equal', adjustable='box')
        ax.legend()
        fig.tight_layout()
        return fig


def gen_hexgrid_topology(batch_size,
                         num_rings,
                         num_ut_per_sector,
                         scenario,
                         min_bs_ut_dist=None,
                         max_bs_ut_dist=None,
                         isd=None,
                         bs_height=None,
                         min_ut_height=None,
                         max_ut_height=None,
                         indoor_probability=None,
                         min_ut_velocity=None,
                         max_ut_velocity=None,
                         downtilt_to_sector_center=True,
                         los=None,
                         return_grid=False,
                         precision=None):
    # pylint: disable=line-too-long
    r"""
    Generates a batch of topologies with hexagonal cells placed on a spiral
    grid, 3 base stations per cell, and user terminals (UT) dropped uniformly at random
    across the cells

    UT orientation and velocity are drawn uniformly randomly within the
    specified bounds, whereas the BSs point toward the center of their respective sector.

    Parameters provided as `None` are set to valid values according to the chosen
    ``scenario`` (see [TR38901]_).

    The returned batch of topologies can be fed into the
    :meth:`~sionna.phy.channel.tr38901.UMa.set_topology` method of the system level models, i.e.
    :class:`~sionna.phy.channel.tr38901.UMi`, :class:`~sionna.phy.channel.tr38901.UMa`,
    and :class:`~sionna.phy.channel.tr38901.RMa`.

    Input
    --------
    batch_size : `int`
        Batch size

    num_ut : `int`
        Number of UTs to sample per batch example

    scenario : "uma" | "umi" | "rma" | "uma-calibration" | "umi-calibration"
        System level model scenario

    min_bs_ut_dist : `None` (default) | `tf.float`
        Minimum BS-UT distance [m]

    max_bs_ut_dist : `None` (default) | `tf.float`
        Maximum BS-UT distance [m]

    isd : `None` (default) | `tf.float`
        Inter-site distance [m]

    bs_height : `None` (default) | `tf.float`
        BS elevation [m]

    min_ut_height : `None` (default) | `tf.float`
        Minimum UT elevation [m]

    max_ut_height : `None` (default) | `tf.float`
        Maximum UT elevation [m]

    indoor_probability : `None` (default) | `tf.float`
        Probability of a UT to be indoor

    min_ut_velocity : `None` (default) | `tf.float`
        Minimum UT velocity [m/s]

    max_ut_velocity : `None` (default) | `tf.float`
        Maximim UT velocity [m/s]

    downtilt_to_sector_center : `bool` (default: `True`)
        If `True`, the BS is mechanically downtilted and points towards the
        sector center. Else, no mechanical downtilting is applied.

    los : `bool` | `None` (default)
         LoS/NLoS states of UTs

    return_grid : `bool` (default: `False`)
        Determines whether the :class:`~sionna.sys.topology.HexGrid` object
        is returned

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Output
    ------
    ut_loc : [batch_size, num_ut, 3], `tf.float`
        UT locations

    bs_loc : [batch_size, num_cells*3, 3], `tf.float`
        BS location

    ut_orientations : [batch_size, num_ut, 3], `tf.float`
        UT orientations [radian]

    bs_orientations : [batch_size, num_cells*3, 3], `tf.float`
        BS orientations [radian]. Oriented toward the center of the sector.

    ut_velocities : [batch_size, num_ut, 3], `tf.float`
        UT velocities [m/s]

    in_state : [batch_size, num_ut], `tf.float`
        Indoor/outdoor state of UTs. `True` means indoor, `False` means
        outdoor.

    los : `None`
        LoS/NLoS states of UTs. This is convenient for directly using the
        function's output as input to
        :meth:`~sionna.phy.channel.SystemLevelScenario.set_topology`, ensuring that
        the LoS/NLoS states adhere to the 3GPP specification (Section 7.4.2 of TR
        38.901). 

    bs_virtual_loc : [batch_size, num_cells*3, num_ut, 3], `tf.float`
        Virtual, i.e., mirror, BS positions for each UT, computed according to
        the wraparound principle 

    grid : :class:`~sionna.sys.topology.HexGrid`
        Hexagonal grid object. Only returned if ``return_grid`` is `True`.

    Example
    -------
    .. code-block:: python

        from sionna.phy.channel.tr38901 import PanelArray, UMi
        from sionna.sys import gen_hexgrid_topology

        # Create antenna arrays
        bs_array = PanelArray(num_rows_per_panel = 4,
                              num_cols_per_panel = 4,
                              polarization = 'dual',
                              polarization_type = 'VH',
                              antenna_pattern = '38.901',
                              carrier_frequency = 3.5e9)

        ut_array = PanelArray(num_rows_per_panel = 1,
                              num_cols_per_panel = 1,
                              polarization = 'single',
                              polarization_type = 'V',
                              antenna_pattern = 'omni',
                              carrier_frequency = 3.5e9)
        # Create channel model
        channel_model = UMi(carrier_frequency = 3.5e9,
                            o2i_model = 'low',
                            ut_array = ut_array,
                            bs_array = bs_array,
                            direction = 'uplink')
        # Generate the topology
        topology = gen_hexgrid_topology(batch_size=100,
                                        num_rings=1,
                                        num_ut_per_sector=3,
                                        scenario='umi')
        # Set the topology
        channel_model.set_topology(*topology)
        channel_model.show_topology()

    .. image:: ../figures/drop_uts_in_hexgrid.png
    """
    # -----------------#
    # 3GPP parameters #
    # -----------------#
    params = set_3gpp_scenario_parameters(scenario,
                                          min_bs_ut_dist,
                                          isd,
                                          bs_height,
                                          min_ut_height,
                                          max_ut_height,
                                          indoor_probability,
                                          min_ut_velocity,
                                          max_ut_velocity,
                                          precision=precision)
    min_bs_ut_dist, isd, bs_height, min_ut_height, max_ut_height, \
        indoor_probability, min_ut_velocity, max_ut_velocity = params

    if precision is None:
        rdtype = config.tf_rdtype
    else:
        rdtype = dtypes[precision]["tf"]["rdtype"]

    # ------------ #
    # BS placement #
    # ------------ #
    grid = HexGrid(isd=isd,
                   cell_height=bs_height,
                   num_rings=num_rings,
                   precision=precision)
    num_cells = grid.num_cells

    # [num_cells*3, 3]
    bs_loc = tf.repeat(grid.cell_loc, 3, axis=0)
    # [1, num_cells*3, 3]
    bs_loc = insert_dims(bs_loc, num_dims=1, axis=0)
    # [batch_size, num_cells*3, 3]
    bs_loc = tf.tile(bs_loc, [batch_size, 1, 1])

    # ----------------#
    # BS orientation #
    # ----------------#
    # Yaw varies according to the sector
    # [num_cells*3]
    bs_yaw = tf.tile([tf.constant(PI/3.0, rdtype),
                      tf.constant(PI, rdtype),
                      tf.constant(5.0*PI/3.0, rdtype)], [num_cells])
    # [1, num_cells*3]
    bs_yaw = insert_dims(bs_yaw, 1, axis=0)
    # [batch_size, num_cells*3]
    bs_yaw = tf.tile(bs_yaw, [batch_size, 1])
    # [batch_size, num_cells*3, 1]
    bs_yaw = insert_dims(bs_yaw, 1, axis=-1)

    # BSs are downtilted towards the sector center
    if downtilt_to_sector_center:
        sector_center = (min_bs_ut_dist + 0.5*isd) * 0.5
        bs_downtilt = 0.5*PI - tf.math.atan(sector_center/bs_height)
    else:
        bs_downtilt = tf.cast(0, rdtype)

    # [batch_size, num_cells*3, 1]
    bs_pitch = tf.fill([batch_size, num_cells*3, 1], bs_downtilt)

    # [batch_size, num_cells*3, 1]
    bs_roll = tf.zeros([batch_size, num_cells*3, 1], rdtype)

    # [batch_size, num_cells*3, 3]
    bs_orientations = tf.concat([bs_yaw, bs_pitch, bs_roll], axis=-1)

    # ----------#
    # Drop UTs #
    # ----------#
    # ut_loc: [batch_size, num_cells, num_sectors, num_ut_per_sector, 3]
    ut_loc, bs_virtual_loc, _ = grid(batch_size,
                                     num_ut_per_sector,
                                     min_bs_ut_dist,
                                     max_bs_ut_dist=max_bs_ut_dist,
                                     min_ut_height=min_ut_height,
                                     max_ut_height=max_ut_height)
    # [batch_size, num_ut, 3]
    ut_loc = flatten_dims(ut_loc, num_dims=3, axis=1)
    num_ut = ut_loc.shape[1]

    # [batch_size, num_ut, num_cells, 3]
    bs_virtual_loc = flatten_dims(bs_virtual_loc, num_dims=3, axis=1)
    # [batch_size, num_ut, num_cells*3, 3]
    bs_virtual_loc = tf.repeat(bs_virtual_loc, 3, axis=2)
    # [batch_size, num_cells*3, num_ut, 3]
    bs_virtual_loc = tf.transpose(bs_virtual_loc, [0, 2, 1, 3])

    # ----------#
    # UT state #
    # ----------#
    # Draw random UT orientation, velocity and indoor state
    ut_orientations, ut_velocities, in_state = \
        random_ut_properties(batch_size,
                              num_ut,
                              indoor_probability,
                              min_ut_velocity,
                              max_ut_velocity,
                              precision=precision)

    if return_grid:
        return ut_loc, bs_loc, ut_orientations, \
            bs_orientations, ut_velocities, in_state, los, bs_virtual_loc, grid
    else:
        return ut_loc, bs_loc, ut_orientations, \
            bs_orientations, ut_velocities, in_state, los, bs_virtual_loc
