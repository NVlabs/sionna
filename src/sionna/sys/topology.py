#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Multicell topology generation for Sionna SYS"""

import math
import warnings
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import torch

from sionna._validation import check_tensor
from sionna.phy import PI, Block, Object, config, dtypes
from sionna.phy.channel.utils import random_ut_properties, set_3gpp_scenario_parameters
from sionna.phy.config import Precision
from sionna.phy.utils import flatten_dims, insert_dims, rand, sample_bernoulli

__all__ = [
    "get_num_hex_in_grid",
    "convert_hex_coord",
    "Hexagon",
    "HexGrid",
    "gen_hexgrid_topology",
    "gen_tr38901_multicell_topology",
    "gen_tr38901_indoor_office_topology",
    "IndoorFactoryTopology",
    "gen_tr38901_indoor_factory_topology",
]


def get_num_hex_in_grid(num_rings: int) -> int:
    r"""Computes the number of hexagons in a spiral hexagonal grid with a given
    number of rings :math:`N`. It equals :math:`1+3N(N+1)`.

    :param num_rings: Number of rings of the hexagonal spiral grid

    :output num_hexagons: Number of hexagons in the spiral hexagonal grid

    .. rubric:: Examples

    .. code-block:: python

        from sionna.sys import get_num_hex_in_grid

        print(get_num_hex_in_grid(1))
        # 7
        print(get_num_hex_in_grid(2))
        # 19
    """
    return 1 + 3 * num_rings * (num_rings + 1)


def convert_hex_coord(
    coord: torch.Tensor,
    conversion_type: str,
    hex_radius: Optional[torch.Tensor] = None,
    precision: Optional[Precision] = None,
    device: Optional[str] = None,
) -> torch.Tensor:
    r"""Converts the center coordinates of a hexagon within a grid between any two
    of the types {"offset", "axial", "euclid"}.

    :param coord: Coordinates of the center of a hexagon contained in a
        hexagonal grid with shape [..., 2]
    :param conversion_type: Type of coordinate conversion. One of
        'offset2euclid', 'euclid2offset', 'euclid2axial', 'offset2axial',
        'axial2offset', 'axial2euclid'.
    :param hex_radius: Hexagon radius, i.e., distance between its center and any of
        its corners with shape [...]. It must be specified if ``conversion_type``
        is 'offset2euclid', 'axial2euclid', 'euclid2offset', or 'euclid2axial'.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :output coord_out: Output coordinates with shape [..., 2]

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.sys import convert_hex_coord

        # Convert offset to Euclidean coordinates
        offset_coord = torch.tensor([1, 2])
        euclid = convert_hex_coord(offset_coord, 'offset2euclid', hex_radius=1.0)
        print(euclid)
        # tensor([1.5000, 4.3301])
    """
    if precision is None:
        dtype = config.dtype
    else:
        dtype = dtypes[precision]["torch"]["dtype"]

    if device is None:
        device = config.device

    def inter_center_distance() -> Tuple[torch.Tensor, torch.Tensor]:
        # Inter-center distance between two horizontally adjacent hexagons
        dist_x = hex_radius * 1.5
        # Inter-center distance between two vertically adjacent hexagons
        dist_y = hex_radius * math.sqrt(3.0)
        return dist_x, dist_y

    valid_types = [
        "offset2euclid",
        "euclid2offset",
        "euclid2axial",
        "offset2axial",
        "axial2offset",
        "axial2euclid",
    ]
    if conversion_type not in valid_types:
        raise ValueError(
            f"Invalid conversion_type. Must be one of {valid_types}"
        )

    if conversion_type.startswith("euclid"):
        coord = coord.to(dtype=dtype, device=device)
    else:
        coord = coord.to(dtype=torch.int32, device=device)

    if hex_radius is not None:
        if not isinstance(hex_radius, torch.Tensor):
            hex_radius = torch.tensor(hex_radius, dtype=dtype, device=device)
        else:
            hex_radius = hex_radius.to(dtype=dtype, device=device)
        # Broadcast to match coord shape (excluding last dim)
        while hex_radius.dim() < coord.dim() - 1:
            hex_radius = hex_radius.unsqueeze(0)

    if conversion_type == "offset2euclid":
        if hex_radius is None:
            raise ValueError("hex_radius must be specified for 'offset2euclid'")
        col, row = coord[..., 0], coord[..., 1]
        dist_x, dist_y = inter_center_distance()
        # Euclidean coordinates
        col_f = col.to(dtype)
        row_f = row.to(dtype)
        x = col_f * dist_x
        y = row_f * dist_y + (col % 2).to(dtype) * dist_y / 2
        coord_out = torch.stack([x, y], dim=-1)

    elif conversion_type == "euclid2offset":
        if hex_radius is None:
            raise ValueError("hex_radius must be specified for 'euclid2offset'")
        x, y = coord[..., 0], coord[..., 1]
        dist_x, dist_y = inter_center_distance()
        col = x / dist_x
        # Use float modulo (matching TF behavior) before casting to int
        row = (y - (col % 2) * dist_y / 2) / dist_y
        col = col.to(torch.int32)
        row = row.to(torch.int32)
        coord_out = torch.stack([col, row], dim=-1)

    elif conversion_type == "euclid2axial":
        if hex_radius is None:
            raise ValueError("hex_radius must be specified for 'euclid2axial'")
        coord_offset = convert_hex_coord(
            coord,
            conversion_type="euclid2offset",
            hex_radius=hex_radius,
            precision=precision,
            device=device,
        )
        coord_out = convert_hex_coord(
            coord_offset,
            conversion_type="offset2axial",
            precision=precision,
            device=device,
        )

    elif conversion_type == "offset2axial":
        col, row = coord[..., 0], coord[..., 1]
        q = col.to(torch.int32)
        r = row - ((col - (col % 2)) // 2).to(torch.int32)
        coord_out = torch.stack([q, r], dim=-1)

    elif conversion_type == "axial2offset":
        q, r = coord[..., 0], coord[..., 1]
        col = q.to(torch.int32)
        row = r + ((q - (q % 2)) // 2).to(torch.int32)
        coord_out = torch.stack([col, row], dim=-1)

    else:  # axial2euclid
        coord_offset = convert_hex_coord(
            coord,
            conversion_type="axial2offset",
            precision=precision,
            device=device,
        )
        coord_out = convert_hex_coord(
            coord_offset,
            conversion_type="offset2euclid",
            hex_radius=hex_radius,
            precision=precision,
            device=device,
        )

    return coord_out


class Hexagon(Object):
    """Class defining a hexagon placed in a hexagonal grid.

    :param radius: Hexagon radius, defined as the distance between the hexagon
        center and any of its corners
    :param coord: Coordinates of the hexagon center within the grid with
        shape [2]. If ``coord_type`` is 'euclid', the unit of measurement
        is meters [m].
    :param coord_type: Coordinate type of ``coord``. One of 'offset'
        (default), 'axial', or 'euclid'.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.
    """

    def __init__(
        self,
        radius: float,
        coord: Union[List[int], Tuple[int, int], torch.Tensor],
        coord_type: str = "offset",
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(precision=precision, device=device)

        self._coord_offset: Optional[torch.Tensor] = None
        self._coord_axial: Optional[torch.Tensor] = None
        self._coord_euclid: Optional[torch.Tensor] = None
        self._radius: Optional[torch.Tensor] = None

        if coord_type not in ["offset", "axial", "euclid"]:
            raise ValueError("Invalid input value for coord_type")

        # Set radius first (needed for coordinate conversions)
        self._radius = torch.tensor(radius, dtype=self.dtype, device=self.device)

        if coord_type == "offset":
            self.coord_offset = coord
        elif coord_type == "axial":
            self.coord_axial = coord
        else:  # coord_type == 'euclid'
            self.coord_euclid = coord

        self._neighbor_axial_directions = torch.tensor(
            [[1, 0], [1, -1], [0, -1], [-1, 0], [-1, 1], [0, 1]],
            dtype=torch.int32,
            device=self.device,
        )

    @property
    def coord_offset(self) -> torch.Tensor:
        """[2], `torch.int32` : Offset coordinates of the hexagon within a grid.
        The first (second) coordinate defines the horizontal (vertical) offset
        with respect to the grid center.

        .. figure:: ../figures/offset_coord.png
            :align: center
        """
        return self._coord_offset

    @coord_offset.setter
    def coord_offset(self, value: Union[List[int], Tuple[int, int], torch.Tensor]) -> None:
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=torch.int32, device=self.device)
        self._coord_offset = value.to(dtype=torch.int32, device=self.device)

        # Compute axial coordinates
        self._coord_axial = convert_hex_coord(
            self._coord_offset,
            conversion_type="offset2axial",
            precision=self.precision,
            device=self.device,
        )

        # Compute Euclidean center
        self._coord_euclid = convert_hex_coord(
            self._coord_offset,
            conversion_type="offset2euclid",
            hex_radius=self._radius,
            precision=self.precision,
            device=self.device,
        )

    @property
    def coord_axial(self) -> torch.Tensor:
        r"""[2], `torch.int32` : Axial coordinates of the hexagon within a grid.

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
    def coord_axial(self, value: Union[List[int], Tuple[int, int], torch.Tensor]) -> None:
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=torch.int32, device=self.device)
        self._coord_axial = value.to(dtype=torch.int32, device=self.device)

        # Compute offset coordinates
        self._coord_offset = convert_hex_coord(
            self._coord_axial,
            conversion_type="axial2offset",
            precision=self.precision,
            device=self.device,
        )

        # Compute Euclidean center
        self._coord_euclid = convert_hex_coord(
            self._coord_offset,
            conversion_type="offset2euclid",
            hex_radius=self._radius,
            precision=self.precision,
            device=self.device,
        )

    @property
    def coord_euclid(self) -> torch.Tensor:
        """[2], `torch.float` : Euclidean coordinates of the hexagon within a grid.

        .. figure:: ../figures/euclid_coord.png
            :align: center
        """
        return self._coord_euclid

    @coord_euclid.setter
    def coord_euclid(self, value: Union[List[float], Tuple[float, float], torch.Tensor]) -> None:
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=self.dtype, device=self.device)
        value = value.to(dtype=self.dtype, device=self.device)

        # Compute offset coordinates
        self._coord_offset = convert_hex_coord(
            value,
            conversion_type="euclid2offset",
            hex_radius=self._radius,
            precision=self.precision,
            device=self.device,
        )

        # Convert back to Euclidean coordinates (snap to grid)
        self._coord_euclid = convert_hex_coord(
            self._coord_offset,
            conversion_type="offset2euclid",
            hex_radius=self._radius,
            precision=self.precision,
            device=self.device,
        )

        # Compute axial coordinates
        self._coord_axial = convert_hex_coord(
            self._coord_offset,
            conversion_type="offset2axial",
            precision=self.precision,
            device=self.device,
        )

    @property
    def radius(self) -> torch.Tensor:
        """`torch.float` : Hexagon radius, defined as the distance between its
        center and any of its corners.
        """
        return self._radius

    @radius.setter
    def radius(self, value: float) -> None:
        self._radius = torch.tensor(value, dtype=self.dtype, device=self.device)
        if self._coord_offset is not None:
            # Update Euclidean coordinates
            self._coord_euclid = convert_hex_coord(
                self._coord_offset,
                conversion_type="offset2euclid",
                hex_radius=self._radius,
                precision=self.precision,
                device=self.device,
            )

    def corners(self) -> torch.Tensor:
        """Computes the Euclidean coordinates of the 6 corners of the hexagon.

        :output corners: Euclidean coordinates of the 6 corners with shape [6, 2],
            `torch.float`
        """
        angles = torch.arange(6, dtype=self.dtype, device=self.device) * PI / 3
        corners = torch.stack(
            [self._radius * torch.cos(angles), self._radius * torch.sin(angles)],
            dim=1,
        )
        return self._coord_euclid.unsqueeze(0) + corners

    def neighbor(self, axial_direction_idx: int) -> "Hexagon":
        """Returns the neighboring hexagon over the specified axial direction.

        :param axial_direction_idx: Index determining the neighbor relative
            axial direction with respect to the current hexagon. Must be one
            of {0,...,5}.

        :output neighbor: :class:`~sionna.sys.topology.Hexagon` -- Neighboring hexagon,
            in the axial relative direction
        """
        neighbor_coord_axial = [
            (self._coord_axial[0] + self._neighbor_axial_directions[axial_direction_idx][0]).item(),
            (self._coord_axial[1] + self._neighbor_axial_directions[axial_direction_idx][1]).item(),
        ]
        return Hexagon(
            radius=self._radius.item(),
            coord=neighbor_coord_axial,
            coord_type="axial",
            precision=self.precision,
            device=self.device,
        )

    def coord_dict(self) -> Dict[str, torch.Tensor]:
        """Returns the hexagon coordinates in the form of a dictionary.

        :output coord_dict: `dict` -- Dictionary containing the three hexagon coordinates,
            with keys 'euclid', 'offset', 'axial'
        """
        return {
            "euclid": self._coord_euclid,
            "offset": self._coord_offset,
            "axial": self._coord_axial,
        }


class HexGrid(Block):
    r"""Creates a hexagonal spiral grid of cells, drops users uniformly at
    random and computes wraparound distances and base station positions.

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

    :param num_rings: Number of spiral rings in the grid
    :param cell_radius: Radius of each hexagonal cell in the grid, defined as
        the distance between the cell center and any of its corners. Either
        ``isd`` or ``cell_radius`` must be specified.
    :param cell_height: Cell height [m]. Defaults to 0.
    :param isd: Inter-site distance. Either ``isd`` or ``cell_radius`` must
        be specified.
    :param center_loc: Coordinates of the grid center with shape [2].
        Defaults to (0, 0).
    :param center_loc_type: Coordinate type of ``center_loc``. One of
        'offset' (default), 'axial', or 'euclid'.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input batch_size: `int`.
        Batch size.
    :input num_ut_per_sector: `int`.
        Number of users to sample per sector and per batch.
    :input min_bs_ut_dist: `float`.
        Minimum distance between a base station (BS) and a user [m].
    :input max_bs_ut_dist: `float` | `None`.
        Maximum distance between a base station (BS) and a user [m]. If
        `None`, it defaults to ``cell_radius``.
    :input min_ut_height: `float`.
        Minimum user height [m]. Defaults to 0.
    :input max_ut_height: `float`.
        Maximum user height [m]. Defaults to 0.

    :output ut_loc: [batch_size, num_cells, num_sectors=3, num_ut_per_sector, 3], `torch.float`.
        Location of users, dropped uniformly at random within each sector.
    :output mirror_cell_per_ut_loc: [batch_size, num_cells, num_sectors=3, num_ut_per_sector, num_cells, 3], `torch.float`.
        Coordinates of the artificial mirror cell centers, located
        at Euclidean distance ``wraparound_dist`` from each user.
    :output wraparound_dist: [batch_size, num_cells, num_sectors=3, num_ut_per_sector, num_cells], `torch.float`.
        Wraparound distance in the X-Y plane between each user
        and the cell centers.

    .. rubric:: Examples

    .. code-block:: python

        from sionna.sys import HexGrid

        # Create a hexagonal grid with a specified radius and number of rings
        grid = HexGrid(cell_radius=1,
                       cell_height=10,
                       num_rings=1,
                       center_loc=(0, 0))

        # Cell center locations
        print(grid.cell_loc)
        # tensor([[ 0.0000,  0.0000, 10.0000],
        #         [-1.5000,  0.8660, 10.0000],
        #         [ 0.0000,  1.7321, 10.0000],
        #         [ 1.5000,  0.8660, 10.0000],
        #         [ 1.5000, -0.8660, 10.0000],
        #         [ 0.0000, -1.7321, 10.0000],
        #         [-1.5000, -0.8660, 10.0000]])
    """

    def __init__(
        self,
        num_rings: int,
        cell_radius: Optional[float] = None,
        cell_height: float = 0.0,
        isd: Optional[float] = None,
        center_loc: Union[List[int], Tuple[int, int]] = (0, 0),
        center_loc_type: str = "offset",
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(precision=precision, device=device)

        if (cell_radius is None and isd is None) or (
            cell_radius is not None and isd is not None
        ):
            raise ValueError(
                "Exactly one of {'cell_radius', 'isd'} must be provided as input"
            )

        self._grid: Dict[int, Hexagon] = {}
        self._num_rings: Optional[int] = None
        self._cell_radius: Optional[torch.Tensor] = None
        self._isd: Optional[torch.Tensor] = None
        self._cell_height: Optional[torch.Tensor] = None
        self._mirror_cell_loc: Optional[torch.Tensor] = None
        self._mirror_displacements_offset: Optional[torch.Tensor] = None
        self._mirror_displacements_euclid: Optional[torch.Tensor] = None
        self._center_loc_type = center_loc_type
        self._center_loc: Optional[torch.Tensor] = None

        self.center_loc = center_loc
        self.cell_height = cell_height
        if cell_radius is not None:
            self.cell_radius = cell_radius
        if isd is not None:
            self.isd = isd
        self.num_rings = num_rings

    @property
    def grid(self) -> Dict[int, Hexagon]:
        """`dict` : Collection of :class:`~sionna.sys.topology.Hexagon` objects
        corresponding to the cells in the grid.
        """
        return self._grid

    @property
    def cell_loc(self) -> torch.Tensor:
        """[num_cells, 3], `torch.float` : Euclidean coordinates of the cell centers [m]."""
        cell_locs = [cell.coord_euclid for _, cell in self._grid.items()]
        cell_loc = torch.stack(cell_locs, dim=0)
        cell_height = self._cell_height.reshape(1, 1).expand(
            cell_loc.shape[0], 1
        )
        return torch.cat([cell_loc, cell_height], dim=-1)

    @property
    def center_loc(self) -> torch.Tensor:
        """[2], `int` | `float` : Grid center coordinates in the X-Y plane,
        of type ``center_loc_type``.
        """
        return self._center_loc

    @center_loc.setter
    def center_loc(self, value: Union[List, Tuple, torch.Tensor]) -> None:
        if self._center_loc_type == "euclid":
            dtype = self.dtype
        else:
            dtype = torch.int32
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=dtype, device=self.device)
        self._center_loc = value.to(dtype=dtype, device=self.device)
        if self._num_rings is not None and self._cell_radius is not None:
            self._compute_grid()
            self._get_mirror_cell_loc()

    @property
    def num_rings(self) -> int:
        """`int` : Number of rings of the spiral grid."""
        return self._num_rings

    @num_rings.setter
    def num_rings(self, value: int) -> None:
        if not (value > 0):
            raise ValueError("The number of rings must be positive")
        self._num_rings = value
        if self._cell_radius is not None:
            self._compute_grid()
            self._get_mirror_displacements()
            self._get_mirror_cell_loc()

    @property
    def num_cells(self) -> int:
        """`int` : Number of cells in the grid."""
        return len(self._grid)

    @property
    def cell_radius(self) -> torch.Tensor:
        """`torch.float` : Radius of any hexagonal cell in the grid [m]."""
        return self._cell_radius

    @cell_radius.setter
    def cell_radius(self, value: float) -> None:
        if not (value > 0):
            raise ValueError("The cell radius must be positive")
        self._cell_radius = torch.tensor(value, dtype=self.dtype, device=self.device)
        self._isd = self._cell_radius * math.sqrt(3.0)
        for _, cell in self._grid.items():
            cell.radius = self._cell_radius.item()
        if self._num_rings is not None:
            self._get_mirror_displacements()
            self._get_mirror_cell_loc()

    @property
    def isd(self) -> torch.Tensor:
        """`torch.float` : Inter-site Euclidean distance [m]."""
        return self._isd

    @isd.setter
    def isd(self, value: float) -> None:
        if not (value > 0):
            raise ValueError("The inter-site distance must be positive")
        self._isd = torch.tensor(value, dtype=self.dtype, device=self.device)
        self._cell_radius = self._isd / math.sqrt(3.0)
        for _, cell in self._grid.items():
            cell.radius = self._cell_radius.item()
        if self._num_rings is not None:
            self._get_mirror_displacements()
            self._get_mirror_cell_loc()

    @property
    def cell_height(self) -> torch.Tensor:
        """`torch.float` : Cell height [m]."""
        return self._cell_height

    @cell_height.setter
    def cell_height(self, value: float) -> None:
        if not (value >= 0):
            raise ValueError("The cell height must be non-negative")
        self._cell_height = torch.tensor(value, dtype=self.dtype, device=self.device)
        if self._mirror_displacements_euclid is not None:
            self._get_mirror_cell_loc()

    @property
    def mirror_cell_loc(self) -> torch.Tensor:
        """[num_cells, num_mirror_grids+1=7, 3], `torch.float` : Euclidean
        (x,y,z) coordinates (axis=2) of the 6 mirror + base cells (axis=1)
        for each base cell (axis=0).
        """
        return self._mirror_cell_loc

    def _get_mirror_cell_loc(self) -> None:
        """For each cell (axis=0), returns the coordinates (axis=2) of the
        corresponding mirror cells (axis=1).
        """
        # [7, 3]
        mirror_displacements_euclid_3d = torch.cat(
            [
                self._mirror_displacements_euclid,
                torch.zeros(7, 1, dtype=self.dtype, device=self.device),
            ],
            dim=-1,
        )
        # [num_cells, 1, 3] + [1, 7, 3]
        self._mirror_cell_loc = (
            self.cell_loc.unsqueeze(1) + mirror_displacements_euclid_3d.unsqueeze(0)
        )

    def _get_mirror_displacements(self) -> None:
        """Computes the 2D displacement between the grid center and the mirror
        grid centers, in both offset and Euclidean coordinates.
        """
        nr = self._num_rings
        # [7, 2]
        self._mirror_displacements_offset = torch.tensor(
            [
                [0, 0],
                [2 * nr + 1, 0],
                [nr, int(3 * nr / 2 + 1 - 0.5 * (nr & 1))],
                [-nr - 1, int(3 * nr / 2 + 0.5 * (nr & 1))],
                [-(2 * nr + 1), -1],
                [-nr, -int(3 * nr / 2 + 0.5 * (nr & 1) + 1)],
                [nr + 1, -int(3 * nr / 2 + 1 - 0.5 * (nr & 1))],
            ],
            dtype=torch.int32,
            device=self.device,
        )

        # [7, 2]
        self._mirror_displacements_euclid = convert_hex_coord(
            self._mirror_displacements_offset,
            conversion_type="offset2euclid",
            hex_radius=self._cell_radius,
            precision=self.precision,
            device=self.device,
        )

    def call(
        self,
        batch_size: int,
        num_ut_per_sector: int,
        min_bs_ut_dist: float,
        max_bs_ut_dist: Optional[float] = None,
        min_ut_height: float = 0.0,
        max_ut_height: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Drops users uniformly at random and computes wraparound distances."""
        if torch.is_tensor(min_ut_height):
            min_ut_height = min_ut_height.detach().clone().to(dtype=self.dtype, device=self.device)
        else:
            min_ut_height = torch.tensor(min_ut_height, dtype=self.dtype, device=self.device)
        if torch.is_tensor(max_ut_height):
            max_ut_height = max_ut_height.detach().clone().to(dtype=self.dtype, device=self.device)
        else:
            max_ut_height = torch.tensor(max_ut_height, dtype=self.dtype, device=self.device)
        check_tensor(
            max_ut_height >= min_ut_height,
            "max_ut_height must be >= min_ut_height",
        )

        # Cast to dtype
        if torch.is_tensor(min_bs_ut_dist):
            min_bs_ut_dist = min_bs_ut_dist.detach().clone().to(dtype=self.dtype, device=self.device)
        else:
            min_bs_ut_dist = torch.tensor(min_bs_ut_dist, dtype=self.dtype, device=self.device)
        if max_bs_ut_dist is None:
            max_bs_ut_dist = self._cell_radius
        elif torch.is_tensor(max_bs_ut_dist):
            max_bs_ut_dist = max_bs_ut_dist.detach().clone().to(dtype=self.dtype, device=self.device)
        else:
            max_bs_ut_dist = torch.tensor(max_bs_ut_dist, dtype=self.dtype, device=self.device)
        check_tensor(
            min_bs_ut_dist <= max_bs_ut_dist,
            "min_bs_ut_dist must not exceed max_bs_ut_dist",
        )

        # Minimum cell-UT vertical distance
        cell_height = self._cell_height
        height_within_range = (
            (max_ut_height >= cell_height) & (cell_height >= min_ut_height)
        )
        cell_ut_min_dist_z = torch.where(
            height_within_range,
            torch.zeros((), dtype=self.dtype, device=self.device),
            torch.minimum(
                torch.abs(cell_height - min_ut_height),
                torch.abs(cell_height - max_ut_height),
            ),
        )

        # Maximum cell-UT vertical distance
        cell_ut_max_dist_z = torch.maximum(
            torch.abs(cell_height - min_ut_height),
            torch.abs(cell_height - max_ut_height),
        )

        # Force minimum BS-UT distance >= their height difference
        min_bs_ut_dist = torch.maximum(min_bs_ut_dist, cell_ut_min_dist_z)

        # Minimum squared distance between BS and UT on the X-Y plane
        r_min2 = min_bs_ut_dist**2 - cell_ut_min_dist_z**2

        # Maximum squared distance between BS and UT on the X-Y plane
        r_max2 = max_bs_ut_dist**2 - cell_ut_max_dist_z**2

        # Check the consistency of input parameters
        check_tensor(
            torch.sqrt(r_min2) <= self._isd / 2,
            "The minimum BS-UT distance cannot be larger than half the "
            "inter-site distance",
        )

        # -------- #
        # UT drop  #
        # -------- #
        # Broadcast to [1, num_cells, 1, 1, 3]
        cell_loc_bcast = insert_dims(self.cell_loc, num_dims=1, axis=0)
        cell_loc_bcast = insert_dims(cell_loc_bcast, num_dims=2, axis=2)
        cell_loc_bcast = cell_loc_bcast.to(self.dtype)

        # Get generator for random numbers
        generator = None if torch.compiler.is_compiling() else self.torch_rng

        # Random angles within half a sector, between [-pi/6; pi/6]
        # [batch_size, num_cells, 3, num_ut_per_sector]
        alpha_half = rand(
            [batch_size, self.num_cells, 3, num_ut_per_sector],
            dtype=self.dtype, device=self.device, generator=generator,
        ) * (PI / 3) - PI / 6

        # Maximum distance (on the X-Y plane) from BS to a point in
        # the sector, at each angle in alpha_half
        r_max = self._isd.to(self.dtype) / (2 * torch.cos(alpha_half))
        r_max = torch.minimum(r_max, torch.sqrt(r_max2))

        # To ensure the UT distribution to be uniformly distributed across the
        # sector, we sample positions such that their *squared* distance from
        # the BS is uniformly distributed within (r_min**2, r_max**2)
        distance2 = rand(
            [batch_size, self.num_cells, 3, num_ut_per_sector],
            dtype=self.dtype, device=self.device, generator=generator,
        ) * (r_max**2 - r_min2) + r_min2
        distance = torch.sqrt(distance2)

        # Randomly assign the UTs to one of the two halves of the sector
        side = sample_bernoulli(
            [batch_size, self.num_cells, 3, num_ut_per_sector],
            0.5,
            precision=self.precision,
            device=self.device,
        ).to(self.dtype)
        side = 2.0 * side + 1.0
        alpha = alpha_half + side * PI / 6

        # Add an offset to angles alpha depending on the sector they belong to
        alpha_offset = torch.tensor(
            [0, 2 * PI / 3, 4 * PI / 3], dtype=self.dtype, device=self.device
        )
        # [1, 1, 3, 1]
        alpha_offset = insert_dims(alpha_offset, num_dims=2, axis=0)
        alpha_offset = insert_dims(alpha_offset, num_dims=1, axis=-1)
        alpha = alpha + alpha_offset

        # Compute UT locations on the X-Y plane
        # [batch_size, num_cells, 3, num_ut_per_sector, 2]
        ut_loc = torch.stack(
            [distance * torch.cos(alpha), distance * torch.sin(alpha)], dim=-1
        )
        ut_loc = ut_loc + cell_loc_bcast[..., :2]

        # Add 3rd dimension
        # [batch_size, num_cells, 3, num_ut_per_sector, 3]
        ut_loc_z = rand(
            [*ut_loc.shape[:-1], 1],
            dtype=self.dtype, device=self.device, generator=generator,
        ) * (max_ut_height - min_ut_height) + min_ut_height
        ut_loc = torch.cat([ut_loc, ut_loc_z], dim=-1)

        # ------------ #
        # Wraparound   #
        # ------------ #
        # [..., 1, 1, 3]
        ut_loc_bcast = insert_dims(ut_loc, num_dims=2, axis=4)

        # [..., num_cells, num_mirror_grids+1=7, 3]
        mirror_loc_bcast = insert_dims(self._mirror_cell_loc, num_dims=4, axis=0)
        mirror_loc_bcast = mirror_loc_bcast.expand(
            batch_size, self.num_cells, 3, num_ut_per_sector, -1, -1, -1
        )

        # Distance between each point and the 6 mirror + 1 base cells
        # [..., num_cells, num_mirror_grids+1=7]
        ut_mirror_cells_dist = torch.norm(
            ut_loc_bcast - mirror_loc_bcast.to(self.dtype),
            p=2,
            dim=-1,
        )

        # Wraparound distance: min across 6 mirror + 1 base cells
        # [..., num_cells]
        wraparound_dist = ut_mirror_cells_dist.min(dim=-1).values

        # The closest among 6 mirror + 1 base cells for each (UT, base cell)
        # [..., num_cells]
        wraparound_mirror_idx = ut_mirror_cells_dist.argmin(dim=-1)

        # Coordinates of the cell at wraparound distance for each (UT, base cell)
        # [..., num_cells, 3]
        # Gather using advanced indexing
        batch_idx = torch.arange(batch_size, device=self.device)
        cell_idx = torch.arange(self.num_cells, device=self.device)
        sector_idx = torch.arange(3, device=self.device)
        ut_idx = torch.arange(num_ut_per_sector, device=self.device)
        cell2_idx = torch.arange(self.num_cells, device=self.device)

        # Create meshgrid for all indices
        b, c, s, u, c2 = torch.meshgrid(
            batch_idx, cell_idx, sector_idx, ut_idx, cell2_idx, indexing="ij"
        )

        mirror_cell_per_ut_loc = mirror_loc_bcast[
            b, c, s, u, c2, wraparound_mirror_idx, :
        ]

        return ut_loc, mirror_cell_per_ut_loc, wraparound_dist

    def _compute_grid(self) -> None:
        """Compute the spiral grid of hexagonal cells."""
        self._grid = {}
        # Add the central hexagon
        self._grid[0] = Hexagon(
            self._cell_radius.item(),
            coord=self._center_loc.tolist(),
            coord_type=self._center_loc_type,
            precision=self.precision,
            device=self.device,
        )
        # Grid center (axial coordinates)
        grid_center_axial = self._grid[0].coord_axial

        # Spiral over concentric circles of radius ring_radius
        hex_key = 1
        for ring_radius in range(1, self._num_rings + 1):
            hex_curr = Hexagon(
                self._cell_radius.item(),
                coord=(
                    -ring_radius + grid_center_axial[0].item(),
                    ring_radius + grid_center_axial[1].item(),
                ),
                coord_type="axial",
                precision=self.precision,
                device=self.device,
            )
            # Loop over 6 corners
            for ii in range(6):
                # Add 'ring_radius' hexagons in the ii-th direction
                for _ in range(ring_radius):
                    self._grid[hex_key] = hex_curr
                    hex_curr = hex_curr.neighbor(axial_direction_idx=ii)
                    hex_key += 1

    def show(
        self,
        show_mirrors: bool = False,
        show_coord: bool = False,
        show_coord_type: str = "euclid",
        show_sectors: bool = False,
        coord_fontsize: int = 8,
        fig: Optional[plt.Figure] = None,
        color: str = "b",
        label: Optional[str] = "base",
    ) -> plt.Figure:
        """Visualizes the base hexagonal grid and, if specified, the mirror
        grids too.

        Note that a mirror grid is a replica of the base grid, repeated
        around its boundaries to enable wraparound.

        :param show_mirrors: If `True`, then the mirror grids are visualized
        :param show_coord: If `True`, then the hexagon coordinates are
            visualized
        :param show_coord_type: Type of coordinates to be visualized. Must be
            one of {'offset', 'axial', 'euclid'}. Only effective if
            ``show_coord`` is `True`.
        :param show_sectors: If `True`, then the three sectors within each
            hexagon are visualized
        :param coord_fontsize: Coordinate fontsize. Only effective if
            ``show_coord`` is `True`.
        :param fig: Existing figure handle on which the grid is overlaid.
            If `None`, then a new figure is created.
        :param color: Matplotlib line color
        :param label: Label for the cells. If `None`, no label is added.

        :output fig: Figure handle
        """
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.gca()

        if show_mirrors:
            for rr in range(6):
                # Mirror spiral grid
                grid_mirror = HexGrid(
                    cell_radius=self._cell_radius.item(),
                    num_rings=self._num_rings,
                    center_loc=(
                        (self._center_loc[:2] + self._mirror_displacements_offset[rr + 1][:2])
                        .tolist()
                    ),
                    center_loc_type="offset",
                    precision=self.precision,
                    device=self.device,
                )
                # Plot mirror grid
                fig = grid_mirror.show(
                    color="r",
                    fig=fig,
                    show_mirrors=False,
                    show_coord=show_coord,
                    show_coord_type=show_coord_type,
                    label="mirror" if rr == 0 else None,
                )

        for cell_idx, cell in self._grid.items():
            # Visualize hexagon edges
            corners = cell.corners().cpu().numpy()
            ax.plot(
                [corners[-1][0]] + [c[0] for c in corners],
                [corners[-1][1]] + [c[1] for c in corners],
                color=color,
            )

            # Visualize sectors
            if show_sectors:
                center = cell.coord_euclid.cpu().numpy()
                for sector, ii in enumerate([0, 2, 4]):
                    ax.plot(
                        [center[0], corners[ii][0]],
                        [center[1], corners[ii][1]],
                        linestyle="--",
                        color=color,
                    )
                    ax.annotate(
                        str(sector + 1),
                        xy=(
                            (center[0] + corners[ii + 1][0]) / 2,
                            (center[1] + corners[ii + 1][1]) / 2,
                        ),
                        horizontalalignment="center",
                        verticalalignment="center",
                    )

            # Visualize hexagon coordinates
            if show_coord:
                center = cell.coord_euclid.cpu().numpy()
                if show_coord_type == "euclid":
                    coord_val = cell.coord_dict()[show_coord_type].cpu().numpy()
                    text = f"({coord_val[0]:.1f},{coord_val[1]:.1f})"
                else:
                    coord_val = cell.coord_dict()[show_coord_type].cpu().numpy()
                    text = f"({coord_val[0]},{coord_val[1]})"
                ax.annotate(
                    text,
                    xy=(center[0], center[1]),
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=coord_fontsize,
                )
            else:
                center = cell.coord_euclid.cpu().numpy()
                ax.plot(
                    *center,
                    marker=".",
                    color=color,
                    label=(label + " cell")
                    if (label is not None) and (cell_idx == 0)
                    else None,
                )
        ax.set_aspect("equal", adjustable="box")
        ax.legend()
        fig.tight_layout()
        return fig


def gen_hexgrid_topology(
    batch_size: int,
    num_rings: int,
    num_ut_per_sector: int,
    scenario: str,
    min_bs_ut_dist: Optional[float] = None,
    max_bs_ut_dist: Optional[float] = None,
    isd: Optional[float] = None,
    bs_height: Optional[float] = None,
    min_ut_height: Optional[float] = None,
    max_ut_height: Optional[float] = None,
    indoor_probability: Optional[float] = None,
    min_ut_velocity: Optional[float] = None,
    max_ut_velocity: Optional[float] = None,
    downtilt_to_sector_center: bool = True,
    los: Optional[bool] = None,
    return_grid: bool = False,
    precision: Optional[Precision] = None,
    device: Optional[str] = None,
) -> Union[
    Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[bool],
        torch.Tensor,
    ],
    Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[bool],
        torch.Tensor,
        HexGrid,
    ],
]:
    r"""Generates UMi/UMa/RMa hexagonal-grid topologies using Tables 7.2-1 and
    7.2-3 of :cite:p:`TR38901V1920`.

    Hexagonal cells are placed on a spiral grid with 3 base stations per cell,
    and user terminals (UTs) are dropped uniformly at random across the cells.

    UT orientation and velocity are drawn uniformly randomly within the
    specified bounds, whereas the base stations point toward the center of their
    respective sector.

    Parameters provided as `None` are set to valid values according to the
    chosen ``scenario``.

    The returned batch of topologies can be fed into the
    :meth:`~sionna.phy.channel.tr38901.UMa.set_topology` method of the system
    level models, i.e.,
    :class:`~sionna.phy.channel.tr38901.UMi`,
    :class:`~sionna.phy.channel.tr38901.UMa`, and
    :class:`~sionna.phy.channel.tr38901.RMa`.

    :param batch_size: Batch size
    :param num_rings: Number of rings in the hexagonal grid
    :param num_ut_per_sector: Number of UTs to sample per sector and per batch
    :param scenario: System level model scenario. One of "uma", "umi", "rma",
        "uma-calibration", "umi-calibration".
    :param min_bs_ut_dist: Minimum BS-UT distance [m]
    :param max_bs_ut_dist: Maximum BS-UT distance [m]
    :param isd: Inter-site distance [m]
    :param bs_height: BS elevation [m]
    :param min_ut_height: Minimum UT elevation [m]
    :param max_ut_height: Maximum UT elevation [m]
    :param indoor_probability: Probability of a UT to be indoor
    :param min_ut_velocity: Minimum UT velocity [m/s]
    :param max_ut_velocity: Maximum UT velocity [m/s]
    :param downtilt_to_sector_center: If `True`, the BS is mechanically
        downtilted and points towards the sector center. Else, no mechanical
        downtilting is applied.
    :param los: LoS/NLoS states of UTs
    :param return_grid: Determines whether the
        :class:`~sionna.sys.topology.HexGrid` object is returned
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :output ut_loc: [batch_size, num_ut, 3], `torch.float`.
        UT locations.
    :output bs_loc: [batch_size, num_cells\*3, 3], `torch.float`.
        BS locations.
    :output ut_orientations: [batch_size, num_ut, 3], `torch.float`.
        UT orientations [radian].
    :output bs_orientations: [batch_size, num_cells\*3, 3], `torch.float`.
        BS orientations [radian]. Oriented toward the center of the sector.
    :output ut_velocities: [batch_size, num_ut, 3], `torch.float`.
        UT velocities [m/s].
    :output in_state: [batch_size, num_ut], `torch.bool`.
        Indoor/outdoor state of UTs. `True` means indoor, `False` means
        non-indoor. On the first RMa topology call,
        :class:`~sionna.phy.channel.tr38901.RMa` interprets every non-indoor UT
        as in-car according to Table 7.2-3. Pass an explicit ``in_car`` mask to
        :meth:`~sionna.phy.channel.tr38901.RMa.set_topology` to model
        unprotected outdoor UTs.
    :output los: `None`.
        LoS/NLoS states of UTs. This is convenient for directly using the
        function's output as input to
        :meth:`~sionna.phy.channel.tr38901.SystemLevelScenario.set_topology`,
        ensuring that the LoS/NLoS states adhere to the 3GPP specification
        (Section 7.4.2 of TR 38.901).
    :output bs_virtual_loc: [batch_size, num_cells\*3, num_ut, 3], `torch.float`.
        Virtual, i.e., mirror, BS positions for each UT, computed according to
        the wraparound principle.
    :output grid: :class:`~sionna.sys.topology.HexGrid`.
        Hexagonal grid object. Only returned if ``return_grid`` is `True`.

    .. rubric:: Examples

    .. code-block:: python

        from sionna.phy.channel.tr38901 import PanelArray, UMi
        from sionna.sys import gen_hexgrid_topology

        # Create antenna arrays
        bs_array = PanelArray(num_rows_per_panel=4,
                              num_cols_per_panel=4,
                              polarization='dual',
                              polarization_type='VH',
                              antenna_pattern='38.901',
                              carrier_frequency=3.5e9)

        ut_array = PanelArray(num_rows_per_panel=1,
                              num_cols_per_panel=1,
                              polarization='single',
                              polarization_type='V',
                              antenna_pattern='omni',
                              carrier_frequency=3.5e9)

        # Create channel model
        channel_model = UMi(carrier_frequency=3.5e9,
                            o2i_model='low',
                            ut_array=ut_array,
                            bs_array=bs_array,
                            direction='uplink')

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
    if precision is None:
        dtype = config.dtype
    else:
        dtype = dtypes[precision]["torch"]["dtype"]

    if device is None:
        device = config.device

    # ----------------- #
    # 3GPP parameters   #
    # ----------------- #
    params = set_3gpp_scenario_parameters(
        scenario,
        min_bs_ut_dist,
        isd,
        bs_height,
        min_ut_height,
        max_ut_height,
        indoor_probability,
        min_ut_velocity,
        max_ut_velocity,
        precision=precision,
        device=device,
    )
    (
        min_bs_ut_dist,
        isd,
        bs_height,
        min_ut_height,
        max_ut_height,
        indoor_probability,
        min_ut_velocity,
        max_ut_velocity,
    ) = params

    # Convert max_bs_ut_dist to tensor if provided as a number
    if max_bs_ut_dist is not None and not isinstance(max_bs_ut_dist, torch.Tensor):
        max_bs_ut_dist = torch.tensor(max_bs_ut_dist, dtype=dtype, device=device)

    # ------------ #
    # BS placement #
    # ------------ #
    grid = HexGrid(
        isd=isd.item(),
        cell_height=bs_height.item(),
        num_rings=num_rings,
        precision=precision,
        device=device,
    )
    num_cells = grid.num_cells

    # [num_cells*3, 3]
    bs_loc = grid.cell_loc.repeat_interleave(3, dim=0)
    # [1, num_cells*3, 3]
    bs_loc = insert_dims(bs_loc, num_dims=1, axis=0)
    # [batch_size, num_cells*3, 3]
    bs_loc = bs_loc.expand(batch_size, -1, -1)

    # ---------------- #
    # BS orientation   #
    # ---------------- #
    # Yaw varies according to the sector
    # [num_cells*3]
    bs_yaw = torch.tensor(
        [PI / 3.0, PI, 5.0 * PI / 3.0], dtype=dtype, device=device
    ).repeat(num_cells)
    # [1, num_cells*3]
    bs_yaw = insert_dims(bs_yaw, 1, axis=0)
    # [batch_size, num_cells*3]
    bs_yaw = bs_yaw.expand(batch_size, -1)
    # [batch_size, num_cells*3, 1]
    bs_yaw = insert_dims(bs_yaw, 1, axis=-1)

    # base stations are downtilted towards the sector center
    if downtilt_to_sector_center:
        sector_center = (min_bs_ut_dist + 0.5 * isd) * 0.5
        bs_downtilt = 0.5 * PI - torch.atan(sector_center / bs_height)
    else:
        bs_downtilt = torch.tensor(0.0, dtype=dtype, device=device)

    # [batch_size, num_cells*3, 1]
    bs_pitch = torch.full(
        (batch_size, num_cells * 3, 1), bs_downtilt.item(), dtype=dtype, device=device
    )

    # [batch_size, num_cells*3, 1]
    bs_roll = torch.zeros(batch_size, num_cells * 3, 1, dtype=dtype, device=device)

    # [batch_size, num_cells*3, 3]
    bs_orientations = torch.cat([bs_yaw, bs_pitch, bs_roll], dim=-1)

    # ---------- #
    # Drop UTs   #
    # ---------- #
    # ut_loc: [batch_size, num_cells, num_sectors, num_ut_per_sector, 3]
    ut_loc, bs_virtual_loc, _ = grid(
        batch_size,
        num_ut_per_sector,
        min_bs_ut_dist.item(),
        max_bs_ut_dist=max_bs_ut_dist.item() if max_bs_ut_dist is not None else None,
        min_ut_height=min_ut_height.item(),
        max_ut_height=max_ut_height.item(),
    )
    # [batch_size, num_ut, 3]
    ut_loc = flatten_dims(ut_loc, num_dims=3, axis=1)
    num_ut = ut_loc.shape[1]

    # [batch_size, num_ut, num_cells, 3]
    bs_virtual_loc = flatten_dims(bs_virtual_loc, num_dims=3, axis=1)
    # [batch_size, num_ut, num_cells*3, 3]
    bs_virtual_loc = bs_virtual_loc.repeat_interleave(3, dim=2)
    # [batch_size, num_cells*3, num_ut, 3]
    bs_virtual_loc = bs_virtual_loc.permute(0, 2, 1, 3)

    # ---------- #
    # UT state   #
    # ---------- #
    # Draw random UT orientation, velocity and indoor state
    ut_orientations, ut_velocities, in_state = random_ut_properties(
        batch_size,
        num_ut,
        indoor_probability,
        min_ut_velocity,
        max_ut_velocity,
        precision=precision,
        device=device,
    )

    if return_grid:
        return (
            ut_loc,
            bs_loc,
            ut_orientations,
            bs_orientations,
            ut_velocities,
            in_state,
            los,
            bs_virtual_loc,
            grid,
        )
    else:
        return (
            ut_loc,
            bs_loc,
            ut_orientations,
            bs_orientations,
            ut_velocities,
            in_state,
            los,
            bs_virtual_loc,
        )


def _tr38901_calibration_defaults(
    scenario: str,
    isd: Optional[float],
    bs_height: Optional[float],
    min_bs_ut_dist: Optional[float],
    indoor_probability: Optional[float],
) -> Tuple[float, float, float, float]:
    """Return scenario defaults for TR 38.901 calibration-style drops."""

    scenario = scenario.lower().replace("-calibration", "")
    if scenario == "umi":
        defaults = (200.0, 10.0, 10.0, 0.8)
    elif scenario == "uma":
        defaults = (500.0, 25.0, 35.0, 0.8)
    elif scenario == "rma":
        defaults = (5000.0, 35.0, 35.0, 0.5)
    else:
        raise ValueError("`scenario` must be one of 'umi', 'uma', or 'rma'")

    isd = defaults[0] if isd is None else isd
    bs_height = defaults[1] if bs_height is None else bs_height
    min_bs_ut_dist = defaults[2] if min_bs_ut_dist is None else min_bs_ut_dist
    indoor_probability = defaults[3] if indoor_probability is None else indoor_probability
    return isd, bs_height, min_bs_ut_dist, indoor_probability


def _tr38901_site_positions(
    num_rings: int,
    isd: float,
    dtype: torch.dtype,
    device: str,
) -> torch.Tensor:
    """Generate the spiral 3GPP calibration site layout."""

    dirs = torch.tensor(
        [
            [1.0, 0.0],
            [0.5, math.sqrt(3.0) / 2.0],
            [-0.5, math.sqrt(3.0) / 2.0],
            [-1.0, 0.0],
            [-0.5, -math.sqrt(3.0) / 2.0],
            [0.5, -math.sqrt(3.0) / 2.0],
        ],
        dtype=dtype,
        device=device,
    ) * isd

    positions = [torch.zeros(2, dtype=dtype, device=device)]
    for ring in range(1, num_rings + 1):
        point = ring * dirs[0]
        for side in range(6):
            move_dir = (side + 2) % 6
            for _ in range(ring):
                positions.append(point.clone())
                point = point + dirs[move_dir]
    return torch.stack(positions, dim=0)


def _tr38901_wrap_vectors(
    num_rings: int,
    isd: float,
    dtype: torch.dtype,
    device: str,
) -> torch.Tensor:
    """Return wraparound translation vectors for the calibration layout."""

    radius = isd / math.sqrt(3.0)
    layer_num = num_rings + 1
    if layer_num == 3:
        values = [
            [0.0, 0.0],
            [0.5 * math.sqrt(3.0), 7.5],
            [4.0 * math.sqrt(3.0), 3.0],
            [3.5 * math.sqrt(3.0), -4.5],
            [-0.5 * math.sqrt(3.0), -7.5],
            [-4.0 * math.sqrt(3.0), -3.0],
            [-3.5 * math.sqrt(3.0), 4.5],
        ]
        cluster = torch.tensor(values, dtype=dtype, device=device) * radius
        vectors = torch.stack([-cluster[:, 1], cluster[:, 0]], dim=-1)
    elif layer_num == 2:
        values = [
            [0.0, 0.0],
            [3.0, 2.0 * math.sqrt(3.0)],
            [4.5, -0.5 * math.sqrt(3.0)],
            [1.5, -2.5 * math.sqrt(3.0)],
            [-3.0, -2.0 * math.sqrt(3.0)],
            [-4.5, 0.5 * math.sqrt(3.0)],
            [-1.5, 2.5 * math.sqrt(3.0)],
        ]
        vectors = torch.tensor(values, dtype=dtype, device=device) * radius
    elif layer_num == 1:
        vectors = torch.zeros((1, 2), dtype=dtype, device=device)
    else:
        raise ValueError("`num_rings` must be 0, 1, or 2")
    return vectors


def _tr38901_drop_sector_ut_batch(
    site_xy: torch.Tensor,
    cell_radius: torch.Tensor,
    min_distance: torch.Tensor,
    boresight: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """Drop UTs uniformly in sector/Voronoi intersections."""

    dtype = site_xy.dtype
    device = site_xy.device
    num_points = site_xy.shape[0]

    # Each 120-degree sector intersects the Voronoi hexagon in a quadrilateral
    # formed by the site and three consecutive hexagon vertices. Triangulate it
    # from the site so that arbitrary triangle areas are handled correctly.
    vertex_offsets = torch.tensor(
        [-math.pi / 3.0, 0.0, math.pi / 3.0],
        dtype=dtype,
        device=device,
    )
    vertex_angles = boresight.unsqueeze(-1) + vertex_offsets
    vertex_vectors = cell_radius * torch.stack(
        [torch.cos(vertex_angles), torch.sin(vertex_angles)], dim=-1
    )
    vertices = site_xy.unsqueeze(1) + vertex_vectors
    triangle_edge_1 = vertex_vectors[:, :-1]
    triangle_edge_2 = vertex_vectors[:, 1:]
    triangle_areas = 0.5 * torch.abs(
        triangle_edge_1[..., 0] * triangle_edge_2[..., 1]
        - triangle_edge_1[..., 1] * triangle_edge_2[..., 0]
    )
    cumulative_areas = torch.cumsum(triangle_areas, dim=-1)

    points = torch.empty(num_points, 2, dtype=dtype, device=device)
    active = torch.ones(num_points, dtype=torch.bool, device=device)

    for _ in range(10000):
        active_idx = torch.where(active)[0]
        if active_idx.numel() == 0:
            return points

        count = active_idx.numel()
        area_draw = torch.rand(
            count, dtype=dtype, device=device, generator=generator
        ) * cumulative_areas[active_idx, -1]
        triangle_idx = torch.sum(
            area_draw.unsqueeze(-1) >= cumulative_areas[active_idx], dim=-1
        ).clamp_max(triangle_areas.shape[-1] - 1)
        vertex_1 = vertices[active_idx, triangle_idx]
        vertex_2 = vertices[active_idx, triangle_idx + 1]

        barycentric = torch.rand(
            count, 2, dtype=dtype, device=device, generator=generator
        )
        barycentric = torch.where(
            (torch.sum(barycentric, dim=-1) > 1.0).unsqueeze(-1),
            1.0 - barycentric,
            barycentric,
        )
        samples = (
            site_xy[active_idx]
            + barycentric[:, :1] * (vertex_1 - site_xy[active_idx])
            + barycentric[:, 1:] * (vertex_2 - site_xy[active_idx])
        )

        ok = torch.linalg.norm(samples - site_xy[active_idx], dim=-1) >= (
            min_distance[active_idx]
        )
        accepted = active_idx[ok]
        points[accepted] = samples[ok]
        active[accepted] = False

    raise RuntimeError("Failed to drop UTs satisfying the minimum distance")


def _tr38901_virtual_bs_locations(
    ut_loc: torch.Tensor,
    site_positions: torch.Tensor,
    bs_height: float,
    wrap_vectors: torch.Tensor,
) -> torch.Tensor:
    """Compute per-UT wrapped BS locations for all sectors."""

    batch_size, num_ut, _ = ut_loc.shape
    num_sites = site_positions.shape[0]
    site_images = (
        site_positions.reshape(1, 1, num_sites, 1, 2)
        + wrap_vectors.reshape(1, 1, 1, -1, 2)
    )
    distances = torch.linalg.norm(
        ut_loc[:, :, :2].reshape(batch_size, num_ut, 1, 1, 2) - site_images,
        dim=-1,
    )
    best = torch.argmin(distances, dim=-1)
    site_idx = torch.arange(num_sites, device=ut_loc.device).reshape(1, 1, num_sites)
    selected_xy = site_images[0, 0, site_idx, best, :]
    z = torch.full(
        (*selected_xy.shape[:-1], 1),
        bs_height,
        dtype=ut_loc.dtype,
        device=ut_loc.device,
    )
    selected_sites = torch.cat([selected_xy, z], dim=-1)
    return selected_sites.repeat_interleave(3, dim=2).permute(0, 2, 1, 3)


def _tr38901_rectangular_site_positions(
    length: float,
    width: float,
    spacing: float,
    dtype: torch.dtype,
    device: str,
) -> torch.Tensor:
    """Return a centered rectangular lattice with the requested spacing."""

    if length <= 0.0:
        raise ValueError("`length` must be positive")
    if width <= 0.0:
        raise ValueError("`width` must be positive")
    if spacing <= 0.0:
        raise ValueError("`spacing` must be positive")

    num_x = max(1, int(math.floor(length / spacing)))
    num_y = max(1, int(math.floor(width / spacing)))
    x_margin = 0.5 * (length - (num_x - 1) * spacing)
    y_margin = 0.5 * (width - (num_y - 1) * spacing)
    x = x_margin + spacing * torch.arange(num_x, dtype=dtype, device=device)
    y = y_margin + spacing * torch.arange(num_y, dtype=dtype, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)


def _tr38901_factory_defaults(
    factory_scenario: str,
) -> Tuple[str, dict[str, float]]:
    """Return Table 7.8-7 InF topology defaults."""

    aliases = {
        "sl": "sl",
        "inf-sl": "sl",
        "sparse-low": "sl",
        "sparse-clutter-low-bs": "sl",
        "dl": "dl",
        "inf-dl": "dl",
        "dense-low": "dl",
        "dense-clutter-low-bs": "dl",
        "sh": "sh",
        "inf-sh": "sh",
        "sparse-high": "sh",
        "sparse-clutter-high-bs": "sh",
        "dh": "dh",
        "inf-dh": "dh",
        "dense-high": "dh",
        "dense-clutter-high-bs": "dh",
    }
    scenario = factory_scenario.strip().lower().replace("_", "-").replace(" ", "-")
    defaults = {
        "sl": {
            "hall_length": 120.0,
            "hall_width": 60.0,
            "hall_height": 10.0,
            "bs_spacing": 20.0,
            "bs_height": 1.5,
            "ut_height": 1.5,
        },
        "dl": {
            "hall_length": 300.0,
            "hall_width": 150.0,
            "hall_height": 10.0,
            "bs_spacing": 50.0,
            "bs_height": 1.5,
            "ut_height": 1.5,
        },
        "sh": {
            "hall_length": 300.0,
            "hall_width": 150.0,
            "hall_height": 10.0,
            "bs_spacing": 50.0,
            "bs_height": 8.0,
            "ut_height": 1.5,
        },
        "dh": {
            "hall_length": 120.0,
            "hall_width": 60.0,
            "hall_height": 10.0,
            "bs_spacing": 20.0,
            "bs_height": 8.0,
            "ut_height": 1.5,
        },
    }
    if scenario not in aliases:
        raise ValueError(
            "`factory_scenario` must be one of 'SL', 'DL', 'SH', or 'DH'. "
            "InF-HH is not part of the TR 38.901 Table 7.8-7 calibration "
            "topology."
        )
    scenario = aliases[scenario]
    return scenario.upper(), defaults[scenario]


def _tr38901_drop_rectangular_ut_xy(
    batch_size: int,
    num_ut: int,
    length: float,
    width: float,
    site_positions: torch.Tensor,
    min_distance: float,
    dtype: torch.dtype,
    device: str,
    generator: torch.Generator,
) -> torch.Tensor:
    """Drop UTs uniformly in a rectangle with a minimum distance to BS sites."""

    if batch_size <= 0:
        raise ValueError("`batch_size` must be positive")
    if num_ut <= 0:
        raise ValueError("`num_ut` must be positive")
    if min_distance < 0.0:
        raise ValueError("`min_distance` must be non-negative")

    xy_flat = torch.empty(batch_size * num_ut, 2, dtype=dtype, device=device)
    active = torch.ones(batch_size * num_ut, dtype=torch.bool, device=device)
    min_distance_t = torch.tensor(min_distance, dtype=dtype, device=device)

    for _ in range(10000):
        active_idx = torch.where(active)[0]
        if active_idx.numel() == 0:
            return xy_flat.reshape(batch_size, num_ut, 2)

        samples = torch.empty(active_idx.numel(), 2, dtype=dtype, device=device)
        samples[:, 0] = length * torch.rand(
            active_idx.numel(), dtype=dtype, device=device, generator=generator
        )
        samples[:, 1] = width * torch.rand(
            active_idx.numel(), dtype=dtype, device=device, generator=generator
        )

        if min_distance <= 0.0:
            ok = torch.ones(active_idx.numel(), dtype=torch.bool, device=device)
        else:
            distance = torch.linalg.norm(
                samples.unsqueeze(1) - site_positions.unsqueeze(0), dim=-1
            )
            ok = distance.min(dim=-1).values >= min_distance_t

        accepted = active_idx[ok]
        xy_flat[accepted] = samples[ok]
        active[accepted] = False

    raise RuntimeError("Failed to drop UTs satisfying the minimum distance")


def _tr38901_unwrapped_virtual_bs_locations(
    bs_loc: torch.Tensor,
    num_ut: int,
) -> torch.Tensor:
    """Broadcast unwrapped BS locations to the per-UT virtual-location tensor."""

    return bs_loc.unsqueeze(2).expand(-1, -1, num_ut, -1).clone()


def gen_tr38901_multicell_topology(
    scenario: str,
    batch_size: int,
    num_ut_per_sector: int,
    carrier_frequency: float,
    num_rings: int = 2,
    use_3gpp_calibration_defaults: bool = True,
    isd: Optional[float] = None,
    bs_height: Optional[float] = None,
    min_bs_ut_dist: Optional[float] = None,
    indoor_probability: Optional[float] = None,
    apply_tr36873_indoor_heights: Optional[bool] = None,
    return_site_positions: bool = False,
    enforce_indoor_distance: bool = True,
    precision: Optional[Precision] = None,
    device: Optional[str] = None,
) -> tuple:
    r"""Generates a TR 38.901 multi-cell topology using Tables 7.8-1 and 7.8-2
    for UMi/UMa and Table 7.2-3 for RMa of
    :cite:p:`TR38901V1920`.

    This helper creates the hexagonal multi-cell topology used for the UMi, UMa,
    and RMa calibration-style scenarios. By default, it generates a two-ring
    layout with 19 sites, three co-located sectors per site, wraparound virtual
    BS locations, and explicit site identifiers for sharing site-level random
    quantities across co-located sectors.

    The sector boresights follow the Table 7.8 convention
    :math:`30^\circ`, :math:`150^\circ`, and :math:`270^\circ`.

    UT locations are generated by a stratified sector drop. For every
    batch item, site, and sector, exactly ``num_ut_per_sector`` UTs are drawn.
    Each drop region is the actual intersection of a 120-degree sector with
    its site's Voronoi hexagon. The intersection is triangulated from the site;
    triangles are selected in proportion to their area and sampled uniformly
    with barycentric coordinates. Samples that violate ``min_bs_ut_dist`` are
    rejected and redrawn. Indoor/outdoor state is drawn before the position.
    For indoor UTs, the rejection distance includes the home-site
    outdoor-to-indoor distance used by the calibration setup. Thus, the drop
    enforces equal UT counts per sector and is uniform over each admissible
    sector/Voronoi intersection.

    .. figure:: ../figures/tr38901_multicell_topology.png
       :align: center
       :width: 90%

       Example one-ring UMi calibration topology with co-located BS sectors
       and stratified UT drops.

    :param scenario: Scenario. One of ``"umi"``, ``"uma"``, or ``"rma"``.
    :param batch_size: Batch size.
    :param num_ut_per_sector: Number of UTs to drop per sector and batch.
    :param carrier_frequency: Carrier frequency [Hz]. For UMi and UMa below
        6 GHz, the indoor distance follows the backward-compatible
        link-specific single-uniform model from Table 7.4.3-3. At 6 GHz and
        above, and for RMa, it is the minimum of two independent uniform
        variables according to Clause 7.4.3.1. Use the same value as for the
        channel model.
    :param num_rings: Number of rings in the hexagonal site layout.
    :param use_3gpp_calibration_defaults: If `True`, missing topology
        parameters are filled with TR 38.901 Table 7.8 calibration defaults.
    :param isd: Inter-site distance [m].
    :param bs_height: BS height [m].
    :param min_bs_ut_dist: Minimum 2D BS-UT distance [m]. If `None`, the
        standard defaults are 10 m for UMi and 35 m for UMa and RMa.
    :param indoor_probability: Probability that a UT is indoor. For RMa, the
        remaining UTs are interpreted as in-car by the channel unless an
        explicit ``in_car`` mask is supplied to
        :meth:`~sionna.phy.channel.tr38901.RMa.set_topology`.
    :param apply_tr36873_indoor_heights: If `True`, indoor UMi/UMa UT heights
        are drawn from the TR 36.873 floor-height model used by the calibration
        drops. If `None`, this is enabled for UMi/UMa and disabled for RMa.
        RMa UT heights are always 1.5 m according to Table 7.2-3.
    :param return_site_positions: If `True`, return
        ``(topology, site_positions)`` instead of only ``topology``. The
        ``topology`` tuple can still be passed directly to
        :meth:`~sionna.phy.channel.tr38901.SystemLevelChannel.set_topology`.
    :param enforce_indoor_distance: If `True`, include the sampled indoor
        distance in the placement rejection threshold. This ensures that the
        outdoor part of every serving link respects ``min_bs_ut_dist``.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :output ut_loc: [batch_size, num_ut, 3], `torch.float`.
        UT locations [m].
    :output bs_loc: [batch_size, num_sites\*3, 3], `torch.float`.
        BS sector locations [m].
    :output ut_orientations: [batch_size, num_ut, 3], `torch.float`.
        UT orientations [radian]. The calibration topology leaves these at zero;
        Phase-2 calibration applies its UT-orientation distribution separately.
    :output bs_orientations: [batch_size, num_sites\*3, 3], `torch.float`.
        BS sector orientations [radian].
    :output ut_velocities: [batch_size, num_ut, 3], `torch.float`.
        UT velocity vectors [m/s]. The calibration topology leaves these at zero.
    :output in_state: [batch_size, num_ut], `torch.bool`.
        Indoor/non-indoor state of UTs. `True` means indoor. For RMa, the
        initial :meth:`~sionna.phy.channel.tr38901.RMa.set_topology` call
        interprets every `False` entry as in-car per Table 7.2-3 unless
        ``in_car`` is supplied explicitly.
    :output los: `None`.
        Placeholder for stochastic LoS/NLoS sampling by the channel model.
    :output bs_virtual_loc: [batch_size, num_sites\*3, num_ut, 3], `torch.float`.
        Wraparound virtual BS sector locations [m].
    :output bs_site_ids: [num_sites\*3], `torch.int64`.
        Site identifier of each BS sector. Co-located sectors share the same
        identifier.
    :output spatial_consistency_track_ids: `None`.
        Placeholder for optional spatial-consistency track identifiers.
    :output distance_2d_in: [batch_size, num_ut] or
        [batch_size, num_sites\*3, num_ut], `torch.float`.
        Indoor 2D distances [m]. UMi and UMa below 6 GHz return link-specific
        distances, with one value shared by the three sectors of each site.
        Other cases return UT-specific distances. Outdoor UTs have zero
        distance. The home-site value is used for placement rejection.
    :output site_positions: [num_sites, 2], `torch.float`.
        Site center positions [m]. Returned separately from ``topology`` only
        if ``return_site_positions`` is `True`.

    .. rubric:: Examples

    .. code-block:: python

        from sionna.sys import gen_tr38901_multicell_topology

        carrier_frequency = 3.5e9
        topology = gen_tr38901_multicell_topology(
            "umi", 1, 2, carrier_frequency)
        channel_model.set_topology(*topology)
    """
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise ValueError("`batch_size` must be a positive integer")
    if (
        not isinstance(num_ut_per_sector, int)
        or isinstance(num_ut_per_sector, bool)
        or num_ut_per_sector <= 0
    ):
        raise ValueError("`num_ut_per_sector` must be a positive integer")
    if (
        not isinstance(num_rings, int)
        or isinstance(num_rings, bool)
        or num_rings not in (0, 1, 2)
    ):
        raise ValueError("`num_rings` must be 0, 1, or 2")

    if precision is None:
        dtype = config.dtype
    else:
        dtype = dtypes[precision]["torch"]["dtype"]
    if device is None:
        device = config.device

    scenario = scenario.lower().replace("-calibration", "")
    if scenario not in ("umi", "uma", "rma"):
        raise ValueError("`scenario` must be one of 'umi', 'uma', or 'rma'")
    if carrier_frequency <= 0.0:
        raise ValueError("`carrier_frequency` must be positive")
    if use_3gpp_calibration_defaults:
        isd, bs_height, min_bs_ut_dist, indoor_probability = (
            _tr38901_calibration_defaults(
                scenario, isd, bs_height, min_bs_ut_dist, indoor_probability
            )
        )
    elif None in (isd, bs_height, min_bs_ut_dist, indoor_probability):
        raise ValueError(
            "`isd`, `bs_height`, `min_bs_ut_dist`, and "
            "`indoor_probability` must be provided when "
            "`use_3gpp_calibration_defaults` is False"
        )

    if isd <= 0.0:
        raise ValueError("`isd` must be positive")
    if bs_height <= 0.0:
        raise ValueError("`bs_height` must be positive")
    if min_bs_ut_dist < 0.0:
        raise ValueError("`min_bs_ut_dist` must be non-negative")
    if not 0.0 <= indoor_probability <= 1.0:
        raise ValueError("`indoor_probability` must be between zero and one")
    if apply_tr36873_indoor_heights is None:
        apply_tr36873_indoor_heights = scenario in ("umi", "uma")
    elif not isinstance(apply_tr36873_indoor_heights, bool):
        raise TypeError("`apply_tr36873_indoor_heights` must be bool or None")
    if scenario == "rma" and apply_tr36873_indoor_heights:
        raise ValueError(
            "RMa UT heights are fixed to 1.5 m by TR 38.901 Table 7.2-3"
        )

    site_positions = _tr38901_site_positions(num_rings, isd, dtype, device)
    num_sites = site_positions.shape[0]
    num_ut = num_sites * 3 * num_ut_per_sector
    cell_radius = torch.tensor(isd / math.sqrt(3.0), dtype=dtype, device=device)
    min_distance = torch.tensor(min_bs_ut_dist, dtype=dtype, device=device)
    indoor_probability_t = torch.tensor(indoor_probability, dtype=dtype, device=device)

    generator = config.torch_rng(device)
    sector_yaws = torch.deg2rad(
        torch.tensor([30.0, 150.0, 270.0], dtype=dtype, device=device)
    )

    per_batch_site_xy = site_positions.repeat_interleave(3*num_ut_per_sector, dim=0)
    per_batch_boresight = sector_yaws.repeat_interleave(num_ut_per_sector)
    per_batch_boresight = per_batch_boresight.repeat(num_sites)
    site_xy_flat = per_batch_site_xy.repeat(batch_size, 1)
    boresight_flat = per_batch_boresight.repeat(batch_size)
    total_ut = batch_size * num_ut

    in_state_flat = (
        torch.rand(total_ut, dtype=dtype, device=device, generator=generator)
        < indoor_probability_t
    )
    in_state = in_state_flat.reshape(batch_size, num_ut)
    max_indoor_distance = 10.0 if scenario == "rma" else 25.0
    legacy_link_specific = (
        scenario in ("umi", "uma") and carrier_frequency < 6e9
    )
    if legacy_link_specific:
        distance_2d_in_by_site = max_indoor_distance * torch.rand(
            batch_size,
            num_sites,
            num_ut,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        distance_2d_in_by_site = torch.where(
            in_state.unsqueeze(1),
            distance_2d_in_by_site,
            torch.zeros_like(distance_2d_in_by_site),
        )
        home_site_ids = torch.arange(
            num_sites, dtype=torch.int64, device=device
        ).repeat_interleave(3 * num_ut_per_sector)
        home_site_index = home_site_ids.reshape(1, 1, num_ut).expand(
            batch_size, 1, -1
        )
        home_distance_2d_in = torch.gather(
            distance_2d_in_by_site, dim=1, index=home_site_index
        ).squeeze(1)
        distance_2d_in = distance_2d_in_by_site.repeat_interleave(3, dim=1)
    else:
        indoor_distance_u1 = torch.rand(
            total_ut, dtype=dtype, device=device, generator=generator
        )
        indoor_distance_u2 = torch.rand(
            total_ut, dtype=dtype, device=device, generator=generator
        )
        indoor_distance_normalized = torch.minimum(
            indoor_distance_u1, indoor_distance_u2
        )
        indoor_distance_flat = max_indoor_distance * indoor_distance_normalized
        indoor_distance_flat = torch.where(
            in_state_flat,
            indoor_distance_flat,
            torch.zeros_like(indoor_distance_flat),
        )
        home_distance_2d_in = indoor_distance_flat.reshape(batch_size, num_ut)
        distance_2d_in = home_distance_2d_in
    if enforce_indoor_distance:
        effective_min_dist = (
            min_distance + home_distance_2d_in.reshape(total_ut)
        )
    else:
        effective_min_dist = (
            torch.zeros_like(home_distance_2d_in).reshape(total_ut)
            + min_distance
        )
    xy_flat = _tr38901_drop_sector_ut_batch(
        site_xy_flat,
        cell_radius,
        effective_min_dist,
        boresight_flat,
        generator,
    )

    if apply_tr36873_indoor_heights:
        num_floors = torch.randint(
            4,
            9,
            (total_ut,),
            dtype=torch.int64,
            device=device,
            generator=generator,
        )
        floor = torch.floor(
            torch.rand(total_ut, dtype=dtype, device=device, generator=generator)
            * num_floors.to(dtype)
        ) + 1.0
        indoor_height = 3.0 * (floor - 1.0) + 1.5
        height_flat = torch.where(
            in_state_flat,
            indoor_height,
            torch.full_like(indoor_height, 1.5),
        )
    else:
        height_flat = torch.full((total_ut,), 1.5, dtype=dtype, device=device)

    ut_loc = torch.cat([xy_flat, height_flat.unsqueeze(-1)], dim=-1)
    ut_loc = ut_loc.reshape(batch_size, num_ut, 3)

    bs_site_loc = torch.cat(
        [
            site_positions,
            torch.full((num_sites, 1), bs_height, dtype=dtype, device=device),
        ],
        dim=-1,
    )
    bs_loc = bs_site_loc.repeat_interleave(3, dim=0).reshape(1, num_sites * 3, 3)
    bs_loc = bs_loc.expand(batch_size, -1, -1).clone()

    bs_yaw = sector_yaws.repeat(num_sites)
    bs_yaw = bs_yaw.reshape(1, num_sites * 3, 1).expand(batch_size, -1, -1)
    bs_orientations = torch.cat(
        [
            bs_yaw,
            torch.zeros(batch_size, num_sites * 3, 2, dtype=dtype, device=device),
        ],
        dim=-1,
    )

    ut_orientations = torch.zeros(batch_size, num_ut, 3, dtype=dtype, device=device)
    ut_velocities = torch.zeros(batch_size, num_ut, 3, dtype=dtype, device=device)

    wrap_vectors = _tr38901_wrap_vectors(num_rings, isd, dtype, device)
    bs_virtual_loc = _tr38901_virtual_bs_locations(
        ut_loc, site_positions, bs_height, wrap_vectors
    )
    bs_site_ids = torch.arange(num_sites, dtype=torch.int64, device=device)
    bs_site_ids = bs_site_ids.repeat_interleave(3)

    output = (
        ut_loc,
        bs_loc,
        ut_orientations,
        bs_orientations,
        ut_velocities,
        in_state,
        None,
        bs_virtual_loc,
        bs_site_ids,
        None,
        distance_2d_in,
    )
    if return_site_positions:
        return output, site_positions
    return output


def gen_tr38901_indoor_office_topology(
    batch_size: int,
    num_ut_per_sector: int,
    room_length: float = 120.0,
    room_width: float = 50.0,
    room_height: float = 3.0,
    isd: float = 20.0,
    bs_height: Optional[float] = None,
    ut_height: float = 1.0,
    min_bs_ut_dist: float = 0.0,
    return_site_positions: bool = False,
    precision: Optional[Precision] = None,
    device: Optional[str] = None,
) -> tuple:
    r"""Generates a TR 38.901 indoor-office topology using Tables 7.2-2 and
    7.8-1 of :cite:p:`TR38901V1920`.

    With the default arguments, the room has size
    :math:`120\,\mathrm{m}\times 50\,\mathrm{m}\times 3\,\mathrm{m}`, the
    inter-site distance is 20 m, and 12 ceiling-mounted BS sites are generated.
    Each site has three co-located sectors with azimuth orientations
    :math:`30^\circ`, :math:`150^\circ`, and :math:`270^\circ`. UTs are dropped
    uniformly over the room and marked as indoor.

    The returned tuple can be passed directly to
    :meth:`~sionna.phy.channel.tr38901.InH.set_topology`.

    .. figure:: ../figures/tr38901_indoor_office_topology.png
       :align: center
       :width: 85%

       Example indoor-office topology with ceiling-mounted BS sites and indoor
       UT drops.

    The topology shown in the figure was generated with:

    .. code-block:: python

        from sionna.phy import config
        from sionna.sys import gen_tr38901_indoor_office_topology

        config.seed = 42
        topology, site_positions = gen_tr38901_indoor_office_topology(
            batch_size=1,
            num_ut_per_sector=1,
            return_site_positions=True,
            precision="single",
            device="cpu")

    :param batch_size: Batch size.
    :param num_ut_per_sector: Number of UTs to drop per sector and batch.
        The total number of UTs is
        ``num_sites*3*num_ut_per_sector``.
    :param room_length: Room length along the x-axis [m].
    :param room_width: Room width along the y-axis [m].
    :param room_height: Room height [m].
    :param isd: Spacing between neighboring BS sites [m].
    :param bs_height: BS height [m]. If `None`, ``room_height`` is used.
    :param ut_height: UT height [m].
    :param min_bs_ut_dist: Minimum 2D distance between each UT and BS site [m].
    :param return_site_positions: If `True`, return
        ``(topology, site_positions)`` instead of only ``topology``. The
        ``topology`` tuple can still be passed directly to
        :meth:`~sionna.phy.channel.tr38901.SystemLevelChannel.set_topology`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :output ut_loc: [batch_size, num_ut, 3], `torch.float`.
        UT locations [m].
    :output bs_loc: [batch_size, num_sites*3, 3], `torch.float`.
        BS sector locations [m].
    :output ut_orientations: [batch_size, num_ut, 3], `torch.float`.
        UT orientations [radian].
    :output bs_orientations: [batch_size, num_sites*3, 3], `torch.float`.
        BS sector orientations [radian].
    :output ut_velocities: [batch_size, num_ut, 3], `torch.float`.
        UT velocity vectors [m/s].
    :output in_state: [batch_size, num_ut], `torch.bool`.
        Indoor state of UTs. Always `True`.
    :output los: `None`.
        Placeholder for stochastic LoS/NLoS sampling by the channel model.
    :output bs_virtual_loc: [batch_size, num_sites*3, num_ut, 3], `torch.float`.
        Virtual BS sector locations [m]. No wraparound is applied.
    :output bs_site_ids: [num_sites*3], `torch.int64`.
        Site identifier of each BS sector. Co-located sectors share the same
        identifier.
    :output site_positions: [num_sites, 2], `torch.float`.
        BS site center positions [m]. Returned separately from ``topology``
        only if ``return_site_positions`` is `True`.

    .. rubric:: Examples

    .. code-block:: python

        from sionna.phy.channel.tr38901 import InH
        from sionna.sys import gen_tr38901_indoor_office_topology

        topology = gen_tr38901_indoor_office_topology(1, 2)
        channel_model = InH(carrier_frequency, ut_array, bs_array, "downlink")
        channel_model.set_topology(*topology)
    """

    if precision is None:
        dtype = config.dtype
    else:
        dtype = dtypes[precision]["torch"]["dtype"]
    if device is None:
        device = config.device
    if bs_height is None:
        bs_height = room_height

    site_positions = _tr38901_rectangular_site_positions(
        room_length, room_width, isd, dtype, device
    )
    num_sites = site_positions.shape[0]
    num_sectors = 3
    num_bs = num_sites * num_sectors
    num_ut = num_bs * num_ut_per_sector

    generator = config.torch_rng(device)
    ut_xy = _tr38901_drop_rectangular_ut_xy(
        batch_size,
        num_ut,
        room_length,
        room_width,
        site_positions,
        min_bs_ut_dist,
        dtype,
        device,
        generator,
    )
    ut_loc = torch.cat(
        [
            ut_xy,
            torch.full((batch_size, num_ut, 1), ut_height,
                       dtype=dtype, device=device),
        ],
        dim=-1,
    )

    bs_site_loc = torch.cat(
        [
            site_positions,
            torch.full((num_sites, 1), bs_height, dtype=dtype, device=device),
        ],
        dim=-1,
    )
    bs_loc_single = bs_site_loc.repeat_interleave(num_sectors, dim=0)
    bs_loc = bs_loc_single.unsqueeze(0).expand(batch_size, -1, -1).clone()

    sector_yaws = torch.deg2rad(
        torch.tensor([30.0, 150.0, 270.0], dtype=dtype, device=device)
    )
    bs_yaw = sector_yaws.repeat(num_sites)
    bs_yaw = bs_yaw.reshape(1, num_bs, 1).expand(batch_size, -1, -1)
    bs_orientations = torch.cat(
        [bs_yaw, torch.zeros(batch_size, num_bs, 2, dtype=dtype, device=device)],
        dim=-1,
    )

    ut_orientations = torch.zeros(batch_size, num_ut, 3, dtype=dtype, device=device)
    ut_velocities = torch.zeros(batch_size, num_ut, 3, dtype=dtype, device=device)
    in_state = torch.ones(batch_size, num_ut, dtype=torch.bool, device=device)
    bs_virtual_loc = _tr38901_unwrapped_virtual_bs_locations(bs_loc, num_ut)
    bs_site_ids = torch.arange(num_sites, dtype=torch.int64, device=device)
    bs_site_ids = bs_site_ids.repeat_interleave(num_sectors)

    output = (
        ut_loc,
        bs_loc,
        ut_orientations,
        bs_orientations,
        ut_velocities,
        in_state,
        None,
        bs_virtual_loc,
        bs_site_ids,
    )
    if return_site_positions:
        return output, site_positions
    return output


class IndoorFactoryTopology(tuple):
    r"""Tuple-compatible indoor-factory topology with InF configuration
    metadata.

    Instances contain the same nine entries accepted by
    :meth:`~sionna.phy.channel.tr38901.InF.set_topology`. They additionally
    record the factory sub-scenario and hall dimensions used to generate the
    deployment. Use :meth:`set_topology` to validate this metadata against an
    InF channel model before applying the topology.

    This class is returned by
    :func:`~sionna.sys.gen_tr38901_indoor_factory_topology` and is not intended
    to be constructed directly.
    """

    def __new__(
        cls,
        values: tuple,
        factory_scenario: str,
        hall_dimensions: Tuple[float, float, float],
        warn_on_unpack: bool = False,
    ) -> "IndoorFactoryTopology":
        instance = super().__new__(cls, values)
        instance._factory_scenario = factory_scenario
        instance._hall_dimensions = hall_dimensions
        instance._warn_on_unpack = warn_on_unpack
        return instance

    @property
    def factory_scenario(self) -> str:
        """Canonical InF sub-scenario, e.g., ``"SH"``."""
        return self._factory_scenario

    @property
    def hall_dimensions(self) -> Tuple[float, float, float]:
        """Hall dimensions ``(length, width, height)`` [m]."""
        return self._hall_dimensions

    def __iter__(self):
        if self._warn_on_unpack:
            warnings.warn(
                "Directly unpacking an IndoorFactoryTopology with non-default "
                "hall dimensions bypasses InF hall-dimension validation. Use "
                "topology.set_topology(channel_model) instead.",
                UserWarning,
                stacklevel=2,
            )
        return super().__iter__()

    def __getnewargs_ex__(self) -> tuple:
        return (
            (
                self[:],
                self.factory_scenario,
                self.hall_dimensions,
                self._warn_on_unpack,
            ),
            {},
        )

    def set_topology(self, channel_model: object) -> None:
        r"""Validate and apply this topology to an InF channel model.

        :param channel_model: :class:`~sionna.phy.channel.tr38901.InF` channel
            model configured with the same ``factory_scenario`` and
            ``hall_dimensions`` as this topology.

        :raises TypeError: If ``channel_model`` is not an InF channel model.
        :raises ValueError: If its factory sub-scenario or hall dimensions do
            not match this topology.
        """
        scenario = getattr(channel_model, "_scenario", None)
        setter = getattr(channel_model, "set_topology", None)
        if (
            scenario is None
            or not callable(setter)
            or not hasattr(scenario, "factory_scenario")
            or not hasattr(scenario, "hall_dimensions")
        ):
            raise TypeError("`channel_model` must be an InF channel model")

        channel_factory_scenario = str(scenario.factory_scenario).upper()
        if channel_factory_scenario != self.factory_scenario:
            raise ValueError(
                "The InF channel model uses factory scenario "
                f"'{channel_factory_scenario}', but the topology was generated "
                f"for '{self.factory_scenario}'."
            )

        channel_hall_dimensions = torch.as_tensor(scenario.hall_dimensions)
        expected_hall_dimensions = torch.tensor(
            self.hall_dimensions,
            dtype=channel_hall_dimensions.dtype,
            device=channel_hall_dimensions.device,
        )
        if (
            channel_hall_dimensions.shape != expected_hall_dimensions.shape
            or not torch.equal(channel_hall_dimensions, expected_hall_dimensions)
        ):
            actual = channel_hall_dimensions.detach().cpu().tolist()
            raise ValueError(
                "The InF channel model uses hall_dimensions="
                f"{actual}, but the topology was generated with "
                f"hall_dimensions={self.hall_dimensions}. Construct InF with "
                "hall_dimensions=topology.hall_dimensions."
            )

        setter(*self[:])


def gen_tr38901_indoor_factory_topology(
    factory_scenario: str,
    batch_size: int,
    num_ut: int,
    hall_length: Optional[float] = None,
    hall_width: Optional[float] = None,
    hall_height: Optional[float] = None,
    bs_spacing: Optional[float] = None,
    bs_height: Optional[float] = None,
    ut_height: Optional[float] = None,
    min_bs_ut_dist: float = 1.0,
    return_site_positions: bool = False,
    precision: Optional[Precision] = None,
    device: Optional[str] = None,
) -> Union[IndoorFactoryTopology, Tuple[IndoorFactoryTopology, torch.Tensor]]:
    r"""Generates a TR 38.901 indoor-factory topology using Tables 7.2-4 and
    7.8-7 of :cite:p:`TR38901V1920`.

    The supported factory sub-scenarios are ``"SL"`` (sparse clutter, low BS
    height), ``"DL"`` (dense clutter, low BS height), ``"SH"`` (sparse
    clutter, high BS height), and ``"DH"`` (dense clutter, high BS height).
    These are the sub-scenarios included in the large-scale calibration
    assumptions. ``"HH"`` is a valid
    :class:`~sionna.phy.channel.tr38901.InF` sub-scenario, but it is not part
    of the Table 7.8-7 calibration topology. With default parameters, 18 base stations
    are placed on a rectangular lattice with spacing :math:`D` and offset
    :math:`D/2` from the hall walls. UTs are dropped uniformly inside the hall,
    marked as indoor, and constrained by ``min_bs_ut_dist``.

    The default calibration deployments use
    :math:`L\times W = 120\,\mathrm{m}\times 60\,\mathrm{m}` and
    :math:`D=20\,\mathrm{m}` for ``"SL"`` and ``"DH"``, and
    :math:`L\times W = 300\,\mathrm{m}\times 150\,\mathrm{m}` and
    :math:`D=50\,\mathrm{m}` for ``"DL"`` and ``"SH"``. The BS height is
    :math:`1.5\,\mathrm{m}` for the low-BS cases and
    :math:`8\,\mathrm{m}` for the high-BS cases.

    The returned :class:`IndoorFactoryTopology` remains tuple-compatible and
    can be passed directly to
    :meth:`~sionna.phy.channel.tr38901.InF.set_topology`. It also records the
    resolved ``factory_scenario`` and ``hall_dimensions``. For custom hall
    dimensions, construct ``InF`` with ``topology.hall_dimensions`` and call
    ``topology.set_topology(channel_model)`` to validate the channel's
    statistics configuration before applying the deployment. Directly
    unpacking a topology with non-default hall dimensions remains supported
    but emits a warning because it bypasses this validation.

    .. figure:: ../figures/tr38901_indoor_factory_topology.png
       :align: center
       :width: 85%

       Example InF-SH indoor-factory topology with a rectangular BS lattice
       and indoor UT drops.

    The topology shown in the figure was generated with:

    .. code-block:: python

        from sionna.phy import config
        from sionna.sys import gen_tr38901_indoor_factory_topology

        config.seed = 42
        topology, site_positions = gen_tr38901_indoor_factory_topology(
            "SH",
            batch_size=1,
            num_ut=80,
            return_site_positions=True,
            precision="single",
            device="cpu")

    :param factory_scenario: Indoor-factory sub-scenario. Must be ``"SL"``,
        ``"DL"``, ``"SH"``, or ``"DH"``.
    :param batch_size: Batch size.
    :param num_ut: Number of UTs to drop per batch.
    :param hall_length: Hall length along the x-axis [m]. If `None`, the
        Table 7.8-7 default for ``factory_scenario`` is used.
    :param hall_width: Hall width along the y-axis [m]. If `None`, the
        Table 7.8-7 default for ``factory_scenario`` is used.
    :param hall_height: Hall height [m]. If `None`, the Table 7.8-7 default is
        used. Hall height does not alter the generated coordinates, but it is
        included in ``topology.hall_dimensions`` because it affects InF
        channel statistics.
    :param bs_spacing: BS lattice spacing [m]. If `None`, the Table 7.8-7
        default for ``factory_scenario`` is used.
    :param bs_height: BS height [m]. If `None`, the Table 7.8-7 default for
        ``factory_scenario`` is used.
    :param ut_height: UT height [m]. If `None`, the Table 7.8-7 default is used.
    :param min_bs_ut_dist: Minimum 2D distance between each UT and BS [m].
    :param return_site_positions: If `True`, return
        ``(topology, site_positions)`` instead of only ``topology``. The
        ``topology`` object still provides the metadata and validation API
        described above.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :output ut_loc: [batch_size, num_ut, 3], `torch.float`.
        UT locations [m].
    :output bs_loc: [batch_size, num_bs, 3], `torch.float`.
        BS locations [m].
    :output ut_orientations: [batch_size, num_ut, 3], `torch.float`.
        UT orientations [radian].
    :output bs_orientations: [batch_size, num_bs, 3], `torch.float`.
        BS orientations [radian].
    :output ut_velocities: [batch_size, num_ut, 3], `torch.float`.
        UT velocity vectors [m/s].
    :output in_state: [batch_size, num_ut], `torch.bool`.
        Indoor state of UTs. Always `True`.
    :output los: `None`.
        Placeholder for stochastic LoS/NLoS sampling by the channel model.
    :output bs_virtual_loc: [batch_size, num_bs, num_ut, 3], `torch.float`.
        Virtual BS locations [m]. No wraparound is applied.
    :output bs_site_ids: [num_bs], `torch.int64`.
        Site identifier of each BS.
    :output site_positions: [num_bs, 2], `torch.float`.
        BS site center positions [m]. Returned separately from ``topology``
        only if ``return_site_positions`` is `True`.

    .. rubric:: Examples

    .. code-block:: python

        from sionna.phy.channel.tr38901 import InF
        from sionna.sys import gen_tr38901_indoor_factory_topology

        topology = gen_tr38901_indoor_factory_topology(
            "SH", 1, 20, hall_length=200.0, hall_width=100.0)
        channel_model = InF(carrier_frequency, ut_array, bs_array, "downlink",
                            factory_scenario=topology.factory_scenario,
                            hall_dimensions=topology.hall_dimensions)
        topology.set_topology(channel_model)
    """

    if precision is None:
        dtype = config.dtype
    else:
        dtype = dtypes[precision]["torch"]["dtype"]
    if device is None:
        device = config.device

    factory_scenario, defaults = _tr38901_factory_defaults(factory_scenario)
    default_hall_dimensions = (
        defaults["hall_length"],
        defaults["hall_width"],
        defaults["hall_height"],
    )
    hall_length = defaults["hall_length"] if hall_length is None else hall_length
    hall_width = defaults["hall_width"] if hall_width is None else hall_width
    hall_height = defaults["hall_height"] if hall_height is None else hall_height
    bs_spacing = defaults["bs_spacing"] if bs_spacing is None else bs_spacing
    bs_height = defaults["bs_height"] if bs_height is None else bs_height
    ut_height = defaults["ut_height"] if ut_height is None else ut_height
    if hall_height <= 0.0:
        raise ValueError("`hall_height` must be positive")
    hall_dimensions = (float(hall_length), float(hall_width), float(hall_height))

    site_positions = _tr38901_rectangular_site_positions(
        hall_length, hall_width, bs_spacing, dtype, device
    )
    num_bs = site_positions.shape[0]
    generator = config.torch_rng(device)
    ut_xy = _tr38901_drop_rectangular_ut_xy(
        batch_size,
        num_ut,
        hall_length,
        hall_width,
        site_positions,
        min_bs_ut_dist,
        dtype,
        device,
        generator,
    )
    ut_loc = torch.cat(
        [
            ut_xy,
            torch.full((batch_size, num_ut, 1), ut_height,
                       dtype=dtype, device=device),
        ],
        dim=-1,
    )

    bs_loc_single = torch.cat(
        [
            site_positions,
            torch.full((num_bs, 1), bs_height, dtype=dtype, device=device),
        ],
        dim=-1,
    )
    bs_loc = bs_loc_single.unsqueeze(0).expand(batch_size, -1, -1).clone()

    ut_orientations = torch.zeros(batch_size, num_ut, 3, dtype=dtype, device=device)
    bs_orientations = torch.zeros(batch_size, num_bs, 3, dtype=dtype, device=device)
    ut_velocities = torch.zeros(batch_size, num_ut, 3, dtype=dtype, device=device)
    in_state = torch.ones(batch_size, num_ut, dtype=torch.bool, device=device)
    bs_virtual_loc = _tr38901_unwrapped_virtual_bs_locations(bs_loc, num_ut)
    bs_site_ids = torch.arange(num_bs, dtype=torch.int64, device=device)

    output = IndoorFactoryTopology(
        (
            ut_loc,
            bs_loc,
            ut_orientations,
            bs_orientations,
            ut_velocities,
            in_state,
            None,
            bs_virtual_loc,
            bs_site_ids,
        ),
        factory_scenario,
        hall_dimensions,
        hall_dimensions != default_hall_dimensions,
    )
    if return_site_positions:
        return output, site_positions
    return output
