Multicell Topology
==================

.. currentmodule:: sionna.sys

.. figure:: ../figures/topology_api.png
   :align: center
   :width: 100%

In system-level simulations with 3GPP channel modeling, it is customary to place
cells on a spiral hexagonal grid. The grid is defined by the inter-site distance,
determining the distance between any two adjacent hexagonal cell centers, and
the number of rings of the grid, typically 1 or 2 (corresponding to 7 and 19
cells, hence 21 and 57 base stations, respectively).

To eliminate edge effects that would result in users at the cell borders
experiencing reduced interference levels, the grid is usually
wrapped around to create a seamless topology.

To learn how to place base stations and drop users on a hexagonal grid in
Sionna, refer to the `Hexagonal Grid Topology notebook <../tutorials/notebooks/HexagonalGrid.ipynb>`_.

.. autosummary::
   :toctree: .

   Hexagon
   HexGrid
   gen_hexgrid_topology
   get_num_hex_in_grid
   convert_hex_coord
