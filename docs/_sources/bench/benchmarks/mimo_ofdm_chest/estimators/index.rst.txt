Channel Estimators
====================

.. currentmodule:: sionna.bench.mimo_ofdm_chest.papers

The following table provides an overview of the implemented channel estimators and indicates
whether they exploit spatial, spectral/delay, or temporal properties of the channel.

For data-driven or statistics-based estimators, the corresponding doc page also documents how
the estimators was trained/fitted.

.. table::
   :class: estimator-overview

   +--------------------------------------------------------+----------+-----------+----------+
   | Channel Estimator                                      | Spatial  | Spectral  | Temporal |
   +========================================================+==========+===========+==========+
   | *Interpolators*                                                                          |
   +--------------------------------------------------------+----------+-----------+----------+
   |   :class:`~sionna.phy.ofdm.NearestNeighborInterpolator`|          |           |          |
   +--------------------------------------------------------+----------+-----------+----------+
   |   :class:`~sionna.phy.ofdm.LinearInterpolator`         |          |           |          |
   +--------------------------------------------------------+----------+-----------+----------+
   |   :class:`~coleri2002.CombInterpolator`                |          |           |          |
   +--------------------------------------------------------+----------+-----------+----------+
   | *Linear estimators*                                                                      |
   +--------------------------------------------------------+----------+-----------+----------+
   |   :class:`~sionna.phy.ofdm.LSChannelEstimator`         | ✘        | ✘         | ✘        |
   +--------------------------------------------------------+----------+-----------+----------+
   |   :class:`~sionna.phy.ofdm.LMMSEChannelEstimator`      | ✔        | ✔         | ✔        |
   +--------------------------------------------------------+----------+-----------+----------+
   |   :class:`~edfors1996.TruncatedChannelEstimator`       | ✘        | ✔         | ✘        |
   +--------------------------------------------------------+----------+-----------+----------+
   | *Sparsity-based estimators*                                                              |
   +--------------------------------------------------------+----------+-----------+----------+
   |   :class:`~wei2017.SBLChannelEstimator`                | ✘        | ✔         | ✘        |
   +--------------------------------------------------------+----------+-----------+----------+
   |   :class:`~lyu2026.SparseDiffuseChannelEstimator`      | ✘        | ✔         | ✘        |
   +--------------------------------------------------------+----------+-----------+----------+
   |   :class:`~delfeld2025.RKHSChannelEstimator`           | ✔        | ✔         | ✘        |
   +--------------------------------------------------------+----------+-----------+----------+
   | *Latent variable-based estimators*                                                       |
   +--------------------------------------------------------+----------+-----------+----------+
   |   :class:`~baur2022.VAEChannelEstimator`               | ✔        | ✘         | ✘        |
   +--------------------------------------------------------+----------+-----------+----------+
   |   :class:`~koller2022.SpatialGMMChannelEstimator`      | ✔        | ✘         | ✘        |
   +--------------------------------------------------------+----------+-----------+----------+
   |   :class:`~koller2022.SpectralGMMChannelEstimator`     | ✘        | ✔         | ✘        |
   +--------------------------------------------------------+----------+-----------+----------+
   | *Neural network-based estimators*                                                        |
   +--------------------------------------------------------+----------+-----------+----------+
   |   :class:`~li2020.ReEsNetChannelEstimator`             | ✘        | ✔         | ✔        |
   +--------------------------------------------------------+----------+-----------+----------+
   |   :class:`~balevi2020.DeepPriorChannelEstimator`       | ✔        | ✔         | ✔        |
   +--------------------------------------------------------+----------+-----------+----------+
   |   :class:`~melgar2022.DNNChannelEstimator`             | ✘        | ✔         | ✘        |
   +--------------------------------------------------------+----------+-----------+----------+
   |   :class:`~melgar2022.InterpolatedDNNChannelEstimator` | ✘        | ✔         | ✘        |
   +--------------------------------------------------------+----------+-----------+----------+






.. toctree::
   :hidden:
   :maxdepth: 5

   linear/index
   interpolators/index
   sparse/index
   latent_variable/index
   neural_networks/index


.. References
.. ==========

.. .. bibliography::
..    :all: