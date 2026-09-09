#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Class for sampling large scale parameters (LSPs) and pathloss following the
3GPP TR38.901 specifications and according to a channel simulation scenario
"""

from typing import Optional

import torch

from sionna.phy.object import Object
from sionna.phy.utils import normal

from .spatial_consistency import (
    spatial_consistency_correlation_matrix,
    spatial_consistency_matrix_sqrt,
)
from .utils import update_topology_buffer

__all__ = ["LSP", "LSPGenerator"]


class LSP:
    r"""Class for conveniently storing LSPs

    :param ds: RMS delay spread [s],
        shape [batch size, num tx, num rx]
    :param asd: Azimuth angle spread of departure [deg],
        shape [batch size, num tx, num rx]
    :param asa: Azimuth angle spread of arrival [deg],
        shape [batch size, num tx, num rx]
    :param sf: Shadow fading,
        shape [batch size, num tx, num rx]
    :param k_factor: Rician K-factor. Only used for LoS,
        shape [batch size, num tx, num rx]
    :param zsa: Zenith angle spread of arrival [deg],
        shape [batch size, num tx, num rx]
    :param zsd: Zenith angle spread of departure [deg],
        shape [batch size, num tx, num rx]
    :param pathloss: Optional path loss [dB] sampled with these LSPs,
        shape [batch size, num tx, num rx]
    """

    def __init__(
        self,
        ds: torch.Tensor,
        asd: torch.Tensor,
        asa: torch.Tensor,
        sf: torch.Tensor,
        k_factor: torch.Tensor,
        zsa: torch.Tensor,
        zsd: torch.Tensor,
        pathloss: Optional[torch.Tensor] = None,
    ) -> None:
        self.ds = ds
        self.asd = asd
        self.asa = asa
        self.sf = sf
        self.k_factor = k_factor
        self.zsa = zsa
        self.zsd = zsd
        self.pathloss = pathloss


class LSPGenerator(Object):
    r"""Sample large scale parameters (LSP) and pathloss given a channel
    scenario, e.g., UMa, UMi, RMa, InH, or InF

    This class implements steps 1 to 4 of the TR 38.901 specifications
    (section 7.5), as well as path-loss generation (Section 7.4.1) with O2I
    low- and high- loss models (Section 7.4.3).

    Note that a global scenario is set for the entire batches when instantiating
    this class (UMa, UMi, RMa, InH, or InF). However, each UT-BS link can have
    its
    specific state (LoS, NLoS, or indoor).

    The batch size is set by the ``scenario`` given as argument when
    constructing the class.

    Spatial filtering is evaluated for the UT locations in the current
    topology snapshot. Every call samples a fresh LSP field; realizations are
    not retained across topology updates.

    :param scenario: Scenario used to generate LSPs

    :output lsp: :class:`~sionna.phy.channel.tr38901.LSP`.
        An LSP instance storing realization of LSPs.

    .. rubric:: Examples

    .. code-block:: python

        # Assuming scenario is a SystemLevelScenario instance
        lsp_generator = LSPGenerator(scenario)
        lsp = lsp_generator()
    """

    def __init__(self, scenario) -> None:
        super().__init__(precision=scenario.precision, device=scenario.device)
        self._scenario = scenario
        self._use_legacy_o2i_model = (
            scenario.scenario_kind in ("umi", "uma")
            and bool(scenario.carrier_frequency < 6e9)
        )
        self.register_buffer(
            "_standard_lsp_order",
            torch.tensor([3, 4, 0, 1, 2, 6, 5], device=self.device),
        )
        self.register_buffer(
            "_internal_lsp_order",
            torch.tensor([2, 3, 4, 0, 1, 6, 5], device=self.device),
        )

    def sample_pathloss(self) -> torch.Tensor:
        """Generate pathlosses [dB] for each BS-UT link.

        :output pathloss: [batch size, number of base stations, number of UTs], `torch.float`.
            Pathloss [dB] for each BS-UT link.
        """
        # Pre-computed basic pathloss
        pl_b = self._scenario.basic_pathloss

        # O2I penetration
        if self._scenario.o2i_pathloss_enabled:
            if self._scenario.o2i_model == "low":
                pl_o2i = self._o2i_low_loss()
            else:  # 'high'
                pl_o2i = self._o2i_high_loss()
        else:
            pl_o2i = torch.zeros_like(pl_b)

        # Total path loss, including building and car penetration
        pl = pl_b + pl_o2i + self._car_penetration_loss()

        return pl

    def __call__(self) -> LSP:
        """Generate LSPs"""
        # LSPs are assumed to follow a log-normal distribution.
        # They are generated in the log-domain (where they follow a normal
        # distribution), where they are correlated as indicated in TR38901
        # specification (Section 7.5, step 4)

        s = normal(
            (self._scenario.batch_size,
             self._scenario.num_bs,
             self._scenario.num_ut,
             7),
            dtype=self.dtype,
            device=self.device,
            generator=self.torch_rng,
        )

        # WINNER II Section 3.3.1 first filters one independent spatial field
        # per LSP, then applies the same-link cross-LSP Cholesky transform.
        s = s.permute(0, 1, 3, 2).unsqueeze(3)
        s = torch.matmul(s, self._spatial_lsp_correlation_matrix_sqrt.transpose(-1, -2))
        s = s.squeeze(3).permute(0, 1, 3, 2)

        # TR 38.901 Step 4 mandates the Cholesky order
        # [SF, K, DS, ASD, ASA, ZSD, ZSA]. Convert back to the public internal
        # order [DS, ASD, ASA, SF, K, ZSA, ZSD] afterwards.
        s = s.index_select(-1, self._standard_lsp_order).unsqueeze(-1)
        s = self._cross_lsp_correlation_matrix_sqrt @ s
        s = s.squeeze(-1).index_select(-1, self._internal_lsp_order)

        # Scaling and transposing LSPs to the right mean and variance
        lsp_log_mean = self._scenario.lsp_log_mean
        lsp_log_std = self._scenario.lsp_log_std
        lsp_log = lsp_log_std * s + lsp_log_mean
        lsp_log = self._scenario.share_by_bs_site(lsp_log)

        # Mapping to linear domain
        lsp = torch.pow(
            torch.tensor(10.0, dtype=self.dtype, device=self.device), lsp_log
        )

        # Limit the RMS azimuth arrival (ASA) and azimuth departure (ASD)
        # spread values to 104 degrees
        # Limit the RMS zenith arrival (ZSA) and zenith departure (ZSD)
        # spread values to 52 degrees
        lsp = LSP(
            ds=lsp[:, :, :, 0],
            asd=torch.minimum(
                lsp[:, :, :, 1],
                torch.tensor(104.0, dtype=self.dtype, device=self.device),
            ),
            asa=torch.minimum(
                lsp[:, :, :, 2],
                torch.tensor(104.0, dtype=self.dtype, device=self.device),
            ),
            sf=lsp[:, :, :, 3],
            k_factor=lsp[:, :, :, 4],
            zsa=torch.minimum(
                lsp[:, :, :, 5],
                torch.tensor(52.0, dtype=self.dtype, device=self.device),
            ),
            zsd=torch.minimum(
                lsp[:, :, :, 6],
                torch.tensor(52.0, dtype=self.dtype, device=self.device),
            ),
            pathloss=(
                self.sample_pathloss()
                if self._scenario.pathloss_enabled
                else None
            ),
        )

        return lsp

    def topology_updated_callback(self) -> None:
        """Updates internal quantities. Must be called at every update of
        the scenario that changes the state of UTs or their locations.
        """
        # Pre-computing these quantities avoid unnecessary calculations at every
        # generation of new LSPs

        # Compute cross-LSP correlation matrix
        self._compute_cross_lsp_correlation_matrix()

        # Compute LSP spatial correlation matrix
        self._compute_lsp_spatial_correlation_sqrt()

        # Compute the correlation matrix for the random O2I penetration term
        self._compute_o2i_penetration_correlation_sqrt()

    def reset_topology(self) -> None:
        """Reset topology-dependent buffers."""
        for name in (
            "_cross_lsp_correlation_matrix_sqrt",
            "_spatial_lsp_correlation_matrix_sqrt",
            "_o2i_penetration_correlation_matrix_sqrt",
        ):
            if hasattr(self, name):
                delattr(self, name)

    def allocate_topology_tensors(self, batch_size: int, num_bs: int, num_ut: int) -> None:
        """Pre-allocate topology-dependent buffers."""
        self.reset_topology()
        self.register_buffer(
            "_cross_lsp_correlation_matrix_sqrt",
            torch.zeros(
                batch_size, num_bs, num_ut, 7, 7, dtype=self.dtype, device=self.device
            ),
        )
        self.register_buffer(
            "_spatial_lsp_correlation_matrix_sqrt",
            torch.zeros(
                batch_size,
                num_bs,
                7,
                num_ut,
                num_ut,
                dtype=self.dtype,
                device=self.device,
            ),
        )
        self.register_buffer(
            "_o2i_penetration_correlation_matrix_sqrt",
            torch.zeros(
                batch_size,
                num_ut,
                num_ut,
                dtype=self.dtype,
                device=self.device,
            ),
        )

    ########################################
    # Internal utility methods
    ########################################

    def _compute_cross_lsp_correlation_matrix(self) -> None:
        """Compute and store as attribute the square-root of the cross-LSPs
        correlation matrices for each BS-UT link, and then the corresponding
        matrix square root for filtering.

        The resulting tensor is of shape
        [batch size, number of base stations, number of UTs, 7, 7),
        7 being the number of LSPs to correlate.
        """
        # The following 7 LSPs are correlated:
        # DS, ASA, ASD, SF, K, ZSA, ZSD
        # We create the correlation matrix initialized to the identity matrix
        cross_lsp_corr_mat = torch.eye(
            7,
            7,
            dtype=self.dtype,
            device=self.device,
        ).expand(
            self._scenario.batch_size,
            self._scenario.num_bs,
            self._scenario.num_ut,
            7,
            7,
        ).clone()

        # Tensors of bool indicating the state of UT-BS links
        # Indoor
        indoor_bool = self._scenario.indoor.unsqueeze(1).expand(
            -1, self._scenario.num_bs, -1
        )
        # LoS
        los_bool = self._scenario.los
        # NLoS (outdoor)
        nlos_bool = (~self._scenario.los) & (~indoor_bool)
        # Expand to allow broadcasting with the BS dimension
        indoor_bool = indoor_bool.unsqueeze(3).unsqueeze(4)
        los_bool = los_bool.unsqueeze(3).unsqueeze(4)
        nlos_bool = nlos_bool.unsqueeze(3).unsqueeze(4)

        # Internal function that adds to the correlation matrix ``mat``
        # ``cross_lsp_corr_mat`` the parameter ``parameter_name`` at location
        # (m,n)
        def _add_param(mat: torch.Tensor, parameter_name: str, m: int, n: int) -> torch.Tensor:
            # Mask to put the parameters in the right spot of the 7x7
            # correlation matrix
            mask = torch.zeros(7, 7, dtype=self.dtype, device=self.device)
            mask[m, n] = 1.0
            mask[n, m] = 1.0
            mask = mask.reshape(1, 1, 1, 7, 7)
            # Get the parameter value according to the link scenario
            p_los = self._scenario._params_los[parameter_name]
            p_nlos = self._scenario._params_nlos[parameter_name]
            p_o2i = self._scenario._params_o2i[parameter_name]
            update = self._scenario.broadcast_params(p_los, p_nlos, p_o2i)
            update = update.unsqueeze(3).unsqueeze(4)
            # Add update
            mat = mat + update * mask
            return mat

        # Fill off-diagonal elements of the correlation matrices
        # ASD vs DS
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, "corrASDvsDS", 0, 1)
        # ASA vs DS
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, "corrASAvsDS", 0, 2)
        # ASA vs SF
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, "corrASAvsSF", 3, 2)
        # ASD vs SF
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, "corrASDvsSF", 3, 1)
        # DS vs SF
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, "corrDSvsSF", 3, 0)
        # ASD vs ASA
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, "corrASDvsASA", 1, 2)
        # ASD vs K
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, "corrASDvsK", 1, 4)
        # ASA vs K
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, "corrASAvsK", 2, 4)
        # DS vs K
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, "corrDSvsK", 0, 4)
        # SF vs K
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, "corrSFvsK", 3, 4)
        # ZSD vs SF
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, "corrZSDvsSF", 3, 6)
        # ZSA vs SF
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, "corrZSAvsSF", 3, 5)
        # ZSD vs K
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, "corrZSDvsK", 6, 4)
        # ZSA vs K
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, "corrZSAvsK", 5, 4)
        # ZSD vs DS
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, "corrZSDvsDS", 6, 0)
        # ZSA vs DS
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, "corrZSAvsDS", 5, 0)
        # ZSD vs ASD
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, "corrZSDvsASD", 6, 1)
        # ZSA vs ASD
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, "corrZSAvsASD", 5, 1)
        # ZSD vs ASA
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, "corrZSDvsASA", 6, 2)
        # ZSA vs ASA
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, "corrZSAvsASA", 5, 2)
        # ZSD vs ZSA
        cross_lsp_corr_mat = _add_param(cross_lsp_corr_mat, "corrZSDvsZSA", 5, 6)

        cross_lsp_corr_mat = cross_lsp_corr_mat.index_select(
            -2, self._standard_lsp_order
        ).index_select(-1, self._standard_lsp_order)
        # Step 4 explicitly prescribes a Cholesky square root in the standard
        # LSP-vector order.
        chol, _ = torch.linalg.cholesky_ex(
            cross_lsp_corr_mat, check_errors=False
        )
        self._update_buffer("_cross_lsp_correlation_matrix_sqrt", chol)

    def _compute_lsp_spatial_correlation_sqrt(self) -> None:
        r"""Compute the square root of the spatial correlation matrices of LSPs.

        The LSPs are correlated across users according to the distance between
        the users. Each LSP is spatially correlated according to a different
        spatial correlation matrix.

        The links involving different base stations are not correlated.
        UTs in different state (LoS, NLoS, O2I) are not assumed to be
        correlated.

        The correlation of the LSPs X of two UTs in the same state related to
        the links of these UTs to a same BS is

        .. math::
            C(X_1,X_2) = \exp(-d/D_X)

        where :math:`d` is the distance between the UTs in the X-Y plane (2D
        distance) and :math:`D_X` the correlation distance of LSP X.

        The resulting tensor is of shape
        [batch size, number of base stations, 7, number of UTs, number of UTs),
        7 being the number of LSPs.
        """
        # Tensors of bool indicating which pair of UTs to correlate.
        # Pairs of UTs that are correlated are those that share the same state
        # (indoor, LoS, or NLoS).
        # Indoor
        indoor = self._scenario.indoor.unsqueeze(1).expand(
            -1, self._scenario.num_bs, -1
        )
        # LoS
        los_ut = self._scenario.los
        los_pair_bool = los_ut.unsqueeze(3) & los_ut.unsqueeze(2)
        # NLoS
        if self._scenario.use_indoor_lsp_params:
            nlos_ut = (~self._scenario.los) & (~indoor)
        else:
            nlos_ut = ~self._scenario.los
        nlos_pair_bool = nlos_ut.unsqueeze(3) & nlos_ut.unsqueeze(2)
        # O2I
        if self._scenario.use_indoor_lsp_params:
            o2i_pair_bool = indoor.unsqueeze(3) & indoor.unsqueeze(2)
        else:
            o2i_pair_bool = torch.zeros_like(nlos_pair_bool)
        region_ids = self._scenario.ut_spatial_region_ids
        same_region = region_ids.unsqueeze(-1) == region_ids.unsqueeze(-2)
        same_region = same_region.unsqueeze(1)
        los_pair_bool = los_pair_bool & same_region
        nlos_pair_bool = nlos_pair_bool & same_region
        o2i_pair_bool = o2i_pair_bool & same_region

        # Stacking the correlation matrix
        # One correlation matrix per LSP
        filtering_matrices = []
        distance_scaling_matrices = []
        for parameter_name in (
            "corrDistDS",
            "corrDistASD",
            "corrDistASA",
            "corrDistSF",
            "corrDistK",
            "corrDistZSA",
            "corrDistZSD",
        ):
            # Matrix used for filtering and scaling the 2D distances
            # For each pair of UTs, the entry is set to 0 if the UTs are in
            # different states, -1/(correlation distance) otherwise.
            # The correlation distance is different for each LSP.
            filtering_matrix = torch.eye(
                self._scenario.num_ut,
                self._scenario.num_ut,
                dtype=self.dtype,
                device=self.device,
            ).expand(
                self._scenario.batch_size, self._scenario.num_bs, -1, -1
            ).clone()

            distance_scaling_matrix = self._scenario.broadcast_params(
                self._scenario._params_los[parameter_name],
                self._scenario._params_nlos[parameter_name],
                self._scenario._params_o2i[parameter_name]
            )
            distance_scaling_matrix = distance_scaling_matrix.unsqueeze(3).expand(
                -1, -1, -1, self._scenario.num_ut
            )
            distance_scaling_matrix = -1.0 / distance_scaling_matrix

            # LoS
            filtering_matrix = torch.where(
                los_pair_bool,
                torch.tensor(1.0, dtype=self.dtype, device=self.device),
                filtering_matrix,
            )
            # NLoS
            filtering_matrix = torch.where(
                nlos_pair_bool,
                torch.tensor(1.0, dtype=self.dtype, device=self.device),
                filtering_matrix,
            )
            # indoor
            filtering_matrix = torch.where(
                o2i_pair_bool,
                torch.tensor(1.0, dtype=self.dtype, device=self.device),
                filtering_matrix,
            )
            # Stacking
            filtering_matrices.append(filtering_matrix)
            distance_scaling_matrices.append(distance_scaling_matrix)

        filtering_matrices = torch.stack(filtering_matrices, dim=2)
        distance_scaling_matrices = torch.stack(distance_scaling_matrices, dim=2)

        ut_dist_2d = self._scenario.matrix_ut_distance_2d
        # Adding a dimension for broadcasting with BS
        ut_dist_2d = ut_dist_2d.unsqueeze(1).unsqueeze(2)

        # Correlation matrix
        spatial_lsp_correlation = torch.exp(
            ut_dist_2d * distance_scaling_matrices
        ) * filtering_matrices

        # Compute and store the square root of the spatial correlation matrix.
        # Co-located terminals lead to positive-semidefinite, but singular,
        # matrices. The spatial-consistency square-root helper handles this
        # case without adding artificial jitter. For co-sited sectors, only
        # representative base stations need to be factorized; the factors are identical
        # within a site because topology updates share the link state by site.
        representatives = self._scenario.bs_site_representatives
        bs_index = torch.arange(
            self._scenario.num_bs, dtype=torch.int64, device=self.device
        ).reshape(1, -1)
        if (
            representatives is not None
            and not torch.compiler.is_compiling()
            and bool(torch.any(representatives != bs_index))
        ):
            chol = torch.empty_like(spatial_lsp_correlation)
            for batch_ind in range(self._scenario.batch_size):
                unique_reps, inverse = torch.unique(
                    representatives[batch_ind],
                    sorted=True,
                    return_inverse=True,
                )
                rep_chol = spatial_consistency_matrix_sqrt(
                    spatial_lsp_correlation[batch_ind, unique_reps],
                    precision=self.precision,
                    device=self.device,
                )
                chol[batch_ind] = rep_chol[inverse]
        else:
            chol = spatial_consistency_matrix_sqrt(
                spatial_lsp_correlation,
                precision=self.precision,
                device=self.device,
            )
        self._update_buffer("_spatial_lsp_correlation_matrix_sqrt", chol)

    def _update_buffer(self, name: str, value: torch.Tensor) -> None:
        """Update or register a buffer for topology-dependent tensors."""
        update_topology_buffer(
            self,
            name,
            value,
            "Call reset_topology() or allocate_topology_tensors() first.",
        )

    def _compute_o2i_penetration_correlation_sqrt(self) -> None:
        """Precompute the 10 m random-penetration correlation factor."""
        if self._scenario._enable_spatial_consistency:
            correlation = spatial_consistency_correlation_matrix(
                self._scenario.matrix_ut_distance_2d,
                torch.tensor(10.0, dtype=self.dtype, device=self.device),
                states=self._scenario.ut_spatial_region_ids,
                precision=self.precision,
                device=self.device,
            )
            factor = spatial_consistency_matrix_sqrt(
                correlation,
                precision=self.precision,
                device=self.device,
            )
        else:
            factor = torch.eye(
                self._scenario.num_ut,
                dtype=self.dtype,
                device=self.device,
            ).expand(self._scenario.batch_size, -1, -1).clone()
        self._update_buffer(
            "_o2i_penetration_correlation_matrix_sqrt", factor
        )

    def _sample_o2i_penetration_random(self, stddev: float) -> torch.Tensor:
        """Sample one spatially consistent penetration term per UT."""
        white = normal(
            (self._scenario.batch_size, 1, self._scenario.num_ut),
            dtype=self.dtype,
            device=self.device,
            generator=self.torch_rng,
        )
        sample = torch.matmul(
            self._o2i_penetration_correlation_matrix_sqrt,
            white.transpose(1, 2),
        ).squeeze(-1)
        sample = sample * stddev
        return sample.unsqueeze(1).expand(-1, self._scenario.num_bs, -1)

    def _car_penetration_loss(self) -> torch.Tensor:
        """Sample the UT-specific car penetration loss of Section 7.4.3.2."""
        if self._scenario.scenario_kind != "rma":
            return torch.zeros_like(self._scenario.basic_pathloss)

        sample = normal(
            (self._scenario.batch_size, 1, self._scenario.num_ut),
            dtype=self.dtype,
            device=self.device,
            generator=self.torch_rng,
        )
        sample = (
            5.0 * sample + self._scenario.car_penetration_loss_mean
        )
        sample = sample.expand(-1, self._scenario.num_bs, -1)
        return sample * self._scenario.in_car.unsqueeze(1).to(self.dtype)

    def _o2i_low_loss(self) -> torch.Tensor:
        """Compute for each BS-UT link the pathloss due to the O2I
        penetration loss in dB with the low-loss model.
        See section 7.4.3.1 of 38.901 specification.

        UTs located outdoor (LoS and NLoS) get O2I pathloss of 0dB.

        :output pl_o2i: [batch size, number of base stations, number of UTs], `torch.float`.
            O2I penetration low-loss in dB for each BS-UT link.
        """
        if self._use_legacy_o2i_model:
            return self._o2i_legacy_loss()

        fc = self._scenario.carrier_frequency / 1e9  # Carrier frequency (GHz)
        batch_size = self._scenario.batch_size
        num_ut = self._scenario.num_ut

        # Material penetration losses
        # fc must be in GHz
        l_glass = 2.0 + 0.2 * fc
        l_concrete = 5.0 + 4.0 * fc

        # Path loss through external wall
        pl_tw = 5.0 - 10.0 * torch.log10(
            0.3 * torch.pow(
                torch.tensor(10.0, dtype=self.dtype, device=self.device),
                -l_glass / 10.0,
            )
            + 0.7 * torch.pow(
                torch.tensor(10.0, dtype=self.dtype, device=self.device),
                -l_concrete / 10.0,
            )
        )

        # Filtering-out the O2I pathloss for UTs located outdoor
        indoor_mask = torch.where(
            self._scenario.indoor,
            torch.tensor(1.0, dtype=self.dtype, device=self.device),
            torch.zeros(batch_size, num_ut, dtype=self.dtype, device=self.device),
        ).unsqueeze(1)
        pl_tw = pl_tw * indoor_mask

        # Pathloss due to indoor propagation
        # The indoor 2D distance for outdoor UTs is 0
        pl_in = 0.5 * self._scenario.distance_2d_in

        # Random path loss component
        # Gaussian distributed with standard deviation 4.4 in dB
        pl_rnd = self._sample_o2i_penetration_random(4.4)
        pl_rnd = pl_rnd * indoor_mask

        return pl_tw + pl_in + pl_rnd

    def _o2i_high_loss(self) -> torch.Tensor:
        """Compute for each BS-UT link the pathloss due to the O2I
        penetration loss in dB with the high-loss model.
        See section 7.4.3.1 of 38.901 specification.

        UTs located outdoor (LoS and NLoS) get O2I pathloss of 0dB.

        :output pl_o2i: [batch size, number of base stations, number of UTs], `torch.float`.
            O2I penetration high-loss in dB for each BS-UT link.
        """
        if self._use_legacy_o2i_model:
            return self._o2i_legacy_loss()

        fc = self._scenario.carrier_frequency / 1e9  # Carrier frequency (GHz)
        batch_size = self._scenario.batch_size
        num_ut = self._scenario.num_ut

        # Material penetration losses
        # fc must be in GHz
        if self._scenario.spec_version == "19.2":
            l_iirglass = 25.4 + 0.11 * fc
        else:
            l_iirglass = 23.0 + 0.3 * fc
        l_concrete = 5.0 + 4.0 * fc

        # Path loss through external wall
        pl_tw = 5.0 - 10.0 * torch.log10(
            0.7 * torch.pow(
                torch.tensor(10.0, dtype=self.dtype, device=self.device),
                -l_iirglass / 10.0,
            )
            + 0.3 * torch.pow(
                torch.tensor(10.0, dtype=self.dtype, device=self.device),
                -l_concrete / 10.0,
            )
        )

        # Filtering-out the O2I pathloss for outdoor UTs
        indoor_mask = torch.where(
            self._scenario.indoor,
            torch.tensor(1.0, dtype=self.dtype, device=self.device),
            torch.zeros(batch_size, num_ut, dtype=self.dtype, device=self.device),
        ).unsqueeze(1)
        pl_tw = pl_tw * indoor_mask

        # Pathloss due to indoor propagation
        # The indoor 2D distance for outdoor UTs is 0
        pl_in = 0.5 * self._scenario.distance_2d_in

        # Random path loss component
        # Gaussian distributed with standard deviation 6.5 in dB for the
        # high loss model
        pl_rnd = self._sample_o2i_penetration_random(6.5)
        pl_rnd = pl_rnd * indoor_mask

        return pl_tw + pl_in + pl_rnd

    def _o2i_legacy_loss(self) -> torch.Tensor:
        """Return the below-6-GHz UMi/UMa penetration loss.

        Table 7.4.3-3 specifies a fixed 20 dB external-wall loss and no
        random penetration-loss component. The indoor propagation loss is
        still 0.5 dB per metre of indoor distance.
        """
        indoor_mask = self._scenario.indoor.unsqueeze(1).to(self.dtype)
        wall_loss = torch.tensor(20.0, dtype=self.dtype, device=self.device)
        return wall_loss * indoor_mask + 0.5 * self._scenario.distance_2d_in
