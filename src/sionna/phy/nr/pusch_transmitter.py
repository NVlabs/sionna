#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""PUSCH Transmitter for the NR (5G) module of Sionna PHY"""

from sionna.phy import Block
from sionna.phy.mapping import Mapper, BinarySource
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, OFDMModulator
from .config import Config
from .pusch_config import PUSCHConfig, check_pusch_configs
from .pusch_pilot_pattern import PUSCHPilotPattern
from .pusch_precoder import PUSCHPrecoder
from .tb_encoder import TBEncoder
from .layer_mapping import LayerMapper

class PUSCHTransmitter(Block):
    # pylint: disable=line-too-long
    r"""
    This block generates batches of 5G NR PUSCH slots for multiple transmitters
    with random or provided payloads. Frequency- or time-domain outputs can be generated.

    It combines multiple processing blocks into a single layer
    as shown in the following figure. Blocks with dashed lines are
    optional and depend on the configuration.

    .. figure:: ../figures/pusch_transmitter_block_diagram.png
        :scale: 30%
        :align: center

    Information bits :math:`\mathbf{b}` that are either randomly generated or
    provided as input are encoded into a transport block by the :class:`~sionna.nr.TBEncoder`.
    The encoded bits are then mapped to QAM constellation symbols by the :class:`~sionna.mapping.Mapper`.
    The :class:`~sionna.nr.LayerMapper` splits the modulated symbols into different layers
    which are then mapped onto OFDM resource grids by the :class:`~sionna.ofdm.ResourceGridMapper`.
    If precoding is enabled in the :class:`~sionna.nr.PUSCHConfig`, the resource grids
    are further precoded so that there is one for each transmitter and antenna port.
    If ``output_domain`` equals "freq", these are the outputs :math:`\mathbf{x}`.
    If ``output_domain`` is chosen to be "time", the resource grids are transformed into
    time-domain signals by the :class:`~sionna.ofdm.OFDMModulator`.

    Parameters
    ----------
    pusch_configs : instance or `list` of :class:`~sionna.nr.PUSCHConfig`
        PUSCH Configurations according to which the resource grid and pilot pattern
        will created. One configuration is needed for each transmitter.

    return_bits : `bool`, (default `True`)
        If set to `True`, the block generates random information bits
        to be transmitted and returns them together with the transmit signal.

    output_domain : "freq" (default) | "time"
        Domain of the output

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    verbose: `bool`, (default `False`)
        If `True`, additional parameters are printed during initialization.

    Input
    -----
    One of:

    batch_size : `int`
        Batch size of random transmit signals to be generated,
        if ``return_bits`` is `True`.

    b : [batch_size, num_tx, tb_size], `tf.float`
        Information bits to be transmitted,
        if ``return_bits`` is `False`.

    Output
    ------
    x : [batch size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size], `tf.complex`, or [batch size, num_tx, num_tx_ant, num_time_samples], `tf.complex`
        Transmit signal in either frequency or time domain, depending on ``output_domain``.

    b : [batch_size, num_tx, tb_size], `tf.float`
        Transmitted information bits.
        Only returned if ``return_bits`` is `True`.

    Example
    -------
    >>> pusch_config = PUSCHConfig()
    >>> pusch_transmitter = PUSCHTransmitter(pusch_config)
    >>> x, b = pusch_transmitter(16)
    >>> print("Shape of x:", x.shape)
    Shape of x: (16, 1, 1, 14, 48)
    >>> print("Shape of b:", b.shape)
    Shape of b: (16, 1, 1352)

    """
    def __init__(self,
                 pusch_configs,
                 return_bits=True,
                 output_domain="freq",
                 precision=None,
                 verbose=False,
                 **kwargs):
        super().__init__(precision=precision, **kwargs)

        # Validate inputs and extract parameters
        assert isinstance(return_bits, bool), "return_bits must be bool"
        self._return_bits = return_bits

        assert output_domain in ["time", "freq"], \
            "output_domain must be 'time' or 'freq'"
        self._output_domain = output_domain

        assert isinstance(verbose, bool), "verbose must be bool"
        self._verbose = verbose

        if isinstance(pusch_configs, PUSCHConfig):
            pusch_configs = [pusch_configs]

        params = check_pusch_configs(pusch_configs)
        for key, value in params.items():
            self.__setattr__(f"_{key}", value)

        self._pusch_configs = pusch_configs

        # (Optionally) Create BinarySource
        if self._return_bits:
            self._binary_source = BinarySource(precision=self.precision)

        # Create TBEncoder
        self._tb_encoder = TBEncoder(
                            target_tb_size=self._tb_size,
                            num_coded_bits=self._num_coded_bits,
                            target_coderate=self._target_coderate,
                            num_bits_per_symbol=self._num_bits_per_symbol,
                            num_layers=self._num_layers,
                            n_rnti=self._n_rnti,
                            n_id=self._n_id,
                            channel_type="PUSCH", # PUSCHTransmitter
                            codeword_index=0, # not supported for PUSCH
                            use_scrambler=True,
                            verbose=self._verbose,
                            precision=self.precision)

        # Create PUSCHLayerMapper
        self._layer_mapper = LayerMapper(
                                num_layers=self._num_layers,
                                precision=self.precision)

        # Create Mapper
        self._mapper = Mapper("qam",
                              self._num_bits_per_symbol,
                              precision=self.precision)

        # Create PUSCHPilotPattern
        self._pilot_pattern = PUSCHPilotPattern(self._pusch_configs,
                                                precision=self.precision)

        # Create ResourceGrid
        self._resource_grid = ResourceGrid(
                            num_ofdm_symbols=self._num_ofdm_symbols,
                            fft_size=self._num_subcarriers,
                            subcarrier_spacing=self._subcarrier_spacing,
                            num_tx=self._num_tx,
                            num_streams_per_tx=self._num_layers,
                            cyclic_prefix_length=self._cyclic_prefix_length,
                            pilot_pattern=self._pilot_pattern,
                            precision=self.precision)

        # Create ResourceGridMapper
        self._resource_grid_mapper = ResourceGridMapper(self._resource_grid,
                                                    precision=self.precision)

        # (Optionally) Create PUSCHPrecoder
        if self._precoding=="codebook":
            self._precoder = PUSCHPrecoder(self._precoding_matrices,
                                           precision=self.precision)

        # (Optionally) Create OFDMModulator
        if self._output_domain=="time":
            self._ofdm_modulator = OFDMModulator(self._cyclic_prefix_length,
                                                 precision=self.precision)

    #########################################
    # Public methods and properties
    #########################################

    @property
    def resource_grid(self):
        """OFDM resource grid underlying the PUSCH transmissions"""
        return self._resource_grid

    @property
    def pilot_pattern(self):
        """Aggregate pilot pattern of all transmitters"""
        return self._pilot_pattern

    def show(self):
        """Print all properties of the PUSCHConfig and children"""
        # CarrierConfig is always the same
        self._pusch_configs[0].carrier.show()
        Config.show(self._pusch_configs[0])
        for idx,p in enumerate(self._pusch_configs):
            print(f"---- UE {idx} ----")
            p.dmrs.show()
            p.tb.show()

    def call(self, inputs):

        if self._return_bits:
            # inputs defines batch_size
            batch_size = inputs
            b = self._binary_source([batch_size, self._num_tx, self._tb_size])
        else:
            b = inputs

        # Encode transport block
        c = self._tb_encoder(b)

        # Map to constellations
        x_map = self._mapper(c)

        # Map to layers
        x_layer = self._layer_mapper(x_map)

        # Apply resource grid mapping
        x_grid = self._resource_grid_mapper(x_layer)

        # (Optionally) apply PUSCH precoding
        if self._precoding=="codebook":
            x_pre = self._precoder(x_grid)
        else:
            x_pre = x_grid

        # (Optionally) apply OFDM modulation
        if self._output_domain=="time":
            x = self._ofdm_modulator(x_pre)
        else:
            x = x_pre

        if self._return_bits:
            return x, b
        else:
            return x


