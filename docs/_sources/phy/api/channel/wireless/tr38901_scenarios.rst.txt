:orphan:

System-Level Scenarios
----------------------

.. currentmodule:: sionna.phy.channel.tr38901

The scenario classes hold the versioned TR 38.901 parameter tables and the
topology-dependent state used by the system-level channel generators. They do
not generate channel impulse responses themselves. The complete
:class:`UMi`, :class:`UMa`, :class:`RMa`, :class:`InH`, and :class:`InF`
channel models construct the corresponding scenario object internally, so most
users should instantiate those channel classes instead.

The lower-level scenario API is useful when working directly with components
such as :class:`LSPGenerator` or :class:`RaysGenerator`. In particular,
:class:`InFScenario` stores the selected factory sub-scenario, hall and clutter
geometry, versioned LoS/NLoS parameters, and the topology state used internally
by :class:`InF`.

:class:`RMaScenario` additionally owns the mutually exclusive indoor and in-car
UT states required by Table 7.2-3. If ``in_car`` is omitted on its first
topology call, non-indoor UTs are interpreted as in-car. The ordinary-window
car penetration model is the default; the optional metallized-window mean from
Section 7.4.3.2 can be selected when constructing :class:`RMa` or
:class:`RMaScenario`.

For InF-SL, InF-DL, InF-SH, and InF-DH, omitted hall and clutter parameters use
the calibration assumptions in Table 7.8-7 of
:cite:p:`TR38901V1920`. Table 7.8-7 does not define an InF-HH
calibration layout. The InF-HH defaults exposed by :class:`InF` and
:class:`InFScenario` are therefore implementation assumptions; pass explicit
hall and clutter values when modelling a specific InF-HH deployment.

.. autosummary::
   :toctree: .

   SystemLevelScenario
   UMiScenario
   UMaScenario
   RMaScenario
   InHScenario
   InFScenario

Lower-Level Generators
~~~~~~~~~~~~~~~~~~~~~~

The following public components support custom assembly of the TR 38.901
generation steps. The high-level channel classes configure and call them
automatically.

.. autosummary::
   :toctree: .

   LSPGenerator
   RaysGenerator
   Topology
   ChannelCoefficientsGenerator
