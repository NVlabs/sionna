Paths
======

A propagation path :math:`i` starts at a transmit antenna and ends at a receive antenna. It is described by
its channel coefficient :math:`a_i` and delay :math:`\tau_i`, as well as the
angles of departure :math:`(\theta_{\text{T},i}, \varphi_{\text{T},i})`
and arrival :math:`(\theta_{\text{R},i}, \varphi_{\text{R},i})`.
For more detail, see the `Primer on Electromagnetics <../em_primer.html>`_.

In Sionna, paths are computed with the help of a path solver (such as :class:`~sionna.rt.PathSolver`)
which returns an instance of :class:`~sionna.rt.Paths`.
Paths can be visualized by providing them as arguments to the functions :meth:`~sionna.rt.Scene.render`,
:meth:`~sionna.rt.Scene.render_to_file`, or :meth:`~sionna.rt.Scene.preview`.

Channel impulse responses (CIRs) can be obtained with :meth:`~sionna.rt.Paths.cir` which can
then be used for link-level simulations. This is for example done in the `Sionna Ray Tracing Tutorial <../tutorials/Introduction.html>`_.

.. autoclass:: sionna.rt.Paths
   :members:

Constants
----------

.. autoclass:: sionna.rt.constants.InteractionType
   :members:

.. autodata:: sionna.rt.constants.INVALID_SHAPE
.. autodata:: sionna.rt.constants.INVALID_PRIMITIVE

References:
   .. [Wiffen2018] F\. Wiffen et al., "`Comparison of OTFS and OFDM in Ray Launched sub-6 GHz and mmWave Line-of-Sight Mobility Channels <https://ieeexplore.ieee.org/abstract/document/8580850>`_", Proc. IEEE Int. Sym. Personal, Indoor and Mobile Radio Commun. (PIMRC), Bologna, Italy, Sep. 2018.
