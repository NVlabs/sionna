Ray Tracing
#############

This module provides a differentiable ray tracer for radio propagation modeling.
The best way to get started is by having a look at the `Sionna Ray Tracing Tutorial <../examples/Sionna_Ray_Tracing_Introduction.html>`_.
The `Primer on Electromagnetics <../em_primer.html>`_ provides useful background knowledge and various definitions that are used throughout the API documentation.

The most important component of the ray tracer is the :class:`~sionna.rt.Scene`.
It has methods for the computation of propagation :class:`~sionna.rt.Paths` (:meth:`~sionna.rt.Scene.compute_paths`) and :class:`~sionna.rt.CoverageMap` (:meth:`~sionna.rt.Scene.coverage_map`).
Sionna has several integrated `Example Scenes`_ that you can use for your own experiments. In this `video <https://youtu.be/7xHLDxUaQ7c>`_, we explain how you can create your own scenes using `OpenStreetMap  <https://www.openstreetmap.org>`_ and `Blender <https://www.blender.org>`_.
You can preview a scene within a Jupyter notebook (:meth:`~sionna.rt.Scene.preview`) or render it to a file from the viewpoint of a camera (:meth:`~sionna.rt.Scene.render` or :meth:`~sionna.rt.Scene.render_to_file`).

Propagation :class:`~sionna.rt.Paths` can be transformed into time-varying channel impulse responses (CIRs) via :meth:`~sionna.rt.Paths.cir`. The CIRs can then be used for link-level simulations in Sionna via the functions :meth:`~sionna.channel.cir_to_time_channel` or :meth:`~sionna.channel.cir_to_ofdm_channel`. Alternatively, you can create a dataset of CIRs that can be used by a channel model with the help of :class:`~sionna.channel.CIRDataset`.

The paper `Sionna RT: Differentiable Ray Tracing for Radio Propagation Modeling <https://nvlabs.github.io/sionna/made_with_sionna.html#sionna-rt-differentiable-ray-tracing-for-radio-propagation-modeling>`_ shows how differentiable ray tracing can be used for various optimization tasks. The related `notebooks <https://nvlabs.github.io/sionna/made_with_sionna.html#sionna-rt-differentiable-ray-tracing-for-radio-propagation-modeling>`_ can be a good starting point for your own experiments.

.. include:: rt_scene.rst.txt
.. include:: rt_paths.rst.txt
.. include:: rt_coverage_map.rst.txt
.. include:: rt_camera.rst.txt
.. include:: rt_scene_object.rst.txt
.. include:: rt_radio_material.rst.txt
.. include:: rt_radio_device.rst.txt
.. include:: rt_antenna_array.rst.txt
.. include:: rt_antenna.rst.txt
.. include:: rt_ris.rst.txt
.. include:: rt_utils.rst.txt

References:
   .. [Balanis97] A\. Balanis, "Antenna Theory: Analysis and Design," 2nd Edition, John Wiley & Sons, 1997.
   .. [ITUR_P2040_2] ITU-R, “Effects of building materials and structures on radiowave propagation above about 100 MHz“, Recommendation ITU-R P.2040-2
   .. [SurfaceIntegral] Wikipedia, "`Surface integral <https://en.wikipedia.org/wiki/Surface_integral>`_", accessed Jun. 22, 2023.
   .. [Wiffen2018] F\. Wiffen et al., "`Comparison of OTFS and OFDM in Ray Launched sub-6 GHz and mmWave Line-of-Sight Mobility Channels <https://ieeexplore.ieee.org/abstract/document/8580850>`_", Proc. IEEE Int. Sym. Personal, Indoor and Mobil Radio Commun. (PIMRC), Bologna, Italy, Sep. 2018.
