.. _dev_custom_antenna_patterns:

Custom Antenna Patterns
=======================

As explained in greater detail in ":ref:`far_field`", an antenna pattern maps a
zenith and azimuth angle to two complex numbers, the zenith and azimuth patterns, respectively.
Mathematically, it is defined as a function :math:`f:(\theta,\varphi)\mapsto (C_\theta(\theta, \varphi), C_\varphi(\theta, \varphi))`.

If you want to add a new :class:`~sionna.rt.AntennaPattern` to Sionna RT, you must register
a factory method for it together with a name, using the function
:meth:`~sionna.rt.register_antenna_pattern`.
Once this is done, the new antenna pattern can be used everywhere by providing
its name. An example is shown below:

.. code-block:: python

    import mitsuba as mi
    import drjit as dr
    from sionna.rt import AntennaPattern, PlanarArray, register_antenna_pattern

    def v_sin_pow_pattern(theta: mi.Float, phi: mi.Float, n: mi.Float) -> mi.Complex2f:
        """Vertically polarized antenna pattern function"""
        return mi.Complex2f(dr.power(dr.sin(theta), n), 0)

    class MyPattern(AntennaPattern):
        def __init__(self, n):
            def my_pattern(theta, phi, n):
                """Adds a zero azimuth component to define a valid antenna pattern"""
                c_theta = v_sin_pow_pattern(theta, phi, n=n)
                c_phi = dr.zeros(mi.Complex2f, dr.width(c_theta))
                return c_theta, c_phi

            # Create compulsory pattern property
            # Add a second pattern here to make it dual-polarized
            self.patterns = [lambda theta, phi: my_pattern(theta, phi, n)]

    def my_pattern_factory(n=3):
        """Factory method that returns an instance of the antenna pattern"""
        return MyPattern(n=n)

    # Register the factory method
    register_antenna_pattern("my_pattern", my_pattern_factory)

    # Use the custom antenna pattern with the rest of Sionna RT
    array = PlanarArray(num_rows=1, num_cols=1, pattern="my_pattern", n=8)
    array.antenna_pattern.compute_gain();

::

    Directivity [dB]: 5.24
    Gain [dB]: 0.0
    Efficiency [%]: 30.0


Rather than specifying an antenna pattern from scratch, you can also register a
factory method for a new :class:`~sionna.rt.PolarizedAntennaPattern` which uses
a vertically polarized antenna pattern function:

.. code-block:: python

    import mitsuba as mi
    import drjit as dr
    from sionna.rt import PolarizedAntennaPattern, PlanarArray, \
                          register_antenna_pattern, register_polarization

    def v_sin_pow_pattern(theta: mi.Float, phi: mi.Float, n: mi.Float) -> mi.Complex2f:
        """Vertically polarized antenna pattern function"""
        return mi.Complex2f(dr.power(dr.sin(theta), n), 0)

    def my_pattern_factory(*, n, polarization, polarization_model):
        """Factory method returning an instance of a PolarizedAntennaPattern
        with the newly created pattern function
        """
        return PolarizedAntennaPattern(
            v_pattern=lambda theta, phi: v_sin_pow_pattern(theta, phi, n),
            polarization=polarization,
            polarization_model=polarization_model
        )

    register_antenna_pattern("my_pattern", my_pattern_factory)

    # Register a custom polarization
    # Since we provide two slant angles, the resulting
    # antenna pattern will be dual-polarized
    register_polarization("my_polarization", [-dr.pi/6, dr.pi*2/6])

    # Use the custom antenna pattern with the rest of Sionna RT
    array = PlanarArray(num_rows=1, num_cols=1,
                        pattern="my_pattern",
                        n=12,
                        polarization="my_polarization",
                        polarization_model="tr38901_1")

In the example above, we have also used :meth:`~sionna.rt.register_polarization`
to create a new polarization which can be used together with any registered
antenna pattern factory method that uses a polarization as keyword argument.

If needed, also new polarization models can be registered via
:meth:`~sionna.rt.register_polarization_model`.


.. _dev_custom_antenna_patterns_grad:

Gradient-based Optimization
---------------------------

Thanks to Dr.Jit's `automatic differentiation <https://drjit.readthedocs.io/en/latest/autodiff.html>`_ capabilities, it is possible to
define antenna patterns with parameters that can be optimized via gradient descent.
In the following example, we will create a new antenna pattern that consists of
a single spherical Gaussian with trainable mean direction and sharpness.

.. code-block:: python

    import mitsuba as mi
    import drjit as dr
    from sionna.rt import r_hat, PolarizedAntennaPattern, register_antenna_pattern

    class Trainable_V_Pattern:
        """Trainable vertically polarized antenna pattern function

        Defined via a spherical Gaussian with trainable mean direction
        and sharpness.
        """
        def __init__(self, opt: mi.ad.Optimizer):

            # Add trainable target directions to optimizer
            opt["theta_t"] = mi.Float(dr.pi/2)
            opt["phi_t"] = mi.Float(0)

            # Add trainable sharpness to optimizer
            opt["lambda"] = mi.Float(1)

            self.opt = opt

        def __call__(self, theta, phi):
            mu = r_hat(self.opt["theta_t"], self.opt["phi_t"])
            v = r_hat(theta, phi)
            gain = 2*self.opt["lambda"]*dr.rcp(1-dr.exp(-2*self.opt["lambda"])) \
                    *dr.exp(self.opt["lambda"]*(dr.dot(mu, v) - 1))
            c_theta_real = dr.sqrt(gain)
            return mi.Complex2f(c_theta_real, 0)

    # The factory method requires a new keyword argument `opt` which must be
    # a Mitsuba optimizer
    def trainable_pattern_factory(*, opt, polarization, polarization_model="tr38901_2"):
        return PolarizedAntennaPattern(
            v_pattern=Trainable_V_Pattern(opt),
            polarization=polarization,
            polarization_model=polarization_model
        )

    register_antenna_pattern("trainable", trainable_pattern_factory)

Let us now load and empty scene in which we place a transmitter and a receiver.
The transmitter has an antenna array using our newly defined trainable antenna
pattern.

.. code-block:: python

    from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver

    # Load empty scene
    scene = load_scene()

    # Create a Mitsuba Optimizer
    opt = mi.ad.Adam(lr=1e-2)

    # Define transmit array with trainable antenna pattern
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1,
                                 pattern="trainable",
                                 opt=opt,
                                 polarization="V")

    scene.rx_array = PlanarArray(num_rows=1, num_cols=1,
                                 pattern="iso",
                                 polarization="V")

    # Add transmitter and receiver to the scene
    scene.add(Transmitter(name="tx", position=[0,0,0]))
    scene.add(Receiver(name="rx", position=[10,10,10]))

Next, we will compute propagation paths and compute gradients of the
total receiver power with respect to the trainable parameters of the
antenna patterns.

.. code-block:: python

    solver = PathSolver()

    # Switch the computation of field loop to "evaluated" mode to
    # enable gradient backpropagation through the loop
    solver.field_calculator.loop_mode = "evaluated"

    # Compute propagation paths
    paths = solver(scene, max_depth=0)

    # Compute total received power
    a_r, a_i = paths.a
    power = dr.sum(a_r**2 + a_i**2)

    # Compute gradients
    dr.backward(power)
    print(opt.variables["theta_t"].grad)
    print(opt.variables["phi_t"].grad)
    print(opt.variables["lambda"].grad)

::

    [-1.35529e-07]
    [1.35529e-07]
    [6.20459e-08]
