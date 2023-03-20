Primer on Electromagnetics
##########################

This section provides useful background for the general understanding of ray tracing for wireless propagation modelling. In particular, our goal is to provide a concise definition of a `channel impulse response` between a transmitting and receiving antenna, as done in (Ch. 2 & 3) [Wiesbeck]_. The notations and definitions will be used in the API documentation of Sionna's :doc:`Ray Tracing module <api/rt>`.


Coordinate system, rotations, and vector fields
***********************************************

We consider a global coordinate system (GCS) with Cartesian standard basis :math:`\hat{\mathbf{x}}`, :math:`\hat{\mathbf{y}}`, :math:`\hat{\mathbf{z}}`.
The spherical unit vectors are defined as

.. math::
    :label: spherical_vecs

    \begin{align}
        \hat{\mathbf{r}}          (\theta, \varphi) &= \sin(\theta)\cos(\varphi) \hat{\mathbf{x}} + \sin(\theta)\sin(\varphi) \hat{\mathbf{y}} + \cos(\theta)\hat{\mathbf{z}}\\
        \hat{\boldsymbol{\theta}} (\theta, \varphi) &= \cos(\theta)\cos(\varphi) \hat{\mathbf{x}} + \cos(\theta)\sin(\varphi) \hat{\mathbf{y}} - \sin(\theta)\hat{\mathbf{z}}\\
        \hat{\boldsymbol{\varphi}}(\theta, \varphi) &=            -\sin(\varphi) \hat{\mathbf{x}} +             \cos(\varphi) \hat{\mathbf{y}}.
    \end{align}

For an arbitrary unit norm vector :math:`\hat{\mathbf{v}} = (x, y, z)`, the elevation and azimuth angles :math:`\theta` and :math:`\varphi` can be computed as

.. math::
    :label: theta_phi

    \theta  &= \cos^{-1}(z) \\
    \varphi &= \mathop{\text{atan2}}(y, x)

where :math:`\mathop{\text{atan2}}(y, x)` is the two-argument inverse tangent function [atan2]_.
A 3D rotation with yaw, pitch, and roll angles :math:`\alpha`, :math:`\beta`, and :math:`\gamma`, respectively, is expressed by the matrix

.. math::
    :label: rotation

    \begin{align}
        \mathbf{R}(\alpha, \beta, \gamma) = \mathbf{R}_z(\alpha)\mathbf{R}_y(\beta)\mathbf{R}_x(\gamma)
    \end{align}

where :math:`\mathbf{R}_z(\alpha)`, :math:`\mathbf{R}_y(\beta)`, and :math:`\mathbf{R}_x(\gamma)` are rotation matrices around the :math:`z`, :math:`y`, and :math:`x` axes, respectively, which are defined as

.. math::
    \begin{align}
        \mathbf{R}_z(\alpha) &= \begin{pmatrix}
                        \cos(\alpha) & -\sin(\alpha) & 0\\
                        \sin(\alpha) & \cos(\alpha) & 0\\
                        0 & 0 & 1
                      \end{pmatrix}\\
        \mathbf{R}_y(\beta) &= \begin{pmatrix}
                        \cos(\beta) & 0 & \sin(\beta)\\
                        0 & 1 & 0\\
                        -\sin(\beta) & 0 & \cos(\beta)
                      \end{pmatrix}\\
        \mathbf{R}_x(\gamma) &= \begin{pmatrix}
                            1 & 0 & 0\\
                            0 & \cos(\gamma) & -\sin(\gamma)\\
                            0 & \sin(\gamma) & \cos(\gamma)
                      \end{pmatrix}.
    \end{align}

A closed-form expression for :math:`\mathbf{R}(\alpha, \beta, \gamma)` can be found in (7.1-4) [TR38901]_.
The reverse rotation is simply defined by :math:`\mathbf{R}^{-1}(\alpha, \beta, \gamma)=\mathbf{R}^\mathsf{T}(\alpha, \beta, \gamma)`.
A vector :math:`\mathbf{x}` defined in a first coordinate system is represented in a second coordinate system rotated by :math:`\mathbf{R}(\alpha, \beta, \gamma)` with respect to the first one as :math:`\mathbf{x}'=\mathbf{R}^\mathsf{T}(\alpha, \beta, \gamma)\mathbf{x}`.
If a point in the first coordinate system has spherical angles :math:`(\theta, \varphi)`, the corresponding angles :math:`(\theta', \varphi')` in the second coordinate system can be found to be

.. math::
    :label: theta_phi_prime

    \begin{align}
        \theta' &= \cos^{-1}\left( \mathbf{z}^\mathsf{T} \mathbf{R}^\mathsf{T}(\alpha, \beta, \gamma)\hat{\mathbf{r}}(\theta, \varphi)          \right)\\
        \varphi' &= \arg\left( \left( \mathbf{x} + j\mathbf{y}\right)^\mathsf{T} \mathbf{R}^\mathsf{T}(\alpha, \beta, \gamma)\hat{\mathbf{r}}(\theta, \varphi) \right).
    \end{align}

For a vector field :math:`\mathbf{F}'(\theta',\varphi')` expressed in local spherical coordinates

.. math::
    \mathbf{F}'(\theta',\varphi') = F_{\theta'}(\theta',\varphi')\hat{\boldsymbol{\theta}}'(\theta',\varphi') + F_{\varphi'}(\theta',\varphi')\hat{\boldsymbol{\varphi}}'(\theta',\varphi')

that are rotated by :math:`\mathbf{R}=\mathbf{R}(\alpha, \beta, \gamma)` with respect to the GCS, the spherical field components in the GCS can be expressed as

.. math::
    :label: F_prime_2_F

    \begin{bmatrix}
        F_\theta(\theta, \varphi) \\
        F_\varphi(\theta, \varphi)
    \end{bmatrix} =
    \begin{bmatrix}
        \hat{\boldsymbol{\theta}}(\theta,\varphi)^\mathsf{T}\mathbf{R}\hat{\boldsymbol{\theta}}'(\theta',\varphi') & \hat{\boldsymbol{\theta}}(\theta,\varphi)^\mathsf{T}\mathbf{R}\hat{\boldsymbol{\varphi}}'(\theta',\varphi') \\
        \hat{\boldsymbol{\varphi}}(\theta,\varphi)^\mathsf{T}\mathbf{R}\hat{\boldsymbol{\theta}}'(\theta',\varphi') & \hat{\boldsymbol{\varphi}}(\theta,\varphi)^\mathsf{T}\mathbf{R}\hat{\boldsymbol{\varphi}}'(\theta',\varphi')
    \end{bmatrix}
    \begin{bmatrix}
        F_{\theta'}(\theta', \varphi') \\
        F_{\varphi'}(\theta', \varphi')
    \end{bmatrix}

so that

.. math::
    \mathbf{F}(\theta,\varphi) = F_{\theta}(\theta,\varphi)\hat{\boldsymbol{\theta}}(\theta,\varphi) + F_{\varphi}(\theta,\varphi)\hat{\boldsymbol{\varphi}}(\theta,\varphi).

Planar Time-Harmonic Waves
**************************

A time-harmonic planar electric wave :math:`\mathbf{E}(\mathbf{x}, t)\in\mathbb{C}^3` travelling in a homogeneous medium with wave vector :math:`\mathbf{k}\in\mathbb{C}^3` can be described at position :math:`\mathbf{x}\in\mathbb{R}^3` and time :math:`t` as

.. math::
    \begin{align}
        \mathbf{E}(\mathbf{x}, t) &= \mathbf{E}_0 e^{j(\omega t -\mathbf{k}^{\mathsf{H}}\mathbf{x})}\\
                                  &= \mathbf{E}(\mathbf{x}) e^{j\omega t}
    \end{align}

where :math:`\mathbf{E}_0\in\mathbb{C}^3` is the field phasor. The wave vector can be decomposed as :math:`\mathbf{k}=k \hat{\mathbf{k}}`, where :math:`\hat{\mathbf{k}}` is a unit norm vector, :math:`k=\omega\sqrt{\varepsilon\mu}` is the wave number, and :math:`\omega=2\pi f` is the angular frequency. The permittivity :math:`\varepsilon` and permeability :math:`\mu` are defined as

.. math::
    :label: epsilon

    \varepsilon = \eta \varepsilon_0

.. math::
    :label: mu

    \mu = \mu_r \mu_0

where :math:`\eta` and :math:`\varepsilon_0` are the complex relative and vacuum permittivities, :math:`\mu_r` and :math:`\mu_0` are the relative and vacuum permeabilities, and :math:`\sigma` is the conductivity.
The complex relative permittivity :math:`\eta` is given as

.. math::
    :label: eta

    \eta = \varepsilon_r - j\frac{\sigma}{\varepsilon_0\omega}

where :math:`\varepsilon_r` is the real relative permittivity of a non-conducting dielectric.

With these definitions, the speed of light is given as (Eq. 4-28d) [Balanis]_

.. math::
    c=\frac{1}{\sqrt{\varepsilon_0\varepsilon_r\mu}}\left\{\frac12\left(\sqrt{1+\left(\frac{\sigma}{\omega\varepsilon_0\varepsilon_r}\right)^2}+1\right)\right\}^{-\frac{1}{2}}

where the factor in curly brackets vanishes for non-conducting materials. The speed of light in vacuum is denoted :math:`c_0=\frac{1}{\sqrt{\varepsilon_0 \mu_0}}` and the vacuum wave number :math:`k_0=\frac{\omega}{c_0}`. In conducting materials, the wave number is complex which translates to propagation losses.

The associated magnetic field :math:`\mathbf{H}(\mathbf{x}, t)\in\mathbb{C}^3` is

.. math::
    \mathbf{H}(\mathbf{x}, t) = \frac{\hat{\mathbf{k}}\times  \mathbf{E}(\mathbf{x}, t)}{Z} = \mathbf{H}(\mathbf{x})e^{j\omega t}

where :math:`Z=\sqrt{\mu/\varepsilon}` is the wave impedance.  The vacuum impedance is denoted :math:`Z_0=\sqrt{\mu_0/\varepsilon_0}\approx 376.73\,\Omega`.

The time-averaged Poynting vector is defined as

.. math::
        \mathbf{S}(\mathbf{x}) = \frac{1}{2} \Re\left\{\mathbf{E}(\mathbf{x})\times  \mathbf{H}(\mathbf{x})\right\}
                               = \frac{1}{2} \Re\left\{\frac{1}{Z} \right\} \lVert \mathbf{E}(\mathbf{x})  \rVert^2 \hat{\mathbf{k}}

which describes the directional energy flux (W/m²), i.e., energy transfer per unit area per unit time.

Note that the actual electromagnetic waves are the real parts of :math:`\mathbf{E}(\mathbf{x}, t)` and :math:`\mathbf{H}(\mathbf{x}, t)`.

.. _far_field:

Far Field of a Transmitting Antenna
***********************************

We assume that the electric far field of an antenna in free space can be described by a spherical wave originating from the center of the antenna:

.. math::
    \mathbf{E}(r, \theta, \varphi, t) = \mathbf{E}(r,\theta, \varphi) e^{j\omega t} = \mathbf{E}_0(\theta, \varphi) \frac{e^{-jk_0r}}{r} e^{j\omega t}

where :math:`\mathbf{E}_0(\theta, \varphi)` is the electric field phasor, :math:`r` is the distance (or radius), :math:`\theta` the zenith angle, and :math:`\varphi` the azimuth angle.
In contrast to a planar wave, the field strength decays as :math:`1/r`.

The complex antenna field pattern :math:`\mathbf{F}(\theta, \varphi)` is defined as

.. math::
    :label: F

    \begin{align}
        \mathbf{F}(\theta, \varphi) = \frac{ \mathbf{E}_0(\theta, \varphi)}{\max_{\theta,\varphi}\lVert  \mathbf{E}_0(\theta, \varphi) \rVert}.
    \end{align}

The time-averaged Poynting vector for such a spherical wave is

.. math::
    :label: S_spherical

    \mathbf{S}(r, \theta, \varphi) = \frac{1}{2Z_0}\lVert \mathbf{E}(r, \theta, \varphi) \rVert^2 \hat{\mathbf{r}}

where :math:`\hat{\mathbf{r}}` is the radial unit vector. It simplifies for an ideal isotropic antenna with input power :math:`P_\text{T}` to

.. math::
    \mathbf{S}_\text{iso}(r, \theta, \varphi) = \frac{P_\text{T}}{4\pi r^2} \hat{\mathbf{r}}.

The antenna gain :math:`G` is the ratio of the maximum radiation power density of the antenna in radial direction and that of an ideal isotropic radiating antenna:

.. math::
    :label: G

        G = \frac{\max_{\theta,\varphi}\lVert \mathbf{S}(r, \theta, \varphi)\rVert}{ \lVert\mathbf{S}_\text{iso}(r, \theta, \varphi)\rVert}
          = \frac{2\pi}{Z_0 P_\text{T}} \max_{\theta,\varphi}\lVert \mathbf{E}_0(\theta, \varphi) \rVert^2.

One can similarly define a gain with directional dependency by ignoring the computation of the maximum the last equation:

.. math::
    :label: Gdir

        G(\theta, \varphi) = \frac{2\pi}{Z_0 P_\text{T}} \lVert \mathbf{E}_0(\theta, \varphi) \rVert^2 = G \lVert \mathbf{F}(\theta, \varphi) \rVert^2.

If one uses in the last equation the radiated power :math:`P=\eta_\text{rad} P_\text{T}`, where :math:`\eta_\text{rad}` is the radiation efficiency, instead of the input power :math:`P_\text{T}`, one obtains the directivity :math:`D(\theta,\varphi)`. Both are related through :math:`G(\theta, \varphi)=\eta_\text{rad} D(\theta, \varphi)`.

.. admonition:: Antenna pattern

    Since :math:`\mathbf{F}(\theta, \varphi)` contains no information about the maximum gain :math:`G` and :math:`G(\theta, \varphi)` does not carry any phase information, we define the `antenna pattern` :math:`\mathbf{C}(\theta, \varphi)` as

    .. math::
        :label: C

        \mathbf{C}(\theta, \varphi) = \sqrt{G}\mathbf{F}(\theta, \varphi)

    such that :math:`G(\theta, \varphi)= \lVert\mathbf{C}(\theta, \varphi) \rVert^2`.

    Using the spherical unit vectors :math:`\hat{\boldsymbol{\theta}}\in\mathbb{R}^3`
    and :math:`\hat{\boldsymbol{\varphi}}\in\mathbb{R}^3`,
    we can rewrite :math:`\mathbf{C}(\theta, \varphi)` as

    .. math::
        \mathbf{C}(\theta, \varphi) = C_\theta(\theta,\varphi) \hat{\boldsymbol{\theta}} + C_\varphi(\theta,\varphi) \hat{\boldsymbol{\varphi}}

    where :math:`C_\theta(\theta,\varphi)\in\mathbb{C}` and :math:`C_\varphi(\theta,\varphi)\in\mathbb{C}` are the
    `zenith pattern` and `azimuth pattern`, respectively.

Combining :eq:`F` and :eq:`G`, we can obtain the following expression of the electric far field

.. math::
    :label: E_T

    \mathbf{E}_\text{T}(r,\theta_\text{T},\varphi_\text{T}) = \sqrt{ \frac{P_\text{T} G_\text{T} Z_0}{2\pi}} \frac{e^{-jk_0 r}}{r} \mathbf{F}_\text{T}(\theta_\text{T}, \varphi_\text{T})

where we have added the subscript :math:`\text{T}` to all quantities that are specific to the transmitting antenna.

The input power :math:`P_\text{T}` of an antenna with (conjugate matched) impedance :math:`Z_\text{T}`, fed by a voltage source with complex amplitude :math:`V_\text{T}`, is given by (see, e.g., [Wikipedia]_)

.. math::
    :label: P_T

    P_\text{T} = \frac{|V_\text{T}|^2}{8\Re\{Z_\text{T}\}}.

.. admonition:: Normalization of antenna patterns

    The radiated power :math:`\eta_\text{rad} P_\text{T}` of an antenna can be obtained by integrating the Poynting vector over the surface of a closed sphere of radius :math:`r` around the antenna:

    .. math::
        \begin{align}
            \eta_\text{rad} P_\text{T} &=  \int_0^{2\pi}\int_0^{\pi} \mathbf{S}(r, \theta, \varphi)^\mathsf{T} \hat{\mathbf{r}} r^2 \sin(\theta)d\theta d\varphi \\
                            &= \int_0^{2\pi}\int_0^{\pi} \frac{1}{2Z_0} \lVert \mathbf{E}(r, \theta, \varphi) \rVert^2 r^2\sin(\theta)d\theta d\varphi \\
                            &= \frac{P_\text{T}}{4 \pi} \int_0^{2\pi}\int_0^{\pi} G(\theta, \varphi) \sin(\theta)d\theta d\varphi.
        \end{align}

    We can see from the last equation that the directional gain of any antenna must satisfy

    .. math::
        \int_0^{2\pi}\int_0^{\pi} G(\theta, \varphi) \sin(\theta)d\theta d\varphi = 4 \pi \eta_\text{rad}.

Modelling of a Receiving Antenna
********************************

Although the transmitting antenna radiates a spherical wave :math:`\mathbf{E}_\text{T}(r,\theta_\text{T},\varphi_\text{T})`,
we assume that the receiving antenna observes a planar incoming wave :math:`\mathbf{E}_\text{R}` that arrives from the angles :math:`\theta_\text{R}` and :math:`\varphi_\text{R}`
which are defined in the local spherical coordinates of the receiving antenna. The Poynting vector of the incoming wave :math:`\mathbf{S}_\text{R}` is hence :eq:`S_spherical`

.. math::
    :label: S_R

    \mathbf{S}_\text{R} = -\frac{1}{2Z_0} \lVert \mathbf{E}_\text{R} \rVert^2 \hat{\mathbf{r}}(\theta_\text{R}, \varphi_\text{R})

where :math:`\hat{\mathbf{r}}(\theta_\text{R}, \varphi_\text{R})` is the radial unit vector in the spherical coordinate system of the receiver.

The aperture or effective area :math:`A_\text{R}` of an antenna with gain :math:`G_\text{R}` is defined as the ratio of the available received power :math:`P_\text{R}` at the output of the antenna and the absolute value of the Poynting vector, i.e., the power density:

.. math::
    :label: A_R

    A_\text{R} = \frac{P_\text{R}}{\lVert \mathbf{S}_\text{R}\rVert} = G_\text{R}\frac{\lambda^2}{4\pi}

where :math:`\frac{\lambda^2}{4\pi}` is the aperture of an isotropic antenna. In the definition above, it is assumed that the antenna is ideally directed towards and polarization matched to the incoming wave.
For an arbitrary orientation of the antenna (but still assuming polarization matching), we can define a direction dependent effective area

.. math::
    :label: A_dir

    A_\text{R}(\theta_\text{R}, \varphi_\text{R}) = G_\text{R}(\theta_\text{R}, \varphi_\text{R})\frac{\lambda^2}{4\pi}.

The available received power at the output of the antenna can be expressed as

.. math::
    :label: P_R

    P_\text{R} = \frac{|V_\text{R}|^2}{8\Re\{Z_\text{R}\}}

where :math:`Z_\text{R}` is the impedance of the receiving antenna and :math:`V_\text{R}` the open circuit voltage.

We can now combine :eq:`P_R`, :eq:`A_dir`, and :eq:`A_R` to obtain the following expression for the absolute value of the voltage :math:`|V_\text{R}|`
assuming matched polarization:

.. math::
    \begin{align}
        |V_\text{R}| &= \sqrt{P_\text{R} 8\Re\{Z_\text{R}\}}\\
                     &= \sqrt{\frac{\lambda^2}{4\pi} G_\text{R}(\theta_\text{R}, \varphi_\text{R}) \frac{8\Re\{Z_\text{R}\}}{2 Z_0} \lVert \mathbf{E}_\text{R} \rVert^2}\\
                     &= \sqrt{\frac{\lambda^2}{4\pi} G_\text{R} \frac{4\Re\{Z_\text{R}\}}{Z_0}} \lVert \mathbf{F}_\text{R}(\theta_\text{R}, \varphi_\text{R})\rVert\lVert\mathbf{E}_\text{R}\rVert.
    \end{align}

By extension of the previous equation, we can obtain an expression for :math:`V_\text{R}` which is valid for
arbitrary polarizations of the incoming wave and the receiving antenna:

.. math::
    :label: V_R

    V_\text{R} = \sqrt{\frac{\lambda^2}{4\pi} G_\text{R} \frac{4\Re\{Z_\text{R}\}}{Z_0}} \mathbf{F}_\text{R}(\theta_\text{R}, \varphi_\text{R})^{\mathsf{H}}\mathbf{E}_\text{R}.

.. admonition:: Example: Recovering Friis equation

    In the case of free space propagation, we have :math:`\mathbf{E}_\text{R}=\mathbf{E}_\text{T}(r,\theta_\text{T},\varphi_\text{T})`.
    Combining :eq:`V_R`, :eq:`P_R`, and :eq:`E_T`, we obtain the following expression for the received power:

    .. math::
        P_\text{R} = \left(\frac{\lambda}{4\pi r}\right)^2 G_\text{R} G_\text{T} P_\text{T} \left|\mathbf{F}_\text{R}(\theta_\text{R}, \varphi_\text{R})^{\mathsf{H}} \mathbf{F}_\text{T}(\theta_\text{T}, \varphi_\text{T})\right|^2.

    It is important that :math:`\mathbf{F}_\text{R}` and :math:`\mathbf{F}_\text{T}` are expressed in the same coordinate system for the last equation to make sense.
    For perfect orientation and polarization matching, we can recover the well-known Friis transmission equation:

    .. math::
        \frac{P_\text{R}}{P_\text{T}} = \left(\frac{\lambda}{4\pi r}\right)^2 G_\text{R} G_\text{T}.


General Propagation Path
************************

A single propagation path consists of a cascade of multiple scattering processes, where a scattering process can be anything that prevents the wave from propagating as in free space. This includes reflection, refraction, diffraction, and diffuse scattering. For each scattering process, one needs to compute a relationship between the incoming field at the scatter center and the created far field at the next scatter center or the receiving antenna.
We can represent this cascade of scattering processes by a single matrix :math:`\widetilde{\mathbf{T}}`
that describes the transformation that the radiated field :math:`\mathbf{E}_\text{T}(r, \theta_\text{T}, \varphi_\text{T})` undergoes until it reaches the receiving antenna:

.. math::
    :label: E_R

    \mathbf{E}_\text{R} = \sqrt{ \frac{P_\text{T} G_\text{T} Z_0}{2\pi}} \widetilde{\mathbf{T}} \mathbf{F}_\text{T}(\theta_\text{T}, \varphi_\text{T}).

Note that we have obtained this expression by replacing the free space propagation term :math:`\frac{e^{-jk_0r}}{r}` in :eq:`E_T` by the matrix :math:`\widetilde{\mathbf{T}}`. This requires that all quantities are expressed in the same coordinate system which is also assumed in the following expressions. Further, it is assumed that the matrix :math:`\widetilde{\mathbf{T}}` includes the necessary coordinate transformations.

Plugging :eq:`E_R` into :eq:`V_R`, we can obtain a general expression for the received voltage of a propagation path:

.. math::
    V_\text{R} = \sqrt{\left(\frac{\lambda}{4\pi}\right)^2 G_\text{R}G_\text{T}P_\text{T} 8\Re\{Z_\text{R}\}} \,\mathbf{F}_\text{R}(\theta_\text{R}, \varphi_\text{R})^{\mathsf{H}}\widetilde{\mathbf{T}} \mathbf{F}_\text{T}(\theta_\text{T}, \varphi_\text{T}).

If the electromagnetic wave arrives at the receiving antenna over :math:`N` propagation paths, we can simply add the received voltages
from all paths to obtain

.. math::
    :label: V_Rmulti

    \begin{align}
    V_\text{R} &= \sqrt{\left(\frac{\lambda}{4\pi}\right)^2 G_\text{R}G_\text{T}P_\text{T} 8\Re\{Z_\text{R}\}} \sum_{n=1}^N\mathbf{F}_\text{R}(\theta_{\text{R},i}, \varphi_{\text{R},i})^{\mathsf{H}}\widetilde{\mathbf{T}}_i \mathbf{F}_\text{T}(\theta_{\text{T},i}, \varphi_{\text{T},i})\\
    &= \sqrt{\left(\frac{\lambda}{4\pi}\right)^2 P_\text{T} 8\Re\{Z_\text{R}\}} \sum_{n=1}^N\mathbf{C}_\text{R}(\theta_{\text{R},i}, \varphi_{\text{R},i})^{\mathsf{H}}\widetilde{\mathbf{T}}_i \mathbf{C}_\text{T}(\theta_{\text{T},i}, \varphi_{\text{T},i})
    \end{align}

where all path-dependent quantities carry the subscript :math:`i`. Note that the matrices :math:`\widetilde{\mathbf{T}}_i` also ensure appropriate scaling so that the total received power can never be larger than the transmit power.


Frequency & Impulse Response
****************************

The channel frequency response :math:`H(f)` at frequency :math:`f=\frac{c}{\lambda}` is defined as the ratio between the received voltage and the voltage at the input to the transmitting antenna:

.. math::
    :label: H

    H(f) = \frac{V_\text{R}}{V_\text{T}} = \frac{V_\text{R}}{|V_\text{T}|}

where it is assumed that the input voltage has zero phase.

It is useful to separate phase shifts due to wave propagation from the transfer matrices :math:`\widetilde{\mathbf{T}}_i`. If we denote by :math:`r_i` the total length of path :math:`i` with average propagation speed :math:`c_i`, the path delay is :math:`\tau_i=r_i/c_i`. We can now define the new transfer matrix

.. math::
    :label: T_tilde

    \mathbf{T}_i=\widetilde{\mathbf{T}}_ie^{j2\pi f \tau_i}.

Using :eq:`P_T` and :eq:`T_tilde` in :eq:`V_Rmulti` while assuming equal real parts of both antenna impedances, i.e., :math:`\Re\{Z_\text{T}\}=\Re\{Z_\text{R}\}` (which is typically the case), we obtain the final expression for the channel frequency response:

.. math::
    :label: H_final

    \boxed{H(f) = \sum_{i=1}^N \underbrace{\frac{\lambda}{4\pi} \mathbf{C}_\text{R}(\theta_{\text{R},i}, \varphi_{\text{R},i})^{\mathsf{H}}\mathbf{T}_i \mathbf{C}_\text{T}(\theta_{\text{T},i}, \varphi_{\text{T},i})}_{\triangleq a_i} e^{-j2\pi f\tau_i}}

Taking the inverse Fourier transform, we finally obtain the channel impulse response

.. math::
    :label: h_final2

    \boxed{h(\tau) = \int_{-\infty}^{\infty} H(f) e^{j2\pi f \tau} df = \sum_{i=1}^N a_i \delta(\tau-\tau_i)}

Reflection and refraction
*************************

When a plane wave hits a plane interface which separates two materials, e.g., air and concrete, a part of the wave gets reflected and the other transmitted (or *refracted*), i.e., it propagates into the other material.  We assume in the following description that both materials are uniform non-magnetic dielectrics, i.e., :math:`\mu_r=1`, and follow the definitions as in [ITURP20402]_. The incoming wave phasor :math:`\mathbf{E}_\text{i}` is expressed by two arbitrary orthogonal polarization components, i.e.,

.. math::
    \mathbf{E}_\text{i} = E_{\text{i},s} \hat{\mathbf{e}}_{\text{i},s} + E_{\text{i},p} \hat{\mathbf{e}}_{\text{i},p}

which are both orthogonal to the incident wave vector, i.e., :math:`\hat{\mathbf{e}}_{\text{i},s}^{\mathsf{T}} \hat{\mathbf{e}}_{\text{i},p}=\hat{\mathbf{e}}_{\text{i},s}^{\mathsf{T}} \hat{\mathbf{k}}_\text{i}=\hat{\mathbf{e}}_{\text{i},p}^{\mathsf{T}} \hat{\mathbf{k}}_\text{i} =0`.

.. _fig_reflection:
.. figure:: figures/reflection.svg
        :align: center
        :width: 90 %

        Reflection and refraction of a plane wave at a plane interface between two materials.

:numref:`fig_reflection` shows reflection and refraction of the incoming wave at the plane interface between two materials with relative permittivities :math:`\eta_1` and :math:`\eta_2`. The coordinate system is chosen such that the wave vectors of the incoming, reflected, and transmitted waves lie within the plane of incidence, which is chosen to be the x-z plane. The normal vector of the interface :math:`\hat{\mathbf{n}}` is pointing toward the negative z axis.
The incoming wave is must be represented in a different basis, i.e., in the form two different orthogonal polarization components :math:`E_{\text{i}, \perp}` and :math:`E_{\text{i}, \parallel}`, i.e.,

.. math::
    \mathbf{E}_\text{i} = E_{\text{i},\perp} \hat{\mathbf{e}}_{\text{i},\perp} + E_{\text{i},\parallel} \hat{\mathbf{e}}_{\text{i},\parallel}

where the former is orthogonal to the plane of incidence and called transverse electric (TE) polarization (left), and the latter is parallel to the plane of incidence and called transverse magnetic (TM) polarization (right). We adopt in the following the convention that all transverse components are coming out of the figure (indicated by the :math:`\odot` symbol). One can easily verify that the following relationships must hold:

.. math::
    \begin{align}
    \hat{\mathbf{e}}_{\text{i},\perp} &= \frac{\hat{\mathbf{k}}_\text{i} \times \hat{\mathbf{n}}}{\lVert \hat{\mathbf{k}}_\text{i} \times \hat{\mathbf{n}} \rVert} \\
    \hat{\mathbf{e}}_{\text{i},\parallel} &= \frac{\hat{\mathbf{e}}_{\text{i},\perp} \times \hat{\mathbf{k}}_\text{i}}{\lVert \hat{\mathbf{e}}_{\text{i},\perp} \times \hat{\mathbf{k}}_\text{i} \rVert}\\
    \begin{bmatrix}E_{\text{i},\perp} \\ E_{\text{i},\parallel} \end{bmatrix} &=
        \begin{bmatrix}
            \hat{\mathbf{e}}_{\text{i},\perp}^\mathsf{T}\hat{\mathbf{e}}_{\text{i},s} & \hat{\mathbf{e}}_{\text{i},\perp}^\mathsf{T}\hat{\mathbf{e}}_{\text{i},p}\\
            \hat{\mathbf{e}}_{\text{i},\parallel}^\mathsf{T}\hat{\mathbf{e}}_{\text{i},s} & \hat{\mathbf{e}}_{\text{i},\parallel}^\mathsf{T}\hat{\mathbf{e}}_{\text{i},p}
        \end{bmatrix}
     \begin{bmatrix}E_{\text{i},s} \\ E_{\text{i},p}\end{bmatrix}.
    \end{align}


While the angles of incidence and reflection are both equal to :math:`\theta_1`, the angle of the refracted wave :math:`\theta_2` is given by Snell's law:

.. math::
    \sin(\theta_2) = \sqrt{\frac{\eta_1}{\eta_2}} \sin(\theta_1)

or, equivalently,

.. math::
    \cos(\theta_2) = \sqrt{1 - \frac{\eta_1}{\eta_2} \sin^2(\theta_1)}.

The reflected and transmitted wave phasors :math:`\mathbf{E}_\text{r}` and :math:`\mathbf{E}_\text{t}` are similarly represented as

.. math::
    \begin{align}
        \mathbf{E}_\text{r} &= E_{\text{r},\perp} \hat{\mathbf{e}}_{\text{r},\perp} + E_{\text{r},\parallel} \hat{\mathbf{e}}_{\text{r},\parallel}\\
        \mathbf{E}_\text{t} &= E_{\text{t},\perp} \hat{\mathbf{e}}_{\text{t},\perp} + E_{\text{t},\parallel} \hat{\mathbf{e}}_{\text{t},\parallel}
    \end{align}

where

.. math::
    \begin{align}
        \hat{\mathbf{e}}_{\text{r},\perp} &= \hat{\mathbf{e}}_{\text{i},\perp}\\
        \hat{\mathbf{e}}_{\text{r},\parallel} &= \frac{\hat{\mathbf{e}}_{\text{r},\perp}\times\hat{\mathbf{k}}_\text{r}}{\lVert \hat{\mathbf{e}}_{\text{r},\perp}\times\hat{\mathbf{k}}_\text{r} \rVert}\\
        \hat{\mathbf{e}}_{\text{t},\perp} &= \hat{\mathbf{e}}_{\text{i},\perp}\\
        \hat{\mathbf{e}}_{\text{t},\parallel} &= \frac{\hat{\mathbf{e}}_{\text{t},\perp}\times\hat{\mathbf{k}}_\text{t}}{ \Vert \hat{\mathbf{e}}_{\text{t},\perp}\times\hat{\mathbf{k}}_\text{t} \rVert}.
    \end{align}

The *Fresnel* equations provide relationships between the incident, reflected, and refracted field components (for :math:`\sqrt{\left| \eta_1/\eta_2 \right|}\sin(\theta_1)<1`):

.. math::
    :label: fresnel

    \begin{align}
        r_{\perp}     &= \frac{E_{\text{r}, \perp    }}{E_{\text{i}, \perp    }} = \frac{ \sqrt{\eta_1}\cos(\theta_1) - \sqrt{\eta_2}\cos(\theta_2) }{ \sqrt{\eta_1}\cos(\theta_1) + \sqrt{\eta_2}\cos(\theta_2) } \\
        r_{\parallel} &= \frac{E_{\text{r}, \parallel}}{E_{\text{i}, \parallel}} = \frac{ \sqrt{\eta_2}\cos(\theta_1) - \sqrt{\eta_1}\cos(\theta_2) }{ \sqrt{\eta_2}\cos(\theta_1) + \sqrt{\eta_1}\cos(\theta_2) } \\
        t_{\perp}     &= \frac{E_{\text{t}, \perp    }}{E_{\text{t}, \perp    }} = \frac{ 2\sqrt{\eta_1}\cos(\theta_1) }{ \sqrt{\eta_1}\cos(\theta_1) + \sqrt{\eta_2}\cos(\theta_2) } \\
        t_{\parallel} &= \frac{E_{\text{t}, \parallel}}{E_{\text{t}, \parallel}} = \frac{ 2\sqrt{\eta_1}\cos(\theta_1) }{ \sqrt{\eta_2}\cos(\theta_1) + \sqrt{\eta_1}\cos(\theta_2) }.
    \end{align}

If :math:`\sqrt{\left| \eta_1/\eta_2 \right|}\sin(\theta_1)\ge 1`, we have :math:`r_{\perp}=r_{\parallel}=1` and :math:`t_{\perp}=t_{\parallel}=0`, i.e., total reflection.

For the case of an incident wave in vacuum, i.e., :math:`\eta_1=1`, the Fresnel equations :eq:`fresnel` simplify to

.. math::
    \begin{align}
        r_{\perp}     &= \frac{\cos(\theta_1) -\sqrt{\eta_2 -\sin^2(\theta_1)}}{\cos(\theta_1) +\sqrt{\eta_2 -\sin^2(\theta_1)}} \\
        r_{\parallel} &= \frac{\eta_2\cos(\theta_1) -\sqrt{\eta_2 -\sin^2(\theta_1)}}{\eta_2\cos(\theta_1) +\sqrt{\eta_2 -\sin^2(\theta_1)}} \\
        t_{\perp}     &= \frac{2\cos(\theta_1)}{\cos(\theta_1) + \sqrt{\eta_2-\sin^2(\theta_1)}}\\
        t_{\parallel} &= \frac{2\sqrt{\eta_2}\cos(\theta_1)}{\eta_2 \cos(\theta_1) + \sqrt{\eta_2-\sin^2(\theta_1)}}.
    \end{align}

Putting everything together, we obtain the following relationships between incident, reflected, and transmitted waves:

.. math::
    \begin{align}
        \begin{bmatrix}E_{\text{r},\perp} \\ E_{\text{r},\parallel} \end{bmatrix} &=
        \begin{bmatrix}
            r_{\perp} & 0 \\
            0         & r_{\parallel}
        \end{bmatrix}
        \begin{bmatrix}
            \hat{\mathbf{e}}_{\text{i},\perp}^\mathsf{T}\hat{\mathbf{e}}_{\text{i},s} & \hat{\mathbf{e}}_{\text{i},\perp}^\mathsf{T}\hat{\mathbf{e}}_{\text{i},p}\\
            \hat{\mathbf{e}}_{\text{i},\parallel}^\mathsf{T}\hat{\mathbf{e}}_{\text{i},s} & \hat{\mathbf{e}}_{\text{i},\parallel}^\mathsf{T}\hat{\mathbf{e}}_{\text{i},p}
        \end{bmatrix}
     \begin{bmatrix}E_{\text{i},s} \\ E_{\text{i},p}\end{bmatrix} \\
     \begin{bmatrix}E_{\text{t},\perp} \\ E_{\text{t},\parallel} \end{bmatrix} &=
        \begin{bmatrix}
            t_{\perp} & 0 \\
            0         & t_{\parallel}
        \end{bmatrix}
        \begin{bmatrix}
            \hat{\mathbf{e}}_{\text{i},\perp}^\mathsf{T}\hat{\mathbf{e}}_{\text{i},s} & \hat{\mathbf{e}}_{\text{i},\perp}^\mathsf{T}\hat{\mathbf{e}}_{\text{i},p}\\
            \hat{\mathbf{e}}_{\text{i},\parallel}^\mathsf{T}\hat{\mathbf{e}}_{\text{i},s} & \hat{\mathbf{e}}_{\text{i},\parallel}^\mathsf{T}\hat{\mathbf{e}}_{\text{i},p}
        \end{bmatrix}
     \begin{bmatrix}E_{\text{i},s} \\ E_{\text{i},p}\end{bmatrix}.
    \end{align}

References:
    .. [atan2] Wikipedia, "`atan2 <https://en.wikipedia.org/wiki/Atan2>`__," accessed 8 Feb. 2023.
    .. [Balanis] A. Balanis, "Advanced Engineering Electromagnetics," John Wiley & Sons, 2012.
    .. [Wiesbeck] N. Geng and W. Wiesbeck, "Planungsmethoden für die Mobilkommunikation," Springer, 1998.
    .. [Wikipedia] Wikipedia, "`Maximum power transfer theorem <https://en.wikipedia.org/wiki/Maximum_power_transfer_theorem>`_," accessed 7 Oct. 2022.
    .. [ITURP20402] ITU, "`Recommendation ITU-R P.2040-2: Effects of building materials and structures on radiowave propagation above about 100 MHz <https://www.itu.int/rec/R-REC-P.2040/en>`_". Sep. 2021.
