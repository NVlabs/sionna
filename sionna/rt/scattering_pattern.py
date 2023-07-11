#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Classes and functions related to Scattering patterns for the Sionna RT Module
"""

import tensorflow as tf
from scipy.special import binom
from abc import ABC
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from .utils import theta_phi_from_unit_vec, r_hat, dot
from sionna.constants import PI

class ScatteringPattern(ABC):
    # pylint: disable=line-too-long
    r"""
    Abstract class defining a scattering pattern for diffuse reflections

    This class implements a mix of the Backscattering, Directive, and
    Lambertian scattering models as described in [Degli-Esposti07]_.

    Parameters
    ----------
    alpha_r : int, [0,1,2,...]
        Parameter related to the width of the scattering lobe in the
        direction of the specular reflection.
        A value of 0 indicates Lambertian scattering.

    alpha_i : int, [1,2,...]
        Parameter related to the width of the scattering lobe in the
        incoming direction. Only plays a role if ``alpha_r``>0.

    lambda_ : float, [0,1]
        Parameter determining the percentage of the diffusely
        reflected energy in the lobe around the specular reflection.

    trainable_lambda_ : bool
        If set to `True`, the parameter `lambda_` is made a trainable variable.
        Defaults to `False`.
    
    dtype : tf.complex64 or tf.complex128
        Datatype used for all computations.
        Defaults to `tf.complex64`.

    Input
    -----
    k_i : [batch_size, 3], dtype.real_dtype
        Tensor of incomning directions

    k_s : [batch_size,3], dtype.real_dtype
        Tensor of outgoing directions

    n_hat : [batch_size, 3], dtype.real_dtype
        Tensor of surface normals

    Output
    ------
    pattern : [batch_size], dtype.real_dtype
        Scattering pattern 
    """
    def __init__(self,
                 alpha_r,
                 alpha_i,
                 lambda_,
                 trainable_lambda_=False,
                 dtype=tf.complex64):
        if dtype not in (tf.complex64, tf.complex128):
            msg = "`dtype` must be `tf.complex64` or `tf.complex128`"
            raise ValueError(msg)
        self._dtype = dtype
        self._rdtype = dtype.real_dtype
        self.alpha_r = alpha_r
        self.alpha_i = alpha_i
        self._lambda_ = tf.Variable(tf.zeros((), self._rdtype))
        self.lambda_ = lambda_
        self.trainable_lambda_ = trainable_lambda_

    ###
    ### Static properties and methods
    ###
    _alpha_max = -1 # Maximum value of alpha_r, alpha_i
    _params_32 = None # Dictionary of precomputed tensors in 32bit precision
    _params_64 = None # Dictionary of precomputed tensors in 64bit precision

    @staticmethod
    def _update_params(alpha_max):
        """Precomputes several tensors needed for the __call__"""
        alpha_max = max(alpha_max, 1)
        ScatteringPattern._alpha_max = alpha_max
        j_even = tf.range(0, alpha_max+1, delta=2, dtype=tf.float64)
        j_odd = tf.range(1, alpha_max+1, delta=2, dtype=tf.float64)
        i_j_even = 2*PI/(j_even+1)
        n_max = (j_odd[-1]-1)/2
        w_range = tf.range(0, n_max+1, dtype=tf.float64)
        w_2 = 2*w_range
        binom_2 = tf.constant(binom(w_2, w_range), tf.float64)
        binom_1 = np.zeros([alpha_max+1, alpha_max+1])
        for alpha in range(alpha_max+1):
            binom_1[alpha, :alpha+1] = binom(alpha, np.arange(alpha+1))/2**alpha
        binom_1_even = tf.cast(binom_1[:,::2], tf.float64)
        binom_1_odd = tf.cast(binom_1[:,1::2], tf.float64)
        f_summands_even = tf.reduce_sum(binom_1_even*i_j_even, axis=-1)
        ScatteringPattern._params_64 = {
                        "binom_1_odd" : binom_1_odd,
                        "binom_2" : binom_2,
                        "f_summands_even" : f_summands_even,
                        "j_odd" : j_odd,
                        "n_max" : int(n_max),
                        "w_2" : w_2
                      }
        ScatteringPattern._params_32 = {
                        "binom_1_odd" : tf.cast(binom_1_odd, tf.float32),
                        "binom_2" : tf.cast(binom_2, tf.float32),
                        "f_summands_even" : tf.cast(f_summands_even, tf.float32),
                        "j_odd" : tf.cast(j_odd, tf.float32),
                        "n_max" : int(n_max),
                        "w_2" : tf.cast(w_2, tf.float32)
                      }

    @staticmethod
    def f_alpha(cos_theta_i, alphas):
        """Compute the normalization factor F_{alpha_R}"""
        cos_theta_i = tf.expand_dims(cos_theta_i, -1)
        sin_theta_i = tf.sqrt(1-cos_theta_i**2)
        dtype = cos_theta_i.dtype
        if  dtype==tf.float32:
            params = ScatteringPattern._params_32
        else:
            params = ScatteringPattern._params_64

        # Compute K_n
        series = params["binom_2"]*(sin_theta_i/2)**(params["w_2"])
        n = tf.constant(0, tf.int32)
        n_max = tf.cast(params["n_max"], tf.int32)
        k_n = tf.TensorArray(dtype=dtype, size=n_max+1)

        def condition(n, n_max, k_n, series): # pylint: disable=unused-argument
            return n <= n_max

        # Iteratively compute the sums of the first n elements of series
        def body(n, n_max, k_n, series):
            k_n = k_n.write(n, tf.reduce_sum(series[...,:n+1], axis=-1))
            return n+1, n_max, k_n, series

        _, _, k_n, _ = tf.while_loop(condition, body,
                                     loop_vars=[n, n_max, k_n, series],
                                     parallel_iterations=params["n_max"]+1)
        k_n = k_n.stack()
        k_n = tf.transpose(k_n, perm=[1, 0])

        # Compute I_j for odd values of j
        i_j_odd = 2*PI/(params["j_odd"]+1)*cos_theta_i*k_n

        # Compute the sum over odd values of j
        f_summands_odd = tf.gather(params["binom_1_odd"], alphas)*i_j_odd
        f_odd = tf.reduce_sum(f_summands_odd, -1)

        # Get the precomputed sum over even values of j
        f_even = tf.gather(params["f_summands_even"], alphas)

        # Compute the final normalization factor F_alpha
        f = f_odd+f_even

        return f

    @staticmethod
    def pattern(k_i, k_s, n_hat, alpha_r, alpha_i, lambda_):
        """Compute the scattering pattern

        The function always computes the BackscatteringPattern as well
        as the LambertianPattern and then selects which one is applied,
        depending on the values of alpha_r (alpha_r=0 means Lambertian).
        The DirectivePattern is a special case for lambda_=0.

        This design choice has been made to allow the computation
        of a different scattering pattern for each pair k_i, k_s.
        """
        dtype = k_i.dtype
        cos_theta_i = -dot(k_i, n_hat)
        cos_theta_s = dot(k_s, n_hat)
        k_r = k_i + 2*tf.expand_dims(cos_theta_i, -1)*n_hat

        # Compute backscattering pattern
        f_alpha_r = ScatteringPattern.f_alpha(cos_theta_i, alpha_r)
        f_alpha_i = ScatteringPattern.f_alpha(cos_theta_i, alpha_i)
        f = lambda_*f_alpha_r + (1-lambda_)*f_alpha_i
        pattern_bs = lambda_*tf.pow((1+dot(k_r, k_s))/2,
                                    tf.cast(alpha_r, k_r.dtype))
        pattern_bs += (1-lambda_)*tf.pow((1-dot(k_i, k_s))/2,
                                         tf.cast(alpha_i, k_r.dtype))
        pattern_bs /= f

        # Compute Lambertian pattern
        pattern_l = cos_theta_s/tf.cast(PI, k_i.dtype)
        pattern_l = tf.where(pattern_l<0., tf.cast(0, dtype), pattern_l)

        # Select one of the two models depending on alpha_r
        pattern = tf.where(alpha_r==0, pattern_l, pattern_bs)

        return pattern

    ###
    ### Instance properties and methods
    ###
    @property
    def alpha_r(self):
        """
        bool : Get/set ``alpha_r``
        """
        return self._alpha_r

    @alpha_r.setter
    def alpha_r(self, value):
        if not isinstance(value, (int, np.integer)):
            raise TypeError("alpha_r must be a positive integer")

        if isinstance(self, LambertianPattern):
            value = 0
        else:
            if value < 1:
                raise ValueError("alpha must be >=1")
        if value > ScatteringPattern._alpha_max:
            ScatteringPattern._update_params(value)
        self._alpha_r = value

    @property
    def alpha_i(self):
        """
        bool : Get/set ``alpha_i``
        """
        return self._alpha_i

    @alpha_i.setter
    def alpha_i(self, value):
        if not isinstance(value, (int, np.integer)):
            raise TypeError("alpha_i must be a positive integer")
        if value < 1:
            raise ValueError("alpha msu be >=1")
        if value > ScatteringPattern._alpha_max:
            ScatteringPattern._update_params(value)
        self._alpha_i = value

    @property
    def lambda_(self):
        """
        bool : Get/set ``lambda_``
        """
        return self._lambda_

    @lambda_.setter
    def lambda_(self, value):
        value = tf.cast(value, self._rdtype)
        if value<0 or value >1:
            raise ValueError("lambda_ must be in [0,1]")
        self._lambda_.assign(value)

    @property
    def trainable_lambda_(self):
        """
        bool : Get/set if ``lambda_`` is trainable or not.
            Defaults to `False`.
        """
        return self._trainable_lambda_

    @trainable_lambda_.setter
    def trainable_lambda_(self, value):
        if not isinstance(value, bool):
            raise TypeError("`trainable_lambda_` must be bool")
        # pylint: disable=protected-access
        self._lambda_._trainable = value
        self._trainable_lambda_ = value

    def __call__(self, k_i, k_s, n_hat):
        return ScatteringPattern.pattern(k_i, k_s, n_hat, self.alpha_r,
                                         self.alpha_i, self._lambda_)

    def visualize(self, k_i=(0.7071, 0., -0.7071), show_directions=False):
        r"""
        Visualizes the scattering pattern

        It is assumed that the surface normal points toward the
        positive z-axis.

        Input
        -----
        k_i : [3], array_like
            Incoming direction

        show_directions : bool
            If `True`, the incoming and specular reflection directions
            are shown.
            Defaults to `False`.

        Output
        ------
        : :class:`matplotlib.pyplot.Figure`
            3D visualization of the scattering pattern

        : :class:`matplotlib.pyplot.Figure`
            Visualization of the incident plane cut through
            the scattering pattern
        """
        k_i_in = k_i

        ###
        ### 3D visualization
        ###
        theta = np.linspace(0.0, PI/2, 50)
        phi = np.linspace(-PI, PI, 100)
        theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
        k_s = tf.cast(r_hat(theta_grid, phi_grid), self._dtype.real_dtype)
        k_i = tf.cast(tf.broadcast_to(k_i_in, k_s.shape), self._dtype.real_dtype)
        k_s = tf.reshape(k_s, [-1, 3])
        k_i = tf.reshape(k_i, [-1, 3])
        n_hat = tf.constant([[0,0,1]], dtype=k_i.dtype)
        n_hat = tf.repeat(n_hat, k_i.shape[0], 0)
        pattern = self(k_i, k_s, n_hat)
        pattern = tf.reshape(pattern, [50, 100])
        x = pattern * np.sin(theta_grid) * np.cos(phi_grid)
        y = pattern * np.sin(theta_grid) * np.sin(phi_grid)
        z = pattern * np.cos(theta_grid)
        p_min = np.min(pattern)
        p_max = np.max(pattern)

        def norm(x):
            """Maps input to [0,1] range"""
            x_min = np.min(x)
            x_max = np.max(x)
            if x_min==x_max:
                x = np.ones_like(x)
            else:
                x -= x_min
                x /= np.abs(x_max-x_min)
            return x

        fig_3d = plt.figure()
        ax = fig_3d.add_subplot(1,1,1, projection='3d', computed_zorder=False)

        ax.plot_surface(x, y, z, rstride=1, cstride=1, linewidth=0,
                            antialiased=False, alpha=0.7,
                            facecolors=cm.turbo(norm(pattern)))

        ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))

        if show_directions:
            r = np.max(pattern)*1.1
            k_i = k_i[0,0]
            uvw = -k_i.numpy()
            plt.quiver(*r*uvw, *-uvw, length=r, linestyle='dashed', color="black", arrow_length_ratio=0.07, alpha=0.5)
            ax.text(r*uvw[0], r*uvw[1], r*uvw[2], r"$\hat{\mathbf{k}}_\mathrm{i}$")

            theta_i, phi_i = theta_phi_from_unit_vec(-k_i)
            theta_r, phi_r = theta_i, phi_i + tf.cast(PI, phi_i.dtype)
            k_r = r_hat(theta_r, phi_r).numpy()
            plt.quiver(*[0,0,0], *k_r, length=r, linestyle='dashed', color="black", arrow_length_ratio=0.07, alpha=0.5)
            ax.text(r*k_r[0], r*k_r[1], r*k_r[2], r"$\hat{\mathbf{k}}_\mathrm{r}$")

        sm = cm.ScalarMappable(cmap=plt.cm.turbo)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation="vertical", location="right",
                                shrink=0.7, pad=0.15)

        xticks = cbar.ax.get_yticks()
        xticklabels = cbar.ax.get_yticklabels()
        xticklabels = p_min + xticks*(p_max-p_min)
        xticklabels = [f"{z:.2f}" for z in xticklabels]
        cbar.ax.set_yticks(xticks)
        cbar.ax.set_yticklabels(xticklabels)
        ax.view_init(elev=30., azim=-60)
        plt.xlabel("x")
        plt.ylabel("y")
        ax.set_zlabel("z")
        ax.set_aspect('auto')
        plt.suptitle(r"3D visualization of the scattering pattern $f_\mathrm{s}(\hat{\mathbf{k}}_\mathrm{i}, \hat{\mathbf{k}}_\mathrm{s})$")

        ###
        ### Incident plane cut through the scattering pattern
        ###
        k_i = tf.constant(k_i_in, self._dtype.real_dtype)
        theta_i, phi_i = theta_phi_from_unit_vec(-k_i)
        theta_r, phi_r = theta_i, phi_i + tf.cast(PI, phi_i.dtype)

        # Pattern around reflected direction
        theta_s = tf.cast(tf.linspace(0.0, PI/2, 100), dtype=self._dtype.real_dtype)
        phi_s = tf.broadcast_to(phi_r, theta_s.shape)
        k_s = r_hat(theta_s, phi_s)
        k_i_1 = tf.broadcast_to(k_i, k_s.shape)
        n_hat = tf.constant([[0,0,1]], dtype=k_i.dtype)
        n_hat = tf.broadcast_to(n_hat, k_i_1.shape)
        pattern = self(k_i_1, k_s, n_hat)

        # Pattern around incident direction
        k_s = r_hat(theta_s, phi_s+PI)
        pattern2 = self(k_i_1, k_s, n_hat)

        fig_cut = plt.figure()
        plt.polar(theta_s, pattern, color='C0')
        plt.polar(2*PI-theta_s , pattern2, color='C0')

        ax = fig_cut.axes[0]
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_thetamin(-90)
        ax.set_thetamax(90)

        if show_directions:
            xticks = list(ax.get_xticks())
            if not theta_i.numpy() in xticks:
                xticks += [theta_i.numpy()]
            if not -theta_i.numpy() in xticks:
                xticks += [-theta_i.numpy()]
            ax.set_xticks(xticks)
            ax.text(-theta_i-10*PI/180, ax.get_yticks()[-1]*2/3, r"$\hat{\mathbf{k}}_\mathrm{i}$", horizontalalignment='center')
            ax.text(theta_i+10*PI/180, ax.get_yticks()[-1]*2/3, r"$\hat{\mathbf{k}}_\mathrm{r}$", horizontalalignment='center')
            plt.quiver([0], [0], [np.sin(theta_i)], [np.cos(theta_i)], scale=1., color="grey",)
            plt.quiver([0], [0], [-np.sin(theta_i)], [np.cos(theta_i)], scale=1., color="grey",)

        plt.title(r"Incident plane cut through the scattering pattern $f_\mathrm{s}(\hat{\mathbf{k}}_\mathrm{i}, \hat{\mathbf{k}}_\mathrm{s})$ ($\phi_\mathrm{s}=\phi_\mathrm{i}+\pi$)")
        plt.tight_layout()

        return fig_3d, fig_cut

class LambertianPattern(ScatteringPattern):
    # pylint: disable=line-too-long
    r"""
    Lambertian scattering model from [Degli-Esposti07]_ as given in :eq:`lambertian_model`

    Parameters
    ----------
    dtype : tf.complex64 or tf.complex128
        Datatype used for all computations.
        Defaults to `tf.complex64`.

    Input
    -----
    k_i : [batch_size, 3], dtype.real_dtype
        Incoming directions

    k_s : [batch_size,3], dtype.real_dtype
        Outgoing directions

    Output
    ------
    pattern : [batch_size], dtype.real_dtype
        Scattering pattern

    Example
    -------
    >>> LambertianPattern().visualize()

    .. figure:: ../figures/lambertian_pattern_3d.png
        :align: center

    .. figure:: ../figures/lambertian_pattern_cut.png
        :align: center
    """
    def __init__(self, dtype=tf.complex64):
        super().__init__(alpha_r=0, alpha_i=1, lambda_=1, dtype=dtype)

class DirectivePattern(ScatteringPattern):
    # pylint: disable=line-too-long
    r"""
    Directive scattering model from [Degli-Esposti07]_ as given in :eq:`directive_model`

    Parameters
    ----------
    alpha_r : int, [1,2,...]
        Parameter related to the width of the scattering lobe in the
        direction of the specular reflection.

    dtype : tf.complex64 or tf.complex128
        Datatype used for all computations.
        Defaults to `tf.complex64`.

    Input
    -----
    k_i : [batch_size, 3], dtype.real_dtype
        Incoming directions

    k_s : [batch_size,3], dtype.real_dtype
        Outgoing directions

    Output
    ------
    pattern : [batch_size], dtype.real_dtype
        Scattering pattern

    Example
    -------
    >>> DirectivePattern(alpha_r=10).visualize()

    .. figure:: ../figures/directive_pattern_3d.png
        :align: center

    .. figure:: ../figures/directive_pattern_cut.png
        :align: center
    """
    def __init__(self,
                 alpha_r,
                 dtype=tf.complex64):
        super().__init__(alpha_r=alpha_r, alpha_i=1, lambda_=1, dtype=dtype)

class BackscatteringPattern(ScatteringPattern):
    # pylint: disable=line-too-long
    r"""
    Backscattering model from [Degli-Esposti07]_ as given in :eq:`backscattering_model`

    Parameters
    ----------
    alpha_r : int, [1,2,...]
        Parameter related to the width of the scattering lobe in the
        direction of the specular reflection.

    alpha_i : int, [1,2,...]
        Parameter related to the width of the scattering lobe in the
        incoming direction.

    lambda_ : float, [0,1]
        Parameter determining the percentage of the diffusely
        reflected energy in the lobe around the specular reflection.

    trainable_lambda_ : bool
        If set to `True`, the parameter `lambda_` is made a trainable variable.
        Defaults to `False`.

    dtype : tf.complex64 or tf.complex128
        Datatype used for all computations.
        Defaults to `tf.complex64`.

    Input
    -----
    k_i : [batch_size, 3], dtype.real_dtype
        Incoming directions

    k_s : [batch_size,3], dtype.real_dtype
        Outgoing directions

    Output
    ------
    pattern : [batch_size], dtype.real_dtype
        Scattering pattern

    Example
    -------
    >>> BackscatteringPattern(alpha_r=20, alpha_i=30, lambda_=0.7).visualize()

    .. figure:: ../figures/backscattering_pattern_3d.png
        :align: center

    .. figure:: ../figures/backscattering_pattern_cut.png
        :align: center
    """
    def __init__(self,
                 alpha_r,
                 alpha_i,
                 lambda_,
                 trainable_lambda_=False,
                 dtype=tf.complex64):
        super().__init__(alpha_r=alpha_r, alpha_i=alpha_i, lambda_=lambda_,
                         trainable_lambda_=trainable_lambda_, dtype=dtype)
