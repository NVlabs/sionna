#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" EDFA Test module.

Todo:

"""

# Standard library imports

# Third party imports
import unittest
import numpy as np
import tensorflow as tf

# Local application imports
import sionna
from sionna.channel.optical import edfa


class TestEDFA(unittest.TestCase):
	def setUp(self):
		self._dtype = tf.complex128

	def test_variance(self):
		F = 10 ** (6 / 10)
		G = 2.0
		h = 6.62607015e-34
		f_c = 193.55e12
		dt = 1e-12
		n_sp = F / tf.cast(2.0, self._dtype.real_dtype) * G / (G - tf.cast(
			1.0, self._dtype.real_dtype))
		rho_n_ASE = tf.cast(n_sp * (
				G - tf.cast(1.0, self._dtype.real_dtype)) * h * f_c,
							self._dtype.real_dtype)  # Noise density in (W/Hz)
		P_n_ASE = tf.cast(
			tf.cast(2.0, self._dtype.real_dtype) * rho_n_ASE *
			(tf.cast(1.0, self._dtype.real_dtype) / dt),
			self._dtype.real_dtype)  # Noise power in (W)
		amplifier = edfa.EDFA(G, F, f_c, dt, self._dtype)
		x = tf.zeros((10, 10, 1000), dtype=self._dtype)
		y = amplifier(x)
		sigma_n_ASE_square = np.mean(np.var(y.numpy(), axis=-1))
		self.assertLessEqual(
			np.abs((P_n_ASE - sigma_n_ASE_square) / P_n_ASE), 1e-2,
			'incorrect_variance'
		)

	def test_gain(self):
		F = 10 ** (6 / 10)
		G = 4.0
		f_c = 193.55e12
		dt = 1e-12
		n_sp = F / tf.cast(2.0, self._dtype.real_dtype) * G / (
				G - tf.cast(1.0, self._dtype.real_dtype))
		amplifier = edfa.EDFA(G, n_sp, f_c, dt, self._dtype)
		shape = (10, 10, 10000)
		x = tf.complex(
			tf.cast(1.0 / tf.sqrt(2.0), self._dtype.real_dtype) *
			tf.ones(shape, dtype=self._dtype.real_dtype),
			tf.cast(1.0 / tf.sqrt(2.0), self._dtype.real_dtype) *
			tf.ones(shape, dtype=self._dtype.real_dtype)
		)
		y = amplifier(x)
		mu_n_ASE = np.mean(np.mean(np.abs(y.numpy()) ** 2.0, axis=-1))
		self.assertLessEqual(np.abs(G - mu_n_ASE), 1e-5, 'incorrect_gain')
