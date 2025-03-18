#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""Blocks for interleaving and utility functions"""

import numpy as np
import tensorflow as tf
from importlib_resources import files, as_file
from sionna.phy import config, Block
from sionna.phy.fec.turbo import coeffs

class RowColumnInterleaver(Block):
     # pylint: disable=line-too-long
    r"""Interleaves a sequence of inputs via row/column swapping.

    Parameters
    ----------
    row_depth: int
        The row depth, i.e., how many values per row can be stored.

    axis: int
        The dimension that should be interleaved.

    inverse: `bool`, (default `False`)
        If `True`,  the inverse permutation is performed.

    precision : `None` (default) | 'single' | 'double'
        Precision used for internal calculations and outputs.
        If set to `None`, :py:attr:`~sionna.phy.config.precision` is used.

    Input
    -----
    x: `tf.DType`
        Tensor of arbitrary shape and arbitrary dtype.

    Output
    ------
    : `tf.DType`
        Tensor of same shape and dtype as ``inputs``.

    Note
    ----
    If the sequence length is not a multiple of ``row_depth``, additional
    filler bits are used for the last row that will be removed internally.
    However, for the last positions the interleaving distance may be
    slightly degraded.

    """

    def __init__(self,
                 row_depth,
                 axis=-1,
                 inverse=False,
                 precision=None,
                 **kwargs):

        super().__init__(precision=precision, **kwargs)

        # store perm_seq
        self._perm_seq = None # initialized during build
        self._perm_seq_inv = None # initialized during build

        if not isinstance(axis, int):
            raise TypeError("axis must be int.")
        self._axis = axis

        if not isinstance(row_depth, int):
            raise TypeError("row_depth must be int.")
        self._row_depth = row_depth

        if not isinstance(inverse, bool):
            raise TypeError("inverse must be bool.")
        self._inverse = inverse

        # should not be changed, only required for associated deinterleaver
        self._keep_state = True

    ###############################
    # Public methods and properties
    ###############################

    @property
    def axis(self):
        """Axis to be permuted"""
        return self._axis

    @property
    def row_depth(self):
        """Row depth of the row-column interleaver"""
        return self._row_depth

    @property
    def perm_seq(self):
        """Permutation sequence"""
        return self._perm_seq

    @property
    def perm_seq_inv(self):
        """Inverse permutation sequence"""
        return self._perm_seq_inv

    @property
    def keep_state(self):
        """Row-column interleaver always uses same internal state."""
        return True

    #################
    # Utility methods
    #################

    def _generate_perm_rc(self, n_seq, r_depth):
        """Generates a row/column permutation to initialize an RC-interleaver.

        If required last positions use "filler" positions.

        Args:
            N_seq (int): An integer defining the sequence length to interleave.

            r_depth (int): An integer defining the depth of the interleaver.
        """

        # round to next multiple of r_depth
        n = tf.cast((tf.math.ceil(n_seq/r_depth)*r_depth), tf.int32)
        nb_rows = tf.cast(n/r_depth, tf.int64)

        ind = tf.range(n, dtype=tf.int32)

        # rearange in row/colum format
        ind_rc = tf.reshape(ind, [nb_rows,-1])

        # and interleave via row/column swapping
        ind_cr = tf.transpose(ind_rc, (1,0))

        # read out indices in column/row ordering
        perm_seq_filler= tf.reshape(ind_cr, [-1])

        # remove filler positions
        mask = tf.math.less(perm_seq_filler, n_seq)
        perm_seq = tf.boolean_mask(perm_seq_filler, mask)
        perm_seq_inv= tf.argsort(perm_seq)
        return perm_seq, perm_seq_inv

    ########################
    # Sionna Block functions
    ########################

    def build(self, input_shape):
        """check shapes for consistency"""

        if self._axis >= len(input_shape):
            raise ValueError("Axis does match input shape")

        # Interleaver can't build pattern for dynamic shapes
        if input_shape[self._axis] is None:
            raise ValueError("Permutation axis cannot be None (dynamic).")

        # and generate permutation patterns
        p, pi = self._generate_perm_rc(input_shape[self._axis], self._row_depth)
        self._perm_seq = p
        self._perm_seq_inv = pi

    def call(self, x, /, *, inverse=None, **kwargs):
        """interleaving function

        This function returns the permuted version of inputs.

        Note that the deinterleaver will provide a seed per default, this will
        be ignored by the **kwargs argument.

        Args:
            inputs (tf.float32): Tensor of arbitrary shape. Must have at least
                rank two.

        Returns:
            `tf.float32`: Tensor of same shape as the input.

        """

        input_shape = x.shape

        # re-init if shape has changed, update perm_seq
        if x.shape[self._axis] != self._perm_seq.shape[0]:
            self.build(x.shape)

        # if not explicitly provided use internal value
        if inverse is None:
            inverse = self._inverse

        if inverse:
            x_int = tf.gather(x, self._perm_seq_inv, axis=self._axis)
        else:
            x_int = tf.gather(x, self._perm_seq, axis=self._axis)

        return tf.ensure_shape(x_int, input_shape)


class RandomInterleaver(Block):
    # pylint: disable=line-too-long
    r"""Random interleaver permuting a sequence of input symbols.

    Parameters
    ----------
    seed: int
        Integer defining the random seed used if option ``keep_state`` is
        True.

    keep_batch_constant: `bool`, (default `True`)
        If set to True each sample in the batch uses the same permutation.
        Otherwise, unique permutations per batch sample are generate (slower).

    inverse: `bool`, (default `False`)
        If `True`,  the inverse permutation is performed.

    keep_state: `bool`, (default `True`)
        If `True`,  the permutation is fixed for multiple calls (defined by
        ``seed`` attribute).

    axis: int, (default -1)
        The dimension that should be interleaved.
        First dimension (`axis=0`) is not allowed.

    precision : `None` (default) | 'single' | 'double'
        Precision used for internal calculations and outputs.
        If set to `None`, :py:attr:`~sionna.phy.config.precision` is used.

    Input
    -----
    x: `tf.DType`
        Tensor of arbitrary shape and dtype.

    seed: `int`
        An integer defining the state of the random number
        generator. If explicitly given, the global internal seed is
        replaced by this seed. Can be used to realize random
        interleaver/deinterleaver pairs (call with same random seed).

    Output
    ------
    : tf.DType
        Tensor of same shape and dtype as the input ``x``.

    Note
    ----
    The interleaver block is stateless, i.e., the seed is either random
    during each call or must be explicitly provided during init/call.
    This simplifies XLA/graph execution.

    This is NOT the 5G interleaver sequence.
    """

    def __init__(self,
                seed=None,
                keep_batch_constant=True,
                inverse=False,
                keep_state=True,
                axis=-1,
                precision=None,
                **kwargs):

        super().__init__(precision=precision, **kwargs)

        # verify and store attributes
        if not isinstance(keep_batch_constant, bool):
            raise TypeError("keep_batch_constant must be bool.")
        self._keep_batch_constant = keep_batch_constant

        if not isinstance(axis, int):
            raise TypeError("axis must be int.")
        self._axis=axis

        # a global seed is stored and used if called with keep_state=True
        if seed is not None:
            if not isinstance(seed, int):
                raise TypeError("seed must be int.")
        else:
            # generate random seed if no value is provided
            seed = int(np.random.uniform(0, 2**31-1))

        # if keep_state==True this seed is used to generate scrambling sequences
        self._seed = (1337, seed)

        if not isinstance(inverse, bool):
            raise TypeError("inverse must be boolean")
        self._inverse = inverse

        if not isinstance(keep_state, bool):
            raise TypeError("keep_state must be boolean")
        self._keep_state = keep_state

        if self._keep_state is False and self._inverse is True:
            print("Note: keep_state=False and, thus, a new realization of " \
                  "the interleaver is generated during each call. Thus, " \
                  "the inverse interleaver does not correspond to a previous " \
                  "interleaver call.")

    ###############################
    # Public methods and properties
    ###############################

    @property
    def seed(self):
        """Seed to generate random sequence"""
        return self._seed[1] # only return the non-fixed seed

    @property
    def axis(self):
        """Axis to be permuted"""
        return self._axis

    @property
    def keep_state(self):
        """Generate new random seed per call"""
        return self._keep_state

    def find_s_min(self, seed, seq_length, s_min_stop=0):
        r"""Find :math:`S` parameter such that :math:`\pi(i)-\pi(j)>S` for all
        :math:`i-j<S`. This can be used to find optimized interleaver patterns.

        ``s_min_stop`` is an additional stopping condition, i.e., stop if
        current :math:`S` is already smaller than ``s_min_stop``.

        Please note that this is a Numpy utility function and usually not part
        of the graph.

        Input
        -----
        seed: int
            seed to draw random permutation that shall be analyzed.

        seq_length: int
            length of permutation sequence to be analyzed.

        s_min_stop: int, (default 0)
            Enables early stop if already current s_min< ``s_min_stop`` .

        Output
        ------
        : float
            The S-parameter for the given ``seed``.
        """

        if not isinstance(seed, int):
            raise TypeError("seed must be int.")
        if not isinstance(seq_length, int):
            raise TypeError("seq_length must be int.")
        if not isinstance(s_min_stop, int):
            raise TypeError("s_min_stop must be int.")

        seed = (1337, seed)
        perm_seq = self._generate_perm_full(seed, seq_length, batch_size=1)
        perm_seq = tf.squeeze(perm_seq, axis=0).numpy()
        s_min = seq_length
        for i in range(len(perm_seq)): # search for all positions in perm_seq
            for j in range(-s_min,s_min,1): # search dist
                if j==0: # ignore identity
                    continue
                if i+j>=0 and i+j<seq_length:
                    d = np.abs(perm_seq[i] - perm_seq[i+j])
                    if d<=np.abs(j):
                        s_min = np.min([s_min, np.abs(j)])
                    if d<s_min and np.abs(j)<s_min:
                        s_min = np.min([s_min, d])
            # early stop
            if s_min<=s_min_stop:
                break
        return int(s_min)

    #################
    # Utility methods
    #################

    def _generate_perm_full(self, seed, seq_length, batch_size, inverse=False):
        """Generates a random permutation for the interleaver.

        Args:
            seed (int): A shape [2] Tensor, the seed to the random number
                generator.

            seq_length (int): The length of the sequence to be permuted.

            batch_size (int): The batch size (=number of independent
                permutations).

            inverse (bool): Defaults to False. If `True`,  the inverse permutation
                for the given seed is generated.
        """
        rand_seq = tf.random.stateless_uniform([batch_size, seq_length],
                                                seed,
                                                minval=0,
                                                maxval=1,
                                                dtype=tf.float32)

        perm_seq =  tf.argsort(rand_seq, axis=-1)

        if inverse:
            # cast to tf.float32 due to improved performance
            perm_seq = tf.cast(perm_seq, tf.float32)
            perm_seq = tf.argsort(perm_seq, axis=-1)

        return perm_seq

    #########################
    # Sionna Block functions
    #########################

    # pylint: disable=(unused-argument)
    def build(self, input_shape, **kwargs):
        """Build block and check consistency of dimensions."""

        if self._axis >= len(input_shape):
            raise ValueError("Axis does not match input shape.")

    def call(self, x, /, *, seed=None, inverse=None):
        """Interleaving function.

        This function returns the permuted version of ``inputs``.

        Args:
            inputs (List): ``[x, seed]``, where
            ``x`` (tf.float32): Tensor of arbitrary shape. Must have at
                least rank two.
            ``seed`` (int): An integer defining the state of the random number
                generator. If explicitly given, the global internal seed is
                replaced by this seed. Can be used the realize random
                interleaver/deinterleaver pairs (call with same random seed).


        Returns:
            `tf.float`: Tensor of same shape as the input.

        Raises:
            InvalidArgumentError
                When rank(``x``)<2.

            AssertionError
                If ``seed`` is not None or int.

        Note:
            In case of inverse interleaving (e.g., at the receiver),
            ``keep_state`` should be `True` as otherwise a new permutation is
            generated and the output is not equal to the original sequence.
            Alternatively, an explicit seed must be provided as function
            argument.
        """

        input_shape = x.shape

        if inverse is None:
            inverse = self._inverse
        else:
            if not isinstance(inverse, bool):
                raise TypeError("inverse must be bool")

        # use seed if explicit seed is provided
        if seed is not None:
            seed = (tf.constant(1337), tf.cast(seed, tf.int32))
        # only generate a new random sequence if keep_state==False
        elif self._keep_state:
            # use sequence as defined by seed
            seed = self._seed
        else:
            if inverse:
                raise ValueError(
                    "Inverse interleaving not possible for " \
                    "random seeds per call (keep_state=False) without " \
                    "explicitly providing the seed as inputs.")
            # generate new seed for each call
            # Note: not necessarily random if XLA is active
            seed = config.tf_rng.uniform([2],
                                         minval=0,
                                         maxval=2**31-1,
                                         dtype=tf.int32)
        # select if each sample in batch needs own perm (computational complex!)
        if self._keep_batch_constant:
            batch_size = 1
        else:
            # special case: no batch dim
            if len(tf.shape(x))==1:
                batch_size = 1
            else:
                batch_size = tf.shape(x)[0]

        perm_seq = self._generate_perm_full(seed,
                                            tf.shape(x)[self._axis],
                                            batch_size,
                                            inverse)

        if self._keep_batch_constant:
            # broadcast single sequence over complete batch
            perm_seq = tf.squeeze(perm_seq, axis=0) # remove batch_dim
            x = tf.gather(x, perm_seq, batch_dims=0, axis=self._axis)
        elif len(tf.shape(x))==1: # special case: no batch dim
            perm_seq = tf.squeeze(perm_seq, axis=0) # remove batch_dim
            x = tf.gather(x, perm_seq, batch_dims=0, axis=self._axis)
        else:
            x = tf.gather(x, perm_seq, batch_dims=1, axis=self._axis)

        return tf.ensure_shape(x, input_shape)

class Deinterleaver(Block):
    """Deinterleaver that reverts the interleaver for a given input sequence.

    Parameters
    ----------
    interleaver: Interleaver
        Associated Interleaver which shall be deinterleaved by this block.
        Can be either
        :class:`~sionna.phy.fec.interleaving.RandomInterleaver` or
        :class:`~sionna.phy.fec.interleaving.RowColumnInterleaver`.

    precision : `None` (default) | 'single' | 'double'
        Precision used for internal calculations and outputs.
        If set to `None`, :py:attr:`~sionna.phy.config.precision` is used.

    Input
    -----
    x: tf.DType
        2+D tensor of arbitrary shape.

    seed: int
        An integer defining the state of the random number
        generator. If explicitly given, the global internal seed is
        replaced by this seed. Can be used to realize random
        interleaver/deinterleaver pairs (call with same random seed).

    Output
    ------
    : tf.DType
        2+D tensor of same shape and dtype as the input ``x``.

    Note
    ----
    This block provides a wrapper of the inverse interleaver function.
    """

    def __init__(self,
                 interleaver,
                 precision=None,
                 **kwargs):

        if not isinstance(interleaver,
                          (RandomInterleaver,
                          RowColumnInterleaver,
                          Turbo3GPPInterleaver)):
            raise ValueError("interleaver is not a valid interleaver instance.")
        self._interleaver = interleaver

        # if dtype is None, use same dtype as associated interleaver
        if precision is None:
            precision = self._interleaver.precision
        super().__init__(precision=precision, **kwargs)

        if self._interleaver._keep_state is False:
            print("Warning: deinterleaver requires interleaver to have " \
            "keep_state=True or to explicitly provide the seed as inputs.")

    #########################################
    # Public methods and properties
    #########################################

    @property
    def interleaver(self):
        """Associated interleaver instance"""
        return self._interleaver

    ########################
    # Sionna block functions
    ########################

    def build(self, input_shape):
        """build block"""
        pass

    def call(self, x, seed=None):
        """deinterleaving function.

        This function returns the permuted version of inputs.

        Args:
        x (tf.float32): Tensor of arbitrary shape. Must have at least
            rank two.

        seed (int): An integer defining the state of the random number
        generator. If explicitly given, the global internal seed is
        replaced by this seed. Can be used the realize random
        interleaver/deinterleaver pairs (call with same random seed).

        Returns:
            `tf.float32`: Tensor of same shape as the input.
        """
        input_dt = x.dtype
        x = self._interleaver(x, seed=seed, inverse=True)

        # cast to original dtype to avoid different dtypes
        # due to call of interleaver block which casts to its internal precision
        return tf.cast(x, input_dt)

class Turbo3GPPInterleaver(Block):
    # pylint: disable=line-too-long
    """Interleaver for 3GPP Turbo codes

    Interleaver as used in the 3GPP Turbo codes [3GPPTS36212_I]_ and, thus,
    the maximum length is given as 6144 elements (only for the dimension as
    specific by ``axis``).

    Parameters
    ----------
    inverse: `bool`, (default `False`)
        If `True`,  the inverse permutation is performed.

    axis: int, (default -1)
        The dimension that should be interleaved.
        First dimension (`axis=0`) is not allowed.

    precision : `None` (default) | 'single' | 'double'
        Precision used for internal calculations and outputs.
        If set to `None`, :py:attr:`~sionna.phy.config.precision` is used.

    Input
    -----
    x: tf.DType
        2+D tensor of arbitrary shape and dtype.

    Output
    ------
    : tf.DType
        2+D tensor of same shape and dtype as the input ``x``.

    Note
    ----
    Note that this implementation slightly deviates from the 3GPP
    standard [3GPPTS36212_I]_ in a sense that zero-padding is introduced
    for cases when the exact interleaver length is not supported by the
    standard.
    """

    def __init__(self,
                 inverse=False,
                 axis=-1,
                 precision=None,
                 **kwargs):

        super().__init__(precision=precision, **kwargs)

        if not isinstance(axis, int):
            raise TypeError("axis must be int.")

        self._axis=axis
        self._keep_state = True # only required for deinterleaver
        self.frame_size = None

        if not isinstance(inverse, bool):
            raise TypeError("inverse must be boolean")
        self._inverse = inverse

        # load interleaver patterns as defined in the 3GPP standard
        self.coeffs_dict = {}
        source = files(coeffs).joinpath("turbo_coeffs.csv")
        with as_file(source) as coeffs.csv:
            csv_reader = np.genfromtxt(coeffs.csv, delimiter=",")

            for (line_count, row) in enumerate(csv_reader):
                if line_count >0: #igonore first line (=header)
                    self.coeffs_dict[int(row[1])] = (int(row[2]), int(row[3]))

    ###############################
    # Public methods and properties
    ###############################

    @property
    def axis(self):
        """Axis to be permuted"""
        return self._axis

    #################
    # Utility methods
    #################
    def _generate_perm_full(self, frame_size, inverse=False):
        """Generates a random permutation for the interleaver.
        Args:

        frame_size (int): The length of the sequence to be permuted.

        batch_size (int): The batch size (=number of independent
            permutations).

        inverse (bool): Defaults to False. If `True`,  the inverse
        permutation for the given seed is generated.
        """
        k = frame_size
        if k not in self.coeffs_dict:
            geqk_sizes = sorted([x for x in self.coeffs_dict if x >= k])
            if len(geqk_sizes)==0:
                print("Input frame size too large for 3GPP Turbo Interleaver.")
            else:
                k = geqk_sizes[0]
        f1, f2 = self.coeffs_dict[k]
        perm_seq = [(f1 * i + f2* (i**2))%k for i in range(k)]

        if frame_size < k:
            perm_seq = [x for x in perm_seq if x < frame_size]

        perm_seq = tf.convert_to_tensor(perm_seq)
        if inverse:
            # cast to tf.float32 due to improved sorting performance
            perm_seq = tf.cast(perm_seq, tf.float32)
            perm_seq = tf.argsort(perm_seq, axis=-1)

        return perm_seq

    #########################
    # Sionna block functions
    #########################

    def build(self, input_shape):
        """Build Block and check consistency of dimensions."""

        if not self.axis < len(input_shape):
            raise ValueError("Axis does not match input shape.")

        frame_size = input_shape[self._axis]
        if frame_size >= 6145:
            msg = "3GPP Turbo Interleaver is defined for block lengths up "\
                  "to 6144."
            raise ValueError(msg)

    # **kwargs to avoid issues if seed is provided during deinterleaving
    # pylint: disable=unused-argument
    def call(self, x, /, *, inverse=None, **kwargs):
        """Interleaving function.

        This function returns the permuted version of ``inputs``.
        """

        input_shape = x.shape
        frame_size = input_shape[self._axis]

        if inverse is None:
            inverse = self._inverse

        perm_seq = self._generate_perm_full(frame_size, inverse)
        x = tf.gather(x, perm_seq, axis=self._axis)

        # set explicitly for xla mode
        return tf.ensure_shape(x, input_shape)
