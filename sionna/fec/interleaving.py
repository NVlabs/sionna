#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Layers for interleaving and utility functions"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from importlib_resources import files, as_file
from sionna.fec.turbo import coeffs

class RowColumnInterleaver(Layer):
     # pylint: disable=line-too-long
    r"""RowColumnInterleaver(row_depth, axis=-1, inverse=False, dtype=tf.float32, **kwargs)

    Interleaves a sequence of inputs via row/column swapping.

    The class inherits from the Keras layer class and can be used as layer in a
    Keras model.

    Parameters
    ----------
        row_depth: int
            The row depth, i.e., how many values per row can be stored.

        axis: int
            The dimension that should be interleaved. First dimension
            (`axis=0`) is not allowed.

        inverse: bool
            A boolean defaults to False. If True, the inverse permutation is
            performed.

        dtype: tf.DType
            Defaults to `tf.float32`. Defines the datatype for internal
            calculations and the output dtype.

    Input
    -----
        inputs: tf.DType
            2+D tensor of arbitrary shape and arbitrary dtype. Must have at
            least rank two.

    Output
    ------
         : tf.DType
            2+D tensor of same shape and dtype as ``inputs``.

    Raises
    ------
         AssertionError
            If ``axis`` is not an integer.

         AssertionError
            If ``row_depth`` is not an integer.

         AssertionError
            If ``axis`` > number of input dimensions.

    Note
    ----
        If the sequence length is not a multiple of ``row_depth``, additional
        filler bits are used for the last row that will be removed internally.
        However, for the last positions the interleaving distance may be
        slightly degraded.

        To permute the batch dimension, expand_dims at `axis=0`, interleave and
        remove new dimension.
    """

    def __init__(self,
                 row_depth,
                 axis=-1,
                 inverse=False,
                 dtype=tf.float32,
                 **kwargs):

        super().__init__(dtype=dtype, **kwargs)

        # store perm_seq
        self._perm_seq = None # initalized during build
        self._perm_seq_inv = None # initalized during build

        assert isinstance(axis, int), "axis must be int."
        self._axis = axis

        assert isinstance(row_depth, int), "row_depth must be int."
        self._row_depth = row_depth

        assert isinstance(inverse, bool), "inverse must be bool."
        self._inverse = inverse

        # cannot be changed, only required for associated interleaver
        self._keep_state = True

    #########################################
    # Public methods and properties
    #########################################

    @property
    def axis(self):
        """Axis to be permuted."""
        return self._axis

    @property
    def row_depth(self):
        """Row depth of the row-column interleaver."""
        return self._row_depth

    @property
    def perm_seq(self):
        """Permutation sequence."""
        return self._perm_seq

    @property
    def perm_seq_inv(self):
        """Inverse permutation sequence."""
        return self._perm_seq_inv

    @property
    def keep_state(self):
        """Row-column interleaver always uses same internal state."""
        return True

    def call_inverse(self, inputs):
        """Implements deinterleaver function corresponding to call().

        Input
        -----
            inputs: tf.DType
                2+D tensor of arbitrary shape and arbitrary dtype. Must have at
                least rank two.

        Output
        ------
            : tf.DType
                2+D tensor of same shape and dtype as ``inputs``.
        """
        input_shape = inputs.shape

        x = tf.gather(inputs, self._perm_seq_inv, axis=self._axis)

        x = tf.ensure_shape(x, input_shape)
        return x

    #########################
    # Utility methods
    #########################

    def _generate_perm_rc(self, n_seq, r_depth):
        """Generates a row/column permutation to initialize an rc-interleaver.

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

    #########################
    # Keras layer functions
    #########################

    def build(self, input_shape):
        assert self._axis < len(input_shape), "Axis does match input shape"
        # init rand sequences during build
        assert input_shape[self._axis] is not None, "Unknown shape at req. dim"
        p, pi = self._generate_perm_rc(input_shape[self._axis], self._row_depth)
        self._perm_seq = p
        self._perm_seq_inv = pi

    def call(self, inputs):
        """interleaving function

        This function returns the permuted version of inputs.

        Args:
            inputs (tf.float32): Tensor of arbitrary shape. Must have at least
                rank two.

        Returns:
            `tf.float32`: Tensor of same shape as the input.

        """

        input_shape = inputs.shape

        # re-init if shape has changed, update perm_seq
        if inputs.shape[self._axis] != self._perm_seq.shape[0]:
            self.build(inputs.shape)

        if self._inverse:
            x = tf.gather(inputs, self._perm_seq_inv, axis=self._axis)
        else:
            x = tf.gather(inputs, self._perm_seq, axis=self._axis)

        x = tf.ensure_shape(x, input_shape)
        return x


class RandomInterleaver(Layer):
    # pylint: disable=line-too-long
    """RandomInterleaver(seed=None, keep_batch_constant=True, inverse=False, keep_state=True, axis=-1, dtype=tf.float32, **kwargs)

    Random interleaver permuting a sequence of input symbols.

    The class inherits from the Keras layer class and can be used as layer in a
    Keras model.

    Parameters
    ----------
        seed: int
            Integer defining the random seed used if option ``keep_state`` is
            True.

        keep_batch_constant: bool
            Defaults to True. If set to True each sample in the batch uses the
            same permutation. Otherwise, unique permutations per batch sample
            are generate (slower).

        inverse: bool
            A boolean defaults to False. If True, the inverse permutation is
            performed.

        keep_state: bool
            A boolean defaults to True. If True, the permutation is fixed for
            multiple calls (defined by ``seed`` attribute).

        axis: int
            Defaults to `-1`. The dimension that should be interleaved.
            First dimension (`axis=0`) is not allowed.

        dtype: tf.DType
            Defaults to `tf.float32`. Defines the datatype for internal
            calculations and the output dtype.

    Input
    -----
        (x, seed):
            Either Tuple ``(x, seed)`` or ``x`` only (no tuple) if the internal
            seed should be used:

        x: tf.DType
            2+D tensor of arbitrary shape and dtype.
        seed: int
            An integer defining the state of the random number
            generator. If explicitly given, the global internal seed is
            replaced by this seed. Can be used to realize random
            interleaver/deinterleaver pairs (call with same random seed).

    Output
    ------
        : tf.DType
            2+D tensor of same shape and dtype as the input ``x``.

    Raises
    ------
        AssertionError
            If ``axis`` is not `int`.

        AssertionError
            If ``seed`` is not `None` or `int`.

        AssertionError
            If ``axis`` > number of input dimensions.

        AssertionError
            If ``inverse`` is not bool.

        AssertionError
            If ``keep_state`` is not bool.

        AssertionError
            If ``keep_batch_constant`` is not bool.

        InvalidArgumentError
            When rank(``x``)<2.

    Note
    ----
        To permute the batch dimension, expand_dims at ``axis=0``, interleave
        and remove new dimension.

        The interleaver layer is stateless, i.e., the seed is either random
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
                dtype=tf.float32,
                **kwargs):

        super().__init__(dtype=dtype, **kwargs)

        # verify and store attributes
        assert isinstance(keep_batch_constant, bool), \
            "keep_batch_constant must be bool."
        self._keep_batch_constant = keep_batch_constant

        assert isinstance(axis, int), "axis must be int."
        assert axis!=0, "Cannot permute batch_dim."
        self._axis=axis

        # a global seed is stored and used if called with keep_state=True
        if seed is not None:
            assert isinstance(seed, int), "seed must be int."
        else:
            # generate random seed if no value is provided
            seed = int(np.random.uniform(0, 2**31-1))

        # if keep_state==True this seed is used to generate scrambling sequences
        self._seed = (1337, seed)

        assert isinstance(inverse, bool), "inverse must be boolean"
        self._inverse = inverse
        assert isinstance(keep_state, bool), "keep_state must be boolean"
        self._keep_state = keep_state

        if self._keep_state is False and self._inverse is True:
            print("Note: keep_state=False and, thus, a new realization of " \
                "the interleaver is generated during each call. Thus, " \
                "the inverse interleaver does not correspond to a previous " \
                "interleaver call.")

    #########################################
    # Public methods and properties
    #########################################

    @property
    def seed(self):
        """Seed to generate random sequence."""
        return self._seed[1] # only return the non-fixed seed

    @property
    def axis(self):
        """Axis to be permuted."""
        return self._axis

    @property
    def keep_state(self):
        """Generate new random seed per call."""
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

            s_min_stop: int
                Defaults to 0. Enables early stop if already current s_min< ``s_min_stop`` .
        Output
        ------
            : float
                The S-parameter for the given ``seed``.
        """

        assert isinstance(seed, int), "seed must be int."
        assert isinstance(seq_length, int), "seq_length must be int."
        assert isinstance(s_min_stop, int), "s_min_stop must be int."

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

    def call_inverse(self, inputs):
        """Implements deinterleaver function corresponding to call().

        Input
        -----
            (x, seed):
                Either Tuple ``(x, seed)`` or ``x`` only (no tuple) if the internal
                seed should be used:

            x: tf.DType
                2+D tensor of arbitrary shape and dtype.
            seed: int
                An integer defining the state of the random number
                generator. If explicitly given, the global internal seed is
                replaced by this seed. Can be used to realize random
                interleaver/deinterleaver pairs (call with same random seed).

        Output
        ------
            : tf.DType
                2+D tensor of same shape and dtype as the input ``x``.

        Raises
        ------
            InvalidArgumentError
                When rank(``x``)<2.

            ValueError
                If ``keep_state`` is False and no explicit seed is provided.

        Note
        ----
            In case of inverse interleaving (e.g., at the receiver),
            ``keep_state`` should be True as otherwise a new permutation is
            generated and the output is not equal to the original sequence.
            Alternatively, an explicit seed must be provided as function
            argument.
        """

        if isinstance(inputs, (tuple, list)):
            if len(inputs)==1: # if user wants to call with call([x])
                seed = None
                x = inputs
            elif len(inputs)==2:
                x, seed = inputs
            else:
                raise TypeError("inputs cannot have more than 2 entries.")
        else:
            seed = None
            x = inputs

        input_shape = x.shape
        tf.debugging.assert_greater(tf.rank(x), 1)

        # use seed if explicit seed is provided
        if seed is not None:
            seed = (tf.constant(1337), tf.cast(seed, tf.int32))
        elif self._keep_state:
            # use sequence as defined by seed
            seed = self._seed
        else:
            # This mode is not supported for
            raise ValueError("Deinterleaving not possible for random " \
                "seeds per call (keep_state=False) without explicitly " \
                "providing the seed as inputs.")
        # select if each sample in batch needs own perm (computational complex!)
        if self._keep_batch_constant:
            batch_size = 1
        else:
            batch_size = tf.shape(x)[0]

        perm_seq = self._generate_perm_full(seed,
                                            tf.shape(x)[self._axis],
                                            batch_size,
                                            inverse=True) # activate inverse

        if self._keep_batch_constant:
            # broadcast single sequence over complete batch
            perm_seq = tf.squeeze(perm_seq, axis=0) # remove batch_dim
            x = tf.gather(x, perm_seq, batch_dims=0, axis=self._axis)
        else:
            x = tf.gather(x, perm_seq, batch_dims=1, axis=self._axis)

        # set explicitly for keras models
        x = tf.ensure_shape(x, input_shape)
        return x

    #########################
    # Utility methods
    #########################

    def _generate_perm_full(self, seed, seq_length, batch_size, inverse=False):
        """Generates a random permutation for the interleaver.

        Args:
            seed (int): A shape [2] Tensor, the seed to the random number
                generator.

            seq_length (int): The length of the sequence to be permuted.

            batch_size (int): The batch size (=number of independent
                permutations).

            inverse (bool): Defaults to False. If True, the inverse permutation
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
    # Keras layer functions
    #########################

    def build(self, input_shape):
        """Build Keras layer and check consistency of dimensions."""
        if isinstance(input_shape, list):
            input_shape=input_shape[0]

        assert self._axis < len(input_shape), "Axis does not match input shape."
        assert len(input_shape) > 1, "At least two dims are required."

    def call(self, inputs):
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
            `tf.float32`: Tensor of same shape as the input.

        Raises:
            InvalidArgumentError
                When rank(``x``)<2.

            AssertionError
                If ``seed`` is not None or int.

        Note:
            In case of inverse interleaving (e.g., at the receiver),
            ``keep_state`` should be True as otherwise a new permutation is
            generated and the output is not equal to the original sequence.
            Alternatively, an explicit seed must be provided as function
            argument.
        """

        if isinstance(inputs, (tuple, list)):
            if len(inputs)==1: # if user wants to call with call([x])
                seed = None
                x = inputs
            elif len(inputs)==2:
                x, seed = inputs
            else:
                raise TypeError("inputs cannot have more than 2 entries.")
        else:
            seed = None
            x = inputs

        input_shape = x.shape
        tf.debugging.assert_greater(tf.rank(x), 1)

        # use seed if explicit seed is provided
        if seed is not None:
            seed = (tf.constant(1337), tf.cast(seed, tf.int32))
        # only generate a new random sequence if keep_state==False
        elif self._keep_state:
            # use sequence as defined by seed
            seed = self._seed
        else:
            # generate new seed for each call
            # Note: not necessarily random if XLA is active
            seed = tf.random.uniform([2],
                                     minval=0,
                                     maxval=2**31-1,
                                     dtype=tf.int32)
        # select if each sample in batch needs own perm (computational complex!)
        if self._keep_batch_constant:
            batch_size = 1
        else:
            batch_size = tf.shape(x)[0]

        perm_seq = self._generate_perm_full(seed,
                                            tf.shape(x)[self._axis],
                                            batch_size,
                                            self._inverse)

        if self._keep_batch_constant:
            # broadcast single sequence over complete batch
            perm_seq = tf.squeeze(perm_seq, axis=0) # remove batch_dim
            x = tf.gather(x, perm_seq, batch_dims=0, axis=self._axis)
        else:
            x = tf.gather(x, perm_seq, batch_dims=1, axis=self._axis)

        # set explicitly for keras models
        x = tf.ensure_shape(x, input_shape)
        return x


class Deinterleaver(Layer):
    """Deinterleaver(interleaver, dtype=None, **kwargs)

    Deinterleaver that reverts the interleaver for a given input sequence.

    The class inherits from the Keras layer class and can be used as layer in a
    Keras model.

    Parameters
    ----------
        interleaver: Interleaver
            Associated Interleaver which shall be deinterleaved by this layer.
            Can be either
            :class:`~sionna.fec.interleaving.RandomInterleaver` or
            :class:`~sionna.fec.interleaving.RowColumnInterleaver`.

        dtype: None or tf.DType
            Defaults to `None`. Defines the datatype for internal calculations
            and the output dtype. If no explicit dtype is provided the dtype
            from the associated interleaver is used.

    Input
    -----
        (x, seed):
            Either Tuple ``(x, seed)`` or ``x`` only (no tuple) if the internal
            seed should be used:

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

    Raises
    ------
        AssertionError
            If ``interleaver`` is not a valid instance of Interleaver.

    Note
    ----
        This layer provides a wrapper of the inverse interleaver function.
    """

    def __init__(self,
                 interleaver,
                 dtype=None,
                 **kwargs):

        if not isinstance(interleaver,
                          (RandomInterleaver,
                          RowColumnInterleaver,
                          Turbo3GPPInterleaver)):
            raise ValueError("interleaver is not a valid interleaver instance.")
        self._interleaver = interleaver

        # if dtype is None, use same dtype as associated interleaver
        if dtype is None:
            dtype = self._interleaver.dtype

        super().__init__(dtype=dtype, **kwargs)

        if self._interleaver._keep_state is False:
            print("Warning: deinterleaver requires interleaver to have " \
            "keep_state=True or to explicitly provide the seed as inputs.")

    #########################################
    # Public methods and properties
    #########################################

    @property
    def interleaver(self):
        """Associated interleaver instance."""
        return self._interleaver

    #########################
    # Utility methods
    #########################

    #########################
    # Keras layer functions
    #########################

    def build(self, input_shape):
        """build layer"""
        pass

    def call(self, inputs):
        """deinterleaving function.

        This function returns the permuted version of inputs.

        Args:
            inputs (tf.float32): Tensor of arbitrary shape. Must have at least
                rank two.

        Returns:
            `tf.float32`: Tensor of same shape as the input.
        """

        x = self._interleaver.call_inverse(inputs)

        x = tf.cast(x, super().dtype) # cast output to correct dtype
        return x


class Turbo3GPPInterleaver(Layer):
    # pylint: disable=line-too-long
    """Turbo3GPPInterleaver(inverse=False, axis=-1, dtype=tf.float32, **kwargs)

    Interleaver as used in the 3GPP Turbo codes [3GPPTS36212_I]_ and, thus,
    the maximum length is given as 6144 elements (only for the dimension as
    specific by ``axis``).

    The class inherits from the Keras layer class and can be used as layer in a
    Keras model.

    Parameters
    ----------
        inverse: bool
            A boolean defaults to False. If True, the inverse permutation is
            performed.

        axis: int
            Defaults to `-1`. The dimension that should be interleaved.
            First dimension (`axis=0`) is not allowed.

        dtype: tf.DType
            Defaults to `tf.float32`. Defines the datatype for internal
            calculations and the output dtype.

    Input
    -----
        x: tf.DType
            2+D tensor of arbitrary shape and dtype.

    Output
    ------
        : tf.DType
            2+D tensor of same shape and dtype as the input ``x``.

    Raises
    ------
        AssertionError
            If ``axis`` is not `int`.

        AssertionError
            If ``axis`` > number of input dimensions.

        AssertionError
            If ``inverse`` is not bool.

        InvalidArgumentError
            When rank(``x``)<2.

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
                 dtype=tf.float32,
                 **kwargs):

        super().__init__(dtype=dtype, **kwargs)

        assert isinstance(axis, int), "axis must be int."
        assert axis!=0, "Cannot permute batch dimension."
        self._axis=axis
        self._keep_state = True # only required for deinterleaver
        self.frame_size = None

        assert isinstance(inverse, bool), "inverse must be boolean"
        self._inverse = inverse

        # load interleaver patterns as defined in the 3GPP standard
        self.coeffs_dict = {}
        source = files(coeffs).joinpath("turbo_coeffs.csv")
        with as_file(source) as coeffs.csv:
            csv_reader = np.genfromtxt(coeffs.csv, delimiter=",")

            for (line_count, row) in enumerate(csv_reader):
                if line_count >0: #igonore first line (=header)
                    self.coeffs_dict[int(row[1])] = (int(row[2]), int(row[3]))
    #########################################
    # Public methods and properties
    #########################################

    @property
    def axis(self):
        """Axis to be permuted."""
        return self._axis

    def find_s_min(self, frame_size, s_min_stop=0):
        r"""Find :math:`S` parameter such that :math:`\pi(i)-\pi(j)>S` for all
        :math:`i-j<S`. This can be used to find optimized interleaver patterns.

        ``s_min_stop`` is an additional stopping condition, i.e., stop if
        current :math:`S` is already smaller than ``s_min_stop``.

        Please note that this is a Numpy utility function and usually not part
        of the graph.

        Input
        -----
        frame_size: int
            length of interleaver.

        s_min_stop: int
            Defaults to 0. Enables early stop if already current
            s_min<``s_min_stop``.

        Output
        ------
        : float
            The S-parameter for the given ``frame_size``.
        """

        assert isinstance(s_min_stop, int), "s_min_stop must be int."
        assert isinstance(frame_size, int), "frame_size must be int."
        assert(frame_size<6145), "Interleaver not defined for this frame_size."

        perm_seq = self._generate_perm_full(frame_size)
        perm_seq = perm_seq.numpy()
        s_min = frame_size

        for i in range(len(perm_seq)): # search for all positions in perm_seq
            for j in range(-s_min,s_min,1): # search dist
                if j==0: # ignore identity
                    continue
                if i+j>=0 and i+j<frame_size:
                    d = np.abs(perm_seq[i] - perm_seq[i+j])
                    if d<=np.abs(j):
                        s_min = np.min([s_min, np.abs(j)])
                    if d<s_min and np.abs(j)<s_min:
                        s_min = np.min([s_min, d])
            # early stop
            if s_min<=s_min_stop:
                break
        return int(s_min)

    def call_inverse(self, inputs):
        """Implements deinterleaver function corresponding to call().

        Input
        -----
         x: tf.DType
            2+D tensor of arbitrary shape and dtype.

        Output
        ------
        : tf.DType
            2+D tensor of same shape and dtype as the input ``x``.

        Raises
        ------
        InvalidArgumentError
            When rank(``x``)<2.
        """

        if isinstance(inputs, (tuple, list)):
            if len(inputs)==1: # if user wants to call with call([x])
                x = inputs
            else:
                raise TypeError("inputs cannot have more than 1 entry.")
        else:
            x = inputs

        input_shape = x.shape
        frame_size = input_shape[self._axis]

        # activate inverse
        perm_seq = self._generate_perm_full(frame_size, inverse=True)
        x = tf.gather(x, perm_seq, batch_dims=0, axis=self._axis)

        # set explicitly for keras models
        x = tf.ensure_shape(x, input_shape)
        return x

    #########################
    # Utility methods
    #########################

    def _generate_perm_full(self, frame_size, inverse=False):
        """Generates a random permutation for the interleaver.
        Args:
            frame_size (int): The length of the sequence to be permuted.

            batch_size (int): The batch size (=number of independent
                permutations).

            inverse (bool): Defaults to False. If True, the inverse permutation
                for the given seed is generated.
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
    # Keras layer functions
    #########################

    def build(self, input_shape):
        """Build Keras layer and check consistency of dimensions."""
        if isinstance(input_shape, list):
            input_shape=input_shape[0]

        assert self.axis < len(input_shape), "Axis does not match input shape."
        assert len(input_shape) > 1, "At least two dims are required."

        frame_size = input_shape[self._axis]
        assert(frame_size< 6145), \
            "3GPP Turbo Interleaver is defined for block lengths up to 6144."

    def call(self, inputs):
        """Interleaving function.

        This function returns the permuted version of ``inputs``.
        """

        if isinstance(inputs, (tuple, list)):
            if len(inputs)==1: # if user wants to call with call([x])
                x = inputs
            else:
                raise TypeError("inputs cannot have more than 1 entry.")
        else:
            x = inputs

        input_shape = x.shape
        frame_size = input_shape[self._axis]

        perm_seq = self._generate_perm_full(frame_size, self._inverse)
        x = tf.gather(x, perm_seq, batch_dims=0, axis=self._axis)

        # set explicitly for keras models
        x = tf.ensure_shape(x, input_shape)
        return x
