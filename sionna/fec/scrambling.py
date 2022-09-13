#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Layers for scrambling, descrambling and utility functions."""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


class Scrambler(Layer):
    # pylint: disable=line-too-long
    r"""Scrambler(seed=None, keep_batch_constant=False, sequence=None, binary=True keep_state=True, dtype=tf.float32, **kwargs)

    Randomly flips the state/sign of a sequence of bits or LLRs, respectively.

    The class inherits from the Keras layer class and can be used as layer in a
    Keras model.

    Parameters
    ----------
        seed: int
            Defaults to None. Defines the initial state of the
            pseudo random generator to generate the scrambling sequence.
            If None, a random integer will be generated. Only used
            when called with ``keep_state`` is True.

        keep_batch_constant: bool
            Defaults to False. If True, all samples in the batch are scrambled
            with the same scrambling sequence. Otherwise, per sample a random
            sequence is generated.

        sequence: Array of 0s and 1s or None
            If provided, the seed will be ignored and the explicit scrambling
            sequence is used. Shape must be broadcastable to ``x``.

        binary: bool
            Defaults to True. Indicates whether bit-sequence should be flipped
            (i.e., binary operations are performed) or the signs should be
            flipped (i.e., soft-value/LLR domain-based).

        keep_state: bool
            Defaults to True. Indicates whether the scrambling sequence should
            be kept constant.

        dtype: tf.DType
            Defaults to `tf.float32`. Defines the datatype for internal
            calculations and the output dtype.

    Input
    -----
        (x, seed, binary):
            Either Tuple ``(x, seed, binary)`` or  ``(x, seed)`` or ``x`` only
            (no tuple) if the internal  seed should be used:

        x: tf.float
            1+D tensor of arbitrary shape.
        seed: int
            An integer defining the state of the random number
            generator. If explicitly given, the global internal seed is
            replaced by this seed. Can be used to realize random
            scrambler/descrambler pairs (call with same random seed).
        binary: bool
            Overrules the init parameter `binary` iff explicitly given.
            Indicates whether bit-sequence should be flipped
            (i.e., binary operations are performed) or the signs should be
            flipped (i.e., soft-value/LLR domain-based).

    Output
    ------
        : tf.float
            1+D tensor of same shape as ``x``.

    Note
    ----
        For inverse scrambling, the same scrambler can be re-used (as the values
        are flipped again, i.e., result in the original state). However,
        ``keep_state`` must be set to True as a new sequence would be generated
        otherwise.

        The scrambler layer is stateless, i.e., the seed is either random
        during each call or must be explicitly provided during init/call.
        This simplifies XLA/graph execution.
        If the seed is provided in the init() function, this fixed seed is used
        for all calls. However, an explicit seed can be provided during
        the call function to realize `true` random states.

        Scrambling is typically used to ensure equal likely `0`  and `1` for
        sources with unequal bit probabilities. As we have a perfect source in
        the simulations, this is not required. However, for all-zero codeword
        simulations and higher-order modulation, so-called "channel-adaptation"
        [Pfister03]_ is required.

    Raises
    ------
        AssertionError
            If ``seed`` is not int.

        AssertionError
            If ``keep_batch_constant`` is not bool.

        AssertionError
            If ``binary`` is not bool.

        AssertionError
            If ``keep_state`` is not bool.

        AssertionError
            If ``seed`` is provided to list of inputs but not an
            int.

        TypeError
            If `dtype` of ``x`` is not as expected.
    """
    def __init__(self,
                 seed=None,
                 keep_batch_constant=False,
                 binary=True,
                 sequence=None,
                 keep_state=True,
                 dtype=tf.float32,
                 **kwargs):

        if dtype not in (tf.float16, tf.float32, tf.float64, tf.int8,
            tf.int32, tf.int64, tf.uint8, tf.uint16, tf.uint32):
            raise ValueError("Unsupported dtype.")

        super().__init__(dtype=dtype, **kwargs)

        assert isinstance(keep_batch_constant, bool), \
            "keep_batch_constant must be bool."
        self._keep_batch_constant = keep_batch_constant

        if seed is not None:
            if sequence is not None:
                print("Note: explicit scrambling sequence provided. " \
                      "Seed will be ignored.")
            assert isinstance(seed, int), "seed must be int."
        else:
            seed = int(np.random.uniform(0, 2**31-1))

        assert isinstance(binary, bool), "binary must be bool."
        assert isinstance(keep_state, bool), "keep_state must be bool."

        self._binary = binary
        self._keep_state = keep_state

        self._check_input = True

        # if keep_state==True this seed is used to generate scrambling sequences
        self._seed = (1337, seed)

        # if an explicit sequence is provided the above parameters will be
        # ignored
        self._sequence = None
        if sequence is not None:
            sequence = tf.cast(sequence, self.dtype)
            # check that sequence is binary
            tf.debugging.assert_equal(
                tf.reduce_min(
                    tf.cast(
                        tf.logical_or(
                            tf.equal(sequence, tf.constant(0, self.dtype)),
                            tf.equal(sequence, tf.constant(1, self.dtype)),),
                        self.dtype)),
                tf.constant(1, self.dtype),
                "Scrambling sequence must be binary.")
            self._sequence = sequence


    #########################################
    # Public methods and properties
    #########################################

    @property
    def seed(self):
        """Seed used to generate random sequence."""
        return self._seed[1] # only return the non-fixed seed

    @property
    def keep_state(self):
        """Indicates if new random sequences are used per call."""
        return self._keep_state

    @property
    def sequence(self):
        """Explicit scrambling sequence if provided."""
        return self._sequence

    #########################
    # Utility methods
    #########################

    def _generate_scrambling(self, input_shape, seed):
        r"""Generates a random sequence of `0`s and `1`s that can be used
        to initialize a scrambler and updates the internal attributes.
        """
        if self._keep_batch_constant:
            input_shape_no_bs = input_shape[1:]
            seq = tf.random.stateless_uniform(input_shape_no_bs,
                                              seed,
                                              minval=0,
                                              maxval=2,
                                              dtype=tf.int32)
            # expand batch dim such that it can be broadcasted
            seq = tf.expand_dims(seq, axis=0)
        else:
            seq = tf.random.stateless_uniform(input_shape,
                                              seed,
                                              minval=0,
                                              maxval=2,
                                              dtype=tf.int32)

        return tf.cast(seq, self.dtype) # enable flexible dtypes

    #########################
    # Keras layer functions
    #########################

    def build(self, input_shape):
        """Build the model and initialize variables."""
        pass

    def call(self, inputs):
        r"""scrambling function.

        This function returns the scrambled version of ``inputs``.

        ``inputs`` can be either a list ``[x, seed]`` or single tensor ``x``.

        Args:
            inputs (List): ``[x, seed]``, where
            ``x`` (tf.float32): Tensor of arbitrary shape.
            ``seed`` (int): An integer defining the state of the random number
                generator. If explicitly given, the global internal seed is
                replaced by this seed. Can be used the realize random
                scrambler/descrambler pairs (call with same random seed).

        Returns:
            `tf.float32`: Tensor of same shape as the input.

        Raises:
            AssertionError
                If ``seed`` is not None or int.

            TypeError
                If `dtype` of ``x`` is not as expected.
        """
        is_binary = self._binary # can be overwritten if explicitly provided

        if isinstance(inputs, (tuple, list)):
            if len(inputs)==1: # if user wants to call with call([x])
                seed = None
                x = inputs
            elif len(inputs)==2:
                x, seed = inputs
            elif len(inputs)==3:
            # allow that flag binary is explicitly provided (for descrambler)
                x, seed, is_binary = inputs
                # is binary can be either a tensor or bool
                if isinstance(is_binary, tf.Tensor):
                    if not is_binary.dtype.is_bool:
                        raise TypeError("binary must be bool.")
                else: # is boolean
                    assert isinstance(is_binary.dtype, bool), \
                    "binary must be bool."
            else:
                raise TypeError("inputs cannot have more than 3 entries.")
        else:
            seed = None
            x = inputs

        tf.debugging.assert_type(x, self.dtype,
                                 "Invalid input dtype.")

        input_shape = tf.shape(x)

        # generate random sequence on-the-fly (due to unknown shapes during
        # compile/build time)
        # use seed if explicit seed is provided
        if seed is not None:
            #assert seed.dtype.is_integer, "seed must be int."
            seed = (1337, seed)
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

        # apply sequence if explicit sequence is provided
        if self._sequence is not None:
            rand_seq = self._sequence
        else:
            rand_seq = self._generate_scrambling(input_shape, seed)

        if is_binary:
            # flipp the bits by substraction and map -1 to 1 via abs(.) operator
            x_out = tf.abs(x - rand_seq)
        else:
            rand_seq_bipol = -2 * rand_seq + 1
            x_out = tf.multiply(x, rand_seq_bipol)

        return x_out

class Descrambler(Layer):
    r"""Descrambler(scrambler, binary=True, dtype=None, **kwargs)

    Descrambler for a given scrambler.

    The class inherits from the Keras layer class and can be used as layer in a
    Keras model.

    Parameters
    ----------
        scrambler: Scrambler
            Associated :class:`~sionna.fec.scrambling.Scrambler` instance which
            should be descrambled.

        binary: bool
            Defaults to True. Indicates whether bit-sequence should be flipped
            (i.e., binary operations are performed) or the signs should be
            flipped (i.e., soft-value/LLR domain-based).

        dtype: None or tf.DType
            Defaults to `None`. Defines the datatype for internal calculations
            and the output dtype. If no explicit dtype is provided the dtype
            from the associated interleaver is used.

    Input
    -----
        (x, seed):
            Either Tuple ``(x, seed)`` or ``x`` only (no tuple) if the internal
            seed should be used:

        x: tf.float
            1+D tensor of arbitrary shape.

        seed: int
            An integer defining the state of the random number
            generator. If explicitly given, the global internal seed is
            replaced by this seed. Can be used to realize random
            scrambler/descrambler pairs (call with same random seed).

    Output
    ------
        : tf.float
            1+D tensor of same shape as ``x``.

    Raise
    -----
        AssertionError
            If ``scrambler`` is not an instance of `Scrambler`.

        AssertionError
            If ``seed`` is provided to list of inputs but not an
            int.

        TypeError
            If `dtype` of ``x`` is not as expected.
    """
    def __init__(self,
                 scrambler,
                 binary=True,
                 dtype=None,
                 **kwargs):

        assert isinstance(scrambler, Scrambler), \
            "scrambler must be an instance of Scrambler."
        self._scrambler = scrambler

        assert isinstance(binary, bool), "binary must be bool."
        self._binary = binary

        # if dtype is None, use same dtype as associated scrambler
        if dtype is None:
            dtype = self._scrambler.dtype

        super().__init__(dtype=dtype, **kwargs)

        if self._scrambler.keep_state is False:
            print("Warning: scrambler uses random sequences that cannot be " \
                  "access by descrambler. Please use keep_state=True and " \
                  "provide explicit random seed as input to call function.")

        if self._scrambler.dtype != self.dtype:
            print("Scrambler and descrambler are using different " \
                "dtypes. This will cause an internal implicit cast.")

    #########################################
    # Public methods and properties
    #########################################

    @property
    def scrambler(self):
        """Associated scrambler instance."""
        return self._scrambler

    #########################
    # Utility methods
    #########################

    #########################
    # Keras layer functions
    #########################

    def build(self, input_shape):
        """Build the model and initialize variables."""
        pass

    def call(self, inputs):
        r"""Descrambling function.

        This function returns the descrambled version of ``inputs``.

        ``inputs`` can be either a list ``[x, seed]`` or single tensor ``x``.

        Args:
            inputs (List): ``[x, seed]``, where
            ``x`` (tf.float32): Tensor of arbitrary shape.
            ``seed`` (int): An integer defining the state of the random number
                generator. If not explicitly given, the global internal seed is
                replaced by this seed. Can be used the realize random
                scrambler/descrambler pairs (must be called with same random
                seed).

        Returns:
            `tf.float32`: Tensor of same shape as the input.

        Raises:
            AssertionError: If ``seed`` is not `None` or `int`.
        """

        if isinstance(inputs, (tuple, list)):
            if len(inputs)>2:
                raise TypeError("inputs cannot have more than 2 entries.")
            else: # seed explicitly given
                inputs.append(self._binary)
        else: # seed not given
            s = self._scrambler.seed # use seed from associated scrambler
            inputs = (inputs, s, self._binary)

        x_out = self._scrambler(inputs)

        # scrambler could potentially have different dtypes
        return tf.cast(x_out, super().dtype)
