#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""Blocks for scrambling, descrambling and utility functions."""
import tensorflow as tf
from sionna.phy import config, Block
from sionna.phy.utils import expand_to_rank
from sionna.phy.nr.utils import generate_prng_seq

class Scrambler(Block):
    # pylint: disable=line-too-long
    r"""Randomly flips the state/sign of a sequence of bits or LLRs, respectively.

    Parameters
    ----------
    seed: `None` (default) | int
        Defaults to None. Defines the initial state of the
        pseudo random generator to generate the scrambling sequence.
        If None, a random integer will be generated. Only used
        when called with ``keep_state`` is True.

    keep_batch_constant: `bool`, (default `False`)
        If `True`,  all samples in the batch are scrambled with the same
        scrambling sequence. Otherwise, per sample a random
        sequence is generated.

    sequence: None | Array of 0s and 1s
        If provided, the seed will be ignored and the explicit scrambling
        sequence is used. Shape must be broadcastable to ``x``.

    binary: `bool`, (default `True`)
        Indicates whether bit-sequence should be flipped
        (i.e., binary operations are performed) or the signs should be
        flipped (i.e., soft-value/LLR domain-based).

    keep_state: `bool`, (default `True`)
        Indicates whether the scrambling sequence should be kept constant.

    precision : `None` (default) | 'single' | 'double'
        Precision used for internal calculations and outputs.
        If set to `None`, :py:attr:`~sionna.phy.config.precision` is used.

    Input
    -----
    x: tf.float
        Tensor of arbitrary shape.

    seed: `None` (default) | int
        An integer defining the state of the random number
        generator. If explicitly given, the global internal seed is
        replaced by this seed. Can be used to realize random
        scrambler/descrambler pairs (call with same random seed).

    binary: `None` (default) | bool
        Overrules the init parameter `binary` iff explicitly given.
        Indicates whether bit-sequence should be flipped
        (i.e., binary operations are performed) or the signs should be
        flipped (i.e., soft-value/LLR domain-based).

    Output
    ------
    : tf.float
        Tensor of same shape as ``x``.

    Note
    ----
    For inverse scrambling, the same scrambler can be re-used (as the values
    are flipped again, i.e., result in the original state). However,
    ``keep_state`` must be set to `True` as a new sequence would be generated
    otherwise.

    The scrambler block is stateless, i.e., the seed is either random
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
    """
    def __init__(self,
                 seed=None,
                 keep_batch_constant=False,
                 binary=True,
                 sequence=None,
                 keep_state=True,
                 precision=None,
                 **kwargs):

        super().__init__(precision=precision, **kwargs)

        if not isinstance(keep_batch_constant, bool):
            raise TypeError("keep_batch_constant must be bool.")
        self._keep_batch_constant = keep_batch_constant

        if seed is not None:
            if sequence is not None:
                print("Note: explicit scrambling sequence provided. " \
                      "Seed will be ignored.")
            if not isinstance(seed, int):
                raise TypeError("seed must be int.")
        else:
            seed = int(config.np_rng.uniform(0, 2**31-1))

        # allow tf.bool as well
        if not (isinstance(binary, bool) or \
                (tf.is_tensor(binary) and binary.dtype == tf.bool)):
            raise TypeError("binary must be bool.")
        self._binary = tf.cast(binary, tf.bool)

        if not isinstance(keep_state, bool):
            raise TypeError("keep_state must be bool.")
        self._keep_state = keep_state

        self._check_input = True

        # if keep_state==True this seed is used to generate scrambling sequences
        self._seed = (1337, seed)

        # if an explicit sequence is provided the above parameters will be
        # ignored
        self._sequence = None
        if sequence is not None:
            sequence = tf.cast(sequence, self.rdtype)
            # check that sequence is binary
            x = tf.logical_or(tf.equal(sequence, tf.constant(0, self.rdtype)),
                              tf.equal(sequence, tf.constant(1, self.rdtype)))
            x = tf.reduce_min(tf.cast(x, self.rdtype))
            tf.debugging.assert_equal(x, tf.constant(1, self.rdtype),
                                     "Scrambling sequence must be binary.")
            self._sequence = sequence

    #########################################
    # Public methods and properties
    #########################################

    @property
    def seed(self):
        """Seed used to generate random sequence"""
        return self._seed[1] # only return the non-fixed seed

    @property
    def keep_state(self):
        """Indicates if new random sequences are used per call"""
        return self._keep_state

    @property
    def sequence(self):
        """Explicit scrambling sequence if provided"""
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

        return tf.cast(seq, self.rdtype) # enable flexible dtypes

    ##################
    # Public functions
    ##################

    def call(self, x, seed=None, binary=None):
        r"""scrambling function.

        This function returns the scrambled version of ``x``.

        Input
        -----
        x: tf.float
            Tensor of arbitrary shape.

        seed: `None` (default) | int
            An integer defining the state of the random number
            generator. If explicitly given, the global internal seed is
            replaced by this seed. Can be used to realize random
            scrambler/descrambler pairs (call with same random seed).

        binary: `None` (default) | bool
            Overrules the init parameter `binary` iff explicitly given.
            Indicates whether bit-sequence should be flipped
            (i.e., binary operations are performed) or the signs should be
            flipped (i.e., soft-value/LLR domain-based).

        Output
        ------
        : tf.float
            Tensor of same shape as ``x``.
        """

        # providing the binary flag explicitly for the Descrambler
        if binary is None:
            binary = self._binary # can be overwritten if explicitly provided
        else:
            # allow tf.bool as well
            if not (isinstance(binary, bool) or \
                    (tf.is_tensor(binary) and binary.dtype == tf.bool)):
                raise TypeError("binary must be bool.")
            binary = tf.cast(binary, tf.bool)

        input_shape = tf.shape(x)
        # we allow non float input dtypes
        input_dtype = x.dtype
        x = tf.cast(x, self.rdtype)

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
            seed = config.tf_rng.uniform([2],
                                         minval=0,
                                         maxval=2**31-1,
                                         dtype=tf.int32)

        # apply sequence if explicit sequence is provided
        if self._sequence is not None:
            rand_seq = self._sequence
        else:
            rand_seq = self._generate_scrambling(input_shape, seed)


        if binary:
            # flip the bits by subtraction and map -1 to 1 via abs(.) operator
            x_out = tf.abs(x - rand_seq)
        else:
            rand_seq_bipol = -2 * rand_seq + 1
            x_out = tf.multiply(x, rand_seq_bipol)

        return tf.cast(x_out, input_dtype)

class TB5GScrambler(Block):
    # pylint: disable=line-too-long
    r"""5G NR Scrambler for PUSCH and PDSCH channel.

    Implements the pseudo-random bit scrambling as defined in
    [3GPPTS38211_scr]_ Sec. 6.3.1.1 for the "PUSCH" channel and in Sec. 7.3.1.1
    for the "PDSCH" channel.

    Only for the "PDSCH" channel, the scrambler can be configured for two
    codeword transmission mode. Hereby, ``codeword_index`` corresponds to the
    index of the codeword to be scrambled.

    If ``n_rnti`` are a list of ints, the scrambler assumes that the second
    last axis contains `len(` ``n_rnti`` `)` elements. This allows independent
    scrambling for multiple independent streams.

    Parameters
    ----------
    n_rnti: int | list of ints
        RNTI identifier provided by higher layer. Defaults to 1 and must be
        in range `[0, 65335]`. If a list is provided, every list element
        defines a scrambling sequence for multiple independent streams.

    n_id: int | list of ints
        Scrambling ID related to cell id and provided by higher layer.
        Defaults to 1 and must be in range `[0, 1023]`. If a list is
        provided, every list element defines a scrambling sequence for
        multiple independent streams.

    binary: `bool`, (default `True`)
        Indicates whether bit-sequence should be flipped
        (i.e., binary operations are performed) or the signs should be
        flipped (i.e., soft-value/LLR domain-based).

    channel_type: str, 'PUSCH' | 'PDSCH'
        Can be either 'PUSCH' or 'PDSCH'.

    codeword_index: int, 0 | 1
        Scrambler can be configured for two codeword transmission.
        ``codeword_index`` can be either 0 or 1.

    precision : `None` (default) | 'single' | 'double'
        Precision used for internal calculations and outputs.
        If set to `None`, :py:attr:`~sionna.phy.config.precision` is used.

    Input
    -----
    x: tf.float
        Tensor of arbitrary shape. If ``n_rnti`` and ``n_id`` are a
        list, it is assumed that ``x`` has shape
        `[...,num_streams, n]` where `num_streams=len(` ``n_rnti`` `)`.

    binary: `None` (default) | bool
        Overrules the init parameter `binary` iff explicitly given.
        Indicates whether bit-sequence should be flipped
        (i.e., binary operations are performed) or the signs should be
        flipped (i.e., soft-value/LLR domain-based).

    Output
    ------
    : tf.float
        Tensor of same shape as ``x``.

    Note
    ----
    The parameters radio network temporary identifier (RNTI) ``n_rnti`` and
    the datascrambling ID ``n_id`` are usually provided be the higher layer protocols.

    For inverse scrambling, the same scrambler can be re-used (as the values
    are flipped again, i.e., result in the original state).
    """
    def __init__(self,
                 n_rnti=1,
                 n_id=1,
                 binary=True,
                 channel_type="PUSCH",
                 codeword_index=0,
                 precision=None,
                 **kwargs):

        super().__init__(precision=precision, **kwargs)

        # check inputs for consistency
        if not isinstance(binary, bool):
            raise TypeError("binary must be bool.")
        self._binary = binary

        if channel_type not in ("PDSCH", "PUSCH"):
            raise TypeError("Unsupported channel_type.")

        if codeword_index not in (0, 1):
            raise ValueError("codeword_index must be 0 or 1.")

        self._check_input = True
        self._input_shape = None

        # allow list input for independent multi-stream scrambling
        if isinstance(n_rnti, (list, tuple)):
            if not isinstance(n_id, (list, tuple)):
                raise TypeError("n_id must be a list of same length as n_rnti.")

            if not len(n_rnti)==len(n_id):
                raise ValueError("n_rnti and n_id must be of same length.")

            self._multi_stream = True
        else:
            n_rnti = [n_rnti]
            n_id = [n_id]
            self._multi_stream = False

        # check all entries for consistency
        for idx, (nr, ni) in enumerate(zip(n_rnti, n_id)):

            # allow floating inputs, but verify that it represent an int value
            if not nr%1==0:
                raise ValueError("n_rnti must be integer.")
            if not nr in range(2**16):
                raise ValueError("n_rnti must be in [0, 65535].")
            n_rnti[idx] = int(nr)

            if not ni%1==0:
                raise ValueError("n_rnti must be integer.")
            if not ni in range(2**10):
                raise ValueError("n_id must be in [0, 1023].")
            n_id[idx] = int(ni)

        self._c_init = []
        if channel_type=="PUSCH":
            # defined in 6.3.1.1 in 38.211
            for nr, ni in zip(n_rnti, n_id):
                self._c_init += [nr * 2**15 + ni]
        elif channel_type =="PDSCH":
            # defined in 7.3.1.1 in 38.211
            for nr, ni in zip(n_rnti, n_id):
                self._c_init += [nr * 2**15 + codeword_index * 2**14 + ni]

    ###############################
    # Public methods and properties
    ###############################

    @property
    def keep_state(self):
        """Required for descrambler, is always `True` for the TB5GScrambler."""
        return True

    #################
    # Utility methods
    #################

    def _generate_scrambling(self, input_shape):
        r"""Returns random sequence of `0`s and `1`s following
        [3GPPTS38211_scr]_ ."""

        seq = generate_prng_seq(input_shape[-1], self._c_init[0])
        seq = tf.constant(seq, self.rdtype) # enable flexible dtypes
        seq = expand_to_rank(seq, len(input_shape), axis=0)

        if self._multi_stream:
            for c in self._c_init[1:]:
                s = generate_prng_seq(input_shape[-1], c)
                s = tf.constant(s, self.rdtype) # enable flexible dtypes
                s = expand_to_rank(s, len(input_shape), axis=0)
                seq = tf.concat([seq, s], axis=-2)

        return seq

    # pylint: disable=(unused-argument)
    def build(self, input_shape, **kwargs):
        """Initialize pseudo-random scrambling sequence."""

        self._input_shape = input_shape

        # in multi-stream mode, the axis=-2 must have dimension=len(c_init)
        if self._multi_stream:
            assert input_shape[-2]==len(self._c_init), \
                "Dimension of axis=-2 must be equal to len(n_rnti)."

        self._sequence = self._generate_scrambling(input_shape)

    def call(self, x, /, *, binary=None):
        r"""This function returns the scrambled version of ``x``.
        """

        if binary is None:
            binary = self._binary
        else:
            # allow tf.bool as well
            if not (isinstance(binary, bool) or \
                (tf.is_tensor(binary) and binary.dtype == tf.bool)):
                raise TypeError("binary must be bool.")

        if not x.shape[-1]==self._input_shape:
            self.build(x.shape)

        # support various non-float dtypes
        input_dtype = x.dtype
        x = tf.cast(x, self.rdtype)

        if binary:
            # flip the bits by subtraction and map -1 to 1 via abs(.) operator
            x_out = tf.abs(x - self._sequence)
        else:
            rand_seq_bipol = -2 * self._sequence + 1
            x_out = tf.multiply(x, rand_seq_bipol)

        return tf.cast(x_out, input_dtype)

class Descrambler(Block):
    r"""Descrambler for a given scrambler.

    Parameters
    ----------
    scrambler: Scrambler, TB5GScrambler
        Associated :class:`~sionna.phy.fec.scrambling.Scrambler` or
        :class:`~sionna.phy.fec.scrambling.TB5GScrambler` instance which
        should be descrambled.

    binary:  `bool`, (default `True`)
        Indicates whether bit-sequence should be flipped (i.e., binary
        operations are performed) or the signs should be flipped (i.e.,
        soft-value/LLR domain-based).

    precision : `None` (default) | 'single' | 'double'
        Precision used for internal calculations and outputs.
        If set to `None`, :py:attr:`~sionna.phy.config.precision` is used.

    Input
    -----
    x: tf.float
        Tensor of arbitrary shape.

    seed: int
        An integer defining the state of the random number
        generator. If explicitly given, the global internal seed is
        replaced by this seed. Can be used to realize random
        scrambler/descrambler pairs (call with same random seed).

    Output
    ------
    : tf.float
        Tensor of same shape as ``x``.

    """
    def __init__(self,
                 scrambler,
                 binary=True,
                 precision=None,
                 **kwargs):

        if not isinstance(scrambler, (Scrambler, TB5GScrambler)):
            raise TypeError("scrambler must be an instance of Scrambler.")
        self._scrambler = scrambler

        # if precision is None, use same precision as associated scrambler
        if precision is None:
            precision = self._scrambler.precision
        super().__init__(precision=precision, **kwargs)

        # allow tf.bool as well
        if not (isinstance(binary, bool) or \
                (tf.is_tensor(binary) and binary.dtype == tf.bool)):
            raise TypeError("binary must be bool.")
        self._binary = tf.cast(binary, tf.bool)

        if self._scrambler.keep_state is False:
            print("Warning: scrambler uses random sequences that cannot be " \
                  "access by descrambler. Please use keep_state=True and " \
                  "provide explicit random seed as input to call function.")

        if self._scrambler.precision != self.precision:
            print("Scrambler and descrambler are using different " \
                 "precision. This will cause an internal implicit cast.")

    ###############################
    # Public methods and properties
    ###############################

    @property
    def scrambler(self):
        """Associated scrambler instance"""
        return self._scrambler

    def call(self, x, /, *,seed=None):
        r"""Descrambling function

        This function returns the descrambled version of ``inputs``.

        Args:
        ``x`` (tf.float): Tensor of arbitrary shape.

        ``seed`` (int): An integer defining the state of the random number
            generator. If not explicitly given, the global internal seed is
            replaced by this seed. Can be used the realize random
            scrambler/descrambler pairs (must be called with same random
            seed).

        Returns:
            `tf.float`: Tensor of same shape as the input.

        """
        # cast to support non float input types
        input_dt = x.dtype
        x = tf.cast(x, self.rdtype)
        # Scrambler
        if isinstance(self._scrambler, Scrambler):
            if seed is not None:
                s = seed
            else:
                s = self._scrambler.seed # use seed from associated scrambler
            x_out = self._scrambler(x, seed=s, binary=self._binary)
        elif isinstance(self._scrambler, TB5GScrambler):
            x_out = self._scrambler(x, binary=self._binary)
        else:
            raise TypeError("Unknown Scrambler type.")

        # scrambler could potentially have different dtypes
        return tf.cast(x_out, input_dt)
