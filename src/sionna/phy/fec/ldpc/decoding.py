#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#
"""Blocks for channel decoding and utility functions."""

import tensorflow as tf
import numpy as np
import scipy as sp # for sparse H matrix computations
from sionna.phy import Block
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
import types

class LDPCBPDecoder(Block):
    # pylint: disable=line-too-long
    r"""Iterative belief propagation decoder for low-density parity-check (LDPC)
    codes and other `codes on graphs`.

    This class defines a generic belief propagation decoder for decoding
    with arbitrary parity-check matrices. It can be used to iteratively
    estimate/recover the transmitted codeword (or information bits) based on the
    LLR-values of the received noisy codeword observation.

    Per default, the decoder implements the flooding message passing algorithm
    [Ryan]_, i.e., all nodes are updated in a parallel fashion. Different check node update functions are available

    (1) `boxplus`

        .. math::
            y_{j \to i} = 2 \operatorname{tanh}^{-1} \left( \prod_{i' \in \mathcal{N}(j) \setminus i} \operatorname{tanh} \left( \frac{x_{i' \to j}}{2} \right) \right)

    (2) `boxplus-phi`

        .. math::
            y_{j \to i} = \alpha_{j \to i} \cdot \phi \left( \sum_{i' \in \mathcal{N}(j) \setminus i} \phi \left( |x_{i' \to j}|\right) \right)

        with :math:`\phi(x)=-\operatorname{log}(\operatorname{tanh} \left(\frac{x}{2}) \right)`

    (3) `minsum`

        .. math::
            \qquad y_{j \to i} = \alpha_{j \to i} \cdot {min}_{i' \in \mathcal{N}(j) \setminus i} \left(|x_{i' \to j}|\right)

    (4) `offset-minsum`

    .. math::
            \qquad y_{j \to i} = \alpha_{j \to i} \cdot {max} \left( {min}_{i' \in \mathcal{N}(j) \setminus i} \left(|x_{i' \to j}| \right)-\beta , 0\right)

    where :math:`\beta=0.5` and and :math:`y_{j \to i}` denotes the message
    from check node (CN) *j* to variable node (VN) *i* and :math:`x_{i \to j}`
    from VN *i* to CN *j*, respectively.  Further, :math:`\mathcal{N}(j)`
    denotes all indices of connected VNs to CN *j* and

    .. math::
        \alpha_{j \to i} = \prod_{i' \in \mathcal{N}(j) \setminus i} \operatorname{sign}(x_{i' \to j})

    is the sign of the outgoing message. For further details we refer to
    [Ryan]_ and [Chen]_ for offset corrected minsum.

    Note that for full 5G 3GPP NR compatibility, the correct puncturing and
    shortening patterns must be applied (cf. [Richardson]_ for details), this
    can be done by :class:`~sionna.phy.fec.ldpc.encoding.LDPC5GEncoder` and
    :class:`~sionna.phy.fec.ldpc.decoding.LDPC5GDecoder`, respectively.

    If required, the decoder can be made trainable and is fully differentiable
    by following the concept of `weighted BP` [Nachmani]_. For this, custom
    callbacks can be registered that scale the messages during decoding. Please
    see the corresponding tutorial notebook for details.

    For numerical stability, the decoder applies LLR clipping of +/- `llr_max`
    to the input LLRs.

    Parameters
    ----------
    pcm: ndarray
        An ndarray of shape `[n-k, n]` defining the parity-check matrix
        consisting only of `0` or `1` entries. Can be also of type `scipy.
        sparse.csr_matrix` or `scipy.sparse.csc_matrix`.

    cn_update: str, "boxplus-phi" (default) | "boxplus" | "minsum" | "offset-minsum" | "identity" | callable
        Check node update rule to be used as described above.
        If a callable is provided, it will be used instead as CN update.
        The input of the function is a ragged tensor of v2c messages of shape
        `[num_cns, None, batch_size]` where the second dimension is ragged
        (i.e., depends on the individual CN degree).

    vn_update: str, "sum" (default) | "identity" | callable
        Variable node update rule to be used.
        If a callable is provided, it will be used instead as VN update.
        The input of the function is a ragged tensor of c2v messages of shape
        `[num_vns, None, batch_size]` where the second dimension is ragged
        (i.e., depends on the individual VN degree).

    cn_schedule: "flooding" | [num_update_steps, num_active_nodes], tf.int
        Defines the CN update scheduling per BP iteration. Can be either
        "flooding" to update all nodes in parallel (recommended) or an 2D tensor
        where each row defines the `num_active_nodes` node indices to be
        updated per subiteration. In this case each BP iteration runs
        `num_update_steps` subiterations, thus the decoder's level of
        parallelization is lower and usually the decoding throughput decreases.

    hard_out: `bool`, (default `True`)
        If `True`,  the decoder provides hard-decided codeword bits instead of
        soft-values.

    num_iter: int
        Defining the number of decoder iteration (due to batching, no early
        stopping used at the moment!).

    llr_max: float (default 20) | `None`
        Internal clipping value for all internal messages. If `None`, no
        clipping is applied.

    v2c_callbacks, `None` (default) | list of callables
        Each callable will be executed after each VN update with the following
        arguments `msg_vn_rag_`, `it`, `x_hat`,where `msg_vn_rag_` are the v2c
        messages as ragged tensor of shape `[num_vns, None, batch_size]`,
        `x_hat` is the current estimate of each VN of shape
        `[num_vns, batch_size]` and `it` is the current iteration counter.
        It must return and updated version of `msg_vn_rag_` of same shape.

    c2v_callbacks: `None` (default) | list of callables
        Each callable will be executed after each CN update with the following
        arguments `msg_cn_rag_` and `it` where `msg_cn_rag_` are the c2v
        messages as ragged tensor of shape `[num_cns, None, batch_size]` and
        `it` is the current iteration counter.
        It must return and updated version of `msg_cn_rag_` of same shape.

    return_state: `bool`, (default `False`)
        If `True`,  the internal VN messages ``msg_vn`` from the last decoding
        iteration are returned, and ``msg_vn`` or `None` needs to be given as a
        second input when calling the decoder.
        This can be used for iterative demapping and decoding.

    precision : `None` (default) | 'single' | 'double'
        Precision used for internal calculations and outputs.
        If set to `None`, :py:attr:`~sionna.phy.config.precision` is used.

    Input
    -----
    llr_ch: [...,n], tf.float
        Tensor containing the channel logits/llr values.

    msg_v2c: `None` | [num_edges, batch_size], tf.float
        Tensor of VN messages representing the internal decoder state.
        Required only if the decoder shall use its previous internal state, e.g.
        for iterative detection and decoding (IDD) schemes.

    Output
    ------
    : [...,n], tf.float
        Tensor of same shape as ``llr_ch`` containing
        bit-wise soft-estimates (or hard-decided bit-values) of all
        codeword bits.

    : [num_edges, batch_size], tf.float:
        Tensor of VN messages representing the internal decoder state.
        Returned only if ``return_state`` is set to `True`.

    Note
    ----
    As decoding input logits :math:`\operatorname{log} \frac{p(x=1)}{p(x=0)}`
    are assumed for compatibility with the learning framework, but internally
    log-likelihood ratios (LLRs) with definition
    :math:`\operatorname{log} \frac{p(x=0)}{p(x=1)}` are used.

    The decoder is not (particularly) optimized for quasi-cyclic (QC) LDPC
    codes and, thus, supports arbitrary parity-check matrices.

    The decoder is implemented by using '"ragged Tensors"' [TF_ragged]_ to
    account for arbitrary node degrees. To avoid a performance degradation
    caused by a severe indexing overhead, the batch-dimension is shifted to
    the last dimension during decoding.
    """

    def __init__(self,
                 pcm,
                 cn_update="boxplus-phi",
                 vn_update="sum",
                 cn_schedule="flooding",
                 hard_out=True,
                 num_iter=20,
                 llr_max=20.,
                 v2c_callbacks=None,
                 c2v_callbacks=None,
                 return_state=False,
                 precision=None,
                 **kwargs):

        super().__init__(precision=precision, **kwargs)

        # check inputs for consistency
        if not isinstance(hard_out, bool):
            raise TypeError('hard_out must be bool.')
        if not isinstance(num_iter, int):
            raise TypeError('num_iter must be int.')
        if num_iter<0:
            raise ValueError('num_iter cannot be negative.')
        if not isinstance(return_state, bool):
            raise TypeError('return_state must be bool.')

        if isinstance(pcm, np.ndarray):
            if not np.array_equal(pcm, pcm.astype(bool)):
                raise ValueError('PC matrix must be binary.')
        elif isinstance(pcm, sp.sparse.csr_matrix):
            if not np.array_equal(pcm.data, pcm.data.astype(bool)):
                raise ValueError('PC matrix must be binary.')
        elif isinstance(pcm, sp.sparse.csc_matrix):
            if not np.array_equal(pcm.data, pcm.data.astype(bool)):
                raise ValueError('PC matrix must be binary.')
        else:
            raise TypeError("Unsupported dtype of pcm.")

        # Deprecation warning for cn_type
        if 'cn_type' in kwargs:
            raise TypeError("'cn_type' is deprecated; use 'cn_update' instead.")

        # init decoder parameters
        self._pcm = pcm
        self._hard_out = hard_out
        self._num_iter = tf.constant(num_iter, dtype=tf.int32)
        self._return_state = return_state

        self._num_cns = pcm.shape[0] # total number of check nodes
        self._num_vns = pcm.shape[1] # total number of variable nodes

        # internal value for llr clipping
        if not isinstance(llr_max, (int, float)):
            raise TypeError("llr_max must be int or float.")
        self._llr_max = tf.cast(llr_max, self.rdtype)

        if v2c_callbacks is None:
            self._v2c_callbacks = []
        else:
            if isinstance(v2c_callbacks, (list, tuple)):
                self._v2c_callbacks = v2c_callbacks
            elif isinstance(v2c_callbacks, types.FunctionType):
                # allow that user provides single function
                self._v2c_callbacks = [v2c_callbacks,]
            else:
                raise TypeError("v2c_callbacks must be a list of callables.")

        if c2v_callbacks is None:
            self._c2v_callbacks = []
        else:
            if isinstance(c2v_callbacks, (list, tuple)):
                self._c2v_callbacks = c2v_callbacks
            elif isinstance(c2v_callbacks, types.FunctionType):
                # allow that user provides single function
                self._c2v_callbacks = [c2v_callbacks,]
            else:
                raise TypeError("c2v_callbacks must be a list of callables.")

        if isinstance(cn_schedule, str) and cn_schedule=="flooding":
            self._scheduling = "flooding"
            self._cn_schedule = tf.stack([tf.range(self._num_cns)], axis=0)
        elif tf.is_tensor(cn_schedule) or isinstance(cn_schedule, np.ndarray):
            cn_schedule = tf.cast(cn_schedule, tf.int32)
            self._scheduling = "custom"
            # check custom schedule for consistency
            if len(cn_schedule.shape)!=2:
                raise ValueError("cn_schedule must be of rank 2.")
            if tf.reduce_max(cn_schedule)>=self._num_cns:
                msg = "cn_schedule can only contain values smaller number_cns."
                raise ValueError(msg)
            if tf.reduce_min(cn_schedule)<0:
                msg = "cn_schedule cannot contain negative values."
                raise ValueError(msg)
            self._cn_schedule = cn_schedule
        else:
            msg = "cn_schedule can be 'flooding' or an array of ints."
            raise ValueError(msg)

        ######################
        # Init graph structure
        ######################

        # make pcm sparse first if ndarray is provided
        if isinstance(pcm, np.ndarray):
            pcm = sp.sparse.csr_matrix(pcm)

        # Assign all edges to CN and VN nodes, respectively
        self._cn_idx, self._vn_idx, _ = sp.sparse.find(pcm)

        # sort indices explicitly, as scipy.sparse.find changed from column to
        # row sorting in scipy>=1.11
        idx = np.argsort(self._vn_idx)
        self._cn_idx = self._cn_idx[idx]
        self._vn_idx = self._vn_idx[idx]

        # number of edges equals number of non-zero elements in the
        # parity-check matrix
        self._num_edges = len(self._vn_idx)

        # pre-load the CN function
        if cn_update=='boxplus':
            # check node update using the tanh function
            self._cn_update = cn_update_tanh
        elif cn_update=='boxplus-phi':
            # check node update using the "_phi" function
            self._cn_update = cn_update_phi
        elif cn_update in ('minsum', 'min'):
            # check node update using the min-sum approximation
            self._cn_update = cn_update_minsum
        elif cn_update=="offset-minsum":
            # check node update using the min-sum approximation
            self._cn_update = cn_update_offset_minsum
        elif cn_update=='identity':
            self._cn_update = cn_node_update_identity
        elif isinstance(cn_update, types.FunctionType):
            self._cn_update = cn_update
        else:
            raise TypeError("Provided cn_update not supported.")

        # pre-load the VN function
        if vn_update=='sum':
            self._vn_update = vn_update_sum
        elif vn_update=='identity':
            self._vn_update = vn_node_update_identity
        elif isinstance(vn_update, types.FunctionType):
            self._vn_update = vn_update
        else:
            raise TypeError("Provided vn_update not supported.")

        ######################
        # init graph structure
        ######################

        # Permutation index to rearrange edge messages into CN perspective
        v2c_perm = np.argsort(self._cn_idx)
        # and the inverse operation;
        v2c_perm_inv = np.argsort(v2c_perm)
        # only required for layered decoding
        self._v2c_perm_inv = tf.constant(v2c_perm_inv)

        # Initialize a ragged tensor that allows to gather
        # from the v2c messages (from VN perspective) and returns
        # a ragged tensor of incoming messages of each CN.
        # This needs to be ragged as the CN degree can be irregular.
        self._v2c_perm = tf.RaggedTensor.from_value_rowids(
                                values=v2c_perm,
                                value_rowids=self._cn_idx[v2c_perm])

        self._c2v_perm = tf.RaggedTensor.from_value_rowids(
                                values=v2c_perm_inv,
                                value_rowids=self._vn_idx)

    ###############################
    # Public methods and properties
    ###############################

    @property
    def pcm(self):
        """Parity-check matrix of LDPC code"""
        return self._pcm

    @property
    def num_cns(self):
        """Number of check nodes"""
        return self._num_cns

    @property
    def num_vns(self):
        """Number of variable nodes"""
        return self._num_vns

    @property
    def n(self):
        """codeword length"""
        return self._num_vns

    @property
    def coderate(self):
        """codrate assuming independent parity checks"""
        return (self._num_vns - self._num_cns) / self._num_vns

    @property
    def num_edges(self):
        """Number of edges in decoding graph"""
        return self._num_edges

    @property
    def num_iter(self):
        "Number of decoding iterations"
        return self._num_iter

    @num_iter.setter
    def num_iter(self, num_iter):
        "Number of decoding iterations"
        if not isinstance(num_iter, int):
            raise TypeError('num_iter must be int.')
        if num_iter<0:
            raise ValueError('num_iter cannot be negative.')
        self._num_iter = tf.constant(num_iter, dtype=tf.int32)

    @property
    def llr_max(self):
        """Max LLR value used for internal calculations and rate-matching"""
        return self._llr_max

    @llr_max.setter
    def llr_max(self, value):
        """Max LLR value used for internal calculations"""
        if value<0:
            raise ValueError('llr_max cannot be negative.')
        self._llr_max = tf.cast(value, dtype=self.rdtype)

    @property
    def return_state(self):
        """Return internal decoder state for IDD schemes"""
        return self._return_state

    #########################
    # Decoding functions
    #########################

    def _bp_iter(self, msg_v2c, msg_c2v, llr_ch, x_hat, it, num_iter):
        """Main decoding loop

        Parameters
        ----------
        msg_v2c: [num_edges, batch_size], tf.float
            Tensor of VN messages representing the internal decoder state.

        msg_c2v: [num_edges, batch_size], tf.float
            Tensor of CN messages representing the internal decoder state.

        llr_ch: [...,n], tf.float
            Tensor containing the channel logits/llr values.

        x_hat : [...,n] or [...,k], tf.float
            Tensor of same shape as ``llr_ch`` containing bit-wise
            soft-estimates of all `n` codeword bits.

        it: tf.int
            Current iteration number

        num_iter: int
            Total number of decoding iterations

        Returns
        -------
        msg_v2c: [num_edges, batch_size], tf.float
            Tensor of VN messages representing the internal decoder state.

        msg_c2v: [num_edges, batch_size], tf.float
            Tensor of CN messages representing the internal decoder state.

        llr_ch: [...,n], tf.float
            Tensor containing the channel logits/llr values.

        x_hat : [...,n] or [...,k], tf.float
            Tensor of same shape as ``llr_ch`` containing bit-wise
            soft-estimates of all `n` codeword bits.

        it: tf.int
            Current iteration number

        num_iter: int
            Total number of decoding iterations
        """

        # Unroll loop to keep XLA / Keras compatibility
        # For flooding this will be unrolled to a single loop iteration
        for j in range(self._cn_schedule.shape[0]):

            # get active check nodes
            if self._scheduling=="flooding":
                # for flooding all CNs are active
                v2c_perm = self._v2c_perm
            else: # select active CNs for j-th subiteration
                cn_idx = tf.gather(self._cn_schedule, j, axis=0)
                v2c_perm = tf.gather(self._v2c_perm, cn_idx, axis=0)

            # Gather ragged tensor of incoming messages at CN.
            # The shape is [num_cns, None, batch_size,...].
            # The None dimension is the ragged dimension and depends on the
            # individual check node degree

            msg_cn_rag = tf.gather(msg_v2c, v2c_perm, axis=0)

            # Apply the CN update
            msg_cn_rag_ = self._cn_update(msg_cn_rag, self.llr_max)

            # Apply CN callbacks
            for cb in self._c2v_callbacks:
                msg_cn_rag_ = cb(msg_cn_rag_, it)

            # Apply partial message updates for layered decoding
            if self._scheduling!="flooding":
                # note: the scatter update operation is quite expensive
                up_idx = tf.gather(self._c2v_perm.flat_values,
                                   v2c_perm.flat_values)
                # update only active cns are updated
                msg_c2v = tf.tensor_scatter_nd_update(
                                     msg_c2v,
                                     tf.expand_dims(up_idx, axis=1),
                                     msg_cn_rag_.flat_values)
            else:
                # for flodding all nodes are updated
                msg_c2v = msg_cn_rag_.flat_values

            # Gather ragged tensor of incoming messages at VN.
            # Note for performance reasons this includes the re-permute
            # of edges from CN to VN perspective.
            # The shape is [num_vns, None, batch_size,...].
            msg_vn_rag = tf.gather(msg_c2v, self._c2v_perm, axis=0)

            # Apply the VN update
            msg_vn_rag_, x_hat = self._vn_update(msg_vn_rag,
                                                 llr_ch,
                                                 self.llr_max)

            # apply v2c callbacks
            for cb in self._v2c_callbacks:
                msg_vn_rag_ = cb(msg_vn_rag_, it+1, x_hat)

            # we return flat values to avoid ragged tensors passing the tf.
            # while boundary (possible issues with XLA)
            msg_v2c = msg_vn_rag_.flat_values

        #increase iteration coutner
        it += 1

        return msg_v2c, msg_c2v, llr_ch, x_hat, it, num_iter

    # pylint: disable=unused-argument,unused-variable
    def _stop_cond(self, msg_v2c, msg_c2v, llr_ch, x_hat, it, num_iter):
        """stops decoding loop after num_iter iterations.
        Most inputs are ignored, just for compatibility with tf.while.
        """
        return it < num_iter

    #########################
    # Sionna Block functions
    #########################

    # pylint: disable=(unused-argument)
    def build(self, input_shape, **kwargs):
        # Raise AssertionError if shape of x is invalid

        assert (input_shape[-1]==self._num_vns), \
                            'Last dimension must be of length n.'

    def call(self, llr_ch, /, *, num_iter=None, msg_v2c=None):
        """Iterative BP decoding function.
        """

        if num_iter is None:
            num_iter=self.num_iter

        # clip LLRs for numerical stability
        llr_ch = tf.clip_by_value(llr_ch,
                                  clip_value_min=-self._llr_max,
                                  clip_value_max=self._llr_max)

        # reshape to support multi-dimensional inputs
        llr_ch_shape = llr_ch.get_shape().as_list()
        new_shape = [-1, self._num_vns]
        llr_ch_reshaped = tf.reshape(llr_ch, new_shape)

        # batch dimension is last dimension due to ragged tensor representation
        llr_ch = tf.transpose(llr_ch_reshaped, (1, 0))

        # logits are converted into "true" LLRs as usually done in literature
        llr_ch *= -1.

        # If no initial decoder state is provided, we initialize it with 0.
        # This is relevant for IDD schemes.
        if msg_v2c is None:
            # init v2c messages with channel LLRs
            msg_v2c = tf.gather(llr_ch, self._vn_idx)
        else:
            msg_v2c *= -1 # invert sign due to logit definition

        # msg_v2c is of shape [num_edges, batch_size]
        # it contains all edge message from VN to CN
        # Hereby, self._vn_idx indicates the index of the associated VN
        # and self._cn_idx the index of the associated CN

        # messages from CN perspective; are inititalized to zero
        msg_c2v = tf.zeros_like(msg_v2c)

        # apply VN callbacks before first iteration
        if self._v2c_callbacks != []:
            msg_vn_rag_ = tf.RaggedTensor.from_value_rowids(
                                values=msg_v2c,
                                value_rowids=self._vn_idx)
            # apply v2c callbacks
            for cb in self._v2c_callbacks:
                msg_vn_rag_ = cb(msg_vn_rag_, tf.constant(0, tf.int32), llr_ch)

            # Ensure shape as otherwise XLA cannot infer
            # the output signature of the loop
            msg_v2c = msg_vn_rag_.flat_values

        #####################
        # Main decoding loop
        #####################

        # msg_v2c : decoder state (from vN perspective)
        # msg_c2v : decoder state (from CN perspective)
        # llr_ch : channel llrs
        # llr_ch:  x_hat; automatically returns llr_ch for 0 iterations
        # tf.constant(0, tf.int32) : iteration counter
        # num_iter : total number of iterations

        inputs = (msg_v2c, msg_c2v, llr_ch, llr_ch,
                  tf.constant(0, tf.int32), num_iter)

        # and run main decoding loop for num_iter iterations
        msg_v2c, _, _, x_hat, _, _ = tf.while_loop(
                                        self._stop_cond,self._bp_iter,
                                        inputs, maximum_iterations=num_iter)

        ######################
        # Post process outputs
        ######################

        # restore batch dimension to first dimension
        x_hat = tf.transpose(x_hat, (1,0))

        if self._hard_out: # hard decide decoder output if required
            x_hat = tf.greater_equal(tf.cast(0, self.rdtype), x_hat)
            x_hat = tf.cast(x_hat, self.rdtype)
        else:
            x_hat *= -1.  # convert LLRs back into logits

        # Reshape c_short so that it matches the original input dimensions
        output_shape = llr_ch_shape
        output_shape[0] = -1 # Dynamic batch dim
        x_reshaped = tf.reshape(x_hat, output_shape)

        if not self._return_state:
            return x_reshaped
        else:
            msg_v2c *= -1 # invert sign due to logit definition
            return x_reshaped, msg_v2c

#######################
# Node update functions
#######################

# pylint: disable=unused-argument,unused-variable
def vn_node_update_identity(msg_c2v_rag, llr_ch, llr_clipping=None, **kwargs):
    # pylint: disable=line-too-long
    r"""Dummy variable node update function for testing.

    Behaves as an identity function and can be used for testing an debugging of
    message passing decoding.

    Marginalizes input messages and returns them as second output.

    Parameters
    ----------
    msg_c2v_rag: [num_edges, batch_size], tf.ragged
        Ragged tensor of shape `[num_nodes, None, batch_size]` where the second
        axis is ragged (represents individual node degrees).
        Represents c2v messages.

    llr_ch: [num_nodes, batch_size], tf.float
        Tensor containing the channel LLRs.

    llr_clipping: `None` (default) | float
        Clipping value used for internal processing. If `None`, no internal
        clipping is applied.

    Returns
    -------
    msg_v2c_rag : tf.ragged
        Updated v2c messages. Ragged tensor of same shape as ``msg_c2v``

    x_tot: tf.float
        Mariginalized LLRs per variable node of shape `[num_nodes, batch_size]`.
        Can be used as final estimate per VN.
    """
    # aggregate all incoming messages per node
    x_tot = tf.reduce_sum(msg_c2v_rag, axis=1) + llr_ch

    return msg_c2v_rag, x_tot

def vn_update_sum(msg_c2v_rag, llr_ch, llr_clipping=None):
    # pylint: disable=line-too-long
    r"""Variable node update function implementing the `sum` update.

    This function implements the (extrinsic) variable node update
    function. It takes the sum over all incoming messages ``msg`` excluding
    the intrinsic (= outgoing) message itself.

    Additionally, the channel LLR ``llr_ch`` is considered in each variable
    node.

    Parameters
    ----------
    msg_c2v_rag: [num_nodes, None, batch_size], tf.ragged
        Ragged tensor of shape `[num_nodes, None, batch_size]` where the second
        axis is ragged (represents individual node degrees).
        Represents c2v messages.

    llr_ch: [num_nodes, batch_size], tf.float
        Tensor containing the channel LLRs.

    llr_clipping: `None` (default) | float
        Clipping value used for internal processing. If `None`, no internal
        clipping is applied.

    Returns
    -------
    msg_v2c_rag : tf.ragged
        Updated v2c messages. Ragged tensor of same shape as ``msg_c2v``

    x_tot: tf.float
        Mariginalized LLRs per variable node of shape `[num_nodes, batch_size]`.
    """
    # aggregate all incoming messages per node
    x = tf.reduce_sum(msg_c2v_rag, axis=1)
    x_tot = tf.add(x, llr_ch)

    # TF2.9 does not support XLA for the addition of ragged tensors
    # the following code provides a workaround that supports XLA

    # subtract extrinsic message from node value
    #x_e = tf.expand_dims(x_tot, axis=1)
    #x_e = tf.add(-msg_c2v, x_e)
    x_e = tf.ragged.map_flat_values(lambda x,y,row_ind: x+tf.gather(y, row_ind),
                            -1.*msg_c2v_rag, x_tot, msg_c2v_rag.value_rowids())

    if llr_clipping is not None:
        x_e = tf.clip_by_value(x_e,
                    clip_value_min=-llr_clipping, clip_value_max=llr_clipping)
        x_tot = tf.clip_by_value(x_tot,
                    clip_value_min=-llr_clipping, clip_value_max=llr_clipping)
    return x_e, x_tot

# pylint: disable=unused-argument,unused-variable
def cn_node_update_identity(msg_v2c_rag, *kwargs):
    # pylint: disable=line-too-long
    r"""Dummy function that returns the first tensor without any processing.

    Used for testing an debugging of message passing decoding.

    Parameters
    ----------
    msg_v2c_rag: [num_nodes, None, batch_size], tf.ragged
        Ragged tensor of shape `[num_nodes, None, batch_size]` where the second
        axis is ragged (represents individual node degrees).
        Represents v2c messages

    Returns
    -------
    msg_c2v_rag : [num_nodes, None, batch_size], tf.ragged
        Updated v2c messages. Ragged tensor of same shape as ``msg_c2v``.
    """
    return msg_v2c_rag

def cn_update_offset_minsum(msg_v2c_rag, llr_clipping=None, offset=0.5):
    # pylint: disable=line-too-long
    r"""Check node update function implementing the offset corrected minsum.

    The function implements

    .. math::
            \qquad y_{j \to i} = \alpha_{j \to i} \cdot {max} \left( {min}_{i' \in \mathcal{N}(j) \setminus i} \left(|x_{i' \to j}| \right)-\beta , 0\right)

    where :math:`\beta=0.5` and :math:`y_{j \to i}` denotes the message from
    check node (CN) *j* to variable node (VN) *i* and :math:`x_{i \to j}` from
    VN *i* to CN *j*, respectively. Further, :math:`\mathcal{N}(j)` denotes
    all indices of connected VNs to CN *j* and

    .. math::
        \alpha_{j \to i} = \prod_{i' \in \mathcal{N}(j) \setminus i} \operatorname{sign}(x_{i' \to j})

    is the sign of the outgoing message. For further details we refer to
    [Chen]_.

    Parameters
    ----------
    msg_v2c_rag: [num_nodes, None, batch_size], tf.ragged
        Ragged tensor of shape `[num_nodes, None, batch_size]` where the second
        axis is ragged (represents individual node degrees).
        Represents v2c messages.

    llr_clipping: `None` (default) | float
        Clipping value used for internal processing. If `None`, no internal
        clipping is applied.

    offset: float (default `0.5`)
        Offset value to be subtracted from each outgoing message.

    Returns
    -------
    msg_c2v : [num_nodes, None, batch_size], tf.ragged
        Ragged tensor of same shape as ``msg_c2v`` containing the updated c2v
        messages.
    """

    def _sign_val_minsum(msg):
        """Helper to replace find sign-value during min-sum decoding.
        Must be called with `map_flat_values`."""

        sign_val = tf.sign(msg)
        sign_val = tf.where(tf.equal(sign_val, 0),
                            tf.ones_like(sign_val),
                            sign_val)
        return sign_val

    # a constant used to overwrite the first min
    large_val = 100000.
    msg_v2c_rag = tf.clip_by_value(msg_v2c_rag,
                               clip_value_min=-large_val,
                               clip_value_max=large_val)

    # only output is clipped (we assume input was clipped by previous function)

    # calculate sign of outgoing msg and the node
    sign_val = tf.ragged.map_flat_values(_sign_val_minsum, msg_v2c_rag)
    sign_node = tf.reduce_prod(sign_val, axis=1)

    # TF2.9 does not support XLA for the multiplication of ragged tensors
    # the following code provides a workaround that supports XLA

    # sign_val = self._stop_ragged_gradient(sign_val) \
    #             * tf.expand_dims(sign_node, axis=1)
    sign_val = tf.ragged.map_flat_values(
                                    lambda x, y, row_ind:
                                    tf.multiply(x, tf.gather(y, row_ind)),
                                    sign_val,
                                    sign_node,
                                    sign_val.value_rowids())

    # remove sign from messages
    msg = tf.ragged.map_flat_values(tf.abs, msg_v2c_rag)

    # Calculate the extrinsic minimum per CN, i.e., for each message of
    # index i, find the smallest and the second smallest value.
    # However, in some cases the second smallest value may equal the
    # smallest value (multiplicity of mins).
    # Please note that this needs to be applied to raggedTensors, e.g.,
    # tf.top_k() is currently not supported and all ops must support graph
    # and XLA mode.

    # find min_value per node
    min_val = tf.reduce_min(msg, axis=1, keepdims=True)

    # TF2.9 does not support XLA for the subtraction of ragged tensors
    # the following code provides a workaround that supports XLA

    # and subtract min; the new array contains zero at the min positions
    # benefits from broadcasting; all other values are positive
    msg_min1 = tf.ragged.map_flat_values(lambda x, y, row_ind:
                                            x - tf.gather(y, row_ind),
                                            msg,
                                            tf.squeeze(min_val, axis=1),
                                            msg.value_rowids())

    # replace 0 (=min positions) with large value to ignore it for further
    # min calculations
    msg = tf.ragged.map_flat_values(
                        lambda x: tf.where(tf.equal(x, 0), large_val, x),
                        msg_min1)

    # find the second smallest element (we add min_val as this has been
    # subtracted before)
    min_val_2 = tf.reduce_min(msg, axis=1, keepdims=True) + min_val

    # Detect duplicated minima (i.e., min_val occurs at two incoming
    # messages). As the LLRs per node are <LLR_MAX and we have
    # replace at least 1 position (position with message "min_val") by
    # large_val, it holds for the sum < large_val + node_degree*LLR_MAX.
    # If the sum > 2*large_val, the multiplicity of the min is at least 2.
    node_sum = tf.reduce_sum(msg, axis=1, keepdims=True) - (2*large_val-1.)
    # indicator that duplicated min was detected (per node)
    double_min = 0.5*(1-tf.sign(node_sum))

    # if a duplicate min occurred, both edges must have min_val, otherwise
    # the second smallest value is taken
    min_val_e = (1-double_min) * min_val + (double_min) * min_val_2

    # replace all values with min_val except the position where the min
    # occurred (=extrinsic min).

    # no XLA support for TF 2.15
    # msg_e = tf.where(msg==large_val, min_val_e, min_val)

    min_1 = tf.squeeze(tf.gather(min_val, msg.value_rowids()), axis=1)
    min_e = tf.squeeze(tf.gather(min_val_e, msg.value_rowids()), axis=1)
    msg_e = tf.ragged.map_flat_values(
                lambda x: tf.where(x==large_val, min_e, min_1), msg)

    # it seems like tf.where does not set the shape of tf.ragged properly
    # we need to ensure the shape manually
    msg_e = tf.ragged.map_flat_values(
                lambda x: tf.ensure_shape(x, msg.flat_values.shape), msg_e)

    # apply offset
    msg_e = tf.ragged.map_flat_values(lambda x,y: tf.maximum(x-y, 0),
                                      msg_e, offset)

    # TF2.9 does not support XLA for the multiplication of ragged tensors
    # the following code provides a workaround that supports XLA

    # and apply sign
    #msg = sign_val * msg_e
    msg = tf.ragged.map_flat_values(tf.multiply, sign_val, msg_e)

    # clip output values if required
    if llr_clipping is not None:
        msg = tf.clip_by_value(msg,
                    clip_value_min=-llr_clipping, clip_value_max=llr_clipping)
    return msg

def cn_update_minsum(msg_v2c_rag, llr_clipping=None):
    # pylint: disable=line-too-long
    r"""Check node update function implementing the `minsum` update.

    The function implements

    .. math::
            \qquad y_{j \to i} = \alpha_{j \to i} \cdot {min}_{i' \in \mathcal{N}(j) \setminus i} \left(|x_{i' \to j}|\right)

    where :math:`y_{j \to i}` denotes the message from check node (CN) *j* to
    variable node (VN) *i* and :math:`x_{i \to j}` from VN *i* to CN *j*,
    respectively. Further, :math:`\mathcal{N}(j)` denotes all indices of
    connected VNs to CN *j* and

    .. math::
        \alpha_{j \to i} = \prod_{i' \in \mathcal{N}(j) \setminus i} \operatorname{sign}(x_{i' \to j})

    is the sign of the outgoing message. For further details we refer to
    [Ryan]_ and [Chen]_.

    Parameters
    ----------
    msg_v2c_rag: [num_nodes, None, batch_size], tf.ragged
        Ragged tensor of shape `[num_nodes, None, batch_size]` where the second
        axis is ragged (represents individual node degrees).
        Represents v2c messages

    llr_clipping: `None` (default) | float
        Clipping value used for internal processing. If `None`, no internal
        clipping is applied.

    Returns
    -------
    msg_c2v : [num_nodes, None, batch_size], tf.ragged
        Ragged tensor of same shape as ``msg_c2v`` containing the updated c2v
        messages.
    """

    msg_c2v = cn_update_offset_minsum(msg_v2c_rag,
                                      llr_clipping=llr_clipping,
                                      offset=0)

    return msg_c2v

def cn_update_tanh(msg, llr_clipping=None):
    # pylint: disable=line-too-long
    r"""Check node update function implementing the `boxplus` operation.

    This function implements the (extrinsic) check node update
    function. It calculates the boxplus function over all incoming messages
    "msg" excluding the intrinsic (=outgoing) message itself.
    The exact boxplus function is implemented by using the tanh function.

    The function implements

    .. math::
            y_{j \to i} = 2 \operatorname{tanh}^{-1} \left( \prod_{i' \in \mathcal{N}(j) \setminus i} \operatorname{tanh} \left( \frac{x_{i' \to j}}{2} \right) \right)

    where :math:`y_{j \to i}` denotes the message from check node (CN) *j* to
    variable node (VN) *i* and :math:`x_{i \to j}` from VN *i* to CN *j*,
    respectively. Further, :math:`\mathcal{N}(j)` denotes all indices of
    connected VNs to CN *j* and

    .. math::
        \alpha_{j \to i} = \prod_{i' \in \mathcal{N}(j) \setminus i} \operatorname{sign}(x_{i' \to j})

    is the sign of the outgoing message. For further details we refer to
    [Ryan]_.

    Note that for numerical stability clipping can be applied.

    Parameters
    ----------
    msg_v2c_rag: [num_nodes, None, batch_size], tf.ragged
        Ragged tensor of shape `[num_nodes, None, batch_size]` where the second
        axis is ragged (represents individual node degrees).
        Represents v2c messages

    llr_clipping: `None` (default) | float
        Clipping value used for internal processing. If `None`, no internal
        clipping is applied.

    Returns
    -------
    msg_c2v : [num_nodes, None, batch_size], tf.ragged
        Ragged tensor of same shape as ``msg_c2v`` containing the updated c2v
        messages.
    """

    # clipping value for the atanh function is applied (tf.float32 is used)
    atanh_clip_value = 1 - 1e-7

    msg = msg / 2
    # tanh is not overloaded for ragged tensors
    msg = tf.ragged.map_flat_values(tf.tanh, msg) # tanh is not overloaded

    # for ragged tensors; map to flat tensor first
    msg = tf.ragged.map_flat_values(
            lambda x: tf.where(tf.equal(x, 0), tf.ones_like(x) * 1e-12, x), msg)

    msg_prod = tf.reduce_prod(msg, axis=1)

    # TF2.9 does not support XLA for the multiplication of ragged tensors
    # the following code provides a workaround that supports XLA

    # ^-1 to avoid division
    # Note this is (potentially) numerically unstable
    # msg = msg**-1 * tf.expand_dims(msg_prod, axis=1) # remove own edge

    msg = tf.ragged.map_flat_values(
                        lambda x,y,row_ind : x * tf.gather(y, row_ind),
                        msg**-1, msg_prod, msg.value_rowids())

    # Overwrite small (numerical zeros) message values with exact zero
    # these are introduced by the previous "_where_ragged" operation
    # this is required to keep the product stable (cf. _phi_update for log
    # sum implementation)
    msg = tf.ragged.map_flat_values(
        lambda x: tf.where(tf.less(tf.abs(x), 1e-7), tf.zeros_like(x), x), msg)

    msg = tf.clip_by_value(msg,
                           clip_value_min=-atanh_clip_value,
                           clip_value_max=atanh_clip_value)

    # atanh is not overloaded for ragged tensors
    msg = 2 * tf.ragged.map_flat_values(tf.atanh, msg)

    # clip output values if required
    if llr_clipping is not None:
        msg = tf.clip_by_value(msg,
                               clip_value_min=-llr_clipping,
                               clip_value_max=llr_clipping)
    return msg

def cn_update_phi(msg, llr_clipping=None):
    # pylint: disable=line-too-long
    r"""Check node update function implementing the `boxplus` operation.

    This function implements the (extrinsic) check node update function
    based on the numerically more stable `"_phi"` function (cf. [Ryan]_).
    It calculates the boxplus function over all incoming messages ``msg``
    excluding the intrinsic (=outgoing) message itself.
    The exact boxplus function is implemented by using the `"_phi"` function
    as in [Ryan]_.

    The function implements

    .. math::
            y_{j \to i} = \alpha_{j \to i} \cdot \phi \left( \sum_{i' \in \mathcal{N}(j) \setminus i} \phi \left( |x_{i' \to j}|\right) \right)

    where :math:`\phi(x)=-\operatorname{log}(\operatorname{tanh} \left(\frac{x} {2}) \right)`
    and :math:`y_{j \to i}` denotes the message from check node
    (CN) *j* to variable node (VN) *i* and :math:`x_{i \to j}` from VN *i* to
    CN *j*, respectively. Further, :math:`\mathcal{N}(j)` denotes all indices
    of connected VNs to CN *j* and

    .. math::
        \alpha_{j \to i} = \prod_{i' \in \mathcal{N}(j) \setminus i} \operatorname{sign}(x_{i' \to j})

    is the sign of the outgoing message. For further details we refer to
    [Ryan]_.

    Note that for numerical stability clipping can be applied.

    Parameters
    ----------
    msg_v2c_rag: [num_nodes, None, batch_size], tf.ragged
        Ragged tensor of shape `[num_nodes, None, batch_size]` where the second
        axis is ragged (represents individual node degrees).
        Represents v2c messages

    llr_clipping: `None` (default) | float
        Clipping value used for internal processing. If `None`, no internal
        clipping is applied.

    Returns
    -------
    msg_c2v : [num_nodes, None, batch_size], tf.ragged
        Ragged tensor of same shape as ``msg_c2v`` containing the updated c2v
        messages.
    """
    def _phi(x):
        # pylint: disable=line-too-long
        r"""Utility function for the boxplus-phi check node update.

        This function implements the (element-wise) `"phi"` function as defined
        in [Ryan]_  :math:`\phi(x)=-\operatorname{log}(\operatorname{tanh} \left(\frac{x}{2}) \right)`.

        Parameters
        ----------
        x : tf.float
            Input tensor of arbitrary shape.

        Returns
        -------
        : tf.float
            Tensor of same shape and dtype as ``x``.

        """
        if x.dtype==tf.float32:
            # the clipping values are optimized for tf.float32
            x = tf.clip_by_value(x,
                    clip_value_min=8.5e-8, clip_value_max=16.635532)
        elif x.dtype==tf.float64:
            x = tf.clip_by_value(x,
                    clip_value_min=1e-12, clip_value_max=28.324079)
        else:
            raise TypeError("Unsupported dtype for phi function.")

        return tf.math.log(tf.math.exp(x)+1) - tf.math.log(tf.math.exp(x)-1)

    ##################
    # Sign of messages
    ##################

    sign_val = tf.sign(msg)
    # TF2.14 does not support XLA for tf.where
    # the following code provides a workaround that supports XLA
    sign_val = tf.ragged.map_flat_values(lambda x : tf.where(tf.equal(x, 0),
                                         tf.ones_like(x), x), sign_val)
    # calculate sign of entire node
    sign_node = tf.reduce_prod(sign_val, axis=1)

    # TF2.9 does not support XLA for the multiplication of ragged tensors
    # the following code provides a workaround that supports XLA
    #sign_val = sign_val * tf.expand_dims(sign_node, axis=1)
    sign_val = tf.ragged.map_flat_values(
                lambda x,y,row_ind : x * tf.gather(y, row_ind),
                sign_val, sign_node, sign_val.value_rowids())

    ###################
    # Value of messages
    ###################
    msg = tf.ragged.map_flat_values(tf.abs, msg) # remove sign

    # apply _phi element-wise
    msg = tf.ragged.map_flat_values(_phi, msg)

    # sum over entire node
    msg_sum = tf.reduce_sum(msg, axis=1)

    # TF2.9 does not support XLA for the addition of ragged tensors
    # the following code provides a workaround that supports XLA
    #msg = tf.add( -msg, tf.expand_dims(msg_sum, axis=1)) # remove own edge
    msg = tf.ragged.map_flat_values(
                            lambda x, y, row_ind : x + tf.gather(y, row_ind),
                            -1.*msg, msg_sum, msg.value_rowids())

    # apply _phi element-wise (does not support ragged Tensors)
    sign_val = sign_val.with_flat_values(tf.stop_gradient(sign_val.flat_values))
    msg_e = sign_val * tf.ragged.map_flat_values(_phi, msg)

    if llr_clipping is not None:
        msg_e = tf.clip_by_value(msg_e,
                    clip_value_min=-llr_clipping, clip_value_max=llr_clipping)
    return msg_e


class LDPC5GDecoder(LDPCBPDecoder):
    # pylint: disable=line-too-long
    r"""Iterative belief propagation decoder for 5G NR LDPC codes.

    Inherits from :class:`~sionna.phy.fec.ldpc.decoding.LDPCBPDecoder` and
    provides a wrapper for 5G compatibility, i.e., automatically handles
    rate-matching according to [3GPPTS38212_LDPC]_.

    Note that for full 5G 3GPP NR compatibility, the correct puncturing and
    shortening patterns must be applied and, thus, the encoder object is
    required as input.

    If required the decoder can be made trainable and is differentiable
    (the training of some check node types may be not supported) following the
    concept of "weighted BP" [Nachmani]_.

    Parameters
    ----------
    encoder: LDPC5GEncoder
        An instance of :class:`~sionna.phy.fec.ldpc.encoding.LDPC5GEncoder`
        containing the correct code parameters.

    cn_update: `str`, "boxplus-phi" (default) | "boxplus" | "minsum" | "offset-minsum" | "identity" | callable
        Check node update rule to be used as described above.
        If a callable is provided, it will be used instead as CN update.
        The input of the function is a ragged tensor of v2c messages of shape
        `[num_cns, None, batch_size]` where the second dimension is ragged
        (i.e., depends on the individual CN degree).

    vn_update: `str`, "sum" (default) | "identity" | callable
        Variable node update rule to be used.
        If a callable is provided, it will be used instead as VN update.
        The input of the function is a ragged tensor of c2v messages of shape
        `[num_vns, None, batch_size]` where the second dimension is ragged
        (i.e., depends on the individual VN degree).

    cn_schedule: "flooding" | "layered" | [num_update_steps, num_active_nodes], tf.int
        Defines the CN update scheduling per BP iteration. Can be either
        "flooding" to update all nodes in parallel (recommended) or "layered"
        to sequentally update all CNs in the same lifting group together or an
        2D tensor where each row defines the `num_active_nodes` node indices to
        be updated per subiteration. In this case each BP iteration runs
        `num_update_steps` subiterations, thus the decoder's level of
        parallelization is lower and usually the decoding throughput decreases.

    hard_out: `bool`, (default `True`)
        If `True`,  the decoder provides hard-decided codeword bits instead of
        soft-values.

    return_infobits: `bool`, (default `True`)
        If `True`, only the `k` info bits (soft or hard-decided) are returned.
        Otherwise all `n` positions are returned.

    prune_pcm: `bool`, (default `True`)
        If `True`, all punctured degree-1 VNs and connected check nodes are
        removed from the decoding graph (see [Cammerer]_ for details). Besides
        numerical differences, this should yield the same decoding result but
        improved the decoding throughput and reduces the memory footprint.
        In HARQ mode, pruning is automatically disabled.

    num_iter: `int` (default: 20)
        Defining the number of decoder iterations (due to batching, no early
        stopping used at the moment!).

    llr_max: `float` (default: 20) | `None`
        Internal clipping value for all internal messages. If `None`, no
        clipping is applied.

    v2c_callbacks, `None` (default) | list of callables
        Each callable will be executed after each VN update with the following
        arguments `msg_vn_rag_`, `it`, `x_hat`,where `msg_vn_rag_` are the v2c
        messages as ragged tensor of shape `[num_vns, None, batch_size]`,
        `x_hat` is the current estimate of each VN of shape
        `[num_vns, batch_size]` and `it` is the current iteration counter.
        It must return and updated version of `msg_vn_rag_` of same shape.

    c2v_callbacks: `None` (default) | list of callables
        Each callable will be executed after each CN update with the following
        arguments `msg_cn_rag_` and `it` where `msg_cn_rag_` are the c2v
        messages as ragged tensor of shape `[num_cns, None, batch_size]` and
        `it` is the current iteration counter.
        It must return and updated version of `msg_cn_rag_` of same shape.

    return_state: `bool`, (default `False`)
        If `True`,  the internal VN messages ``msg_vn`` from the last decoding
        iteration are returned, and ``msg_vn`` or `None` needs to be given as a
        second input when calling the decoder.
        This can be used for iterative demapping and decoding.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`, :py:attr:`~sionna.phy.config.precision` is used.

    harq_mode: `bool`, (default `False`)
        If `True`, the decoder is used in HARQ mode (incremental
        redundancy) where successive LLR receptions are inserted in a circular
        buffer and accumulated. The circular buffer is of size [batch_size, n_cb].

    accumulator: `None` (default) | callable
        Function to accumulate LLRs from multiple transmissions. If `None`, uses
        simple addition. The callable should have signature:
        `accumulator(llr_accumulated, llr_new, transmission_idx)`
        where `llr_accumulated` is the accumulated LLRs so far (or None for first
        transmission), `llr_new` is the new LLRs to accumulate, and
        `transmission_idx` is the transmission index (0, 1, 2, ...).
        Should return the updated accumulated LLRs.

    Input
    -----
    llr_ch: [...,n] or [...,num_rv,n], tf.float
        Tensor containing the channel logits/llr values.
        If ``rv`` is not `None`, the expected shape is shape [..., num_rv, n] where `num_rv = len(rv)`.

    num_iter: `None` | `int`, (default: None)
        Number of decoding iterations to be performed. When None is given,
        the decoder uses the value set in the constructor (self._num_iter).

    msg_v2c: `None` | [num_edges, batch_size], tf.float
        Tensor of VN messages representing the internal decoder state.
        Required only if the decoder shall use its previous internal state, e.g.
        for iterative detection and decoding (IDD) schemes.

    rv: `None` (default) | list of str
        List of redundancy version strings to decode. Can contain
        any combination of `["rv0", "rv1", "rv2", "rv3"]` in any order,
        including repeats. If `None`, defaults to single RV0 decoding.
        When provided, the decoder accumulates LLRs from all RVs before decoding.

    Output
    ------
    : [...,n] or [...,k], tf.float
        Tensor of same shape as ``llr_ch`` containing
        bit-wise soft-estimates (or hard-decided bit-values) of all
        `n` codeword bits or only the `k` information bits if
        ``return_infobits`` is True.

    : [num_edges, batch_size], tf.float:
        Tensor of VN messages representing the internal decoder state.
        Returned only if ``return_state`` is set to `True`.
        Remark: always retruns entire decoder state, even if
        ``return_infobits`` is True.

    Note
    ----
    As decoding input logits :math:`\operatorname{log} \frac{p(x=1)}{p(x=0)}`
    are assumed for compatibility with the learning framework, but internally
    LLRs with definition :math:`\operatorname{log} \frac{p(x=0)}{p(x=1)}` are
    used.

    The decoder is not (particularly) optimized for Quasi-cyclic (QC) LDPC
    codes and, thus, supports arbitrary parity-check matrices.

    The decoder is implemented by using '"ragged Tensors"' [TF_ragged]_ to
    account for arbitrary node degrees. To avoid a performance degradation
    caused by a severe indexing overhead, the batch-dimension is shifted to
    the last dimension during decoding.

    """

    def __init__(self,
                 encoder,
                 cn_update="boxplus-phi",
                 vn_update="sum",
                 cn_schedule="flooding",
                 hard_out=True,
                 return_infobits=True,
                 num_iter=20,
                 llr_max=20.,
                 v2c_callbacks=None,
                 c2v_callbacks=None,
                 prune_pcm=True,
                 return_state=False,
                 precision=None,
                 harq_mode=False,
                 accumulator=None,
                 **kwargs):

        # needs the 5G Encoder to access all 5G parameters
        if not isinstance(encoder, LDPC5GEncoder):
            raise TypeError("encoder must be of class LDPC5GEncoder.")

        self._encoder = encoder
        pcm = encoder.pcm

        if not isinstance(harq_mode, bool):
            raise TypeError('harq_mode must be bool.')
        self._harq_mode = harq_mode
        if self._harq_mode:
            prune_pcm = False

        if not isinstance(return_infobits, bool):
            raise TypeError('return_info must be bool.')
        self._return_infobits = return_infobits

        if not isinstance(return_state, bool):
            raise TypeError('return_state must be bool.')
        self._return_state = return_state

        # Deprecation warning for cn_type
        if 'cn_type' in kwargs:
            raise TypeError("'cn_type' is deprecated; use 'cn_update' instead.")

        # prune punctured degree-1 VNs and connected CNs. A punctured
        # VN-1 node will always "send" llr=0 to the connected CN. Thus, this
        # CN will only send 0 messages to all other VNs, i.e., does not
        # contribute to the decoding process.
        if not isinstance(prune_pcm, bool):
            raise TypeError('prune_pcm must be bool.')
        self._prune_pcm = prune_pcm
        if prune_pcm:
            # find index of first position with only degree-1 VN
            dv = np.sum(pcm, axis=0) # VN degree
            last_pos = encoder._n_ldpc
            for idx in range(encoder._n_ldpc-1, 0, -1):
                if dv[0, idx]==1:
                    last_pos = idx
                else:
                    break
            # number of filler bits
            k_filler = self.encoder.k_ldpc - self.encoder.k

            # number of punctured bits
            nb_punc_bits = ((self.encoder.n_ldpc - k_filler)
                                     - self.encoder.n - 2*self.encoder.z)

            # if layered decoding is used, qunatized number of punctured bits
            # to a multiple of z; otherwise scheduling groups of Z CNs becomes
            # impossible
            if cn_schedule=="layered":
                nb_punc_bits = np.floor(nb_punc_bits/self.encoder.z) \
                             * self.encoder.z
                nb_punc_bits = int (nb_punc_bits) # cast to int

            # effective codeword length after pruning of vn-1 nodes
            self._n_pruned = np.max((last_pos, encoder._n_ldpc - nb_punc_bits))
            self._nb_pruned_nodes = encoder._n_ldpc - self._n_pruned

            # remove last CNs and VNs from pcm
            pcm = pcm[:-self._nb_pruned_nodes, :-self._nb_pruned_nodes]

            #check for consistency
            if self._nb_pruned_nodes<0:
                msg = "Internal error: number of pruned nodes must be positive."
                raise ArithmeticError(msg)
        else:
            # no pruning; same length as before
            self._nb_pruned_nodes = 0
            self._n_pruned = encoder._n_ldpc

        if cn_schedule=="layered":
            z = self._encoder.z
            num_blocks = int(pcm.shape[0]/z)
            cn_schedule = []
            for i in range(num_blocks):
                cn_schedule.append(np.arange(z) + i*z)
            cn_schedule = tf.stack(cn_schedule, axis=0)

        # Set up accumulator function
        if accumulator is None:
            self._accumulator = self._default_accumulator
        elif callable(accumulator):
            self._accumulator = accumulator
        else:
            raise TypeError("accumulator must be callable or None.")

        super().__init__(pcm,
                         cn_update=cn_update,
                         vn_update=vn_update,
                         cn_schedule=cn_schedule,
                         hard_out=hard_out,
                         num_iter=num_iter,
                         llr_max=llr_max,
                         v2c_callbacks=v2c_callbacks,
                         c2v_callbacks=c2v_callbacks,
                         return_state=return_state,
                         precision=precision,
                         **kwargs)

    ###############################
    # Public methods and properties
    ###############################

    @property
    def encoder(self):
        """LDPC Encoder used for rate-matching/recovery"""
        return self._encoder

    ########################
    # Sionna block functions
    ########################

    def _default_accumulator(self, llr_accumulated, llr_new, transmission_idx):
        """Default LLR accumulator: simple addition.

        Args:
            llr_accumulated: Previous accumulated LLRs or None for first transmission
            llr_new: New LLRs to accumulate
            transmission_idx: Index of current transmission (0, 1, 2, ...)

        Returns:
            Updated accumulated LLRs
        """
        if llr_accumulated is None:
            return llr_new
        else:
            return llr_accumulated + llr_new
    
    def build(self, input_shape, **kwargs):
        """Build block"""

        # check input dimensions for consistency
        if input_shape[-1]!=self.encoder.n:
            raise ValueError('Last dimension must be of length n.')

        self._old_shape_5g = input_shape

    def call(self, llr_ch, /, *, num_iter=None, msg_v2c=None, rv=None):
        """Iterative BP decoding function and rate matching.

        Args:
            llr_ch: tf.float
                Tensor containing the channel logits/llr values.
                - If not self._harq_mode: shape [..., n]
                - Else: shape [..., num_rv, n] where num_rv = len(rv)
            num_iter: int, optional
                Number of decoding iterations.
            msg_v2c: tf.Tensor, optional
                VN messages for decoder state.
            rv: list, optional
                List of redundancy version strings to decode. Can contain
                any combination of ["rv0", "rv1", "rv2", "rv3"] in any order.
                If None, assumes single RV0 decoding.
        """
        llr_ch_shape = llr_ch.get_shape().as_list()
        llr_ch_shape[0] = -1  # it can be None

        if rv is None or not self._harq_mode:
            rv = ["rv0"]
        else:
            self.encoder.validate_rv_list(rv)
            if tf.shape(llr_ch)[-2]!=len(rv_list):
                msg = "In HARQ mode, second last dimension of llr_ch"
                msg += "must equal len(rv)."
                raise ValueError(msg)
    
        k = self.encoder.k
        n = tf.shape(llr_ch)[-1]  # or self.encoder.n
        n_cb = self.encoder.n_cb  # circular buffer length (n_ldpc - k_filler)
        nb_pruned_nodes = self._nb_pruned_nodes  # 0 in HARQ mode
        num_rv = len(rv)  # 1 in non-HARQ mode

        new_shape = [-1, num_rv, n]
        llr_ch_reshaped = tf.reshape(llr_ch, new_shape)
        batch_size = tf.shape(llr_ch_reshaped)[0]

        rv_starts = self.encoder.get_rv_starts()
        start_pos = [rv_starts[rv_name] for rv_name in rv]

        llr_accumulated = None
        for rv_idx, _ in enumerate(rv):
            # extract LLRs for this RV: [batch_size, n] and squeeze the
            # RV dimension
            llr_rv = llr_ch_reshaped[:, rv_idx, :]

            # apply output interleaver inverse if needed
            # see Sec. 5.4.2.2 in 38.212
            if self.encoder.num_bits_per_symbol is not None:
                llr_rv = tf.gather(llr_rv, self.encoder.out_int_inv, axis=-1)

            # reconstruct full circular buffer for this RV
            # pad to circular buffer length (filler bits are added later).
            # nb_pruned_nodes is non-zero only in non-HARQ mode.
            llr_rv_padded = tf.pad(llr_rv, [[0, 0], [0, n_cb - n - nb_pruned_nodes]])

            # apply circular shift to reconstruct original position
            # Implicit undoing of shortening of first two 2*Z positions
            llr_rv_unrolled = tf.roll(llr_rv_padded, shift=start_pos[rv_idx], axis=1)

            # accumulate using the configured accumulator function
            llr_accumulated = self._accumulator(llr_accumulated, llr_rv_unrolled, rv_idx)

        # use accumulated LLRs for further processing
        llr_5g = llr_accumulated

        # insert filler bits
        x1 = tf.slice(llr_5g, [0,0], [batch_size, k])
        x2 = tf.slice(llr_5g,
                      [0, k],
                      [batch_size, n_cb - k - nb_pruned_nodes])

        # negative sign due to logit definition
        z = -tf.cast(self._llr_max, self.rdtype) \
            * tf.ones([batch_size, self.encoder.k_filler], self.rdtype)

        llr_5g = tf.concat([x1, z, x2], axis=1)

        # and run the core decoder
        output = super().call(llr_5g, num_iter=num_iter, msg_v2c=msg_v2c)

        if self._return_state:
            x_hat, msg_v2c = output
        else:
            x_hat = output

        if self._return_infobits:  # return only info bits
            # reconstruct u_hat
            # 5G NR code is systematic
            u_hat = tf.slice(x_hat, [0,0], [batch_size, k])
            # reshape u_hat
            # remove RV dimension: [..., num_rv, n] -> [..., k]
            if self._harq_mode:
                output_shape = llr_ch_shape[0:-2] + [k]
            else:
                output_shape = llr_ch_shape[0:-1] + [k]
            output_shape[0] = -1  # It can be None
            u_reshaped = tf.reshape(u_hat, output_shape)

            if self._return_state:
                return u_reshaped, msg_v2c
            else:
                return u_reshaped

        else:  # return all codeword bits (after rate-matching)
            # The transmitted CW bits are not the same as used during decoding
            # cf. last parts of 5G encoding function

            # remove filler bits at pos (k, k_ldpc). n_pruned is n_ldpc in
            # HARQ mode
            x_no_filler1 = tf.slice(x_hat, [0, 0], [batch_size, k])
            x_no_filler2 = tf.slice(x_hat,
                                    [0, self.encoder.k_ldpc],
                                    [batch_size,
                                     self._n_pruned-self.encoder.k_ldpc])

            x_no_filler = tf.concat([x_no_filler1, x_no_filler2], 1)

            # replicate x to [-1, num_rv, n_ldpc or n_pruned] shape
            x_no_filler = tf.expand_dims(x_no_filler, axis=1)  # [-1, 1, n_ldpc or n_pruned]
            x_tiled = tf.tile(x_no_filler, [1, num_rv, 1]) # [-1, num_rv, n_ldpc or n_pruned]

            # rotate back to input position (rv)
            x_rolled = tf.stack(
                [tf.roll(x_tiled[:, rv_idx, :], shift=-start_pos[rv_idx], axis=1)
                   for rv_idx, _ in enumerate(rv)
                ], axis=1)

            # get back to length n
            x_short = tf.slice(x_rolled, [0, 0, 0], [batch_size, num_rv, n])

            # if used, apply rate-matching output interleaver again as
            # Sec. 5.4.2.2 in 38.212
            if self.encoder.num_bits_per_symbol is not None:
                x_short = tf.gather(x_short, self.encoder.out_int, axis=-1)

            # reshape to original size
            x_short= tf.reshape(x_short, llr_ch_shape)

            if self._return_state:
                return x_short, msg_v2c
            else:
                return x_short
