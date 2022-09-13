#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Layers for channel decoding and utility functions."""

import tensorflow as tf
import numpy as np
import scipy as sp # for sparse H matrix computations
from tensorflow.keras.layers import Layer
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.fec.utils import llr2mi
import matplotlib.pyplot as plt

class LDPCBPDecoder(Layer):
    # pylint: disable=line-too-long
    r"""LDPCBPDecoder(pcm, trainable=False, cn_type='boxplus-phi', hard_out=True, track_exit=False, num_iter=20, stateful=False,output_dtype=tf.float32, **kwargs)

    Iterative belief propagation decoder for low-density parity-check (LDPC)
    codes and other `codes on graphs`.

    This class defines a generic belief propagation decoder for decoding
    with arbitrary parity-check matrices. It can be used to iteratively
    estimate/recover the transmitted codeword (or information bits) based on the
    LLR-values of the received noisy codeword observation.

    The decoder implements the flooding SPA algorithm [Ryan]_, i.e., all nodes
    are updated in a parallel fashion. Different check node update functions are
    available

    (1) `boxplus`

        .. math::
            y_{j \to i} = 2 \operatorname{tanh}^{-1} \left( \prod_{i' \in \mathcal{N}_(j) \setminus i} \operatorname{tanh} \left( \frac{x_{i' \to j}}{2} \right) \right)

    (2) `boxplus-phi`

        .. math::
            y_{j \to i} = \alpha_{j \to i} \cdot \phi \left( \sum_{i' \in \mathcal{N}_(j) \setminus i} \phi \left( |x_{i' \to j}|\right) \right)

        with :math:`\phi(x)=-\operatorname{log}(\operatorname{tanh} \left(\frac{x}{2}) \right)`

    (3) `minsum`

        .. math::
            \qquad y_{j \to i} = \alpha_{j \to i} \cdot {min}_{i' \in \mathcal{N}_(j) \setminus i} \left(|x_{i' \to j}|\right)

    where :math:`y_{j \to i}` denotes the message from check node (CN) *j* to
    variable node (VN) *i* and :math:`x_{i \to j}` from VN *i* to CN *j*,
    respectively. Further, :math:`\mathcal{N}_(j)` denotes all indices of
    connected VNs to CN *j* and

    .. math::
        \alpha_{j \to i} = \prod_{i' \in \mathcal{N}_(j) \setminus i} \operatorname{sign}(x_{i' \to j})

    is the sign of the outgoing message. For further details we refer to
    [Ryan]_.

    Note that for full 5G 3GPP NR compatibility, the correct puncturing and
    shortening patterns must be applied (cf. [Richardson]_ for details), this
    can be done by :class:`~sionna.fec.ldpc.decoding.LDPC5GEncoder` and
    :class:`~sionna.fec.ldpc.decoding.LDPC5GDecoder`, respectively.

    If required, the decoder can be made trainable and is fully differentiable
    by following the concept of `weighted BP` [Nachmani]_ as shown in Fig. 1
    leading to

    .. math::
        y_{j \to i} = 2 \operatorname{tanh}^{-1} \left( \prod_{i' \in \mathcal{N}_(j) \setminus i} \operatorname{tanh} \left( \frac{\textcolor{red}{w_{i' \to j}} \cdot x_{i' \to j}}{2} \right) \right)

    where :math:`w_{i \to j}` denotes the trainable weight of message :math:`x_{i \to j}`.
    Please note that the training of some check node types may be not supported.

    ..  figure:: ../figures/weighted_bp.png

        Fig. 1: Weighted BP as proposed in [Nachmani]_.


    The class inherits from the Keras layer class and can be used as layer in a
    Keras model.

    Parameters
    ----------
        pcm: ndarray
            An ndarray of shape `[n-k, n]` defining the parity-check matrix
            consisting only of `0` or `1` entries. Can be also of type `scipy.
            sparse.csr_matrix` or `scipy.sparse.csc_matrix`.

        trainable: bool
            Defaults to False. If True, every outgoing variable node message is
            scaled with a trainable scalar.

        cn_type: str
            A string defaults to '"boxplus-phi"'. One of
            {`"boxplus"`, `"boxplus-phi"`, `"minsum"`} where
            '"boxplus"' implements the single-parity-check APP decoding rule.
            '"boxplus-phi"' implements the numerical more stable version of
            boxplus [Ryan]_.
            '"minsum"' implements the min-approximation of the CN
            update rule [Ryan]_.

        hard_out: bool
            Defaults to True. If True, the decoder provides hard-decided
            codeword bits instead of soft-values.

        track_exit: bool
            Defaults to False. If True, the decoder tracks EXIT
            characteristics. Note that this requires the all-zero
            CW as input.

        num_iter: int
            Defining the number of decoder iteration (no early stopping used at
            the moment!).

        stateful: bool
            Defaults to False. If True, the internal VN messages ``msg_vn``
            from the last decoding iteration are returned, and ``msg_vn`` or
            `None` needs to be given as a second input when calling the decoder.
            This is required for iterative demapping and decoding.

        output_dtype: tf.DType
            Defaults to tf.float32. Defines the output datatype of the layer
            (internal precision remains tf.float32).

    Input
    -----
    llrs_ch or (llrs_ch, msg_vn):
        Tensor or Tuple (only required if ``stateful`` is True):

    llrs_ch: [...,n], tf.float32
        2+D tensor containing the channel logits/llr values.

    msg_vn: None or RaggedTensor, tf.float32
        Ragged tensor of VN messages.
        Required only if ``stateful`` is True.

    Output
    ------
        : [...,n], tf.float32
            2+D Tensor of same shape as ``inputs`` containing
            bit-wise soft-estimates (or hard-decided bit-values) of all
            codeword bits.

        : RaggedTensor, tf.float32:
            Tensor of VN messages.
            Returned only if ``stateful`` is set to True.

    Attributes
    ----------
        pcm: ndarray
            An ndarray of shape `[n-k, n]` defining the parity-check matrix
            consisting only of `0` or `1` entries. Can be also of type `scipy.
            sparse.csr_matrix` or `scipy.sparse.csc_matrix`.

        num_cns: int
            Defining the number of check nodes.

        num_vns: int
            Defining the number of variable nodes.

        num_edges: int
            Defining the total number of edges.

        trainable: bool
            If True, the decoder uses trainable weights.

        _atanh_clip_value: float
            Defining the internal clipping value before the atanh is applied
            (relates to the CN update).

        _cn_type: str
            Defining the CN update function type.

        _cn_update:
            A function defining the CN update.

        _hard_out: bool
            If True, the decoder outputs hard-decided bits.

        _cn_con: ndarray
            An ndarray of shape `[num_edges]` defining all edges from check
            node perspective.

        _vn_con: ndarray
            An ndarray of shape `[num_edges]` defining all edges from variable
            node perspective.

        _vn_mask_tf: tf.float32
            A ragged Tensor of shape `[num_vns, None]` defining the incoming
            message indices per VN. The second dimension is ragged and depends
            on the node degree.

        _cn_mask_tf: tf.float32
            A ragged Tensor of shape `[num_cns, None]` defining the incoming
            message indices per CN. The second dimension is ragged and depends
            on the node degree.

        _ind_cn: ndarray
            An ndarray of shape `[num_edges]` defining the permutation index to
            rearrange messages from variable into check node perspective.

        _ind_cn_inv: ndarray
            An ndarray of shape `[num_edges]` defining the permutation index to
            rearrange messages from check into variable node perspective.

        _vn_row_splits: ndarray
            An ndarray of shape `[num_vns+1]` defining the row split positions
            of a 1D vector consisting of all edges messages. Used to build a
            ragged Tensor of incoming VN messages.

        _cn_row_splits: ndarray
            An ndarray of shape `[num_cns+1]` defining the row split positions
            of a 1D vector consisting of all edges messages. Used to build a
            ragged Tensor of incoming CN messages.

        _edge_weights: tf.float32
            A Tensor of shape `[num_edges]` defining a (trainable) weight per
            outgoing VN message.

    Raises:
        ValueError
            If the shape of ``pcm`` is invalid or contains other values than
            `0` or `1` or dtype is not `tf.float32`.

        AssertionError
            If ``trainable`` is not `bool`.

        AssertionError
            If ``track_exit`` is not `bool`.

        AssertionError
            If ``hard_out`` is not `bool`.

        AssertionError
            If ``stateful`` is not `bool`.

        AssertionError
            If ``cn_type`` is not `str`.

        ValueError
            If ``num_iter`` is not an integer greater (or equal) `0`.

        ValueError
            If ``output_dtype`` is not
            {tf.float16, tf.float32, tf.float64}.

        ValueError
            If ``inputs`` is not of shape `[batch_size, n]`.

        InvalidArgumentError
            When rank(``inputs``)<2.
    Note
    ----
        As decoding input logits
        :math:`\operatorname{log} \frac{p(x=1)}{p(x=0)}` are
        assumed for compatibility with the learning framework, but internally
        log-likelihood ratios (LLRs) with definition :math:`\operatorname{log} \frac{p(x=0)}{p(x=1)}` are used.

        The decoder is not (particularly) optimized for quasi-cyclic (QC) LDPC
        codes and, thus, supports arbitrary parity-check matrices.

        The decoder is implemented by using '"ragged Tensors"' [TF_ragged]_ to
        account for arbitrary node degrees. To avoid a performance degradation
        caused by a severe indexing overhead, the batch-dimension is shifted to
        the last dimension during decoding.

        If the decoder is made trainable [Nachmani]_, for performance
        improvements only variable to check node messages are scaled as the VN
        operation is linear and, thus, would not increase the expressive power
        of the weights.

    """

    def __init__(self,
                 pcm,
                 trainable=False,
                 cn_type='boxplus-phi',
                 hard_out=True,
                 track_exit=False,
                 num_iter=20,
                 stateful=False,
                 output_dtype=tf.float32,
                 **kwargs):

        super().__init__(dtype=output_dtype, **kwargs)

        assert isinstance(trainable, bool), 'trainable must be bool.'
        assert isinstance(hard_out, bool), 'hard_out must be bool.'
        assert isinstance(track_exit, bool), 'track_exit must be bool.'
        assert isinstance(cn_type, str) , 'cn_type must be str.'
        assert isinstance(num_iter, int), 'num_iter must be int.'
        assert num_iter>=0, 'num_iter cannot be negative.'
        assert isinstance(stateful, bool), 'stateful must be bool.'
        assert isinstance(output_dtype, tf.DType), \
                                'output_dtype must be tf.Dtype.'

        if isinstance(pcm, np.ndarray):
            assert np.array_equal(pcm, pcm.astype(bool)), 'PC matrix \
                must be binary.'
        elif isinstance(pcm, sp.sparse.csr_matrix):
            assert np.array_equal(pcm.data, pcm.data.astype(bool)), \
                'PC matrix must be binary.'
        elif isinstance(pcm, sp.sparse.csc_matrix):
            assert np.array_equal(pcm.data, pcm.data.astype(bool)), \
                'PC matrix must be binary.'
        else:
            raise TypeError("Unsupported dtype of pcm.")

        if output_dtype not in (tf.float16, tf.float32, tf.float64):
            raise ValueError(
                'output_dtype must be {tf.float16, tf.float32, tf.float64}.')

        if output_dtype is not tf.float32:
            print('Note: decoder uses tf.float32 for internal calculations.')

        # init decoder parameters
        self._pcm = pcm
        self._trainable = trainable
        self._cn_type = cn_type
        self._hard_out = hard_out
        self._track_exit = track_exit
        self._num_iter = tf.constant(num_iter, dtype=tf.int32)
        self._stateful = stateful
        self._output_dtype = output_dtype

        # clipping value for the atanh function is applied (tf.float32 is used)
        self._atanh_clip_value = 1 - 1e-7
        # clipping for min-sum decoding
        self._llr_max_minsum = 20

        # init code parameters
        self._num_cns = pcm.shape[0] # total number of check nodes
        self._num_vns = pcm.shape[1] # total number of variable nodes

        # make pcm sparse first if ndarray is provided
        if isinstance(pcm, np.ndarray):
            pcm = sp.sparse.csr_matrix(pcm)
        # find all edges from variable and check node perspective
        self._cn_con, self._vn_con, _ = sp.sparse.find(pcm)

        # number of edges equals number of non-zero elements in the
        # parity-check matrix
        self._num_edges = len(self._vn_con)

        # permutation index to rearrange messages into check node perspective
        self._ind_cn = np.argsort(self._cn_con)

        # inverse permutation index to rearrange messages back into variable
        # node perspective
        self._ind_cn_inv = np.argsort(self._ind_cn)

        # generate row masks (array of integers defining the row split pos.)
        self._vn_row_splits = self._gen_node_mask_row(self._vn_con)
        self._cn_row_splits = self._gen_node_mask_row(
                                                    self._cn_con[self._ind_cn])
        # pre-load the CN function for performance reasons
        if self._cn_type=='boxplus':
            # check node update using the tanh function
            self._cn_update = self._cn_update_tanh
        elif self._cn_type=='boxplus-phi':
            # check node update using the "_phi" function
            self._cn_update = self._cn_update_phi
        elif self._cn_type=='minsum':
            # check node update using the min-sum approximation
            self._cn_update = self._cn_update_minsum
        else:
            raise ValueError('Unknown node type.')

        # init trainable weights if needed
        self._has_weights = False  # indicates if trainable weights exist
        if self._trainable:
            self._has_weights = True
            self._edge_weights = tf.Variable(tf.ones(self._num_edges),
                                             trainable=self._trainable,
                                             dtype=tf.float32)

        # track mutual information during decoding
        self._ie_c = 0
        self._ie_v = 0

    #########################################
    # Public methods and properties
    #########################################

    @property
    def pcm(self):
        """Parity-check matrix of LDPC code."""
        return self._pcm

    @property
    def num_cns(self):
        """Number of check nodes."""
        return self._num_cns

    @property
    def num_vns(self):
        """Number of variable nodes."""
        return self._num_vns

    @property
    def num_edges(self):
        """Number of edges in decoding graph."""
        return self._num_edges

    @property
    def has_weights(self):
        """Indicates if decoder has trainable weights."""
        return self._has_weights

    @property
    def edge_weights(self):
        """Trainable weights of the BP decoder."""
        if not self._has_weights:
            return []
        else:
            return self._edge_weights

    @property
    def output_dtype(self):
        """Output dtype of decoder."""
        return self._output_dtype

    @property
    def ie_c(self):
        "Extrinsic mutual information at check node."
        return self._ie_c

    @property
    def ie_v(self):
        "Extrinsic mutual information at variable node."
        return self._ie_v

    @property
    def num_iter(self):
        "Number of decoding iterations."
        return self._num_iter

    @num_iter.setter
    def num_iter(self, num_iter):
        "Number of decoding iterations."
        assert isinstance(num_iter, int), 'num_iter must be int.'
        assert num_iter>=0, 'num_iter cannot be negative.'
        self._num_iter = tf.constant(num_iter, dtype=tf.int32)

    def show_weights(self, size=7):
        """Show histogram of trainable weights.

        Input
        -----
            size: float
                Figure size of the matplotlib figure.

        """
        # only plot if weights exist
        if self._has_weights:
            weights = self._edge_weights.numpy()

            plt.figure(figsize=(size,size))
            plt.hist(weights, density=True, bins=20, align='mid')
            plt.xlabel('weight value')
            plt.ylabel('density')
            plt.grid(True, which='both', axis='both')
            plt.title('Weight Distribution')
        else:
            print("No weights to show.")

    #########################
    # Utility methods
    #########################

    def _gen_node_mask(self, con):
        """ Generates internal node masks indicating which msg index belongs
        to which node index.
        """
        ind = np.argsort(con)
        con = con[ind]

        node_mask = []

        cur_node = 0
        cur_mask = []
        for i in range(self._num_edges):
            if con[i] == cur_node:
                cur_mask.append(ind[i])
            else:
                node_mask.append(cur_mask)
                cur_mask = [ind[i]]
                cur_node += 1
        node_mask.append(cur_mask)
        return node_mask

    def _gen_node_mask_row(self, con):
        """ Defining the row split positions of a 1D vector consisting of all
        edges messages.

        Used to build a ragged Tensor of incoming node messages.
        """
        node_mask = [0] # the first element indicates the first node index (=0)

        cur_node = 0
        for i in range(self._num_edges):
            if con[i] != cur_node:
                node_mask.append(i)
                cur_node += 1
        node_mask.append(self._num_edges) # last element must be the number of
        # elements (delimiter)
        return node_mask

    def _vn_update(self, msg, llr_ch):
        """ Variable node update function.

        This function implements the (extrinsic) variable node update
        function. It takes the sum over all incoming messages ``msg`` excluding
        the intrinsic (= outgoing) message itself.

        Additionally, the channel LLR ``llr_ch`` is added to each message.
        """
        # aggregate all incoming messages per node
        x = tf.reduce_sum(msg, axis=1)
        x = tf.add(x, llr_ch)

        # TF2.9 does not support XLA for the addition of ragged tensors
        # the following code provides a workaround that supports XLA

        # subtract extrinsic message from node value
        # x = tf.expand_dims(x, axis=1)
        # x = tf.add(-msg, x)
        x = tf.ragged.map_flat_values(lambda x, y, row_ind :
                                      x + tf.gather(y, row_ind),
                                      -1.*msg,
                                      x,
                                      msg.value_rowids())
        return x

    def _extrinsic_min(self, msg):
        """ Provides the extrinsic min operation for the minsum approximation
        of the CN function.

        This function implements the extrinsic min operation, i.e.,
        the min is taken over all values excluding the value at the current
        index.

        Note that the input is expected to be a Tensor and NOT a ragged Tensor.
        """
        num_val = tf.shape(msg)[0]
        msg = tf.transpose(msg, (1,0))
        msg = tf.expand_dims(msg, axis=1)
        id_mat = tf.eye(num_val)

        msg = (tf.tile(msg, (1, num_val, 1)) # create outgoing tensor per value
               + 1000. * id_mat) # "ignore" intrinsic msg by adding large const.


        msg = tf.math.reduce_min(msg, axis=2)
        msg = tf.transpose(msg, (1,0))
        return msg

    def _where_ragged(self, msg):
        """Helper to replace 0 elements from ragged tensor (called with
        map_flat_values)."""
        return tf.where(tf.equal(msg, 0), tf.ones_like(msg) * 1e-12, msg)

    def _where_ragged_inv(self, msg):
        """Helper to replace small elements from ragged tensor (called with
        map_flat_values) with exact `0`."""
        msg_mod =  tf.where(tf.less(tf.abs(msg), 1e-7),
                            tf.zeros_like(msg),
                            msg)
        return msg_mod

    def _cn_update_tanh(self, msg):
        """Check node update function implementing the exact boxplus operation.

        This function implements the (extrinsic) check node update
        function. It calculates the boxplus function over all incoming messages
        "msg" excluding the intrinsic (=outgoing) message itself.
        The exact boxplus function is implemented by using the tanh function.

        The input is expected to be a ragged Tensor of shape
        `[num_cns, None, batch_size]`.

        Note that for numerical stability clipping is applied.
        """

        msg = msg / 2
        # tanh is not overloaded for ragged tensors
        msg = tf.ragged.map_flat_values(tf.tanh, msg) # tanh is not overloaded

        # for ragged tensors; map to flat tensor first
        msg = tf.ragged.map_flat_values(self._where_ragged, msg)

        msg_prod = tf.reduce_prod(msg, axis=1)

        # TF2.9 does not support XLA for the multiplication of ragged tensors
        # the following code provides a workaround that supports XLA

        # ^-1 to avoid division
        # Note this is (potentially) numerically unstable
        # msg = msg**-1 * tf.expand_dims(msg_prod, axis=1) # remove own edge

        msg = tf.ragged.map_flat_values(lambda x, y, row_ind :
                                        x * tf.gather(y, row_ind),
                                        msg**-1,
                                        msg_prod,
                                        msg.value_rowids())

        # Overwrite small (numerical zeros) message values with exact zero
        # these are introduced by the previous "_where_ragged" operation
        # this is required to keep the product stable (cf. _phi_update for log
        # sum implementation)
        msg = tf.ragged.map_flat_values(self._where_ragged_inv, msg)

        msg = tf.clip_by_value(msg,
                               clip_value_min=-self._atanh_clip_value,
                               clip_value_max=self._atanh_clip_value)

        # atanh is not overloaded for ragged tensors
        msg = 2 * tf.ragged.map_flat_values(tf.atanh, msg)
        return msg

    def _phi(self, x):
        """Helper function for the check node update.

        This function implements the (element-wise) `"_phi"` function as defined
        in [Ryan]_.
        """
        # the clipping values are optimized for tf.float32
        x = tf.clip_by_value(x, clip_value_min=8.5e-8, clip_value_max=16.635532)
        return tf.math.log(tf.math.exp(x)+1) - tf.math.log(tf.math.exp(x)-1)

    def _cn_update_phi(self, msg):
        """Check node update function implementing the exact boxplus operation.

        This function implements the (extrinsic) check node update function
        based on the numerically more stable `"_phi"` function (cf. [Ryan]_).
        It calculates the boxplus function over all incoming messages ``msg``
        excluding the intrinsic (=outgoing) message itself.
        The exact boxplus function is implemented by using the `"_phi"` function
        as in [Ryan]_.

        The input is expected to be a ragged Tensor of shape
        `[num_cns, None, batch_size]`.

        Note that for numerical stability clipping is applied.
        """

        sign_val = tf.sign(msg)

        sign_val = tf.where(tf.equal(sign_val, 0),
                            tf.ones_like(sign_val),
                            sign_val)

        sign_node = tf.reduce_prod(sign_val, axis=1)

        # TF2.9 does not support XLA for the multiplication of ragged tensors
        # the following code provides a workaround that supports XLA

        # sign_val = sign_val * tf.expand_dims(sign_node, axis=1)
        sign_val = tf.ragged.map_flat_values(lambda x, y, row_ind :
                                             x * tf.gather(y, row_ind),
                                             sign_val,
                                             sign_node,
                                             sign_val.value_rowids())

        msg = tf.ragged.map_flat_values(tf.abs, msg) # remove sign

        # apply _phi element-wise (does not support ragged Tensors)
        msg = tf.ragged.map_flat_values(self._phi, msg)
        msg_sum = tf.reduce_sum(msg, axis=1)

        # TF2.9 does not support XLA for the addition of ragged tensors
        # the following code provides a workaround that supports XLA

        # msg = tf.add( -msg, tf.expand_dims(msg_sum, axis=1)) # remove own edge
        msg = tf.ragged.map_flat_values(lambda x, y, row_ind :
                                        x + tf.gather(y, row_ind),
                                        -1.*msg,
                                        msg_sum,
                                        msg.value_rowids())

        # apply _phi element-wise (does not support ragged Tensors)
        msg = self._stop_ragged_gradient(sign_val) * tf.ragged.map_flat_values(
                                                            self._phi, msg)
        return msg

    def _stop_ragged_gradient(self, rt):
        """Helper function as TF 2.5 does not support ragged gradient
        stopping"""
        return rt.with_flat_values(tf.stop_gradient(rt.flat_values))

    def _sign_val_minsum(self, msg):
        """Helper to replace find sign-value during min-sum decoding.
        Must be called with `map_flat_values`."""

        sign_val = tf.sign(msg)
        sign_val = tf.where(tf.equal(sign_val, 0),
                            tf.ones_like(sign_val),
                            sign_val)
        return sign_val

    def _cn_update_minsum_mapfn(self, msg):
        """ Check node update function implementing the min-sum approximation.

        This function approximates the (extrinsic) check node update
        function based on the min-sum approximation (cf. [Ryan]_).
        It calculates the "extrinsic" min function over all incoming messages
        ``msg`` excluding the intrinsic (=outgoing) message itself.

        The input is expected to be a ragged Tensor of shape
        `[num_vns, None, batch_size]`.

        This function uses tf.map_fn() to call the CN updates.
        It is currently not used, but can be used as template to implement
        modified CN functions (e.g., offset-corrected minsum).
        Please note that tf.map_fn lowers the throughput significantly.
        """

        sign_val = tf.ragged.map_flat_values(self._sign_val_minsum, msg)

        sign_node = tf.reduce_prod(sign_val, axis=1)
        sign_val = self._stop_ragged_gradient(sign_val) * tf.expand_dims(
                                                             sign_node, axis=1)

        msg = tf.ragged.map_flat_values(tf.abs, msg) # remove sign

        # calculate extrinsic messages and include the sign
        msg_e = tf.map_fn(self._extrinsic_min, msg, infer_shape=False)

        # ensure shape after map_fn
        msg_fv = msg_e.flat_values
        msg_fv = tf.ensure_shape(msg_fv, msg.flat_values.shape)
        msg_e = msg.with_flat_values(msg_fv)

        msg = sign_val * msg_e

        return msg

    def _cn_update_minsum(self, msg):
        """ Check node update function implementing the min-sum approximation.

        This function approximates the (extrinsic) check node update
        function based on the min-sum approximation (cf. [Ryan]_).
        It calculates the "extrinsic" min function over all incoming messages
        ``msg`` excluding the intrinsic (=outgoing) message itself.

        The input is expected to be a ragged Tensor of shape
        `[num_vns, None, batch_size]`.
        """
        # a constant used overwrite the first min
        LARGE_VAL = 10000. # pylint: disable=invalid-name

        # clip values for numerical stability
        msg = tf.clip_by_value(msg,
                               clip_value_min=-self._llr_max_minsum,
                               clip_value_max=self._llr_max_minsum)

        # calculate sign of outgoing msg
        sign_val = tf.ragged.map_flat_values(self._sign_val_minsum, msg)

        sign_node = tf.reduce_prod(sign_val, axis=1)

        # TF2.9 does not support XLA for the multiplication of ragged tensors
        # the following code provides a workaround that supports XLA

        # sign_val = self._stop_ragged_gradient(sign_val) \
        #             * tf.expand_dims(sign_node, axis=1)
        sign_val = tf.ragged.map_flat_values(
                                        lambda x, y, row_ind:
                                        tf.multiply(x, tf.gather(y, row_ind)),
                                        self._stop_ragged_gradient(sign_val),
                                        sign_node,
                                        sign_val.value_rowids())

        msg = tf.ragged.map_flat_values(tf.abs, msg) # remove sign

        # Calculate the extrinsic minimum per CN, i.e., for each message of
        # index i, find the smallest and the second smallest value.
        # However, in some cases the second smallest value may equal the
        # smallest value (multiplicity of mins).
        # Please note that this needs to be applied to raggedTensors, e.g.,
        # tf.top_k() is currently not supported and the ops must support graph
        # # mode.

        # find min_value per node
        min_val = tf.reduce_min(msg, axis=1, keepdims=True)

        # TF2.9 does not support XLA for the subtraction of ragged tensors
        # the following code provides a workaround that supports XLA

        # and subtract min; the new array contains zero at the min positions
        # benefits from broadcasting; all other values are positive
        # msg_min1 = msg - min_val
        msg_min1 = tf.ragged.map_flat_values(lambda x, y, row_ind:
                                             x- tf.gather(y, row_ind),
                                             msg,
                                             tf.squeeze(min_val, axis=1),
                                             msg.value_rowids())

        # replace 0 (=min positions) with large value to ignore it for further
        # min calculations
        msg = tf.ragged.map_flat_values(lambda x:
                                        tf.where(tf.equal(x, 0), LARGE_VAL, x),
                                        msg_min1)

        # find the second smallest element (we add min_val as this has been
        # subtracted before)
        min_val2 = tf.reduce_min(msg, axis=1, keepdims=True) + min_val

        # Detect duplicated minima (i.e., min_val occurs at two incoming
        # messages). As the LLRs per node are <LLR_MAX and we have
        # replace at least 1 position (position with message "min_val") by
        # LARGE_VAL, it holds for the sum < LARGE_VAL + node_degree*LLR_MAX.
        # if the sum > 2*LARGE_VAL, the multiplicity of the min is at least 2.
        node_sum = tf.reduce_sum(msg, axis=1, keepdims=True) - (2*LARGE_VAL-1.)
        # indicator that duplicated min was detected (per node)
        double_min = 0.5*(1-tf.sign(node_sum))

        # if a duplicate min occurred, both edges must have min_val, otherwise
        # the second smallest value is taken
        min_val_e = (1-double_min) * min_val + (double_min) * min_val2

        # replace all values with min_val except the position where the min
        # occurred (=extrinsic min).
        msg_e = tf.where(msg==LARGE_VAL, min_val_e, min_val)

        # it seems like tf.where does not set the shape of tf.ragged properly
        # we need to ensure the shape manually
        msg_e = tf.ragged.map_flat_values(
                                    lambda x:
                                    tf.ensure_shape(x, msg.flat_values.shape),
                                    msg_e)

        # TF2.9 does not support XLA for the multiplication of ragged tensors
        # the following code provides a workaround that supports XLA

        # and apply sign
        #msg = sign_val * msg_e
        msg = tf.ragged.map_flat_values(tf.multiply,
                                        sign_val,
                                        msg_e)

        return msg

    def _mult_weights(self, x):
        """Multiply messages with trainable weights for weighted BP."""
        # transpose for simpler broadcasting of training variables
        x = tf.transpose(x, (1, 0))
        x = tf.math.multiply(x, self._edge_weights)
        x = tf.transpose(x, (1, 0))
        return x

    #########################
    # Keras layer functions
    #########################

    def build(self, input_shape):
        # Raise AssertionError if shape of x is invalid
        if self._stateful:
            assert(len(input_shape)==2), \
                "For stateful decoding, a tuple of two inputs is expected."
            input_shape = input_shape[0]

        assert (input_shape[-1]==self._num_vns), \
                            'Last dimension must be of length n.'
        assert (len(input_shape)>=2), 'The inputs must have at least rank 2.'

    def call(self, inputs):
        """Iterative BP decoding function.

        This function performs ``num_iter`` belief propagation decoding
        iterations and returns the estimated codeword.

        Args:
        llr_ch or (llr_ch, msg_vn):

            llr_ch (tf.float32): Tensor of shape `[...,n]` containing the
                channel logits/llr values.

            msg_vn (tf.float32) : Ragged tensor containing the VN
                messages, or None. Required if ``stateful`` is set to True.

        Returns:
            `tf.float32`: Tensor of shape `[...,n]` containing
            bit-wise soft-estimates (or hard-decided bit-values) of all
            codeword bits.

        Raises:
            ValueError: If ``inputs`` is not of shape `[batch_size, n]`.

            InvalidArgumentError: When rank(``inputs``)<2.
        """

        # Extract inputs
        if self._stateful:
            llr_ch, msg_vn = inputs
        else:
            llr_ch = inputs

        tf.debugging.assert_type(llr_ch, self.dtype, 'Invalid input dtype.')

        # internal calculations still in tf.float32
        llr_ch = tf.cast(llr_ch, tf.float32)

        # last dim must be of length n
        tf.debugging.assert_equal(tf.shape(llr_ch)[-1],
                                  self._num_vns,
                                  'Last dimension must be of length n.')

        llr_ch_shape = llr_ch.get_shape().as_list()
        new_shape = [-1, self._num_vns]
        llr_ch_reshaped = tf.reshape(llr_ch, new_shape)

        # must be done during call, as XLA fails otherwise due to ragged
        # indices placed on the CPU device.
        # create permutation index from cn perspective
        self._cn_mask_tf = tf.ragged.constant(self._gen_node_mask(self._cn_con),
                                              row_splits_dtype=tf.int32)

        # batch dimension is last dimension due to ragged tensor representation
        llr_ch = tf.transpose(llr_ch_reshaped, (1,0))

        llr_ch = -1. * llr_ch # logits are converted into "true" llrs

        # init internal decoder state if not explicitly
        # provided (e.g., required to restore decoder state for iterative
        # detection and decoding)
        # load internal state from previous iteration
        # required for iterative det./dec.
        if not self._stateful or msg_vn is None:
            msg_shape = tf.stack([tf.constant(self._num_edges),
                                   tf.shape(llr_ch)[1]],
                                   axis=0)
            msg_vn = tf.zeros(msg_shape, dtype=tf.float32)
        else:
            msg_vn = msg_vn.flat_values

        # track exit decoding trajectory; requires all-zero cw?
        if self._track_exit:
            self._ie_c = tf.zeros(self._num_iter+1)
            self._ie_v = tf.zeros(self._num_iter+1)

        # perform one decoding iteration
        # Remark: msg_vn cannot be ragged as input for tf.while_loop as
        # otherwise XLA will not be supported (with TF 2.5)
        def dec_iter(llr_ch, msg_vn, it):
            it += 1

            msg_vn = tf.RaggedTensor.from_row_splits(
                        values=msg_vn,
                        row_splits=tf.constant(self._vn_row_splits, tf.int32))
            # variable node update
            msg_vn = self._vn_update(msg_vn, llr_ch)

            # track exit decoding trajectory; requires all-zero cw
            if self._track_exit:
                # neg values as different llr def is expected
                mi = llr2mi(-1. * msg_vn.flat_values)
                self._ie_v = tf.tensor_scatter_nd_add(self._ie_v,
                                                     tf.reshape(it, (1, 1)),
                                                     tf.reshape(mi, (1)))

            # scale outgoing vn messages (weighted BP); only if activated
            if self._has_weights:
                msg_vn = tf.ragged.map_flat_values(self._mult_weights,
                                                   msg_vn)
            # permute edges into CN perspective
            msg_cn = tf.gather(msg_vn.flat_values, self._cn_mask_tf, axis=None)

            # check node update using the pre-defined function
            msg_cn = self._cn_update(msg_cn)

            # track exit decoding trajectory; requires all-zero cw?
            if self._track_exit:
                # neg values as different llr def is expected
                mi = llr2mi(-1.*msg_cn.flat_values)
                # update pos i+1 such that first iter is stored as 0
                self._ie_c = tf.tensor_scatter_nd_add(self._ie_c,
                                                     tf.reshape(it, (1, 1)),
                                                     tf.reshape(mi, (1)))

            # re-permute edges to variable node perspective
            msg_vn = tf.gather(msg_cn.flat_values, self._ind_cn_inv, axis=None)
            return llr_ch, msg_vn, it

        # stopping condition (required for tf.while_loop)
        def dec_stop(llr_ch, msg_vn, it): # pylint: disable=W0613
            return tf.less(it, self._num_iter)

        # start decoding iterations
        it = tf.constant(0)
        # maximum_iterations required for XLA
        _, msg_vn, _ = tf.while_loop(dec_stop,
                                     dec_iter,
                                     (llr_ch, msg_vn, it),
                                     parallel_iterations=1,
                                     maximum_iterations=self._num_iter)


        # raggedTensor for final marginalization
        msg_vn = tf.RaggedTensor.from_row_splits(
                        values=msg_vn,
                        row_splits=tf.constant(self._vn_row_splits, tf.int32))

        # marginalize and remove ragged Tensor
        x_hat = tf.add(llr_ch, tf.reduce_sum(msg_vn, axis=1))

        # restore batch dimension to first dimension
        x_hat = tf.transpose(x_hat, (1,0))

        x_hat = -1. * x_hat # convert llrs back into logits

        if self._hard_out: # hard decide decoder output if required
            x_hat = tf.cast(tf.less(0.0, x_hat), self._output_dtype)

        # Reshape c_short so that it matches the original input dimensions
        output_shape = llr_ch_shape
        output_shape[0] = -1 # overwrite batch dim (can be None in Keras)

        x_reshaped = tf.reshape(x_hat, output_shape)

        # cast output to output_dtype
        x_out = tf.cast(x_reshaped, self._output_dtype)

        if not self._stateful:
            return x_out
        else:
            return x_out, msg_vn

class LDPC5GDecoder(LDPCBPDecoder):
    # pylint: disable=line-too-long
    r"""LDPC5GDecoder(encoder, trainable=False, cn_type='boxplus-phi', hard_out=True, track_exit=False, return_infobits=True, prune_pcm=True, num_iter=20, stateful=False, output_dtype=tf.float32, **kwargs)

    (Iterative) belief propagation decoder for 5G NR LDPC codes.

    Inherits from :class:`~sionna.fec.ldpc.decoding.LDPCBPDecoder` and provides
    a wrapper for 5G compatibility, i.e., automatically handles puncturing and
    shortening according to [3GPPTS38212_LDPC]_.

    Note that for full 5G 3GPP NR compatibility, the correct puncturing and
    shortening patterns must be applied and, thus, the encoder object is
    required as input.

    If required the decoder can be made trainable and is differentiable
    (the training of some check node types may be not supported) following the
    concept of "weighted BP" [Nachmani]_.

    The class inherits from the Keras layer class and can be used as layer in a
    Keras model.

    Parameters
    ----------
        encoder: LDPC5GEncoder
            An instance of :class:`~sionna.fec.ldpc.encoding.LDPC5GEncoder`
            containing the correct code parameters.

        trainable: bool
            Defaults to False. If True, every outgoing variable node message is
            scaled with a trainable scalar.

        cn_type: str
            A string defaults to '"boxplus-phi"'. One of
            {`"boxplus"`, `"boxplus-phi"`, `"minsum"`} where
            '"boxplus"' implements the single-parity-check APP decoding rule.
            '"boxplus-phi"' implements the numerical more stable version of
            boxplus [Ryan]_.
            '"minsum"' implements the min-approximation of the CN
            update rule [Ryan]_.

        hard_out: bool
            Defaults to True. If True, the decoder provides hard-decided
            codeword bits instead of soft-values.

        track_exit: bool
            Defaults to False. If True, the decoder tracks EXIT characteristics.
            Note that this requires the all-zero CW as input.

        return_infobits: bool
            Defaults to True. If True, only the `k` info bits (soft or
            hard-decided) are returned. Otherwise all `n` positions are
            returned.

        prune_pcm: bool
            Defaults to True. If True, all punctured degree-1 VNs and
            connected check nodes are removed from the decoding graph (see
            [Cammerer]_ for details). Besides numerical differences, this should
            yield the same decoding result but improved the decoding throughput
            and reduces the memory footprint.

        num_iter: int
            Defining the number of decoder iteration (no early stopping used at
            the moment!).

        stateful: bool
            Defaults to False. If True, the internal VN messages ``msg_vn``
            from the last decoding iteration are returned, and ``msg_vn`` or
            `None` needs to be given as a second input when calling the decoder.
            This is required for iterative demapping and decoding.

        output_dtype: tf.DType
            Defaults to tf.float32. Defines the output datatype of the layer
            (internal precision remains tf.float32).

    Input
    -----
    llrs_ch or (llrs_ch, msg_vn):
        Tensor or Tuple (only required if ``stateful`` is True):

    llrs_ch: [...,n], tf.float32
        2+D tensor containing the channel logits/llr values.

    msg_vn: None or RaggedTensor, tf.float32
        Ragged tensor of VN messages.
        Required only if ``stateful`` is True.

    Output
    ------
        : [...,n] or [...,k], tf.float32
            2+D Tensor of same shape as ``inputs`` containing
            bit-wise soft-estimates (or hard-decided bit-values) of all
            codeword bits. If ``return_infobits`` is True, only the `k`
            information bits are returned.

        : RaggedTensor, tf.float32:
            Tensor of VN messages.
            Returned only if ``stateful`` is set to True.
    Raises
    ------
        ValueError
            If the shape of ``pcm`` is invalid or contains other
            values than `0` or `1`.

        AssertionError
            If ``trainable`` is not `bool`.

        AssertionError
            If ``track_exit`` is not `bool`.

        AssertionError
            If ``hard_out`` is not `bool`.

        AssertionError
            If ``return_infobits`` is not `bool`.

        AssertionError
            If ``encoder`` is not an instance of
            :class:`~sionna.fec.ldpc.encoding.LDPC5GEncoder`.

        ValueError
            If ``output_dtype`` is not {tf.float16, tf.float32, tf.
            float64}.

        ValueError
            If ``inputs`` is not of shape `[batch_size, n]`.

        ValueError
            If ``num_iter`` is not an integer greater (or equal) `0`.

        InvalidArgumentError
            When rank(``inputs``)<2.

    Note
    ----
        As decoding input logits
        :math:`\operatorname{log} \frac{p(x=1)}{p(x=0)}` are assumed for
        compatibility with the learning framework, but
        internally llrs with definition
        :math:`\operatorname{log} \frac{p(x=0)}{p(x=1)}` are used.

        The decoder is not (particularly) optimized for Quasi-cyclic (QC) LDPC
        codes and, thus, supports arbitrary parity-check matrices.

        The decoder is implemented by using '"ragged Tensors"' [TF_ragged]_ to
        account for arbitrary node degrees. To avoid a performance degradation
        caused by a severe indexing overhead, the batch-dimension is shifted to
        the last dimension during decoding.

        If the decoder is made trainable [Nachmani]_, for performance
        improvements only variable to check node messages are scaled as the VN
        operation is linear and, thus, would not increase the expressive power
        of the weights.
    """

    def __init__(self,
                 encoder,
                 trainable=False,
                 cn_type='boxplus-phi',
                 hard_out=True,
                 track_exit=False,
                 return_infobits=True,
                 prune_pcm=True,
                 num_iter=20,
                 stateful=False,
                 output_dtype=tf.float32,
                 **kwargs):

        # needs the 5G Encoder to access all 5G parameters
        assert isinstance(encoder, LDPC5GEncoder), 'encoder must \
                          be of class LDPC5GEncoder.'
        self._encoder = encoder
        pcm = encoder.pcm

        assert isinstance(return_infobits, bool), 'return_info must be bool.'
        self._return_infobits = return_infobits

        self._llr_max = 20 # internal max value for LLR initialization

        assert isinstance(output_dtype, tf.DType), \
                                'output_dtype must be tf.DType.'
        if output_dtype not in (tf.float16, tf.float32, tf.float64):
            raise ValueError(
                'output_dtype must be {tf.float16, tf.float32, tf.float64}.')
        self._output_dtype = output_dtype

        assert isinstance(stateful, bool), 'stateful must be bool.'
        self._stateful = stateful

        assert isinstance(prune_pcm, bool), 'prune_pcm must be bool.'
        # prune punctured degree-1 VNs and connected CNs. A punctured
        # VN-1 node will always "send" llr=0 to the connected CN. Thus, this
        # CN will only send 0 messages to all other VNs, i.e., does not
        # contribute to the decoding process.
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
            # effective codeword length after pruning of vn-1 nodes
            self._n_pruned = np.max((last_pos, encoder._n_ldpc - nb_punc_bits))
            self._nb_pruned_nodes = encoder._n_ldpc - self._n_pruned
            # remove last CNs and VNs from pcm
            pcm = pcm[:-self._nb_pruned_nodes, :-self._nb_pruned_nodes]

            #check for consistency
            assert(self._nb_pruned_nodes>=0), "Internal error: number of \
                        pruned nodes must be positive."
        else:
            self._nb_pruned_nodes = 0
            # no pruning; same length as before
            self._n_pruned = encoder._n_ldpc



        super().__init__(pcm,
                         trainable,
                         cn_type,
                         hard_out,
                         track_exit,
                         num_iter=num_iter,
                         stateful=stateful,
                         output_dtype=output_dtype,
                         **kwargs)

    #########################################
    # Public methods and properties
    #########################################

    @property
    def llr_max(self):
        """Max LLR value used for rate-matching."""
        return self._llr_max

    @property
    def encoder(self):
        """LDPC Encoder used for rate-matching/recovery."""
        return self._encoder

    #########################
    # Keras layer functions
    #########################

    def build(self, input_shape):
        """Build model."""
        if self._stateful:
            assert(len(input_shape)==2), \
                "For stateful decoding, a tuple of two inputs is expected."
            input_shape = input_shape[0]

        # check input dimensions for consistency
        assert (input_shape[-1]==self.encoder.n), \
                                'Last dimension must be of length n.'
        assert (len(input_shape)>=2), 'The inputs must have at least rank 2.'

        self._old_shape_5g = input_shape

    def call(self, inputs):
        """Iterative BP decoding function.

        This function performs ``num_iter`` belief propagation decoding
        iterations and returns the estimated codeword.

        Args:
            inputs (tf.float32): Tensor of shape `[...,n]` containing the
                channel logits/llr values.

        Returns:
            `tf.float32`: Tensor of shape `[...,n]` or `[...,k]`
            (``return_infobits`` is True) containing bit-wise soft-estimates
            (or hard-decided bit-values) of all codeword bits (or info
            bits, respectively).

        Raises:
            ValueError: If ``inputs`` is not of shape `[batch_size, n]`.

            ValueError: If ``num_iter`` is not an integer greater (or equal)
                `0`.

            InvalidArgumentError: When rank(``inputs``)<2.
        """

        # Extract inputs
        if self._stateful:
            llr_ch, msg_vn = inputs
        else:
            llr_ch = inputs

        tf.debugging.assert_type(llr_ch, self.dtype, 'Invalid input dtype.')

        llr_ch_shape = llr_ch.get_shape().as_list()
        new_shape = [-1, llr_ch_shape[-1]]
        llr_ch_reshaped = tf.reshape(llr_ch, new_shape)
        batch_size = tf.shape(llr_ch_reshaped)[0]

        # invert if rate-matching output interleaver was applied as defined in
        # Sec. 5.4.2.2 in 38.212
        if self._encoder.num_bits_per_symbol is not None:
            llr_ch_reshaped = tf.gather(llr_ch_reshaped,
                                        self._encoder.out_int_inv,
                                        axis=-1)


        # undo puncturing of the first 2*Z bit positions
        llr_5g = tf.concat(
            [tf.zeros([batch_size, 2*self.encoder.z], self._output_dtype),
                          llr_ch_reshaped],
                          1)

        # undo puncturing of the last positions
        # total length must be n_ldpc, while llr_ch has length n
        # first 2*z positions are already added
        # -> add n_ldpc - n - 2Z punctured positions
        k_filler = self.encoder.k_ldpc - self.encoder.k # number of filler bits
        nb_punc_bits = ((self.encoder.n_ldpc - k_filler)
                                     - self.encoder.n - 2*self.encoder.z)


        llr_5g = tf.concat([llr_5g,
                   tf.zeros([batch_size, nb_punc_bits - self._nb_pruned_nodes],
                            self._output_dtype)],
                            1)

        # undo shortening (= add 0 positions after k bits, i.e. LLR=LLR_max)
        # the first k positions are the systematic bits
        x1 = tf.slice(llr_5g, [0,0], [batch_size, self.encoder.k])

        # parity part
        nb_par_bits = (self.encoder.n_ldpc - k_filler
                       - self.encoder.k - self._nb_pruned_nodes)
        x2 = tf.slice(llr_5g,
                      [0, self.encoder.k],
                      [batch_size, nb_par_bits])

        # negative sign due to logit definition
        z = -self._llr_max * tf.ones([batch_size, k_filler], self._output_dtype)

        llr_5g = tf.concat([x1, z, x2], 1)

        # and execute the decoder
        if not self._stateful:
            x_hat = super().call(llr_5g)
        else:
            x_hat,msg_vn = super().call([llr_5g, msg_vn])

        if self._return_infobits: # return only info bits
            # reconstruct u_hat # code is systematic
            u_hat = tf.slice(x_hat, [0,0], [batch_size, self.encoder.k])
            # Reshape u_hat so that it matches the original input dimensions
            output_shape = llr_ch_shape[0:-1] + [self.encoder.k]
            # overwrite first dimension as this could be None (Keras)
            output_shape[0] = -1
            u_reshaped = tf.reshape(u_hat, output_shape)

            # enable other output datatypes than tf.float32
            u_out = tf.cast(u_reshaped, self._output_dtype)

            if not self._stateful:
                return u_out
            else:
                return u_out, msg_vn

        else: # return all codeword bits
            # the transmitted CW bits are not the same as used during decoding
            # cf. last parts of 5G encoding function

            # remove last dim
            x = tf.reshape(x_hat, [batch_size, self._n_pruned])

            # remove filler bits at pos (k, k_ldpc)
            x_no_filler1 = tf.slice(x, [0, 0], [batch_size, self.encoder.k])

            x_no_filler2 = tf.slice(x,
                                    [0, self.encoder.k_ldpc],
                                    [batch_size,
                                    self._n_pruned-self.encoder.k_ldpc])

            x_no_filler = tf.concat([x_no_filler1, x_no_filler2], 1)

            # shorten the first 2*Z positions and end after n bits
            x_short = tf.slice(x_no_filler,
                               [0, 2*self.encoder.z],
                               [batch_size, self.encoder.n])

            # if used, apply rate-matching output interleaver again as
            # Sec. 5.4.2.2 in 38.212
            if self._encoder.num_bits_per_symbol is not None:
                x_short = tf.gather(x_short, self._encoder.out_int, axis=-1)

            # Reshape x_short so that it matches the original input dimensions
            # overwrite first dimension as this could be None (Keras)
            llr_ch_shape[0] = -1
            x_short= tf.reshape(x_short, llr_ch_shape)

            # enable other output datatypes than tf.float32
            x_out = tf.cast(x_short, self._output_dtype)

            if not self._stateful:
                return x_out
            else:
                return x_out, msg_vn
