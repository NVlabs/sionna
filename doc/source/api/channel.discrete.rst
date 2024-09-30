========
Discrete
========

This module provides layers and functions that implement channel
models with discrete input/output alphabets.

All channel models support binary inputs :math:`x \in \{0, 1\}` and `bipolar`
inputs :math:`x \in \{-1, 1\}`, respectively. In the later case, it is assumed
that each `0` is mapped to `-1`.

The channels can either return discrete values or log-likelihood ratios (LLRs).
These LLRs describe the channel transition probabilities
:math:`L(y|X=1)=L(X=1|y)+L_a(X=1)` where :math:`L_a(X=1)=\operatorname{log} \frac{P(X=1)}{P(X=0)}` depends only on the `a priori` probability of :math:`X=1`. These LLRs equal the `a posteriori` probability if :math:`P(X=1)=P(X=0)=0.5`.

Further, the channel reliability parameter :math:`p_b` can be either a scalar
value or a tensor of any shape that can be broadcasted to the input. This
allows for the efficient implementation of
channels with non-uniform error probabilities.

The channel models are based on the `Gumble-softmax trick` [GumbleSoftmax]_ to
ensure differentiability of the channel w.r.t. to the channel reliability
parameter. Please see [LearningShaping]_ for further details.


Setting-up:

>>> bsc = BinarySymmetricChannel(return_llrs=False, bipolar_input=False)

Running:

>>> x = tf.zeros((128,)) # x is the channel input
>>> pb = 0.1 # pb is the bit flipping probability
>>> y = bsc((x, pb))


BinaryMemorylessChannel
=======================

.. autoclass:: sionna.channel.BinaryMemorylessChannel
   :members:
   :exclude-members: call, build

BinarySymmetricChannel
======================

.. autoclass:: sionna.channel.BinarySymmetricChannel
   :members:
   :exclude-members: call, build

BinaryErasureChannel
====================

.. autoclass:: sionna.channel.BinaryErasureChannel
   :members:
   :exclude-members: call, build

BinaryZChannel
==============

.. autoclass:: sionna.channel.BinaryZChannel
   :members:
   :exclude-members: call, build


References:
   .. [GumbleSoftmax] E\. Jang, G\. Shixiang, and B\. Poole. `"Categorical reparameterization with gumbel-softmax,"` arXiv preprint arXiv:1611.01144 (2016).

   .. [LearningShaping] M\. Stark, F\. Ait Aoudia, and J\. Hoydis. `"Joint learning of geometric and probabilistic constellation shaping,"` 2019 IEEE Globecom Workshops (GC Wkshps). IEEE, 2019.

