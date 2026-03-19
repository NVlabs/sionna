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

The channel models are based on the `Gumble-softmax trick` :cite:p:`GumbleSoftmax` to
ensure differentiability of the channel w.r.t. to the channel reliability
parameter. Please see :cite:p:`LearningShaping` for further details.


Setting-up:

>>> bsc = BinarySymmetricChannel(return_llrs=False, bipolar_input=False)

Running:

>>> x = torch.zeros((128,)) # x is the channel input
>>> pb = 0.1 # pb is the bit flipping probability
>>> y = bsc((x, pb))

.. autoclass:: sionna.phy.channel.BinaryErasureChannel
   :members:
   :exclude-members: call, build

.. autoclass:: sionna.phy.channel.BinaryMemorylessChannel
   :members:
   :exclude-members: call, build

.. autoclass:: sionna.phy.channel.BinarySymmetricChannel
   :members:
   :exclude-members: call, build

.. autoclass:: sionna.phy.channel.BinaryZChannel
   :members:
   :exclude-members: call, build
