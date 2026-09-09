#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Definition of Sionna Block"""

from typing import Any, Optional
from .object import Object
from .config import Precision

__all__ = ["Block"]


class Block(Object):
    """Abstract class for Sionna PHY processing blocks.

    All Sionna PHY processing blocks inherit from this class. It provides
    automatic input casting to the block's precision, lazy building based
    on input shapes, and compatibility with ``torch.compile``.

    :param precision: Precision used for internal calculations and outputs.
        `None` (default) | ``"single"`` | ``"double"``.
        If `None`, :attr:`~sionna.phy.config.Config.precision` is used.
        Defaults to `None`.
    :param device: Device for computation (e.g., ``'cpu'``, ``'cuda:0'``).
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.
        Defaults to `None`.

    .. rubric:: Notes

    Input conversion and lazy building happen on every call, so a block
    behaves the same in eager mode and under ``torch.compile``.

    Compiling a block with a traceable :meth:`build` before its first call costs
    one additional trace, because :attr:`built` changes during that call.
    Calling the block once eagerly beforehand avoids the additional trace.

    With ``fullgraph=True``, :meth:`build` must itself be traceable. Call the
    block once eagerly first if :meth:`build` creates
    :class:`torch.nn.Parameter` instances or uses unsupported Python or NumPy
    operations. Otherwise, compilation raises
    ``torch._dynamo.exc.Unsupported``. Without ``fullgraph=True``, unsupported
    build operations cause graph breaks instead.
    """

    def __init__(
        self,
        *args: Any,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(precision=precision, device=device)
        self._built = False

    @property
    def built(self) -> bool:
        """Indicates if the block's build function was called."""
        return self._built

    def build(self, *arg_shapes, **kwarg_shapes):
        r"""Initialize the block based on the inputs' shapes.

        Subclasses can override this method to create tensors or
        sub-blocks whose sizes depend on the input shapes.

        :param \*arg_shapes: Shapes of the positional arguments. Can be
            tuples (for tensors) or nested structures thereof (for
            lists/dicts of tensors).
        :param \*\*kwarg_shapes: Shapes of the keyword arguments. Can be
            tuples (for tensors) or nested structures thereof (for
            lists/dicts of tensors).
        """

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Convert inputs, build the block if needed, and run its forward pass."""
        args = self._convert(args)
        kwargs = self._convert(kwargs)

        if not self._built:
            arg_shapes = self._get_shape(args)
            kwarg_shapes = self._get_shape(kwargs)
            self.build(*arg_shapes, **kwarg_shapes)
            self._built = True

        return super().__call__(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Delegates to :meth:`call`."""
        return self.call(*args, **kwargs)

    def call(self, *args: Any, **kwargs: Any) -> Any:
        """Process inputs. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement 'call'.")
