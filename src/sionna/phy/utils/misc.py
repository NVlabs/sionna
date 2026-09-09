#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Miscellaneous utility functions of Sionna PHY and SYS"""

from abc import ABC, abstractmethod
import math
from numbers import Integral, Real
import time
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.interpolate import RectBivariateSpline, griddata

from sionna.phy import config, dtypes, Block
from sionna.phy.config import Precision
from sionna.phy.utils.metrics import count_errors, count_block_errors
from sionna.phy.utils.random import (  # pylint: disable=unused-import
    complex_normal,  # re-exported so `utils.misc.complex_normal` keeps working
    rand as _rand,
)

__all__ = [
    "lin_to_db",
    "db_to_lin",
    "watt_to_dbm",
    "dbm_to_watt",
    "ebnodb2no",
    "hard_decisions",
    "sample_bernoulli",
    "sim_ber",
    "to_list",
    "dict_keys_to_int",
    "scalar_to_shaped_tensor",
    "DeepUpdateDict",
    "Interpolate",
    "SplineGriddataInterpolation",
    "MCSDecoder",
    "TransportBlock",
    "SingleLinkChannel",
]


def lin_to_db(
    x: Union[float, torch.Tensor],
    precision: Optional[Precision] = None,
    device: Optional[str] = None,
) -> torch.Tensor:
    r"""Converts the input in linear scale to dB scale.

    :param x: Input value in linear scale.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for the output tensor. If `None` and `x` is a tensor,
        uses the device of `x`. Otherwise uses :attr:`~sionna.phy.config.Config.device`.

    :output x_db: Input value converted to dB.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.utils import lin_to_db

        x = torch.tensor(100.0)
        print(lin_to_db(x).item())
        # 20.0
    """
    if precision is None:
        dtype = config.dtype
    else:
        dtype = dtypes[precision]["torch"]["dtype"]

    # Explicit device wins, then the device of the input tensor, then config
    if device is None:
        if isinstance(x, torch.Tensor):
            device = x.device
        else:
            device = config.device

    # For existing tensors use .to() to avoid new_tensor under torch.compile
    if isinstance(x, torch.Tensor):
        x = x.to(dtype=dtype, device=device)
    else:
        x = torch.as_tensor(x, dtype=dtype, device=device)
    return 10.0 * torch.log10(x)


def db_to_lin(
    x: Union[float, torch.Tensor],
    precision: Optional[Precision] = None,
    device: Optional[str] = None,
) -> torch.Tensor:
    r"""Converts the input [dB] to linear scale.

    :param x: Input value in dB.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for the output tensor. If `None` and `x` is a tensor,
        uses the device of `x`. Otherwise uses :attr:`~sionna.phy.config.Config.device`.

    :output x_lin: Input value converted to linear scale.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.utils import db_to_lin

        x = torch.tensor(20.0)
        print(db_to_lin(x).item())
        # 100.0
    """
    if precision is None:
        dtype = config.dtype
    else:
        dtype = dtypes[precision]["torch"]["dtype"]

    # Explicit device wins, then the device of the input tensor, then config
    if device is None:
        if isinstance(x, torch.Tensor):
            device = x.device
        else:
            device = config.device

    # For existing tensors use .to() to avoid new_tensor under torch.compile
    if isinstance(x, torch.Tensor):
        x = x.to(dtype=dtype, device=device)
    else:
        x = torch.as_tensor(x, dtype=dtype, device=device)
    return 10.0 ** (x / 10.0)


def watt_to_dbm(
    x_w: Union[float, torch.Tensor],
    precision: Optional[Precision] = None,
    device: Optional[str] = None,
) -> torch.Tensor:
    r"""Converts the input [Watt] to dBm.

    :param x_w: Input value in Watt.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for the output tensor. If `None` and `x_w` is a tensor,
        uses the device of `x_w`. Otherwise uses :attr:`~sionna.phy.config.Config.device`.

    :output x_dbm: Input value converted to dBm.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.utils import watt_to_dbm

        x = torch.tensor(1.0)  # 1 Watt
        print(watt_to_dbm(x).item())
        # 30.0
    """
    return lin_to_db(x_w, precision=precision, device=device) + 30.0


def dbm_to_watt(
    x_dbm: Union[float, torch.Tensor],
    precision: Optional[Precision] = None,
    device: Optional[str] = None,
) -> torch.Tensor:
    r"""Converts the input [dBm] to Watt.

    :param x_dbm: Input value in dBm.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for the output tensor. If `None` and `x_dbm` is a tensor,
        uses the device of `x_dbm`. Otherwise uses :attr:`~sionna.phy.config.Config.device`.

    :output x_w: Input value converted to Watt.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.utils import dbm_to_watt

        x = torch.tensor(30.0)  # 30 dBm
        print(dbm_to_watt(x).item())
        # 1.0
    """
    return db_to_lin(x_dbm, precision=precision, device=device) * 0.001


def ebnodb2no(
    ebno_db: Union[float, torch.Tensor],
    num_bits_per_symbol: int,
    coderate: float,
    resource_grid: Optional[Any] = None,
    precision: Optional[Precision] = None,
    device: Optional[str] = None,
) -> torch.Tensor:
    r"""Computes the noise variance `No` for a given `Eb/No` in dB.

    The function takes into account the number of coded bits per constellation
    symbol, the coderate, as well as possible additional overheads related to
    OFDM transmissions, such as the cyclic prefix and pilots.

    The value of `No` is computed according to the following expression

    .. math::
        N_o = \left(\frac{E_b}{N_o} \frac{r M}{E_s}\right)^{-1}

    where :math:`2^M` is the constellation size, i.e., :math:`M` is the
    average number of coded bits per constellation symbol,
    :math:`E_s=1` is the average energy per constellation per symbol,
    :math:`r\in(0,1]` is the coderate,
    :math:`E_b` is the energy per information bit,
    and :math:`N_o` is the noise power spectral density.
    For OFDM transmissions, :math:`E_s` is scaled
    according to the ratio between the total number of resource elements in
    a resource grid with non-zero energy and the number
    of resource elements used for data transmission. Also the additionally
    transmitted energy during the cyclic prefix is taken into account, as
    well as the number of transmitted streams per transmitter.

    :param ebno_db: Eb/No value in dB.
    :param num_bits_per_symbol: Number of bits per symbol.
    :param coderate: Coderate.
    :param resource_grid: An optional resource grid for OFDM transmissions.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :output no: Value of :math:`N_o` in linear scale.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.utils import ebnodb2no

        no = ebnodb2no(ebno_db=10.0, num_bits_per_symbol=4, coderate=0.5)
        print(no.item())
    """
    if isinstance(num_bits_per_symbol, bool) or not isinstance(
        num_bits_per_symbol, Integral
    ):
        raise TypeError("num_bits_per_symbol must be an integer.")
    if num_bits_per_symbol <= 0:
        raise ValueError("num_bits_per_symbol must be positive.")
    if isinstance(coderate, bool) or not isinstance(coderate, Real):
        raise TypeError("coderate must be a real number.")
    if not math.isfinite(coderate) or not 0 < coderate <= 1:
        raise ValueError("coderate must be finite and within (0, 1].")

    if precision is None:
        dtype = config.dtype
    else:
        dtype = dtypes[precision]["torch"]["dtype"]

    if device is None:
        device = config.device

    # Use as_tensor to avoid copy warning from torch.compile when input is tensor
    ebno_db = torch.as_tensor(ebno_db, dtype=dtype, device=device)

    ebno = 10.0 ** (ebno_db / 10.0)

    energy_per_symbol = 1.0
    if resource_grid is not None:
        # Divide energy per symbol by the number of transmitted streams
        energy_per_symbol /= resource_grid.num_streams_per_tx

        # Number of nonzero energy symbols.
        # We do not account for the nulled DC and guard carriers.
        cp_overhead = resource_grid.cyclic_prefix_length / resource_grid.fft_size
        num_syms = (
            resource_grid.num_ofdm_symbols
            * (1 + cp_overhead)
            * resource_grid.num_effective_subcarriers
        )
        energy_per_symbol *= num_syms / resource_grid.num_data_symbols

    no = energy_per_symbol / (ebno * coderate * num_bits_per_symbol)
    return no


def hard_decisions(llr: torch.Tensor) -> torch.Tensor:
    r"""Transforms LLRs into hard decisions.

    Positive values are mapped to :math:`1`.
    Nonpositive values are mapped to :math:`0`.

    :param llr: Tensor of LLRs (must be non-complex dtype).

    :output b: Hard decisions with the same shape and dtype as ``llr``.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.utils import hard_decisions

        llr = torch.tensor([-1.5, 0.0, 2.3, -0.1])
        print(hard_decisions(llr))
        # tensor([0., 0., 1., 0.])
    """
    return (llr > 0).to(llr.dtype)


def sample_bernoulli(
    shape: Union[List[int], Tuple[int, ...], torch.Size],
    p: Union[float, torch.Tensor],
    precision: Optional[Precision] = None,
    device: Optional[str] = None,
) -> torch.Tensor:
    r"""Generates samples from a Bernoulli distribution with probability ``p``.

    :param shape: Shape of the tensor to sample.
    :param p: Probability (broadcastable with ``shape``). Values outside
        [0, 1] are rejected, except for a tensor-valued ``p``, where checking
        it would force a device synchronization.
    :param precision: Precision used for internal calculations.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :output samples: Binary samples (boolean tensor).

    .. rubric:: Examples

    .. code-block:: python

        from sionna.phy.utils import sample_bernoulli

        samples = sample_bernoulli([100], p=0.3)
        print(samples.sum().item())  # Approximately 30
    """
    if precision is None:
        dtype = config.dtype
    else:
        dtype = dtypes[precision]["torch"]["dtype"]

    if device is None:
        device = config.device

    # Validate on the host only. A scalar broadcasts against the samples for
    # free, whereas copying it to the device blocks until the queued work
    # drains, and reading back a device tensor would stall just the same.
    if isinstance(p, Real):
        if not (math.isfinite(p) and 0.0 <= p <= 1.0):
            raise ValueError("p must contain finite values within [0, 1].")
    elif not isinstance(p, torch.Tensor):
        values = np.asarray(p, dtype=np.float64)
        if not np.all(np.isfinite(values) & (values >= 0.0) & (values <= 1.0)):
            raise ValueError("p must contain finite values within [0, 1].")
        p = torch.as_tensor(p, dtype=dtype, device=device)

    generator = None if torch.compiler.is_compiling() else config.torch_rng(device)
    z = _rand(shape, dtype=dtype, device=device, generator=generator)
    return z < p


class _SimStatus:
    """Status codes for sim_ber SNR points."""

    NA = 0
    MAX_ITER = 1
    NO_ERROR = 2
    TARGET_BIT = 3
    TARGET_BLOCK = 4
    TARGET_BER = 5
    TARGET_BLER = 6
    CB_STOP = 7

    LABELS = {
        0: "not simulated",
        1: "reached max iterations",
        2: "no errors - early stop",
        3: "reached target bit errors",
        4: "reached target block errors",
        5: "reached target BER - early stop",
        6: "reached target BLER - early stop",
        7: "callback triggered stopping",
    }


_SIM_BER_HEADER = [
    "EbNo [dB]", "BER", "BLER", "bit errors", "num bits",
    "block errors", "num blocks", "runtime [s]", "status",
]

_SIM_BER_ROW_FMT = (
    "{: >9} |{: >11} |{: >11} |{: >12} |{: >12} "
    "|{: >13} |{: >12} |{: >12} |{: >10}"
)

_EARLY_STOP_MESSAGES = {
    _SimStatus.NO_ERROR: "no error occurred",
    _SimStatus.TARGET_BER: "target BER is reached",
    _SimStatus.TARGET_BLER: "target BLER is reached",
}


def _print_sim_progress(ebno_dbs, bit_errors, block_errors, nb_bits, nb_blocks,
                        status, snr_idx, mc_iter, max_mc_iter, elapsed,
                        is_final):
    """Print a single data row of sim_ber progress."""
    ber_val = float(bit_errors[snr_idx]) / float(nb_bits[snr_idx])
    if np.isnan(ber_val):
        ber_val = 0.0
    bler_val = float(block_errors[snr_idx]) / float(nb_blocks[snr_idx])
    if np.isnan(bler_val):
        bler_val = 0.0

    if status[snr_idx] == _SimStatus.NA:
        status_txt = f"iter: {mc_iter:.0f}/{max_mc_iter:.0f}"
    else:
        status_txt = _SimStatus.LABELS[int(status[snr_idx])]

    row = [
        str(np.round(ebno_dbs[snr_idx].cpu().numpy(), 3)),
        f"{ber_val:.4e}",
        f"{bler_val:.4e}",
        np.round(bit_errors[snr_idx], 0),
        np.round(nb_bits[snr_idx], 0),
        np.round(block_errors[snr_idx], 0),
        np.round(nb_blocks[snr_idx], 0),
        np.round(elapsed, 1),
        status_txt,
    ]

    print(_SIM_BER_ROW_FMT.format(*row), end="\n" if is_final else "\r")


def _check_mc_stopping(bit_errors_i, block_errors_i,
                       num_target_bit_errors, num_target_block_errors):
    """Return _SimStatus if an MC-level stopping condition is met, else None."""
    if (num_target_bit_errors is not None
            and bit_errors_i >= num_target_bit_errors):
        return _SimStatus.TARGET_BIT
    if (num_target_block_errors is not None
            and block_errors_i >= num_target_block_errors):
        return _SimStatus.TARGET_BLOCK
    return None


def _check_early_stop(bit_errors_i, block_errors_i, nb_bits_i, nb_blocks_i,
                      target_ber, target_bler):
    """Return _SimStatus if an early-stop condition across SNR points is met."""
    if nb_bits_i == 0 or nb_blocks_i == 0:
        return None
    if block_errors_i == 0:
        return _SimStatus.NO_ERROR
    if bit_errors_i / nb_bits_i < target_ber:
        return _SimStatus.TARGET_BER
    if block_errors_i / nb_blocks_i < target_bler:
        return _SimStatus.TARGET_BLER
    return None


def sim_ber(
    mc_fun: Callable,
    ebno_dbs: torch.Tensor,
    batch_size: int,
    max_mc_iter: int,
    soft_estimates: bool = False,
    num_target_bit_errors: Optional[int] = None,
    num_target_block_errors: Optional[int] = None,
    target_ber: Optional[float] = None,
    target_bler: Optional[float] = None,
    early_stop: bool = True,
    compile_mode: Optional[str] = None,
    verbose: bool = True,
    forward_keyboard_interrupt: bool = True,
    callback: Optional[Callable] = None,
    precision: Optional[Precision] = None,
    device: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Simulates until target number of errors is reached and returns BER/BLER.

    The simulation continues with the next SNR point if either
    ``num_target_bit_errors`` bit errors or ``num_target_block_errors`` block
    errors is achieved. Further, it continues with the next SNR point after
    ``max_mc_iter`` batches of size ``batch_size`` have been simulated.
    Early stopping allows to stop the simulation after the first error-free SNR
    point or after reaching a certain ``target_ber`` or ``target_bler``.

    :param mc_fun: Callable that yields the transmitted bits `b` and the
        receiver's estimate `b_hat` for a given ``batch_size`` and
        ``ebno_db``. If ``soft_estimates`` is `True`, `b_hat` is interpreted as
        logit.
    :param ebno_dbs: A tensor containing SNR points to be evaluated.
    :param batch_size: Batch-size for evaluation.
    :param max_mc_iter: Maximum number of Monte-Carlo iterations per SNR point.
    :param soft_estimates: If `True`, `b_hat` is interpreted as logit and an
        additional hard-decision is applied internally. Defaults to `False`.
    :param num_target_bit_errors: Target number of bit errors per SNR point
        until the simulation continues to next SNR point. Defaults to `None`.
    :param num_target_block_errors: Target number of block errors per SNR point
        until the simulation continues. Defaults to `None`.
    :param target_ber: The simulation stops after the first SNR point which
        achieves a lower bit error rate as specified by ``target_ber``.
        This requires ``early_stop`` to be `True`. Defaults to `None`.
    :param target_bler: The simulation stops after the first SNR point which
        achieves a lower block error rate as specified by ``target_bler``.
        This requires ``early_stop`` to be `True`. Defaults to `None`.
    :param early_stop: If `True`, the simulation stops after the first
        error-free SNR point (i.e., no error occurred after ``max_mc_iter``
        Monte-Carlo iterations). Defaults to `True`.
    :param compile_mode: A string describing the compilation mode of ``mc_fun``.
        If `None`, ``mc_fun`` is executed as is.
        Options: `None`, "default", "reduce-overhead", "max-autotune".
    :param verbose: If `True`, the current progress will be printed.
        Defaults to `True`.
    :param forward_keyboard_interrupt: If `False`, KeyboardInterrupts will be
        caught internally and not forwarded (e.g., will not stop outer loops).
        If `True`, the simulation ends and returns the intermediate simulation
        results. Defaults to `True`.
    :param callback: If specified, ``callback`` will be called after each
        Monte-Carlo step. Can be used for logging or advanced early stopping.
        Input signature of ``callback`` must match
        ``callback(mc_iter, snr_idx, ebno_dbs, bit_errors, block_errors,
        nb_bits, nb_blocks)`` where ``mc_iter`` denotes the number of processed
        batches for the current SNR point, ``snr_idx`` is the index of the
        current SNR point, ``ebno_dbs`` is the vector of all SNR points to be
        evaluated, ``bit_errors`` the vector of number of bit errors for each
        SNR point, ``block_errors`` the vector of number of block errors,
        ``nb_bits`` the vector of number of simulated bits, ``nb_blocks`` the
        vector of number of simulated blocks, respectively.
        If ``callback`` returns `sim_ber.CALLBACK_NEXT_SNR`, early stopping is
        detected and the simulation will continue with the next SNR point.
        If ``callback`` returns `sim_ber.CALLBACK_STOP`, the simulation is
        stopped immediately. For `sim_ber.CALLBACK_CONTINUE` continues with
        the simulation.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :output ber: [n], `torch.float`.
        Bit-error rate.

    :output bler: [n], `torch.float`.
        Block-error rate.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.utils import sim_ber

        def mc_fun(batch_size, ebno_db):
            # Simple example: generate random bits and add noise
            b = torch.randint(0, 2, (batch_size, 100))
            b_hat = b.clone()  # Perfect decoding for demo
            return b, b_hat

        ebno_dbs = torch.linspace(0, 10, 5)
        ber, bler = sim_ber(mc_fun, ebno_dbs, batch_size=100, max_mc_iter=10)
    """
    if precision is None:
        precision = config.precision
    if device is None:
        device = config.device

    dtype = dtypes[precision]["torch"]["dtype"]

    # Input validation
    if not isinstance(early_stop, bool):
        raise TypeError("early_stop must be bool.")
    if not isinstance(soft_estimates, bool):
        raise TypeError("soft_estimates must be bool.")
    if not isinstance(verbose, bool):
        raise TypeError("verbose must be bool.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral):
        raise TypeError("batch_size must be an integer.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if isinstance(max_mc_iter, bool) or not isinstance(max_mc_iter, Integral):
        raise TypeError("max_mc_iter must be an integer.")
    if max_mc_iter <= 0:
        raise ValueError("max_mc_iter must be positive.")

    # Handle target_ber / target_bler
    if target_ber is not None:
        if not early_stop:
            print("Warning: early stop is deactivated. target_ber is ignored.")
    else:
        target_ber = -1.0

    if target_bler is not None:
        if not early_stop:
            print("Warning: early stop is deactivated. target_bler is ignored.")
    else:
        target_bler = -1.0

    # Handle compilation mode
    if compile_mode is not None:
        if not isinstance(compile_mode, str):
            raise TypeError("compile_mode must be str or None.")
        # Apply torch.compile to mc_fun. Since shapes are fixed in BER
        # simulations (only scalar ebno_db values change), dynamic=False
        # is appropriate and avoids issues with complex algorithms like SCL/OSD.
        mc_fun = torch.compile(mc_fun, mode=compile_mode)

    # Initialize internal variables
    # Convert list or numpy array to tensor if needed
    if isinstance(ebno_dbs, list):
        ebno_dbs = torch.tensor(ebno_dbs)
    elif isinstance(ebno_dbs, np.ndarray):
        ebno_dbs = torch.from_numpy(ebno_dbs)
    ebno_dbs = ebno_dbs.to(dtype).to(device)
    num_points = ebno_dbs.shape[0]

    # Use numpy arrays for accumulation (int64 for precision)
    bit_errors = np.zeros(num_points, dtype=np.int64)
    block_errors = np.zeros(num_points, dtype=np.int64)
    nb_bits = np.zeros(num_points, dtype=np.int64)
    nb_blocks = np.zeros(num_points, dtype=np.int64)

    # Track status and runtime
    status = np.zeros(num_points)
    runtime = np.zeros(num_points)

    snr_idx = 0
    try:
        for snr_idx in range(num_points):
            start_time = time.perf_counter()
            cb_state = sim_ber.CALLBACK_CONTINUE
            mc_iter = 0

            for mc_iter in range(max_mc_iter):
                with torch.no_grad():
                    b, b_hat = mc_fun(
                        batch_size=batch_size, ebno_db=ebno_dbs[snr_idx])
                    if not isinstance(b, torch.Tensor) or not isinstance(
                        b_hat, torch.Tensor
                    ):
                        raise TypeError("mc_fun must return two tensors.")
                    if b.shape != b_hat.shape:
                        raise ValueError(
                            "mc_fun must return tensors with the same shape."
                        )
                    if b.ndim == 0 or b.numel() == 0 or b.shape[-1] == 0:
                        raise ValueError(
                            "mc_fun must return tensors with a non-empty bit dimension."
                        )
                    if soft_estimates:
                        b_hat = hard_decisions(b_hat)

                bit_errors[snr_idx] += count_errors(b, b_hat).item()
                block_errors[snr_idx] += count_block_errors(b, b_hat).item()
                nb_bits[snr_idx] += b.numel()
                nb_blocks[snr_idx] += b[..., -1].numel()

                cb_state = sim_ber.CALLBACK_CONTINUE
                if callback is not None:
                    cb_state = callback(
                        mc_iter, snr_idx, ebno_dbs,
                        torch.tensor(bit_errors, dtype=torch.int64),
                        torch.tensor(block_errors, dtype=torch.int64),
                        torch.tensor(nb_bits, dtype=torch.int64),
                        torch.tensor(nb_blocks, dtype=torch.int64),
                    )
                    if cb_state in (sim_ber.CALLBACK_STOP,
                                    sim_ber.CALLBACK_NEXT_SNR):
                        status[snr_idx] = _SimStatus.CB_STOP
                        break

                if verbose:
                    if snr_idx == 0 and mc_iter == 0:
                        print(_SIM_BER_ROW_FMT.format(*_SIM_BER_HEADER))
                        print("-" * 135)
                    _print_sim_progress(
                        ebno_dbs, bit_errors, block_errors, nb_bits,
                        nb_blocks, status, snr_idx, mc_iter, max_mc_iter,
                        time.perf_counter() - start_time, is_final=False,
                    )

                stop = _check_mc_stopping(
                    bit_errors[snr_idx], block_errors[snr_idx],
                    num_target_bit_errors, num_target_block_errors,
                )
                if stop is not None:
                    status[snr_idx] = stop
                    break
            else:
                status[snr_idx] = _SimStatus.MAX_ITER

            runtime[snr_idx] = time.perf_counter() - start_time

            if verbose:
                _print_sim_progress(
                    ebno_dbs, bit_errors, block_errors, nb_bits, nb_blocks,
                    status, snr_idx, mc_iter, max_mc_iter,
                    runtime[snr_idx], is_final=True,
                )

            if early_stop:
                reason = _check_early_stop(
                    bit_errors[snr_idx], block_errors[snr_idx],
                    nb_bits[snr_idx], nb_blocks[snr_idx],
                    target_ber, target_bler,
                )
                if reason is not None:
                    status[snr_idx] = reason
                    if verbose:
                        msg = _EARLY_STOP_MESSAGES[reason]
                        ebno_val = ebno_dbs[snr_idx].cpu().numpy()
                        print(
                            f"\nSimulation stopped as {msg} "
                            f"@ EbNo = {ebno_val:.1f} dB.\n"
                        )
                    break

            if cb_state is sim_ber.CALLBACK_STOP:
                status[snr_idx] = _SimStatus.CB_STOP
                if verbose:
                    ebno_val = ebno_dbs[snr_idx].cpu().numpy()
                    print(
                        f"\nSimulation stopped by callback function "
                        f"@ EbNo = {ebno_val:.1f} dB.\n"
                    )
                break

    except KeyboardInterrupt as e:
        if forward_keyboard_interrupt:
            raise e

        print(
            f"\nSimulation stopped by the user "
            f"@ EbNo = {ebno_dbs[snr_idx].cpu().numpy()} dB."
        )
        for idx in range(snr_idx + 1, num_points):
            bit_errors[idx] = -1
            block_errors[idx] = -1
            nb_bits[idx] = 1
            nb_blocks[idx] = 1

    with np.errstate(invalid="ignore"):
        ber = bit_errors.astype(np.float64) / nb_bits.astype(np.float64)
        bler = block_errors.astype(np.float64) / nb_blocks.astype(np.float64)
    ber = np.nan_to_num(ber, nan=0.0)
    bler = np.nan_to_num(bler, nan=0.0)

    return (torch.tensor(ber, dtype=dtype, device=device),
            torch.tensor(bler, dtype=dtype, device=device))


# Callback constants for sim_ber
sim_ber.CALLBACK_CONTINUE = None
sim_ber.CALLBACK_STOP = 2
sim_ber.CALLBACK_NEXT_SNR = 1


def to_list(x: Any) -> Optional[List]:
    """Converts the input to a list.

    :param x: Input to be converted to a list. Can be `list`, `float`, `int`,
        `str`, or `None`.

    :output x: Input converted to a list, or `None` if input is `None`.

    .. rubric:: Examples

    .. code-block:: python

        from sionna.phy.utils import to_list

        print(to_list(5))
        # [5]
        print(to_list([1, 2, 3]))
        # [1, 2, 3]
        print(to_list("hello"))
        # ['hello']
    """
    if x is not None:
        if isinstance(x, str) or not hasattr(x, "__len__"):
            x = [x]
        else:
            x = list(x)
    return x


def dict_keys_to_int(x: Any) -> Any:
    r"""Converts the string keys of an input dictionary to integers whenever
    possible.

    :param x: Input dictionary.

    :output x: Dictionary with integer keys where conversion is possible.

    .. rubric:: Examples

    .. code-block:: python

        from sionna.phy.utils import dict_keys_to_int

        dict_in = {'1': {'2': [45, '3']}, '4.3': 6, 'd': [5, '87']}
        print(dict_keys_to_int(dict_in))
        # {1: {'2': [45, '3']}, '4.3': 6, 'd': [5, '87']}
    """
    if isinstance(x, dict):
        x_new = {}
        for k, v in x.items():
            try:
                k_new = int(k)
            except ValueError:
                k_new = k
            x_new[k_new] = v
        return x_new
    else:
        return x


def scalar_to_shaped_tensor(
    inp: Union[int, float, bool, torch.Tensor],
    dtype: torch.dtype,
    shape: List[int],
    device: Optional[str] = None,
) -> torch.Tensor:
    r"""Converts a scalar input to a tensor of specified shape, or validates
    and casts an existing input tensor.

    If the input is a scalar, creates a tensor of the specified shape filled
    with that value. Otherwise, verifies the input tensor matches the required
    shape and casts it to the specified dtype.

    :param inp: Input value. If scalar (`int`, `float`, `bool`, or shapeless
        tensor), it will be used to fill a new tensor. If a shaped tensor, its
        shape must match the specified shape.
    :param dtype: Desired data type of the output tensor.
    :param shape: Required shape of the output tensor.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :output tensor: A tensor of shape ``shape`` and type ``dtype``. Either filled with
        the scalar input value or the input tensor cast to the specified dtype.

    .. rubric:: Examples

    .. code-block:: python

        import torch
        from sionna.phy.utils import scalar_to_shaped_tensor

        # From scalar
        t = scalar_to_shaped_tensor(5.0, torch.float32, [2, 3])
        print(t)
        # tensor([[5., 5., 5.],
        #         [5., 5., 5.]])

        # From tensor
        t = scalar_to_shaped_tensor(torch.ones(2, 3), torch.float64, [2, 3])
        print(t.dtype)
        # torch.float64
    """
    if device is None:
        device = config.device

    if isinstance(inp, (int, float, bool)):
        return torch.full(shape, inp, dtype=dtype, device=device)
    elif inp.dim() == 0:
        # Use broadcasting instead of .item() to avoid torch.compile graph breaks
        return inp.to(dtype=dtype, device=device) + torch.zeros(
            shape, dtype=dtype, device=device
        )
    else:
        if list(inp.shape) != list(shape):
            raise ValueError(
                f"Inconsistent shape: expected {list(shape)}, got {list(inp.shape)}"
            )
        return inp.to(dtype=dtype, device=device)


class DeepUpdateDict(dict):
    r"""Dictionary class inheriting from `dict` enabling nested merging of
    the dictionary with a new one."""

    def _deep_update(
        self,
        dict_orig: dict,
        delta: dict,
        stop_at_keys: Tuple = (),
    ) -> None:
        for key in delta:
            if (
                (key not in dict_orig)
                or (not isinstance(delta[key], dict))
                or (not isinstance(dict_orig[key], dict))
                or (key in to_list(stop_at_keys))
            ):
                dict_orig[key] = delta[key]
            else:
                self._deep_update(dict_orig[key], delta[key], stop_at_keys=stop_at_keys)

    def deep_update(self, delta: dict, stop_at_keys: Tuple = ()) -> None:
        r"""Merges ``self`` with the input ``delta`` in nested fashion.

        In case of conflict, the values of the new dictionary prevail.
        The two dictionaries are merged at intermediate keys ``stop_at_keys``,
        if provided.

        :param delta: Dictionary to be merged with ``self``.
        :param stop_at_keys: Tuple of keys at which the subtree of ``delta``
            replaces the corresponding subtree of ``self``.

        .. rubric:: Examples

        .. code-block:: python

            from sionna.phy.utils import DeepUpdateDict

            # Merge without conflicts
            dict1 = DeepUpdateDict(
                {'a': 1, 'b': {'b1': 10, 'b2': 20}})
            dict_delta1 = {'c': -2, 'b': {'b3': 30}}
            dict1.deep_update(dict_delta1)
            print(dict1)
            # {'a': 1, 'b': {'b1': 10, 'b2': 20, 'b3': 30}, 'c': -2}

            # Compare against the classic "update" method, which is not nested
            dict1 = DeepUpdateDict(
                {'a': 1, 'b': {'b1': 10, 'b2': 20}})
            dict1.update(dict_delta1)
            print(dict1)
            # {'a': 1, 'b': {'b3': 30}, 'c': -2}

            # Handle key conflicts
            dict2 = DeepUpdateDict(
                {'a': 1, 'b': {'b1': 10, 'b2': 20}})
            dict_delta2 = {'a': -2, 'b': {'b1': {'f': 3, 'g': 4}}}
            dict2.deep_update(dict_delta2)
            print(dict2)
            # {'a': -2, 'b': {'b1': {'f': 3, 'g': 4}, 'b2': 20}}

            # Merge at intermediate keys
            dict2 = DeepUpdateDict(
                {'a': 1, 'b': {'b1': 10, 'b2': 20}})
            dict2.deep_update(dict_delta2, stop_at_keys='b')
            print(dict2)
            # {'a': -2, 'b': {'b1': {'f': 3, 'g': 4}}}
        """
        self._deep_update(self, delta, stop_at_keys=stop_at_keys)


class Interpolate(ABC):
    r"""Abstract class template for interpolating data defined on unstructured
    or rectangular grids.

    Used in :class:`~sionna.sys.PHYAbstraction` for BLER and SNR interpolation.
    """

    @abstractmethod
    def unstruct(
        self,
        z: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        x_interp: np.ndarray,
        y_interp: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        r"""Interpolates unstructured data.

        :param z: Co-domain sample values of shape [N].
            Informally, ``z`` = f(``x``, ``y``).
        :param x: First coordinate of the domain sample values of shape [N].
        :param y: Second coordinate of the domain sample values of shape [N].
        :param x_interp: Interpolation grid for the first (x) coordinate of
            shape [L]. Typically, :math:`L \gg N`.
        :param y_interp: Interpolation grid for the second (y) coordinate of
            shape [J]. Typically, :math:`J \gg N`.

        :output z_interp: Interpolated data of shape [L, J].
        """

    @abstractmethod
    def struct(
        self,
        z: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        x_interp: np.ndarray,
        y_interp: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        r"""Interpolates data structured in rectangular grids.

        :param z: Co-domain sample values of shape [N, M].
            Informally, ``z`` = f(``x``, ``y``).
        :param x: First coordinate of the domain sample values of shape [N].
        :param y: Second coordinate of the domain sample values of shape [M].
        :param x_interp: Interpolation grid for the first (x) coordinate of
            shape [L]. Typically, :math:`L \gg N`.
        :param y_interp: Interpolation grid for the second (y) coordinate of
            shape [J]. Typically, :math:`J \gg M`.

        :output z_interp: Interpolated data of shape [L, J].
        """


class SplineGriddataInterpolation(Interpolate):
    r"""Interpolates data defined on rectangular or unstructured grids via
    Scipy's `interpolate.RectBivariateSpline` and `interpolate.griddata`,
    respectively.

    Inherits from :class:`~sionna.phy.utils.Interpolate`.
    """

    def struct(
        self,
        z: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        x_interp: np.ndarray,
        y_interp: np.ndarray,
        spline_degree: int = 1,
        **kwargs,
    ) -> np.ndarray:
        r"""Performs spline interpolation via Scipy's
        `interpolate.RectBivariateSpline`.

        :param z: Co-domain sample values of shape [N, M].
            Informally, ``z`` = f(``x``, ``y``).
        :param x: First coordinate of the domain sample values of shape [N].
        :param y: Second coordinate of the domain sample values of shape [M].
        :param x_interp: Interpolation grid for the first (x) coordinate of
            shape [L]. Typically, :math:`L \gg N`.
        :param y_interp: Interpolation grid for the second (y) coordinate of
            shape [J]. Typically, :math:`J \gg M`.
        :param spline_degree: Spline interpolation degree. Defaults to 1.

        :output z_interp: Interpolated data of shape [L, J].
        """
        if len(x) <= spline_degree:
            raise ValueError("Too few points for interpolation")

        # Compute log10(mat), replacing zeros with a "low" value to avoid inf
        log_mat = np.zeros(z.shape)
        mat_is0 = z == 0
        if np.all(mat_is0):
            return np.zeros((len(x_interp), len(y_interp)), dtype=float)
        if mat_is0.sum() > 0:
            log_mat_not0 = np.log10(z[~mat_is0])
            min_log_mat_not0 = min(log_mat_not0)
            log_mat[~mat_is0] = log_mat_not0
            log_mat[mat_is0] = min(log_mat_not0) - 2
        else:
            log_mat = np.log10(z)
            min_log_mat_not0 = -np.inf

        # Spline interpolation on log10(BLER) for numerical precision
        fun_interp = RectBivariateSpline(
            x, y, log_mat, kx=spline_degree, ky=spline_degree
        )

        log_mat_interp = fun_interp(x_interp, y_interp)

        # Retrieve the BLER
        mat_interp = np.power(10, log_mat_interp)
        # Replace "low" values with zeros
        mat_interp[mat_interp < 10**min_log_mat_not0] = 0

        return mat_interp

    def unstruct(
        self,
        z: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        x_interp: np.ndarray,
        y_interp: np.ndarray,
        griddata_method: str = "linear",
        **kwargs,
    ) -> np.ndarray:
        r"""Interpolates unstructured data via Scipy's `interpolate.griddata`.

        :param z: Co-domain sample values of shape [N].
            Informally, ``z`` = f(``x``, ``y``).
        :param x: First coordinate of the domain sample values of shape [N].
        :param y: Second coordinate of the domain sample values of shape [N].
        :param x_interp: Interpolation grid for the first (x) coordinate of
            shape [L]. Typically, :math:`L \gg N`.
        :param y_interp: Interpolation grid for the second (y) coordinate of
            shape [J]. Typically, :math:`J \gg N`.
        :param griddata_method: Interpolation method ("linear", "nearest",
            "cubic"). See Scipy's `interpolate.griddata` for more details.
            Defaults to "linear".

        :output z_interp: Interpolated data of shape [L, J].
        """
        y_grid, x_grid = np.meshgrid(y_interp, x_interp)
        z_interp = griddata(
            list(zip(y, x)), z, (y_grid, x_grid), method=griddata_method
        )
        return z_interp


class MCSDecoder(Block):
    r"""Class template for mapping a Modulation and Coding Scheme (MCS) index
    to the corresponding modulation order, i.e., number of bits per symbol,
    and coderate.

    :input mcs_index: [...], `torch.int32`. MCS index.

    :input mcs_table_index: [...], `torch.int32`. MCS table index. Different tables contain different mappings.

    :input mcs_category: [...], `torch.int32`. Table category which may correspond, e.g., to uplink or
        downlink transmission.

    :input check_index_validity: `bool`. If `True`, a ValueError is thrown if the input MCS indices are not
        valid for the given configuration. Defaults to `True`.

    :output modulation_order: [...], `torch.int32`. Modulation order corresponding to the input MCS index.

    :output coderate: [...], `torch.float`. Coderate corresponding to the input MCS index.
    """

    def call(
        self,
        mcs_index: torch.Tensor,
        mcs_table_index: torch.Tensor,
        mcs_category: torch.Tensor,
        check_index_validity: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process MCS index to return modulation order and coderate."""
        raise NotImplementedError("Subclasses must implement 'call'.")


class TransportBlock(Block):
    r"""Class template for computing the number and size (measured in number of
    bits) of code blocks within a transport block, given the modulation order,
    coderate and the total number of coded bits of a transport block.

    Used in :class:`~sionna.sys.PHYAbstraction`.

    :input modulation_order: [...], `torch.int32`. Modulation order, i.e., number of bits per symbol.

    :input target_coderate: [...], `torch.float`. Target coderate.

    :input num_coded_bits: [...], `torch.float`. Total number of coded bits across all codewords.

    :output cb_size: [...], `torch.int32`. Code block (CB) size, i.e., number of information bits per code block.

    :output num_cb: [...], `torch.int32`. Number of code blocks that the transport block is segmented into.
    """

    @abstractmethod
    def call(
        self,
        modulation_order: torch.Tensor,
        target_coderate: torch.Tensor,
        num_coded_bits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute code block size and count."""
        ...


class SingleLinkChannel(Block):
    r"""Class template for simulating single-link, i.e., single-carrier and
    single-stream, channels.

    Used for generating BLER tables in
    :meth:`~sionna.sys.PHYAbstraction.new_bler_table`.

    :param num_bits_per_symbol: Number of bits per symbol, i.e., modulation
        order.
    :param num_info_bits: Number of information bits per code block.
    :param target_coderate: Target code rate, i.e., the target ratio between
        the information and the coded bits within a block.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input batch_size: `int`. Size of the simulation batches.

    :input ebno_db: `float`. Eb/No value in dB.

    :output bits: [``batch_size``, ``num_info_bits``], `torch.int`. Transmitted bits.

    :output bits_hat: [``batch_size``, ``num_info_bits``], `torch.int`. Decoded bits.

    .. rubric:: Examples

    .. code-block:: python

        # Create a channel (subclass implementation required)
        channel = SingleLinkChannel(
            num_bits_per_symbol=4,
            num_info_bits=1024,
            target_coderate=0.5
        )

        # Call returns (bits, bits_hat)
        bits, bits_hat = channel(batch_size=100, ebno_db=5.0)
    """

    def __init__(
        self,
        num_bits_per_symbol: Optional[int],
        num_info_bits: Optional[int],
        target_coderate: Optional[float],
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
    ):
        super().__init__(precision=precision, device=device)
        self._num_bits_per_symbol: Optional[int] = None
        self._num_info_bits: Optional[int] = None
        self._target_coderate: Optional[float] = None
        self._num_coded_bits: Optional[float] = None

        if num_bits_per_symbol is not None:
            self.num_bits_per_symbol = num_bits_per_symbol
        if target_coderate is not None:
            self.target_coderate = target_coderate
        if num_info_bits is not None:
            self.num_info_bits = num_info_bits

    @property
    def num_bits_per_symbol(self) -> Optional[int]:
        """Get/set the modulation order."""
        return self._num_bits_per_symbol

    @num_bits_per_symbol.setter
    def num_bits_per_symbol(self, value: int) -> None:
        if not (value > 0):
            raise ValueError("num_bits_per_symbol must be a positive integer")
        self._num_bits_per_symbol = int(value)
        self.set_num_coded_bits()

    @property
    def num_info_bits(self) -> Optional[int]:
        """Get/set the number of information bits per code block."""
        return self._num_info_bits

    @num_info_bits.setter
    def num_info_bits(self, value: int) -> None:
        if not (value > 0):
            raise ValueError("num_info_bits must be a positive integer")
        self._num_info_bits = int(value)
        self.set_num_coded_bits()

    @property
    def target_coderate(self) -> Optional[float]:
        """Get/set the target coderate."""
        return self._target_coderate

    @target_coderate.setter
    def target_coderate(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("target_coderate must be a real number")
        if not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError("target_coderate must be finite and within (0, 1]")
        self._target_coderate = float(value)
        self.set_num_coded_bits()

    @property
    def num_coded_bits(self) -> Optional[float]:
        """Number of coded bits in a code block (read-only)."""
        return self._num_coded_bits

    def set_num_coded_bits(self) -> None:
        """Compute the number of coded bits per code block."""
        if (
            self.num_info_bits is not None
            and self.target_coderate is not None
            and self.num_bits_per_symbol is not None
        ):
            num_coded_bits = self.num_info_bits / self.target_coderate
            # Ensure num_coded_bits is a multiple of num_bits_per_symbol
            self._num_coded_bits = (
                np.ceil(num_coded_bits / self.num_bits_per_symbol)
                * self.num_bits_per_symbol
            )

    @abstractmethod
    def call(
        self,
        batch_size: int,
        ebno_db: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simulate the channel."""
