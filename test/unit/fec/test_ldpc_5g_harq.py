#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import tensorflow as tf
import pytest
import numpy as np

from sionna.phy.fec.ldpc.decoding import LDPC5GDecoder
from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
from sionna.phy.fec.utils import GaussianPriorSource
from sionna.phy.utils import ebnodb2no, compute_ber
from sionna.phy.mapping import BinarySource
from sionna.phy.channel import AWGN
from sionna.phy.mapping import Mapper, Demapper, Constellation


def _spec_k0(rv_name, n_cb, bg):
    """RV start k0 from TS 38.212 Table 5.4.2.1-2 for I_LBRM=0.

    Written in table form with floor operators to mirror the spec:
      BG1: floor(N_cb * [0, 17, 33, 56] / 66)
      BG2: floor(N_cb * [0, 13, 25, 43] / 50)
    """
    if bg == "bg1":
        numerators = {"rv0": 0, "rv1": 17, "rv2": 33, "rv3": 56}
        denominator = 66
    elif bg == "bg2":
        numerators = {"rv0": 0, "rv1": 13, "rv2": 25, "rv3": 43}
        denominator = 50
    else:
        raise ValueError(f"Unsupported basegraph {bg}.")

    if rv_name not in numerators:
        raise ValueError(f"Invalid RV name {rv_name}.")

    # floor() from the spec table
    return (n_cb * numerators[rv_name]) // denominator


def _build_uncompressed_rm_buffer(encoder, bits):
    """Build uncompressed RM buffer d in TS 38.212 indexing space.

    Returns the sequence d of length N_cb (I_LBRM=0), i.e., after removing the
    first 2*z bits and before removing filler bits.
    """
    bits = tf.reshape(bits, [-1, encoder.k])
    batch_size = tf.shape(bits)[0]

    u_fill = tf.concat([
        bits,
        tf.zeros([batch_size, encoder.k_ldpc - encoder.k], encoder.rdtype)
    ], axis=1)

    c = encoder._encode_fast(u_fill)
    c = tf.reshape(c, [batch_size, encoder.n_ldpc])
    n_cb = encoder.n_ldpc - 2*encoder.z
    d = c[:, 2*encoder.z:2*encoder.z + n_cb]
    return d


def _spec_rate_match_from_d(d, encoder, rv_name, e_len):
    """Rate-match according to TS 38.212 §5.4.2.1 with filler skipping.

    Args:
        d: ndarray, shape [B, N_cb] in uncompressed spec indexing space.
        encoder: LDPC5GEncoder instance.
        rv_name: One of {"rv0", "rv1", "rv2", "rv3"}.
        e_len: Number of output bits E.

    Returns:
        ndarray of shape [B, E] containing RV-selected bits.
    """
    n_cb = encoder.n_ldpc - 2*encoder.z
    bg = encoder._bg
    k0 = _spec_k0(rv_name, n_cb, bg)

    filler_start = encoder.k - 2*encoder.z
    filler_end = filler_start + encoder.k_filler
    null_mask = np.zeros(n_cb, dtype=bool)
    null_mask[filler_start:filler_end] = True

    out_idx = []
    j = 0
    while len(out_idx) < e_len:
        idx = (k0 + j) % n_cb
        # TS 38.212: skip NULL entries in circular buffer d.
        if not null_mask[idx]:
            out_idx.append(idx)
        j += 1

    return d[:, out_idx]

#############################
# Test cases for LDPC5G HARQ
#############################

@pytest.mark.parametrize("k_n", [(400, 900), (5000, 12000)])
@pytest.mark.parametrize("rv_list", [["rv0", "rv2", "rv3", "rv1"]])
@pytest.mark.parametrize("use_graph_mode,use_xla", [
    (True, True),   # Graph mode with XLA
    (True, False),  # Graph mode without XLA
    (False, False)  # Eager mode (XLA not applicable)
])
def test_harq_encoder(k_n, rv_list, use_graph_mode, use_xla):
    """Test HARQ encoder functionality with different RV combinations.

    This test verifies that the HARQ encoder correctly generates codewords
    for different redundancy versions (RVs) by comparing against a reference
    encoder with circular shifting.

    Tests:
    - Various code rates (k/n combinations)
    - HARQ transmissions (4 RVs)
    - Correct RV-specific codeword generation
    - Encoder in HARQ matches expectations
    - Graph mode and XLA compilation compatibility
    """
    # Set random seed for reproducibility
    tf.random.set_seed(42)

    k, n = k_n
    batch_size = 2
    num_rv = len(rv_list)

    # HARQ encoder
    encoder = LDPC5GEncoder(k, n)
    starts_comp = encoder.rv_starts_comp
    n_cb_comp = encoder.n_cb_comp

    # The reference encoder needs to produce the full circular buffer for
    # comparison.  However, rate k/n_cb_comp can fall below the encoder's
    # minimum (1/3 BG1, 1/5 BG2) due to filler bits or small k_b, so we
    # use the lowest accepted rate instead.
    n_ref = 5 * k if 5 * k < n_cb_comp else 3 * k  # BG2 or BG1
    encoder_ref = LDPC5GEncoder(k, n_ref)

    # Generate encoded bits
    source = BinarySource()

    @tf.function(jit_compile=use_xla)
    def encode_harq():
        bits = source([batch_size, k])
        x_ref = encoder_ref(bits)  # Shape: [batch_size, n_ref]; uses default RV0
        x = encoder(bits, rv=rv_list)  # Shape: [batch_size, num_rv, n]; uses list of RVs
        return bits, x_ref, x

    if use_graph_mode:
        bits, x_ref, x = encode_harq()
    else:
        # In eager mode, XLA is not applicable
        bits = source([batch_size, k])
        x_ref = encoder_ref(bits)  # Shape: [batch_size, n_ref]; uses default RV0
        x = encoder(bits, rv=rv_list)  # Shape: [batch_size, num_rv, n]; uses list of RVs
    
    # Pad x_ref to n_cb_comp with marker value -1 on the right side
    x_ref = tf.pad(x_ref, [[0, 0], [0, n_cb_comp - n_ref]], mode="CONSTANT", constant_values=-1)

    # Validate encoded bits
    assert x.shape == [batch_size, num_rv, n]

    for i, rv in enumerate(rv_list):
        start = starts_comp[rv]
        # Align reference codeword for comparison
        x_ref_unrolled = tf.roll(x_ref, shift=-start+starts_comp["rv0"], axis=-1)
        x_ref_unrolled = x_ref_unrolled[:, :n]  # Adjust to match length n
        # Compare only non-marker positions (ignore -1 markers)
        mask = x_ref_unrolled != -1
        x_rv = x[:, i, :]
        matches = tf.logical_or(~mask, tf.equal(x_ref_unrolled, x_rv))
        all_match = tf.reduce_all(matches).numpy()
        assert all_match, f"RV {rv} encoding mismatch in mode Graph={use_graph_mode}, XLA={use_xla}"

    # Print test completion info
    mode_str = f"Graph={use_graph_mode}, XLA={use_xla}"
    print(f"HARQ encoder test passed for k={k}, n={n}, {mode_str}")


def test_harq_rv_selection_matches_spec():
    """Comprehensive HARQ RV conformance against TS 38.212 indexing.

    The test sweeps many valid (k, n) operating points and representative RV
    patterns. For each case, it compares encoder
    HARQ output to a spec-faithful reference built from uncompressed RM buffer
    indexing with filler-bit skipping.
    """
    tf.random.set_seed(1234)

    rv_patterns = [
        ["rv0", "rv3"],
        ["rv0", "rv2", "rv3", "rv1"],
    ]

    k_values = [
        56, 104, 136, 248, 344, 472, 632, 856,
        1032, 1480, 1832, 2360, 2840, 3320, 3832,
        4312, 5032, 5960, 7032, 7848, 8416,
    ]
    rate_targets = [0.92, 0.85, 0.77, 0.69, 0.61, 0.53, 0.45, 0.37, 0.34, 0.28, 0.24, 0.22]

    source = BinarySource()
    batch_size = 2
    num_cases_tested = 0

    for k in k_values:
        for rate in rate_targets:
            n = int(np.ceil(k / rate))

            # For k > 3824, auto-selection can force BG1, which requires
            # r >= 1/3 in this implementation.
            # Skip only this known unsupported region instead of catching all
            # ValueErrors, so unexpected failures remain visible.
            if k > 3824 and rate < (1/3):
                continue

            encoder = LDPC5GEncoder(k, n)

            bits = source([batch_size, k])
            d = _build_uncompressed_rm_buffer(encoder, bits).numpy()

            for rv_list in rv_patterns:
                x_impl = encoder(bits, rv=rv_list).numpy()
                x_ref = np.stack([
                    _spec_rate_match_from_d(d, encoder, rv_name, n)
                    for rv_name in rv_list
                ], axis=1)

                if not np.array_equal(x_impl, x_ref):
                    pytest.fail(
                        "HARQ RV mismatch against TS 38.212 reference: "
                        f"k={k}, n={n}, rv_list={rv_list}, "
                        f"bg={encoder._bg}, z={encoder.z}, "
                        f"k_filler={encoder.k_filler}, n_cb={encoder.n_cb}."
                    )

                num_cases_tested += 1

    # Ensure the sweep really exercised many operating points and RV patterns.
    assert num_cases_tested >= 300, (
        f"Insufficient HARQ sweep coverage ({num_cases_tested} cases)."
    )

@pytest.mark.parametrize("k_n_esno", [(300, 6*140, -0.5), (4500, 6*1800, 1.0)])
@pytest.mark.parametrize("use_graph_mode", [True, False])
@pytest.mark.parametrize("use_xla", [True, False])
def test_harq_decoder(k_n_esno, use_graph_mode, use_xla):
    """Test HARQ decoder functionality with different RV combinations.

    This test verifies that the HARQ decoder correctly decodes codewords
    for different redundancy versions (RVs).

    Tests:
    - Various code rates (k/n combinations)
    - HARQ transmissions (4 RVs)
    - Decoder in HARQ matches expectations
    - Graph mode and XLA compilation compatibility
    """
    # Set random seed for reproducibility
    tf.random.set_seed(42)

    k, n, esno = k_n_esno
    batch_size = 3

    source = BinarySource()
    encoder = LDPC5GEncoder(k, n)
    decoder = LDPC5GDecoder(encoder, harq_mode=True, num_iter=20)
    channel = AWGN()

    # Set up 64QAM
    constellation = Constellation("qam", num_bits_per_symbol=6)
    mapper = Mapper(constellation=constellation)
    demapper = Demapper(demapping_method="app", constellation=constellation)

    # HARQ transmission with different RV levels
    rv_list = ["rv0", "rv2", "rv3", "rv1"]
    # Assume EsNo and therefore we set the parameters to 1
    no = ebnodb2no(esno, num_bits_per_symbol=1, coderate=1)

    @tf.function(jit_compile=use_xla)
    def run_e2e_harq():
        # Generate information bits
        bits = source([batch_size, k])

        # Encode and transmit all RVs at once
        codeword = encoder(bits, rv=rv_list)  # Shape: [batch_size, num_rv, n]
        x = mapper(codeword)
        y = channel(x, no)
        llr = demapper(y, no)
        decoded_bits = decoder(llr, rv=rv_list)

        return bits, decoded_bits

    if use_graph_mode:
        bits, decoded_bits = run_e2e_harq()
    else:
        # Generate information bits
        bits = source([batch_size, k])

        # Encode and transmit all RVs at once
        codeword = encoder(bits, rv=rv_list)  # Shape: [batch_size, num_rv, n]
        x = mapper(codeword)
        y = channel(x, no)
        llr = demapper(y, no)
        decoded_bits = decoder(llr, rv=rv_list)

    # Check decoding success
    ber = compute_ber(bits, decoded_bits)

    # Print BER for analysis
    mode_str = f"Graph={use_graph_mode}, XLA={use_xla}"
    print(f"EsNo: {esno} dB, BER: {ber.numpy():.6f} ({mode_str})")

    # Verify the system runs end-to-end and produces reasonable output
    assert ber <= 0.05  # BER should not exceed 5%
    assert not tf.reduce_any(tf.math.is_nan(decoded_bits))  # No NaN values
    # Note: With proper SNR levels and HARQ, BER should be quite low

@pytest.mark.parametrize("use_graph_mode", [True, False])
@pytest.mark.parametrize("use_xla", [True, False])
def test_return_codeword(use_graph_mode, use_xla):
    """Test HARQ with return_infobits=False to get full codewords per RV.

    This test verifies that the HARQ decoder correctly returns full codewords
    (not just info bits) and that the decoded bits match the transmitted bits
    after RV accumulation.

    Tests:
    - Graph mode and XLA compilation compatibility
    - End-to-end transmission: encoder -> channel -> decoder with hard output comparison
    - RV-specific codeword recovery matches transmitted codewords
    """
    tf.random.set_seed(42)

    k, n = 150, 6*100
    batch_size = 3

    # Set up components similar to decoder test
    source = BinarySource()
    encoder = LDPC5GEncoder(k, n)
    decoder = LDPC5GDecoder(encoder,
                            harq_mode=True,
                            return_infobits=False,  # Get full codewords
                            hard_out=True,
                            num_iter=20)
    channel = AWGN()

    # Set up 64QAM like in decoder test
    constellation = Constellation("qam", num_bits_per_symbol=6)
    mapper = Mapper(constellation=constellation)
    demapper = Demapper(demapping_method="app", constellation=constellation)

    # HARQ transmission with different RV levels
    rv_list = ["rv0", "rv2", "rv3", "rv1"]

    esno_db = -1.0  # SNR for testing
    no = ebnodb2no(esno_db, num_bits_per_symbol=1, coderate=1)

    @tf.function(jit_compile=use_xla)
    def run_e2e_harq():
        # Generate information bits
        bits = source([batch_size, k])

        # Encode and transmit all RVs - get reference transmitted codewords
        codeword_tx = encoder(bits, rv=rv_list)  # Shape: [batch_size, num_rv, n]
        x = mapper(codeword_tx)
        y = channel(x, no)
        llr = demapper(y, no)

        # Decode to get full codewords per RV
        decoded_codewords = decoder(llr, rv=rv_list)  # Should be [batch_size, num_rv, n]

        return codeword_tx, decoded_codewords

    if use_graph_mode:
        codeword_tx, decoded_codewords = run_e2e_harq()
    else:
        # Generate information bits
        bits = source([batch_size, k])

        # Encode and transmit all RVs - get reference transmitted codewords
        codeword_tx = encoder(bits, rv=rv_list)  # Shape: [batch_size, num_rv, n]
        x = mapper(codeword_tx)
        y = channel(x, no)
        llr = demapper(y, no)

        # Decode to get full codewords per RV
        decoded_codewords = decoder(llr, rv=rv_list)  # Should be [batch_size, num_rv, n]

    # Check output shape
    expected_shape = [batch_size, len(rv_list), n]  # Per-RV results
    assert decoded_codewords.shape == expected_shape

    # Calculate BER for each RV and overall
    ber_total = 0.0
    for i, rv in enumerate(rv_list):
        tx_rv = codeword_tx[:, i, :]  # [batch_size, n]
        decoded_rv = decoded_codewords[:, i, :]  # [batch_size, n]

        ber = tf.reduce_sum(tf.cast(tf.not_equal(tf.cast(tx_rv, tf.float32),
                                                 tf.cast(decoded_rv > 0, tf.float32)),
                                                 tf.float32))
        ber /= (batch_size * n)
        ber_total += ber

    ber_total /= len(rv_list)

    # Print test completion info
    mode_str = f"Graph={use_graph_mode}, XLA={use_xla}"
    print(f"BER = {ber_total:.4f} ({mode_str})")
    assert ber_total <= 0.05, f"BER: {ber_total:.4f} ({mode_str})"

@pytest.mark.parametrize("use_state", [True, False])
def test_harq_with_decoder_state(use_state):
    """Test HARQ functionality with decoder state management.

    The LDPC decoder can optionally return its internal state (edge messages
    from the belief propagation algorithm). This test verifies that HARQ mode
    works correctly both with and without state return.

    Tests:
    - return_state=True: Decoder should return (result, state) tuple
    - return_state=False: Decoder should return only result
    - State tensor should have proper dimensions when returned
    - Normal decoding should work in both cases
    """
    k, n = 100, 200
    batch_size = 6
    encoder = LDPC5GEncoder(k, n)
    decoder = LDPC5GDecoder(encoder,
                           harq_mode=True,
                           return_state=use_state,
                           num_iter=5)

    source = GaussianPriorSource()
    rv_list = ["rv0", "rv1", "rv3"]
    llr_ch = source([batch_size, len(rv_list), n], 0.5)

    if use_state:
        result, state = decoder(llr_ch, rv=rv_list)
        assert state is not None
        assert state.shape[0] > 0  # Should have some edges
        assert state.shape[1] == batch_size
        assert result.shape == [batch_size, k]  # Default HARQ accumulates
    else:
        result = decoder(llr_ch, rv=rv_list)
        assert result.shape == [batch_size, k]

@pytest.mark.parametrize("k_n", [(100, 200), (4000, 7000)])
def test_harq_error_conditions(k_n):
    """Test error handling in HARQ mode.

    This test verifies that the HARQ decoder properly handles invalid inputs
    and edge cases. Good error handling prevents silent failures and helps
    users identify configuration problems.

    Tests:
    - Invalid RV names should raise ValueError
    - Graceful handling of dimension mismatches (implementation-dependent)
    - Validates that encoder and decoder both validate RV names properly
    """
    k, n = k_n
    encoder = LDPC5GEncoder(k, n)
    decoder = LDPC5GDecoder(encoder, harq_mode=True)

    batch_size = 3

    # Test dimension mismatch - this may or may not raise an error depending on implementation
    # The decoder might handle this gracefully, so we just check it doesn't crash
    try:
        llr_ch = tf.random.normal([batch_size, 2, n])  # 2 RVs in tensor
        result = decoder(llr_ch, rv=["rv0", "rv1", "rv2"])  # 3 RVs in list
        # If no error, just verify output is reasonable
        assert result.shape[0] == batch_size  # Batch dimension should be preserved
        assert result.shape[1] == k  # Should return info bits by default
        print(f"Dimension mismatch handled gracefully, result shape: {result.shape}")
    except (ValueError, tf.errors.InvalidArgumentError) as e:
        # Error is also acceptable for mismatched dimensions
        print(f"Dimension mismatch raised error as expected: {e}")

    # Test invalid RV names in decoder - this should raise a ValueError
    llr_ch = tf.random.normal([batch_size, 1, n])
    with pytest.raises(ValueError):
        decoder(llr_ch, rv=["invalid_rv"])

    # Test invalid RV names in encoder - this should also raise a ValueError
    bits = tf.cast(tf.random.uniform([batch_size, k], maxval=2, dtype=tf.int32), tf.float32)
    with pytest.raises(ValueError):
        encoder(bits, rv=["invalid_rv"])

    # Test that valid RV names work for both encoder and decoder
    try:
        encoded = encoder(bits, rv=["rv0", "rv1"])
        assert encoded.shape == [batch_size, 2, n]

        llr_ch = tf.random.normal([batch_size, 2, n])
        result = decoder(llr_ch, rv=["rv0", "rv1"])
        assert result.shape == [batch_size, k]
        print("Valid RV tests passed")
    except Exception as e:
        pytest.fail(f"Valid RV test failed unexpectedly: {e}")

@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("dtype_in", [tf.float32, tf.float64])
def test_harq_dtypes(precision, dtype_in):
    """Test HARQ with different data types and precision settings.

    Neural network frameworks often support multiple numerical precisions.
    Single precision (float32) is faster but less accurate, while double
    precision (float64) is slower but more accurate. This test ensures
    HARQ works correctly with both.

    Tests:
    - Input data types: tf.float32 and tf.float64
    - Decoder precision settings: "single" and "double"
    - Output data type consistency with precision setting
    - Internal state data type consistency
    - Validates numerical precision handling in HARQ mode
    """

    k, n = 100, 200
    batch_size = 8
    encoder = LDPC5GEncoder(k, n)
    decoder = LDPC5GDecoder(encoder, harq_mode=True, precision=precision, return_state=True)

    rv_list = ["rv0", "rv1"]
    llr_ch = tf.zeros([batch_size, len(rv_list), n], dtype_in)

    result, state = decoder(llr_ch, rv=rv_list)

    # Check output precision
    if precision == "single":
        assert result.dtype == tf.float32
        assert state.dtype == tf.float32
    else:
        assert result.dtype == tf.float64
        assert state.dtype == tf.float64


@pytest.mark.parametrize("k_n", [(100, 200), (400, 900)])
@pytest.mark.parametrize("batch_shape", [
    [2, 3],      # 2D batch
    [2, 3, 4],   # 3D batch
])
def test_harq_multidimensional_batch(k_n, batch_shape):
    """Test HARQ encoder/decoder with multi-dimensional batch shapes.

    This test verifies that the encoder and decoder correctly handle
    multiple batch dimensions, not just a single batch dimension.
    The output should match element-by-element when compared to
    processing each batch element individually.

    Tests:
    - Multi-dimensional batch shapes work correctly
    - Results match single-batch processing
    - Both encoder and decoder preserve batch dimensions
    """
    k, n = k_n
    rv_list = ["rv0", "rv2"]
    num_rv = len(rv_list)

    # Set up encoder and decoder
    encoder = LDPC5GEncoder(k, n)
    decoder = LDPC5GDecoder(encoder, harq_mode=True, num_iter=10)

    # Generate source bits with multi-dimensional batch
    source = BinarySource()
    u = source(batch_shape + [k])

    # Encode with multi-dimensional batch
    c = encoder(u, rv=rv_list)
    assert c.shape == batch_shape + [num_rv, n]

    # Add noise
    source_llr = GaussianPriorSource()
    llr = source_llr(batch_shape + [num_rv, n], 0.8)

    # Decode with multi-dimensional batch
    u_hat = decoder(llr, rv=rv_list)
    assert u_hat.shape == batch_shape + [k]

    # Verify against element-by-element processing
    # Flatten all batch dimensions except the last one
    total_batch_size = np.prod(batch_shape)
    u_flat = tf.reshape(u, [total_batch_size, k])
    c_flat = tf.reshape(c, [total_batch_size, num_rv, n])
    llr_flat = tf.reshape(llr, [total_batch_size, num_rv, n])
    u_hat_flat = tf.reshape(u_hat, [total_batch_size, k])

    # Process each batch element individually and compare
    for i in range(total_batch_size):
        # Encode single element
        c_ref = encoder(u_flat[i:i+1, :], rv=rv_list)
        assert np.allclose(c_ref.numpy(), c_flat[i:i+1, :, :].numpy())

        # Decode single element
        u_hat_ref = decoder(llr_flat[i:i+1, :, :], rv=rv_list)
        assert np.allclose(u_hat_ref.numpy(), u_hat_flat[i:i+1, :].numpy())
