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

    ldpc_params = LDPC5GEncoder(k, n)
    n_cb = ldpc_params.n_cb

    # Reference encoder assumes no rate matching
    encoder_ref = LDPC5GEncoder(k, n_cb, allow_low_rates=True)

    # HARQ encoder
    encoder = LDPC5GEncoder(k, n, allow_low_rates=True)
    starts = encoder.get_rv_starts()

    # Generate encoded bits
    source = BinarySource()

    @tf.function(jit_compile=use_xla)
    def encode_harq():
        bits = source([batch_size, k])
        x_ref = encoder_ref(bits)  # Shape: [batch_size, n_cb]
        x_ref = tf.roll(x_ref, shift=starts["rv0"], axis=-1)  # Undo shift of RV0
        x = encoder(bits, rv=rv_list)  # Shape: [batch_size, num_rv, n]
        return bits, x_ref, x

    if use_graph_mode:
        bits, x_ref, x = encode_harq()
    else:
        # In eager mode, XLA is not applicable
        bits = source([batch_size, k])
        x_ref = encoder_ref(bits)  # Shape: [batch_size, n_cb]
        x_ref = tf.roll(x_ref, shift=starts["rv0"], axis=-1)  # Undo shift of RV0
        x = encoder(bits, rv=rv_list)  # Shape: [batch_size, num_rv, n]

    # Validate encoded bits
    assert x.shape == [batch_size, num_rv, n]

    for i, rv in enumerate(rv_list):
        start = starts[rv]
        x_ref_unrolled = tf.roll(x_ref, shift=-start, axis=-1)
        x_ref_unrolled = x_ref_unrolled[:, :n]  # Adjust to match n length
        assert tf.equal(x_ref_unrolled, x[:, i, :]).numpy().all(), \
            f"RV {rv} encoding mismatch in mode Graph={use_graph_mode}, XLA={use_xla}"

    # Print test completion info
    mode_str = f"Graph={use_graph_mode}, XLA={use_xla}"
    print(f"HARQ encoder test passed for k={k}, n={n}, {mode_str}")

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
    encoder = LDPC5GEncoder(k, n, allow_low_rates=True)
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
    encoder = LDPC5GEncoder(k, n, allow_low_rates=True)
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

def test_custom_accumulator():
    """Test that custom accumulator callable works syntactically.

    This test verifies that the HARQ decoder can accept and use a custom
    accumulator function instead of the default one. This is a syntactic
    test to ensure the callable replacement mechanism works correctly.
    """
    k, n = 100, 200
    batch_size = 3
    encoder = LDPC5GEncoder(k, n)

    # Define a simple custom accumulator (just for syntax testing)
    def simple_accumulator(llr_accumulated, llr_new, transmission_idx):
        """Simple custom accumulator: same as default but with logging capability."""
        if llr_accumulated is None:
            return llr_new
        else:
            # Could add logging here if needed
            return llr_accumulated + llr_new

    # Create decoder with custom accumulator
    decoder = LDPC5GDecoder(encoder,
                           harq_mode=True,
                           accumulator=simple_accumulator,
                           num_iter=5)

    # Test that it works syntactically
    source = GaussianPriorSource()
    rv_list = ["rv0", "rv1"]
    llr_ch = source([batch_size, len(rv_list), n], 0.5)

    result = decoder(llr_ch, rv=rv_list)
    assert result.shape == [batch_size, k]
