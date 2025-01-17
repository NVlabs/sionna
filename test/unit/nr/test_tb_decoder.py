#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Test NR receiver components."""
import pytest
import unittest
import numpy as np
import tensorflow as tf
from sionna.nr import TBEncoder, TBDecoder
from sionna.utils import BinarySource

@pytest.mark.usefixtures("only_gpu")
class TestTBDecoder(unittest.TestCase):
    """Test TBDecoder"""

    def test_identity(self):
        """Test that receiver can recover info bits."""

        source = BinarySource()

        # define test parameters
        # the tests cover the following scenarios
        # 1.) Single CB segmentation
        # 2.) Long CB / multiple CWs
        # 3.) Deactivated scrambler
        # 4.) N-dimensional inputs
        # 5.) zero padding

        bs = [[10], [10], [10], [10, 13, 14], [2]]
        tb_sizes = [6656, 60456, 984, 984, 50000]
        num_coded_bits = [13440, 100800, 2880, 2880, 100000]
        num_bits_per_symbols = [4, 8, 2, 2, 4]
        num_layers = [1, 1, 2, 4, 2]
        n_rntis = [1337, 45678, 1337, 1337, 1337]
        sc_ids = [1, 1023, 2, 42, 42]
        use_scramblers = [True, True, False, True, True]

        for i,_ in enumerate(tb_sizes):
            encoder = TBEncoder(
                        target_tb_size=tb_sizes[i],
                        num_coded_bits=num_coded_bits[i],
                        target_coderate=tb_sizes[i]/num_coded_bits[i],
                        num_bits_per_symbol=num_bits_per_symbols[i],
                        num_layers=num_layers[i],
                        n_rnti=n_rntis[i], # used for scrambling
                        n_id=sc_ids[i], # used for scrambling
                        channel_type="PUSCH",
                        codeword_index=0,
                        use_scrambler=use_scramblers[i],
                        verbose=False,
                        output_dtype=tf.float32,
                        )

            decoder = TBDecoder(encoder=encoder,
                                num_bp_iter=10,
                                cn_type="minsum")

            u = source(bs[i] + [encoder.k])
            c = encoder(u)
            llr_ch = (2*c-1) # apply bpsk
            u_hat, crc_status = decoder(llr_ch)

            # all info bits can be recovered
            self.assertTrue(np.array_equal(u.numpy(), u_hat.numpy()))
            # all crc checks are valid
            self.assertTrue(np.array_equal(crc_status.numpy(),
                                           np.ones_like(crc_status.numpy())))

    def test_scrambling(self):
        """Test that (de-)scrambling works as expected."""

        source = BinarySource()
        bs = 10

        n_rnti_ref = 1337
        sc_id_ref = 42

        # add offset to both scrambling indices
        n_rnti_offset = [0, 1, 0]
        sc_id_offset = [0, 0, 1]

        init = True
        for i, _ in enumerate(n_rnti_offset):
            encoder = TBEncoder(
                        target_tb_size=60456,
                        num_coded_bits=100800,
                        target_coderate=60456/100800,
                        num_bits_per_symbol=4,
                        n_rnti=n_rnti_ref + n_rnti_offset[i],
                        n_id=sc_id_ref + sc_id_offset[i],
                        use_scrambler=True,
                        verbose=False)

            if init: # init decoder only once
                decoder = TBDecoder(encoder=encoder,
                                      num_bp_iter=20,
                                      cn_type="minsum")

            # as scrambling IDs do not match, all TBs must be wrong
            if not init:
                u = source([bs, encoder.k])
                c = encoder(u)
                llr_ch = (2*c-1) # apply bpsk
                u_hat, crc_status = decoder(llr_ch)
                # all info bits can be recovered
                self.assertFalse(np.array_equal(u.numpy(), u_hat.numpy()))
                # all CRC checks are wrong
                self.assertTrue(np.array_equal(crc_status.numpy(),
                                             np.zeros_like(crc_status.numpy())))
            init = False

    def test_crc(self):
        """Test that crc detects the correct erroneous positions."""

        source = BinarySource()
        bs = 10

        encoder = TBEncoder(
                    target_tb_size=60456,
                    num_coded_bits=100800,
                    target_coderate=60456/100800,
                    num_bits_per_symbol=4,
                    n_rnti=12367,
                    n_id=312,
                    use_scrambler=True)

        decoder = TBDecoder(encoder=encoder,
                            num_bp_iter=20,
                            cn_type="minsum")


        u = source([bs, encoder.k])
        c = encoder(u)
        llr_ch = (2*c-1) # apply bpsk

        # destroy TB at batch index 7
        # all others are correctly received
        err_pos = 7
        llr_ch = llr_ch.numpy()
        llr_ch[err_pos, 500:590] = -10 # overwrite some llr positions

        u_hat, crc_status = decoder(llr_ch)

        # all CRC checks are correct expect at pos err_ pos
        crc_status_ref = np.ones_like(crc_status.numpy())
        crc_status_ref[err_pos] = 0

        self.assertTrue(np.array_equal(crc_status.numpy(),crc_status_ref))

    def test_tf_fun(self):
        """Test tf.function"""

        @tf.function
        def run_graph(bs):
            u = source([bs, encoder.k])
            c = encoder(u)
            llr_ch = (2*c-1) # apply bpsk
            return decoder(llr_ch)

        @tf.function(jit_compile=True)
        def run_graph_xla(bs):
            u = source([bs, encoder.k])
            c = encoder(u)
            llr_ch = (2*c-1) # apply bpsk
            return decoder(llr_ch)

        source = BinarySource()
        bs = 10

        encoder = TBEncoder(
                    target_tb_size=60456,
                    num_coded_bits=100800,
                    target_coderate=60456/100800,
                    num_bits_per_symbol=4,
                    n_rnti=12367,
                    n_id=312,
                    use_scrambler=True)

        decoder = TBDecoder(encoder=encoder,
                              num_bp_iter=20,
                              cn_type="minsum")

        x = run_graph(bs)
        self.assertTrue(x[0].shape[0]==bs) # verify correct size

        # change batch_size
        x = run_graph(2*bs)
        self.assertTrue(x[0].shape[0]==2*bs) # verify correct size

        # build for dynamic bs
        x = run_graph(tf.constant(bs))
        self.assertTrue(x[0].shape[0]==bs) # verify correct size
        # change dynamic bs
        x = run_graph(tf.constant(bs+1))
        self.assertTrue(x[0].shape[0]==(bs+1)) # verify correct size

        # again with jit_compile=True
        x = run_graph_xla(bs)
        self.assertTrue(x[0].shape[0]==bs) # verify correct size

        # change batch_size
        x = run_graph_xla(2*bs)
        self.assertTrue(x[0].shape[0]==2*bs) # verify correct size

        # build for dynamic bs
        x = run_graph_xla(tf.constant(bs))
        self.assertTrue(x[0].shape[0]==bs) # verify correct size
        # change dynamic bs
        x = run_graph_xla(tf.constant(bs+1))
        self.assertTrue(x[0].shape[0]==(bs+1)) # verify correct size
