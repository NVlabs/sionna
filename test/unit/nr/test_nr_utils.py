#
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0#

import unittest
import numpy as np
import tensorflow as tf
from sionna.phy.nr.utils import decode_mcs_index, generate_prng_seq, \
    calculate_tb_size
from utils import calculate_tb_size_numpy, decode_mcs_index_numpy
from sionna.phy.utils.tensors import random_tensor_from_values, enumerate_indices


class TestNRUtils(unittest.TestCase):
    """Test nr_utils function"""

    def test_mcs_pdsch(self):
        """Test MCS selection for PDSCH against values from 38.214."""

        # Tab. 5.1.3.1-1
        qs = [2,2,2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,6,6,6,6,6,6,6,6,6,6,6,6]
        rs = [120,157,193,251,308,379,449,526,602,679,340,378,434,490,553,616,
              658,438,466,517,567,616,666,719,772,822,873,910,948]

        for bpsk in (True, False): # no impact
            for idx, q in enumerate(qs):
                m, r = decode_mcs_index(mcs_index=idx,
                                        table_index=1,
                                        is_pusch=False,
                                        pi2bpsk=bpsk)
                self.assertTrue(m==q)
                self.assertTrue(rs[idx]/1024==r.numpy())

        # test that next index raises error
        with self.assertRaises(tf.errors.InvalidArgumentError):
            decode_mcs_index(mcs_index=idx+1,
                             table_index=1,
                             is_pusch=False,
                             pi2bpsk=False)

        # Tab. 5.1.3.1-2
        qs = [2,2,2,2,2,4,4,4,4,4,4,6,6,6,6,6,6,6,6,6,8,8,8,8,8,8,8,8]
        rs = [120,193,308,449,602,378,434,490,553,616,658,466,517,567,616,666,
              719,772,822,873,682.5,711,754,797,841,885,916.5,948]

        for bpsk in (True, False): # no impact
            for idx, q in enumerate(qs):
                m, r = decode_mcs_index(mcs_index=idx,
                                  table_index=2,
                                  is_pusch=False,
                                  pi2bpsk=bpsk)
                self.assertTrue(m==q)
                self.assertTrue(rs[idx]/1024==r.numpy())

        # test that next index raises error
        with self.assertRaises(tf.errors.InvalidArgumentError):
            decode_mcs_index(mcs_index=idx+1,
                       table_index=2,
                       is_pusch=False,
                       pi2bpsk=False)

        # Tab. 5.1.3.1-3
        qs = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,4,4,4,4,4,4,6,6,6,6,6,6,6,6]
        rs = [30,40,50,64,78,99,120,157,193,251,308,379,449,526,602,340,378,434,
              490,553,616,438,466,517,567,616,666,719,772]

        for bpsk in (True, False): # no impact
            for idx, q in enumerate(qs):
                m, r = decode_mcs_index(mcs_index=idx,
                                  table_index=3,
                                  is_pusch=False,
                                  pi2bpsk=bpsk)
                self.assertTrue(m==q)
                self.assertTrue(rs[idx]/1024==r.numpy())

        # test that next index raises error
        with self.assertRaises(tf.errors.InvalidArgumentError):
            decode_mcs_index(mcs_index=idx+1,
                       table_index=3,
                       is_pusch=False,
                       pi2bpsk=False)

        # Tab. 5.1.3.1-4
        qs = [2,2,2,4,4,4,6,6,6,6,6,6,6,6,6,8,8,8,8,8,8,8,8,10,10,10,10]
        rs = [120,193,449,378,490,616,466,517,567,616,666,719,772,822,873,682.5,711,754,797,841,885,916.5,948,805.5,853,900.5,948]

        for bpsk in (True, False): # no impact
            for idx, q in enumerate(qs):
                m, r = decode_mcs_index(mcs_index=idx,
                                  table_index=4,
                                  is_pusch=False,
                                  pi2bpsk=bpsk)
                self.assertTrue(m==q)
                self.assertTrue(rs[idx]/1024==r.numpy())

        # test that next index raises error
        with self.assertRaises(tf.errors.InvalidArgumentError):
            decode_mcs_index(mcs_index=idx+1,
                       table_index=4,
                       is_pusch=False,
                       pi2bpsk=False)


    def test_mcs_pusch(self):
        """Test MCS selection for PUSCH against values from 38.214."""

        # without precoding
        # Tab. 5.1.3.1-1
        qs = [2,2,2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,6,6,6,6,6,6,6,6,6,6,6,6]
        rs = [120,157,193,251,308,379,449,526,602,679,340,378,434,490,553,616,
              658,438,466,517,567,616,666,719,772,822,873,910,948]

        for bpsk in (True, False): # no impact
            for idx, q in enumerate(qs):
                m, r = decode_mcs_index(mcs_index=idx,
                                  table_index=1,
                                  is_pusch=True,
                                  pi2bpsk=bpsk,
                                  transform_precoding=False)
                self.assertTrue(m==q)
                self.assertTrue(rs[idx]/1024==r.numpy())

        # test that next index raises error
        with self.assertRaises(tf.errors.InvalidArgumentError):
            decode_mcs_index(mcs_index=idx+1,
                       table_index=1,
                       is_pusch=True,
                       pi2bpsk=False,
                       transform_precoding=False)

        # Tab. 5.1.3.1-2
        qs = [2,2,2,2,2,4,4,4,4,4,4,6,6,6,6,6,6,6,6,6,8,8,8,8,8,8,8,8]
        rs = [120,193,308,449,602,378,434,490,553,616,658,466,517,567,616,666,
              719,772,822,873,682.5,711,754,797,841,885,916.5,948]

        for bpsk in (True, False): # no impact
            for idx, q in enumerate(qs):
                m, r = decode_mcs_index(mcs_index=idx,
                                  table_index=2,
                                  is_pusch=True,
                                  pi2bpsk=bpsk,
                                  transform_precoding=False)
                self.assertTrue(m==q)
                self.assertTrue(rs[idx]/1024==r.numpy())

        # test that next index raises error
        with self.assertRaises(tf.errors.InvalidArgumentError):
            decode_mcs_index(mcs_index=idx+1,
                       table_index=2,
                       is_pusch=True,
                       pi2bpsk=False,
                       transform_precoding=False)

        # Tab. 5.1.3.1-3
        qs = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,4,4,4,4,4,4,6,6,6,6,6,6,6,6]
        rs = [30,40,50,64,78,99,120,157,193,251,308,379,449,526,602,340,378,434,
              490,553,616,438,466,517,567,616,666,719,772]

        for bpsk in (True, False): # no impact
            for idx, q in enumerate(qs):
                m, r = decode_mcs_index(mcs_index=idx,
                                  table_index=3,
                                  is_pusch=True,
                                  pi2bpsk=bpsk,
                                  transform_precoding=False)
                self.assertTrue(m==q)
                self.assertTrue(rs[idx]/1024==r.numpy())

        # test that next index raises error
        with self.assertRaises(tf.errors.InvalidArgumentError):
            decode_mcs_index(mcs_index=idx+1,
                       table_index=3,
                       is_pusch=True,
                       pi2bpsk=False,
                       transform_precoding=False)

        #### with precoding
        # Tab. 6.1.4.1-1
        pi2bpsk = False #(q=2)
        qs = [2,2,2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,6,6,6,6,6,6,6,6,6,6,6]
        rs = [120,157,193,251,308,379,449,526,602,679,340,378,434,490,553,616,
              658,466,517,567,616,666,719,772,822,873,910,948]

        for idx, q in enumerate(qs):
            m, r = decode_mcs_index(mcs_index=idx,
                              table_index=1,
                              transform_precoding=True,
                              is_pusch=True,
                              pi2bpsk=pi2bpsk)
            self.assertTrue(m==q)
            self.assertTrue(rs[idx]/1024==r.numpy())

        # test that next index raises error
        with self.assertRaises(tf.errors.InvalidArgumentError):
            decode_mcs_index(mcs_index=idx+1,
                       table_index=1,
                       is_pusch=True,
                       pi2bpsk=pi2bpsk,
                       transform_precoding=True)

        # Tab. 6.1.4.1-1
        pi2bpsk = True #(q=1)
        qs = [1,1,2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,6,6,6,6,6,6,6,6,6,6,6]
        rs = [240,314,193,251,308,379,449,526,602,679,340,378,434,490,553,616,
              658,466,517,567,616,666,719,772,822,873,910,948]

        for idx, q in enumerate(qs):
            m, r = decode_mcs_index(mcs_index=idx,
                              table_index=1,
                              is_pusch=True,
                              transform_precoding=True,
                              pi2bpsk=pi2bpsk)
            self.assertTrue(m==q)
            self.assertTrue(rs[idx]/1024==r.numpy())

        # test that next index raises error
        with self.assertRaises(tf.errors.InvalidArgumentError):
            decode_mcs_index(mcs_index=idx+1,
                       table_index=1,
                       is_pusch=True,
                       pi2bpsk=pi2bpsk,
                       transform_precoding=True)

        # Tab. 6.1.4.1-2
        pi2bpsk = False #(q=2)
        qs = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,4,6,6,6,6]
        rs = [30,40,50,64,78,99,120,157,193,251,308,379,449,526,602,679,378,434,
              490,553,616,658,699,772,567,616,666,772]

        for idx, q in enumerate(qs):
            m, r = decode_mcs_index(mcs_index=idx,
                              table_index=2,
                              is_pusch=True,
                              pi2bpsk=pi2bpsk,
                              transform_precoding=True)
            self.assertTrue(m==q)
            self.assertTrue(rs[idx]/1024==r.numpy())

        # test that next index raises error
        with self.assertRaises(tf.errors.InvalidArgumentError):
            decode_mcs_index(mcs_index=idx+1,
                       table_index=2,
                       is_pusch=True,
                       pi2bpsk=pi2bpsk,
                       transform_precoding=True)

        # Tab. 6.1.4.1-2
        pi2bpsk = True #(q=1)
        qs = [1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,4,6,6,6,6]
        rs = [60,80,100,128,156,198,120,157,193,251,308,379,449,526,602,679,378,
              434,490,553,616,658,699,772,567,616,666,772]

        for idx, q in enumerate(qs):
            m, r = decode_mcs_index(mcs_index=idx,
                              table_index=2,
                              is_pusch=True,
                              pi2bpsk=pi2bpsk,
                              transform_precoding=True)
            self.assertTrue(m==q)
            self.assertTrue(rs[idx]/1024==r.numpy())

        # test that next index raises error
        with self.assertRaises(tf.errors.InvalidArgumentError):
            decode_mcs_index(mcs_index=idx+1,
                       table_index=2,
                       is_pusch=True,
                       pi2bpsk=pi2bpsk,
                       transform_precoding=True)

    def test_gen_rand_seq(self):
        """Test random sequence generator."""

        # test against invalid inputs
        testcases = [[-1, 10], [10, -1], [100, 2**32], [10.2, 10], [10, 10.2]]
        for tc in testcases:
            with self.assertRaises(AssertionError):
                generate_prng_seq(tc[0], tc[1])

        # test against reference example
        n_rnti = 20001
        n_id = 41
        c_init = n_rnti * 2**15 + n_id # defined in 6.3.1.1 in 38.211
        l = 100
        s_ref = np.array([0., 1., 1., 1., 1., 0., 0., 0., 0., 1., 0., 1., 0.,
                          1., 1., 1., 0., 0., 0., 1., 1., 1., 0., 0., 0., 1.,
                          1., 0., 0., 1., 1., 1., 0., 1., 0., 0., 1., 1., 1.,
                          0., 1., 0., 0., 0., 0., 0., 1., 1., 1., 0., 1., 1.,
                          0., 1., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 1.,
                          0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 1., 0., 0.,
                          1., 1., 0., 1., 1., 1., 0., 0., 0., 0., 0., 1., 0.,
                          1., 1., 1., 1., 1., 1., 1., 0., 0.])
        s = generate_prng_seq(l, c_init)
        self.assertTrue(np.array_equal(s, s_ref))

        # different sequence expected as c_init changes
        s = generate_prng_seq(l, c_init+1)
        self.assertFalse(np.array_equal(s, s_ref))


    def test_tb_size(self):
        """Test TB size calculation"""

        def verify_results(retval, target_rate, n_res):
            """Run consistency tests"""

            tb_size = retval[0].numpy()
            cb_size = retval[1].numpy()
            num_cbs = retval[2].numpy()
            tb_crc_length = retval[3].numpy()
            cb_crc_length = retval[4].numpy()
            cw_length = retval[5].numpy()

            # tb_size must equal number of CB bits (+CRC overhead)
            self.assertTrue(
                tb_size==num_cbs*(cb_size-cb_crc_length)-tb_crc_length)

            # individual cw length for each cb is returned
            self.assertTrue(num_cbs==len(cw_length))

            # single cw TB has no CB CRC
            if num_cbs==1:
                self.assertTrue(cb_crc_length==0)
            else:
                self.assertTrue(cb_crc_length==24)

            # codeword lengths consist of only two different values
            t = np.setdiff1d(np.array(cw_length),
                            np.array([np.min(cw_length), np.max(cw_length)]))
            self.assertTrue(len(t)==0)

            # TB CRC is 16 or 24
            if tb_size>3824:
                self.assertTrue(tb_crc_length==24)
            else:
                self.assertTrue(tb_crc_length==16)

            # ignore scnearios where n_res is clipped (at 156),
            # as the rate can significantly differ for larger values of n_res
            if n_res<=156:
                # effective rate is close to target rate
                eff_rate = tb_size / np.sum(cw_length)
                if tb_size>4000: # for very short blocks the rate does not match
                    # allow 2% difference for long codes
                    self.assertTrue(np.abs(eff_rate-target_rate)<2e-2)
                elif tb_size>200:
                    # allow 10% for medium codes
                    self.assertTrue(np.abs(eff_rate-target_rate)<1e-1)
                else: # ignore ultra short scenarios
                    pass


        for mcs_index in (0, 4, 16, 20, 27):
            for num_layers in (1, 2, 3, 4):
                for num_prbs in (1, 20, 200, 275):
                    for num_ofdm_symbols in (8, 10, 14):
                        for num_dmrs_per_prb in (0, 10, 20):
                            q, r = decode_mcs_index(mcs_index, 2)
                            q, r = q.numpy(), r.numpy()

                            retval = calculate_tb_size(
                                            target_coderate=r,
                                            modulation_order=q,
                                            num_layers=num_layers,
                                            num_prbs=num_prbs,
                                            num_ofdm_symbols=num_ofdm_symbols,
                                            num_dmrs_per_prb=num_dmrs_per_prb,
                                            verbose=True)

                            # number of resource elements per prb
                            n_res = q * num_layers \
                                   *  (12 * num_ofdm_symbols - num_dmrs_per_prb)

                            #### verify results #####
                            verify_results(retval, r, n_res)

    def test_tb_size_vs_numpy(self):
        """
        Validate calculate_tb_size_tf, accepting Tensor inputs,
        against the Numpy version "calculate_tb_size". Test in Eager, Graph and
        XLA modes
        """
        @tf.function
        def calculate_tb_size_graph(modulation_order,
                      target_coderate,
                      target_tb_size=None,
                      num_coded_bits=None,
                      num_prbs=None,
                      num_ofdm_symbols=None,
                      num_dmrs_per_prb=None,
                      num_layers=None,
                      num_ov=None,
                      tb_scaling=None,
                      return_cw_length=True,
                      verbose=False):
            return calculate_tb_size(modulation_order,
                      target_coderate,
                      target_tb_size=target_tb_size,
                      num_coded_bits=num_coded_bits,
                      num_prbs=num_prbs,
                      num_ofdm_symbols=num_ofdm_symbols,
                      num_dmrs_per_prb=num_dmrs_per_prb,
                      num_layers=num_layers,
                      num_ov=num_ov,
                      tb_scaling=tb_scaling,
                      return_cw_length=return_cw_length,
                      verbose=verbose)

        @tf.function(jit_compile=True)
        def calculate_tb_size_xla(modulation_order,
                      target_coderate,
                      target_tb_size=None,
                      num_coded_bits=None,
                      num_prbs=None,
                      num_ofdm_symbols=None,
                      num_dmrs_per_prb=None,
                      num_layers=None,
                      num_ov=None,
                      tb_scaling=None,
                      return_cw_length=True,
                      verbose=False):
            return calculate_tb_size(modulation_order,
                      target_coderate,
                      target_tb_size=target_tb_size,
                      num_coded_bits=num_coded_bits,
                      num_prbs=num_prbs,
                      num_ofdm_symbols=num_ofdm_symbols,
                      num_dmrs_per_prb=num_dmrs_per_prb,
                      num_layers=num_layers,
                      num_ov=num_ov,
                      tb_scaling=tb_scaling,
                      return_cw_length=return_cw_length,
                      verbose=verbose)

        fun_dict = {'eager': calculate_tb_size,
                    'graph': calculate_tb_size_graph,
                    'xla': calculate_tb_size_xla}

        def validate_against_np(tb_size, cb_size, num_cb, cw_length, tb_crc_length, cb_crc_length):
            # Validate against the Numpy version "calculate_tb_size"
            for idx in enumerate_indices(shape).numpy():

                tb_size_orig, cb_size_orig, num_cbs_orig, \
                    tb_crc_length_orig, cb_crc_length_orig, cw_length_orig = \
                    calculate_tb_size_numpy(tf.gather_nd(modulation_order, idx).numpy(),
                                    tf.gather_nd(target_coderate, idx).numpy(),
                                    num_prbs=tf.gather_nd(num_prbs, idx).numpy(),
                                    num_ofdm_symbols=tf.gather_nd(num_ofdm_symbols, idx).numpy(),
                                    num_dmrs_per_prb=tf.gather_nd(num_dmrs_per_prb, idx).numpy(),
                                    num_layers=tf.gather_nd(num_layers, idx).numpy(),
                                    num_ov=tf.gather_nd(num_ov, idx).numpy(),
                                    tb_scaling=tf.gather_nd(tb_scaling, idx).numpy(),
                                    verbose=False)
                self.assertTrue(tb_size_orig==tf.gather_nd(tb_size, idx).numpy())
                self.assertTrue(cb_size_orig==tf.gather_nd(cb_size, idx).numpy())
                self.assertTrue(num_cbs_orig==tf.gather_nd(num_cb, idx).numpy())
                if not np.all(cw_length_orig==tf.gather_nd(cw_length, idx).numpy()[:len(cw_length_orig)]):
                    print(f'{cw_length_orig=}')
                    print(f'TF version = {tf.gather_nd(cw_length, idx).numpy()}')
                self.assertTrue(np.all(cw_length_orig==tf.gather_nd(cw_length, idx).numpy()[:len(cw_length_orig)]))
                self.assertTrue(tb_crc_length_orig==tf.gather_nd(tb_crc_length, idx).numpy())
                self.assertTrue(cb_crc_length_orig==tf.gather_nd(cb_crc_length, idx).numpy())
                self.assertTrue(num_cbs_orig==tf.gather_nd(num_cb, idx).numpy())
            return None


        shape = [10, 12, 15]
        int_dtype = tf.int32
        float_dtype = tf.float32

        modulation_order = random_tensor_from_values(
            values=[2, 4, 6], shape=shape, dtype=int_dtype)
        target_coderate = random_tensor_from_values(
            values=np.linspace(.5, .95, 30), shape=shape, dtype=float_dtype)
        num_ofdm_symbols = random_tensor_from_values(
            values=[8, 9, 10, 11, 12], shape=shape, dtype=int_dtype)
        num_dmrs_per_prb = random_tensor_from_values(values=[12, 24, 36],
                                                     shape=shape, dtype=int_dtype)
        num_prbs = random_tensor_from_values(values=list(
            range(1, 100)), shape=shape, dtype=int_dtype)
        num_layers = random_tensor_from_values(values=[1, 2, 3],
                                               shape=shape, dtype=int_dtype)
        num_ov = random_tensor_from_values(values=[0, 1, 2, 3],
                                           shape=shape, dtype=int_dtype)
        tb_scaling = random_tensor_from_values(
            values=[1.0], shape=shape, dtype=float_dtype)

        for mode, fun in fun_dict.items():
            # mode a): compute target_tb_size and num_coded bits
            tb_size, cb_size, num_cb, tb_crc_length, cb_crc_length, cw_length = \
                fun(modulation_order,
                    target_coderate,
                    num_prbs=num_prbs,
                    num_ofdm_symbols=num_ofdm_symbols,
                    num_dmrs_per_prb=num_dmrs_per_prb,
                    num_layers=num_layers,
                    num_ov=num_ov,
                    tb_scaling=tb_scaling)

            # validate against the Numpy version
            validate_against_np(tb_size, cb_size, num_cb, cw_length,
                                tb_crc_length, cb_crc_length)

            # mode b): provide already target_tb_size, num_coded_bits
            # Use the actual TB size computed in mode a) to ensure consistency
            # This avoids the quantization issue where providing an unquantized
            # target_tb_size bypasses the 3GPP quantization steps
            target_tb_size = tf.cast(tb_size, float_dtype)

            # Compute data symbols per PRB for num_coded_bits calculation
            n_re_per_prb = 12 * num_ofdm_symbols - num_dmrs_per_prb - num_ov
            # The max. number of REs per PRB is limited to 156 in 38.214
            n_re_per_prb = tf.minimum(156, n_re_per_prb)

            # number of coded bits that fit into the given slot configuration
            num_coded_bits = tb_scaling * \
                tf.cast(n_re_per_prb * modulation_order *
                        num_layers * num_prbs, float_dtype)
            num_coded_bits = tf.cast(num_coded_bits, int_dtype)

            tb_size1, cb_size1, num_cb1, tb_crc_length1, cb_crc_length1, cw_length1 = \
                fun(modulation_order,
                    target_coderate,
                    target_tb_size=target_tb_size,
                    num_coded_bits=num_coded_bits,
                    num_layers=num_layers,
                    num_ov=num_ov,
                    tb_scaling=tb_scaling)

            # validate against the mode a) version
            self.assertTrue(tf.reduce_all(tb_size==tb_size1))
            self.assertTrue(tf.reduce_all(cb_size==cb_size1))
            self.assertTrue(tf.reduce_all(num_cb==num_cb1))
            self.assertTrue(tf.reduce_all(cw_length==cw_length1))
            self.assertTrue(tf.reduce_all(tb_crc_length==tb_crc_length1))
            self.assertTrue(tf.reduce_all(cb_crc_length==cb_crc_length1))

            # The sum of codeword lengths must equal the n. coded bits
            self.assertTrue(
                tf.reduce_all(tf.reduce_sum(cw_length1, axis=-1) == num_coded_bits))

    def test_decode_mcs_index_tf(self):
        """Unittest for decode_mcs_index_tf. Compares against its Numpy version.
        Test in Eager, Graph and XLA modes"""

        @tf.function
        def decode_mcs_index_graph(mcs_index,
                                   table_index=None,
                                   is_pusch=None,
                                   transform_precoding=None,
                                   pi2bpsk=None):
            return decode_mcs_index(mcs_index,
                                    table_index=table_index,
                                    is_pusch=is_pusch,
                                    transform_precoding=transform_precoding,
                                    pi2bpsk=pi2bpsk)

        @tf.function(jit_compile=True)
        def decode_mcs_index_xla(mcs_index,
                                 table_index=None,
                                 is_pusch=None,
                                 transform_precoding=None,
                                 pi2bpsk=None):
            return decode_mcs_index(mcs_index,
                                    table_index=table_index,
                                    is_pusch=is_pusch,
                                    transform_precoding=transform_precoding,
                                    pi2bpsk=pi2bpsk)
        fun_dict = {'eager': decode_mcs_index,
                    'graph': decode_mcs_index_graph,
                    'xla': decode_mcs_index_xla}

        shape = [10, 10, 10]
        mcs_index = random_tensor_from_values(values=list(range(27)), shape=shape)
        table_index = random_tensor_from_values(values=[1, 2], shape=shape)
        is_pusch = random_tensor_from_values(
            values=[True, False], shape=shape)
        transform_precoding = random_tensor_from_values(
            values=[True, False], shape=shape)
        pi2bpsk = random_tensor_from_values(values=[True, False], shape=shape)

        for mod, fun in fun_dict.items():
            mod_orders_tf, target_rates_tf = fun(
                mcs_index,
                table_index=table_index,
                is_pusch=is_pusch,
                transform_precoding=transform_precoding,
                pi2bpsk=pi2bpsk)

            # compare against Numpy
            target_rates_np = np.zeros(shape)
            mod_orders_np = np.zeros(shape)

            for s in enumerate_indices(shape):
                channel_type = 'PUSCH' if tf.gather_nd(
                    is_pusch, s).numpy() else 'PDSCH'
                mod_orders_np[tuple(s)], target_rates_np[tuple(s)] = \
                    decode_mcs_index_numpy(
                        int(tf.gather_nd(mcs_index, s).numpy()),
                    table_index=int(tf.gather_nd(table_index, s).numpy()),
                    channel_type=channel_type,
                    transform_precoding=bool(tf.gather_nd(
                        transform_precoding, s).numpy()),
                    pi2bpsk=bool(tf.gather_nd(pi2bpsk, s).numpy()),
                    verbose=False)
            self.assertTrue(tf.reduce_sum(tf.abs(mod_orders_np - mod_orders_tf.numpy()))==0)
            self.assertTrue(tf.reduce_sum(tf.abs(target_rates_np - target_rates_tf.numpy()))==0)
