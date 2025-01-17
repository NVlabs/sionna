#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import unittest
import numpy as np
import tensorflow as tf
from sionna.nr.utils import select_mcs, generate_prng_seq, calculate_tb_size


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
                m, r = select_mcs(mcs_index=idx,
                                  table_index=1,
                                  channel_type="PDSCH",
                                  pi2bpsk=bpsk)
                self.assertTrue(m==q)
                self.assertTrue(rs[idx]/1024==r)

        # test that next index raises error
        with self.assertRaises(AssertionError):
            select_mcs(mcs_index=idx+1,
                       table_index=1,
                       channel_type="PDSCH",
                       pi2bpsk=False)

        # Tab. 5.1.3.1-2
        qs = [2,2,2,2,2,4,4,4,4,4,4,6,6,6,6,6,6,6,6,6,8,8,8,8,8,8,8,8]
        rs = [120,193,308,449,602,378,434,490,553,616,658,466,517,567,616,666,
              719,772,822,873,682.5,711,754,797,841,885,916.5,948]

        for bpsk in (True, False): # no impact
            for idx, q in enumerate(qs):
                m, r = select_mcs(mcs_index=idx,
                                  table_index=2,
                                  channel_type="PDSCH",
                                  pi2bpsk=bpsk)
                self.assertTrue(m==q)
                self.assertTrue(rs[idx]/1024==r)

        # test that next index raises error
        with self.assertRaises(AssertionError):
            select_mcs(mcs_index=idx+1,
                       table_index=2,
                       channel_type="PDSCH",
                       pi2bpsk=False)

        # Tab. 5.1.3.1-3
        qs = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,4,4,4,4,4,4,6,6,6,6,6,6,6,6]
        rs = [30,40,50,64,78,99,120,157,193,251,308,379,449,526,602,340,378,434,
              490,553,616,438,466,517,567,616,666,719,772]

        for bpsk in (True, False): # no impact
            for idx, q in enumerate(qs):
                m, r = select_mcs(mcs_index=idx,
                                  table_index=3,
                                  channel_type="PDSCH",
                                  pi2bpsk=bpsk)
                self.assertTrue(m==q)
                self.assertTrue(rs[idx]/1024==r)

        # test that next index raises error
        with self.assertRaises(AssertionError):
            select_mcs(mcs_index=idx+1,
                       table_index=3,
                       channel_type="PDSCH",
                       pi2bpsk=False)
                       
        # Tab. 5.1.3.1-4
        qs = [2,2,2,4,4,4,6,6,6,6,6,6,6,6,6,8,8,8,8,8,8,8,8,10,10,10,10]
        rs = [120,193,449,378,490,616,466,517,567,616,666,719,772,822,873,682.5,711,754,797,841,885,916.5,948,805.5,853,900.5,948]

        for bpsk in (True, False): # no impact
            for idx, q in enumerate(qs):
                m, r = select_mcs(mcs_index=idx,
                                  table_index=4,
                                  channel_type="PDSCH",
                                  pi2bpsk=bpsk)
                self.assertTrue(m==q)
                self.assertTrue(rs[idx]/1024==r)

        # test that next index raises error
        with self.assertRaises(AssertionError):
            select_mcs(mcs_index=idx+1,
                       table_index=4,
                       channel_type="PDSCH",
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
                m, r = select_mcs(mcs_index=idx,
                                  table_index=1,
                                  channel_type="PUSCH",
                                  pi2bpsk=bpsk,
                                  transform_precoding=False)
                self.assertTrue(m==q)
                self.assertTrue(rs[idx]/1024==r)

        # test that next index raises error
        with self.assertRaises(AssertionError):
            select_mcs(mcs_index=idx+1,
                       table_index=1,
                       channel_type="PUSCH",
                       pi2bpsk=False,
                       transform_precoding=False)

        # Tab. 5.1.3.1-2
        qs = [2,2,2,2,2,4,4,4,4,4,4,6,6,6,6,6,6,6,6,6,8,8,8,8,8,8,8,8]
        rs = [120,193,308,449,602,378,434,490,553,616,658,466,517,567,616,666,
              719,772,822,873,682.5,711,754,797,841,885,916.5,948]

        for bpsk in (True, False): # no impact
            for idx, q in enumerate(qs):
                m, r = select_mcs(mcs_index=idx,
                                  table_index=2,
                                  channel_type="PUSCH",
                                  pi2bpsk=bpsk,
                                  transform_precoding=False)
                self.assertTrue(m==q)
                self.assertTrue(rs[idx]/1024==r)

        # test that next index raises error
        with self.assertRaises(AssertionError):
            select_mcs(mcs_index=idx+1,
                       table_index=2,
                       channel_type="PUSCH",
                       pi2bpsk=False,
                       transform_precoding=False)

        # Tab. 5.1.3.1-3
        qs = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,4,4,4,4,4,4,6,6,6,6,6,6,6,6]
        rs = [30,40,50,64,78,99,120,157,193,251,308,379,449,526,602,340,378,434,
              490,553,616,438,466,517,567,616,666,719,772]

        for bpsk in (True, False): # no impact
            for idx, q in enumerate(qs):
                m, r = select_mcs(mcs_index=idx,
                                  table_index=3,
                                  channel_type="PUSCH",
                                  pi2bpsk=bpsk,
                                  transform_precoding=False)
                self.assertTrue(m==q)
                self.assertTrue(rs[idx]/1024==r)

        # test that next index raises error
        with self.assertRaises(AssertionError):
            select_mcs(mcs_index=idx+1,
                       table_index=3,
                       channel_type="PUSCH",
                       pi2bpsk=False,
                       transform_precoding=False)

        #### with precoding
        # Tab. 6.1.4.1-1
        pi2bpsk = False #(q=2)
        qs = [2,2,2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,6,6,6,6,6,6,6,6,6,6,6]
        rs = [120,157,193,251,308,379,449,526,602,679,340,378,434,490,553,616,
              658,466,517,567,616,666,719,772,822,873,910,948]

        for idx, q in enumerate(qs):
            m, r = select_mcs(mcs_index=idx,
                              table_index=1,
                              transform_precoding=True,
                              channel_type="PUSCH",
                              pi2bpsk=pi2bpsk)
            self.assertTrue(m==q)
            self.assertTrue(rs[idx]/1024==r)

        # test that next index raises error
        with self.assertRaises(AssertionError):
            select_mcs(mcs_index=idx+1,
                       table_index=1,
                       channel_type="PUSCH",
                       pi2bpsk=pi2bpsk,
                       transform_precoding=True)

        # Tab. 6.1.4.1-1
        pi2bpsk = True #(q=1)
        qs = [1,1,2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,6,6,6,6,6,6,6,6,6,6,6]
        rs = [240,314,193,251,308,379,449,526,602,679,340,378,434,490,553,616,
              658,466,517,567,616,666,719,772,822,873,910,948]

        for idx, q in enumerate(qs):
            m, r = select_mcs(mcs_index=idx,
                              table_index=1,
                              channel_type="PUSCH",
                              transform_precoding=True,
                              pi2bpsk=pi2bpsk)
            self.assertTrue(m==q)
            self.assertTrue(rs[idx]/1024==r)

        # test that next index raises error
        with self.assertRaises(AssertionError):
            select_mcs(mcs_index=idx+1,
                       table_index=1,
                       channel_type="PUSCH",
                       pi2bpsk=pi2bpsk,
                       transform_precoding=True)

        # Tab. 6.1.4.1-2
        pi2bpsk = False #(q=2)
        qs = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,4,6,6,6,6]
        rs = [30,40,50,64,78,99,120,157,193,251,308,379,449,526,602,679,378,434,
              490,553,616,658,699,772,567,616,666,772]

        for idx, q in enumerate(qs):
            m, r = select_mcs(mcs_index=idx,
                              table_index=2,
                              channel_type="PUSCH",
                              pi2bpsk=pi2bpsk,
                              transform_precoding=True)
            self.assertTrue(m==q)
            self.assertTrue(rs[idx]/1024==r)

        # test that next index raises error
        with self.assertRaises(AssertionError):
            select_mcs(mcs_index=idx+1,
                       table_index=2,
                       channel_type="PUSCH",
                       pi2bpsk=pi2bpsk,
                       transform_precoding=True)

        # Tab. 6.1.4.1-2
        pi2bpsk = True #(q=1)
        qs = [1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,4,4,4,4,4,4,4,4,6,6,6,6]
        rs = [60,80,100,128,156,198,120,157,193,251,308,379,449,526,602,679,378,
              434,490,553,616,658,699,772,567,616,666,772]

        for idx, q in enumerate(qs):
            m, r = select_mcs(mcs_index=idx,
                              table_index=2,
                              channel_type="PUSCH",
                              pi2bpsk=pi2bpsk,
                              transform_precoding=True)
            self.assertTrue(m==q)
            self.assertTrue(rs[idx]/1024==r)

        # test that next index raises error
        with self.assertRaises(AssertionError):
            select_mcs(mcs_index=idx+1,
                       table_index=2,
                       channel_type="PUSCH",
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

            tb_size = retval[0]
            cb_size = retval[1]
            num_cbs = retval[2]
            cw_length = retval[3]
            tb_crc_length = retval[4]
            cb_crc_length = retval[5]

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
                            q, r = select_mcs(mcs_index, 2)

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
