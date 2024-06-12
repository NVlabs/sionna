#
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
try:
    import sionna
except ImportError as e:
    import sys
    sys.path.append("../")

import unittest
import numpy as np
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print('Number of GPUs available :', len(gpus))
if gpus:
    gpu_num = 0 # Number of the GPU to be used
    try:
        tf.config.set_visible_devices(gpus[gpu_num], 'GPU')
        print('Only GPU number', gpu_num, 'used.')
        tf.config.experimental.set_memory_growth(gpus[gpu_num], True)
    except RuntimeError as e:
        print(e)

from sionna.nr.utils import select_mcs, generate_prng_seq, generate_low_papr_seq_type_1, calculate_tb_size


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

    def test_generate_low_papr_seq_type_1(self):
        # test against invalid inputs
        testcases = [[36, 30, 0, 0], [5, 0, 0, 0], [36, 0, 0, -1], [36, 0, 1, 0], [72, 0, 2, 0]]
        for inputs in testcases:
            with self.assertRaises(ValueError):
                generate_low_papr_seq_type_1(*inputs)

        testcases = [
            [(6, 1, 0, 2),
             [-0.70711 - 0.70711j, -0.34871 - 0.93723j, -0.99734 - 0.072944j, 0.48137 - 0.87652j, -0.5967 - 0.80247j,
              0.20863 + 0.97799j]],
            [(12, 28, 0, 3),
             [0.70711 + 0.70711j, -0.60024 + 0.79982j, -0.48137 + 0.87652j, -0.93568 - 0.35285j, 0.97611 + 0.21728j,
              -0.077358 + 0.997j, -0.064114 - 0.99794j, 0.2043 + 0.97891j, 0.94028 - 0.3404j, -0.46969 - 0.88283j,
              -0.80772 + 0.58957j, -0.71643 + 0.69766j]],
            [(30, 23, 0, 4),
             [0.15143 + 0.98847j, -0.3916 + 0.92014j, -0.6933 - 0.72065j, 0.49314 + 0.86995j, 0.91416 - 0.40534j,
              0.8911 - 0.4538j, 0.62623 + 0.77964j, -0.85955 - 0.51104j, -0.026691 + 0.99964j, 0.5932 + 0.80506j,
              -0.12174 + 0.99256j, -0.74711 - 0.6647j, 0.3808 + 0.92466j, 0.99599 - 0.089513j, 0.99824 + 0.059348j,
              -0.056355 + 0.99841j, -0.098464 - 0.99514j, -0.91885 + 0.39461j, -0.64888 + 0.76089j,
              -0.99548 - 0.094925j, 0.78507 - 0.61941j, -0.99992 + 0.012268j, -0.4719 + 0.88165j, -0.74674 + 0.66512j,
              -0.50378 - 0.86383j, 0.46204 + 0.88686j, 0.83391 - 0.55189j, 0.66673 - 0.7453j, 0.94878 + 0.31594j,
              -0.97159 + 0.23666j]],
            [(36, 20, 0, 1),
             [1 + 0j, -0.99342 + 0.11451j, -0.22459 + 0.97445j, -0.85411 + 0.52009j, 0.6491 - 0.76071j,
              -0.66374 - 0.74796j, -0.1308 - 0.99141j, 0.60622 + 0.7953j, 0.75484 - 0.65591j, 0.94815 - 0.31783j,
              -0.50082 + 0.86555j, 0.96696 + 0.25493j, 0.90173 + 0.4323j, -0.88771 + 0.4604j, 0.81219 + 0.58339j,
              0.81995 + 0.57244j, -0.86847 + 0.49575j, 0.92868 + 0.37088j, 0.9866 + 0.16313j, -0.39291 + 0.91958j,
              0.8911 - 0.4538j, 0.62956 - 0.77695j, 0.75296 + 0.65806j, -0.35158 - 0.93616j, -0.8309 - 0.55642j,
              0.41199 - 0.91119j, -0.6558 + 0.75493j, 0.10869 + 0.99408j, -0.88837 + 0.45913j, 0.92525 - 0.37935j,
              0.15425 - 0.98803j, 0.91474 - 0.40404j, -0.86246 + 0.50612j, 0.18828 + 0.98212j, -0.57115 + 0.82084j,
              0.2864 - 0.95811j]],
            [(100, 15, 1, 1),
             [1 + 0j, -0.6689 - 0.74335j, -0.056579 - 0.9984j, -0.44178 + 0.89713j, -0.72417 + 0.68962j,
              0.84155 - 0.54019j, 0.85653 - 0.5161j, -0.78018 + 0.62556j, -0.56418 + 0.82565j, 0.14153 - 0.98993j,
              -0.45945 - 0.8882j, 0.95169 + 0.30706j, 0.80735 - 0.59007j, 0.16479 + 0.98633j, 0.99048 + 0.13769j,
              -0.27622 + 0.96109j, 0.96648 + 0.25674j, -0.077426 + 0.997j, 0.96468 - 0.26343j, 0.698 + 0.7161j,
              0.12987 - 0.99153j, 0.76506 - 0.64396j, -0.99272 + 0.12041j, -0.95642 - 0.29201j, 0.85179 + 0.52389j,
              0.79936 + 0.60085j, -0.83877 - 0.54448j, -0.94106 - 0.33824j, 0.99887 - 0.047603j, 0.82408 - 0.56648j,
              -0.24937 + 0.96841j, 0.58627 + 0.81012j, -0.99539 + 0.095917j, -0.26902 + 0.96313j,
              -0.88751 - 0.46079j, -0.49987 + 0.8661j, -0.91868 - 0.39501j, -0.12643 + 0.99198j,
              -0.95096 + 0.30932j, 0.79418 + 0.60768j, 0.11175 + 0.99374j, 0.50704 - 0.86192j, 0.84872 - 0.52885j,
              -0.97196 + 0.23516j, -0.99684 + 0.079376j, 0.99652 - 0.083377j, 0.96905 - 0.24685j,
              -0.83793 + 0.54578j, -0.48263 + 0.87583j, -0.14757 - 0.98905j, -0.82023 - 0.57203j, 0.93353 - 0.3585j,
              0.066509 - 0.99779j, 0.94347 + 0.33145j, 0.43243 - 0.90167j, 0.92316 + 0.38443j, 0.17908 - 0.98383j,
              0.98077 - 0.19516j, -0.67046 - 0.74195j, 0.13521 - 0.99082j, -0.7474 + 0.66438j, -0.98383 + 0.17912j,
              0.97915 + 0.20313j, 0.91011 + 0.41437j, -0.88327 - 0.46887j, -0.92611 - 0.37726j, 0.99236 + 0.1234j,
              0.95494 - 0.2968j, -0.63071 + 0.77602j, 0.066408 + 0.99779j, -0.829 - 0.55924j, -0.8873 + 0.46118j,
              -0.14278 - 0.98975j, -0.99952 - 0.030838j, 0.043009 - 0.99907j, -0.99431 + 0.10653j,
              -0.40633 - 0.91373j, -0.62781 + 0.77837j, -0.99907 - 0.043146j, 0.68481 + 0.72872j,
              0.14251 + 0.98979j, 0.30164 - 0.95342j, 0.55959 - 0.82877j, -0.65895 + 0.75219j, -0.63114 + 0.77567j,
              0.46593 - 0.88482j, 0.12408 - 0.99227j, 0.3874 + 0.92191j, 0.88288 + 0.46959j, -0.93686 + 0.34971j,
              -0.20407 + 0.97896j, -0.82174 - 0.56986j, -0.74812 + 0.66356j, -0.60408 - 0.79692j,
              -0.74277 + 0.66954j, -0.83078 - 0.5566j, -0.18043 + 0.98359j, -0.92515 + 0.37961j, 0.90102 + 0.43379j,
              0.43134 + 0.90219j]]
        ]
        for inputs, outputs in testcases:
            np.testing.assert_array_almost_equal(generate_low_papr_seq_type_1(*inputs), outputs, decimal=5)

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
