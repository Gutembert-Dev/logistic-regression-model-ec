"""
Tests for the preprocessing functions.
"""
# pylint: disable=E0401,C0103

import pandas as pd
from pandas.testing import assert_frame_equal

# System imports
from unittest import TestCase

from explore_ai_demo.preprocess import *


class TestApp(TestCase):
    """
    Tests for the preprocessing functions.
    """

    def setUp(self):
        """
        Setup.
        """

    def test_replace_missing(self):
        """
        Test replace missing values function.
        """

        test_df = pd.DataFrame({"CON001OTH": [None, None, "AAA", "BBB", "CCC"]})
        expected_df = pd.DataFrame({"CON001OTH": [-999, -999, "AAA", "BBB", "CCC"]})
        actual_df = replace_missing(test_df)
        assert_frame_equal(actual_df, expected_df)

    def test_exclusion(self):
        """
        Test exclusion function
        :return:
        """
        test_df = pd.DataFrame(
            {
                "CON001OTH": [0, 0, 1, 1, 1],
                "CON002OTH": [0, 1, 1, 1, 1],
                "ACC100CRT": [-1000, 1, 100, 4, 6],
            }
        )
        expected_df = pd.DataFrame(
            {"CON001OTH": [0], "CON002OTH": [0], "ACC100CRT": [-1000], "Exclusion": [0]}
        )
        actual_df = exclusion(test_df)
        assert_frame_equal(actual_df, expected_df)

    def test_mob(self):
        """
        Test mob function.
        """

        actual_df = pd.DataFrame({"months_on_book": [-1, 3, 40, 1000, 99]})
        expected_df = pd.DataFrame(
            {"months_on_book": [None, 0.31, -0.01, -0.12, -0.01]}
        )
        actual_df["months_on_book"] = actual_df.apply(mob, axis=1)
        assert_frame_equal(actual_df, expected_df)

    def test_ptp24(self):
        """
        Test ptp24 function.
        """

        actual_df = pd.DataFrame({"NumPTPsL24M": [0, 1, 3, 4, 2, -1]})
        expected_df = pd.DataFrame(
            {"NumPTPsL24M": [-0.27, 0.70, 1.35, 1.35, 0.93, None]}
        )
        actual_df["NumPTPsL24M"] = actual_df.apply(ptp24, axis=1)
        assert_frame_equal(actual_df, expected_df)

    def test_rpc9(self):
        """
        Test rpc9 function.
        """

        actual_df = pd.DataFrame({"NumRPCsL9M": [0, 1, 3, 4, 2, -1]})
        expected_df = pd.DataFrame(
            {"NumRPCsL9M": [-0.30, 0.43, 1.08, 1.08, 0.74, None]}
        )
        actual_df["NumRPCsL9M"] = actual_df.apply(rpc9, axis=1)
        assert_frame_equal(actual_df, expected_df)

    def test_rec9(self):
        """
        Test rec9 function.
        """

        actual_df = pd.DataFrame({"NumRecsL9M": [0, 1, 3, 4, 2, -1]})
        expected_df = pd.DataFrame(
            {"NumRecsL9M": [-0.07, 1.09, 1.09, 1.09, 1.09, 1.39]}
        )
        actual_df["NumRecsL9M"] = actual_df.apply(rec9, axis=1)
        assert_frame_equal(actual_df, expected_df)

    def test_rec9(self):
        """
        Test rec9 function.
        """

        actual_df = pd.DataFrame({"NumRecsL9M": [0, 1, 3, 4, 2, -1]})
        expected_df = pd.DataFrame(
            {"NumRecsL9M": [-0.07, 1.09, 1.09, 1.09, 1.09, 1.39]}
        )
        actual_df["NumRecsL9M"] = actual_df.apply(rec9, axis=1)
        assert_frame_equal(actual_df, expected_df)

    def test_age(self):
        """
        Test age function.
        """

        actual_df = pd.DataFrame({"Age": [-999, 23, 30, 35, 40, 60]})
        expected_df = pd.DataFrame({"Age": [0.04, 0.33, 0.1, 0.03, -0.03, -0.19]})
        actual_df["Age"] = actual_df.apply(age, axis=1)
        assert_frame_equal(actual_df, expected_df)

    def test_acc001ucr(self):
        """
        Test acc001ucr function.
        """

        actual_df = pd.DataFrame({"ACC001UCR": [-999, 0.5, 2, 3.5, 6, 9]})
        expected_df = pd.DataFrame(
            {"ACC001UCR": [-0.11, 0.12, -0.01, -0.09, -0.15, 0.02]}
        )
        actual_df["ACC001UCR"] = actual_df.apply(acc001ucr, axis=1)
        print(actual_df)
        assert_frame_equal(actual_df, expected_df)

    def test_acc011crt(self):
        """
        Test acc011crt function.
        """

        actual_df = pd.DataFrame({"ACC011CRT": [-999, 0, 1, 3, 4, 2, 5, 6]})
        expected_df = pd.DataFrame(
            {"ACC011CRT": [-0.11, 0.18, 0.1, -0.11, -0.22, 0.0, -0.31, -0.31]}
        )
        actual_df["ACC011CRT"] = actual_df.apply(acc011crt, axis=1)
        print(actual_df)
        assert_frame_equal(actual_df, expected_df)

    def test_acc100crt(self):
        """
        Test acc011crt function.
        """

        actual_df = pd.DataFrame({"ACC100CRT": [-999, 30, 40, 70, 100, 190, 1000, 6]})
        expected_df = pd.DataFrame(
            {"ACC100CRT": [-0.11, 0.01, -0.13, 0.02, 0.1, 0.0, 0.0, 0.01]}
        )
        actual_df["ACC100CRT"] = actual_df.apply(acc100crt, axis=1)
        print(actual_df)
        assert_frame_equal(actual_df, expected_df)

    def test_acc101rev(self):
        """
        Test acc101rev function.
        """

        actual_df = pd.DataFrame({"ACC101REV": [-999, -3, -2, 10, 20, 70, 50, 700]})
        expected_df = pd.DataFrame(
            {"ACC101REV": [-0.11, -0.08, -0.05, 0.15, -0.04, 0.09, -0.04, 0.09]}
        )
        actual_df["ACC101REV"] = actual_df.apply(acc101rev, axis=1)
        print(actual_df)
        assert_frame_equal(actual_df, expected_df)

    def test_acc104crt(self):
        """
        Test acc104crt function.
        """

        actual_df = pd.DataFrame({"ACC104CRT": [-999, 30, 40, 10, 20, 70, 50, 700]})
        expected_df = pd.DataFrame(
            {"ACC104CRT": [-0.11, 0.05, -0.04, 0.05, 0.05, -0.04, -0.04, 0.05]}
        )
        actual_df["ACC104CRT"] = actual_df.apply(acc104crt, axis=1)
        print(actual_df)
        assert_frame_equal(actual_df, expected_df)

    def test_acc209crt(self):
        """
        Test acc209crt function.
        """

        actual_df = pd.DataFrame({"ACC209CRT": [-999, 3, 1, 2, 4, 5, 6, 7]})
        expected_df = pd.DataFrame(
            {"ACC209CRT": [-0.11, -0.04, 0.16, 0.1, -0.13, -0.25, -0.25, -0.25]}
        )
        actual_df["ACC209CRT"] = actual_df.apply(acc209crt, axis=1)
        print(actual_df)
        assert_frame_equal(actual_df, expected_df)

    def test_acc230crt(self):
        """
        Test acc209crt function.
        """

        actual_df = pd.DataFrame({"ACC230CRT": [-999, 3, 1, -2, 4, -5, 6, 0, 7, 10]})
        expected_df = pd.DataFrame(
            {
                "ACC230CRT": [
                    -0.11,
                    0.33,
                    0.33,
                    -0.76,
                    0.26,
                    -0.4,
                    0.1,
                    -0.11,
                    -0.1,
                    -0.23,
                ]
            }
        )
        actual_df["ACC230CRT"] = actual_df.apply(acc230crt, axis=1)
        print(actual_df)
        assert_frame_equal(actual_df, expected_df)

    def test_acc233crt(self):
        """
        Test acc209crt function.
        """

        actual_df = pd.DataFrame({"ACC233CRT": [-999, 3, 1, -2, 4, -5, 6, 0, 7, 10]})
        expected_df = pd.DataFrame(
            {
                "ACC233CRT": [
                    -0.11,
                    -0.12,
                    -0.12,
                    None,
                    0.17,
                    None,
                    0.17,
                    -0.12,
                    0.17,
                    0.0,
                ]
            }
        )
        actual_df["ACC233CRT"] = actual_df.apply(acc233crt, axis=1)
        print(actual_df)
        assert_frame_equal(actual_df, expected_df)

    def test_acc234crt(self):
        """
        Test acc234crt function.
        """

        actual_df = pd.DataFrame(
            {"ACC234CRT": [-999, 3, 1, -2, 4, 50, 60, 0, 70, 1000]}
        )
        expected_df = pd.DataFrame(
            {
                "ACC234CRT": [
                    -0.11,
                    0.13,
                    0.13,
                    None,
                    0.13,
                    0.04,
                    -0.05,
                    -0.31,
                    -0.05,
                    0.17,
                ]
            }
        )
        actual_df["ACC234CRT"] = actual_df.apply(acc234crt, axis=1)
        print(actual_df)
        assert_frame_equal(actual_df, expected_df)

    def test_acc309nct(self):
        """
        Test acc309nct function.
        """

        actual_df = pd.DataFrame(
            {"ACC309NCT": [-999, 3, 1, -2, 4, 50, 600, 0, 70, 1000, 3000, -1]}
        )
        expected_df = pd.DataFrame(
            {
                "ACC309NCT": [
                    -0.11,
                    0.13,
                    0.13,
                    0.07,
                    0.13,
                    0.13,
                    0.13,
                    0.13,
                    0.13,
                    -0.08,
                    -0.33,
                    -0.01,
                ]
            }
        )
        actual_df["ACC309NCT"] = actual_df.apply(acc309nct, axis=1)
        assert_frame_equal(actual_df, expected_df)

    def test_acc314crt(self):
        """
        Test acc309nct function.
        """

        actual_df = pd.DataFrame(
            {
                "ACC314CRT": [
                    -999,
                    3,
                    1,
                    -2,
                    4,
                    500,
                    60,
                    0,
                    70,
                    900,
                    1000,
                    1800,
                    2500,
                    3600,
                    4600,
                    5600,
                    7000,
                    10000,
                    45000,
                    200000,
                    20000,
                ]
            }
        )
        expected_df = pd.DataFrame(
            {
                "ACC314CRT": [
                    -0.11,
                    0.21,
                    0.21,
                    -0.76,
                    0.21,
                    0.21,
                    0.21,
                    -0.14,
                    0.21,
                    0.47,
                    0.47,
                    0.33,
                    0.26,
                    0.21,
                    0.16,
                    0.1,
                    0.04,
                    -0.04,
                    -0.24,
                    -0.42,
                    -0.16,
                ]
            }
        )
        actual_df["ACC314CRT"] = actual_df.apply(acc314crt, axis=1)
        assert_frame_equal(actual_df, expected_df)

    def test_enq004oth(self):
        """
        Test enq004oth function.
        """

        actual_df = pd.DataFrame({"ENQ004OTH": [-999, 3, 1, -2, -4, 5, 6, 0, 7, 2]})
        expected_df = pd.DataFrame(
            {
                "ENQ004OTH": [
                    -0.11,
                    0.15,
                    0.14,
                    None,
                    -0.17,
                    0.15,
                    0.15,
                    0.08,
                    0.15,
                    0.26,
                ]
            }
        )
        actual_df["ENQ004OTH"] = actual_df.apply(enq004oth, axis=1)
        assert_frame_equal(actual_df, expected_df)

    def test_leg200oth(self):
        """
        Test leg200oth function.
        """

        actual_df = pd.DataFrame({"LEG200OTH": [-999, 3, 1, -2, -4, 5, 6, 0, 7]})
        expected_df = pd.DataFrame(
            {"LEG200OTH": [-0.11, 0.18, 0.18, None, None, 0.18, 0.18, -0.01, 0.18]}
        )
        actual_df["LEG200OTH"] = actual_df.apply(leg200oth, axis=1)
        assert_frame_equal(actual_df, expected_df)

    def test_fill_in_missing_values(self):
        """
        Test fill_in_missing_values function.
        """

        test_df = pd.DataFrame(
            {"Gender": [None, "Male", "Female", "Female"], "Age": [None, None, 30, 20]}
        )
        expected_df = pd.DataFrame(
            {
                "Gender": [-999, "Male", "Female", "Female"],
                "Age": [25.0, 25.0, 30.0, 20.0],
            }
        )
        actual_df = fill_in_missing_values(test_df)
        assert_frame_equal(actual_df, expected_df)

    def test_preprocess_categorical_columns(self):
        """
        Test preprocess_categorical_columns function.
        """

        test_df = pd.DataFrame({"Gender": [None, "M", "F", "F"]})
        expected_df = pd.DataFrame({"Gender": [0, 0, 1, 1]})
        actual_df = preprocess_categorical_columns(test_df)
        assert_frame_equal(actual_df, expected_df)
