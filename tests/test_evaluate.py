"""Unit tests for src.evaluate pure helpers (synthetic data only)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binomtest, chi2

from src.evaluate import (
    align_predictions,
    compute_macro_f1,
    compute_topk_accuracy,
    mcnemar_test,
)


def _paired_correct(b: int, c: int, both_right: int = 5, both_wrong: int = 4):
    """Build paired correctness arrays with exactly b/c discordant pairs."""
    a = np.array([1] * both_right + [0] * both_wrong + [1] * b + [0] * c, dtype=bool)
    b_arr = np.array([1] * both_right + [0] * both_wrong + [0] * b + [1] * c,
                     dtype=bool)
    return a, b_arr


class TestMcNemar:
    def test_b10_c2_matches_hand_computed_binomial(self):
        a, b_arr = _paired_correct(10, 2)
        res = mcnemar_test(a, b_arr)
        assert res["b"] == 10
        assert res["c"] == 2
        assert res["n_discordant"] == 12
        # Exact two-sided p for Binom(12, 0.5): 2 * P(X <= 2) = 158/4096.
        assert res["exact_p"] == pytest.approx(158 / 4096)
        assert res["exact_p"] == pytest.approx(
            binomtest(2, 12, 0.5, alternative="two-sided").pvalue
        )
        # Continuity-corrected chi-square: (|10-2| - 1)^2 / 12 = 49/12.
        assert res["chi2_stat"] == pytest.approx(49 / 12)
        assert res["chi2_p"] == pytest.approx(float(chi2.sf(49 / 12, df=1)))
        assert res["exact_p"] < 0.05  # significant at alpha=0.05

    def test_equal_discordant_counts_give_p_one(self):
        a, b_arr = _paired_correct(5, 5)
        res = mcnemar_test(a, b_arr)
        assert res["b"] == 5
        assert res["c"] == 5
        assert res["exact_p"] == pytest.approx(1.0)

    def test_no_discordant_pairs(self):
        a, b_arr = _paired_correct(0, 0)
        res = mcnemar_test(a, b_arr)
        assert res == {
            "b": 0, "c": 0, "n_discordant": 0,
            "exact_p": 1.0, "chi2_stat": 0.0, "chi2_p": 1.0,
        }

    def test_symmetric_in_model_order(self):
        a, b_arr = _paired_correct(10, 2)
        res_ab = mcnemar_test(a, b_arr)
        res_ba = mcnemar_test(b_arr, a)
        assert res_ab["b"] == res_ba["c"]
        assert res_ab["c"] == res_ba["b"]
        assert res_ab["exact_p"] == pytest.approx(res_ba["exact_p"])

    def test_accepts_int_arrays(self):
        res = mcnemar_test(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]))
        assert res["b"] == 1
        assert res["c"] == 1

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            mcnemar_test(np.ones(3, dtype=bool), np.ones(4, dtype=bool))


class TestAlignPredictions:
    def test_aligns_on_clip_id_regardless_of_row_order(self):
        df_a = pd.DataFrame(
            {"clip_id": ["hello_signbsl_001", "cat_signbsl_001", "dog_signbsl_001"],
             "correct": [1, 0, 1]}
        )
        df_b = pd.DataFrame(
            {"clip_id": ["dog_signbsl_001", "hello_signbsl_001", "cat_signbsl_001"],
             "correct": [0, 1, 1]}
        )
        ca, cb = align_predictions(df_a, df_b)
        # Pairs after alignment: hello (1,1), cat (0,1), dog (1,0).
        res = mcnemar_test(ca, cb)
        assert res["b"] == 1
        assert res["c"] == 1

    def test_inner_join_drops_unmatched_clips(self):
        df_a = pd.DataFrame({"clip_id": ["a_x_001", "b_x_001"], "correct": [1, 0]})
        df_b = pd.DataFrame({"clip_id": ["b_x_001"], "correct": [1]})
        ca, cb = align_predictions(df_a, df_b)
        assert len(ca) == len(cb) == 1
        assert not ca[0] and cb[0]

    def test_no_overlap_raises(self):
        df_a = pd.DataFrame({"clip_id": ["a_x_001"], "correct": [1]})
        df_b = pd.DataFrame({"clip_id": ["b_x_001"], "correct": [1]})
        with pytest.raises(ValueError):
            align_predictions(df_a, df_b)


class TestTopkAccuracy:
    # 4 samples, 5 classes; hits at k=1: s0, s2; at k=2: + s3; at k=3: + s1.
    LOGITS = np.array(
        [
            [5.0, 1.0, 0.0, 0.0, 0.0],  # label 0: top-1 hit
            [1.0, 5.0, 0.0, 0.0, 0.0],  # label 2: only in top-3
            [0.0, 0.0, 3.0, 2.0, 1.0],  # label 2: top-1 hit
            [0.0, 1.0, 0.0, 0.0, 2.0],  # label 1: top-2 hit
        ]
    )
    LABELS = np.array([0, 2, 2, 1])

    def test_hand_computed_topk(self):
        assert compute_topk_accuracy(self.LOGITS, self.LABELS, 1) == pytest.approx(0.5)
        assert compute_topk_accuracy(self.LOGITS, self.LABELS, 2) == pytest.approx(0.75)
        assert compute_topk_accuracy(self.LOGITS, self.LABELS, 3) == pytest.approx(1.0)

    def test_perfect_one_hot_logits(self):
        labels = np.array([0, 1, 2])
        logits = np.eye(3)
        assert compute_topk_accuracy(logits, labels, 1) == 1.0

    def test_k_equal_n_classes_is_always_one(self):
        assert compute_topk_accuracy(self.LOGITS, self.LABELS, 5) == 1.0


class TestMacroF1:
    def test_hand_computed_three_classes(self):
        # preds = [0, 1, 1, 1] for labels [0, 0, 1, 2]:
        # class 0: P=1, R=1/2 -> F1=2/3; class 1: P=1/3, R=1 -> F1=1/2;
        # class 2: no predictions -> F1=0. macro = (2/3 + 1/2 + 0) / 3 = 7/18.
        logits = np.array(
            [[3.0, 1.0, 0.0], [0.5, 2.0, 0.0], [0.0, 5.0, 1.0], [0.0, 4.0, 2.0]]
        )
        labels = np.array([0, 0, 1, 2])
        assert compute_macro_f1(logits, labels) == pytest.approx(7 / 18)

    def test_averages_over_all_model_classes(self):
        # Same case padded to 5 classes: absent classes contribute F1=0,
        # so the macro average divides by 5, not 3.
        logits = np.array(
            [[3.0, 1.0, 0.0], [0.5, 2.0, 0.0], [0.0, 5.0, 1.0], [0.0, 4.0, 2.0]]
        )
        logits = np.pad(logits, ((0, 0), (0, 2)), constant_values=-10.0)
        labels = np.array([0, 0, 1, 2])
        assert compute_macro_f1(logits, labels) == pytest.approx(7 / 30)

    def test_perfect_predictions(self):
        labels = np.array([0, 1, 2, 0, 1, 2])
        logits = np.eye(3)[labels]
        assert compute_macro_f1(logits, labels) == pytest.approx(1.0)
