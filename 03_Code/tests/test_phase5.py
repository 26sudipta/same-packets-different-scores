"""Tests for the Phase 5 analysis: variance shares, Kendall's W, bootstrap counts.

Run from the workspace root:  .venv/bin/python -m unittest 03_Code/tests/test_phase5.py
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import phase5_analysis as P  # noqa: E402


class TestShares(unittest.TestCase):
    def test_matches_statsmodels_type1(self):
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import anova_lm
        rng = np.random.default_rng(1)
        for _ in range(4):
            A, M, S = int(rng.integers(2, 6)), 3, 5
            F = rng.random((A, M, S)) + rng.random((A, 1, 1)) + 0.3 * rng.random((1, M, 1))
            df = pd.DataFrame([{"arm": a, "model": m, "seed": s, "f1": F[a, m, s]}
                               for a in range(A) for m in range(M) for s in range(S)])
            fit = smf.ols("f1 ~ C(arm) * C(model) + C(seed)", data=df).fit()
            t = anova_lm(fit, typ=1)
            tot = t["sum_sq"].sum()
            want = [t.loc["C(arm)", "sum_sq"] / tot, t.loc["C(model)", "sum_sq"] / tot,
                    t.loc["C(arm):C(model)", "sum_sq"] / tot, t.loc["C(seed)", "sum_sq"] / tot,
                    t.loc["Residual", "sum_sq"] / tot]
            np.testing.assert_allclose(P.ss_shares(F), want, atol=1e-10)

    def test_constant_is_nan(self):
        self.assertTrue(np.isnan(P.ss_shares(np.ones((2, 3, 5)))).all())


class TestKendall(unittest.TestCase):
    def test_perfect_and_reversed(self):
        X = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.9], [0.0, 0.5, 0.7]])
        self.assertAlmostEqual(float(P.kendall_w(X)), 1.0)
        Y = np.array([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]])
        self.assertAlmostEqual(float(P.kendall_w(Y)), 0.0)

    def test_textbook_value(self):
        # 3 raters rank 4 items (higher score, higher rank); rank sums 4, 6, 9, 11,
        # mean 7.5, S = 29, W = 12 S / (m^2 (n^3 - n)) = 348 / 540
        X = np.array([[1, 2, 3, 4], [1, 3, 2, 4], [2, 1, 4, 3]], dtype=float)
        self.assertAlmostEqual(float(P.kendall_w(X)), 12 * 29 / (9 * 60))

    def test_ties(self):
        # one rater ties two items: T = 2^3 - 2 = 6
        X = np.array([[1.0, 1.0, 2.0], [1.0, 2.0, 3.0]])
        R = np.array([[1.5, 1.5, 3.0], [1.0, 2.0, 3.0]]).sum(axis=0)
        S = ((R - R.mean()) ** 2).sum()
        self.assertAlmostEqual(float(P.kendall_w(X)), 12 * S / (4 * 24 - 2 * 6))


class TestBootWeights(unittest.TestCase):
    def test_class_sizes_kept(self):
        u = np.array(["a"] * 5 + ["b"] * 3 + ["c"], dtype=object)
        W = P.boot_weights(u, 20, np.random.default_rng(0))
        np.testing.assert_array_equal(W[:, u == "a"].sum(axis=1), 5)
        np.testing.assert_array_equal(W[:, u == "b"].sum(axis=1), 3)
        np.testing.assert_array_equal(W[:, u == "c"].sum(axis=1), 1)


if __name__ == "__main__":
    unittest.main()
