"""Phase 4 checks: the pattern based Krippendorff's alpha in agreement.py against the
independent `krippendorff` package, and the split cause rules on synthetic records."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import agreement as A


def ours(codes, K):
    P, c = A.patterns(codes, K)
    return A.alpha(P, c, K)["alpha"]


def theirs(codes):
    import krippendorff
    rd = codes.T.astype(float)
    rd[rd < 0] = np.nan
    return krippendorff.alpha(reliability_data=rd, level_of_measurement="nominal")


class TestAlpha(unittest.TestCase):
    def test_against_package_random(self):
        rng = np.random.default_rng(1)
        for K, raters, miss in [(2, 7, 0.0), (5, 7, 0.3), (12, 2, 0.1), (3, 4, 0.5)]:
            truth = rng.integers(0, K, 3000)
            codes = np.where(rng.random((3000, raters)) < 0.85, truth[:, None],
                             rng.integers(0, K, (3000, raters))).astype(np.int16)
            codes[rng.random(codes.shape) < miss] = -1
            self.assertAlmostEqual(ours(codes, K), theirs(codes), places=9)

    def test_perfect_and_merge(self):
        codes = np.array([[0, 0, 0], [1, 1, -1], [2, 2, 2], [0, 0, -1]], dtype=np.int16)
        self.assertAlmostEqual(ours(codes, 3), 1.0)
        a, b = codes[:2], codes[2:]
        P, c = A.merge_patterns([A.patterns(a, 3), A.patterns(b, 3)], 3)
        P2, c2 = A.patterns(codes, 3)
        self.assertAlmostEqual(A.alpha(P, c, 3)["alpha"], A.alpha(P2, c2, 3)["alpha"])

    def test_bootstrap_contains_point(self):
        rng = np.random.default_rng(2)
        codes = rng.integers(0, 2, (2000, 3)).astype(np.int16)
        codes[:1500, 1] = codes[:1500, 0]
        P, c = A.patterns(codes, 2)
        r = A.alpha(P, c, 2, boot=200, rng=np.random.default_rng(0))
        self.assertLess(r["alpha_lo"], r["alpha"])
        self.assertGreater(r["alpha_hi"], r["alpha"])


class TestSplitCauses(unittest.TestCase):
    def test_cases(self):
        # one arm with microsecond clock; conversation 1: a 120 s record cut by the
        # cap, its continuation, then a 1 ms tail; conversation 2: a record after a
        # 5 s silence; conversation 3: a cumulative record starting at the same time
        r = 1e-6
        rows = [(1, 0.0, 120.0), (1, 120.0, 130.0), (1, 130.2, 130.201),
                (2, 0.0, 1.0), (2, 6.0, 7.0),
                (3, 0.0, 2.0), (3, 0.0, 4.0)]
        rec = pd.DataFrame(rows, columns=["conv", "t0", "end"])
        rec["t1"] = rec["end"] + r
        rec["arm"] = "lycos"
        rec["sched"] = "benign"
        rec["rev"] = False
        s = A.split_counts(rec, {"lycos": 120.0})
        s = s[s["class"] == "ALL"].iloc[0]
        self.assertEqual(s["extra_records"], 4)
        self.assertEqual((s["cut"], s["tail"], s["idle"], s["overlap"], s["other"]),
                         (1, 1, 1, 1, 0))


class TestMatrixCounts(unittest.TestCase):
    def test_pair_classes_after_records_dropped(self):
        # records 0 to 3 of arm x and 4 to 7 of arm y, paired 0-4, 2-6, 3-7; record 1 is
        # dropped (as variant unsw_dups_dropped drops duplicates), so from record 2 on an
        # id is no longer a position in the table
        rec = pd.DataFrame({"idx": np.arange(8), "arm": ["x"] * 4 + ["y"] * 4,
                            "conv": [0, 0, 1, 2] * 2,
                            "sched": ["benign", "benign", "dos", "worms"] * 2})
        rec = rec[rec["idx"] != 1]
        pairs = pd.DataFrame({"ia": [0, 2, 3], "ib": [4, 6, 7], "arm_a": "x", "arm_b": "y"})
        t = pd.DataFrame({"conv": [0, 1, 2], "sched": ["benign", "dos", "worms"],
                          "n_x": [1, 1, 1], "n_y": [2, 1, 1]})
        m = A.matrix_counts(t, rec, pairs, ["x", "y"]).set_index("class")
        self.assertEqual(m.loc["benign", "flow_pairs"], 1)
        self.assertEqual(m.loc["dos", "flow_pairs"], 1)
        self.assertEqual(m.loc["worms", "flow_pairs"], 1)
        self.assertEqual(m.loc["ATTACK", "flow_pairs"], 2)
        self.assertEqual(m.loc["ALL", "flow_pairs"], 3)


if __name__ == "__main__":
    unittest.main()
