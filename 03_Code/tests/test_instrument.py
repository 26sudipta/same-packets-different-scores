"""
Unit tests for the Phase 3 instrument (steps 3.1 to 3.8).

Run from the workspace root:
    .venv/bin/python -m unittest discover -s 03_Code/tests -v

The hand check tests rebuild conversations and pairs from the records of the
step 3.4 hand check cases (fixtures/) and compare them with the decisions made
by hand, recorded in 00_Plan/Phase3_Hand_Check.md.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import conv_scoring as S  # noqa: E402
import instrument as I  # noqa: E402

FIX = HERE / "fixtures"
FROZEN = HERE.parent.parent / "04_Results" / "instrument"
CIC_ARMS = I.BENCHMARKS["CICIDS2017"]["arms"]
CIC_DEF = I.BENCHMARKS["CICIDS2017"]["defining"]


def frame(rows):
    """Records from (arm, src, sport, dst, dport, proto, t0, t1, label) tuples."""
    df = pd.DataFrame(rows, columns=["arm", "src", "sport", "dst", "dport", "proto",
                                     "t0", "t1", "label"])
    df["file"] = "f"
    df["row"] = np.arange(len(df))
    df["bytes"] = 1.0
    return I.add_keys(df)


class TestConversationKey(unittest.TestCase):
    def test_gap_rule(self):
        r = frame([("lycos", "a", 1, "b", 2, 6, 0.0, 10.0, "x"),
                   ("lycos", "a", 1, "b", 2, 6, 69.0, 70.0, "x"),     # 59 s gap: same
                   ("lycos", "a", 1, "b", 2, 6, 131.0, 132.0, "x")])  # 61 s gap: new
        conv, t, _ = I.build_conversations(r, 60, ["lycos"])
        self.assertEqual(conv[0], conv[1])
        self.assertNotEqual(conv[1], conv[2])

    def test_other_protocols_share_one_key(self):
        # UNSW case 17: nProbe writes protocol 14, Argus a name with no number (-1)
        r = frame([("unsw_original", "a", 0, "b", 0, -1, 0.0, 1.0, "x"),
                   ("nf_unsw_v3", "a", 0, "b", 0, 14, 0.2, 0.3, "x"),
                   ("nf_unsw_v3", "a", 0, "b", 0, 1, 0.2, 0.3, "x")])
        conv, t, _ = I.build_conversations(r, 15, ["unsw_original", "nf_unsw_v3"])
        self.assertEqual(conv[0], conv[1])
        self.assertNotEqual(conv[0], conv[2])
        self.assertEqual(list(r["proto"]), [-1, 14, 1])  # the record keeps its own

    def test_gap_measured_from_latest_end(self):
        # a long record keeps the conversation open past a later short one
        r = frame([("lycos", "a", 1, "b", 2, 6, 0.0, 500.0, "x"),
                   ("lycos", "a", 1, "b", 2, 6, 10.0, 11.0, "x"),
                   ("lycos", "a", 1, "b", 2, 6, 520.0, 521.0, "x")])
        conv, _, _ = I.build_conversations(r, 60, ["lycos"])
        self.assertEqual(len(set(conv)), 1)

    def test_direction_and_reversed(self):
        r = frame([("lycos", "b", 2, "a", 1, 6, 5.0, 6.0, "x"),
                   ("bccc2017", "a", 1, "b", 2, 6, 1.0, 9.0, "x")])
        conv, t, _ = I.build_conversations(r, 60, ["lycos", "bccc2017"])
        self.assertEqual(len(t), 1)
        self.assertEqual((t.loc[0, "a_ip"], t.loc[0, "a_port"]), ("a", 1))
        rev = I.reversed_flag(r, conv, t)
        self.assertEqual(rev.tolist(), [True, False])

    def test_minute_clock_picks_by_duration(self):
        # two conversations start inside the same minute; the 30 s record joins the 30 s one
        r = frame([("lycos", "a", 1, "b", 2, 6, 100.0, 110.0, "x"),
                   ("lycos", "a", 1, "b", 2, 6, 130.0, 160.0, "x"),
                   ("cic_original", "a", 1, "b", 2, 6, 120.0, 210.0, "x")])
        conv, _, st = I.build_conversations(r, 15, ["lycos"])
        self.assertEqual(conv[2], conv[1])
        self.assertEqual(st["n_ambiguous"], 1)

    def test_minute_clock_mapped_not_defining(self):
        # cic_original at minute resolution joins the conversation it can reach
        r = frame([("lycos", "a", 1, "b", 2, 6, 125.0, 126.0, "x"),
                   ("cic_original", "a", 1, "b", 2, 6, 120.0, 181.0, "x"),
                   ("cic_original", "a", 1, "b", 2, 6, 600.0, 661.0, "x")])
        conv, t, st = I.build_conversations(r, 60, ["lycos"])
        self.assertEqual(conv[0], conv[1])
        self.assertNotEqual(conv[0], conv[2])
        self.assertEqual(st["n_mapped"], 1)
        self.assertEqual(st["n_unmapped"], 1)

    def test_ports_zero_off_tcp_udp(self):
        r = frame([("lycos", "a", 8, "b", 0, 1, 0.0, 1.0, "x")])
        self.assertTrue(r["key_str"].astype(str).iloc[0].endswith("|1"))


class TestGroundTruth(unittest.TestCase):
    def table(self, rows):
        return pd.DataFrame(rows, columns=["conv", "a_ip", "a_port", "b_ip", "b_port",
                                           "proto", "cs", "ce"])

    def test_schedule_either_order_and_window(self):
        t = self.table([
            (0, "172.16.0.1", 5000, "192.168.10.50", 21, 6, 1499170700.0, 1499170710.0),
            (1, "192.168.10.50", 21, "172.16.0.1", 5001, 6, 1499170700.0, 1499170710.0),
            (2, "172.16.0.1", 5002, "192.168.10.50", 21, 6, 1499170000.0, 1499170900.0),
            (3, "172.16.0.1", 5003, "192.168.10.51", 21, 6, 1499170700.0, 1499170710.0)])
        lab, _, _ = I.label_cic(t, "Tuesday", "start")
        self.assertEqual(lab.tolist(), ["ftp_patator", "ftp_patator", "benign", "benign"])
        alt, _, _ = I.label_cic(t, "Tuesday", "overlap")
        self.assertEqual(alt[2], "ftp_patator")

    def test_heartbleed_ports(self):
        t = self.table([(0, "172.16.0.1", 45022, "192.168.10.51", 444, 6, 1.0, 2.0),
                        (1, "172.16.0.1", 45023, "192.168.10.51", 444, 6, 1.0, 2.0)])
        lab, _, _ = I.label_cic(t, "Wednesday")
        self.assertEqual(lab.tolist(), ["heartbleed", "benign"])

    def test_excluded(self):
        t = self.table([(0, "1.1.1.1", 1, "2.2.2.2", 2, 6, 1499353300.0, 1499353301.0),
                        (1, "52.6.13.28", 80, "192.168.10.5", 3, 6, 1499430000.0, 1.0)])
        self.assertEqual(I.label_cic(t, "Thursday")[0][0], I.EXCLUDED)
        self.assertEqual(I.label_cic(t, "Friday")[0][1], I.EXCLUDED)

    def test_canonical_names(self):
        f = lambda x: I.canonical_cic(x)
        self.assertEqual(f("Web Attack \x96 Brute Force"), ("web_bruteforce", False))
        self.assertEqual(f("DoS Slowhttptest - Attempted"), ("dos_slowhttptest", True))
        self.assertEqual(f("DDoS_LOIT"), ("ddos", False))
        self.assertEqual(f("Botnet_ARES"), ("bot", False))
        self.assertEqual(f("Port_scan"), ("portscan", False))
        self.assertEqual(f("webattack_sql_injection"), ("web_sqli", False))
        self.assertEqual(f("DoS_Golden_Eye")[0], "dos_goldeneye")
        self.assertEqual(I.canonical_unsw("Backdoors"), "backdoor")
        self.assertEqual(I.canonical_unsw(" Reconnaissance "), "reconnaissance")


class TestPairing(unittest.TestCase):
    def test_minute_clock_pairs_by_duration(self):
        # hand check case 2: two CIC records in the same minute, the long one is the match
        r = frame([("cic_original", "a", 1, "b", 2, 6, 440.0, 500.000092, "x"),
                   ("cic_original", "a", 1, "b", 2, 6, 440.0, 515.277691, "x"),
                   ("wtmc2021", "a", 1, "b", 2, 6, 457.0, 473.278421, "x")])
        conv, _, _ = I.build_conversations(r, 60, ["wtmc2021"])
        r["rev"] = False
        p, _ = I.pair_arms(r, conv, "cic_original", "wtmc2021")
        self.assertEqual(p[["ia", "ib"]].values.tolist(), [[1, 2]])

    def test_minute_clock_start_window_is_one_sided(self):
        # hand check case 5: a CIC record stamped in the next minute cannot share the
        # first packet of a record that starts at second 58, however close its duration
        r = frame([("cic_original", "a", 1, "b", 2, 6, 300.0, 360.236820, "x"),
                   ("cic_original", "a", 1, "b", 2, 6, 360.0, 465.056008, "x"),
                   ("wtmc2021", "a", 1, "b", 2, 6, 358.0, 450.988697, "x")])
        conv, _, _ = I.build_conversations(r, 15, ["wtmc2021"])
        r["rev"] = False
        p, _ = I.pair_arms(r, conv, "cic_original", "wtmc2021")
        self.assertEqual(p[["ia", "ib"]].values.tolist(), [[0, 2]])

    def test_no_pair_without_overlap(self):
        r = frame([("lycos", "a", 1, "b", 2, 6, 10.0, 10.000004, "x"),
                   ("bccc2017", "a", 1, "b", 2, 6, 10.024, 10.024001, "x")])
        conv, _, _ = I.build_conversations(r, 60, ["lycos", "bccc2017"])
        r["rev"] = False
        p, _ = I.pair_arms(r, conv, "lycos", "bccc2017")
        self.assertEqual(len(p), 0)

    def test_start_match_beats_overlap(self):
        # hand check case 14: a merged record pairs with the piece it starts with
        r = frame([("lycos", "a", 1, "b", 2, 17, 90.1976, 90.2259, "x"),
                   ("lycos", "a", 1, "b", 2, 17, 344.2807, 344.3287, "x"),
                   ("bccc2017", "a", 1, "b", 2, 17, 90.1976, 344.3287, "x")])
        conv, _, _ = I.build_conversations(r, 300, ["lycos", "bccc2017"])
        r["rev"] = False
        p, _ = I.pair_arms(r, conv, "lycos", "bccc2017")
        self.assertEqual(p[["ia", "ib"]].values.tolist(), [[0, 2]])

    def test_cardinality(self):
        t = pd.DataFrame({"n_x": [1, 2, 1, 2, 1, 0, 0], "n_y": [1, 1, 3, 2, 0, 1, 0]})
        self.assertEqual(I.cardinality(t, "x", "y").tolist(),
                         ["1:1", "n:1", "1:n", "n:m", "a_only", "b_only", "none"])


class TestSplits(unittest.TestCase):
    def test_deterministic_and_stratified(self):
        units = [f"u{i}" for i in range(1000)]
        strata = ["a"] * 900 + ["b"] * 100
        s1 = I.stratified_hash_split(units, strata, 0)
        s2 = I.stratified_hash_split(units, strata, 0)
        s3 = I.stratified_hash_split(units, strata, 1)
        self.assertTrue((s1 == s2).all())
        self.assertFalse((s1 == s3).all())
        self.assertEqual(int(s1[:900].sum()), 270)
        self.assertEqual(int(s1[900:].sum()), 30)

    def test_order_free(self):
        units = [f"u{i}" for i in range(50)]
        strata = ["a"] * 50
        s = I.stratified_hash_split(units, strata, 4)
        rev = I.stratified_hash_split(units[::-1], strata, 4)[::-1]
        self.assertTrue((s == rev).all())


class TestScoring(unittest.TestCase):
    def test_flow_equals_conversation_on_one_to_one(self):
        rng = np.random.default_rng(0)
        classes = np.array(["benign", "dos", "scan", "brute"])
        n = 2000
        conv = rng.permutation(n)            # one record per conversation
        y = classes[rng.integers(0, 4, n)]
        p = np.where(rng.random(n) < 0.7, y, classes[rng.integers(0, 4, n)])
        flow = S.score(conv, y, p, "flow")
        for rule in S.RULES:
            c = S.score(conv, y, p, "conversation", rule, weight=rng.random(n))
            self.assertAlmostEqual(flow["macro_f1"], c["macro_f1"], places=12)
            self.assertEqual(flow["per_class"], c["per_class"])

    def test_rules(self):
        conv = [0, 0, 0, 1, 1, 2]
        pred = ["benign", "dos", "benign", "scan", "scan", "benign"]
        w = [100, 1, 100, 1, 1, 5]
        self.assertEqual(S.aggregate(conv, pred, "any").tolist(), ["dos", "scan", "benign"])
        self.assertEqual(S.aggregate(conv, pred, "majority").tolist(),
                         ["benign", "scan", "benign"])
        self.assertEqual(S.aggregate(conv, pred, "bytes", w).tolist(),
                         ["benign", "scan", "benign"])

    def test_truth_must_be_constant(self):
        with self.assertRaises(ValueError):
            S.conversation_truth([0, 0], ["dos", "benign"])


def load_fixture(name):
    f = FIX / f"handcheck_{name}.csv"
    if not f.is_file():
        return None
    fx = pd.read_csv(f, dtype={"src": str, "dst": str, "file": str, "label": str})
    fx = fx.sort_values("idx", kind="mergesort").reset_index(drop=True)
    orig = fx["idx"].to_numpy()
    cols = ["arm", "file", "row", "src", "sport", "dst", "dport", "proto", "t0", "t1",
            "bytes", "label"]
    r = I.add_keys(fx[cols])
    return fx, r, orig


@unittest.skipUnless((FIX / "handcheck_CICIDS2017_Tuesday.csv").is_file(), "no fixture")
class TestHandCheckCIC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fx, cls.r, cls.orig = load_fixture("CICIDS2017_Tuesday")
        cls.conv, cls.t, _ = I.build_conversations(cls.r, I.DEFAULT_T, CIC_DEF)
        cls.r["rev"] = I.reversed_flag(cls.r, cls.conv, cls.t)
        cls.pos = {o: i for i, o in enumerate(cls.orig)}

    def test_partition_matches_frozen(self):
        a = pd.Series(self.conv).groupby(self.fx["conv"].to_numpy()).nunique()
        b = pd.Series(self.fx["conv"].to_numpy()).groupby(self.conv).nunique()
        self.assertTrue((a == 1).all() and (b == 1).all())

    def pairs(self, a, b):
        p, _ = I.pair_arms(self.r, self.conv, a, b)
        return {(int(self.orig[x]), int(self.orig[y])) for x, y in zip(p["ia"], p["ib"])}

    def test_hand_decisions(self):
        # each line: arm a, arm b, (record a, record b) expected, or (record a, None)
        expected = [
            ("cic_original", "wtmc2021", (417392, 594869)),     # case 2
            ("cic_original", "nfs2023_te", (415866, 2027670)),  # case 2, tails
            ("cic_original", "wtmc2021", (21544, 595279)),      # case 1
            ("cic_original", "nfs2023_te", (21208, 1844288)),   # case 1, tails
            ("cic_original", "wtmc2021", (204214, 649641)),     # case 7
            ("lycos", "bccc2017", (1331481, None)),             # case 7, no overlap
            ("cic_original", "bccc2017", (19170, 2231135)),     # case 14
            ("lycos", "bccc2017", (1116861, 2231135)),          # case 14
            ("lycos", "bccc2017", (1394537, None)),             # case 15
            ("cic_original", "lycos", (154711, 1277997)),       # case 5
            ("cic_original", "wtmc2021", (155227, None)),       # case 5, starts a minute late
        ]
        cache = {}
        for a, b, (x, y) in expected:
            if x not in self.pos:
                self.skipTest(f"record {x} not in fixture")
            got = cache.setdefault((a, b), self.pairs(a, b))
            if y is None:
                self.assertFalse(any(p[0] == x for p in got), (a, b, x))
            else:
                self.assertIn((x, y), got)

    def test_one_to_one(self):
        for a, b in [("cic_original", "bccc2017"), ("nfs2023_nte", "nfs2023_te"),
                     ("lycos", "bccc2017")]:
            p, _ = I.pair_arms(self.r, self.conv, a, b)
            self.assertFalse(p["ia"].duplicated().any())
            self.assertFalse(p["ib"].duplicated().any())


@unittest.skipUnless((FROZEN / "CICIDS2017" / "Tuesday" / "records.parquet").is_file(),
                     "no frozen Tuesday table")
class TestFrozenTuesday(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rec = pd.read_parquet(FROZEN / "CICIDS2017" / "Tuesday" / "records.parquet")
        cls.tab = pd.read_parquet(FROZEN / "CICIDS2017" / "Tuesday" / "conversations.parquet")

    def test_scoring_one_to_one_arm_real(self):
        # LYCOS conversations with exactly one LYCOS record: flow and conversation scores agree
        one = set(self.tab.loc[self.tab["n_lycos"] == 1, "conv"])
        r = self.rec[(self.rec["arm"] == "lycos") & self.rec["conv"].isin(one) &
                     (self.rec["sched"] != I.EXCLUDED)]
        rng = np.random.default_rng(1)
        cls = np.array(sorted(r["sched"].unique()))
        p = np.where(rng.random(len(r)) < 0.8, r["sched"].to_numpy(),
                     cls[rng.integers(0, len(cls), len(r))])
        f, c = S.score_both(r["conv"].to_numpy(), r["sched"].to_numpy(), p)
        self.assertAlmostEqual(f["macro_f1"], c["macro_f1"], places=12)

    def test_grouped_split_has_no_leakage(self):
        for s in I.SEEDS:
            g = self.rec[self.rec[f"g{s}"] >= 0].groupby("conv")[f"g{s}"].nunique()
            self.assertTrue((g == 1).all())

    def test_split_same_traffic_every_arm(self):
        # a conversation is on the same side in every arm
        x = self.rec[self.rec["g0"] >= 0].groupby("conv")["g0"].nunique()
        self.assertEqual(int(x.max()), 1)

    def test_labels_constant_in_conversation(self):
        self.assertEqual(int(self.rec.groupby("conv")["sched"].nunique().max()), 1)


if __name__ == "__main__":
    unittest.main()
