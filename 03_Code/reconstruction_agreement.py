"""
Reconstruction agreement probe for CICIDS2017.

Measures how far three independent reconstructions of the same packet captures
agree: CIC original (CICFlowMeter), Engelen et al. WTMC2021 corrected, and
Rosay et al. LYCOS-IDS2017 (LycoSTand).

Produced the numbers in Research_Gap_Analysis_Journal_12Aug2026.md, section 3.
Run:  python3 reconstruction_agreement.py [day ...]

Caveat that matters: the conversation key is a DIRECTIONLESS five tuple with no
time component, so a five tuple reused later in the day collapses into one key.
Same key family as the E4 leakage-free split, so it is internally consistent,
but a time aware key is needed for the paper itself.
"""
import sys
import pandas as pd

DATA = ("/home/sudipta/Research/AI-Driven Cyber Threat Detection and Log Analysis/Datasets")

ORIG_FILES = {
    "Tuesday":   ["Tuesday-WorkingHours.pcap_ISCX.csv"],
    "Wednesday": ["Wednesday-workingHours.pcap_ISCX.csv"],
    "Thursday":  ["Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
                  "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"],
    "Friday":    ["Friday-WorkingHours-Morning.pcap_ISCX.csv",
                  "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
                  "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"],
    "Monday":    ["Monday-WorkingHours.pcap_ISCX.csv"],
}


def conv_key(df, sip, sp, dip, dp, proto):
    """Directionless five tuple: lower endpoint first, so A->B and B->A collide."""
    for c in (sp, dp, proto):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64").astype(str)
    x = df[sip].astype(str) + ":" + df[sp]
    y = df[dip].astype(str) + ":" + df[dp]
    lo = x.where(x <= y, y)
    hi = y.where(x <= y, x)
    return lo + "|" + hi + "|" + df[proto]


def load_original(day):
    cols = [" Source IP", " Source Port", " Destination IP",
            " Destination Port", " Protocol", " Label"]
    parts = [pd.read_csv(f"{DATA}/CICIDS2017_Original_LabelledFlows/{f}",
                         encoding="latin-1", low_memory=False, usecols=cols)
             for f in ORIG_FILES[day]]
    df = pd.concat(parts, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    df["k"] = conv_key(df, "Source IP", "Source Port",
                       "Destination IP", "Destination Port", "Protocol")
    return df


def load_corrected(day):
    df = pd.read_csv(f"{DATA}/CICIDS2017_Corrected_WTMC2021/{day}-WorkingHours.csv",
                     low_memory=False,
                     usecols=["Src IP", "Src Port", "Dst IP", "Dst Port",
                              "Protocol", "Label"])
    df["k"] = conv_key(df, "Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol")
    return df


def load_lycos(day):
    """LYCOS CSVs ship unlabelled (label == NeedLabel). Run their labelling.py first
    if you need labels; flow counts and keys work either way."""
    df = pd.read_csv(f"{DATA}/LYCOS-IDS2017_GitHubMirror/pcap_lycos/"
                     f"{day}-WorkingHours.pcap_lycos.csv", low_memory=False,
                     usecols=["src_addr", "src_port", "dst_addr", "dst_port",
                              "ip_prot", "label"])
    df["k"] = conv_key(df, "src_addr", "src_port", "dst_addr", "dst_port", "ip_prot")
    return df


def report(day, with_lycos=True):
    o, w = load_original(day), load_corrected(day)
    print(f"\n{'=' * 72}\n{day}\n{'=' * 72}")
    print(f"flow rows      original={len(o):>9,}  corrected={len(w):>9,}"
          f"  ratio={len(o)/len(w):.2f}")

    keys_o, keys_w = set(o.k), set(w.k)
    print(f"conversations  original={len(keys_o):>9,}  corrected={len(keys_w):>9,}"
          f"  jaccard={len(keys_o & keys_w) / len(keys_o | keys_w):.4f}")

    if with_lycos:
        try:
            l = load_lycos(day)
            keys_l = set(l.k)
            union = keys_o | keys_w | keys_l
            print(f"               lycos    ={len(l):>9,} flows, {len(keys_l):,} conversations")
            print(f"three way key agreement = "
                  f"{len(keys_o & keys_w & keys_l) / len(union):.4%}")
        except FileNotFoundError:
            print("               lycos file not found for this day, skipping")

    # cardinality of the original-to-corrected mapping, per conversation
    j = pd.concat([o.k.value_counts().rename("o"),
                   w.k.value_counts().rename("w")], axis=1).fillna(0).astype(int)
    cases = [("1:1", (j.o == 1) & (j.w == 1)),
             ("n:1  original split", (j.o > 1) & (j.w == 1)),
             ("1:n  corrected split", (j.o == 1) & (j.w > 1)),
             ("n:m", (j.o > 1) & (j.w > 1)),
             ("original only", j.w == 0),
             ("corrected only", j.o == 0)]
    print(f"\nmapping cardinality over {len(j):,} conversations")
    for name, mask in cases:
        print(f"  {name:24s} {mask.sum():>8,}  ({mask.sum()/len(j):6.2%})")

    # the same, restricted to conversations the original calls an attack
    atk = j.loc[j.index.intersection(o.loc[o.Label != "BENIGN", "k"].unique())]
    if len(atk):
        one_to_one = ((atk.o == 1) & (atk.w == 1)).sum()
        print(f"\nattack-labelled conversations: {len(atk):,}")
        print(f"  1:1 among them: {one_to_one:,} ({one_to_one/len(atk):.2%})")
        print(f"  flow rows on those keys: original={atk.o.sum():,}"
              f"  corrected={atk.w.sum():,}  ratio={atk.o.sum()/atk.w.sum():.2f}")

    # per class flows and conversations
    ol = o.groupby("Label").k.agg(orig_flows="size", orig_convs="nunique")
    wl = w.groupby("Label").k.agg(corr_flows="size", corr_convs="nunique")
    m = ol.join(wl, how="outer").fillna(0).astype(int)
    m["orig_fpc"] = (m.orig_flows / m.orig_convs.where(m.orig_convs > 0)).round(2)
    m["corr_fpc"] = (m.corr_flows / m.corr_convs.where(m.corr_convs > 0)).round(2)
    print("\nper class (fpc = flows per conversation)")
    print(m.to_string())


if __name__ == "__main__":
    days = sys.argv[1:] or ["Tuesday", "Wednesday", "Thursday", "Friday"]
    for d in days:
        report(d)
