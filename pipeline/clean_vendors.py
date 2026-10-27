"""
Stage 1a - Find duplicate vendors (the data-quality layer).

How it works:
  1. Normalize each vendor name (lowercase, & -> and, drop punctuation and
     endings like Inc / LLC / Corp). Records with the same normalized name merge.
  2. Fuzzy-match the remaining names inside small blocks (same first 2 letters).
       - score >= 95 and same state       -> merged automatically
       - 85-95, or >= 95 but other state  -> sent to a human review queue
       - one name contained in the other  -> review queue (e.g. "Grainger" vs "W W Grainger")
  3. Each group of merged records gets one vendor_id and a canonical name
     (the name with the most spend).

Creates tables vendor_map and vendor_review in data/spend.duckdb, and
writes eval/vendor_review_queue.csv for you to approve or reject.
Run after load.py:
    python pipeline/clean_vendors.py
"""
from collections import defaultdict
from pathlib import Path
import re
import time

import duckdb
import pandas as pd
from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "spend.duckdb"
REVIEW_CSV = ROOT / "eval" / "vendor_review_queue.csv"

AUTO_MERGE = 95
REVIEW_MIN = 85

LEGAL_ENDINGS = {
    "inc", "incorporated", "llc", "llp", "ltd", "limited", "lp", "corp",
    "corporation", "co", "company", "the", "plc", "pc", "pllc",
}


def normalize(name: str) -> str:
    n = str(name).lower()
    n = n.replace("&", " and ")
    n = re.sub(r"\bd/?b/?a\b.*$", " ", n)          # drop "dba ..." trade names
    n = re.sub(r"\bl\s*l\s*c\b", "llc", n)          # "l l c" -> llc
    n = re.sub(r"\bl\s*p\b", "lp", n)               # "l p" -> lp
    n = re.sub(r"[^a-z0-9 ]", " ", n)
    words = [w for w in n.split() if w not in LEGAL_ENDINGS]
    return " ".join(words)


class UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def main() -> None:
    start = time.time()
    con = duckdb.connect(str(DB_PATH))

    # One record per (vendor_code, vendor_name) with its address and spend
    v = con.execute("""
        SELECT vendor_code, vendor_name,
               any_value(vendor_state) AS state,
               any_value(vendor_zip5)  AS zip5,
               count(*)                AS line_items,
               coalesce(sum(line_total), 0) AS spend
        FROM po
        WHERE vendor_name IS NOT NULL
        GROUP BY vendor_code, vendor_name
    """).df()
    v["rec_id"] = range(len(v))
    v["norm"] = v["vendor_name"].map(normalize)

    uf = UnionFind(v["rec_id"])
    method = {}

    # Step 1: same normalized name -> merge
    for _, grp in v.groupby("norm"):
        ids = grp["rec_id"].tolist()
        for other in ids[1:]:
            uf.union(ids[0], other)
            method[other] = "exact_normalized"

    # Step 2: fuzzy match one representative per normalized name, inside blocks
    reps = v.sort_values("spend", ascending=False).drop_duplicates("norm")
    reps = reps[reps["norm"].str.len() >= 3]
    # Two blocking keys so "grainger" and "w w grainger" still get compared:
    # first 2 letters of the name, and first 5 letters of its longest word.
    blocks = defaultdict(list)
    for r in reps.to_dict("records"):
        longest = max(r["norm"].split(), key=len)
        blocks["A:" + r["norm"][:2]].append(r)
        if len(longest) >= 4:
            blocks["B:" + longest[:5]].append(r)

    pairs = {}
    for members in blocks.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                key = tuple(sorted((a["rec_id"], b["rec_id"])))
                pairs[key] = (a, b)

    review_rows = []
    auto_pairs = 0
    for a, b in pairs.values():
        score = fuzz.token_sort_ratio(a["norm"], b["norm"])
        contained = fuzz.token_set_ratio(a["norm"], b["norm"])
        same_state = a["state"] == b["state"]

        if score >= AUTO_MERGE and same_state:
            uf.union(a["rec_id"], b["rec_id"])
            method.setdefault(b["rec_id"], "fuzzy_auto")
            auto_pairs += 1
            continue

        needs_review = (REVIEW_MIN <= score < AUTO_MERGE) or (score >= AUTO_MERGE and not same_state)
        if not needs_review and contained == 100 and min(len(a["norm"]), len(b["norm"])) >= 5:
            needs_review = True
        if needs_review:
            review_rows.append({
                "name_a": a["vendor_name"], "code_a": a["vendor_code"], "state_a": a["state"],
                "spend_a": round(a["spend"], 2),
                "name_b": b["vendor_name"], "code_b": b["vendor_code"], "state_b": b["state"],
                "spend_b": round(b["spend"], 2),
                "similarity": round(score, 1), "one_contains_other": contained == 100,
                "decision": "",   # fill in: merge / keep_separate
            })

    # Step 3: assign vendor_id and canonical name (highest-spend name in each group)
    v["vendor_id"] = v["rec_id"].map(uf.find)
    canon = (v.sort_values("spend", ascending=False)
               .drop_duplicates("vendor_id")[["vendor_id", "vendor_name"]]
               .rename(columns={"vendor_name": "canonical_name"}))
    v = v.merge(canon, on="vendor_id")
    v["match_method"] = v["rec_id"].map(method).fillna("self")

    vendor_map = v[["vendor_code", "vendor_name", "norm", "vendor_id", "canonical_name",
                    "match_method", "state", "line_items", "spend"]]
    review = pd.DataFrame(review_rows)
    if len(review):
        review = review.sort_values(["similarity", "spend_a"], ascending=[False, False])

    con.execute("DROP TABLE IF EXISTS vendor_map")
    con.register("vm", vendor_map)
    con.execute("CREATE TABLE vendor_map AS SELECT * FROM vm")
    con.unregister("vm")
    con.execute("DROP TABLE IF EXISTS vendor_review")
    if len(review):
        con.register("vr", review)
        con.execute("CREATE TABLE vendor_review AS SELECT * FROM vr")
        con.unregister("vr")
    con.close()

    REVIEW_CSV.parent.mkdir(exist_ok=True)
    review.to_csv(REVIEW_CSV, index=False)

    records = len(v)
    groups = v["vendor_id"].nunique()
    print(f"Vendor records (code + name):        {records:,}")
    print(f"Distinct raw names:                  {v['vendor_name'].nunique():,}")
    print(f"Merged by normalized name:           {(v['match_method'] == 'exact_normalized').sum():,}")
    print(f"Merged by fuzzy match (auto):        {(v['match_method'] == 'fuzzy_auto').sum():,}")
    print(f"Unique vendors after cleaning:       {groups:,}  "
          f"({records - groups:,} duplicates removed, {1 - groups / records:.1%})")
    print(f"Pairs sent to human review:          {len(review):,}")

    merged = v[v["match_method"] != "self"]
    if len(merged):
        print("\nExamples of merged names:")
        ex = (v[v["vendor_id"].isin(merged["vendor_id"])]
                .groupby("vendor_id")["vendor_name"].apply(lambda s: " | ".join(sorted(set(s))))
                .head(12))
        for names in ex:
            print(f"  {names[:120]}")
    if len(review):
        print("\nTop review-queue pairs:")
        for _, r in review.head(10).iterrows():
            print(f"  {r['similarity']:5.1f}  {r['name_a'][:40]:<40} <-> {r['name_b'][:40]}")

    print(f"\nReview queue saved to {REVIEW_CSV}")
    print(f"Done in {time.time() - start:.1f} seconds")


if __name__ == "__main__":
    main()
