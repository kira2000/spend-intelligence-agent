"""
Stage 2b - Local LLM second pass for low-confidence items.

The embeddings step (categorize_embed.py) is confident on about half the items.
For the rest, Qwen 2.5 (running free in Ollama) reads the purchase text and
picks ONE category from the 5 candidates the embeddings step suggested.
The answer is forced into a JSON schema whose only allowed values are those
5 codes, so the model cannot invent a category.

Measures, on a random sample of low-confidence test items:
  - accuracy before (embeddings) vs after (LLM)
  - seconds per item on this laptop
  - projected overall accuracy for the full hybrid pipeline

Run after categorize_embed.py (Ollama must be running):
    python pipeline/categorize_llm.py                 # 300 items, threshold 0.7
    python pipeline/categorize_llm.py --limit 0       # all low-confidence items
Options: --threshold 0.7  --limit 300  --model qwen2.5:7b
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

import sys

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from categorize_embed import embed_texts, ollama_embed  # noqa: E402  (reuses cached embeddings)

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "spend.duckdb"
REPORT_PATH = ROOT / "eval" / "stage2_llm_report.txt"
SEED = 42
PROMPT_VERSION = "v2-examples"   # change this whenever the prompt changes, so old answers are not reused
K = 10

report: list[str] = []


def out(text: str = "") -> None:
    print(text)
    report.append(text)


# ---------------------------------------------------------------- category guide
def build_class_guide(con) -> dict[str, str]:
    """One short description per NIGP class, like a page of the NIGP codebook.

    Uses the official class name (3-digit code rows) when the data has one,
    otherwise the most common item names filed under that class.
    """
    rows = con.execute("""
        WITH names AS (
            SELECT nigp_class, commodity_desc, count(*) AS n,
                   max(CASE WHEN length(commodity_code) = 3 THEN 1 ELSE 0 END) AS is_class_row
            FROM po WHERE commodity_desc IS NOT NULL
            GROUP BY 1, 2
        ),
        ranked AS (
            SELECT *, row_number() OVER (PARTITION BY nigp_class
                                         ORDER BY is_class_row DESC, n DESC) AS r
            FROM names
        )
        SELECT nigp_class, string_agg(left(commodity_desc, 60), '; ' ORDER BY r) AS guide
        FROM ranked WHERE r <= 3 GROUP BY 1
    """).fetchall()
    return {c: g for c, g in rows}


# ---------------------------------------------------------------- evidence (retrieval)
def build_evidence(con, sample: pd.DataFrame) -> list[dict]:
    """For each item: per candidate class, its vote share from the 10 nearest past
    purchases and the 2 most similar past purchases filed under that class."""
    train = con.execute("SELECT text, nigp_class FROM knn_train").df()
    train_vecs = embed_texts(con, train["text"].tolist(), ollama_embed)
    test_vecs = embed_texts(con, sample["text"].tolist(), ollama_embed)
    labels = train["nigp_class"].to_numpy()
    texts = train["text"].to_numpy()

    evidence = []
    for i, (_, row) in enumerate(sample.iterrows()):
        sims = train_vecs @ test_vecs[i]
        top = np.argpartition(-sims, K)[:K]
        votes: dict[str, float] = {}
        for j in top:
            votes[labels[j]] = votes.get(labels[j], 0.0) + max(float(sims[j]), 0.0)
        total = sum(votes.values()) or 1.0
        item = {}
        for c in [c for c in row["top5"] if c]:
            idx = np.where(labels == c)[0]
            best = idx[np.argsort(-sims[idx])[:2]] if len(idx) else []
            item[c] = {"share": votes.get(c, 0.0) / total,
                       "examples": [str(texts[j])[:90] for j in best]}
        evidence.append(item)
    return evidence


# ---------------------------------------------------------------- LLM call
SYSTEM = (
    "You are a procurement analyst for the City of Austin. You assign purchase lines "
    "to NIGP commodity classes the way Austin's buyers do. For each candidate you get "
    "a short class description, a similarity vote from past purchases, and real past "
    "purchases Austin filed under that class. Austin's past filings matter more than "
    "the class name. The first candidate is the favourite from similar past purchases: "
    "keep it unless the purchase text clearly fits another candidate better. Ignore "
    "delivery locations, people's names and contract numbers."
)


def ollama_choose(model: str, text: str, candidates: list[str], guide: dict[str, str],
                  evidence: dict) -> dict:
    import ollama
    blocks = []
    for rank, c in enumerate(candidates, 1):
        ev = evidence.get(c, {"share": 0.0, "examples": []})
        examples = "\n".join(f"      past purchase: {e}" for e in ev["examples"]) or "      (no examples)"
        blocks.append(f"{rank}. class {c} - {guide.get(c, 'no description')[:120]}\n"
                      f"      similarity vote: {ev['share']:.0%}\n{examples}")
    prompt = (f"Purchase text:\n\"{text[:600]}\"\n\nCandidates (most likely first):\n"
              + "\n".join(blocks) +
              "\n\nAnswer with the class code and a short reason (max 15 words).")
    schema = {
        "type": "object",
        "properties": {
            "choice": {"type": "string", "enum": candidates},
            "reason": {"type": "string"},
        },
        "required": ["choice", "reason"],
    }
    resp = ollama.chat(
        model=model,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        format=schema,
        options={"temperature": 0},
    )
    content = resp["message"]["content"] if isinstance(resp, dict) else resp.message.content
    data = json.loads(content)
    if data.get("choice") not in candidates:      # safety net
        data["choice"] = candidates[0]
    return data


# ---------------------------------------------------------------- main
def main(choose_fn=ollama_choose) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--limit", type=int, default=300, help="0 = all low-confidence items")
    parser.add_argument("--model", default="qwen2.5:7b")
    args = parser.parse_args()

    con = duckdb.connect(str(DB_PATH))
    test = con.execute("SELECT * FROM knn_test").df()
    test["top5"] = test["top5"].str.split(",")
    guide = build_class_guide(con)

    low = test[test["confidence"] < args.threshold]
    sample = low if args.limit == 0 or args.limit >= len(low) else low.sample(args.limit, random_state=SEED)

    print("Gathering past-purchase examples for each candidate (uses cached embeddings)...")
    evidence = build_evidence(con, sample)

    con.execute("""CREATE TABLE IF NOT EXISTS llm_cache
                   (h VARCHAR PRIMARY KEY, model VARCHAR, choice VARCHAR, reason VARCHAR, seconds DOUBLE)""")

    out("STAGE 2b - LOCAL LLM SECOND PASS")
    out("=" * 60)
    out(f"Model: {args.model}   Confidence threshold: {args.threshold}")
    out(f"Low-confidence test items: {len(low):,} of {len(test):,} ({len(low) / len(test):.1%})")
    out(f"Items in this run: {len(sample):,}")
    out()

    results = []
    start = time.time()
    new_calls = 0
    for n, (_, row) in enumerate(sample.iterrows(), 1):
        cands = [c for c in row["top5"] if c]
        key = hashlib.sha1(f"{PROMPT_VERSION}|{args.model}|{row['text']}|{','.join(cands)}".encode()).hexdigest()
        hit = con.execute("SELECT choice, reason, seconds FROM llm_cache WHERE h = ?", [key]).fetchall()
        if hit:
            choice, reason, secs = hit[0]
        else:
            t0 = time.time()
            try:
                ans = choose_fn(args.model, row["text"], cands, guide, evidence[n - 1])
                choice, reason = ans["choice"], ans.get("reason", "")
            except Exception as exc:                       # keep going if one call fails
                choice, reason = cands[0], f"LLM error: {exc}"[:200]
            secs = time.time() - t0
            con.execute("INSERT OR REPLACE INTO llm_cache VALUES (?, ?, ?, ?, ?)",
                        [key, args.model, choice, reason, secs])
            new_calls += 1
        results.append({"llm_choice": choice, "llm_reason": reason, "llm_seconds": secs})
        if n % 25 == 0 or n == len(sample):
            elapsed = time.time() - start
            eta = elapsed / n * (len(sample) - n)
            print(f"  {n:,}/{len(sample):,}   about {eta / 60:.1f} min left")

    res = pd.concat([sample.reset_index(drop=True), pd.DataFrame(results)], axis=1)
    res["llm_correct"] = res["llm_choice"] == res["nigp_class"]

    knn_acc = res["correct"].mean()
    llm_acc = res["llm_correct"].mean()
    ceiling = res["in_top5"].mean()
    picked_right = res.loc[res["in_top5"], "llm_correct"].mean()
    fixed = int((~res["correct"] & res["llm_correct"]).sum())
    broke = int((res["correct"] & ~res["llm_correct"]).sum())
    secs = res["llm_seconds"].mean()

    out("RESULTS ON THE LOW-CONFIDENCE SAMPLE")
    out("-" * 60)
    out(f"Embeddings accuracy on these items:   {knn_acc:.1%}")
    out(f"LLM accuracy on the same items:       {llm_acc:.1%}   ({llm_acc - knn_acc:+.1%})")
    out(f"Ceiling (true class among the 5):     {ceiling:.1%}")
    out(f"LLM picked right when it could:       {picked_right:.1%}")
    out(f"Items the LLM fixed / broke:          {fixed} / {broke}")
    out(f"Average seconds per item:             {secs:.1f}  (new calls this run: {new_calls})")

    agree_wrong = res[(res["pred"] == res["llm_choice"]) & ~res["llm_correct"]]
    out(f"Both methods agree, official code differs: {len(agree_wrong)} "
        f"({len(agree_wrong) / len(res):.1%})  <- possible coding errors in the source data")

    out()
    out("WHERE DOES THE LLM HELP? (by embeddings confidence)")
    out("-" * 60)
    out(f"  {'confidence':<12}{'items':>7}{'embeddings':>12}{'LLM':>8}")
    for lo, hi in [(0.0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, args.threshold)]:
        part = res[(res["confidence"] >= lo) & (res["confidence"] < hi)]
        if len(part):
            out(f"  {lo:.1f}-{hi:.1f}{len(part):>11}{part['correct'].mean():>12.1%}{part['llm_correct'].mean():>8.1%}")

    out()
    out("WHICH RULE IS BEST? (use the LLM only below a cut-off, embeddings otherwise)")
    out("-" * 60)
    best_rule, best_acc = None, test["correct"].mean()
    for cut in [0.3, 0.4, 0.5, 0.6, args.threshold]:
        # items below the cut-off use the LLM accuracy measured on this sample's matching slice
        slice_ = res[res["confidence"] < cut]
        if not len(slice_):
            continue
        below = test["confidence"] < cut
        llm_gain = slice_["llm_correct"].mean() - slice_["correct"].mean()
        projected = test["correct"].mean() + below.mean() * llm_gain
        out(f"  LLM below {cut:.1f}: {below.mean():5.1%} of items to LLM -> projected accuracy {projected:.1%}"
            f"  (sample size {len(slice_)})")
        if projected > best_acc:
            best_rule, best_acc = cut, projected
    out(f"  Embeddings only:                               projected accuracy {test['correct'].mean():.1%}")
    out(f"  Best rule: " + (f"use the LLM below confidence {best_rule:.1f} ({best_acc:.1%})"
                            if best_rule else "embeddings only - the LLM does not add accuracy here"))

    # Projected accuracy of the whole hybrid pipeline on the full test set
    sure = test[test["confidence"] >= args.threshold]
    share_sure = len(sure) / len(test)
    # Use the accuracy GAIN measured on the sample (not its raw accuracy), because a random
    # sample can be a little easier or harder than the full low-confidence set.
    hybrid = test["correct"].mean() + (1 - share_sure) * (llm_acc - knn_acc)
    out()
    out("PROJECTED FULL PIPELINE (embeddings when confident, LLM otherwise)")
    out("-" * 60)
    out(f"Embeddings only:                      {test['correct'].mean():.1%}")
    out(f"Hybrid (embeddings + LLM):            {hybrid:.1%}")
    out(f"Time to run the LLM on all {len(low):,} low-confidence items: about {len(low) * secs / 3600:.1f} hours")

    for title, mask in [("Examples the LLM FIXED", ~res["correct"] & res["llm_correct"]),
                        ("Examples the LLM got WRONG", ~res["llm_correct"]),
                        ("Possible coding errors (both methods agree, official code differs)",
                         (res["pred"] == res["llm_choice"]) & ~res["llm_correct"])]:
        out()
        out(title + ":")
        for _, r in res[mask].head(6).iterrows():
            out(f"  text:   {r['text'][:90]}")
            out(f"  true {r['nigp_class']} ({guide.get(r['nigp_class'], '')[:45]})"
                f" | embeddings {r['pred']} | LLM {r['llm_choice']}")
            out(f"  reason: {str(r['llm_reason'])[:90]}")

    save = res.copy()
    save["top5"] = save["top5"].map(lambda c: ",".join(c))
    con.execute("DROP TABLE IF EXISTS llm_test")
    con.register("f", save)
    con.execute("CREATE TABLE llm_test AS SELECT * FROM f")
    con.unregister("f")
    con.close()

    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")
    print(f"\nReport saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
