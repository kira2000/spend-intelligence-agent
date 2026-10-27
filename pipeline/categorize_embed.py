"""
Stage 2a - Categorize purchase lines with embeddings + nearest neighbours.

1. Take usable lines from items_clean. Identical texts are collapsed into one
   "unique text" so the same sentence can never appear in both train and test
   (otherwise accuracy would be inflated by memorization).
2. Split unique texts into TRAIN (labels visible) and TEST (labels hidden).
3. Embed every text with nomic-embed-text in Ollama (cached, so re-runs are fast
   and an interrupted run picks up where it stopped).
4. For each test text, find the 10 most similar train texts and let them vote
   (weighted by similarity). Vote share = confidence. Keep the top 5 candidates
   for the LLM second pass.
5. Score against the hidden NIGP classes and save a report.

Run after clean_descriptions.py (Ollama must be running):
    python pipeline/categorize_embed.py
Options:
    --train 60000 --test 8000   sample sizes (defaults shown)
"""
from pathlib import Path
import argparse
import hashlib
import time

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "spend.duckdb"
REPORT_PATH = ROOT / "eval" / "stage2_knn_report.txt"

EMBED_MODEL = "nomic-embed-text"
BATCH = 64
K = 10
MAX_PER_CLASS_TRAIN = 1000   # stops a few huge classes from drowning out small ones
SEED = 42

report: list[str] = []


def out(text: str = "") -> None:
    print(text)
    report.append(text)


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- embeddings
def embed_texts(con, texts: list[str], embed_fn) -> np.ndarray:
    """Return an (n, dim) array, using and filling the emb_cache table."""
    hashes = [text_hash(t) for t in texts]
    has_cache = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = 'emb_cache'"
    ).fetchall()[0][0] > 0
    cached = set()
    if has_cache:
        cached = {r[0] for r in con.execute(
            "SELECT h FROM emb_cache WHERE model = ?", [EMBED_MODEL]).fetchall()}

    todo = [(h, t) for h, t in dict(zip(hashes, texts)).items() if h not in cached]
    if todo:
        out(f"Embedding {len(todo):,} new texts ({len(cached):,} already cached)...")
        start = time.time()
        for i in range(0, len(todo), BATCH):
            chunk = todo[i:i + BATCH]
            vectors = embed_fn([t for _, t in chunk])
            if not has_cache:
                dim = len(vectors[0])
                con.execute(f"CREATE TABLE emb_cache (h VARCHAR PRIMARY KEY, model VARCHAR, v FLOAT[{dim}])")
                has_cache = True
            batch = pd.DataFrame({"h": [h for h, _ in chunk], "model": EMBED_MODEL,
                                  "v": [list(map(float, v)) for v in vectors]})
            con.register("batch", batch)
            con.execute("INSERT OR REPLACE INTO emb_cache SELECT h, model, v FROM batch")
            con.unregister("batch")
            done = min(i + BATCH, len(todo))
            if done % (BATCH * 20) == 0 or done == len(todo):
                rate = done / (time.time() - start)
                eta = (len(todo) - done) / rate if rate else 0
                print(f"  {done:,}/{len(todo):,}  ({rate:.0f} texts/sec, about {eta / 60:.1f} min left)")
        out(f"Embedding took {(time.time() - start) / 60:.1f} minutes")
    else:
        out(f"All {len(texts):,} embeddings found in cache")

    # Read the vectors back in the same order as `texts`, straight into numpy
    need = pd.DataFrame({"idx": range(len(hashes)), "h": hashes})
    con.register("need", need)
    result = con.sql(f"""
        SELECT need.idx, emb_cache.v FROM need JOIN emb_cache USING (h)
        WHERE emb_cache.model = '{EMBED_MODEL}' ORDER BY need.idx
    """)
    for method in ("to_arrow_table", "fetch_arrow_table", "arrow"):   # name differs across DuckDB versions
        if hasattr(result, method):
            table = getattr(result, method)()
            break
    con.unregister("need")
    if hasattr(table, "read_all"):
        table = table.read_all()
    col = table.column("v").combine_chunks()
    matrix = col.values.to_numpy(zero_copy_only=False).reshape(len(col), -1).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
    return matrix


def ollama_embed(texts: list[str]) -> list[list[float]]:
    import ollama
    # nomic-embed-text expects a task prefix; long texts are cut to keep it fast
    prepared = ["classification: " + t[:500] for t in texts]
    return ollama.embed(model=EMBED_MODEL, input=prepared)["embeddings"]


# ---------------------------------------------------------------- data split
def build_split(con, n_train: int, n_test: int):
    lines = con.execute("""
        SELECT model_input, nigp_class FROM items_clean WHERE usable
    """).df()
    lines["key"] = lines["model_input"].str.lower().str.strip()

    # One row per unique text; label = the class it was most often bought under
    grouped = (lines.groupby(["key", "nigp_class"]).size().rename("n").reset_index()
                    .sort_values("n", ascending=False))
    label = grouped.drop_duplicates("key")[["key", "nigp_class"]]
    counts = lines.groupby("key").agg(text=("model_input", "first"), lines=("model_input", "size"))
    uniq = counts.join(label.set_index("key")).reset_index()

    out(f"Usable lines: {len(lines):,}  ->  unique texts: {len(uniq):,}")

    rng = np.random.default_rng(SEED)
    uniq = uniq.sample(frac=1, random_state=SEED).reset_index(drop=True)
    test = uniq.iloc[:n_test].copy()
    pool = uniq.iloc[n_test:]
    # Train: cap each class, then sample down to n_train
    pool = pool.groupby("nigp_class").head(MAX_PER_CLASS_TRAIN)
    train = pool.sample(n=min(n_train, len(pool)), random_state=SEED).copy()

    seen = set(train["nigp_class"])
    test["class_in_train"] = test["nigp_class"].isin(seen)
    out(f"Train texts: {len(train):,} covering {train['nigp_class'].nunique()} classes")
    out(f"Test texts:  {len(test):,} covering {test['nigp_class'].nunique()} classes "
        f"({(~test['class_in_train']).sum()} test texts have a class never seen in train)")
    return train.reset_index(drop=True), test.reset_index(drop=True)


# ---------------------------------------------------------------- k-NN
def knn_predict(train_vecs, train_labels, test_vecs):
    labels = np.asarray(train_labels)
    results = []
    for i in range(0, len(test_vecs), 1000):
        sims = test_vecs[i:i + 1000] @ train_vecs.T
        top = np.argpartition(-sims, K, axis=1)[:, :K]
        for row, idx in enumerate(top):
            votes: dict[str, float] = {}
            for j in idx:
                votes[labels[j]] = votes.get(labels[j], 0.0) + max(float(sims[row, j]), 0.0)
            ranked = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
            total = sum(votes.values()) or 1.0
            results.append({
                "pred": ranked[0][0],
                "confidence": ranked[0][1] / total,
                "top5": [c for c, _ in ranked[:5]],
                "nearest_sim": float(sims[row, idx].max()),
            })
    return pd.DataFrame(results)


# ---------------------------------------------------------------- main
def main(embed_fn=ollama_embed) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=60000)
    parser.add_argument("--test", type=int, default=8000)
    args = parser.parse_args()

    start = time.time()
    con = duckdb.connect(str(DB_PATH))
    out("STAGE 2a - EMBEDDINGS + NEAREST NEIGHBOURS")
    out("=" * 60)
    train, test = build_split(con, args.train, args.test)

    out()
    train_vecs = embed_texts(con, train["text"].tolist(), embed_fn)
    test_vecs = embed_texts(con, test["text"].tolist(), embed_fn)

    preds = knn_predict(train_vecs, train["nigp_class"], test_vecs)
    test = pd.concat([test, preds], axis=1)
    test["correct"] = test["pred"] == test["nigp_class"]
    test["in_top5"] = [t in c for t, c in zip(test["nigp_class"], test["top5"])]

    out()
    out("RESULTS ON HELD-OUT TEST TEXTS (labels hidden during prediction)")
    out("-" * 60)
    out(f"Top-1 accuracy (exact class):         {test['correct'].mean():.1%}")
    out(f"Top-5 recall (true class in top 5):   {test['in_top5'].mean():.1%}   <- ceiling for the LLM pass")
    w = test["lines"]
    out(f"Accuracy weighted by purchase lines:  {(test['correct'] * w).sum() / w.sum():.1%}")

    out()
    out("Accuracy by confidence (how much we can trust each prediction):")
    out(f"  {'confidence':<12}{'share of test':>15}{'accuracy':>12}")
    for lo, hi in [(0.9, 1.01), (0.7, 0.9), (0.5, 0.7), (0.0, 0.5)]:
        part = test[(test["confidence"] >= lo) & (test["confidence"] < hi)]
        if len(part):
            out(f"  {lo:.1f}-{min(hi, 1.0):.1f}{len(part) / len(test):>15.1%}{part['correct'].mean():>12.1%}")

    out()
    out("If low-confidence items go to the LLM:")
    for threshold in [0.5, 0.6, 0.7]:
        sure = test["confidence"] >= threshold
        out(f"  threshold {threshold:.1f}: {sure.mean():.1%} auto-categorized at "
            f"{test.loc[sure, 'correct'].mean():.1%} accuracy; {(~sure).sum():,} items "
            f"({(~sure).mean():.1%}) sent to the LLM")

    # Save everything the LLM pass and the app will need
    save = test.copy()
    save["top5"] = save["top5"].map(lambda c: ",".join(c))
    for name, frame in [("knn_test", save), ("knn_train", train)]:
        con.execute(f"DROP TABLE IF EXISTS {name}")
        con.register("f", frame)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM f")
        con.unregister("f")
    con.close()

    out()
    out(f"Total time: {(time.time() - start) / 60:.1f} minutes")
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")
    print(f"Report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
