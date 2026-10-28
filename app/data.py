"""
Every table the app displays, computed in one place.

Local mode: computed live from data/spend.duckdb.
Demo mode (e.g. Streamlit Community Cloud): read from the small files in demo_data/,
which app/export_demo.py writes using these same functions, so both modes show identical numbers.
"""
from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "demo_data"
CHATS_FILE = DEMO_DIR / "recorded_chats.json"


def _has(con, table: str) -> bool:
    return con.cursor().execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [table]).fetchall()[0][0] > 0


def compute_frames(con, tools) -> dict[str, pd.DataFrame]:
    q = lambda sql: con.cursor().execute(sql).df()   # noqa: E731
    f = {}
    f["totals"] = q("""
        SELECT count(*) AS lines, sum(line_total) AS spend, count(DISTINCT purchase_order) AS pos,
               (SELECT count(DISTINCT vendor_id) FROM vendor_map) AS vendors,
               (SELECT count(*) FROM vendor_map) AS vendor_records,
               (SELECT count(*) FROM items_clean) AS text_lines,
               (SELECT count(*) FILTER (WHERE usable) FROM items_clean) AS usable_lines
        FROM po""")
    if _has(con, "knn_test"):
        knn = q("SELECT avg(correct::INT) AS acc, avg(in_top5::INT) AS top5 FROM knn_test")
        f["totals"]["knn_acc"] = knn["acc"][0]
        f["totals"]["knn_top5"] = knn["top5"][0]
    f["spend_by_year"] = q("""
        SELECT CAST(year(award_date) AS VARCHAR) AS year, round(sum(line_total) / 1e6, 1) AS spend_millions
        FROM po GROUP BY 1 ORDER BY 1""")
    f["top_categories"] = tools.query("""
        SELECT coalesce(c.class_name, r.nigp_class) AS category,
               round(sum(line_total) / 1e6, 1) AS spend_millions
        FROM recent_po r LEFT JOIN class_names c USING (nigp_class)
        GROUP BY 1 ORDER BY 2 DESC LIMIT 10""")
    f["vendor_review"] = q("""
        SELECT name_a, state_a, name_b, state_b, similarity, one_contains_other,
               round(spend_a + spend_b) AS combined_spend
        FROM vendor_review ORDER BY combined_spend DESC LIMIT 100""") if _has(con, "vendor_review") \
        else pd.DataFrame()
    f["price_variance"] = q("SELECT * FROM savings_price_variance")
    f["fragmentation"] = q("SELECT * FROM savings_fragmentation")
    f["off_contract"] = q("SELECT * FROM savings_off_contract")
    return f


def save_demo(frames: dict[str, pd.DataFrame]) -> None:
    DEMO_DIR.mkdir(exist_ok=True)
    for name, df in frames.items():
        df.to_parquet(DEMO_DIR / f"{name}.parquet", index=False)


def load_demo() -> dict[str, pd.DataFrame]:
    return {p.stem: pd.read_parquet(p) for p in DEMO_DIR.glob("*.parquet")}


def load_chats() -> list[dict]:
    if CHATS_FILE.exists():
        return json.loads(CHATS_FILE.read_text(encoding="utf-8"))
    return []
