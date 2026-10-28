"""
Prepare the online demo (Streamlit Community Cloud cannot run Ollama or hold the 300 MB database).

1. Saves every table the app shows as small Parquet files in demo_data/.
2. Copies the top-opportunities CSV.
3. Records real agent conversations (answers, tool steps, memos) from the local
   Qwen model, so the online demo can replay them - clearly labeled as recorded.

Run from the project folder (stop the Streamlit app first; Ollama must be running):
    python app/export_demo.py
    python app/export_demo.py --no-record      # only refresh the data files
Then commit demo_data/ to GitHub.
"""
from pathlib import Path
import argparse
import json
import shutil
import sys

import duckdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "app"))
from tools import SpendTools, DB_PATH  # noqa: E402
from agent import run_agent, draft_memo, check_numbers  # noqa: E402
from data import compute_frames, save_demo, DEMO_DIR, CHATS_FILE  # noqa: E402

DEMO_QUESTIONS = [
    "What are the top 5 savings opportunities?",
    "Where can we save money on traffic signs?",
    "Is there price variance for Motorola?",
    "How has our fuel spending changed over the years?",
    "What will we spend next year?",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-record", action="store_true", help="skip recording agent conversations")
    args = parser.parse_args()

    con = duckdb.connect(str(DB_PATH), read_only=True)
    tools = SpendTools(con)

    frames = compute_frames(con, tools)
    save_demo(frames)
    csv = ROOT / "eval" / "top_savings_opportunities.csv"
    if csv.exists():
        shutil.copy(csv, DEMO_DIR / csv.name)
    size = sum(p.stat().st_size for p in DEMO_DIR.iterdir()) / 1e6
    print(f"Saved {len(frames)} tables to {DEMO_DIR} ({size:.1f} MB)")

    if args.no_record:
        return
    chats = []
    for q in DEMO_QUESTIONS:
        print(f"Recording: {q}")
        res = run_agent(q, tools)
        has_data = any(s["tool"] != "number_check" for s in res["steps"])
        memo = draft_memo(q, res) if has_data else ""
        check = check_numbers(memo, res["steps"]) if memo else {"verified": [], "unverified": []}
        print(f"   {len(res['steps'])} steps, {res['seconds']:.1f} s, memo figures verified: "
              f"{len(check['verified'])}, unverified: {len(check['unverified'])}")
        chats.append({"question": q, **res, "memo": memo})
    CHATS_FILE.write_text(json.dumps(chats, indent=1, default=str), encoding="utf-8")
    print(f"Recorded {len(chats)} conversations to {CHATS_FILE}")


if __name__ == "__main__":
    main()
