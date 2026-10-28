"""
Spend Intelligence Agent - Streamlit app.

Local (full app, live agent; Ollama must be running):
    streamlit run app/app.py
Online demo (Streamlit Community Cloud): runs automatically in DEMO MODE when the
database is not present, using the small files in demo_data/ and recorded agent
conversations. Force demo mode locally with:  set SIA_DEMO=1  (PowerShell: $env:SIA_DEMO=1)
"""
from datetime import datetime
from pathlib import Path
import os
import sys

import duckdb
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "app"))
from tools import SpendTools, DB_PATH  # noqa: E402
from agent import run_agent, draft_memo, check_numbers, MODEL, NUMBER_CHECK  # noqa: E402
from data import compute_frames, load_demo, load_chats, DEMO_DIR  # noqa: E402

DEMO = os.environ.get("SIA_DEMO") == "1" or not DB_PATH.exists()
CSV_PATH = (DEMO_DIR if DEMO else ROOT / "eval") / "top_savings_opportunities.csv"
BAR_COLOR = "#2a78d6"   # single-series charts use one hue
REPO_URL = "https://github.com/kira2000/spend-intelligence-agent"

st.set_page_config(page_title="Spend Intelligence Agent", page_icon="📊", layout="wide")


# ---------------------------------------------------------------- data access
@st.cache_resource
def get_con():
    return duckdb.connect(str(DB_PATH))


@st.cache_resource
def get_tools():
    return SpendTools(get_con())


@st.cache_data(ttl=60)
def get_frames() -> dict:
    return load_demo() if DEMO else compute_frames(get_con(), get_tools())


def money(x) -> str:
    x = float(x or 0)
    if abs(x) >= 1e9:
        return f"${x / 1e9:,.2f}B"
    if abs(x) >= 1e6:
        return f"${x / 1e6:,.1f}M"
    if abs(x) >= 1e3:
        return f"${x / 1e3:,.0f}K"
    return f"${x:,.0f}"


def show_check(text: str, steps: list) -> bool:
    """Show the number check. Returns True when every $ figure is backed by the data."""
    c = check_numbers(text, steps)
    if c["unverified"]:
        st.warning("Number check: these figures were NOT found in the data: " + ", ".join(c["unverified"]))
        return False
    if c["verified"]:
        st.caption(f"Number check: all {len(c['verified'])} dollar figures match the data.")
    return True


def show_turn(turn: dict) -> None:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.markdown(turn["answer"])
        show_check(turn["answer"], turn["steps"])
        with st.expander(f"How I got this ({len(turn['steps'])} steps, {float(turn['seconds']):.0f} s)"):
            for s in turn["steps"]:
                st.markdown(f"**{s['tool']}** `{s['args']}`  ·  {s['seconds']} s")
                st.code(str(s["result"])[:1500], language="json")


F = get_frames()
T = F["totals"].iloc[0]
pv_all = F["price_variance"]
pv = pv_all[~pv_all["is_capital"].astype(bool)]
savings_total = pv["potential_savings"].sum() + F["fragmentation"]["potential_savings"].sum() \
    + F["off_contract"]["potential_savings"].sum()

# ---------------------------------------------------------------- header
st.title("Spend Intelligence Agent")
st.caption("City of Austin purchase orders, Oct 2009 – Sep 2026 · cleaned, categorized and analyzed "
           "with a local LLM (Qwen 2.5 7B via Ollama) · all savings are potential and need validation")
if DEMO:
    st.info(f"**Online demo.** Dashboards show the real results. The agent runs locally on a laptop "
            f"(Ollama), so the *Ask the agent* tab replays real recorded conversations. "
            f"Code and setup: [{REPO_URL.split('//')[1]}]({REPO_URL})")

tab_overview, tab_quality, tab_savings, tab_agent = st.tabs(
    ["Overview", "Data quality", "Savings opportunities", "Ask the agent"])

# ---------------------------------------------------------------- overview
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total spend (2009–2026)", money(T["spend"]))
    c2.metric("Purchase lines", f"{int(T['lines']):,}")
    c3.metric("Unique vendors (after de-duplication)", f"{int(T['vendors']):,}")
    c4.metric("Potential savings (last 3 years)", money(savings_total))

    left, right = st.columns(2)
    with left:
        st.subheader("Spend by year ($M)")
        st.bar_chart(F["spend_by_year"], x="year", y="spend_millions", color=BAR_COLOR,
                     x_label="Year", y_label="Spend ($M)")
    with right:
        st.subheader("Top 10 categories, last 3 years ($M)")
        top_cats = F["top_categories"].copy()
        top_cats["category"] = top_cats["category"].str.slice(0, 40)
        st.bar_chart(top_cats, x="category", y="spend_millions", color=BAR_COLOR,
                     horizontal=True, sort="-spend_millions", x_label="Spend ($M)", y_label="")

# ---------------------------------------------------------------- data quality
with tab_quality:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vendor records → unique vendors", f"{int(T['vendor_records']):,} → {int(T['vendors']):,}",
              f"{int(T['vendor_records'] - T['vendors']):,} duplicates merged", delta_color="off")
    c2.metric("Lines with usable text", f"{T['usable_lines'] / T['text_lines']:.1%}")
    if "knn_acc" in T and pd.notna(T["knn_acc"]):
        c3.metric("Categorization accuracy (embeddings)", f"{T['knn_acc']:.1%}")
        c4.metric("True category in top 5", f"{T['knn_top5']:.1%}")

    st.subheader("Vendor pairs waiting for a human decision")
    st.caption("Borderline matches the cleaner would not merge on its own. Choose merge or keep_separate, then save.")
    review = F["vendor_review"].copy()
    if len(review):
        if not DEMO and get_con().cursor().execute(
                "SELECT count(*) FROM information_schema.tables WHERE table_name = 'vendor_review_decisions'"
        ).fetchall()[0][0]:
            saved = get_con().cursor().execute("SELECT name_a, name_b, decision FROM vendor_review_decisions").df()
            review = review.merge(saved, on=["name_a", "name_b"], how="left")
        else:
            review["decision"] = None
        edited = st.data_editor(
            review, hide_index=True, width="stretch", disabled=[c for c in review.columns if c != "decision"],
            column_config={"decision": st.column_config.SelectboxColumn("decision", options=["merge", "keep_separate"]),
                           "combined_spend": st.column_config.NumberColumn(format="$%d")})
        if st.button("Save decisions", disabled=DEMO, help="Saving is turned off in the online demo" if DEMO else None):
            decided = edited[edited["decision"].notna()][["name_a", "name_b", "decision"]]
            con = get_con().cursor()
            con.execute("CREATE TABLE IF NOT EXISTS vendor_review_decisions "
                        "(name_a VARCHAR, name_b VARCHAR, decision VARCHAR, decided_at TIMESTAMP)")
            con.execute("DELETE FROM vendor_review_decisions")
            con.register("d", decided.assign(decided_at=datetime.now()))
            con.execute("INSERT INTO vendor_review_decisions SELECT * FROM d")
            st.success(f"Saved {len(decided)} decisions.")

# ---------------------------------------------------------------- savings
with tab_savings:
    c1, c2, c3 = st.columns(3)
    c1.metric("Price variance (comparable items)", money(pv["potential_savings"].sum()),
              f"{pv['potential_savings'].sum() / max(pv['spend'].sum(), 1):.0%} of that spend", delta_color="off")
    c2.metric("Vendor consolidation (illustrative 5%)", money(F["fragmentation"]["potential_savings"].sum()))
    c3.metric("Contract compliance (illustrative 5%)", money(F["off_contract"]["potential_savings"].sum()))

    if CSV_PATH.exists():
        opp = pd.read_csv(CSV_PATH, dtype={"category_code": str})
        kinds = st.multiselect("Show", sorted(opp["type"].unique()), default=sorted(opp["type"].unique()))
        view = opp[opp["type"].isin(kinds)]
        st.dataframe(view, hide_index=True, width="stretch",
                     column_config={"spend": st.column_config.NumberColumn(format="$%d"),
                                    "potential_savings": st.column_config.NumberColumn(format="$%d")})
        st.download_button("Download as CSV", view.to_csv(index=False), "top_savings_opportunities.csv", "text/csv")
    else:
        st.info("Run `python pipeline/find_savings.py` to create the opportunities list.")

    with st.expander("Capital items (not counted as savings)"):
        st.caption("One code covers very different specs (e.g. a pickup and a specialty truck). "
                   "Standardize specifications, then compare bids.")
        cap = pv_all[pv_all["is_capital"].astype(bool)]
        st.dataframe(cap[["item", "purchases", "vendors", "min_price", "median_price", "max_price", "spend"]]
                     .sort_values("spend", ascending=False), hide_index=True, width="stretch")

# ---------------------------------------------------------------- agent
with tab_agent:
    if DEMO:
        chats = load_chats()
        st.caption(f"Recorded conversations with the agent ({MODEL}, running locally). "
                   "Every answer shows the tool calls it made and the automatic number check.")
        if not chats:
            st.warning("No recorded conversations yet. Run `python app/export_demo.py` locally.")
        else:
            pick = st.radio("Choose a question", [c["question"] for c in chats], horizontal=False)
            turn = next(c for c in chats if c["question"] == pick)
            show_turn(turn)
            if turn.get("memo"):
                st.divider()
                st.subheader("Savings memo drafted by the agent (needs human approval)")
                numbers_ok = show_check(turn["memo"], turn["steps"])
                st.text_area("Memo", turn["memo"], height=300, disabled=True)
                a, r, d = st.columns(3)
                a.button("Approve", type="primary", disabled=True, help="Approval is turned off in the online demo")
                r.button("Reject", disabled=True)
                d.download_button("Download memo (.md)", turn["memo"], "savings_memo.md", "text/markdown")
                if not numbers_ok:
                    st.error("Approve would be blocked: the memo contains figures not found in the data.")
            elif any(s["tool"] == NUMBER_CHECK for s in turn["steps"]):
                st.caption("No memo: this answer had no data behind it, so the app does not offer one.")
    else:
        st.caption(f"The agent ({MODEL}, running locally) answers by calling read-only tools on the database. "
                   "Expand 'How I got this' to see every step.")
        if "chat" not in st.session_state:
            st.session_state.chat = []

        examples = ["What are the top 5 savings opportunities?",
                    "Where can we save money on traffic signs?",
                    "Is there price variance for Motorola?",
                    "What will we spend next year?"]
        cols = st.columns(len(examples))
        clicked = None
        for col, ex in zip(cols, examples):
            if col.button(ex, width="stretch"):
                clicked = ex

        for turn in st.session_state.chat:
            show_turn(turn)

        question = st.chat_input("Ask about spend, vendors or savings...") or clicked
        if question:
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"), st.spinner("Thinking and querying the data..."):
                try:
                    result = run_agent(question, get_tools(), history=st.session_state.chat)
                    st.session_state.chat.append({"question": question, **result})
                    st.session_state.pop("memo", None)
                    st.rerun()
                except Exception as exc:
                    st.error(f"The agent could not run. Is Ollama running? ({exc})")

        if st.session_state.chat:
            last = st.session_state.chat[-1]
            st.divider()
            st.subheader("Savings memo (needs human approval)")
            if not any(s["tool"] != NUMBER_CHECK for s in last["steps"]):
                st.info("The last answer did not use any data, so there is nothing to put in a memo.")
            elif st.button("Draft a memo from the last answer"):
                with st.spinner("Drafting..."):
                    st.session_state.memo = draft_memo(last["question"], last)
            if "memo" in st.session_state:
                memo = st.text_area("Edit before approving", st.session_state.memo, height=320)
                numbers_ok = show_check(memo, last["steps"])
                if not numbers_ok:
                    st.error("Approve is blocked: edit or remove the flagged figures above.")
                a, r, d = st.columns(3)
                con = get_con().cursor()
                con.execute("""CREATE TABLE IF NOT EXISTS memos
                               (created_at TIMESTAMP, question VARCHAR, memo VARCHAR, status VARCHAR)""")
                if a.button("Approve", type="primary", disabled=not numbers_ok):
                    con.execute("INSERT INTO memos VALUES (?, ?, ?, 'approved')", [datetime.now(), last["question"], memo])
                    st.success("Approved and saved to the memos table.")
                if r.button("Reject"):
                    con.execute("INSERT INTO memos VALUES (?, ?, ?, 'rejected')", [datetime.now(), last["question"], memo])
                    st.warning("Rejected. Nothing will be sent.")
                d.download_button("Download memo (.md)", memo, "savings_memo.md", "text/markdown")
