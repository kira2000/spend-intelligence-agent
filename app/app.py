"""
Spend Intelligence Agent - Streamlit app.

Run from the project folder (Ollama must be running for the agent tab):
    streamlit run app/app.py
"""
from datetime import datetime
from pathlib import Path
import sys

import duckdb
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
from tools import SpendTools, DB_PATH  # noqa: E402
from agent import run_agent, draft_memo, check_numbers, MODEL  # noqa: E402

CSV_PATH = ROOT / "eval" / "top_savings_opportunities.csv"
BAR_COLOR = "#2a78d6"   # single-series charts use one hue

st.set_page_config(page_title="Spend Intelligence Agent", page_icon="📊", layout="wide")


@st.cache_resource
def get_con():
    return duckdb.connect(str(DB_PATH))


def q(sql: str, params=None) -> pd.DataFrame:
    return get_con().cursor().execute(sql, params or []).df()


def money(x) -> str:
    x = float(x or 0)
    if abs(x) >= 1e9:
        return f"${x / 1e9:,.2f}B"
    if abs(x) >= 1e6:
        return f"${x / 1e6:,.1f}M"
    if abs(x) >= 1e3:
        return f"${x / 1e3:,.0f}K"
    return f"${x:,.0f}"


def show_check(text: str, steps: list) -> None:
    c = check_numbers(text, steps)
    if c["unverified"]:
        st.warning("Number check: these figures were NOT found in the data - fix them before approving: "
                   + ", ".join(c["unverified"]))
    elif c["verified"]:
        st.caption(f"Number check: all {len(c['verified'])} dollar figures match the data.")


def has_table(name: str) -> bool:
    return bool(q("SELECT count(*) AS n FROM information_schema.tables WHERE table_name = ?", [name])["n"][0])


@st.cache_resource
def get_tools():
    return SpendTools(get_con())


# ---------------------------------------------------------------- header
st.title("Spend Intelligence Agent")
st.caption("City of Austin purchase orders, Oct 2009 – Sep 2026 · cleaned, categorized and analyzed "
           "with a local LLM (Qwen 2.5 7B via Ollama) · all savings are potential and need validation")

tab_overview, tab_quality, tab_savings, tab_agent = st.tabs(
    ["Overview", "Data quality", "Savings opportunities", "Ask the agent"])

# ---------------------------------------------------------------- overview
with tab_overview:
    totals = q("""SELECT count(*) AS lines, sum(line_total) AS spend,
                         count(DISTINCT purchase_order) AS pos FROM po""").iloc[0]
    vendors = q("SELECT count(DISTINCT vendor_id) AS n FROM vendor_map")["n"][0]
    savings = q("""SELECT
        (SELECT sum(potential_savings) FROM savings_price_variance WHERE NOT is_capital) +
        (SELECT sum(potential_savings) FROM savings_fragmentation) +
        (SELECT sum(potential_savings) FROM savings_off_contract) AS s""")["s"][0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total spend (2009–2026)", money(totals["spend"]))
    c2.metric("Purchase lines", f"{int(totals['lines']):,}")
    c3.metric("Unique vendors (after de-duplication)", f"{int(vendors):,}")
    c4.metric("Potential savings (last 3 years)", money(savings))

    left, right = st.columns(2)
    with left:
        st.subheader("Spend by year ($M)")
        by_year = q("""SELECT CAST(year(award_date) AS VARCHAR) AS year,
                              round(sum(line_total) / 1e6, 1) AS spend_millions
                       FROM po GROUP BY 1 ORDER BY 1""")
        st.bar_chart(by_year, x="year", y="spend_millions", color=BAR_COLOR,
                     x_label="Year", y_label="Spend ($M)")
    with right:
        st.subheader("Top 10 categories, last 3 years ($M)")
        top_cats = get_tools().query("""SELECT coalesce(c.class_name, r.nigp_class) AS category,
                               round(sum(line_total) / 1e6, 1) AS spend_millions
                        FROM recent_po r LEFT JOIN class_names c USING (nigp_class)
                        GROUP BY 1 ORDER BY 2 DESC LIMIT 10""")
        top_cats["category"] = top_cats["category"].str.slice(0, 40)
        st.bar_chart(top_cats, x="category", y="spend_millions", color=BAR_COLOR,
                     horizontal=True, sort="-spend_millions", x_label="Spend ($M)", y_label="")

# ---------------------------------------------------------------- data quality
with tab_quality:
    vm = q("""SELECT count(*) AS records, count(DISTINCT vendor_id) AS vendors,
                     count(*) FILTER (WHERE match_method <> 'self') AS merged FROM vendor_map""").iloc[0]
    txt = q("""SELECT count(*) AS lines, count(*) FILTER (WHERE usable) AS usable FROM items_clean""").iloc[0]
    knn = q("SELECT avg(correct::INT) AS acc, avg(in_top5::INT) AS top5 FROM knn_test").iloc[0] \
        if has_table("knn_test") else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vendor records → unique vendors", f"{int(vm['records']):,} → {int(vm['vendors']):,}",
              f"{int(vm['records'] - vm['vendors']):,} duplicates merged", delta_color="off")
    c2.metric("Lines with usable text", f"{txt['usable'] / txt['lines']:.1%}")
    if knn is not None:
        c3.metric("Categorization accuracy (embeddings)", f"{knn['acc']:.1%}")
        c4.metric("True category in top 5", f"{knn['top5']:.1%}")

    st.subheader("Vendor pairs waiting for a human decision")
    st.caption("Borderline matches the cleaner would not merge on its own. "
               "Choose merge or keep_separate, then save.")
    if has_table("vendor_review"):
        review = q("""SELECT name_a, state_a, name_b, state_b, similarity, one_contains_other,
                             round(spend_a + spend_b) AS combined_spend
                      FROM vendor_review ORDER BY combined_spend DESC LIMIT 100""")
        if has_table("vendor_review_decisions"):
            saved = q("SELECT name_a, name_b, decision FROM vendor_review_decisions")
            review = review.merge(saved, on=["name_a", "name_b"], how="left")
        else:
            review["decision"] = None
        edited = st.data_editor(
            review, hide_index=True, width="stretch", disabled=[c for c in review.columns if c != "decision"],
            column_config={"decision": st.column_config.SelectboxColumn(
                "decision", options=["merge", "keep_separate"]),
                "combined_spend": st.column_config.NumberColumn(format="$%d")})
        if st.button("Save decisions"):
            decided = edited[edited["decision"].notna()][["name_a", "name_b", "decision"]]
            con = get_con().cursor()
            con.execute("CREATE TABLE IF NOT EXISTS vendor_review_decisions "
                        "(name_a VARCHAR, name_b VARCHAR, decision VARCHAR, decided_at TIMESTAMP)")
            con.execute("DELETE FROM vendor_review_decisions")
            decided = decided.assign(decided_at=datetime.now())
            con.register("d", decided)
            con.execute("INSERT INTO vendor_review_decisions SELECT * FROM d")
            st.success(f"Saved {len(decided)} decisions.")

# ---------------------------------------------------------------- savings
with tab_savings:
    pv = q("SELECT sum(potential_savings) AS s, sum(spend) AS spend FROM savings_price_variance WHERE NOT is_capital").iloc[0]
    fr = q("SELECT sum(potential_savings) AS s FROM savings_fragmentation")["s"][0]
    oc = q("SELECT sum(potential_savings) AS s FROM savings_off_contract")["s"][0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Price variance (comparable items)", money(pv["s"]), f"{pv['s'] / pv['spend']:.0%} of that spend",
              delta_color="off")
    c2.metric("Vendor consolidation (illustrative 5%)", money(fr))
    c3.metric("Contract compliance (illustrative 5%)", money(oc))

    if CSV_PATH.exists():
        opp = pd.read_csv(CSV_PATH, dtype={"category_code": str})
        kinds = st.multiselect("Show", sorted(opp["type"].unique()), default=sorted(opp["type"].unique()))
        view = opp[opp["type"].isin(kinds)]
        st.dataframe(view, hide_index=True, width="stretch",
                     column_config={"spend": st.column_config.NumberColumn(format="$%d"),
                                    "potential_savings": st.column_config.NumberColumn(format="$%d")})
        st.download_button("Download as CSV", view.to_csv(index=False), "top_savings_opportunities.csv",
                           "text/csv")
    else:
        st.info("Run `python pipeline/find_savings.py` to create the opportunities list.")

    with st.expander("Capital items (not counted as savings)"):
        st.caption("One code covers very different specs (e.g. a pickup and a specialty truck). "
                   "Standardize specifications, then compare bids.")
        st.dataframe(q("""SELECT item, purchases, vendors, min_price, median_price, max_price, round(spend) AS spend
                          FROM savings_price_variance WHERE is_capital ORDER BY spend DESC"""),
                     hide_index=True, width="stretch")

# ---------------------------------------------------------------- agent
with tab_agent:
    st.caption(f"The agent ({MODEL}, running locally) answers by calling read-only tools on the database. "
               "Expand 'How I got this' to see every step.")
    if "chat" not in st.session_state:
        st.session_state.chat = []

    examples = ["What are the top 5 savings opportunities?",
                "Where can we save money on safety equipment?",
                "Which vendors do we use most for electrical supplies?",
                "How much did we spend off-contract on police equipment?"]
    cols = st.columns(len(examples))
    clicked = None
    for col, ex in zip(cols, examples):
        if col.button(ex, width="stretch"):
            clicked = ex

    for i, turn in enumerate(st.session_state.chat):
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.markdown(turn["answer"])
            show_check(turn["answer"], turn["steps"])
            with st.expander(f"How I got this ({len(turn['steps'])} tool calls, {turn['seconds']:.0f} s)"):
                for s in turn["steps"]:
                    st.markdown(f"**{s['tool']}** `{s['args']}`  ·  {s['seconds']} s")
                    st.code(s["result"][:1500], language="json")

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
        if st.button("Draft a memo from the last answer"):
            with st.spinner("Drafting..."):
                st.session_state.memo = draft_memo(last["question"], last)
        if "memo" in st.session_state:
            memo = st.text_area("Edit before approving", st.session_state.memo, height=320)
            show_check(memo, last["steps"])
            a, r, d = st.columns(3)
            con = get_con().cursor()
            con.execute("""CREATE TABLE IF NOT EXISTS memos
                           (created_at TIMESTAMP, question VARCHAR, memo VARCHAR, status VARCHAR)""")
            if a.button("Approve", type="primary"):
                con.execute("INSERT INTO memos VALUES (?, ?, ?, 'approved')", [datetime.now(), last["question"], memo])
                st.success("Approved and saved to the memos table.")
            if r.button("Reject"):
                con.execute("INSERT INTO memos VALUES (?, ?, ?, 'rejected')", [datetime.now(), last["question"], memo])
                st.warning("Rejected. Nothing will be sent.")
            d.download_button("Download memo (.md)", memo, "savings_memo.md", "text/markdown")
