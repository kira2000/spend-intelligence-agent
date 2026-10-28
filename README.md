# Spend Intelligence Agent

An AI agent that cleans, categorizes, and finds savings in messy procurement data, running free on a local LLM (Ollama + Qwen 2.5).

Built on real public data: **City of Austin purchase orders, Oct 2009 – Sep 2026 (321,625 line items, $4.07B in spend).**

> Work in progress. Results below are updated as each stage is finished.

## The problem
Large organizations have hundreds of thousands of purchase lines with inconsistent vendor names and free-text descriptions. That makes basic questions such as "how much do we spend on safety gear, and with how many suppliers?" hard to answer. Companies pay spend-analytics vendors to fix this.

## How it works
1. **Clean** (done): merge duplicate vendors, strip form junk from descriptions, flag lines with too little text
2. **Categorize** (done): predict the spend category (NIGP class) from free text using embeddings + nearest neighbours; a local LLM re-ranks only the least-confident items, using real past purchases as evidence
3. **Find savings** (done): price variance on identical items, vendor fragmentation, off-contract spend, tail spend; ranked opportunities exported to CSV for procurement teams
4. **Agent + app** (done): a local LLM agent answers questions in plain English by calling read-only tools, drafts savings memos, and every $ figure is automatically checked against the database before a human approves it

## Results so far
| Stage | Result |
|---|---|
| Vendor cleanup | 6,283 vendor records → 5,955 unique vendors (328 duplicates merged); 609 borderline pairs queued for review |
| Description cleanup | 82.7% of lines have usable text, covering 218 of 219 categories |
| Categorization, embeddings only | **70.7%** exact class (216 classes) on 8,000 held-out texts; true class in top 5 for 89.5% |
| High-confidence half | 54% of items auto-categorized at **92.3%** accuracy |
| Hybrid (LLM for confidence < 0.5) | **73.9%** projected overall (+3.2 pts); LLM doubles accuracy on the least-confident items (18.6% → 37.1%) |
| LLM speed | 1.2 s per item on a laptop with Qwen 2.5 7B, $0 cost |
| Likely miscoded source data | On 29% of hard items, both methods agree on a class that differs from the official code (e.g. trailers coded as welding equipment) |
| Savings analysis (last 3 years, $1.05B spend) | **$23.0M potential** (2.2% of spend): $7.1M price variance on comparable items, $14.9M vendor consolidation (illustrative 5%), $0.9M contract compliance (illustrative 5%) |
| Tail spend | 93% of vendors share the last 20% of spend but generate 60% of purchase orders |
| Agent | Answers questions like "top 5 savings opportunities" in ~10 s on a laptop with 6 read-only tools; every $ figure is verified against the data |
| Label-leakage check | The official commodity description almost always matches its code (92% of codes used 5+ times), so it is excluded from model input to keep accuracy honest |

## What I learned (iterations)
- **Leak-free evaluation:** identical texts are grouped so the same sentence can never be in both train and test.
- **LLM v1 (category names only) made results worse** (−9 pts on hard items, fixed 12 / broke 21). Category names alone were misleading because Austin files items in non-obvious classes.
- **LLM v2 (retrieval-augmented):** showing the model the 2 most similar past purchases per candidate plus the similarity vote turned it into a net gain (+4.3 pts, fixed 43 / broke 30).
- **The LLM only helps where the embeddings are unsure.** Below confidence 0.5 it clearly helps; between 0.5 and 0.7 it hurts, so the pipeline uses it only below 0.5.
- **Sanity-checking savings:** the first price-variance estimate was $60M, but it was driven by vehicles where one code covers a $62K pickup and a $770K specialty truck. Separating capital items (median > $10K) and non-comparable units (monthly bills, lots, hours) gave a defensible **$7.1M**; capital items are reported separately with the advice to standardize specs before comparing bids.
- **LLMs are bad at arithmetic:** the first drafted memo stated a total of $10,811,794; the real sum was $10,891,694. Fix: totals are computed in code by the tool, the model is told never to calculate, and an automatic number check flags any $ figure not found in the tool results before the memo can be approved.
- **Safety by design:** the agent's SQL tool accepts only a single SELECT statement (writes and multi-statement queries are rejected), and memos need explicit human approval.
- Caveat: the hybrid figure is measured on a 300-item sample (about 60 items per confidence band), so it carries a few points of uncertainty.

## The app
`streamlit run app/app.py` opens four tabs:
- **Overview** - spend, vendors, potential savings, spend by year and by category
- **Data quality** - cleanup results, categorization accuracy, and a vendor review queue where a human decides merge / keep separate
- **Savings opportunities** - ranked opportunities with evidence and suggested actions, CSV download, capital items listed separately
- **Ask the agent** - chat with the agent; "How I got this" shows every tool call; draft a memo, edit it, Approve or Reject

## Run it
```
python -m venv .venv
.venv\Scripts\activate          # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```
Download "Purchase Order Quantity Price detail for Commodity/Goods procurements" (City of Austin, data.gov) as CSV to `data/austin_po.csv`, then:
```
python pipeline/load.py
python pipeline/explore.py
python pipeline/clean_vendors.py
python pipeline/clean_descriptions.py
python pipeline/categorize_embed.py      # ~20 min the first time (embeddings are cached)
python pipeline/categorize_llm.py        # 300-item LLM evaluation, ~6 min
python pipeline/find_savings.py          # seconds; writes eval/top_savings_opportunities.csv
streamlit run app/app.py                 # the app (stop it before re-running pipeline scripts)
```

## Tech
Python · DuckDB · pandas · rapidfuzz · Ollama (Qwen 2.5 7B, nomic-embed-text) · Streamlit
