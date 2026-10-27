# Spend Intelligence Agent

An AI agent that cleans, categorizes, and finds savings in messy procurement data, running free on a local LLM (Ollama + Qwen 2.5).

Built on real public data: **City of Austin purchase orders, Oct 2009 – Sep 2026 (321,625 line items, $4.07B in spend).**

> Work in progress. Results below are updated as each stage is finished.

## The problem
Large organizations have hundreds of thousands of purchase lines with inconsistent vendor names and free-text descriptions. That makes basic questions such as "how much do we spend on safety gear, and with how many suppliers?" hard to answer. Companies pay spend-analytics vendors to fix this.

## How it works
1. **Clean** (done): merge duplicate vendors, strip form junk from descriptions, flag lines with too little text
2. **Categorize** (done): predict the spend category (NIGP class) from free text using embeddings + nearest neighbours; a local LLM re-ranks only the least-confident items, using real past purchases as evidence
3. **Find savings** (planned): price variance on identical items, vendor fragmentation, off-contract spend
4. **Agent** (planned): answers questions in plain English and drafts savings memos, with human approval

## Results so far
| Stage | Result |
|---|---|
| Vendor cleanup | 6,283 vendor records → 5,957 unique vendors (326 duplicates merged); 609 borderline pairs queued for review |
| Description cleanup | 82.7% of lines have usable text, covering 218 of 219 categories |
| Categorization, embeddings only | **70.7%** exact class (216 classes) on 8,000 held-out texts; true class in top 5 for 89.5% |
| High-confidence half | 54% of items auto-categorized at **92.3%** accuracy |
| Hybrid (LLM for confidence < 0.5) | **73.9%** projected overall (+3.2 pts); LLM doubles accuracy on the least-confident items (18.6% → 37.1%) |
| LLM speed | 1.2 s per item on a laptop with Qwen 2.5 7B, $0 cost |
| Likely miscoded source data | On 29% of hard items, both methods agree on a class that differs from the official code (e.g. trailers coded as welding equipment) |
| Label-leakage check | The official commodity description almost always matches its code (92% of codes used 5+ times), so it is excluded from model input to keep accuracy honest |

## What I learned (iterations)
- **Leak-free evaluation:** identical texts are grouped so the same sentence can never be in both train and test.
- **LLM v1 (category names only) made results worse** (−9 pts on hard items, fixed 12 / broke 21). Category names alone were misleading because Austin files items in non-obvious classes.
- **LLM v2 (retrieval-augmented):** showing the model the 2 most similar past purchases per candidate plus the similarity vote turned it into a net gain (+4.3 pts, fixed 43 / broke 30).
- **The LLM only helps where the embeddings are unsure.** Below confidence 0.5 it clearly helps; between 0.5 and 0.7 it hurts, so the pipeline uses it only below 0.5.
- Caveat: the hybrid figure is measured on a 300-item sample (about 60 items per confidence band), so it carries a few points of uncertainty.

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
```

## Tech
Python · DuckDB · pandas · rapidfuzz · Ollama (Qwen 2.5 7B, nomic-embed-text) · Streamlit
