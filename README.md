# Spend Intelligence Agent

An AI agent that cleans, categorizes, and finds savings in messy procurement data, running free on a local LLM (Ollama + Qwen 2.5).

Built on real public data: **City of Austin purchase orders, Oct 2009 – Sep 2026 (321,625 line items, $4.07B in spend).**

> Work in progress. Results below are updated as each stage is finished.

## The problem
Large organizations have hundreds of thousands of purchase lines with inconsistent vendor names and free-text descriptions. That makes basic questions such as "how much do we spend on safety gear, and with how many suppliers?" hard to answer. Companies pay spend-analytics vendors to fix this.

## How it works
1. **Clean** (done): merge duplicate vendors, strip form junk from descriptions, flag lines with too little text
2. **Categorize** (in progress): predict the spend category (NIGP class) from free text using embeddings, with the local LLM handling low-confidence items
3. **Find savings** (planned): price variance on identical items, vendor fragmentation, off-contract spend
4. **Agent** (planned): answers questions in plain English and drafts savings memos, with human approval

## Results so far
| Stage | Result |
|---|---|
| Vendor cleanup | 6,283 vendor records → 5,957 unique vendors (326 duplicates merged); 609 borderline pairs queued for review |
| Description cleanup | 82.7% of lines have usable text, covering 218 of 219 categories |
| Label-leakage check | The official commodity description almost always matches its code (92% of codes used 5+ times), so it is excluded from model input to keep accuracy honest |

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
```

## Tech
Python · DuckDB · pandas · rapidfuzz · Ollama (Qwen 2.5 7B, nomic-embed-text) · Streamlit
