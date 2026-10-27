"""
Stage 1b - Clean item descriptions so the AI sees useful text.

Why: COMMODITY_DESCRIPTION is basically the official name of the NIGP code
(label leakage), so the AI must categorize from the buyer-written text:
EXTENDED_DESCRIPTION + CONTRACT_NAME. Those fields contain blank-form junk
("RC LN___ QTY DEL___ ..."), quote numbers and dollar amounts, which we remove.

Creates table items_clean in data/spend.duckdb and prints before/after examples.
Run after load.py:
    python pipeline/clean_descriptions.py
"""
from pathlib import Path
import re
import time

import duckdb
import pandas as pd
from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "spend.duckdb"

# Blank receiving-form text, e.g. "RC LN_____ QTY DEL_____ P/F_____ B/O___DEL DATE_____"
FORM_JUNK = re.compile(
    r"\b(RC\s*LN|QTY\s*DEL|P\s*/\s*F|B\s*/\s*O|DEL(IVERY)?\s*DATE)\s*_*", re.IGNORECASE
)
UNDERSCORES = re.compile(r"_{2,}")
MONEY = re.compile(r"\$\s?[\d,]+(\.\d+)?")
PRICE_PHRASE = re.compile(r"@\s*[\d,.]+\s*(ea|each)?", re.IGNORECASE)
REFERENCE = re.compile(
    r"\b(ref\.?\s*)?(quote|qt|po|pr|req|requisition|ticket|invoice|inv)\s*(#|no\.?|number)?\s*:?\s*[A-Z0-9-]*\d[A-Z0-9-]*",
    re.IGNORECASE,
)
LONG_NUMBER = re.compile(r"\b[A-Z]{0,4}\d{6,}\b", re.IGNORECASE)  # part / quote / PR numbers
SEE_ATTACHED = re.compile(r"\b(see|per)\s+attached\b.*$", re.IGNORECASE)
SPACES = re.compile(r"\s+")
# Delivery / admin phrases that say nothing about WHAT was bought
GENERIC_PHRASES = [
    re.compile(r"\bmaterials?\s+(needed\s+)?(at|for)\b", re.IGNORECASE),
    re.compile(r"\b[a-z]{0,3}bsc\b", re.IGNORECASE),                  # Austin Water service centers (WBSC, AWWBSC...)
    re.compile(r"\bglen\s+bell(\s+warehouse)?\b", re.IGNORECASE),
    re.compile(r"\bshall\s+be\s+no\s+later\s+than\b", re.IGNORECASE),
    re.compile(r"\bsend\s+invoices?\s+to\b:?", re.IGNORECASE),
    re.compile(r"\bnon-?contract\b", re.IGNORECASE),
    re.compile(r"\S+@\S+"),                                           # email addresses
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),                       # dates
    re.compile(r"\(?\b\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),            # phone numbers
    re.compile(r"\b(quote|qt)\s+dated\b", re.IGNORECASE),
]
STRAY_SYMBOLS = re.compile(r"(?<=\s)[@#;`~]+(?=\s|$)|^[@#;`~]+\s|\s[A-Z]{1,3}-(?=\s|$)")
# Words that do not help identify the category
FILLER_WORDS = {
    "material", "materials", "needed", "at", "for", "the", "and", "of", "to", "in", "a",
    "msc", "cm", "warehouse", "various", "sections", "items", "item", "catalogue", "catalog",
    "deliver", "delivery", "deliveries", "order", "orders", "purchase", "contract", "per",
    "quantity", "each", "be", "on", "by", "from", "with", "as", "is", "no", "yard",
}
WITH_WITHOUT = [(re.compile(r"(?<![A-Za-z])w/o(?![A-Za-z])", re.IGNORECASE), " without "),
                (re.compile(r"(?<![A-Za-z])w/", re.IGNORECASE), " with "),
                (re.compile(r"(?<![A-Za-z])s/s(?![A-Za-z])", re.IGNORECASE), " stainless steel ")]

# Common purchasing abbreviations -> words (matched as whole words, any case)
ABBREVIATIONS = {
    "ss": "stainless steel", "galv": "galvanized",
    "zn": "zinc", "ci": "cast iron", "mech jt": "mechanical joint",
    "mj": "mechanical joint", "alum": "aluminum", "assy": "assembly", "accy": "accessory",
    "lg": "large", "xl": "extra large", "sm": "small",
    "blk": "black", "wht": "white", "yel": "yellow", "grn": "green",
    "elec": "electrical", "equip": "equipment", "maint": "maintenance", "repl": "replacement",
    "hvy": "heavy", "dty": "duty", "hd": "heavy duty", "fltr": "filter", "vlv": "valve",
    "pnl": "panel", "xfmr": "transformer", "kva": "kVA",
    "pkg": "package", "pk": "pack", "bx": "box", "gal": "gallon",
    "ft": "foot", "lb": "pound", "pr": "pair", "ea": "each", "qty": "quantity",
    "misc": "miscellaneous", "svc": "service", "svcs": "services", "lic": "license",
    "sw": "software", "hw": "hardware", "dia": "diameter", "thk": "thick",
}
ABBREV_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(" + "|".join(re.escape(k) for k in sorted(ABBREVIATIONS, key=len, reverse=True)) + r")(?![A-Za-z0-9])",
    re.IGNORECASE,
)


def clean_text(text: str | None) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    t = str(text)
    t = UNDERSCORES.sub(" ", t)   # first, so "B/O___DEL DATE" splits into words
    t = FORM_JUNK.sub(" ", t)
    t = SEE_ATTACHED.sub(" ", t)
    t = MONEY.sub(" ", t)
    t = PRICE_PHRASE.sub(" ", t)
    t = REFERENCE.sub(" ", t)
    t = LONG_NUMBER.sub(" ", t)
    for pattern in GENERIC_PHRASES:
        t = pattern.sub(" ", t)
    for pattern, word in WITH_WITHOUT:
        t = pattern.sub(word, t)
    t = ABBREV_PATTERN.sub(lambda m: ABBREVIATIONS[m.group(1).lower()], t)
    t = SPACES.sub(" ", t)
    t = STRAY_SYMBOLS.sub(" ", t)
    t = SPACES.sub(" ", t).strip(" -,.;:#|@`")
    return t


def combine(row) -> str:
    """Description + contract name, without repeating the same text twice."""
    desc, contract = row["clean_desc"], row["clean_contract"]
    if contract and desc and (contract.lower() in desc.lower() or desc.lower() in contract.lower()):
        return max(desc, contract, key=len)
    return " | ".join(p for p in [desc, contract] if p)


def informative_words(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z]{3,}", text.lower())
    return [w for w in words if w not in FILLER_WORDS]


def text_status(row) -> str:
    """ok / too_little_text / copies_label (would leak the answer)."""
    if len(informative_words(row["model_input"])) < 2:
        return "too_little_text"
    label = str(row["commodity_desc"] or "").lower()[:60]
    if label and fuzz.ratio(row["clean_desc"].lower()[:60], label) >= 90:
        return "copies_label"
    return "ok"


def main() -> None:
    start = time.time()
    con = duckdb.connect(str(DB_PATH))
    df = con.execute(
        "SELECT line_id, nigp_class, commodity_code, commodity_desc, extended_desc, contract_name FROM po"
    ).df()

    df["clean_desc"] = df["extended_desc"].map(clean_text)
    df["clean_contract"] = df["contract_name"].map(clean_text)

    # Text the AI will see. Commodity description is deliberately NOT included.
    df["model_input"] = df.apply(combine, axis=1)
    df["text_status"] = df.apply(text_status, axis=1)
    df["usable"] = df["text_status"].eq("ok")

    con.execute("DROP TABLE IF EXISTS items_clean")
    con.register("df_view", df[["line_id", "nigp_class", "commodity_code", "clean_desc",
                                "clean_contract", "model_input", "text_status", "usable"]])
    con.execute("CREATE TABLE items_clean AS SELECT * FROM df_view")
    con.unregister("df_view")

    total = len(df)
    usable = int(df["usable"].sum())
    had_text = int(df["extended_desc"].notna().sum())
    print(f"Lines: {total:,}")
    print(f"Lines with any extended description: {had_text:,} ({had_text / total:.1%})")
    for status, n in df["text_status"].value_counts().items():
        print(f"  {status:<18} {n:>9,}  ({n / total:.1%})")
    print(f"Lines with usable text for the AI:   {usable:,} ({usable / total:.1%})")
    print(f"NIGP classes covered by usable lines: {df.loc[df['usable'], 'nigp_class'].nunique()} "
          f"of {df['nigp_class'].nunique()}")

    print("\nBefore -> after examples:")
    sample = df[df["extended_desc"].notna()].sample(min(12, had_text), random_state=7)
    for _, r in sample.iterrows():
        print(f"  BEFORE: {str(r['extended_desc'])[:110]}")
        print(f"  AFTER : {r['model_input'][:110]}")
        print()

    con.close()
    print(f"Saved table items_clean. Done in {time.time() - start:.1f} seconds")


if __name__ == "__main__":
    main()
