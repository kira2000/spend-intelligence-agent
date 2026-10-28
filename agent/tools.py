"""
Read-only tools the spend agent can call.

Every tool returns a short JSON string (max 50 rows) so a small local model can read it.
Nothing here can change the database: run_sql only accepts a single SELECT statement.
"""
from pathlib import Path
import re

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "spend.duckdb"
MAX_ROWS = 50
RECENT_YEARS = 3

SCHEMA_HELP = """
Tables you can query with run_sql (DuckDB SQL, SELECT only):
  po(line_id, commodity_code, nigp_class, commodity_desc, extended_desc, quantity, uom,
     unit_price, line_total, master_agreement, contract_name, purchase_order, award_date,
     vendor_code, vendor_name, vendor_state)            -- every purchase line, 2009-2026
  vendor_map(vendor_code, vendor_name, vendor_id, canonical_name, match_method, spend)
                                                          -- duplicate vendors merged into vendor_id
  savings_price_variance(commodity_code, uom, nigp_class, item, purchases, vendors, min_price,
     p25_price, median_price, max_price, spend, potential_savings, lowest_price_vendor, is_capital)
  savings_fragmentation(nigp_class, class_name, vendors, spend, top3_share, spend_outside_top3,
     potential_savings, top3_vendors)
  savings_off_contract(nigp_class, class_name, spend, off_contract_spend, off_contract_share,
     off_lines, off_vendors, potential_savings)
Savings tables cover the last 3 years. Money columns are US dollars.
"""


# Helper views, written as CTEs so they work on any connection (also read-only ones)
HELPER_CTES = f"""WITH recent_po AS (
    SELECT p.*, coalesce(v.vendor_id, -1) AS vendor_id,
           coalesce(v.canonical_name, p.vendor_name) AS vendor
    FROM po p LEFT JOIN vendor_map v USING (vendor_code, vendor_name)
    WHERE p.award_date >= (SELECT max(award_date) FROM po) - INTERVAL {RECENT_YEARS} YEAR
      AND p.line_total > 0
),
class_names AS (
    SELECT nigp_class, arg_max(commodity_desc, is_class_row * 1e9 + n) AS class_name
    FROM (SELECT nigp_class, commodity_desc, count(*) AS n,
                 max(CASE WHEN length(commodity_code) = 3 THEN 1 ELSE 0 END) AS is_class_row
          FROM po WHERE commodity_desc IS NOT NULL GROUP BY 1, 2)
    GROUP BY 1
)"""


# Search by words, not exact phrases: "Motorola Inc." must also find "MOTOROLA SOLUTIONS INC"
STOP_WORDS = {"inc", "llc", "llp", "ltd", "lp", "corp", "corporation", "co", "company", "incorporated",
              "the", "of", "and", "for", "a", "an", "plc"}


def _words(text: str) -> list[str]:
    words = []
    for w in re.findall(r"[a-z0-9]+", str(text).lower()):
        if w in STOP_WORDS or len(w) < 2:
            continue
        words.append(w[:-1] if len(w) > 4 and w.endswith("s") and not w.endswith("ss") else w)  # vests -> vest
    return words or [str(text).lower().strip()]


def _match_all_words(columns: list[str], text: str):
    """SQL condition: every word appears in at least one of the columns."""
    conds, params = [], []
    for w in _words(text):
        conds.append("(" + " OR ".join(f"{c} ILIKE ?" for c in columns) + ")")
        params += [f"%{w}%"] * len(columns)
    return " AND ".join(conds), params


def _json(df) -> str:
    if len(df) == 0:
        return "[]  (no rows matched)"
    return df.head(MAX_ROWS).to_json(orient="records", date_format="iso", double_precision=2)


class SpendTools:
    def __init__(self, con: duckdb.DuckDBPyConnection):
        self.con = con

    def query(self, sql: str, params=None):
        """Run SQL that may use the helper views recent_po and class_names."""
        sql = sql.strip()
        if sql.upper().startswith("WITH"):
            sql = HELPER_CTES + ",\n" + sql[4:]
        else:
            sql = HELPER_CTES + "\n" + sql
        return self.con.cursor().execute(sql, params or []).df()

    _df = query

    # ------------------------------------------------------------ tools
    def run_sql(self, query: str) -> str:
        """Run one read-only SELECT query (max 50 rows returned)."""
        cur = self.con.cursor()
        try:
            statements = cur.extract_statements(query)
            if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
                return "ERROR: only a single SELECT query is allowed."
            return _json(cur.sql(query).limit(MAX_ROWS).df())
        except Exception as exc:
            return f"ERROR: {str(exc)[:400]}  -- fix the query and try again."

    def search_categories(self, text: str) -> str:
        """Find spend categories whose name or items match the text (last 3 years)."""
        cond, params = _match_all_words(["commodity_desc", "extended_desc"], text)
        return _json(self._df(f"""
            WITH hits AS (
                SELECT DISTINCT nigp_class FROM po WHERE {cond}
            )
            SELECT r.nigp_class, c.class_name,
                   round(sum(r.line_total)) AS spend_3y,
                   count(DISTINCT r.vendor_id) AS vendors,
                   count(*) AS lines
            FROM recent_po r JOIN hits USING (nigp_class) LEFT JOIN class_names c USING (nigp_class)
            GROUP BY 1, 2 ORDER BY spend_3y DESC LIMIT 10
        """, params))

    def category_details(self, nigp_class: str) -> str:
        """Spend, top vendors, contract coverage and savings for one category (3-digit NIGP class)."""
        code = str(nigp_class).strip().zfill(3)[:3]
        summary = self._df("""
            SELECT ? AS nigp_class, any_value(c.class_name) AS class_name,
                   round(sum(line_total)) AS spend_3y,
                   count(DISTINCT vendor_id) AS vendors,
                   round(sum(line_total) FILTER (WHERE master_agreement IS NULL)) AS off_contract_spend
            FROM recent_po r LEFT JOIN class_names c USING (nigp_class) WHERE r.nigp_class = ?
        """, [code, code])
        vendors = self._df("""
            SELECT vendor, round(sum(line_total)) AS spend, count(*) AS lines
            FROM recent_po WHERE nigp_class = ? GROUP BY 1 ORDER BY spend DESC LIMIT 5
        """, [code])
        items = self._df("""
            SELECT item, uom, purchases, vendors, min_price, median_price, max_price,
                   round(potential_savings) AS potential_savings, lowest_price_vendor
            FROM savings_price_variance WHERE nigp_class = ? AND NOT is_capital
            ORDER BY potential_savings DESC LIMIT 5
        """, [code])
        frag = self._df("""SELECT vendors, top3_share, round(potential_savings) AS consolidation_savings
                           FROM savings_fragmentation WHERE nigp_class = ?""", [code])
        return (f'{{"summary": {_json(summary)}, "top_vendors": {_json(vendors)}, '
                f'"price_variance_items": {_json(items)}, "fragmentation": {_json(frag)}}}')

    def price_variance(self, item_text: str) -> str:
        """Price spread and potential savings for items whose name matches the text."""
        cond, params = _match_all_words(["item"], item_text)
        return _json(self._df(f"""
            SELECT item, uom, nigp_class, purchases, vendors, min_price, p25_price, median_price,
                   max_price, round(spend) AS spend, round(potential_savings) AS potential_savings,
                   lowest_price_vendor, is_capital
            FROM savings_price_variance WHERE {cond}
            ORDER BY is_capital, potential_savings DESC LIMIT 10
        """, params))

    def top_savings(self, kind: str = "all", limit: int = 10) -> str:
        """Biggest savings opportunities. kind = price_variance | fragmentation | off_contract | all."""
        limit = max(1, min(int(limit or 10), 25))
        parts = {
            "price_variance": """SELECT 'price_variance' AS type, item AS name, round(spend) AS spend,
                                 round(potential_savings) AS potential_savings
                                 FROM savings_price_variance WHERE NOT is_capital""",
            "fragmentation": """SELECT 'fragmentation' AS type, class_name AS name, round(spend) AS spend,
                                round(potential_savings) AS potential_savings FROM savings_fragmentation""",
            "off_contract": """SELECT 'off_contract' AS type, class_name AS name,
                               round(off_contract_spend) AS spend,
                               round(potential_savings) AS potential_savings FROM savings_off_contract""",
        }
        chosen = list(parts.values()) if kind not in parts else [parts[kind]]
        df = self._df(" UNION ALL ".join(chosen) + f" ORDER BY potential_savings DESC LIMIT {limit}")
        total = round(float(df["potential_savings"].sum())) if len(df) else 0
        # The total is computed here, in code, so the model never has to add numbers itself
        return f'{{"opportunities": {_json(df)}, "total_potential_savings_of_these": {total}}}'

    def vendor_lookup(self, name: str) -> str:
        """Find vendors by name (word match, merged duplicates included), their spend, and the
        price variance on items they supplied in the last 3 years."""
        cond, params = _match_all_words(["vendor_name", "canonical_name"], name)
        vendors = self._df(f"""
            WITH v AS (
                SELECT vendor_id, any_value(canonical_name) AS vendor,
                       string_agg(DISTINCT vendor_name, ' | ') AS names_merged
                FROM vendor_map
                WHERE vendor_id IN (SELECT vendor_id FROM vendor_map WHERE {cond})
                GROUP BY 1
            )
            SELECT v.vendor_id, v.vendor, v.names_merged,
                   round(sum(p.line_total)) AS spend_all_years,
                   count(*) AS lines,
                   min(p.award_date) AS first_purchase, max(p.award_date) AS last_purchase,
                   string_agg(DISTINCT p.nigp_class, ', ') AS categories
            FROM v JOIN vendor_map m USING (vendor_id)
            JOIN po p ON p.vendor_code = m.vendor_code AND p.vendor_name = m.vendor_name
            GROUP BY v.vendor_id, v.vendor, v.names_merged ORDER BY spend_all_years DESC LIMIT 5
        """, params)
        if not len(vendors):
            return '{"vendors": [], "note": "no vendor matched these words"}'
        ids = ",".join(str(int(i)) for i in vendors["vendor_id"])
        above = self._df(f"""
            SELECT s.item, l.uom, any_value(l.vendor) AS vendor, count(*) AS purchases,
                   any_value(s.vendors) AS vendors_selling_this_item,
                   round(min(l.unit_price), 2) AS vendor_min_price,
                   round(median(l.unit_price), 2) AS vendor_median_price,
                   round(max(l.unit_price), 2) AS vendor_max_price,
                   any_value(s.median_price) AS all_vendors_median_price,
                   any_value(s.p25_price) AS p25_price,
                   round(sum(greatest(l.unit_price - s.p25_price, 0) * l.quantity)) AS potential_savings
            FROM recent_po l JOIN savings_price_variance s USING (commodity_code, uom)
            WHERE l.vendor_id IN ({ids}) AND NOT s.is_capital
              AND l.quantity > 0 AND l.unit_price > 0 AND l.unit_price <= 5 * s.median_price
            GROUP BY s.item, l.uom
            HAVING sum(greatest(l.unit_price - s.p25_price, 0) * l.quantity) > 0
            ORDER BY potential_savings DESC LIMIT 10
        """)
        total = round(float(above["potential_savings"].sum())) if len(above) else 0
        return (f'{{"vendors": {_json(vendors.drop(columns="vendor_id"))}, '
                f'"price_variance_items_last_3y": {_json(above)}, '
                f'"note": "potential_savings = cost of this vendor\'s purchases above the item\'s '
                f'25th-percentile price; the spread can be within this vendor\'s own purchases", '
                f'"total_potential_savings_of_these": {total}}}')

    def spend_by_year(self, category_text: str = "") -> str:
        """Actual spend per year (all years), overall or for categories matching the words.
        Historical only - the data has no forecasts."""
        if category_text and category_text.strip():
            cond, params = _match_all_words(["commodity_desc", "extended_desc"], category_text)
            where = f"WHERE nigp_class IN (SELECT DISTINCT nigp_class FROM po WHERE {cond})"
        else:
            where, params = "", []
        df = self._df(f"""
            SELECT year(award_date) AS year, round(sum(line_total)) AS spend,
                   count(*) AS lines, count(DISTINCT purchase_order) AS purchase_orders
            FROM po {where} GROUP BY 1 ORDER BY 1
        """, params)
        return (f'{{"spend_by_year": {_json(df)}, "note": "calendar years by award date; '
                f'the last year may be partial (data ends {self._df("SELECT max(award_date) AS d FROM po")["d"][0]})"}}')

    # ------------------------------------------------------------ registry
    def functions(self) -> dict:
        return {
            "run_sql": self.run_sql,
            "search_categories": self.search_categories,
            "category_details": self.category_details,
            "price_variance": self.price_variance,
            "top_savings": self.top_savings,
            "vendor_lookup": self.vendor_lookup,
            "spend_by_year": self.spend_by_year,
        }


def _spec(name, description, props, required):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": props, "required": required}}}


TOOL_SPECS = [
    _spec("top_savings", "List the biggest potential savings opportunities (last 3 years).",
          {"kind": {"type": "string", "enum": ["all", "price_variance", "fragmentation", "off_contract"]},
           "limit": {"type": "integer", "description": "1-25, default 10"}}, []),
    _spec("search_categories", "Find spend categories (NIGP classes) related to a word, e.g. 'safety', 'fuel'.",
          {"text": {"type": "string"}}, ["text"]),
    _spec("category_details", "Spend, top vendors, contract coverage and savings for one 3-digit NIGP class.",
          {"nigp_class": {"type": "string", "description": "3-digit code, e.g. '345'"}}, ["nigp_class"]),
    _spec("price_variance", "Price spread and potential savings for items whose name contains the text.",
          {"item_text": {"type": "string"}}, ["item_text"]),
    _spec("vendor_lookup", "Look up vendors by name (partial words are fine, e.g. 'motorola'), with spend "
                           "and price variance on items the vendor supplied (last 3 years).",
          {"name": {"type": "string"}}, ["name"]),
    _spec("spend_by_year", "Actual spend per year (history, all years), overall or for a category word "
                           "such as 'fuel'. Use for trend or 'next year' questions; there are no forecasts.",
          {"category_text": {"type": "string", "description": "optional, e.g. 'fuel'; empty = all spend"}}, []),
    _spec("run_sql", "Run one read-only DuckDB SELECT query when the other tools are not enough."
                     + SCHEMA_HELP, {"query": {"type": "string"}}, ["query"]),
]
