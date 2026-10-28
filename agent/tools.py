"""
Read-only tools the spend agent can call.

Every tool returns a short JSON string (max 50 rows) so a small local model can read it.
Nothing here can change the database: run_sql only accepts a single SELECT statement.
"""
from pathlib import Path

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
        like = f"%{text}%"
        return _json(self._df("""
            WITH hits AS (
                SELECT DISTINCT nigp_class FROM po
                WHERE commodity_desc ILIKE ? OR extended_desc ILIKE ?
            )
            SELECT r.nigp_class, c.class_name,
                   round(sum(r.line_total)) AS spend_3y,
                   count(DISTINCT r.vendor_id) AS vendors,
                   count(*) AS lines
            FROM recent_po r JOIN hits USING (nigp_class) LEFT JOIN class_names c USING (nigp_class)
            GROUP BY 1, 2 ORDER BY spend_3y DESC LIMIT 10
        """, [like, like]))

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
        return _json(self._df("""
            SELECT item, uom, nigp_class, purchases, vendors, min_price, p25_price, median_price,
                   max_price, round(spend) AS spend, round(potential_savings) AS potential_savings,
                   lowest_price_vendor, is_capital
            FROM savings_price_variance WHERE item ILIKE ?
            ORDER BY is_capital, potential_savings DESC LIMIT 10
        """, [f"%{item_text}%"]))

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
        """Find a vendor (including merged duplicate names) and its spend and main categories."""
        return _json(self._df("""
            WITH v AS (
                SELECT vendor_id, any_value(canonical_name) AS vendor,
                       string_agg(DISTINCT vendor_name, ' | ') AS names_merged
                FROM vendor_map
                WHERE vendor_id IN (SELECT vendor_id FROM vendor_map
                                    WHERE vendor_name ILIKE ? OR canonical_name ILIKE ?)
                GROUP BY 1
            )
            SELECT v.vendor, v.names_merged,
                   round(sum(p.line_total)) AS spend_all_years,
                   count(*) AS lines,
                   min(p.award_date) AS first_purchase, max(p.award_date) AS last_purchase,
                   string_agg(DISTINCT p.nigp_class, ', ') AS categories
            FROM v JOIN vendor_map m USING (vendor_id)
            JOIN po p ON p.vendor_code = m.vendor_code AND p.vendor_name = m.vendor_name
            GROUP BY v.vendor, v.names_merged ORDER BY spend_all_years DESC LIMIT 5
        """, [f"%{name}%", f"%{name}%"]))

    # ------------------------------------------------------------ registry
    def functions(self) -> dict:
        return {
            "run_sql": self.run_sql,
            "search_categories": self.search_categories,
            "category_details": self.category_details,
            "price_variance": self.price_variance,
            "top_savings": self.top_savings,
            "vendor_lookup": self.vendor_lookup,
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
    _spec("vendor_lookup", "Look up a vendor by name, including merged duplicate names.",
          {"name": {"type": "string"}}, ["name"]),
    _spec("run_sql", "Run one read-only DuckDB SELECT query when the other tools are not enough."
                     + SCHEMA_HELP, {"query": {"type": "string"}}, ["query"]),
]
