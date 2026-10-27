"""
Step 2 - Profile the loaded data so we know what needs cleaning.

Prints a data-quality report and saves it to eval/data_profile.txt.
Run after load.py:
    python pipeline/explore.py
"""
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "spend.duckdb"
REPORT_PATH = ROOT / "eval" / "data_profile.txt"

lines: list[str] = []


def out(text: str = "") -> None:
    print(text)
    lines.append(text)


def section(title: str) -> None:
    out()
    out("=" * 70)
    out(title)
    out("=" * 70)


def show(con: duckdb.DuckDBPyConnection, sql: str) -> None:
    out(con.sql(sql).df().to_string(index=False))


def scalar(con: duckdb.DuckDBPyConnection, sql: str):
    return con.execute(sql).fetchall()[0][0]


def main() -> None:
    con = duckdb.connect(str(DB_PATH), read_only=True)

    section("1. SIZE AND TIME RANGE")
    show(con, """
        SELECT count(*)                     AS line_items,
               count(DISTINCT purchase_order) AS purchase_orders,
               min(award_date)              AS first_award,
               max(award_date)              AS last_award,
               round(sum(line_total) / 1e6, 1) AS total_spend_millions
        FROM po
    """)
    out()
    out("Line items and spend by year:")
    show(con, """
        SELECT year(award_date) AS year,
               count(*) AS line_items,
               round(sum(line_total) / 1e6, 1) AS spend_millions
        FROM po GROUP BY 1 ORDER BY 1
    """)

    section("2. COMMODITY CODES (our answer key for categorization)")
    show(con, """
        SELECT length(commodity_code) AS code_length, count(*) AS line_items
        FROM po GROUP BY 1 ORDER BY 1
    """)
    out()
    out(f"Distinct NIGP classes (3-digit): {scalar(con, 'SELECT count(DISTINCT nigp_class) FROM po'):,}")
    out(f"Distinct class-items (5-digit):  {scalar(con, 'SELECT count(DISTINCT nigp_class_item) FROM po'):,}")
    out(f"Distinct full codes:            {scalar(con, 'SELECT count(DISTINCT commodity_code) FROM po'):,}")
    out()
    out("Top 15 classes by spend:")
    show(con, """
        SELECT nigp_class,
               any_value(commodity_desc) AS example_description,
               count(*) AS line_items,
               round(sum(line_total) / 1e6, 2) AS spend_millions
        FROM po GROUP BY 1 ORDER BY spend_millions DESC NULLS LAST LIMIT 15
    """)

    section("3. LABEL LEAKAGE CHECK")
    out("If every code always has the same description, the description is just the")
    out("code's official label, and using it to predict the code would be cheating.")
    show(con, """
        WITH per_code AS (
            SELECT commodity_code, count(DISTINCT commodity_desc) AS n_desc
            FROM po GROUP BY 1
        )
        SELECT CASE WHEN n_desc = 1 THEN 'exactly 1 description'
                    WHEN n_desc <= 5 THEN '2-5 descriptions'
                    ELSE 'more than 5' END AS descriptions_per_code,
               count(*) AS codes
        FROM per_code GROUP BY 1 ORDER BY 1
    """)

    section("4. DESCRIPTION QUALITY")
    total = scalar(con, "SELECT count(*) FROM po")
    for label, cond in [
        ("Missing commodity description", "commodity_desc IS NULL"),
        ("Missing extended description", "extended_desc IS NULL"),
        ("Extended description is form junk (RC LN / QTY DEL)", "extended_desc ILIKE '%RC LN%' OR extended_desc ILIKE '%QTY DEL%'"),
        ("Commodity description under 10 characters", "length(commodity_desc) < 10"),
    ]:
        n = scalar(con, f"SELECT count(*) FROM po WHERE {cond}")
        out(f"{label:<55} {n:>10,}  ({n / total:.1%})")

    section("5. QUANTITY, UNITS AND PRICES (needed for price comparison)")
    for label, cond in [
        ("Quantity is 0 or missing (lump-sum lines)", "coalesce(quantity, 0) = 0"),
        ("Unit price is 0 or missing", "coalesce(unit_price, 0) = 0"),
        ("Unit of measure missing", "uom IS NULL"),
        ("Usable for price comparison (qty>0, price>0, unit known)", "quantity > 0 AND unit_price > 0 AND uom IS NOT NULL"),
        ("Total does not match qty x price (off by > $1)", "quantity > 0 AND unit_price > 0 AND abs(quantity * unit_price - line_total) > 1"),
        ("Negative line total (credits / corrections)", "line_total < 0"),
    ]:
        n = scalar(con, f"SELECT count(*) FROM po WHERE {cond}")
        out(f"{label:<60} {n:>10,}  ({n / total:.1%})")
    out()
    out("Most common units of measure:")
    show(con, """
        SELECT uom, any_value(uom_desc) AS meaning, count(*) AS line_items
        FROM po GROUP BY 1 ORDER BY 3 DESC LIMIT 15
    """)

    section("6. VENDORS (what the de-duplication step will need to fix)")
    out(f"Distinct vendor codes: {scalar(con, 'SELECT count(DISTINCT vendor_code) FROM po'):,}")
    out(f"Distinct vendor names: {scalar(con, 'SELECT count(DISTINCT vendor_name) FROM po'):,}")
    out(f"Names used by more than one vendor code: "
        f"{scalar(con, 'SELECT count(*) FROM (SELECT vendor_name FROM po GROUP BY 1 HAVING count(DISTINCT vendor_code) > 1)'):,}")
    out()
    out("Examples - same name, several vendor codes:")
    show(con, """
        SELECT vendor_name, count(DISTINCT vendor_code) AS vendor_codes, count(*) AS line_items
        FROM po GROUP BY 1 HAVING count(DISTINCT vendor_code) > 1
        ORDER BY 2 DESC, 3 DESC LIMIT 10
    """)
    out()
    out("Top 15 vendors by spend:")
    show(con, """
        SELECT vendor_name, vendor_state,
               count(*) AS line_items,
               round(sum(line_total) / 1e6, 2) AS spend_millions
        FROM po GROUP BY 1, 2 ORDER BY spend_millions DESC NULLS LAST LIMIT 15
    """)

    section("7. CONTRACT COVERAGE")
    show(con, """
        SELECT CASE WHEN master_agreement IS NULL THEN 'No master agreement'
                    ELSE 'Under a master agreement' END AS bought,
               count(*) AS line_items,
               round(sum(line_total) / 1e6, 1) AS spend_millions
        FROM po GROUP BY 1
    """)

    con.close()
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    out()
    out(f"Report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
