"""
Stage 3 - Find savings opportunities.

Looks at the most recent N years of purchases (default 3, so prices are comparable)
and finds four kinds of opportunity:

  1. PRICE VARIANCE   - the same item (same commodity code + unit of measure) bought
                        at different unit prices. Potential savings = what would have
                        been saved if every purchase had been made at the item's
                        25th-percentile price. Obvious unit errors (price > 5x or
                        < 0.2x the item's median) are excluded and counted separately.
  2. FRAGMENTATION    - categories where spend is split across many vendors.
                        Illustrative savings: 5% of the spend outside the top 3 vendors
                        (typical result of consolidating volume into fewer contracts).
  3. OFF-CONTRACT     - spend with no master agreement in categories where most spend
                        IS under contract (so a contract probably already exists).
  4. TAIL SPEND       - many small purchases spread over rarely used vendors.

Every figure is a POTENTIAL saving for a procurement team to validate,
not a guaranteed amount.

Outputs:
  tables  savings_price_variance, savings_fragmentation, savings_off_contract
  eval/savings_report.txt
  eval/top_savings_opportunities.csv   (open in Excel)

Run after clean_vendors.py:
    python pipeline/find_savings.py            # last 3 years
    python pipeline/find_savings.py --years 5
"""
from pathlib import Path
import argparse

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "spend.duckdb"
REPORT_PATH = ROOT / "eval" / "savings_report.txt"
CSV_PATH = ROOT / "eval" / "top_savings_opportunities.csv"

CONSOLIDATION_RATE = 0.05     # illustrative saving on spend outside the top-3 vendors
MIN_PURCHASES = 5             # an item needs at least this many priced purchases
CAPITAL_THRESHOLD = 10_000    # items with a higher median unit price are capital purchases
                              # (vehicles, switchgear...) whose price differences are mostly
                              # specification differences, so they are reported separately
NON_COMPARABLE_UNITS = ("MO", "MON", "LOT", "DOL", "JOB", "LS", "YR", "HR", "HRS", "DAY", "WK")

report: list[str] = []


def out(text: str = "") -> None:
    print(text)
    report.append(text)


def money(x: float) -> str:
    if x >= 1e6:
        return f"${x / 1e6:,.1f}M"
    if x >= 1e3:
        return f"${x / 1e3:,.0f}K"
    return f"${x:,.0f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=int, default=3)
    args = parser.parse_args()

    con = duckdb.connect(str(DB_PATH))

    # Recent purchases, with cleaned vendor identity and a readable category name
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW recent AS
        SELECT p.*,
               coalesce(v.vendor_id, -1)                 AS vendor_id,
               coalesce(v.canonical_name, p.vendor_name) AS vendor
        FROM po p
        LEFT JOIN vendor_map v USING (vendor_code, vendor_name)
        WHERE p.award_date >= (SELECT max(award_date) FROM po) - INTERVAL {args.years} YEAR
          AND p.line_total > 0
    """)
    con.execute("""
        CREATE OR REPLACE TEMP VIEW class_names AS
        WITH names AS (
            SELECT nigp_class, commodity_desc, count(*) AS n,
                   max(CASE WHEN length(commodity_code) = 3 THEN 1 ELSE 0 END) AS is_class_row
            FROM po WHERE commodity_desc IS NOT NULL GROUP BY 1, 2
        )
        SELECT nigp_class, arg_max(commodity_desc, is_class_row * 1e9 + n) AS class_name
        FROM names GROUP BY 1
    """)

    first, last, lines, spend = con.execute(
        "SELECT min(award_date), max(award_date), count(*), sum(line_total) FROM recent"
    ).fetchall()[0]
    out("STAGE 3 - SAVINGS OPPORTUNITIES")
    out("=" * 70)
    out(f"Window: {first} to {last}  ({args.years} years)")
    out(f"Purchase lines: {lines:,}   Spend: {money(spend)}")
    out("All savings are POTENTIAL and need validation by a procurement team.")

    # ------------------------------------------------------------ 1. price variance
    con.execute(f"""
        CREATE OR REPLACE TABLE savings_price_variance AS
        WITH priced AS (
            SELECT * FROM recent
            WHERE quantity > 0 AND unit_price > 0 AND uom IS NOT NULL
              AND uom NOT IN {NON_COMPARABLE_UNITS}             -- monthly bills, lots, hours...
              AND length(commodity_code) >= 7          -- item-level codes only
        ),
        stats AS (
            SELECT commodity_code, uom,
                   median(unit_price)                AS median_price,
                   quantile_cont(unit_price, 0.25)   AS p25_price
            FROM priced GROUP BY 1, 2
        ),
        flagged AS (
            SELECT p.*, s.median_price, s.p25_price,
                   (p.unit_price > 5 * s.median_price OR p.unit_price < 0.2 * s.median_price) AS unit_error
            FROM priced p JOIN stats s USING (commodity_code, uom)
        ),
        vendor_prices AS (
            SELECT commodity_code, uom, vendor, median(unit_price) AS vendor_median, count(*) AS n
            FROM flagged WHERE NOT unit_error GROUP BY 1, 2, 3
        ),
        best AS (
            SELECT commodity_code, uom,
                   arg_min(vendor, vendor_median) FILTER (WHERE n >= 2) AS lowest_price_vendor
            FROM vendor_prices GROUP BY 1, 2
        )
        SELECT f.commodity_code, f.uom,
               any_value(f.nigp_class)                                   AS nigp_class,
               mode(f.commodity_desc)                                    AS item,
               count(*) FILTER (WHERE NOT f.unit_error)                  AS purchases,
               count(*) FILTER (WHERE f.unit_error)                      AS unit_errors,
               count(DISTINCT f.vendor_id)                               AS vendors,
               round(min(f.unit_price) FILTER (WHERE NOT f.unit_error), 2) AS min_price,
               round(any_value(f.p25_price), 2)                          AS p25_price,
               round(any_value(f.median_price), 2)                       AS median_price,
               round(max(f.unit_price) FILTER (WHERE NOT f.unit_error), 2) AS max_price,
               round(sum(f.line_total) FILTER (WHERE NOT f.unit_error), 2) AS spend,
               round(sum(greatest(f.unit_price - f.p25_price, 0) * f.quantity)
                     FILTER (WHERE NOT f.unit_error), 2)                 AS potential_savings,
               any_value(b.lowest_price_vendor)                          AS lowest_price_vendor
        FROM flagged f LEFT JOIN best b USING (commodity_code, uom)
        GROUP BY f.commodity_code, f.uom
        HAVING count(*) FILTER (WHERE NOT f.unit_error) >= {MIN_PURCHASES}
        ORDER BY potential_savings DESC
    """)
    con.execute(f"""ALTER TABLE savings_price_variance ADD COLUMN is_capital BOOLEAN;
                    UPDATE savings_price_variance SET is_capital = median_price > {CAPITAL_THRESHOLD}""")
    allpv = con.execute("SELECT * FROM savings_price_variance").df()
    cap = allpv[allpv["is_capital"]]
    pv = allpv[~allpv["is_capital"]].reset_index(drop=True)
    unit_errs = int(allpv["unit_errors"].sum())
    out()
    out("1. PRICE VARIANCE (same item + unit, different prices)")
    out("-" * 70)
    out(f"Comparable items with {MIN_PURCHASES}+ priced purchases: {len(pv):,}   "
        f"bought from 2+ vendors: {(pv['vendors'] >= 2).sum():,}")
    out(f"Spend on these items: {money(pv['spend'].sum())}")
    out(f"Potential savings at 25th-percentile price: {money(pv['potential_savings'].sum())} "
        f"({pv['potential_savings'].sum() / max(pv['spend'].sum(), 1):.1%} of that spend)")
    out(f"Lines excluded as probable unit-of-measure errors: {unit_errs:,}")
    out(f"Units excluded as not comparable: {', '.join(NON_COMPARABLE_UNITS)}")
    out("Note: part of the variance is price change over time (inflation) inside the window.")
    out()
    out(f"  {'item':<40}{'unit':>5}{'buys':>6}{'vend':>5}{'low':>9}{'median':>9}{'high':>9}{'savings':>10}")
    for _, r in pv.head(12).iterrows():
        out(f"  {str(r['item'])[:39]:<40}{r['uom']:>5}{r['purchases']:>6}{r['vendors']:>5}"
            f"{money(r['min_price']):>9}{money(r['median_price']):>9}{money(r['max_price']):>9}"
            f"{money(r['potential_savings']):>10}")
    out()
    out(f"Capital items (median unit price over {money(CAPITAL_THRESHOLD)}): {len(cap):,} items, "
        f"{money(cap['spend'].sum())} spend - NOT counted as savings.")
    out("Their price spread mostly reflects different specifications (e.g. a pickup vs a")
    out("specialty truck under one code). Action: standardize specs, then compare bids.")
    for _, r in cap.sort_values("spend", ascending=False).head(5).iterrows():
        out(f"  {str(r['item'])[:40]:<41}{r['purchases']:>4} buys  {money(r['min_price'])}-{money(r['max_price'])}"
            f"  spend {money(r['spend'])}")

    # ------------------------------------------------------------ 2. fragmentation
    con.execute(f"""
        CREATE OR REPLACE TABLE savings_fragmentation AS
        WITH by_vendor AS (
            SELECT nigp_class, vendor_id, any_value(vendor) AS vendor, sum(line_total) AS spend
            FROM recent GROUP BY 1, 2
        ),
        ranked AS (
            SELECT *, row_number() OVER (PARTITION BY nigp_class ORDER BY spend DESC) AS r,
                   sum(spend) OVER (PARTITION BY nigp_class) AS class_spend
            FROM by_vendor
        )
        SELECT r.nigp_class, c.class_name,
               count(*)                                                   AS vendors,
               round(any_value(class_spend), 2)                           AS spend,
               round(sum(spend) FILTER (WHERE r <= 3) / any_value(class_spend), 3) AS top3_share,
               round(sum(spend) FILTER (WHERE r > 3), 2)                  AS spend_outside_top3,
               count(*) FILTER (WHERE spend < 0.01 * class_spend)         AS vendors_under_1pct,
               round({CONSOLIDATION_RATE} * coalesce(sum(spend) FILTER (WHERE r > 3), 0), 2) AS potential_savings,
               string_agg(vendor, '; ' ORDER BY r) FILTER (WHERE r <= 3)   AS top3_vendors
        FROM ranked r LEFT JOIN class_names c USING (nigp_class)
        GROUP BY r.nigp_class, c.class_name
        HAVING count(*) >= 10
        ORDER BY potential_savings DESC
    """)
    fr = con.execute("SELECT * FROM savings_fragmentation").df()
    out()
    out("2. VENDOR FRAGMENTATION (spend split across many vendors)")
    out("-" * 70)
    out(f"Categories with 10+ vendors: {len(fr):,}   spend outside their top-3 vendors: "
        f"{money(fr['spend_outside_top3'].sum())}")
    out(f"Illustrative consolidation savings ({CONSOLIDATION_RATE:.0%} of that): {money(fr['potential_savings'].sum())}")
    out()
    out(f"  {'category':<44}{'vendors':>8}{'spend':>10}{'top-3 share':>12}{'savings':>10}")
    for _, r in fr.head(10).iterrows():
        out(f"  {str(r['class_name'])[:43]:<44}{r['vendors']:>8}{money(r['spend']):>10}"
            f"{r['top3_share']:>12.0%}{money(r['potential_savings']):>10}")

    # ------------------------------------------------------------ 3. off-contract
    con.execute(f"""
        CREATE OR REPLACE TABLE savings_off_contract AS
        WITH c AS (
            SELECT nigp_class,
                   sum(line_total)                                          AS spend,
                   sum(line_total) FILTER (WHERE master_agreement IS NULL)  AS off_spend,
                   count(*) FILTER (WHERE master_agreement IS NULL)         AS off_lines,
                   count(DISTINCT vendor_id) FILTER (WHERE master_agreement IS NULL) AS off_vendors,
                   count(DISTINCT master_agreement)                         AS contracts
            FROM recent GROUP BY 1
        )
        SELECT c.nigp_class, n.class_name, c.contracts,
               round(c.spend, 2) AS spend,
               round(c.off_spend, 2) AS off_contract_spend,
               round(c.off_spend / c.spend, 3) AS off_contract_share,
               c.off_lines, c.off_vendors,
               round({CONSOLIDATION_RATE} * c.off_spend, 2) AS potential_savings
        FROM c LEFT JOIN class_names n USING (nigp_class)
        WHERE c.contracts > 0 AND c.off_spend > 0
          AND c.off_spend / c.spend BETWEEN 0.05 AND 0.5      -- contract exists and is mostly used
        ORDER BY c.off_spend DESC
    """)
    oc = con.execute("SELECT * FROM savings_off_contract").df()
    total_off = con.execute("SELECT sum(line_total) FROM recent WHERE master_agreement IS NULL").fetchall()[0][0] or 0
    out()
    out("3. OFF-CONTRACT SPEND (no master agreement, in categories that have contracts)")
    out("-" * 70)
    out(f"All spend without a master agreement in the window: {money(total_off)} ({total_off / spend:.1%})")
    out(f"...of which in categories where 50%+ of spend IS on contract: {money(oc['off_contract_spend'].sum())} "
        f"across {len(oc)} categories")
    out(f"Illustrative savings if moved onto contract ({CONSOLIDATION_RATE:.0%}): {money(oc['potential_savings'].sum())}")
    out()
    out(f"  {'category':<44}{'off-contract':>13}{'share':>8}{'lines':>7}{'vendors':>8}")
    for _, r in oc.head(10).iterrows():
        out(f"  {str(r['class_name'])[:43]:<44}{money(r['off_contract_spend']):>13}"
            f"{r['off_contract_share']:>8.0%}{r['off_lines']:>7}{r['off_vendors']:>8}")

    # ------------------------------------------------------------ 4. tail spend
    tail = con.execute("""
        WITH v AS (SELECT vendor_id, sum(line_total) AS spend, count(DISTINCT purchase_order) AS pos
                   FROM recent GROUP BY 1),
        r AS (SELECT *, sum(spend) OVER (ORDER BY spend DESC ROWS UNBOUNDED PRECEDING)
                        / sum(spend) OVER () AS cum FROM v)
        SELECT count(*) AS vendors,
               count(*) FILTER (WHERE cum > 0.8) AS tail_vendors,
               sum(spend) FILTER (WHERE cum > 0.8) AS tail_spend,
               sum(pos) FILTER (WHERE cum > 0.8) AS tail_pos,
               sum(pos) AS pos
        FROM r
    """).fetchall()[0]
    vendors, tail_vendors, tail_spend, tail_pos, pos = tail
    out()
    out("4. TAIL SPEND (vendors outside the top 80% of spend)")
    out("-" * 70)
    out(f"{tail_vendors:,} of {vendors:,} vendors ({tail_vendors / vendors:.0%}) share the last 20% of spend "
        f"({money(tail_spend or 0)}) but generate {tail_pos:,} of {pos:,} purchase orders "
        f"({tail_pos / pos:.0%}). Candidates for p-cards, catalogs or consolidation to cut admin cost.")

    # ------------------------------------------------------------ summary + CSV
    total = pv["potential_savings"].sum() + fr["potential_savings"].sum() + oc["potential_savings"].sum()
    out()
    out("SUMMARY OF POTENTIAL SAVINGS")
    out("-" * 70)
    out(f"  Price variance (25th-percentile price):  {money(pv['potential_savings'].sum())}")
    out(f"  Vendor consolidation (illustrative 5%):  {money(fr['potential_savings'].sum())}")
    out(f"  Contract compliance (illustrative 5%):   {money(oc['potential_savings'].sum())}")
    out(f"  Total (categories can overlap):          {money(total)}  over {args.years} years")

    rows = []
    for _, r in pv.head(25).iterrows():
        rows.append({
            "type": "Price variance", "category_code": r["nigp_class"], "item_or_category": r["item"],
            "unit": r["uom"], "purchases": r["purchases"], "vendors": r["vendors"], "spend": r["spend"],
            "potential_savings": r["potential_savings"],
            "evidence": f"unit price ranged {r['min_price']:,.2f}-{r['max_price']:,.2f} (median {r['median_price']:,.2f})",
            "suggested_action": (f"Negotiate toward {r['p25_price']:,.2f}/{r['uom']}"
                                 + (f"; lowest-price vendor: {r['lowest_price_vendor']}"
                                    if pd.notna(r['lowest_price_vendor']) else "")),
        })
    for _, r in fr.head(15).iterrows():
        rows.append({
            "type": "Vendor fragmentation", "category_code": r["nigp_class"], "item_or_category": r["class_name"],
            "unit": "", "purchases": None, "vendors": r["vendors"], "spend": r["spend"],
            "potential_savings": r["potential_savings"],
            "evidence": f"{r['vendors']} vendors; top 3 hold {r['top3_share']:.0%} ({r['top3_vendors']})",
            "suggested_action": "Consolidate volume into 1-3 contracts",
        })
    for _, r in oc.head(15).iterrows():
        rows.append({
            "type": "Off-contract spend", "category_code": r["nigp_class"], "item_or_category": r["class_name"],
            "unit": "", "purchases": r["off_lines"], "vendors": r["off_vendors"], "spend": r["off_contract_spend"],
            "potential_savings": r["potential_savings"],
            "evidence": f"{r['off_contract_share']:.0%} of category spend bought without a master agreement",
            "suggested_action": "Route these purchases to the existing contract",
        })
    csv = pd.DataFrame(rows).sort_values("potential_savings", ascending=False)
    csv.insert(0, "rank", range(1, len(csv) + 1))
    csv["purchases"] = csv["purchases"].astype("Int64")
    CSV_PATH.parent.mkdir(exist_ok=True)
    csv.to_csv(CSV_PATH, index=False)

    con.close()
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")
    print(f"\nReport saved to {REPORT_PATH}")
    print(f"Top opportunities saved to {CSV_PATH} ({len(csv)} rows)")


if __name__ == "__main__":
    main()
