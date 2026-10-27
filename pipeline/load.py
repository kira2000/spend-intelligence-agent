"""
Step 1 - Load the City of Austin purchase-order CSV into DuckDB.

Creates two tables in data/spend.duckdb:
  raw_po : the CSV exactly as downloaded (every column kept as text)
  po     : a typed, lightly cleaned copy that the rest of the pipeline uses

Run from the project folder:
    python pipeline/load.py
"""
from pathlib import Path
import time

import duckdb

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "austin_po.csv"
DB_PATH = ROOT / "data" / "spend.duckdb"


def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(f"CSV not found: {CSV_PATH}")

    start = time.time()
    con = duckdb.connect(str(DB_PATH))

    # 1) Raw copy. all_varchar keeps codes like 00512 from losing their leading zeros.
    con.execute("DROP TABLE IF EXISTS raw_po")
    con.execute(
        "CREATE TABLE raw_po AS SELECT * FROM read_csv(?, header = true, all_varchar = true)",
        [CSV_PATH.as_posix()],
    )

    # 2) Typed copy.
    #    NIGP codes come in 3, 5, 7 or 11 digits (class / class-item / group / detail).
    #    If a leading zero was ever dropped upstream, pad the code back to a valid length.
    con.execute("DROP TABLE IF EXISTS po")
    con.execute(
        """
        CREATE TABLE po AS
        WITH base AS (
            SELECT
                *,
                regexp_replace(trim(COMMODITY), '[^0-9]', '', 'g') AS code_digits
            FROM raw_po
        ),
        fixed AS (
            SELECT
                *,
                CASE
                    WHEN length(code_digits) IN (3, 5, 7, 11) THEN code_digits
                    WHEN length(code_digits) + 1 IN (3, 5, 7, 11) THEN lpad(code_digits, CAST(length(code_digits) + 1 AS INTEGER), '0')
                    WHEN length(code_digits) + 2 IN (3, 5, 7, 11) THEN lpad(code_digits, CAST(length(code_digits) + 2 AS INTEGER), '0')
                    ELSE nullif(code_digits, '')
                END AS commodity_code
            FROM base
        )
        SELECT
            row_number() OVER ()                                         AS line_id,
            commodity_code,
            left(commodity_code, 3)                                      AS nigp_class,
            CASE WHEN length(commodity_code) >= 5
                 THEN left(commodity_code, 5) END                        AS nigp_class_item,
            nullif(trim(COMMODITY_DESCRIPTION), '')                      AS commodity_desc,
            nullif(trim(EXTENDED_DESCRIPTION), '')                       AS extended_desc,
            try_cast(replace(QUANTITY, ',', '') AS DOUBLE)               AS quantity,
            nullif(upper(trim(UNIT_OF_MEASURE)), '')                     AS uom,
            nullif(trim(UNIT_OF_MEAS_DESC), '')                          AS uom_desc,
            try_cast(replace(replace(UNIT_PRICE, ',', ''), '$', '') AS DOUBLE)  AS unit_price,
            try_cast(replace(replace(ITM_TOT_AM, ',', ''), '$', '') AS DOUBLE)  AS line_total,
            nullif(trim(MASTER_AGREEMENT), '')                           AS master_agreement,
            nullif(trim(CONTRACT_NAME), '')                              AS contract_name,
            nullif(trim(PURCHASE_ORDER), '')                             AS purchase_order,
            try_strptime(trim(AWARD_DATE), '%m/%d/%Y')::DATE             AS award_date,
            nullif(trim(VENDOR_CODE), '')                                AS vendor_code,
            nullif(trim(LGL_NM), '')                                     AS vendor_name,
            nullif(trim(AD_LN_1), '')                                    AS vendor_address,
            nullif(trim(CITY), '')                                       AS vendor_city,
            nullif(trim(ST), '')                                         AS vendor_state,
            left(nullif(trim(ZIP), ''), 5)                               AS vendor_zip5,
            nullif(trim(CTRY), '')                                       AS vendor_country
        FROM fixed
        """
    )

    raw_rows = con.execute("SELECT count(*) FROM raw_po").fetchall()[0][0]
    po_rows = con.execute("SELECT count(*) FROM po").fetchall()[0][0]
    bad_totals = con.execute(
        "SELECT count(*) FROM raw_po WHERE ITM_TOT_AM IS NOT NULL "
        "AND try_cast(replace(replace(ITM_TOT_AM, ',', ''), '$', '') AS DOUBLE) IS NULL"
    ).fetchall()[0][0]
    bad_dates = con.execute(
        "SELECT count(*) FROM raw_po WHERE AWARD_DATE IS NOT NULL "
        "AND try_strptime(trim(AWARD_DATE), '%m/%d/%Y') IS NULL"
    ).fetchall()[0][0]
    con.close()

    print(f"Loaded {raw_rows:,} rows into raw_po and {po_rows:,} rows into po")
    print(f"Line totals that could not be read as numbers: {bad_totals:,}")
    print(f"Award dates that could not be read: {bad_dates:,}")
    print(f"Database: {DB_PATH}")
    print(f"Done in {time.time() - start:.1f} seconds")


if __name__ == "__main__":
    main()
