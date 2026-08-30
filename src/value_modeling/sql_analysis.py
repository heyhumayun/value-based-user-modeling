from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


def run_sql(data_dir: str, query_file: str, output_path: str) -> None:
    con = duckdb.connect()
    data = Path(data_dir)
    con.execute(f"CREATE VIEW users AS SELECT * FROM read_csv_auto('{data / 'users.csv'}')")
    con.execute(f"CREATE VIEW content AS SELECT * FROM read_csv_auto('{data / 'content.csv'}')")
    con.execute(f"CREATE VIEW interactions AS SELECT * FROM read_csv_auto('{data / 'interactions.csv'}')")
    query = Path(query_file).read_text()
    result = con.execute(query).fetchdf()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DuckDB SQL analysis for OTT value data.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--query-file", default="sql/ott_value_analysis.sql")
    parser.add_argument("--output-path", default="results/sql_summary.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sql(args.data_dir, args.query_file, args.output_path)
