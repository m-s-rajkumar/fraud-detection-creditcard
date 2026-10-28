"""Run every query in sql/fraud_analysis.sql and print the results."""
import re
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "fraud.db"
SQL_PATH = ROOT / "sql" / "fraud_analysis.sql"

text = SQL_PATH.read_text()
blocks = re.split(r"^-- name: (\S+)\s*$", text, flags=re.MULTILINE)
# blocks = ['', name1, sql1, name2, sql2, ...]

pd.set_option("display.width", 140)
with sqlite3.connect(DB_PATH) as conn:
    for name, sql in zip(blocks[1::2], blocks[2::2]):
        print(f"\n=== {name} ===")
        print(pd.read_sql_query(sql.strip().rstrip(";"), conn).to_string(index=False))
