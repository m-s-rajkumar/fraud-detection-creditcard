"""Load the Kaggle CSV into a local SQLite database so we can analyse it with SQL."""
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "creditcard.csv"
DB_PATH = ROOT / "data" / "fraud.db"

df = pd.read_csv(CSV_PATH)
df.insert(0, "txn_id", range(len(df)))

with sqlite3.connect(DB_PATH) as conn:
    df.to_sql("transactions", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_class ON transactions (Class)")
    n = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]

print(f"Loaded {n:,} rows into {DB_PATH}")
