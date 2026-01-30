import sqlite3
import pandas as pd
import os

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DB_PATH = os.path.join(DATA_DIR, "solar_programs.db")
CSV_PATH = os.path.join(DATA_DIR, "solar_programs.csv")  # CSV with federal + state programs

# --- Ensure data folder exists ---
os.makedirs(DATA_DIR, exist_ok=True)

# --- Create/connect to DB ---
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# --- Create table ---
cursor.execute("""
CREATE TABLE IF NOT EXISTS Programs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    Type TEXT,
    State TEXT,
    Program TEXT,
    Subsidy TEXT,
    MaxSizeBonus TEXT,
    Link TEXT,
    ProgramType TEXT
)
""")
conn.commit()

# --- Read CSV ---
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

# --- Clean column names ---
df.columns = df.columns.str.strip()  # remove spaces

# --- Optional: clear existing rows (avoid duplicates) ---
cursor.execute("DELETE FROM Programs")
conn.commit()

# --- Insert into DB ---
records = []
for _, row in df.iterrows():
    records.append((
        row["Type"],
        row["State"],
        row["Program"],
        row["Subsidy"],
        row["Max Size/Bonus"],
        row["Link"],
        row["ProgramType"]  # <- important fix
    ))

cursor.executemany("""
    INSERT INTO Programs (Type, State, Program, Subsidy, MaxSizeBonus, Link, ProgramType)
    VALUES (?, ?, ?, ?, ?, ?, ?)
""", records)

conn.commit()
print(f"DB population completed with {len(df)} programs.")

conn.close()
