# create_encoders.py

import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# 1. Load your CSV
df = pd.read_csv('AKTU_Counselling.csv')

# 2. (Optional) Clean column names exactly as you do in app2.py
df.columns = df.columns.str.replace('▲▼', '').str.strip()

# 3. List all categorical columns you need encoders for
categorical_cols = [
    'Institute',
    'Program',
    'Stream',
    'Quota',
    'Category',
    'Seat Gender',
    'Round'
]

# 4. For each column, fit a LabelEncoder and save it to disk
for col in categorical_cols:
    if col not in df.columns:
        print(f"⚠️  Column '{col}' not found in CSV. Skipping.")
        continue

    # Ensure every value is a string before encoding
    le = LabelEncoder()
    df[col] = df[col].astype(str).str.strip()
    le.fit(df[col])

    # Save the encoder
    file_name = f"{col}_encoder.pkl"
    joblib.dump(le, file_name)
    print(f"✅ Saved encoder for '{col}' → {file_name}")
