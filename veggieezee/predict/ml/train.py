import os
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "ml/data.xlsx")

df = pd.read_excel(DATA_PATH)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date", "region", "vegetable", "price_npr"])
def norm(s): 
    return str(s).strip().lower()

def add_date_features(d):
    d = d.copy()
    d["year"] = d["date"].dt.year
    d["month"] = d["date"].dt.month
    d["m_sin"] = np.sin(2*np.pi*d["month"]/12)
    d["m_cos"] = np.cos(2*np.pi*d["month"]/12)
    return d

df["region"] = df["region"].astype(str).str.strip()
df["vegetable"] = df["vegetable"].astype(str).str.strip()
if "vegetable_nepali" not in df.columns:
    df["vegetable_nepali"] = df["vegetable"]
df["vegetable_nepali"] = df["vegetable_nepali"].astype(str).str.strip()

df = df.sort_values("date").reset_index(drop=True)

name_map = {}
for _, r in df[["vegetable","vegetable_nepali"]].drop_duplicates().iterrows():
    name_map[norm(r["vegetable"])] = r["vegetable"]
    name_map[norm(r["vegetable_nepali"])] = r["vegetable"]

data = add_date_features(df)

FEATURES = ["region", "vegetable", "year", "month", "m_sin", "m_cos"]
TARGET = "price_npr"

X = data[FEATURES]
y = data[TARGET]

cat_cols = ["region", "vegetable"]
num_cols = ["year", "month", "m_sin", "m_cos"]

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols),
])

rf = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=2
)

pipe = Pipeline([("pre", pre), ("rf", rf)])

#--Evaluation(MAE)--
tscv = TimeSeriesSplit(n_splits=5)
maes = []
for tr, te in tscv.split(X):
    pipe.fit(X.iloc[tr], y.iloc[tr])
    pred = pipe.predict(X.iloc[te])
    maes.append(mean_absolute_error(y.iloc[te], pred))

print("MAE avg (NPR):", round(float(np.mean(maes)), 2))
print("MAE splits:", [round(float(m),2) for m in maes])

pipe.fit(X, y)
joblib.dump(pipe, 'model.pkl') 
joblib.dump(name_map, 'name_map.pkl')
print("Model saved")

