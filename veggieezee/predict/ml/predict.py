import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILE_PATH = os.path.join(BASE_DIR, "ml/data.xlsx")

def norm(s): 
    return str(s).strip().lower()

def add_date_features(d):
    d = d.copy()
    d["year"] = d["date"].dt.year
    d["month"] = d["date"].dt.month
    d["m_sin"] = np.sin(2*np.pi*d["month"]/12)
    d["m_cos"] = np.cos(2*np.pi*d["month"]/12)
    return d

pipe = joblib.load(os.path.join(BASE_DIR, "ml/model.pkl"))
name_map = joblib.load(os.path.join(BASE_DIR, "ml/name_map.pkl"))

df = pd.read_excel(FILE_PATH)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date", "region", "vegetable", "price_npr"])
df["region"] = df["region"].astype(str).str.strip()
df["vegetable"] = df["vegetable"].astype(str).str.strip()
df = df.sort_values("date").reset_index(drop=True)

def resolve_veg(user_text):
    key = norm(user_text)
    if key in name_map:
        return name_map[key]
    raise ValueError(f"Vegetable not found in dataset: {user_text}")

veg_in = input("Vegetable (English or Nepali/roman): ").strip()
region_in = input("Region: ").strip()
date_in = input("Future date (YYYY-MM-DD): ").strip()

veg = resolve_veg(veg_in)
region = region_in
fut = pd.to_datetime(date_in)

one = pd.DataFrame([{"date": fut, "region": region, "vegetable": veg}])
one = add_date_features(one)

FEATURES = ["region", "vegetable", "year", "month", "m_sin", "m_cos"]
pred = float(pipe.predict(one[FEATURES])[0])

hist = df[(df["region"] == region) & (df["vegetable"] == veg)].copy()
hist = hist.sort_values("date")

last4 = hist.tail(4)[["date", "price_npr"]].copy()
last8 = hist.tail(8)[["date", "price_npr"]].copy()

print(f"\nInput -> Veg: {veg_in} (resolved: {veg}), Region: {region}, Future date: {fut.date()}")
print(f"Predicted price (NPR): {pred:.2f}\n")

print("Past 4 records (latest):")
if len(last4) == 0:
    print("No history found for this veg+region in dataset.")
else:
    for _, r in last4.iterrows():
        print(f"{r['date'].date()}  ->  {float(r['price_npr']):.2f}")

plt.figure()
if len(last8) > 0:
    plt.plot(last8["date"], last8["price_npr"], marker="o")
plt.scatter([fut], [pred], marker="X")
plt.title(f"Price trend: {veg} in {region} (last 8 months + prediction)")
plt.xlabel("Date")
plt.ylabel("Price (NPR)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()