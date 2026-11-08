"""Train the final model and save artifacts for the API.

Usage:  python train.py path/to/india_ports.csv
Creates: artifacts/model.pkl  artifacts/meta.json
Re-run this when new PortWatch data arrives (monthly), then redeploy.
"""
import sys, json, warnings
import numpy as np, pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
warnings.filterwarnings("ignore")

CSV = sys.argv[1] if len(sys.argv) > 1 else "india_ports.csv"
ORDER, SEASONAL = (0, 1, 2), (1, 1, 1, 12)     # chosen by AIC in the notebook
H = 12

raw = pd.read_csv(CSV)
raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize(None)
raw["v"] = raw["import_container"] + raw["export_container"]
daily = raw.groupby("date")["v"].sum()
monthly = daily.resample("MS").sum()
complete = daily.resample("MS").count() == monthly.index.days_in_month
y = monthly[complete].astype(float)
y.index.freq = "MS"
print("Months:", y.index[0].date(), "->", y.index[-1].date(), len(y))

def fit(series):
    return SARIMAX(series, order=ORDER, seasonal_order=SEASONAL,
                   enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)

# uncertainty from rolling-origin errors (same idea as the notebook)
errs = []
for k in range(4, 0, -1):
    cut = len(y) - k * H
    tr, va = y.iloc[:cut], y.iloc[cut:cut + H]
    fc = np.asarray(fit(tr).forecast(H))
    errs.append((va.values - fc) / va.values)
pooled = float(1.4826 * np.median(np.abs(np.array(errs))))

res = fit(y)
res.save("artifacts/model.pkl")
meta = {"last_date": y.index[-1].strftime("%Y-%m-%d"), "pooled_rel_error": pooled,
        "history": {d.strftime("%Y-%m"): float(v) for d, v in y.items()},
        "unit_note": "PortWatch estimate of container import + export volume, all Indian ports"}
json.dump(meta, open("artifacts/meta.json", "w"))
print("Saved artifacts. pooled relative error =", round(pooled, 4))
