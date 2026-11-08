import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

warnings.filterwarnings("ignore")
ART = Path(__file__).resolve().parent.parent / "artifacts"
model = SARIMAXResults.load(str(ART / "model.pkl"))
meta = json.loads((ART / "meta.json").read_text())

app = FastAPI(title="India Port Container Volume Forecast",
              description="Independent project using public IMF PortWatch data. "
                          "Not affiliated with Maersk or the IMF.", version="1.0")

def make_forecast(h: int):
    fc = np.asarray(model.forecast(h))
    idx = pd.date_range(pd.Timestamp(meta["last_date"]) + pd.offsets.MonthBegin(1), periods=h, freq="MS")
    steps = np.arange(1, h + 1)
    sd = meta["pooled_rel_error"] * np.sqrt(steps / steps.mean())
    return [{"month": d.strftime("%Y-%m"), "forecast": round(float(f)),
             "lower_95": round(float(f * np.exp(-1.96 * s))),
             "upper_95": round(float(f * np.exp(1.96 * s)))}
            for d, f, s in zip(idx, fc, sd)]

@app.get("/health")
def health():
    return {"status": "ok", "last_data_month": meta["last_date"][:7]}

@app.get("/forecast")
def forecast(months: int = Query(12, ge=1, le=12)):
    return {"model": "SARIMA(0,1,2)(1,1,1,12)", "note": meta["unit_note"],
            "forecast": make_forecast(months)}

@app.get("/history")
def history():
    return {"history": meta["history"]}

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>India Port Volume Forecast</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>body{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#222}
small{color:#666}table{border-collapse:collapse;width:100%;margin-top:1rem}
td,th{border-bottom:1px solid #ddd;padding:.4rem;text-align:right}th:first-child,td:first-child{text-align:left}</style></head>
<body><h1>India Port Container Volume Forecast</h1>
<small>Independent project using public IMF PortWatch data (estimates). Not affiliated with Maersk or the IMF.</small>
<canvas id="c" height="120"></canvas><table id="t"></table>
<script>
(async()=>{
 const [h,f]=await Promise.all([fetch('/history').then(r=>r.json()),fetch('/forecast').then(r=>r.json())]);
 const hk=Object.keys(h.history).slice(-48), fk=f.forecast.map(x=>x.month), labels=hk.concat(fk);
 const pad=n=>Array(n).fill(null);
 new Chart(document.getElementById('c'),{type:'line',data:{labels,datasets:[
  {label:'History',data:hk.map(k=>h.history[k]).concat(pad(fk.length)),borderColor:'#1f77b4',pointRadius:0},
  {label:'Forecast',data:pad(hk.length).concat(f.forecast.map(x=>x.forecast)),borderColor:'#d62728',pointRadius:2},
  {label:'Lower 95%',data:pad(hk.length).concat(f.forecast.map(x=>x.lower_95)),borderColor:'#f4a6a6',borderDash:[4,4],pointRadius:0},
  {label:'Upper 95%',data:pad(hk.length).concat(f.forecast.map(x=>x.upper_95)),borderColor:'#f4a6a6',borderDash:[4,4],pointRadius:0}]}});
 document.getElementById('t').innerHTML='<tr><th>Month</th><th>Forecast</th><th>Lower 95%</th><th>Upper 95%</th></tr>'+
  f.forecast.map(x=>`<tr><td>${x.month}</td><td>${x.forecast.toLocaleString()}</td><td>${x.lower_95.toLocaleString()}</td><td>${x.upper_95.toLocaleString()}</td></tr>`).join('');
})();
</script></body></html>"""

@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE
