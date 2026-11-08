# India Port Container Volume Forecast API

Independent project using public **IMF PortWatch** data (estimates from vessel tracking).
Not affiliated with Maersk or the IMF. Please cite IMF PortWatch (portwatch.imf.org).

Forecasts monthly container import + export volume across Indian ports with a SARIMA model
(chosen by cross-validation against Holt-Winters, gradient boosting, an ensemble and a seasonal-naive baseline).

## Endpoints
| Path | What it returns |
|---|---|
| `/` | Dashboard (history, forecast, 95% interval) |
| `/forecast?months=12` | Forecast with intervals (1 to 12 months) |
| `/history` | Monthly history used for training |
| `/health` | Status and last data month |
| `/docs` | Interactive API docs (Swagger) |

## Run locally
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload        # open http://127.0.0.1:8000
```

## Retrain with new data
```bash
python train.py path/to/india_ports.csv   # rewrites artifacts/, then commit and redeploy
```

## Deploy on Render (free tier)
1. Push this folder to a new GitHub repository.
2. On render.com: **New +** -> **Web Service** -> connect the repo.
3. Choose **Docker** as the runtime and the **Free** instance type, then **Create Web Service**.
4. When the build finishes, Render shows a public URL like `https://your-name.onrender.com`.

Free instances go to sleep when idle, so the first request after a pause can take about a minute.

## Limits (say these in an interview)
- Volumes are PortWatch estimates, not customs counts.
- Hold-out error (MAPE) is about 6.5%, versus about 8.9% for the seasonal-naive baseline: a real but modest gain.
- The model uses past volume only. It does not know about future shocks (strikes, weather, Red Sea changes).
- The 95% interval comes from past forecast errors and widens with the horizon.
