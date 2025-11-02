"""test


Details about the model:
- Log-transform counts WITH seasonal components
- Transform data with log(x+1) before fitting SARIMAX
- optionnaly with exog

"""

import requests  # type: ignore[import-untyped]
import pandas as pd  # type: ignore[import-untyped]
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX  # type: ignore[import-untyped]

# --- Parameters ---
vuln_id = input("Vulnerability id: ")
per_page = 100
page = 1

sightings = []

# --- Fetch sightings with pagination ---
while True:
    url = f"https://vulnerability.circl.lu/api/sighting?page={page}&per_page={per_page}&vuln_id={vuln_id}"
    response = requests.get(url, headers={"accept": "application/json"})
    data = response.json()
    
    sightings.extend(data["data"])
    
    # Check if we fetched all pages
    total = data["metadata"]["count"]
    if page * per_page >= total:
        break
    page += 1

# --- Convert to DataFrame ---
df_sightings = pd.DataFrame(sightings)

# --- Parse timestamps ---
df_sightings["creation_ts"] = pd.to_datetime(
    df_sightings["creation_timestamp"], utc=True, errors="coerce"
)

# --- Aggregate daily counts ---
daily_counts = df_sightings.groupby(df_sightings["creation_ts"].dt.floor("D")).size().rename("sightings")
daily_series = daily_counts.asfreq("D", fill_value=0)
daily_series.index.name = "date"

print("Daily series:")
print(daily_series)

# --- Log-transform counts ---
daily_series_log = np.log1p(daily_series)  # log(x+1) to handle zeros

# --- Exogenous variable (constant for now) ---
# vlai_score = 0.8
# exog = pd.DataFrame({"vla_score": [vlai_score] * len(daily_series)}, index=daily_series.index)

# --- Fit SARIMAX on log-transformed data ---
model = SARIMAX(
    daily_series_log,
    # exog=exog,
    order=(1,1,1),
    seasonal_order=(1,1,1,7),  # weekly seasonality
    enforce_stationarity=False,
    enforce_invertibility=False
)
results = model.fit(disp=False)

# --- Prepare future exogenous variable ---
future_days = 7
future_dates = pd.date_range(start=daily_series.index[-1] + pd.Timedelta(days=1), periods=future_days)
# future_exog = pd.DataFrame({"vla_score": [vlai_score]*future_days}, index=future_dates)

# --- Forecast next 7 days (log scale) ---
# forecast_log = results.get_forecast(steps=future_days, exog=future_exog)
forecast_log = results.get_forecast(steps=future_days)
pred_mean_log = forecast_log.predicted_mean
conf_int_log = forecast_log.conf_int()

# --- Back-transform to original scale ---
pred_mean = np.expm1(pred_mean_log)
conf_int = np.expm1(conf_int_log)

# --- Plot observed + forecast ---
plt.figure(figsize=(10,4))
daily_series.plot(label="Observed")
pred_mean.plot(label="Forecast", color="orange")
plt.fill_between(conf_int.index, conf_int.iloc[:,0], conf_int.iloc[:,1], color="lightblue", alpha=0.3)
plt.title(f"SARIMAX Forecast of Sightings for {vuln_id} (log-transformed)")
plt.xlabel("Date")
plt.ylabel("Sightings")
plt.legend()
plt.grid(True)
plt.show()
