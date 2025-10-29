
"""
Poisson-based Forecast of Vulnerability Sightings (Adaptive Daily/Weekly)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
import statsmodels.formula.api as smf

# --- Parameters ---
vuln_id = input("Vulnerability id: ")
per_page = 100
page = 1
forecast_horizon = 4  # number of periods to forecast (days or weeks)

# --- Fetch sightings ---
sightings = []
while True:
    url = f"https://vulnerability.circl.lu/api/sighting?page={page}&per_page={per_page}&vuln_id={vuln_id}"
    response = requests.get(url, headers={"accept": "application/json"})
    response.raise_for_status()
    data = response.json()
    sightings.extend(data["data"])
    total = data["metadata"]["count"]
    if page * per_page >= total:
        break
    page += 1

if not sightings:
    raise ValueError("No sightings found for this vulnerability.")

# --- Convert to DataFrame ---
df = pd.DataFrame(sightings)
df["creation_ts"] = pd.to_datetime(df["creation_timestamp"], utc=True, errors="coerce")

# --- Aggregate daily counts ---
daily_counts = df.groupby(df["creation_ts"].dt.floor("D")).size().rename("sightings")
daily_series = daily_counts.asfreq("D", fill_value=0)
daily_series.index.name = "date"

# --- Decide aggregation level ---
zero_ratio = (daily_series == 0).mean()
if zero_ratio > 0.5:
    # Sparse daily data → use weekly aggregation
    series = daily_series.resample("W-MON").sum()
    agg_level = "week"
    freq_str = "W-MON"
else:
    series = daily_series
    agg_level = "day"
    freq_str = "D"

# --- Prepare data for Poisson regression ---
df_series = series.reset_index()
df_series['time_index'] = np.arange(len(df_series))

# --- Fit Poisson regression ---
model = smf.glm("sightings ~ time_index", data=df_series, family=sm.families.Poisson())
results = model.fit()
print(results.summary())

# --- Forecast future periods ---
future_index = np.arange(len(df_series), len(df_series) + forecast_horizon)
df_future = pd.DataFrame({"time_index": future_index})
forecast_mean = results.get_prediction(df_future).predicted_mean
forecast_ci = results.get_prediction(df_future).conf_int()

# --- Prepare forecast dates ---
last_date = df_series["date"].iloc[-1]
forecast_dates = pd.date_range(
    last_date + pd.Timedelta(days=1 if agg_level=="day" else 7),
    periods=forecast_horizon,
    freq=freq_str
)

# Convert forecast to Series
forecast_mean_series = pd.Series(forecast_mean, index=forecast_dates)
forecast_ci_lower = pd.Series(forecast_ci[:,0], index=forecast_dates)
forecast_ci_upper = pd.Series(forecast_ci[:,1], index=forecast_dates)

# Combine observed + forecast for plotting
all_dates = pd.concat([df_series["date"], forecast_mean_series.index.to_series()])
all_values = pd.concat([df_series["sightings"], forecast_mean_series])

# --- Plot ---
plt.figure(figsize=(10,4))
plt.bar(df_series["date"], df_series["sightings"], label="Observed", color="skyblue")
plt.plot(all_dates, all_values, color="orange", label="Forecast")
plt.fill_between(forecast_dates, forecast_ci_lower, forecast_ci_upper,
                 color="lightblue", alpha=0.3, label="95% CI")
plt.title(f"Forecast of {agg_level}-aggregated sightings for {vuln_id}")
plt.xlabel("Date")
plt.ylabel("Sightings")
plt.legend()
plt.grid(True)
plt.show()
