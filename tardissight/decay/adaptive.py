"""
Adaptive Logistic or Exponential Forecast.

| Step                | Description                                                                           |
| ------------------- | ------------------------------------------------------------------------------------- |
| **Trend detection** | Checks if recent sightings are increasing or decreasing (via linear slope).           |
| **Model selection** | Uses **logistic** if slope > 0, else **exponential decay**.                           |
| **Curve fitting**   | Uses `scipy.optimize.curve_fit` to estimate parameters.                               |
| **Forecasting**     | Extends the model 10 days into the future.                                            |
| **Plotting**        | Shows observed data (blue), forecast (red dashed), and forecast region (orange area). |


"""

import numpy as np
import pandas as pd # type: ignore[import-untyped]
import requests # type: ignore[import-untyped]
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit # type: ignore[import-untyped]

# --- Parameters ---
vuln_id = input("Vulnerability id: ")
per_page = 1000
page = 1
sightings = []

# --- Fetch sightings with pagination ---
while True:
    url = f"https://vulnerability.circl.lu/api/sighting?page={page}&per_page={per_page}&vuln_id={vuln_id}"
    response = requests.get(url, headers={"accept": "application/json"})
    data = response.json()
    sightings.extend(data["data"])
    total = data["metadata"]["count"]
    if page * per_page >= total:
        break
    page += 1

# --- Convert to DataFrame ---
df_sightings = pd.DataFrame(sightings)

# --- Parse timestamps to dates ---
df_sightings["creation_date"] = pd.to_datetime(
    df_sightings["creation_timestamp"], utc=True, errors="coerce"
).dt.date

# --- Count sightings per day ---
daily_counts = df_sightings.groupby("creation_date").size().rename("sightings")

# --- Convert to pandas Series indexed by datetime ---
daily_series = pd.Series(daily_counts)
daily_series.index = pd.to_datetime(daily_series.index)
daily_series.index.name = "date"
daily_series = daily_series.sort_index()

print("Daily series:")
print(daily_series)

# --- Helper models ---
def logistic(t, L, k, t0):
    """Logistic growth model."""
    return L / (1 + np.exp(-k * (t - t0)))

def exp_decay(t, a, b, c):
    """Exponential decay model."""
    return a * np.exp(-b * t) + c

# --- Prepare data ---
t = np.arange(len(daily_series))
y = daily_series.values

if len(y) < 3:
    print("⚠️ Not enough data points to fit a model.")
    exit()

# --- Detect growth or decay phase ---
trend_slope = np.polyfit(t, y, 1)[0]
print(f"Trend slope: {trend_slope:.3f}")

model_type = "logistic" if trend_slope > 0 else "decay"
print(f"Selected model: {model_type}")

# --- Fit appropriate model with bounds ---
try:
    # Bounds in curve_fit: ensures all fitted parameters are non-negative.
    if model_type == "logistic":
        # Initial guess
        L0 = max(y) * 1.5
        k0 = 0.3
        t0 = np.median(t)
        popt, _ = curve_fit(
            logistic, t, y, p0=[L0, k0, t0],
            bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
            maxfev=10000
        )
        forecast_func = lambda tt: logistic(tt, *popt)
    else:
        # Initial guess
        a0 = max(y)
        b0 = 0.3
        c0 = min(y)
        popt, _ = curve_fit(
            exp_decay, t, y, p0=[a0, b0, c0],
            bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
            maxfev=10000
        )
        forecast_func = lambda tt: exp_decay(tt, *popt)
except RuntimeError:
    print("⚠️ Curve fitting failed. Exiting.")
    exit()

# --- Forecast next days correctly after last observed date ---
future_steps = 15
last_date = daily_series.index.max()

# Build future date index starting after the last observed day
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_steps, freq="D")

# Compute t values for fitting: days since first observed day
t_full = np.arange(len(daily_series))
t_future = np.arange(len(daily_series), len(daily_series) + future_steps)

# Forecast values for future days
forecast_future_values = np.maximum(forecast_func(t_future), 0)  # clamp negatives

# --- Combine observed + forecast for plotting ---
all_dates = daily_series.index.append(forecast_dates)
all_values = np.concatenate([daily_series.values, forecast_future_values])

# --- Plot ---
plt.figure(figsize=(10, 4))
plt.plot(daily_series.index, daily_series.values, "bo-", label="Observed")
plt.plot(forecast_dates, forecast_future_values, "r--", label=f"Forecast ({model_type.capitalize()} Model)")
plt.axvspan(forecast_dates[0], forecast_dates[-1], color="orange", alpha=0.1)

plt.title(f"Adaptive Forecast of Sightings for {vuln_id}")
plt.xlabel("Date")
plt.ylabel("Sightings")
plt.legend()

# --- Overlay textual summary below plot ---
if model_type == "logistic":
    summary = f"{vuln_id} shows a rising trend, expected to plateau around {popt[0]:.0f} sightings."
else:
    summary = f"{vuln_id} shows a decreasing trend, approaching {popt[2]:.0f} sightings."

plt.figtext(0.5, -0.05, summary, ha="center", fontsize=10, color="black")

plt.tight_layout()
plt.show()
