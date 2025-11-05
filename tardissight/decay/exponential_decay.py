"""
Exponential Decay Model for short-term forecasting

y(t) = a⋅e^-bt + c

Fits an exponential decay curve to the observed sightings.
Extends it smoothly for 10 more days.
Produces a visually intuitive curve: a rise followed by a fading tail.

Notes:
- If a vulnerability is still rising (no decay yet), the fit might look flat or strange.
In that case, a growth-then-decay hybrid (logistic model) could be more appropriate.
- For newly published vulnerabilities (e.g. first 1-3 days only), we could skip fitting and instead set a rule-based extrapolation like:
forecast = daily_series[-1] * np.exp(-0.3 * np.arange(1, 11))

When to use this model:
Use exponential decay for vulnerabilities already past their peak.
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

# --- Define exponential decay model ---
def exp_decay(t, a, b, c):
    return a * np.exp(-b * t) + c

# --- Prepare data for fitting ---
t = np.arange(len(daily_series))
y = daily_series.values

# --- Fit the exponential model ---
try:
    popt, pcov = curve_fit(exp_decay, t, y, p0=(max(y), 0.5, min(y)),
                           bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))  # clamp to positive
    a, b, c = popt
    print(f"Fitted parameters: a={a:.2f}, b={b:.3f}, c={c:.2f}")
except RuntimeError:
    print("⚠️ Curve fitting failed. Too few points or unstable data.")
    exit()

# --- Forecast next 10 days ---
future_steps = 10
last_date = daily_series.index.max()  # <-- CHANGE: start forecast after last observed date
t_future = np.arange(len(daily_series), len(daily_series) + future_steps)
forecast_values = np.maximum(exp_decay(t_future, a, b, c), 0)  # clamp negatives

# --- Build date index for forecast starting after last observation ---
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_steps, freq="D")  # <-- CHANGE

# --- Plot results ---
plt.figure(figsize=(10, 4))
plt.plot(daily_series.index, y, "bo-", label="Observed")
plt.plot(forecast_dates, forecast_values, "r--", label="Forecast (Exponential Decay)")

# Highlight forecast region
plt.axvspan(forecast_dates[0], forecast_dates[-1], color="orange", alpha=0.1)

plt.title(f"Exponential Decay Forecast of Sightings for {vuln_id}")
plt.xlabel("Date")
plt.ylabel("Sightings")
plt.legend()
plt.tight_layout()
plt.show()
