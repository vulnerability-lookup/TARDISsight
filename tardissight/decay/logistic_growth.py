"""

Fits a logistic growth curve of the form

y(t) = L / 1 + e^(-k(t-t0))

where:
L: Maximum expected number of sightings (plateau),
k: Growth rate
t0: Day of inflection (midpoint of growth)
Forecasts 10 future days.

for “burst-and-fade” dynamics such as CVE mentions, social signals, etc.
It captures what we typically observe:
- a rapid increase in reports just after disclosure,
- then a plateau and slow decay as attention fades.

Result:
A smooth S-shaped curve when there's an early rise and saturation.
A flat extension when activity already plateaued.

When to use this model:
For newly published or trending vulnerabilities.
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

# --- Define logistic growth function ---
def logistic(t, L, k, t0):
    return L / (1 + np.exp(-k * (t - t0)))

# --- Prepare data ---
t = np.arange(len(daily_series))
y = daily_series.values

if len(y) < 3:
    print("⚠️ Not enough data points to fit a model.")
    exit()

# --- Initial guess for parameters ---
L0 = max(y) * 1.5  # estimated upper bound
k0 = 0.3           # moderate growth rate
t0 = np.median(t)  # inflection around middle

# --- Fit logistic model with non-negative bounds (IMPROVEMENT) ---
try:
    # Bounds ensure L, k, and t0 are non-negative for a sensible growth model
    popt, _ = curve_fit(
        logistic, t, y, p0=[L0, k0, t0], 
        bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
        maxfev=10000
    )
    L, k, t0 = popt
    print(f"Fitted parameters: L={L:.2f}, k={k:.3f}, t0={t0:.2f}")
except RuntimeError:
    print("⚠️ Curve fitting failed. Not enough variation in data or poor initial guess.")
    exit()

# --- Forecast next 10 days ---
future_steps = 10

# --- Forecast starts after last observed date ---
last_date = daily_series.index.max()
t_future = np.arange(len(daily_series), len(daily_series) + future_steps)  # days after last observation

# Apply logistic function and clamp at 0
forecast_values = np.maximum(logistic(t_future, L, k, t0), 0)

# --- Build date index starting after last observed day ---
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_steps, freq="D")

# --- Plot ---
plt.figure(figsize=(10, 4))
plt.plot(daily_series.index, y, "bo-", label="Observed")
plt.plot(forecast_dates, forecast_values, "r--", label="Forecast (Logistic Model)")

# Highlight forecast region
plt.axvspan(forecast_dates[0], forecast_dates[-1], color="orange", alpha=0.1)

plt.title(f"Logistic Forecast of Sightings for {vuln_id}")
plt.xlabel("Date")
plt.ylabel("Sightings")
plt.legend()

# --- Overlay textual summary below plot (IMPROVEMENT) ---
summary = f"{vuln_id} (Logistic Model) suggests growth, expected to plateau around {L:.0f} sightings."
plt.figtext(0.5, -0.05, summary, ha="center", fontsize=10, color="black")

plt.tight_layout()
plt.show()