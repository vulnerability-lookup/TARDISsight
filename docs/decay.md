## Logistic growth and exponential decay

### Logistic growth

Fits a logistic growth curve of the form

y(t) = L / 1 + e^(-k(t-t0))

where:
- L: Maximum expected number of sightings (plateau),
- k: Growth rate
- t0: Day of inflection (midpoint of growth)
- Forecasts 10 future days.

for “burst-and-fade” dynamics such as CVE mentions, social signals, etc.
It captures what we typically observe:
- a rapid increase in reports just after disclosure,
- then a plateau and slow decay as attention fades.

Result:
- A smooth S-shaped curve when there's an early rise and saturation.
- A flat extension when activity already plateaued.

When to use this model:  
For newly published or trending vulnerabilities.


### Exponential decay

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