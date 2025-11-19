## ARIMA/SARIMAX


- Log-transform counts WITH seasonal components
- Transform data with log(x+1) before fitting SARIMAX
- optionnaly with exog



### Examples

Using SARIMAX Log-transform counts WITHOUT seasonal components.

```bash
python arima/sarimax.py
```

#### Sightings forecast for CVE-2025-54236 on 10/29/2025

![Example forecast for CVE-2025-54236](/docs/img/arima-forecast-1.png)


#### Sightings forecast for CVE-2025-8088 on 10/29/2025


![Example forecast for CVE-2025-8088](/docs/img/arima-forecast-2.png)


SARIMAX needs a lot more data for this use case.


#### Sightings forecast for CVE-2025-8088 on 10/30/2025

```bash
python arima/sarimax1.py
```

Poisson-based Forecast of Vulnerability Sightings (Adaptive Daily/Weekly)

![alt text](/docs/img/arima-forecast-3.png)


#### Sightings forecast for CVE-2025-59287 on 10/30/2025

Poisson-based Forecast of Vulnerability Sightings (Adaptive Daily/Weekly)

![alt text](/docs/img/arima-forecast-4.png)


#### Sightings forecast for CVE-2025-54236 on 10/30/2025

Poisson-based Forecast of Vulnerability Sightings (Adaptive Daily/Weekly)

![alt text](/docs/img/arima-forecast-5.png)


### Note

ARIMA/SARIMAX is not the right paradigm.
We’re not dealing with a long stationary time series; we're dealing with short-lived, bursty event sequences that often decay or saturate after initial visibility.