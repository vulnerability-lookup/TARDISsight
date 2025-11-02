## ARIMA/SARIMAX

### Examples

Using SARIMAX Log-transform counts WITHOUT seasonal components.

```bash
python forecast1.py
```

#### Sightings forecast for CVE-2025-54236 on 10/29/2025

![Example forecast for CVE-2025-54236](/docs/arima-forecast-1.png)


#### Sightings forecast for CVE-2025-8088 on 10/29/2025


![Example forecast for CVE-2025-8088](/docs/arima-forecast-2.png)


SARIMAX needs a lot more data for this use case.


#### Sightings forecast for CVE-2025-8088 on 10/30/2025

```bash
python forecast2.py
```

Poisson-based Forecast of Vulnerability Sightings (Adaptive Daily/Weekly)

![alt text](/docs/arima-forecast-3.png)


#### Sightings forecast for CVE-2025-59287 on 10/30/2025

Poisson-based Forecast of Vulnerability Sightings (Adaptive Daily/Weekly)

![alt text](/docs/arima-forecast-4.png)


#### Sightings forecast for CVE-2025-54236 on 10/30/2025

Poisson-based Forecast of Vulnerability Sightings (Adaptive Daily/Weekly)

![alt text](/docs/arima-forecast-5.png)




### Conclusion

ARIMA/SARIMAX is not the right paradigm.
We’re not dealing with a long stationary time series; we're dealing with short-lived, bursty event sequences that often decay or saturate after initial visibility.