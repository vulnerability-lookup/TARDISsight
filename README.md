# TARDISsight

## Usage


### Examples

Using SARIMAX Log-transform counts WITHOUT seasonal components.

```bash
python forecast1.py
```

#### Sightings forecast for CVE-2025-54236 on 10/29/2025

![Example forcast for CVE-2025-54236](docs/example-forecast-1.png)


#### Sightings forecast for CVE-2025-8088 on 10/29/2025


![Example forcast for CVE-2025-8088](docs/example-forecast-2.png)


SARIMAX needs a lot more data for this use case.


#### Sightings forecast for CVE-2025-8088 on 10/30/2025

```bash
python forecast2.py
```

Poisson-based Forecast of Vulnerability Sightings (Adaptive Daily/Weekly)

![alt text](docs/example-forecast-3.png)


#### Sightings forecast for CVE-2025-59287 on 10/30/2025

Poisson-based Forecast of Vulnerability Sightings (Adaptive Daily/Weekly)

![alt text](docs/example-forecast-4.png)