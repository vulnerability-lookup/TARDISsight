## Adaptive model

Adaptive Logistic or Exponential Forecast.

| Step                | Description                                                                           |
| ------------------- | ------------------------------------------------------------------------------------- |
| **Trend detection** | Checks if recent sightings are increasing or decreasing (via linear slope).           |
| **Model selection** | Uses **logistic** if slope > 0, else **exponential decay**.                           |
| **Curve fitting**   | Uses `scipy.optimize.curve_fit` to estimate parameters.                               |
| **Forecasting**     | Extends the model 10 days into the future.                                            |
| **Plotting**        | Shows observed data (blue), forecast (red dashed), and forecast region (orange area). |


