# Ex.No: 6               HOLT WINTERS METHOD
### Date: 06-05-2025

### AIM:
To implement the Holt Winters Method Model using Python.

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
   
### PROGRAM:

####  Name : Bhuvaneshwaran H
#### Reg No : 212223240018

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = pd.read_csv('silver.csv', parse_dates=['Date'], index_col='Date')

monthly_usd = data['USD'].resample('MS').mean()

monthly_usd.plot(title='Monthly Average Silver Price (USD)', figsize=(10, 4))
plt.ylabel('Price (USD)')
plt.show()

scaler = MinMaxScaler()
scaled = pd.Series(
    scaler.fit_transform(monthly_usd.values.reshape(-1,1)).flatten(),
    index=monthly_usd.index
)

scaled.plot(title='Scaled Monthly Silver Price', figsize=(10, 4))
plt.ylabel('Scaled Value')
plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose
decomp = seasonal_decompose(monthly_usd, model='additive', period=12)
decomp.plot()
plt.show()

scaler = MinMaxScaler()
scaled_data = pd.Series(
    scaler.fit_transform(monthly_usd.values.reshape(-1,1)).flatten(),
    index=monthly_usd.index
)

scaled_data = scaled_data + 1  

split = int(len(scaled_data) * 0.8)
train_data = scaled_data[:split]
test_data  = scaled_data[split:]

model_add = ExponentialSmoothing(
    train_data, 
    trend='add', 
    seasonal='mul', 
    seasonal_periods=12
).fit()

test_predictions_add = model_add.forecast(steps=len(test_data))

ax = train_data.plot(label='train_data', figsize=(10, 6))
test_predictions_add.plot(ax=ax, label='test_predictions_add')
test_data.plot(ax=ax, label='test_data')
ax.legend()
ax.set_title('Visual evaluation')
plt.show()

rmse = np.sqrt(mean_squared_error(test_data, test_predictions_add))
print(f"Test RMSE: {rmse:.4f}")

print(f"Std dev of scaled_data: {scaled_data.std():.4f}")
print(f"Mean of scaled_data:    {scaled_data.mean():.4f}")

import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

data = pd.read_csv('silver.csv', parse_dates=['Date'], index_col='Date')

data_monthly = data['USD'].resample('MS').mean()

final_model = ExponentialSmoothing(
    data_monthly, 
    trend='add', 
    seasonal='mul', 
    seasonal_periods=12
).fit()

forecast_steps = 12
final_predictions = final_model.forecast(steps=forecast_steps)

ax = data_monthly.plot(label='Observed Silver Price', figsize=(10, 6))
final_predictions.plot(ax=ax, label='12‑Month Forecast', linestyle='--')

ax.set_xlabel('Date')
ax.set_ylabel('Silver Price (USD)')
ax.set_title('Silver Price Forecast for Next Year')
ax.legend()
plt.tight_layout()
plt.show()
```

### OUTPUT:

Scaled_data plot:

![image](https://github.com/user-attachments/assets/90e1f0da-1d88-4c32-977d-df37a3c83b4b)

Decomposed plot:

![image](https://github.com/user-attachments/assets/cca53668-62b2-48c0-8cc7-8ab39e86e64d)

Test prediction:


![image](https://github.com/user-attachments/assets/edb71670-ce0e-452b-9d28-e67fe0b26534)

Model performance metrics:

![image](https://github.com/user-attachments/assets/1b56404f-73d1-4f8d-9c51-a6e295f25c18)

Final prediction:

![image](https://github.com/user-attachments/assets/fc50832c-5693-49be-92c0-8a58d0f225d0)


### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
