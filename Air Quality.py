#!/usr/bin/env python
# coding: utf-8

# # Predicting Air Quality in Abuja, Nigeria using Time Series

# Air quality is a critical environmental concern, impacting public health,  well-being, and the ecosystem.  Predicting air quality trends can empower proactive measures and informed decision-making.

# # Introduction
# Air pollution poses a significant concern for urban areas worldwide, and Abuja, Nigeria, is no exception.  Pollutants can adversely affect respiratory health, exacerbate existing conditions, and harm the environment.  To support preventative measures and strategic policy decisions, this project seeks to develop a time series forecasting model to predict air quality levels in Abuja. By analyzing historical patterns and identifying trends, the model aims to provide insights for proactive public health interventions and environmental management.
# 
# #### Aim of the Project
# 
# The primary aim of this project is to develop an accurate and reliable time series forecasting model to predict future air quality levels in Abuja, Nigeria.  Specifically, the model will focus on predicting PM2.5 concentrations

# In[2]:


import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import mean_absolute_error, r2_score
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.api import AutoReg
from statsmodels.tsa.api import AutoReg

import time
import warnings


# In[3]:


# Import the files using glob
directory = "downloads/"

# List to store DataFrames
dfs = []
file_pattern = directory + "sensor_data_archive*"
file_paths = glob.glob(file_pattern)
print(file_paths)


# In[4]:


for file_path in file_paths:
    # Read the CSV file in chunks
    chunks = pd.read_csv(file_path, sep=";", chunksize=100000)
    # Processing each chunk
    for chunk in chunks:
        # Filter rows where value type is "P2"
        filtered_chunk = chunk[chunk["value_type"] == "P2"]
        # Append the filtered chunk to the list
        dfs.append(filtered_chunk)


# In[5]:


# Concatenate all filtered chunks into a single DataFrame
air_df = pd.concat(dfs, ignore_index=True)
air_df.shape


# In[6]:


air_df.tail()


# In[7]:


air_df.drop(columns=["sensor_id", "sensor_type", "location", "lat", "lon", "value_type"], inplace=True)


# In[8]:


air_df.head()


# In[9]:


# Function to parse timestamp
def parse_timestamp(timestamp):
    try:
        return pd.to_datetime(timestamp)
    except ValueError:
        return pd.Nat


# In[10]:


# Passing timestamps
air_df["timestamp"] = air_df["timestamp"].apply(parse_timestamp)

# Set timestamp feature as index
air_df.set_index("timestamp", inplace=True)


# In[11]:


air_df.head()


# In[12]:


# Localize Timezone
air_df.index = air_df.index.tz_convert("Africa/Lagos")

# Round seconds to 2 decimal places
air_df.index = air_df.index.round("1s")

# Note: All states in Nigeria use the same timezone.


# In[13]:


air_df.head()


# In[14]:


missing_timestamps = air_df.index[air_df.index.isna()]
print("Missing timestamps:", missing_timestamps)


# In[15]:


air_df = air_df.rename(columns={"value": "P2"})
air_df.head()


# In[16]:


air_df.shape


# In[17]:


air_df.info()


# In[18]:


air_df["P2"] = air_df["P2"].astype(float)


# In[19]:


fig, ax = plt.subplots(figsize=(15, 6))
air_df["P2"].plot(kind="box", ax=ax, vert=False, title="Distribution of PM2.5 Readings");


# From the plot above, we can deduce that:  
# More data points are concentrated towards higher PM2.5 values compared to lower values. 
# 
# The interquartile range (IQR) suggests moderate variability in PM2.5 readings within the middle 50% of the data.  
# 
# Several outliers are present above the upper whisker. These outliers represent instances of significantly higher PM2.5 pollution levels compared to the majority of the data.

# In[20]:


# Handle the outliers
air_df = air_df[air_df["P2"] < 500]


# In[21]:


air_df.shape


# In[22]:


fig, ax = plt.subplots(figsize=(15, 6))
air_df["P2"].plot(kind="box", ax=ax, vert=False, title="Distribution of PM2.5 Readings");


# In[23]:


fig, ax = plt.subplots(figsize=(15, 6))
air_df["P2"].plot(ax=ax, xlabel="Time", ylabel="PM2.5 Time Series")
plt.xlabel("Date")
plt.ylabel("PM 2.5 Level")
plt.title("Hourly PM2.5 Concentrations in Abuja, Nigeria from October 2023 to March 2024");


# The time series plot suggests a potential seasonal pattern in PM2.5 concentrations, with a noticeable increase starting around December 2023. This trend might be associated with the dry season in Abuja, which can worsen air pollution levels due to factors like dust suspension and reduced dispersion of pollutants.  
# 
# The data exhibits significant daily variability in PM2.5 readings, with occasional spikes throughout the observed period. This variability highlights the dynamic nature of air quality in the region, influenced by various factors.  
# 
# There are periods of consistently higher PM2.5 levels, particularly around mid-November 2023 and mid-February 2024. These periods warrant further investigation to identify potential causes, which could include seasonal weather patterns, specific events, or changes in human activities.  
# 
# There seem to be diurnal patterns (daily variations) in the PM2.5 levels, potentially related to human activities, traffic patterns, or weather conditions.

# In[24]:


# Resample to 1H, ffill missing values
y = air_df["P2"].resample("1H").mean().fillna(method="ffill")


# In[25]:


y.tail()


# In[26]:


y.shape


# In[27]:


# the rolling average of the readings
fig, ax = plt.subplots(figsize=(15,6))
y.rolling(168).mean().plot(ax=ax, xlabel="Date", ylabel="PM2.5", title="Abuja PM2.5 Levels, 7-Day Rolling Average");


# The plot reveals significant fluctuations in PM2.5 levels, with multiple peaks exceeding 60 μg/m³ and some surpassing 80 μg/m³. These spikes highlight periods of severe air pollution in Abuja, potentially posing risks to public health.  
# 
# 
# A potential seasonal pattern emerges, with elevated PM2.5 concentrations observed during specific months, notably around November 2023 and February-March 2024.  
# 
# 
# The 7-day rolling average underscores that periods of elevated air pollution are not limited to isolated spikes. There are sustained periods where the average PM2.5 concentration remains above 50 μg/m³.  
# 
# 
# The PM2.5 levels observed in Abuja significantly exceed the World Health Organization's (WHO) recommended air quality guidelines. The annual mean guideline is 5 μg/m³, and the 24-hour mean should not surpass 15 μg/m³. This substantial discrepancy raises serious concerns about the potential health impacts on the local population.

# In[28]:


fig, ax = plt.subplots(figsize=(15, 6))
plot_acf(y, ax=ax)
plt.xlabel("Lag Hours")
plt.ylabel("Correlation Coefficient")
plt.title("Abuja PM2.5 Readings, ACF");


# The ACF plot exhibits a significant positive spike at lag 1, indicating a strong autocorrelation between PM2.5 concentrations on consecutive days. This suggests a short-term dependence within the data. 
# 
# The autocorrelation values steadily decrease as the lag increases. This pattern implies that past PM2.5 readings have a diminishing influence on current readings over time.  
# 
# While the immediate past (lag 1) has significant influence, there is no strong evidence of seasonality or recurring cyclical patterns beyond the one-day lag. This lack of additional significant spikes suggests an absence of long-term periodic trends within the observed timeframe.

# In[29]:


fig, ax = plt.subplots(figsize=(15, 6))
plot_pacf(y, ax=ax)
plt.xlabel("Lag Hours")
plt.ylabel("Correlation Coefficient")
plt.title("Abuja PM2.5 Readings, PACF");


# In[30]:


# Splitting the data into train and test sets
cutoff_test = int(len(y) * 0.90)
y_train = y.iloc[:cutoff_test]
y_test = y.iloc[cutoff_test:]
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# In[31]:


# Get Baseline for the model
y_train_mean = y_train.mean()
y_pred_baseline = [y_train_mean] * len(y_train)
mae_baseline = mean_absolute_error(y_train, y_pred_baseline)

print("Mean P2 Reading:", y_train_mean)
print("Baseline Mae:", mae_baseline )


# In[32]:


# Create range to test the different lags
p_params = range(1, 31)

# Create empty list to hold the mean absolute error scores
maes = []

# Iterate through all values of p in p_param
for p in p_params:
    start_time = time.time()
    # Build model
    model = AutoReg(y_train, lags=p).fit()
    
    elapsed_time = round(time.time() - start_time, 2)
    print(f"Trained AR {p} in {elapsed_time} seconds.")
    
    # Make predictions on training data, dropping null values caused by lag
    y_pred = model.predict().dropna()
    
    # Calculate mean absolute error for training data vs predictions
    mae = mean_absolute_error(y_train.iloc[p:], y_pred)
    
    # Append `mae` to list `maes`
    maes.append(mae)
    
# Put list `maes` into Series with index `p_params`
mae_series = pd.Series(maes, name="mae", index=p_params)

mae_series.head()


# In[33]:


best_p = mae_series.idxmin()
best_model = AutoReg(y_train, lags=best_p).fit()


# In[34]:


y_train_resid = model.resid
y_train_resid.name = "residuals"
y_train_resid.head()


# In[35]:


y_train_resid.hist()
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Best Model, Training Residuals");


# In[36]:


fig, ax = plt.subplots(figsize=(15, 6))
plot_acf(y_train_resid, ax=ax)
plt.xlabel("Lag [hours]")
plt.ylabel("Correlation Coefficient")
plt.title("Abuja, Training Residuals ACF");


# In[37]:


# Walk-forward validation for the model
y_pred_wfv = []
history = y_train.copy()
for i in range(len(y_test)):
    model = AutoReg(history, lags=best_p).fit()
    y_pred = model.predict(start=len(history), end=len(history))
    y_pred_wfv.append(y_pred[0]) 
    history = pd.concat([history, y_test.iloc[i:i+1]])

y_pred_wfv = pd.Series(y_pred_wfv, index=y_test.index)
y_pred_wfv.name = "prediction"
y_pred_wfv.index.name = "timestamp"
y_pred_wfv.head()


# In[38]:


df_pred_test = pd.DataFrame({"y_test": y_test, "y_pred_wfv": y_pred_wfv})
fig = px.line(df_pred_test, labels={"value": "PM2.5"})
fig.update_layout(
    title="Abuja, WFV Predictions",
    xaxis_title="Date",
    yaxis_title="PM2.5 Level",
)


# #### Result
# 

# The plot compares actual (blue line: y_test) and predicted (red line: y_pred_wfv) PM2.5 concentrations over a specific time period. Both lines exhibit substantial fluctuations with noticeable peaks, indicating dynamic air quality conditions in Abuja.  
# 
# There are periods of strong agreement between the observed and predicted values, highlighting the Weather Forecast Model's ability to make accurate predictions under certain circumstances.  
# 
# However, deviations between the two lines also exist, suggesting potential limitations in the model's accuracy or a need to consider additional variables to improve its predictive power.  
# 
# The high PM2.5 levels (reaching around 120 µg/m³) observed in both the actual and predicted data warrant further investigation. These peaks might reflect specific pollution events or recurring factors influencing air quality in Abuja.

# In[39]:


# Make predictions on the test set using the best AR model
y_pred_test = best_model.predict(start=len(y_train), end=len(y) - 1)

# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print(f"Test Set MAE: {mae:.2f}")
print(f"Test Set R-squared: {r2:.2f}")


# In[169]:


from joblib import dump
filename = 'best_ar_model.joblib'
dump({'model': best_model, 'best_lag': best_p}, filename)


# #### Seasonal ARIMA

# In[118]:


order = (1, 1, 0)  
seasonal_order = (1, 1, 0, 20)

# Create the SARIMAX model
sarima_model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order) 

# Fit the model
sarima_results = sarima_model.fit()

# Make predictions on the test set
y_pred_test_sarima = sarima_results.predict(start=len(y_train), end=len(y) - 1)

# Calculate SARIMA performance metrics
mae_sarima = mean_absolute_error(y_test, y_pred_test_sarima)
r2_sarima = r2_score(y_test, y_pred_test_sarima)

print(f"SARIMA Test Set MAE: {mae_sarima:.2f}")
print(f"SARIMA Test Set R-squared: {r2_sarima:.2f}")


# In[119]:


print("Performance Comparison:")
print(f"AR Model Test MAE: {mae:.2f}")
print(f"SARIMA Model Test MAE: {mae_sarima:.2f}")


# In[124]:


# Assuming your data is in 'y_train'
order = (1, 1, 0)  # AR(1)
# Adjust seasonal_order based on your ACF plot analysis
seasonal_order = (1, 0, 0, 20)  # Example: Seasonal period of 20 (adjust as needed)

# Create the SARIMA model
sarima_model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order)

# Fit the model
sarima_results = sarima_model.fit()

# Make predictions on the test set
pred = sarima_results.get_prediction(start=len(y_train), end=len(y) - 1) 
y_pred_test_sarima = pred.predicted_mean

# Calculate SARIMA performance metrics
from sklearn.metrics import mean_absolute_error, r2_score
mae_sarima = mean_absolute_error(y_test, y_pred_test_sarima)
r2_sarima = r2_score(y_test, y_pred_test_sarima)

# Print the results
print(f"SARIMA Test Set MAE: {mae_sarima:.2f}")
print(f"SARIMA Test Set R-squared: {r2_sarima:.2f}")


# In[ ]:




