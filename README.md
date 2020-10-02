# Predicting-Stock-Prices-Using-fbProphet-Model
The most commonly used models for forecasting predictions are the autoregressive models. Briefly, the autoregressive model specifies that the output variable depends linearly on its own previous values and on a stochastic term an imperfectly predictable term. Recently, in an attempt to develop a model that could capture seasonality in time-series data, Facebook developed the famous Prophet model that is publicly available for everyone; I will use this state of the art model the Prophet model. Prophet is able to capture daily, weekly and yearly seasonality along with holiday effects, by implementing additive regression models in this project Use the Prophet mode to predict Google’s stock price in the future.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset using pandas
data = pd.read_csv("C:/Users/Tarun/YPYTHON/My Project/Predicting Stock Prices Using Facebook’s Prophet Model/GOOG.csv")
data.head(5)

# Select only the important features i.e. the date and price
data = data[["Date","Close"]] # select Date and Price
# Rename the features: These names are NEEDED for the model fitting
data = data.rename(columns = {"Date":"ds","Close":"y"}) #renaming the columns of the dataset
data.head(5)

from fbprophet import Prophet
m = Prophet(daily_seasonality = True) # the Prophet class (model)
m.fit(data) # fit the model using all data

future = m.make_future_dataframe(periods=365) #we need to specify the number of days in future
prediction = m.predict(future)
m.plot(prediction)
plt.title("Prediction of the Google Stock Price using the Prophet")
plt.xlabel("Date")
plt.ylabel("Close Stock Price")
plt.show()

m.plot_components(prediction)
plt.show()
