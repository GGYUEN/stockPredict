import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import keras
from keras import layers
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

model = keras.Sequential()
model.add(layers.Dense(10, activation="relu",input_shape=(10,)))
model.add(layers.Dense(10, activation="relu",input_shape=(10,)))
model.add(layers.Dense(10, activation="relu",input_shape=(10,)))
model.add(layers.Dense(10, activation="relu",input_shape=(10,)))
model.add(layers.Dense(2, activation="softmax"))

stockName = "TSLA"
stock = yf.Ticker(stockName).history(period='1258d', interval='1d')
stock.head()

training_set = stock.iloc[:,3:4].values
print("The size of the original data is "+ str(training_set.shape))

sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

X_train = []
Y_train = []

X_train = training_set_scaled[0:1257]
Y_train = training_set_scaled[1:1258]
X_train = np.reshape(X_train, (1257, 1, 1))

print("Size of X_train is " + str(X_train.shape))
print("Size of Y_train is " + str(Y_train.shape))

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))

model = Sequential()
model.add(LSTM(units = 50, return_sequences=True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.summary()

model.compile(optimizer= 'adam', loss='mean_squared_error')

model.fit(X_train, Y_train, epochs= 100, batch_size=50)

dataset_test = yf.Ticker(stockName).history(period='1258d', interval='1d')
real_stock_price = dataset_test.iloc[:, 3:4].values
dataset_test.head(20)

X_test = sc.transform(real_stock_price)
X_test = np.reshape(X_test, (1258, 1 , 1))
Y_test = model.predict(X_test)
Y_test = sc.inverse_transform(Y_test)

plt.figure(figsize=(14, 7))
plt.plot(dataset_test.index,real_stock_price, color="red", label = 'Real stock price')
plt.plot(dataset_test.index,Y_test, color='blue', label='Predicted')
plt.title(stockName+" Stock Price Prediction")
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()
