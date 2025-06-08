import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
#stockName=input("Stock Name:")
stockName="NVDA"
stock = yf.Ticker(stockName).history(period="1y", interval="1d", start=None, end=None, actions=True, auto_adjust=True, back_adjust=False)
dates = stock.index
df = stock

#print(df)

"""
print(df.shape)
print(df.describe())
print(df.info())

plt.figure(figsize=(15,5))
plt.plot(df['Close'])
plt.title(stockName+' Close price.', fontsize=15)
plt.ylabel('Price in dollars.')
plt.show()
"""
####
"""
features = ['Open', 'High', 'Low', 'Close', 'Volume']

plt.subplots(figsize=(20,10))

for i, col in enumerate(features):
  plt.subplot(2,3,i+1)
  sb.distplot(df[col])
plt.show()
###
plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
  plt.subplot(2,3,i+1)
  sb.boxplot(df[col])
plt.show()
###
"""

for df.index in df:
    for date in df.index:
        date=str(date).split(' ')[0]
        splitted =date.split('-')
        print(splitted)
        df['day'] = splitted[2]
        df['month'] = splitted[1]
        df['year'] = splitted[0]
print(df)