

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

tweets_df = pd.read_csv("stock_tweets.csv")
stocks_df = pd.read_csv("stock_yfinance_data.csv")

tweets_df["Date"] = pd.to_datetime(tweets_df["Date"]).dt.date
stocks_df["Date"] = pd.to_datetime(stocks_df["Date"]).dt.date

def get_sentiment(text):
    return TextBlob(str(text)).sentiment.polarity

tweets_df["Sentiment"] = tweets_df["Tweet"].apply(get_sentiment)
daily_sentiment = tweets_df.groupby(["Date", "Stock Name"])["Sentiment"].mean().reset_index()
merged_df = pd.merge(stocks_df, daily_sentiment, on=["Date", "Stock Name"], how="left")
merged_df["Sentiment"].fillna(0, inplace=True)

features = ["Open", "High", "Low", "Close", "Volume", "Sentiment"]
scaler = MinMaxScaler()
merged_df[features] = scaler.fit_transform(merged_df[features])

SEQ_LENGTH = 30
data_values = merged_df[features].values
target = merged_df["Close"].values

split_idx = int(len(data_values) * 0.8)
train_data, test_data = data_values[:split_idx], data_values[split_idx:]
train_target, test_target = target[:split_idx], target[split_idx:]

train_gen = TimeseriesGenerator(train_data, train_target, length=SEQ_LENGTH, batch_size=16)
test_gen = TimeseriesGenerator(test_data, test_target, length=SEQ_LENGTH, batch_size=16)

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(SEQ_LENGTH, len(features))),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.fit(train_gen, validation_data=test_gen, epochs=20)

predictions = model.predict(test_gen)

plt.figure(figsize=(10,5))
plt.plot(test_target[SEQ_LENGTH:], label="Actual Price", color='blue')
plt.plot(predictions, label="Predicted Price", color='red', linestyle="dashed")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Stock Price (Scaled)")
plt.title("Stock Price Prediction using LSTM & Sentiment Analysis")
plt.show()

