import pandas as pd
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#Import data from csv.
df = pd.read_csv("EURUSDDATA.csv")

#Remove data rows with missing SMA values.
df = df.dropna(subset=["SMA20","SMA50","SMA200"])

#Encode day of week catagory.
days = df[["DayOfWeek"]]
def encode_day(day):
    return time.strptime(day, "%A").tm_wday
days_encoded = pd.Series(map(encode_day, days["DayOfWeek"]))
df[["DayOfWeek"]] = days_encoded

#Convert time inputs to oscillating signals.
df["YearSin"] = np.sin(2*np.pi*df["DayOfYear"]/364)
df["YearCos"] = np.cos(2*np.pi*df["DayOfYear"]/364)
df["DaySin"] = np.sin((2*np.pi/6)*df[["DayOfWeek"]])
df["DayCos"] = np.cos((2*np.pi/6)*df[["DayOfWeek"]])
df["HourSin"] = np.sin((2*np.pi/23)*df[["Hour"]])
df["HourCos"] = np.cos((2*np.pi/23)*df[["Hour"]])
df["MinuteSin"] = np.sin((2*np.pi/59)*df[["Minute"]])
df["MinuteCos"] = np.cos((2*np.pi/59)*df[["Minute"]])
df["SecondSin"] = np.sin((2*np.pi/59)*df[["Second"]])
df["SecondCos"] = np.cos((2*np.pi/59)*df[["Second"]])
df["MillisecondSin"] = np.sin((2*np.pi/999)*df[["Millisecond"]])
df["MillisecondCos"] = np.cos((2*np.pi/999)*df[["Millisecond"]])

#Remove original time columns.
df = df.drop(columns=["DayOfYear", "DayOfWeek", "Hour", "Minute", "Second", "Millisecond"])

#Normalise ask, sma20 and sma50 using sma200.
df["Ask"] = df["Ask"]-df["SMA200"]
df["SMA20"] = df["SMA20"]-df["SMA200"]
df["SMA50"] = df["SMA50"]-df["SMA200"]
df["Ask"] = df["Ask"]/max(abs(df["Ask"]))
df["SMA20"] = df["SMA20"]/max(abs(df["SMA20"]))
df["SMA50"] = df["SMA50"]/max(abs(df["SMA50"]))

#Remove SMA200.
df = df.drop(columns=["SMA200"])

#Function to split data into training, validation and test sets.
def split_train_val_test(data, train_prop=0.7, val_prop=0.2):
    n = len(data)
    train_df = data[0:int(n*train_prop)]
    val_df = data[int(n*train_prop):int(n*(train_prop+val_prop))]
    test_df = data[int(n*(train_prop+val_prop)):]
    return train_df, val_df, test_df

#Extract labels from data.
train_df, val_df, test_df = split_train_val_test(df)
train_data = train_df.drop(columns=["Ask"])
val_data = val_df.drop(columns=["Ask"])
test_data = test_df.drop(columns=["Ask"])
train_labels = train_df["Ask"]
val_labels = val_df["Ask"]
test_labels = test_df["Ask"]

#Reshape data.
train_data = np.expand_dims(train_data, axis=0)
val_data = np.expand_dims(val_data, axis=0)
test_data = np.expand_dims(test_data, axis=0)
train_labels = np.expand_dims(train_labels, axis=0)
val_labels = np.expand_dims(val_labels, axis=0)
test_labels = np.expand_dims(test_labels, axis=0)

