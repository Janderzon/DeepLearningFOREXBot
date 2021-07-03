import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt

#Import data from csv.
df = pd.read_csv("EURUSDDATA.csv")

#Remove data rows with missing SMA values.
df = df.dropna(subset=["SMA20","SMA50"])

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

#Remove original time columns
df = df.drop(columns=["DayOfYear", "DayOfWeek", "Hour", "Minute", "Second", "Millisecond"])

#Function to split data into training, validation and test sets.
def split_train_val_test(data, train_prop=0.7, val_prop=0.2):
    n = len(data)
    train_df = data[0:int(n*train_prop)]
    val_df = data[int(n*train_prop):int(n*(train_prop+val_prop))]
    test_df = data[int(n*(train_prop+val_prop)):]
    return train_df, val_df, test_df