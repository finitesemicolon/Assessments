# Lab 2 Logistic Regression

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler,StandardScaler #importing the libraries for logisitic regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report,r2_score,mean_squared_error

log_df = pd.read_csv('https://raw.githubusercontent.com/Deepsphere-AI/LVA-Batch5-Assessment/main/booking.csv')
log_df

log_df.head()

log_df.duplicated().sum() # there are no duplicate values

log_df.isnull().sum() # there are null values in the room type and average price

# replacing the average price and room type with the median of the data
log_df['room type'].fillna('median')

# replacing the average price and room type with the median of the data
log_df['average price'].fillna('mode')

log_df.isnull().sum() # there are null values in the room type and average price

log_df.info()

# encoding the categorical data

len = LabelEncoder()
log_df['Booking_ID'] = len.fit_transform(log_df['Booking_ID'])
log_df['type of meal'] = len.fit_transform(log_df['type of meal'])
log_df['room type'] = len.fit_transform(log_df['room type'])
log_df['market segment type'] = len.fit_transform(log_df['market segment type'])
log_df['date of reservation'] = len.fit_transform(log_df['date of reservation'])
log_df['booking status'] = len.fit_transform(log_df['booking status'])

#

x = log_df.drop(column = ['booking status'])
y = log_df['booking status']

# splitting the data into training and testing
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_split =42)

#scaling the data -- normalisation
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# model implementation
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# evaluation metrics for Logistic Regression accuracy , precision, recall and f1_score

print("Accuracy : ",accuracy_score)
print("Precision : ",precision_score)
print("Recall : ",recall_score)
print("F1_score : ",f1_score)

# confusion matrix can also be used to denote the

con_matrix = confusion_matrix(y_pred, y_test)
print(con_matrix)