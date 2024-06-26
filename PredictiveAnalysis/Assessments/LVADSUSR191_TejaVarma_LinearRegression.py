# -*- coding: utf-8 -*-
"""LVADSUSR191_TejaVarma_IA1

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1mnn_lBj7P08wW9fdJW47vUhVQcYmzczg
"""

# Lab 1 Linear Regression

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler,StandardScaler #importing the libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report,r2_score,mean_squared_error

lin_df = pd.read_csv('https://raw.githubusercontent.com/Deepsphere-AI/LVA-Batch5-Assessment/main/expenses.csv')
lin_df

# checking null values or duplicates
lin_df.isnull().sum()

# there are 16 null values in the bmi column.

lin_df.isnull().sum()

lin_df

lin_df.head()

lin_df.describe()

lin_df.info()

#encoding categorical data ----- sex, smoker, region,

from sklearn.preprocessing import LabelEncoder
len = LabelEncoder()
lin_df['sex'] = len.fit_transform(lin_df['sex'])
lin_df['smoker'] = len.fit_transform(lin_df['smoker'])
lin_df['region'] = len.fit_transform(lin_df['region'])

lin_df # now even all the categorical data is encoded in the form of numerical ones.

# feature selection

print('Duplicates :',lin_df.duplicated().sum()) # the row has one dupplicate value
print('Before :',lin_df.shape)

lin_df.drop_duplicates() # dropping the duplicate
print('After :',lin_df.shape)

numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
# Compute the correlation matrix for numerical variables
correlation_matrix = df[numerical_columns].corr()
print("Correlation matrix:\n", correlation_matrix)

#feature selection and data cleaning

colum = lin_df.select_dtypes(include=['int64','float64']).columns
for col in colum:
  sns.boxplot(lin_df,x=lin_df[col])
  plt.title(f'Box Plot of {col}')
  plt.xlabel(col)
  plt.show()

# outlier detection in the dataset

#q1 = quantile(0.25)
#q3 = quantile(0.75)
#iqr = q3 - q1

#outliers = ((lin_df < q1 - 1.5*iqr) | (lin_df > q3 + 1.5*iqr))
#df = lin_df[~outliers] # negating the outliers in the data
#df

# splitting to train and test set

x = lin_df.drop(columns=['Charges'])
y = lin_df['Charges']

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_split =42)

scaler = MinMaxScaler() # scaling the
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# model evaluation metrics ----- mse, mae, rmse.

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
rmse = mean_squared_error(y_test, y_pred,squared=False)
print("Root Mean Squared Error:", rmse)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
r2_s = r2_score(y_test, y_pred)
print("R2 Score:", r2_s)

############################################################################################################################

