# -*- coding: utf-8 -*-
"""LVADSUSR191_TejaVarma_Reassessment_Clustering

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Y7gwkaZ1rjFc3-qLarhcm8oVtWmy6C1c
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler,StandardScaler,LabelEncoder
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from numpy import loadtxt

from sklearn.cluster import KMeans

from sklearn.metrics import r2_score,mean_squared_error , silhouette_score
from sklearn.metrics import accuracy_score , precision_score , recall_score , f1_score , confusion_matrix

df =pd.read_csv('https://raw.githubusercontent.com/Deepsphere-AI/LVA-Batch5-Assessment/main/Credit%20Card%20Customer%20Data.csv')

df.head()

df.info()

df.describe()

df.isna().sum()
# there are 13 null values in the total_visits_online column

# univariate analysis
for column in df.select_dtypes(include=['int64','float64']).columns:
  plt.figure(figsize=(10,5))
  sns.histplot(df[column])

df.columns

# imputing the null values
# total_visits_online is left skewed so we are considering replacing with the median values
df['Total_visits_online']=df['Total_visits_online'].fillna(df['Total_visits_online'].median())

# checking for null values again
df.isna().sum()
# null values imputed

df.duplicated().sum()
# no duplicates found

#visualising the distribution of data
for i in df.columns:
  sns.boxplot(df[i])
  plt.show()

for i in df.columns:
  plt.hist(df[i])
  plt.title(i)
  plt.show()

#check for outliers
for c in df.select_dtypes(include=['int64','float64']).columns:
  plt.figure(figsize=(10,5))
  sns.boxplot(df[c])

# outlier treatment
for c in df.select_dtypes(include=['int64','float64']).columns:
  q1 = df[c].quantile(0.25)
  q3 = df[c].quantile(0.75)
  iqr = q3-q1
  lwr = q1-1.5*iqr
  upr = q3+1.5*iqr
  df.loc[df[c]>upr,c]=upr
  df.loc[df[c]<lwr,c]=lwr

# checking again for outliers
for c in df.select_dtypes(include=['int64','float64']).columns:
  plt.figure(figsize=(10,5))
  sns.boxplot(df[c])

# correlation matrix

numeric = df.select_dtypes(include = ['int64','float64']).columns
heat = df[numeric].corr()
plt.figure(figsize=(15,10))
sns.heatmap(heat,annot =True)

#feature selection
df.drop(columns = ['Customer Key'],inplace = True)
# dropping the customer key column as it is redundant

#encoding objects, here there are no object datatypes in the given data so we can neglect the encoding part in Clustering

# label_encoder = LabelEncoder()

# for col in df.columns:
#     if df[col].dtype == 'object':  # Check if the column contains categorical data
#         df[col] = label_encoder.fit_transform(df[col])

# # scaling
S = MinMaxScaler()
X = S.fit_transform(df)

# Elbow method to find number of clusters in the given data points
see = []
k_values = range(1, 11)
for k in k_values:
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(df)
    see.append(kmeans.inertia_)

plt.figure(figsize=(16, 6))
plt.plot(k_values, see, marker='o',color='#8B4513')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within Cluster Sum of Squares')
plt.show()

# finally taking number of clusters =2

#model implementation
kmeans = KMeans(n_clusters=2)
kmeans.fit(df)
cluster_labels = kmeans.labels_
df['cluster'] =  kmeans.labels_

df.head() # checking the clusters

# Plotting the clusters

df1 = df[df.cluster==0]
df2 = df[df.cluster==1]

plt.scatter(df1['Total_visits_online'],df1['Avg_Credit_Limit'],color='green')
plt.scatter(df2['Total_visits_online'],df2['Avg_Credit_Limit'],color='red')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.xlabel('Total_visits_online')
plt.ylabel('Avg_Credit_Limit')
plt.legend()

sil_score = silhouette_score(df, kmeans.labels_)
print("Silhouette Score:", sil_score)
# silhoutte score -- 0.7179

df1 # cluster 0 which is low spending customers

df2 # cluster 1 which are high spending customers

# Business Recommendations and Insights based on the clustering

# Cluster 1 -- High spending customers ---- some new loyalty programs to them to get increase customer retention value and have some high satisfaction score, and also payback cards and offers
# Cluster 0 -- Low Spending customers ---- enhance their experience by new low end products for their daily use and increase the brand value and promotions, also giving loans at a cheaper interest rates would be much helpful