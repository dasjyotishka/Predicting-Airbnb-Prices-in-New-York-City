# -*- coding: utf-8 -*-
"""project_airbnb.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mTNzbuzq-H_5fWaO1AwE08IapVGhYE0u

#Mount drive
"""

from google.colab import drive
drive.mount('/content/drive')

"""#Importing libraries"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

"""#Read dataset"""

data = pd.read_csv("/content/drive/MyDrive/ensemble/project_airbnb/dataset/AB_NYC_2019.csv")
data.head()

"""#Feature analysis

"""

data.info()

# Checking how many null values there are in each column
data.isnull().sum()

corr = data.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)

"""#Data Cleaning

###Removing null values
"""

name_null_values = data.name.isnull()
host_null_values = data.host_name.isnull()
print('Percentage of data with an empty name:', len(data.loc[name_null_values])/len(data.id)*100)
print('Percentage of data with an empty host name:', len(data.loc[host_null_values])/len(data.id)*100)

"""Since these are insignificant percentages of data, hence these rows can be ignored/removed."""

data = data.loc[~name_null_values]
data = data.loc[~host_null_values]
data.isnull().sum()

data[data['last_review'].isnull()][['number_of_reviews','reviews_per_month']].head()

"""If a certain listing has a null for its "last_review," that means it has not gotten a review at all(also proved from previous dataframe), so "reviews_per_month" must be 0. 
Also, now we can drop the last-review column containing null values fro a cleaner dataset.

"""

data.fillna({'reviews_per_month':0}, inplace=True)
data.drop('last_review', inplace=True, axis=1)
data.isnull().sum()

"""#Data Visualisations"""

#room-type counts graph
ax4 = data.room_type.value_counts().plot.bar(rot=45)
ax4.set_ylabel("Number of listings")
ax4.bar_label(ax4.containers[0])

"""We see maximum listings are either Private room or Entire home/apt"""

#Avg. price and listings per neighbourhood_group(large area)
neighbourhood_group_df = data.groupby('neighbourhood_group').agg(avg_price=('price','mean'),
                       count=('price','count')).reset_index()
    
neighbourhood_group_df

#Pie chart to show listings in each neighbourhood group
ax1 = data.neighbourhood_group.value_counts().plot.pie(title='Neighborhood Groups',
                                               figsize=(8,8),
                                              autopct='%1.1f%%')
ax1.set_ylabel(None)

"""We see that more than 85% of listings are located in Manhattan and Brooklyn."""

ax4 = neighbourhood_group_df.plot.bar(rot=45, x='neighbourhood_group', y='avg_price')
ax4.bar_label(ax4.containers[0])
ax4

"""We see that highest average price is of listings are located in Manhattan"""

data.groupby('neighbourhood').agg(avg_price=('price','mean'),
                       count=('price','count')).sort_values('count', ascending=False).reset_index()

neighbourhood_df = data.groupby('neighbourhood', as_index=False)['price'].count()
neighbourhood_df = neighbourhood_df.rename(columns={'price':'count'})
neighbourhood_df

total_listings = len(data)
def others(row):
    if (row['count']/total_listings* 100) > 4.0:
        return False
    return True

# Apply the above function to create "Others" column
neighbourhood_df['Others'] = neighbourhood_df.apply(others, axis=1)
neighbourhood_df.head()

def map_others(row):
    if (row['Others'] == True):
        return "Other"
    return row['neighbourhood']

# Apply the above function to create "Others" column
neighbourhood_df['new_neighbourhood_value'] = neighbourhood_df.apply(map_others, axis=1)
len(neighbourhood_df)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Create a label encoder object
le = LabelEncoder()

print(data.shape)
data.head()

# Label Encoding will be used for columns with 2 or less unique values
le_count = 0
for col in data.columns[1:]:
    if data[col].dtype == 'object':
        if len(list(data[col].unique())) <= 2:
            le.fit(data[col])
            data[col] = le.transform(data[col])
            le_count += 1
print('{} columns were label encoded.'.format(le_count))

# convert rest of categorical variable into dummy
data = pd.get_dummies(data, drop_first=True)

print(data.shape)
data.head()

#Feature scaling using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 5))
data_col = list(data.columns)
data_col.remove('price')
for col in data_col:
    data[col] = data[col].astype(float)
    data[[col]] = scaler.fit_transform(data[[col]])
#data['price'] = pd.to_numeric(data['price'], downcast='float')
data.head()

data['last_review'] = pd.to_datetime(data['last_review'])
data.head()

data[data['last_review'].isnull()][['number_of_reviews','reviews_per_month']]

#if a certain listing has a null for its "last_review," that means it has not gotten a review at all(also proved from previous dataframe), so "reviews_per_month" must be 0.
data.fillna({'reviews_per_month':0}, inplace=True)
data.isnull().sum()

df = data[['id','host_id','neighbourhood_group','neighbourhood', 
             'latitude', 'longitude', 'room_type', 'minimum_nights', 
             'number_of_reviews', 'reviews_per_month', 
             'calculated_host_listings_count', 'availability_365']]
df.head()