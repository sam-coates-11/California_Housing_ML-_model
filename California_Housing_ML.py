# The data contains information from the 1990 California census


# longitude: A measure of how far west a house is; a higher value is farther west

# latitude: A measure of how far north a house is; a higher value is farther north

# housingMedianAge: Median age of a house within a block; a lower number is a newer building

# totalRooms: Total number of rooms within a block

# totalBedrooms: Total number of bedrooms within a block

# population: Total number of people residing within a block

# households: Total number of households, a group of people residing within a home unit, for a block

# medianIncome: Median income for households within a block of houses (measured in tens of thousands of US Dollars)

# medianHouseValue: Median house value for households within a block (measured in US Dollars)

# oceanProximity: Location of the house w.r.t ocean/sea

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv(r"/Users/samcoates/Documents/ML models/Final Model/california_housing.csv")

# ANALYSIS AND DATA CLEANING

df.isnull().sum()

df.describe().transpose()

df.head()

len(df)

plt.figure(figsize=(10, 6))
sns.distplot(df['median_house_value'])
plt.show()

indexMHV = df[df['median_house_value'] >= 490000.0].index

df.drop(indexMHV, inplace=True)

df.head()

plt.figure(figsize=(10, 6))
sns.distplot(df['median_house_value'])

plt.figure(figsize=(10, 6))
sns.distplot(df['total_bedrooms'], bins=50)

indexTB = df[df['total_bedrooms'] >= 2000.0].index

df.drop(indexTB, inplace=True)

plt.figure(figsize=(10, 6))
sns.distplot(df['total_bedrooms'], bins=100)

df.corr()

100 * df.isnull().sum()['total_bedrooms'] / df.count()['total_bedrooms']

# using total_rooms to fill in remaining missing total_bedroom values

total_rooms_avg = df.groupby('total_rooms').mean()['total_bedrooms']


def fill_total_bedrooms(total_rooms, total_bedrooms):
    if np.isnan(total_bedrooms):
        return total_rooms_avg[total_rooms]
    else:
        return total_bedrooms


df['total_bedrooms'] = df.apply(lambda x: fill_total_bedrooms(x['total_rooms'], x['total_bedrooms']), axis=1)

df.isnull().sum()

df = df.dropna()

df.isnull().sum()

df.select_dtypes(['object']).columns

df['ocean_proximity'].value_counts()

df['ocean_proximity'] = df['ocean_proximity'].replace(['<1H OCEAN'], 'INLAND')

df['ocean_proximity'].value_counts()

df['ocean_proximity'] = df['ocean_proximity'].replace(['NEAR OCEAN', 'NEAR BAY', 'ISLAND'], 'NEAR WATER')

df.head()

df['ocean_proximity'].value_counts()

dummies = pd.get_dummies(df[['ocean_proximity']], drop_first=True)

df = pd.concat([df.drop('ocean_proximity', axis=1), dummies], axis=1)

df.head()

df.rename(columns={'ocean_proximity_NEAR WATER': 'ocean_proximity'}, inplace=True)

len(df)

df.columns

# TRAINING AND TESTING THE DATA

X = df.drop('ocean_proximity', axis=1).values
y = df['ocean_proximity'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

X_train.shape()

model = Sequential()

# input layer
model.add(Dense(9, activation='relu'))
model.add(Dropout(0.2))

# hidden layer
model.add(Dense(5, activation='relu'))
model.add(Dropout(0.2))

# hidden layer
model.add(Dense(3, activation='relu'))
model.add(Dropout(0.2))

# output layer
model.add(Dense(units=1, activation='relu'))

# Compile model
model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

model.fit(x=X_train, y=y_train, epochs=100, batch_size=256, validation_data=(X_test, y_test))

# EVALUATING MODEL PERFORMANCE

plt.figure(figsize=(10, 6))
ylim = (0.1, 0.25)
losses = pd.DataFrame(model.history.history)
losses.plot()

# PREDICTIONS

predictions = model.predict_classes(X_test)

print(classification_report(y_test, predictions))

print(confusion_matrix(y_test, predictions))

df['ocean_proximity'].value_counts()

14857 / len(df)

# TEST

random.seed(101)
random_ind = random.randint(0, len(df))

new_housing_block_data = df.drop('ocean_proximity', axis=1).iloc[random_ind]

new_housing_block_data = scaler.transform(new_housing_block_data.values.reshape(1, 9))

model.predict_classes(new_housing_block_data)

# CHECK

df.iloc[random_ind]['ocean_proximity']
plt.show()
