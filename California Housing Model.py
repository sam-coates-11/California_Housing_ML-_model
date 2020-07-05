#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv("/Users/samcoates/Documents/ML models/california_housing.csv")


# In[4]:


# ANALYSIS AND DATA CLEANING


# In[5]:


df.isnull().sum()


# In[6]:


df.describe().transpose()


# In[7]:


df.head()


# In[8]:


len(df)


# In[9]:


plt.figure(figsize=(10,6))
sns.distplot(df['median_house_value'])


# In[10]:


indexMHV = df[df['median_house_value'] >= 490000.0 ].index


# In[11]:


df.drop(indexMHV , inplace=True)


# In[12]:


df.head()


# In[13]:


plt.figure(figsize=(10,6))
sns.distplot(df['median_house_value'])


# In[14]:


len(df)


# In[15]:


plt.figure(figsize=(10,6))
sns.distplot(df['total_bedrooms'],bins=50)


# In[16]:


indexTB = df[df['total_bedrooms'] >= 2000.0].index


# In[17]:


df.drop(indexTB , inplace=True)


# In[18]:


plt.figure(figsize=(10,6))
sns.distplot(df['total_bedrooms'],bins=100)


# In[19]:


df.corr()


# In[20]:


100 * df.isnull().sum()['total_bedrooms']/df.count()['total_bedrooms']


# In[21]:


# using total_rooms to fill in remaining missing total_bedroom values


# In[22]:


total_rooms_avg = df.groupby('total_rooms').mean()['total_bedrooms']


# In[23]:


def fill_total_bedrooms(total_rooms,total_bedrooms):
    
    if np.isnan(total_bedrooms):
        return total_rooms_avg[total_rooms]
    else:
        return total_bedrooms


# In[24]:


df['total_bedrooms'] = df.apply(lambda x:fill_total_bedrooms(x['total_rooms'],x['total_bedrooms']),axis=1)


# In[25]:


df.isnull().sum()


# In[26]:


df = df.dropna()


# In[27]:


df.isnull().sum()


# In[28]:


df.select_dtypes(['object']).columns


# In[29]:


df['ocean_proximity'].value_counts()


# In[30]:


df['ocean_proximity'] = df['ocean_proximity'].replace(['<1H OCEAN'],'INLAND')


# In[31]:


df['ocean_proximity'].value_counts()


# In[32]:


df['ocean_proximity'] = df['ocean_proximity'].replace(['NEAR OCEAN','NEAR BAY','ISLAND'],'NEAR WATER')


# In[33]:


df.head()


# In[34]:


df['ocean_proximity'].value_counts()


# In[35]:


dummies = pd.get_dummies(df[['ocean_proximity']],drop_first=True)

df = pd.concat([df.drop('ocean_proximity',axis=1),dummies],axis=1)


# In[36]:


df.head()


# In[40]:


df.rename(columns = {'ocean_proximity_NEAR WATER':'ocean_proximity'}, inplace = True)


# In[41]:


len(df)


# In[42]:


df.columns


# In[43]:


# TRAINING AND TESTING THE DATA


# In[44]:


from sklearn.model_selection import train_test_split


# In[45]:


X = df.drop('ocean_proximity',axis=1).values
y = df['ocean_proximity'].values


# In[46]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# In[47]:


from sklearn.preprocessing import MinMaxScaler


# In[48]:


scaler = MinMaxScaler()


# In[49]:


X_train = scaler.fit_transform(X_train)


# In[50]:


X_test = scaler.transform(X_test)


# In[51]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout


# In[52]:


X_train.shape


# In[53]:


model = Sequential()

# input layer
model.add(Dense(9,  activation='relu'))
model.add(Dropout(0.2))

# hidden layer
model.add(Dense(5, activation='relu'))
model.add(Dropout(0.2))

# hidden layer
model.add(Dense(3, activation='relu'))
model.add(Dropout(0.2))

# output layer
model.add(Dense(units=1,activation='relu'))

# Compile model
model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())


# In[54]:


model.fit(x=X_train,y=y_train,epochs=100,batch_size=256,validation_data=(X_test,y_test))


# In[55]:


# EVALUATING MODEL PERFORMANCE


# In[56]:


losses = pd.DataFrame(model.history.history)


# In[57]:


losses.plot()


# In[58]:


# PREDICTIONS


# In[59]:


from sklearn.metrics import classification_report,confusion_matrix


# In[63]:


predictions = model.predict_classes(X_test)


# In[64]:


print(classification_report(y_test,predictions))


# In[65]:


confusion_matrix(y_test,predictions)


# In[66]:


df['ocean_proximity'].value_counts()


# In[67]:


14857/len(df)


# In[68]:


# TEST


# In[70]:


import random
random.seed(101)
random_ind = random.randint(0,len(df))

new_housing_block_data = df.drop('ocean_proximity',axis=1).iloc[random_ind]
new_housing_block_data


# In[74]:


new_housing_block_data = scaler.transform(new_housing_block_data.values.reshape(1,9))


# In[75]:


new_housing_block_data


# In[76]:


model.predict_classes(new_housing_block_data)


# In[77]:


# CHECK


# In[78]:


df.iloc[random_ind]['ocean_proximity']


# In[ ]:




