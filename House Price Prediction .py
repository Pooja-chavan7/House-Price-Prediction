#!/usr/bin/env python
# coding: utf-8

# IMPORTED REQUIRED LIBRARIES

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBRegressor


# Loading the dataset to pandas dataframe

# In[2]:


df = pd.read_csv(r"C:\Users\POOJA CHAVAN\OneDrive\Documents\Python project\Housing_dataset.csv")


# In[3]:


print(df)


# print first 5 rows of our dataframes

# In[4]:


df.head(10)


# Checking the number of rows and columns in the dataframe 

# In[5]:


df.shape


# Check for missing values

# In[6]:


df.isnull().sum()


# To find missing values

# In[7]:


df.loc[df.price.isna()==True] 


# To fill the null values in price by mean

# In[8]:


df.loc[(df.price.isna()==True),"price"]


# In[9]:


df.price.mean()


# In[10]:


df.loc[(df.price.isna()==True),"price"] = df.price.mean()


# In[11]:


df.isnull().sum()


# In[12]:


df.shape


#  statistical measures of the datasets  

# In[13]:


df.describe()    


# Understanding the  corelation between various features in the datasets
# positive correlation
# Negative correlation

# In[14]:


correlation = df.corr()
print(correlation)


# Constructing a heatmap to understand the correlation

# In[15]:


plt.figure(figsize =(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt=".1f", annot=True, annot_kws={"size":8},cmap="Blues")


# Splitting the data 
# (If you are dropping a column mention axis = 1, for row axis = 0)

# In[16]:


x = df.drop(["price"],axis=1)
y = df["price"]
print(x)
print(y)


# Spiltting the data into training data and test data

# In[17]:


X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.1, random_state = 2)


# In[18]:


print(x.shape,X_train.shape, X_test.shape)


# Model training
# /XGBoost regressor
# /Loading the model

# In[19]:


model = XGBRegressor()


# In[20]:


model.fit(X_train, Y_train)


# Evaluation

# Prediction on training data

# In[21]:


#accuracy for prediction on training data
training_data_prediction = model.predict(X_train)


# In[22]:


print(training_data_prediction)


# In[24]:


#R squared error 
score_1 = metrics.r2_score(Y_train, training_data_prediction)

#mean absolute error
score_2 = metrics.mean_absolute_error(Y_train, training_data_prediction)


# In[27]:


print("R Squared error:",score_1)
print("Mean Absolute Error:", score_2)


# Visualizing the actual prices and predicted prices

# In[36]:


plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual prices")
plt.ylabel("Predicted prices")
plt.title("Actual price vs Predicted price")
plt.show()


# Prediction on Test data

# In[30]:


#accuracy for prediction on test data
test_data_prediction = model.predict(X_test)

#R squared error 
score_1 = metrics.r2_score(Y_test, test_data_prediction)

#mean absolute error
score_2 = metrics.mean_absolute_error(Y_test, test_data_prediction)


# In[37]:


print("R squared error:", score_1)
print("Mean Absolute Error:",score_2)

