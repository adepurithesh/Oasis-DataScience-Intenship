#!/usr/bin/env python
# coding: utf-8

# # Sales Prediction using Python 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("Advertising.csv")
print(data.head())


# In[2]:


print(data.isnull().sum())


# So this dataset does not contain any null values. Now let’s take a look at the correlation between features before we start training a machine learning model to predict future sales:

# In[3]:


plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(12, 10))
sns.heatmap(data.corr())
plt.show()


# Now let’s prepare the data to fit into a machine learning model and then I will use a linear regression algorithm to train a sales prediction model using Python:

# In[4]:


x = np.array(data.drop(["Sales"], 1))
y = np.array(data["Sales"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)

data = pd.DataFrame(data={"Predicted Sales": ypred.flatten()})
print(data)


# In[ ]:




