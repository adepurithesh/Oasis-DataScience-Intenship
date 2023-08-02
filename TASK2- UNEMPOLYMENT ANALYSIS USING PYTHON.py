#!/usr/bin/env python
# coding: utf-8

# In[1]:


#load require liabraries of python
get_ipython().system('pip install plotly')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[2]:


df = pd.read_csv('Unemployment_Rate_upto_11_2020.csv') #Read dataset
df.sample(5)  #fetch five sample of dataset


# In[3]:


df.columns


# In[4]:


df[' Frequency'].value_counts()


# In[5]:


print(df['Region.1'].value_counts())
print(df['Region'].value_counts())


# In[6]:


df.isnull().sum()


# In[7]:


df.duplicated().sum()


# In[8]:


print('row count--->',df.shape[0])
print('column count--->',df.shape[1])


# In[9]:


df.dtypes


# In[10]:


df[["day", "month", "year"]] = df[' Date'].str.split("-", expand = True)
df


# In[11]:


df.drop(columns=[' Frequency'],axis=1,inplace=True)


# In[12]:


df[:5]


# In[13]:


import matplotlib.pyplot as plt
plt.figure(figsize=(5,5))
sns.heatmap(df.corr(),annot=True)
plt.show()
     


# Heatmap gives me correlation matrix in grapical format.It also gives us correlation between each independent feature. From the above visualisation we can say that,Estimated Labour Participation Rate,latitude having highly positive correlated feat i.e 40% and estimated employed,estimated unemployment rate is highly negative correlated feat. They may play very importtant role in the analysis.
# 
# when the independent and dependent feat are correlated with each other then it can be say that they will perform better while training model but when independent are highly correlerated with independent feat it may act as duplicate,so insted of use them in training we must remove them forever for better performance of model.this is the part of feature selection.

# In[14]:


df.columns


# In[15]:


plt.figure(figsize=(10,10))
plt.title("Unemployment in india")
sns.histplot(x=' Estimated Unemployment Rate (%)',hue= "Region", data=df,kde=False)
plt.show()
     


# The above histplot shows regionwise count of Estimated Unemployment Rate(%)

# In[16]:


df.columns


# In[17]:


df.month.unique()


# In[18]:


sns.barplot(x='month',y=' Estimated Unemployment Rate (%)',hue='year',data=df)


# In 2020 pendamic year of covid-19,5th month has the maximum unemployment rate approximatetly near 23-23.5% and minimum rate of month is 10th which is 8 to 8.5%

# In[19]:


df.day.unique()


# In[20]:


sns.barplot(x='day',y=' Estimated Unemployment Rate (%)',hue='year',data=df)


# from the above bar plot we have seen monthwise rate now it's time to check daywise rate and it is clear to see that day 30th which is nearly a month end date having the maximum peoples lost their job

# In[21]:


import seaborn as sns
sns.set_theme(style="ticks", palette="pastel")

# Draw a nested boxplot to show bills by day and time
sns.boxplot(x="month", y=' Estimated Employed', palette=["m", "g"],
            data=df)
sns.despine(offset=10, trim=True)


# Boxplot shows the quantile monthly data distribution.Here it shows that except 4th and 5th month peoples get job rate not affected so much during 2020 pandemic

# In[22]:


df[:5]


# we can also drop this year column,beacause it also contains constant values of year 2020

# In[23]:


df.drop('year',axis=1)


# In[24]:


plt.figure(figsize=(10,10))
plt.title("Unemployment in india")
sns.barplot(x='month',y =' Estimated Unemployment Rate (%)',hue='Region.1', data=df)
plt.show()


# In[25]:


plt.figure(figsize=(10,10))
plt.title("Unemployment in india")
sns.barplot(x='day',y =' Estimated Unemployment Rate (%)',hue='Region.1', data=df)
plt.show()


# In[26]:


unemploment = df[["Region",' Estimated Unemployment Rate (%)']]
figure = px.sunburst(unemploment, path=["Region"], 
                     values=' Estimated Unemployment Rate (%)',
                     width=700, height=700, color_continuous_scale="RdY1Gn", 
                     title="Unemployment Rate in India")
figure.show()   


# In[27]:


df.columns


# In[28]:


unemploment = df[["Region.1",' Estimated Employed']]
figure = px.sunburst(unemploment, path=["Region.1"], 
                     values=' Estimated Employed',
                     width=700, height=700, color_continuous_scale="RdY1Gn", 
                     title="employment Rate in India")
figure.show()


# In[ ]:




