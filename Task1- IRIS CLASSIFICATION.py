#!/usr/bin/env python
# coding: utf-8

# In[5]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import xgboost as xg


# In[6]:


#load and make the copy of Iris dataset to keep track of changes.
df = pd.read_csv('Iris.csv') #read comma seperated values
df_copy=df.copy() #copy dataset


# In[7]:


#fetch first five rows from dataset
df_copy


# # Check data properties and Data cleaning or preprocessing

# In[8]:


#Remove unnecessary feat from dataset Id
df_copy.drop(columns=['Id'],axis=0,inplace=True)
#Check datatypes of each feat
df_copy.dtypes


# In[9]:


#check number of records and feilds present in dataset
df_copy.shape
print('Rows ---->',df.shape[0])
print('Columns ---->',df.shape[1])


# In[10]:


#see the descriptive statistics
df_copy.describe()


# In[11]:


#check the space complexicity taken by data
df_copy.size
#checking if there is any inconsistency in the dataset
#as we see there are no null values in the dataset, so the data can be processed
df_copy.info()


# In[12]:


df_copy.columns = ['sl','sw','pl','pw','species']
df_split_iris=df_copy.species.str.split('-',n=-1,expand=True) #Remove prefix 'Iris-' from species col
df_split_iris.drop(columns=0,axis=1,inplace=True)#Drop 'Iris-' col
df_split_iris


# In[13]:


df3_full=df_copy.join(df_split_iris)
df3_full


# In[14]:


df3_full.rename({1:'species1'},axis=1,inplace=True) #Rename column
df3_full


# In[15]:


df3_full.drop(columns='species',axis=1,inplace=True) #Drop excessive column


# In[16]:


#final dataframe
df3_full


# In[17]:


df3_full.shape #check propertise like shape


# In[18]:


#check for missing entries
df3_full.isna() 


# In[19]:


#In each feat,count of missing entries
df3_full.isna().sum()


# In[20]:


df3_full.corr() # check the correlation matrix


# In[21]:


#statistical description of numerical  data only
df3_full.describe()


# check for balanced or unbalanced?

# In[22]:


#categoriwise frequency of data
df3_full.species1.value_counts()


# # Data Visualisation and gain meaningfull insights from data

# In[23]:


sns.scatterplot(x=df3_full.sl,y=df3_full.pl,hue=df3_full.species1)


# The above graph shows relationship between the sepal length and width.It also shows that species setosa having lesser petal lenght as compared to versicolor and verginica.
# Now we will check relationship between the petal length and width.

# In[24]:


sns.scatterplot(x=df3_full.pl,y=df3_full.pw,hue=df3_full.species1)


# As we can see that the Petal Features are giving a better cluster division compared to the Sepal features. This is an indication that the Petals can help in better and accurate Predictions over the Sepal. We will check that later.

# In[27]:


sns.scatterplot(x=df3_full.sl,y=df3_full.pw,hue=df3_full.species1)


# In[28]:


sns.scatterplot(x=df3_full.sw,y=df3_full.pl,hue=df3_full.species1)


# From both above scatter plots we can clearly see that sw and pl data distribution and not represents specific pattern

# In[29]:


df3_full.columns # check column names


# In[30]:


sns.distplot(df3_full.pl)


# In[31]:


sns.distplot(df3_full.sl)


# In[32]:


sns.distplot(df3_full.sw)


# In[33]:


sns.distplot(df3_full.pw)


# In[34]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='species1',y='pl',data=df3_full)
plt.subplot(2,2,2)
sns.violinplot(x='species1',y='pw',data=df3_full)
plt.subplot(2,2,3)
sns.violinplot(x='species1',y='sl',data=df3_full)
plt.subplot(2,2,4)
sns.violinplot(x='species1',y='sw',data=df3_full)


# The violinplot shows density of the length and width in the species. The thinner part denotes that there is less density whereas the fatter part conveys higher density

# In[35]:


plt.figure(figsize=(7,4)) 
sns.heatmap(df3_full.corr(),annot=True,cmap='cubehelix_r') 
plt.show()


# Observation--->
# 
# The Sepal Width and Length are not correlated The Petal Width and Length are highly correlated
# 
# We will use all the features for training the algorithm and check the accuracy.
# 
# Then we will use 1 Petal Feature and 1 Sepal Feature to check the accuracy of the algorithm as we are using only 2 features that are not correlated. Thus we can have a variance in the dataset which may help in better accuracy. We will check it later.

# # convert categorical features into numerical feature

# In[36]:


from sklearn.preprocessing import LabelEncoder
 
# Creating a instance of label Encoder.
le = LabelEncoder()
 
# Using .fit_transform function to fit label
# encoder and return encoded label
le.fit_transform(df3_full['species1'])
df3_full['species1']=le.fit_transform(df3_full['species1'])
df3_full


# In[37]:


df3_full


# # Divide independent feat and target feat to train model

# In[38]:


x = df3_full.iloc[:,:-1]
x


# In[39]:


y = df3_full.iloc[:,-1]
y


# In[40]:


df3_full.species1.unique()


# In[41]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3,random_state=20)


# In[42]:


xtrain.shape


# In[43]:


ytrain.shape


# In[44]:


xtest.shape


# In[45]:


ytest.shape


# # Train model with LR,DT,RFC of classification techniques

# In[47]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
logi = LogisticRegression()
logi.fit(xtrain,ytrain)
logi_prediction = logi.predict(xtest)
logi_prediction


# In[48]:


print(logi.score(xtrain,ytrain)*100)
print(logi.score(xtest,ytest)*100)


# In[49]:


accuracy_score(ytest,logi_prediction)*100


# In[50]:


from sklearn.model_selection import GridSearchCV


# In[51]:


para = {'penalty':['l1','l2','elasticnet'],
        'C':[1,2,3,4,5,6,10,20,30,40,50,1.5,2.3,1.6,1.9],
        'max_iter':[100,200,300,50,70,60,50]
        }


# In[52]:


classifier_logistic = GridSearchCV(logi,param_grid = para,scoring='accuracy',cv=5)


# In[73]:


classifier_logistic.fit(xtrain,ytrain)

In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
# In[74]:


classifier_logistic.best_estimator_


# In[76]:


classifier_logistic.best_params_



# In[77]:


classifier_logistic.best_score_


# In[78]:


prediction = classifier_logistic.predict(xtest)
prediction


# In[79]:


from sklearn.metrics import accuracy_score,classification_report
grid_logi_accuracy_score1 = accuracy_score(ytest,prediction)
grid_logi_accuracy_score1=(np.round(grid_logi_accuracy_score1*100))
grid_logi_accuracy_score1


# In[80]:


confusion_matrix(ytest,prediction)


# In[81]:


class_pre_rec = classification_report(ytest,prediction)
print(class_pre_rec)


# In[82]:


from sklearn.tree import DecisionTreeClassifier


# In[83]:


tree_classifier = DecisionTreeClassifier(criterion='gini',
    splitter='best', 
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=1,
    random_state=1,
    max_leaf_nodes=2,
    class_weight='balanced',
    ccp_alpha=0.01,)


# In[84]:


tree_classifier.fit(xtrain,ytrain)


# In[85]:


tree_classifier.score(xtrain,ytrain)


# In[86]:


tree_classifier.score(xtest,ytest)


# In[87]:


tree_classifier.predict(xtest)


# In[88]:


tree_pred=tree_classifier.predict(xtest)


# In[89]:


from sklearn.metrics import confusion_matrix,classification_report
confusion_matrix(ytest,tree_pred)


# In[90]:


print(classification_report(ytest,tree_pred))


# In[91]:


import sklearn
sklearn.metrics.get_scorer_names()


# In[92]:


param_dict = {"criterion":['gini','entropy'],"max_depth":[1,2,3,4,5,6,7,None]}


# In[93]:


from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(tree_classifier,param_grid=param_dict,n_jobs=-1)
grid


# In[94]:


grid.fit(xtrain,ytrain)


# In[96]:


grid.best_params_



# In[97]:


grid.best_score_


# In[99]:


grid_pred2=grid.predict(xtest)


# In[102]:


accuracy_score(ytest,grid_pred2)*100


# In[103]:


from sklearn.metrics import confusion_matrix,classification_report
confusion_matrix(ytest,grid_pred2)


# In[104]:


print(classification_report(ytest,grid_pred2))


# In[ ]:




