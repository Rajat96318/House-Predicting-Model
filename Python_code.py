#!/usr/bin/env python
# coding: utf-8

# In[2]:


#importing the modules needed
import pandas as pd
import numpy as np
#Pandas provide high performance, fast, easy to use data structures and data analysis tools for manipulating numeric data and time series.
# Pandas is built on the numpy library and written in languages like Python, Cython, and C.
import seaborn as sns      #Python data visualization library, The graphs created can also be customized easily
import matplotlib.pyplot as plt   
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


# In[12]:


df=pd.read_csv("C:\Boston Dataset.csv")
df.drop(columns=['Unnamed: 0'], axis=0, inplace=True) #delecting non required column
df.head()


# In[14]:


#gives the statistical inforamtion
df.describe()


# In[16]:


df.info()
#gives the datatype information, all values are either in float or integer


# In[17]:


#preprocessing the data
#check for null values
df.isnull().sum()


# In[28]:


# creatring box plots from the 14 attributes
# use the identify the outlayers from the dataset
# box plot= a simple way of representing statistical data on a plot (graph)

fig, ax = plt.subplots(ncols=7, nrows=2, figsize=(20,10))
#we have 14 attributes 7*2=14, figsize(width,height)
index=0    
ax = ax.flatten()
 
for col, value in df.items():
    #df.items() return colums and the corresponding values
    sns.boxplot(y=col, data= df , ax=ax[index])
    #for each index we are creating the plot
    index +=1
    
# the plot we get now are overlapping

plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)
#after this no overlapping in plot 


# In[31]:


fig, ax = plt.subplots(ncols=7, nrows=2, figsize=(20,10))
index=0    
ax = ax.flatten()

for col, value in df.items():
    #A Distplot or distribution plot, depicts the variation in the data distribution
    sns.distplot(value, ax=ax[index]) 
    index +=1
    
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)

#if the plot is on left we say left skewwed plot
#if the plot is on right we say right skewwed plot
#last two plots are uniformly distributed


# In[46]:


# doing min max normalization
# creating column list
## IF outliers are more we do min max normaliztion, if less outliers we ignore them
# if the values have so much diffrence then we use min max normaliztion
cols = ['crim', 'zn', 'tax', 'black']
for col in cols:
    #find min and max of that column
    minimum = min(df[col])
    maximum = max(df[col])
    df[col] = (df[col] - minimum) / (maximum - minimum)
    

#after doing min max normaliztion we will get range 0 to 1


# In[47]:


# agin printing to see the changes done to make range 0 to 1

fig, ax = plt.subplots(ncols=7, nrows=2, figsize=(20,10))
index=0    
ax = ax.flatten()

for col, value in df.items():
    #A Distplot or distribution plot, depicts the variation in the data distribution
    sns.distplot(value, ax=ax[index]) 
    index +=1
    
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)


# In[52]:


#standardization when we have uniform distribution
#it uses mean and standard deviation to create an standard score

from sklearn import preprocessing
scalar = preprocessing.StandardScaler()

#fit our data
scaled_cols = scalar.fit_transform(df[cols])
scaled_cols = pd.DataFrame(scaled_cols,columns=cols)
scaled_cols.head()


# In[53]:


for col in cols:
    df[col]= scaled_cols[col]


# In[54]:


# agin printing to see the changes done by standardization

fig, ax = plt.subplots(ncols=7, nrows=2, figsize=(20,10))
index=0    
ax = ax.flatten()

for col, value in df.items():
    #A Distplot or distribution plot, depicts the variation in the data distribution
    sns.distplot(value, ax=ax[index]) 
    index +=1
    
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)




# In[59]:


#correlation matrix
#A correlation matrix is simply a table which displays the correlation
corr = df.corr()
#to improve fig size
plt.figure(figsize=(20,10))
sns.heatmap(corr, annot=True, cmap='coolwarm')


# In[60]:


sns.regplot(y=df['medv'], x=df['lstat'])
#regplot() : This method is used to plot data and a linear regression model fit
#price drecreses when the lstat is incresing


# In[61]:


sns.regplot(y=df['medv'], x= df['rm'])
#price of houses increses when the rm increses


# In[62]:


#doing regression part
X = df.drop(columns=['medv','rad'], axis=1)
y= df['medv']


# ## Now training the data

# In[71]:


from sklearn.model_selection import cross_val_score, train_test_split
# using cross_val_score we will determine which model is performing better
from sklearn.metrics import mean_squared_error
def train(model, X, y):
    #train the model
    # X is input attribute, y is output attribute
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model.fit(x_train, y_train)
    
    #predict the training set
    pred = model.predict(x_test)
    
    #performing cross-validation
    cv_score = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)
    #gives cv_score in negative
    cv_score = np.abs(np.mean(cv_score))
    
    print("Model Report")
    print("MSE:",mean_squared_error(y_test,pred))
    print('CV score:', cv_score)


# In[78]:


from sklearn.linear_model import LinearRegression
model = LinearRegression(normalize=True)
train(model, X, y)
coef = pd.Series(model.coef_, X.columns).sort_values()
coef.plot(kind='bar', title='Model Coefficients')
#using this to display the plot


# In[ ]:




