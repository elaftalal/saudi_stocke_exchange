#!/usr/bin/env python
# coding: utf-8

# # **Read in and Explore the data**

# In[109]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import patsy
from sklearn.metrics import mean_squared_error
from math import sqrt
from datetime import datetime


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[110]:


## Read in the Tarin_data set
stock_train = pd.read_csv('Tadawul_stcks.csv')
stock_train


# In[111]:


## Read in the Test_data set
stock_test = pd.read_csv('Tadawul_stcks_23_4.csv') 
stock_test


# In[112]:


stock_train.info()


# In[95]:


stock_test.info()


# In[96]:


stock_train.shape


# In[97]:


stock_test.shape


# In[98]:


stock_train.describe()


# # **Data Cleaning**

# In[99]:


stock_train.dropna(inplace = True)


# In[100]:


stock_train.open[stock_train.open>47.0]= stock_train.open.median()


# In[101]:


stock_train.describe()


# In[113]:


stock_train['date'] =  pd.to_datetime(stock_train['date'])


# In[114]:


stock_train.info()


# In[116]:


stock_train['year'] = pd. DatetimeIndex(stock_train['date']). year


# In[117]:


stock_train.info()


# In[118]:


stock_train.high[stock_train.high>50]= stock_train.high.median()


# In[119]:


stock_train.low[stock_train.low>47.0]= stock_train.low.median()


# In[120]:


stock_train.close[stock_train.close>55]= stock_train.close.median()


# In[121]:


stock_train.change[stock_train.change>50]= stock_train.close.median()


# In[122]:


stock_train.change[stock_train.change<-50]= stock_train.close.median()


# In[ ]:


stock_train.perc_Change[stock_train.perc_Change>99.9]= stock_train.perc_Change.median()
stock_train.perc_Change[stock_train.perc_Change<-99.9]= stock_train.perc_Change.median()


# In[ ]:


stock_train.columns = stock_train.columns.str.replace(' ', '')


# In[ ]:


stock_train.no_trades[stock_train.no_trades>1500]= stock_train.no_trades.median()


# In[ ]:


stock_train.describe()


# In[ ]:


stock_train.hist(edgecolor='red', linewidth=1.2, figsize=(11, 11));


# # **Data Visualization**

# # •	What is the highest profit Sector?

# In[ ]:


stock_train[['sectoer', 'close']].max()


# .

# # •	What is the most valuable year in the Saudi stock market? 

# In[ ]:


a=stock_train.groupby(['year'])[['high']].sum().sort_values('high', ascending=False)
a


# In[ ]:


plt.figure(figsize=[12,6]);
plt.bar(a.index,a.high);
plt.title('the most valuable year in the Saudi stock market',fontsize=30,color='black',family='serif');
plt.xlabel('year',fontsize=20,color='black',family='serif');
plt.ylabel('high',fontsize=20,color='black',family='serif');
plt.xticks(rotation=45)


# .

# # •	What is the lowiest valuable year in the Saudi stock market?

# In[ ]:


b=stock_train.groupby(['year'])[['low']].sum().sort_values('low')
b


# In[ ]:


plt.figure(figsize=[12,6]);
plt.bar(b.index,b.low);
plt.title('the less valuable year in the Saudi stock market',fontsize=30,color='black',family='serif');
plt.xlabel('year',fontsize=20,color='black',family='serif');
plt.ylabel('high',fontsize=20,color='black',family='serif');


# .

# # •	What is the percentage change in the sector, and what is the highest sector?

# In[ ]:





# # ***predict the stocks prices***

# # ***Regression ***

# ## Fit a linear regression model on the train data set

# In[104]:


#Separate our features from our target

x = stock_train.loc[:,['open', 'high', 'low', 'change',]]
y = stock_train['close']


# In[105]:


x.info


# In[106]:


#Fit on  train dataset
re = LinearRegression()
re.fit(stock_train[['open','high', 'low', 'change']], stock_train['close'])


# In[107]:


#Check the R-squared value
re.score(stock_train[['open', 'high', 'low', 'change']], stock_train['close'])


# In[108]:


# Check the R-squared on test data
re.score(stock_test[['open','high', 'low', 'change']], stock_test['close'])


# ## **R-squared**

# In[ ]:


# Compare the actual y values in the training set with the predicted values
actual_train = stock_train['close']
predicted_train = re.predict(stock_train[['open','high', 'low', 'change']])
predicted_train.shape


# In[ ]:


# Check the RMSE (root mean squared error) on the training data
sqrt(mean_squared_error(actual_train, predicted_train))


# In[ ]:


# Compare the actual y values in the test set with the predicted values
actual_test = stock_test['close']
predicted_test = re.predict(stock_test[['open','high', 'low', 'change']])
predicted_test.shape

