#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()


# In[15]:


raw_data = pd.read_csv('1.03. Dummies.csv')


# In[16]:


raw_data


# In[18]:


data = raw_data.copy()


# In[19]:


data['Attendance'] = data['Attendance'].map({'Yes':1, 'No':0})


# In[20]:


data


# In[21]:


data.describe()


# In[26]:


## Regression


# In[27]:


y = data['GPA']
x1 = data[['SAT','Attendance']]


# In[31]:


x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
results.summary()


# In[37]:


plt.scatter(data['SAT'],y, c=data['Attendance'], cmap='RdYlGn_r')
yhat_no = 0.6439 + 0.0014 * data['SAT']
yhat_yes = 0.6439 + 0.2226 + 0.0014 * data['SAT']
yhat = 0.0017 * data['SAT'] + 0.275
fig = plt.plot(data['SAT'], yhat_no, lw=2, c='red', label='regression line 1')
fig = plt.plot(data['SAT'], yhat_yes, lw=2, c='orange', label ='regression line 2')
fig = plt.plot(data['SAT'], yhat, lw=3, c='blue', label ='regressionline')
plt.xlabel('SAT', fontsize=20)
plt.ylabel('GPA', fontsize=20)
plt.show()


# In[ ]:




