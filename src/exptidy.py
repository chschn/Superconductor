#!/usr/bin/env python
# coding: utf-8

# # Explorative analysis on cleaned and normalized instances

# ## Importing the required modules

# In[7]:


from math import sqrt
import time
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import ppscore as pps


# ## Registering the start time for runtime calculation

# In[8]:


start = time.time()


# ## Reading the data file into the Dataframe - deduplicated and normalized

# In[9]:


sup = pd.read_csv("../data/sup_dedup_norm.csv",sep=',',header=0)
sup


# ## Display basic stats

# In[10]:


sup.describe()


# ## Display boxplot of all attributes without target

# In[22]:


split = 41
endr = len(sup.columns) - 1

fig = plt.figure()
fig.subplots_adjust(hspace=2.0)
fig = plt.figure(figsize=(15,10))
ax1 = fig.add_subplot(2, 1, 1)
ax1.set_title("Boxplot after normalization")
ax1.boxplot(sup.iloc[:,0:split],labels=range(0,split),medianprops=dict(color="#1ACC94"))
ax1.set_xticklabels(labels=range(0,split), rotation=90)
ax2 = fig.add_subplot(2, 1, 2)
ax2.set_title("Boxplot after normalization")
ax2.boxplot(sup.iloc[:,split:endr],labels=range(split,endr),medianprops=dict(color="#1ACC94"))
ax2.set_xticklabels(labels=range(split,endr), rotation=90)
fig.savefig('../graph/Box_norm.jpg')


# In[23]:


endr = len(sup.columns) - 1

fig = plt.figure()
fig.subplots_adjust(hspace=2.0)
fig = plt.figure(figsize=(18,4))
ax1 = fig.add_subplot(1, 1, 1)
ax1.set_title("Boxplot after normalization")
ax1.boxplot(sup.iloc[:,0:endr],labels=range(0,endr),medianprops=dict(color="#1ACC94"))
ax1.set_xticklabels(labels=range(0,endr), rotation=90)
fig.savefig('../graph/Box_norm_one.jpg')


# ## Display color coded absolute value correlation matrix

# In[13]:


crmx = sup.corr().abs()
crmx.style.background_gradient(cmap='YlOrRd')


# In[14]:


colnum = list(range(0,len(crmx.columns)))

fig = plt.figure(figsize=(20,20))
ax = sns.heatmap(crmx, cmap='YlOrRd', xticklabels=colnum, yticklabels=colnum)
ax.set_xticklabels(labels=colnum, rotation=90)
ax.set_yticklabels(labels=colnum, rotation=0)
fig.savefig('../graph/Corr_norm_heatmap.jpg')


# ## Display color coded absolute value correlation matrix of selected attributes

# Each property has 10 statistical mesures. Selecting one property as a starting point (ix = [i+j*10 for i in range(1,11) with j between 0 and 7)

# Alternatively select the same statistical measure for each property (ix = [i+j for i in range(1,81,10) with j between 0 and 7)

# In[15]:


ix = [i for i in range(1,81,10)]
ix.append(-1)
sup.iloc[:,ix].corr().abs().style.background_gradient(cmap='YlOrRd')


# In[16]:


colnum = [list(sup.columns).index(i) for i in sup.iloc[:,ix].columns]

fig = plt.figure(figsize=(10,10))
ax = sns.heatmap(sup.iloc[:,ix].corr().abs(), cmap='YlOrRd', annot=True, xticklabels=colnum, yticklabels=colnum)
ax.set_xticklabels(labels=colnum, rotation=90)
ax.set_yticklabels(labels=colnum, rotation=0)
fig.savefig('../graph/Corr_norm_select_heatmap_means.jpg')


# ## Display scatter matrix for selected attributes plus target

# Display of all 82 attributes exceeds computing resources

# Each property has 10 statistical mesures. Selecting one property as a starting point (ix = [i+j*10 for i in range(1,11) with j between 0 and 7)

# Alternatively select the same statistical measure for each property (ix = [i+j for i in range(1,81,10) with j between 0 and 7)

# In[17]:


pd.plotting.scatter_matrix(sup.iloc[:,ix], figsize=(30,30), c="#1ACC94")
plt.savefig('../graph/Scatter_norm_select_means.jpg')


# ## Display PPS Matrix

# In[18]:


pps.matrix(sup.iloc[:,ix], output='list', sample=0)


# In[19]:


predictors_df = pps.predictors(sup.iloc[:,ix], y="critical_temp")
fig = plt.figure(figsize=(15,7))
sns.barplot(data=predictors_df, x="x", y="ppscore")
plt.xticks(rotation=15)
plt.savefig('../graph/PPS_norm_predictors_means')


# In[20]:


matrix_df = pps.matrix(sup.iloc[:,ix])[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')
fig = plt.figure(figsize=(12,12))
sns.heatmap(matrix_df, vmin=0, vmax=1, cmap="YlOrRd", linewidths=0.5, annot=True)
plt.savefig('../graph/PPS_norm_matrix_means')


# In[21]:


end = time.time()
runtime = end - start
print(f"Runtime = {runtime:.2f} s")


# In[ ]:




