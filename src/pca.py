#!/usr/bin/env python
# coding: utf-8

# # Principle Component Analysis

# ## Importing the required modules

# In[20]:


from math import sqrt
import time
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from prep import dedup, norm, reduce
from util import tsplit, read_train_test
from regr import linreg, knnreg, treereg, randreg, adareg, svmreg, votreg, xgbreg
from hypopt import knnhyp, treehyp, randhyp, adahyp, svmhyp, xgbhyp


# ## Registering the start time for runtime calculation

# In[21]:


start = time.time()


# ## Read the tidy training and test datasets from file

# In[22]:


x_train, x_test, y_train, y_test = read_train_test('../data/sup_norm_tr.csv', '../data/sup_norm_te.csv')


# ## Perform the principal components fit on the training data only

# Set the parameter n_components to cut off the components beyond some threshold

# If n_components is a float between 0 and 1 it represents the conserved fraction of variance instead

# In[23]:


comp_it = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.98,0.99,0.999,0.9995, 0.9999, 0.99995, 0.99999, 0.999995, 0.999999, 0.9999995, 0.999999999999]
cpca = []

for comp_i in comp_it:
    pca = PCA(n_components=comp_i)
    pca.fit(x_train)
    pc_train = pca.transform(x_train)
    pc_train = pd.DataFrame(data = pc_train)
    cpca.append(len(pc_train.columns))
    


# In[24]:


fig = plt.figure(figsize=(10,6))
plt.plot(comp_it, cpca)
plt.xlabel("$C_{comp}$", fontdict={'fontsize': 14})
plt.ylabel("principal components", fontdict={'fontsize': 14})
plt.grid()
plt.xlim(0.8, 1.01)
fig.savefig('../graph/Pca_Ccomp.jpg')


# In[25]:


pca = PCA(n_components=0.93)
pca.fit(x_train)


# ## Perform the principal components transformation on the training data

# Also convert into a pandas DataFrame

# In[26]:


cl_raw = x_train.columns
pc_train = pca.transform(x_train)
x_train = pd.DataFrame(data = pc_train)


# Display the transformed training data

# In[27]:


x_train


# ## Check the sum of principal component variances (should be slightly above threshold)

# In[28]:


pca.explained_variance_ratio_.sum()


# ## Display mapping matrix of principal components

# In[29]:


pcs_train = pd.DataFrame(data=pca.components_.reshape(len(x_train.columns),-1), columns=cl_raw)
pcs_train


# ## Heatmap of PCA mapping matrix

# In[30]:


colnum = list(range(0,len(pcs_train.columns)))

fig = plt.figure(figsize=(20,4))
ax = sns.heatmap(pcs_train, cmap='RdBu_r', annot=False)
ax.set_title("PCA - feature mapping", fontdict={'fontsize': 20})
ax.set_xticklabels(labels=colnum, rotation=90)
ax.set_yticklabels(labels=pcs_train.index, rotation=0)
ax.set_xlabel("orig. features", fontdict={'fontsize': 14})
ax.set_ylabel("principal components", fontdict={'fontsize': 14})
fig.savefig('../graph/Pca_mapping_heatmap.jpg')


# In[31]:


fig = plt.figure(figsize=(12,6))
plt.bar(pcs_train.index + 1, pca.explained_variance_ratio_, color = "blue")
plt.title("PCA vs. explained variance", fontdict={'fontsize': 20})
plt.xlim(0.0,14)
plt.xticks(list(range(1,14)))
plt.xlabel("principal components", fontdict={'fontsize': 14})
plt.ylabel("explained variance", fontdict={'fontsize': 14})
plt.annotate(f"explained ratio sum = {pca.explained_variance_ratio_.sum():.4f}", xy=(9,0.4), xycoords='data')
fig.savefig('../graph/Pca_explained_var.jpg')


# ## Heatmap of the correlation matrix of training after PCA

# In[32]:


crmx_red = x_train.corr().abs()


# In[33]:


fig = plt.figure(figsize=(20,20))
sns.heatmap(crmx_red, cmap='YlOrRd', annot=True, fmt='.2f')


# ## Perform the principal components transformation on the test data

# In[34]:


pc_test = pca.transform(x_test)
x_test = pd.DataFrame(data = pc_test)


# Display the transformed test data

# In[35]:


x_test


# ## Aggregate x and y into one dataframe for training and test each

# In[36]:


train = x_train.assign(critical_temp=y_train)
test = x_test.assign(critical_temp=y_test)


# ## Export resulting dataframes into csv files

# In[37]:


train.to_csv('../data/sup_pca_tr.csv', index=False)
test.to_csv('../data/sup_pca_te.csv', index=False)


# In[38]:


end = time.time()
runtime = end - start
print(f"Runtime = {runtime:.2f} s")

