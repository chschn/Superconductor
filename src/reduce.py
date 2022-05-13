#!/usr/bin/env python
# coding: utf-8

# # Feature reduction of the instance cleaned dataset

# ## Importing the required modules

# In[10]:


import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prep import reduce


# ## Registering the start time for runtime calculation

# In[11]:


start = time.time()


# ## Reducing the features with a threshold of 0.7

# In[12]:


reduce('../data/sup_dedup_norm.csv', '../data/sup_dedup_norm_red.csv', 0.7)


# ## Reading the reduced data into dataframe

# In[13]:


sup = pd.read_csv("../data/sup_dedup_norm_red.csv",sep=',',header=0)


# ## Display basic dataset statistics

# In[14]:


sup.describe()


# ## Normalize the attributes, but don't normalize the target

# Normalizing the target would make the interpretation of predictions and errors difficult (wouldn't represent temperature in Kelvin anymore)

# ## Build correlation matrix and take absolute values

# Only the strength of the correlation matters

# In[15]:


crmx = sup.corr().abs()
crmx.style.background_gradient(cmap='YlOrRd')


# In[16]:


with open('../graph/corr_red.html', 'w') as f:
    print(sup.corr().abs().style.background_gradient(cmap='YlOrRd').to_html(), file=f)


# ## Heatmap of the reduced correlation matrix

# In[17]:


colnum = [list(sup.columns).index(i) for i in sup.columns]

fig = plt.figure(figsize=(20,20))
ax = sns.heatmap(crmx, cmap='YlOrRd', annot=True, xticklabels=colnum, yticklabels=colnum)
ax.set_xticklabels(labels=colnum, rotation=90)
ax.set_yticklabels(labels=colnum, rotation=0)
fig.savefig('../graph/Corr_reduced_heatmap.jpg', dpi=150)


# ## Output of elapsed time

# In[18]:


end = time.time()
runtime = end - start
print(f"Runtime = {runtime:.2f} s")

