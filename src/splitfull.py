#!/usr/bin/env python
# coding: utf-8

# # Predicting superconductor critical temperatures

# ## Importing the required modules

# In[8]:


from math import sqrt
import time
import numpy as np
import pandas as pd
from prep import norm
from util import tsplit


# ## Registering the start time for runtime calculation

# In[9]:


start = time.time()


# ## Normalizing the features

# In[10]:


norm('../data/sup.csv', '../data/sup_norm.csv')


# ## Split into training and test data

# In[11]:


tsplit('../data/sup_norm.csv', '../data/sup_norm_tr.csv', '../data/sup_norm_te.csv')


# In[12]:


end = time.time()
runtime = end - start
print(f"Runtime = {runtime:.2f} s")

