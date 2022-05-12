#!/usr/bin/env python
# coding: utf-8

# # Split the data into training and test data - feature reduced

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


# ## Split into training and test data

# In[11]:


tsplit('../data/sup_dedup_norm_red.csv', '../data/sup_dedup_norm_red_tr.csv', '../data/sup_dedup_norm_red_te.csv')


# In[12]:


end = time.time()
runtime = end - start
print(f"Runtime = {runtime:.2f} s")

