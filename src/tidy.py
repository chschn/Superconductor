#!/usr/bin/env python
# coding: utf-8

# # Instances cleanup

# ## Importing the required modules

# In[7]:


import time
import numpy as np
import pandas as pd
from prep import dedup, norm, reduce
from util import tsplit, read_train_test
from regr import linreg, knnreg, treereg, randreg, adareg, svmreg, votreg, xgbreg
from hypopt import knnhyp, treehyp, randhyp, adahyp, svmhyp, xgbhyp


# ## Registering the start time for runtime calculation

# In[8]:


start = time.time()


# ## Reading the data file into the Dataframe

# In[9]:


sup = pd.read_csv("../data/sup.csv",sep=',',header=0)
sup


# ## Deduplicate the instances

# In[10]:


dedup('../data/sup.csv', '../data/sup_dedup.csv')


# ## Normalize the features

# In[11]:


norm('../data/sup_dedup.csv', '../data/sup_dedup_norm.csv')


# In[12]:


end = time.time()
runtime = end - start
print(f"Runtime = {runtime:.2f} s")

