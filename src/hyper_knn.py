#!/usr/bin/env python
# coding: utf-8

# # Hyperparameter optimization for kNN

# ## Importing the required modules

# In[14]:


import time
import numpy as np
import pandas as pd
from util import read_train_test
from hypopt import knnhyp


# ## Registering the start time for runtime calculation

# In[15]:


start = time.time()


# ## Reading the training and test data from file

# In[16]:


x_train, x_test, y_train, y_test = read_train_test('../data/sup_dedup_norm_red_tr.csv', '../data/sup_dedup_norm_red_te.csv')


# ## Hyperparameter optimization - kNN

# In[17]:


kopt, rmse_knn = knnhyp(x_train, x_test, y_train, y_test, kmin=1, kmax=20, outfile_log='../hyper/hyperlog_knn.txt', outfile_dat='../hyper/hyperdata_knn.csv', outfile_fig='../hyper/hypergraph_knn.jpg')


# ## Output of elapsed time

# In[18]:


end = time.time()
runtime = end - start
print(f"Runtime = {runtime:.2f} s")

