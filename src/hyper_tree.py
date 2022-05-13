#!/usr/bin/env python
# coding: utf-8

# # Hyperparameter optimization for Decision Tree

# ## Importing the required modules

# In[1]:


import time
import numpy as np
import pandas as pd
from util import read_train_test
from hypopt import treehyp


# ## Registering the start time for runtime calculation

# In[2]:


start = time.time()


# ## Reading the training and test data from file

# In[3]:


x_train, x_test, y_train, y_test = read_train_test('../data/sup_dedup_norm_red_tr.csv', '../data/sup_dedup_norm_red_te.csv')


# ## Hyperparameter optimization - Decision Tree

# In[4]:


max_depth_tr_opt, min_samples_tr_opt, rmse_tr = treehyp(x_train, x_test, y_train, y_test, max_depth_min=1, max_depth_max=201, max_depth_n=101, min_samples_min=2, min_samples_max=102, min_samples_n=101, outfile_log='../hyper/hyperlog_dt.txt', outfile_dat='../hyper/hyperdata_dt.csv', outfile_fig='../hyper/hypergraph_dt.jpg')


# ## Output of elapsed time

# In[5]:


end = time.time()
runtime = end - start
print(f"Runtime = {runtime:.2f} s")

