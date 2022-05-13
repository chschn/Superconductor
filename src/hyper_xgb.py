#!/usr/bin/env python
# coding: utf-8

# # Hyperparameter optimization for XGBoost

# ## Importing the required modules

# In[1]:


import time
import numpy as np
import pandas as pd
from util import read_train_test
from hypopt import xgbhyp


# ## Registering the start time for runtime calculation

# In[2]:


start = time.time()


# ## Reading the training and test data from file

# In[3]:


x_train, x_test, y_train, y_test = read_train_test('../data/sup_dedup_norm_red_tr.csv', '../data/sup_dedup_norm_red_te.csv')


# ## Hyperparameter optimization - XGBoost

# In[4]:


learning_rate_opt, subsample_opt, rmse_xgb = xgbhyp(x_train, x_test, y_train, y_test, learning_rate_min=0.01, learning_rate_max=0.5, learning_rate_n=50, subsample_min=0.5, subsample_max=1.0, subsample_n=51, outfile_log='../hyper/hyperlog_xgb.txt', outfile_dat='../hyper/hyperdata_xgb.csv', outfile_fig='../hyper/hypergraph_xgb.jpg')


# ## Output of elapsed time

# In[ ]:


end = time.time()
runtime = end - start
print(f"Runtime = {runtime:.2f} s")

