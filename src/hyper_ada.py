#!/usr/bin/env python
# coding: utf-8

# # Hyperparameter optimization for Ada Boost

# ## Importing the required modules

# In[1]:


import time
import numpy as np
import pandas as pd
from util import read_train_test
from hypopt import adahyp


# ## Registering the start time for runtime calculation

# In[2]:


start = time.time()


# ## Reading the training and test data from file

# In[3]:


x_train, x_test, y_train, y_test = read_train_test('../data/sup_dedup_norm_red_tr.csv', '../data/sup_dedup_norm_red_te.csv')


# ## Hyperparameter optimization - Ada Boost

# In[4]:


learning_rate_opt, n_estimators_opt, rmse_ada = adahyp(x_train, x_test, y_train, y_test, learning_rate_min=0.05, learning_rate_max=2.5, learning_rate_n=50, n_estimators_min=2, n_estimators_max=102, n_estimators_n=51, outfile_log='../hyper/hyperlog_ada.txt', outfile_dat='../hyper/hyperdata_ada.csv', outfile_fig='../hyper/hypergraph_ada.jpg')


# ## Output of elapsed time

# In[ ]:


end = time.time()
runtime = end - start
print(f"Runtime = {runtime:.2f} s")

