#!/usr/bin/env python
# coding: utf-8

# # Hyperparameter optimization for SVM

# ## Importing the required modules

# In[1]:


import time
import numpy as np
import pandas as pd
from util import read_train_test
from hypopt import svmhyp


# ## Registering the start time for runtime calculation

# In[2]:


start = time.time()


# ## Reading the training and test data from file

# In[3]:


x_train, x_test, y_train, y_test = read_train_test('../data/sup_dedup_norm_red_tr.csv', '../data/sup_dedup_norm_red_te.csv')


# ## Hyperparameter optimization - SVM

# In[4]:


kernel_lst = ["linear", "poly", "rbf", "sigmoid"]
c_opt, kernel_opt, rmse_svm = svmhyp(x_train, x_test, y_train, y_test, c_min=0.05, c_max=250, c_n=25, kernels=kernel_lst, outfile_log='../hyper/hyperlog_svm.txt', outfile_dat='../hyper/hyperdata_svm.csv', outfile_fig='../hyper/hypergraph_svm.jpg')


# ## Output of elapsed time

# In[ ]:


end = time.time()
runtime = end - start
print(f"Runtime = {runtime:.2f} s")

