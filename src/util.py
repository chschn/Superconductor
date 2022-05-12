#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Split dataset into training and test dataset and write into separate csv-files
#
def tsplit(infile, out_train, out_test, test_size=None):
    """ 
    Split data into training and test datasets. Read and write from file.
    
    parameter:
    
        infile    :   name of input csv file with full dataset
        out_train :   name of output csv file for training data
        out_test  :   name of output csv file for test data
        
    returns:
        None
        
    outputs:
        screen output with stats
        output files for training and test datasets each
        
    """
    inpath = "./" + infile
    outpath_train = "./" + out_train
    outpath_test = "./" + out_test
    
    # read the contents of infile into dataframe "df" 
    df = pd.read_csv(inpath, sep=',', header=0)
    fullc = df.shape[0]
    
    # split the data into training and test data
    x_train, x_test, y_train, y_test = train_test_split(df.drop(["critical_temp"],axis=1),df["critical_temp"],test_size=test_size)
    
    # aggregate x and y into one dataframe for training and test each
    train = x_train.assign(critical_temp=y_train)
    test = x_test.assign(critical_temp=y_test)
    trainc = train.shape[0]
    testc = test.shape[0]
    smpl = testc / trainc * 100
    
    # export resulting dataframes into csv files out_train and out_test
    train.to_csv(outpath_train, index=False)
    test.to_csv(outpath_test, index=False)
    
    # screen output
    print(f"{fullc} instances split into {trainc} training instances and {testc} test instances -> {smpl:.2f}%")
    
    
# Read training and test data from csv files and distribute into x_train, y_train, x_test and y_test
#
def read_train_test(in_train, in_test):
    """ 
    Read training and test datasets from file. Put the data into x_train, x_test, y_train, y_test.
    
    parameter:
    
        in_train :   name of input csv file for training data
        in_test  :   name of input csv file for test data
        
    returns:
        x_train  :   pandas data frame with features for training
        x_test   :   padas data frame with features for testing
        y_train  :   pandas series with target for training
        y_test   :   pandas series with target for testing
        
    outputs:
        screen output with stats
        
    """    
    inpath_train = "./" + in_train
    inpath_test = "./" + in_test

    # read the contents of files in_train and in_test into dataframes train and test 
    train = pd.read_csv(inpath_train, sep=',', header=0)
    test = pd.read_csv(inpath_test, sep=',', header=0)
    
    # split into x and y
    return train.iloc[:,:-1], test.iloc[:,:-1], train.iloc[:,-1:].squeeze(), test.iloc[:,-1:].squeeze()