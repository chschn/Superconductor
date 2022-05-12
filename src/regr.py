#!/usr/bin/env python
# coding: utf-8

from math import sqrt
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor, VotingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

# Linear regression
#
def linreg(x_train, x_test, y_train, y_test, grd=1):
    """ 
    Linear Regression wrapper
    
    parameter:
    
        x_train  :   pandas data frame with features for training
        x_test   :   padas data frame with features for testing
        y_train  :   pandas series with target for training
        y_test   :   pandas series with target for testing

        grd=1 :  polynomial grade of fitted function
        
    returns:
        lin       :  linear regression model
        lin_pred  :  predictions
        rmse_lin  :  prediction RMSE value
        
    outputs:
        screen output of the prediction RMSE value
        
    """    
    # polynomial transformation
    poly = PolynomialFeatures(grd)
    x_poly_train = poly.fit_transform(x_train)
    x_poly_test = poly.fit_transform(x_test)
    
    # linear regression training
    lin = LinearRegression()
    lin.fit(x_poly_train,y_train)
    
    # prediction
    lin_pred = lin.predict(x_poly_test)
    
    # calculation of RMSE
    rmse_lin = sqrt(mean_squared_error(lin_pred, y_test))
    
    # screen output
    print(f"Linear Regression -> RMSE = {rmse_lin:.4f}")
    
    # return values
    return lin, lin_pred, rmse_lin


# kNN regression
#
def knnreg(x_train, x_test, y_train, y_test, k=1):
    """ 
    k Nearest Neighbor Regression wrapper
    
    parameter:
    
        x_train  :   pandas data frame with features for training
        x_test   :   padas data frame with features for testing
        y_train  :   pandas series with target for training
        y_test   :   pandas series with target for testing
        
        k=1 :  number of nearest neighbors
    
    note:
        The option weights='distance' is being used
        
    returns:
        knn       :  kNN regression model
        knn_pred  :  predictions
        rmse_knn  :  prediction RMSE value
        
    outputs:
        screen output of the prediction RMSE value
    
    """     
    # kNN regression training
    knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
    knn.fit(x_train,y_train)
    
    # prediction
    knn_pred = knn.predict(x_test)
    
    # calculation of RMSE
    rmse_knn = sqrt(mean_squared_error(knn_pred, y_test))
    
    # screen output
    print(f"kNN Regression -> RMSE = {rmse_knn:.4f}")
    
    # return values
    return knn, knn_pred, rmse_knn


# Decision Tree regression
#
def treereg(x_train, x_test, y_train, y_test, max_depth=None, min_samples_split=2):
    """ 
    Decision Tree Regression wrapper
    
    parameter:
    
        x_train  :   pandas data frame with features for training
        x_test   :   padas data frame with features for testing
        y_train  :   pandas series with target for training
        y_test   :   pandas series with target for testing
        
        max_depth=None       :   maximum depth of decision tree      
        min_samples_split=2  :   minimum number of samples for split
        
    returns:
        tr        :  Decision Tree regression model
        tr_pred   :  predictions
        rmse_tr   :  prediction RMSE value
        
    outputs:
        screen output of the prediction RMSE value
        
    """         
    # Decision Tree regression training
    tr = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split)
    tr.fit(x_train,y_train)
    
    # prediction
    tr_pred = tr.predict(x_test)
    
    # calculation of RMSE
    rmse_tr = sqrt(mean_squared_error(tr_pred, y_test))
    
    # screen output
    print(f"Decision Tree Regression -> RMSE = {rmse_tr:.4f}")
    
    # return values
    return tr, tr_pred, rmse_tr


# Random Forest regression
#
def randreg(x_train, x_test, y_train, y_test, max_depth=None, min_samples_split=2):
    """ 
    Random Forest Regression wrapper
    
    parameter:
    
        x_train  :   pandas data frame with features for training
        x_test   :   padas data frame with features for testing
        y_train  :   pandas series with target for training
        y_test   :   pandas series with target for testing
        
        max_depth=None       :   maximum depth of decision tree      
        min_samples_split=2  :   minimum number of samples for split
        
    returns:
        rf        :  Random Forest regression model
        rf_pred   :  predictions
        rmse_rf   :  prediction RMSE value
        
    outputs:
        screen output of the prediction RMSE value
        
    """       
    # Random Forest regression training
    rf = RandomForestRegressor(max_depth=max_depth, min_samples_split=min_samples_split)
    rf.fit(x_train,y_train)
    
    # prediction
    rf_pred = rf.predict(x_test)
    
    # calculation of RMSE
    rmse_rf = sqrt(mean_squared_error(rf_pred, y_test))
    
    # screen output
    print(f"Random Forest Regression -> RMSE = {rmse_rf:.4f}")
    
    # return values
    return rf, rf_pred, rmse_rf


# Ada Boost regression
#
def adareg(x_train, x_test, y_train, y_test, learning_rate=1.0, n_estimators=50):
    """ 
    Ada Boost Regression wrapper
    
    parameter:
    
        x_train  :   pandas data frame with features for training
        x_test   :   padas data frame with features for testing
        y_train  :   pandas series with target for training
        y_test   :   pandas series with target for testing
        
        learning_rate=1.0  :   learning rate      
        n_estimators=50    :   number of estimators
        
    returns:
        ada       :  Ada Boost regression model
        ada_pred  :  predictions
        rmse_ada  :  prediction RMSE value
        
    outputs:
        screen output of the prediction RMSE value
        
    """       
    # Ada Boost regression training
    ada = AdaBoostRegressor(learning_rate=learning_rate,n_estimators=n_estimators)
    ada.fit(x_train,y_train)
    
    # prediction
    ada_pred = ada.predict(x_test)
    
    # calculation of RMSE
    rmse_ada = sqrt(mean_squared_error(ada_pred, y_test))
    
    # screen output
    print(f"Ada Boost Regression -> RMSE = {rmse_ada:.4f}")
    
    # return values
    return ada, ada_pred, rmse_ada


# SVM regression
#
def svmreg(x_train, x_test, y_train, y_test, C=1.0, kernel='rbf'):
    """ 
    Support Vector Machine Regression wrapper
    
    parameter:
    
        x_train  :   pandas data frame with features for training
        x_test   :   padas data frame with features for testing
        y_train  :   pandas series with target for training
        y_test   :   pandas series with target for testing
        
        C=1.0          :   complexity      
        kernel='rbf'   :   kernel function
        
    returns:
        svm       :  SVM regression model
        svm_pred  :  predictions
        rmse_svm  :  prediction RMSE value
        
    outputs:
        screen output of the prediction RMSE value
        
    """    
    # SVM regression training
    svm = SVR(C=C,kernel=kernel)
    svm.fit(x_train,y_train)
    
    # prediction
    svm_pred = svm.predict(x_test)
    
    # calculation of RMSE
    rmse_svm = sqrt(mean_squared_error(svm_pred, y_test))
    
    # screen output
    print(f"SVM Regression -> RMSE = {rmse_svm:.4f}")
    
    # return values
    return svm, svm_pred, rmse_svm


# Voting regression
#
def votreg(x_train, x_test, y_train, y_test, estimators=[]):
    """ 
    Voting Regression wrapper
    
    parameter:
    
        x_train  :   pandas data frame with features for training
        x_test   :   padas data frame with features for testing
        y_train  :   pandas series with target for training
        y_test   :   pandas series with target for testing
        
        estimators=[]   :   list of tuples with estimator names and estimator models      

    example:   
        est = [("kNN",knn),("lin",lin),("tree",tr),("random",rf)]
        
        here e.g. in the tuple ("kNN",knn): 
            "kNN" is the name string for the model (could be deliberately been chosen)
            knn is the model itself (must be a previously trained model)
            
    returns:
        vot       :  Voting regression model
        vot_pred  :  predictions
        rmse_vot  :  prediction RMSE value
        
    outputs:
        screen output of the prediction RMSE value
        
    """    
    # Voting regression training
    vot = VotingRegressor(estimators)
    vot.fit(x_train,y_train)
    
    # prediction
    vot_pred = vot.predict(x_test)
    
    # calculation of RMSE
    rmse_vot = sqrt(mean_squared_error(vot_pred, y_test))
    
    # screen output
    print(f"Voting Regression -> RMSE = {rmse_vot:.4f}")
    
    # return values
    return vot, vot_pred, rmse_vot


# XGBoost regression
#
def xgbreg(x_train, x_test, y_train, y_test, learning_rate=0.05, objective ='reg:squarederror', max_depth=16, subsample=1.0, colsample_bytree=0.5, min_child_weight=1, eval_metric='rmse'):
    """ 
    XGBoost Regression wrapper
    
    parameter:
    
        x_train  :   pandas data frame with features for training
        x_test   :   padas data frame with features for testing
        y_train  :   pandas series with target for training
        y_test   :   pandas series with target for testing
        
        learning_rate=0.05              :   learning rate      
        objective ='reg:squarederror'   :   learning objective
        max_depth=16                    :   maximum depth of estimators
        subsample=1.0                   :   the subsample fraction taken into each iteraton step
        colsample_bytree=0.5            :   the column subsample fraction for each individual tree
        min_child_weight=1              :   the minimum weight for children
        eval_metric='rmse'              :   type of metrics taken into account for optimization
       
    returns:
        xgb       :  XGBoost regression model
        xgb_pred  :  predictions
        rmse_xgb  :  prediction RMSE value
        
    outputs:
        screen output of the prediction RMSE value
        
    hint:
        The given standard values align with the values used by Kam Hamidieh for his publication:
        
        Kam Hamidieh. “A data-driven statistical model for predicting the critical temperature
        of a superconductor”. In: Computational Materials Science 154 (2018), pp. 346–354. doi:
        https://doi.org/10.1016/j.commatsci.2018.07.052
        
    """       
    # XGBoost regression training
    xgb = XGBRegressor(learning_rate=learning_rate, objective=objective, max_depth=max_depth, subsample=subsample, colsample_bytree=colsample_bytree, min_child_weight=min_child_weight, eval_metric=eval_metric)
    xgb.fit(x_train,y_train)
    
    # prediction
    xgb_pred = xgb.predict(x_test)
    
    # calculation of RMSE
    rmse_xgb = sqrt(mean_squared_error(xgb_pred, y_test))
    
    # screen output
    print(f"XGBoost Regression -> RMSE = {rmse_xgb:.4f}")
    
    # return values
    return xgb, xgb_pred, rmse_xgb