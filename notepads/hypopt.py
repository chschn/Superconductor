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
import matplotlib.pyplot as plt


# kNN regression
#
def knnhyp(x_train, x_test, y_train, y_test, kmin, kmax, outfile_log, outfile_dat, outfile_fig):
    """ 
    k Nearest Neighbor Regression hyperparameter gridsearch optimizer
    
    parameter:
    
        x_train  :   pandas data frame with features for training
        x_test   :   padas data frame with features for testing
        y_train  :   pandas series with target for training
        y_test   :   pandas series with target for testing
        
        kmin :  minimum iteration value for k
        kmax :  maximum iteration value for k
        
        outfile_log  :   output text file to save the optimization log
        outfile_dat  :   output csv file to save the RMSE values for each run
        outfile_fig  :   outpit jpg file to save the graphic with the optimization curve
   
    note:
        The option weights='distance' is being used
        
    returns:
        kopt      :  optimum k value
        rmse_knn  :  RMSE value for optimum k
        
    outputs:
        screen output of the prediction RMSE values for each run and optimum values at the end
        txt file output of the screen log messages
        csv file output of the data
        jpg output of the optimization graphic
    
    """ 
    min_rmse = np.inf
    best_point = []
    score = []
    
    knn_surf = pd.DataFrame(columns=["k","rmse_knn"])
    
    # regression training loop
    with open(outfile_log, 'w') as f:
        for i in range(1, kmax+1):
            knn = KNeighborsRegressor(n_neighbors=i, weights='distance')
            knn.fit(x_train, y_train)
            knn_pred = knn.predict(x_test)
            rmse_knn = sqrt(mean_squared_error(knn_pred.reshape(-1,1), y_test.to_numpy().reshape(-1,1)))
            print(f"k = {i}, rmse_knn = {rmse_knn:.4f}")
            f.write((f"k = {i}, rmse_knn = {rmse_knn:.4f}\n"))
            knn_surf = knn_surf.append({'k': int(i), 'rmse_knn': rmse_knn}, ignore_index = True)
            score.append(rmse_knn)
            if rmse_knn < min_rmse:
                min_rmse = rmse_knn
                best_point = [i]
                
    # evaluate optimum k and RMSE value
        rmse_knn = min(score)
        score = pd.Series(score)
        kopt = (score[score==rmse_knn].index + 1).values.tolist()[0]    
    
    # screen and file log output
        print("Optimum point with RMSE = {:.4f} found at {}".format(min_rmse, best_point))
        f.write("Optimum point with RMSE = {:.4f} found at {}\n".format(min_rmse, best_point))
        
    # file output
        knn_surf.to_csv(outfile_dat, index=False)
        
    # graph output
    fig = plt.figure(figsize=(8,8))
    plt.title("kNN Hyperparameter Optimization", fontdict={'fontsize': 16})
    plt.plot(range(1,kmax+1), score)
    plt.xlabel("k", fontdict={'fontsize': 14})
    plt.ylabel("RMSE", fontdict={'fontsize': 14})
    plt.xlim(0, kmax)
    plt.xticks(range(0, kmax+1, kmax//10), fontsize=12)
    plt.yticks(fontsize=12)
    fig.savefig(outfile_fig, dpi=150)
    
    # return values
    return kopt, rmse_knn


# Decision Tree regression
#
def treehyp(x_train, x_test, y_train, y_test, max_depth_min, max_depth_max, max_depth_n, min_samples_min, min_samples_max, min_samples_n, outfile_log, outfile_dat, outfile_fig):
    """ 
    Decision Tree Regression hyperparameter gridsearch optimizer
    
    parameter:
    
        x_train  :   pandas data frame with features for training
        x_test   :   padas data frame with features for testing
        y_train  :   pandas series with target for training
        y_test   :   pandas series with target for testing
        
        max_depth_min   :  minimum iteration value for max_depth
        max_depth_max   :  maximum iteration value for max_depth
        max_depth_n     :  number of iterations for max_depth
        min_samples_min :  minimum iteration value for min_samples
        min_samples_max :  maximum iteration value for min_samples
        min_samples_n   :  number of iterations for min_samples
        
        outfile_log  :   output text file to save the optimization log
        outfile_dat  :   output csv file to save the RMSE values for each run
        outfile_fig  :   outpit jpg file to save the graphic with the optimization curve
        
    returns:
        max_depth_opt    :  optimum max_depth value
        min_samples_opt  :  optimum min_samples value
        rmse_tr          :  RMSE value for optimum parameters
        
    outputs:
        screen output of the prediction RMSE values for each run and optimum values at the end
        txt file output of the screen log messages
        csv file output of the data
        jpg output of the optimization graphic
    
    """ 
    max_depth_it = np.linspace(max_depth_min, max_depth_max, max_depth_n)
    min_samples_it = np.linspace(min_samples_min, min_samples_max, min_samples_n)
    
    min_rmse = np.inf
    best_point = []
    
    ct = pd.DataFrame(columns=["max_depth","min_samples","rmse_tr"])
    
    # regression training loop
    with open(outfile_log, 'w') as f:
        for max_depth_i in max_depth_it:
            for min_samples_i in min_samples_it:
                tr_opt = DecisionTreeRegressor(max_depth=int(max_depth_i),min_samples_split=int(min_samples_i))
                tr_opt.fit(x_train,y_train)
                tr_pred = tr_opt.predict(x_test)
                rmse_tr = sqrt(mean_squared_error(tr_pred.reshape(-1,1), y_test.to_numpy().reshape(-1,1)))
                print(f"max_depth = {max_depth_i}, min_samples = {min_samples_i}, rmse_tr = {rmse_tr:.4f}")
                f.write(f"max_depth = {max_depth_i}, min_samples = {min_samples_i}, rmse_tr = {rmse_tr:.4f}\n")
                ct = ct.append({'max_depth': int(max_depth_i), 'min_samples': int(min_samples_i), 'rmse_tr': rmse_tr}, ignore_index = True)
                if rmse_tr < min_rmse:
                    min_rmse = rmse_tr
                    best_point = [max_depth_i,min_samples_i]  
    
    # screen and file log output
        print("Optimum point with RMSE = {:.4f} found at {}".format(min_rmse, best_point))
        f.write("Optimum point with RMSE = {:.4f} found at {}\n".format(min_rmse, best_point))
        
    # file output
        ct.to_csv(outfile_dat, index=False)
        
    # graph output
    x = ct.iloc[:,0].to_numpy().reshape(ct.iloc[:,0].unique().shape[0],ct.iloc[:,1].unique().shape[0])
    y = ct.iloc[:,1].to_numpy().reshape(ct.iloc[:,0].unique().shape[0],ct.iloc[:,1].unique().shape[0])
    z = ct.iloc[:,2].to_numpy().reshape(ct.iloc[:,0].unique().shape[0],ct.iloc[:,1].unique().shape[0])
    
    z1 = np.quantile(z,0.0)
    z2 = np.quantile(z,0.95)
    
    fig = plt.figure(figsize=(10,10))
    zl = np.linspace(z1, z2, 20)
    cset = plt.contour(x, y, z, levels=zl, alpha=0.3, linewidths=0.5)
    plt.contourf(x, y, z, levels=zl, cmap="plasma_r")
    plt.plot(best_point[0], best_point[1], marker='o', markersize=20, color='green')
    plt.clabel(cset, inline=1, fontsize=10)
    plt.xlabel(ct.columns[0])
    plt.ylabel(ct.columns[1])
    plt.colorbar(format='%.1f')
    plt.title("Decision Tree Hyperparameter Optimization")
    
    fig.savefig(outfile_fig, dpi=150)
    
    # return values
    return best_point[0], best_point[1], min_rmse


# Random Forest regression
#
def randhyp(x_train, x_test, y_train, y_test, max_depth_min, max_depth_max, max_depth_n, min_samples_min, min_samples_max, min_samples_n, outfile_log, outfile_dat, outfile_fig):
    """ 
    Random Forest Regression hyperparameter gridsearch optimizer
    
    parameter:
    
        x_train  :   pandas data frame with features for training
        x_test   :   padas data frame with features for testing
        y_train  :   pandas series with target for training
        y_test   :   pandas series with target for testing
        
        max_depth_min   :  minimum iteration value for max_depth
        max_depth_max   :  maximum iteration value for max_depth
        max_depth_n     :  number of iterations for max_depth
        min_samples_min :  minimum iteration value for min_samples
        min_samples_max :  maximum iteration value for min_samples
        min_samples_n   :  number of iterations for min_samples
        
        outfile_log  :   output text file to save the optimization log
        outfile_dat  :   output csv file to save the RMSE values for each run
        outfile_fig  :   outpit jpg file to save the graphic with the optimization curve
        
    returns:
        max_depth_opt    :  optimum max_depth value
        min_samples_opt  :  optimum min_samples value
        rmse_rf          :  RMSE value for optimum parameters
        
    outputs:
        screen output of the prediction RMSE values for each run and optimum values at the end
        txt file output of the screen log messages
        csv file output of the data
        jpg output of the optimization graphic
    
    """ 
    max_depth_it = np.linspace(max_depth_min, max_depth_max, max_depth_n)
    min_samples_it = np.linspace(min_samples_min, min_samples_max, min_samples_n)
    
    min_rmse = np.inf
    best_point = []
    
    ct = pd.DataFrame(columns=["max_depth","min_samples","rmse_rf"])
    
    # regression training loop
    with open(outfile_log, 'w') as f:
        for max_depth_i in max_depth_it:
            for min_samples_i in min_samples_it:
                rf_opt = RandomForestRegressor(max_depth=int(max_depth_i),min_samples_split=int(min_samples_i))
                rf_opt.fit(x_train,y_train)
                rf_pred = rf_opt.predict(x_test)
                rmse_rf = sqrt(mean_squared_error(rf_pred.reshape(-1,1), y_test.to_numpy().reshape(-1,1)))
                print(f"max_depth = {max_depth_i}, min_samples = {min_samples_i}, rmse_rf = {rmse_rf:.4f}")
                f.write(f"max_depth = {max_depth_i}, min_samples = {min_samples_i}, rmse_rf = {rmse_rf:.4f}\n")
                ct = ct.append({'max_depth': int(max_depth_i), 'min_samples': int(min_samples_i), 'rmse_rf': rmse_rf}, ignore_index = True)
                if rmse_rf < min_rmse:
                    min_rmse = rmse_rf
                    best_point = [max_depth_i,min_samples_i]  
    
    # screen and file log output
        print("Optimum point with RMSE = {:.4f} found at {}".format(min_rmse, best_point))
        f.write("Optimum point with RMSE = {:.4f} found at {}\n".format(min_rmse, best_point))
        
    # file output
        ct.to_csv(outfile_dat, index=False)
        
    # graph output
    x = ct.iloc[:,0].to_numpy().reshape(ct.iloc[:,0].unique().shape[0],ct.iloc[:,1].unique().shape[0])
    y = ct.iloc[:,1].to_numpy().reshape(ct.iloc[:,0].unique().shape[0],ct.iloc[:,1].unique().shape[0])
    z = ct.iloc[:,2].to_numpy().reshape(ct.iloc[:,0].unique().shape[0],ct.iloc[:,1].unique().shape[0])
    
    z1 = np.quantile(z,0.0)
    z2 = np.quantile(z,0.95)
    
    fig = plt.figure(figsize=(10,10))
    zl = np.linspace(z1, z2, 20)
    cset = plt.contour(x, y, z, levels=zl, alpha=0.3, linewidths=0.5)
    plt.contourf(x, y, z, levels=zl, cmap="plasma_r")
    plt.plot(best_point[0], best_point[1], marker='o', markersize=20, color='green')
    plt.clabel(cset, inline=1, fontsize=10)
    plt.xlabel(ct.columns[0])
    plt.ylabel(ct.columns[1])
    plt.colorbar(format='%.1f')
    plt.title("Random Forest Hyperparameter Optimization")
    
    fig.savefig(outfile_fig, dpi=150)
    
    # return values
    return best_point[0], best_point[1], min_rmse


# Ada Boost regression
#
def adahyp(x_train, x_test, y_train, y_test, learning_rate_min, learning_rate_max, learning_rate_n, n_estimators_min, n_estimators_max, n_estimators_n, outfile_log, outfile_dat, outfile_fig):
    """ 
    Ada Boost Regression hyperparameter gridsearch optimizer
    
    parameter:
    
        x_train  :   pandas data frame with features for training
        x_test   :   padas data frame with features for testing
        y_train  :   pandas series with target for training
        y_test   :   pandas series with target for testing
        
        learning_rate_min   :  minimum iteration value for learning_rate
        learning_rate_max   :  maximum iteration value for learning_rate
        learning_rate_n     :  number of iterations for learning_rate
        n_estimators_min    :  minimum iteration value for n_estimators
        n_estimators_max    :  maximum iteration value for n_estimators
        n_estimators_n      :  number of iterations for n_estimators
        
        outfile_log  :   output text file to save the optimization log
        outfile_dat  :   output csv file to save the RMSE values for each run
        outfile_fig  :   outpit jpg file to save the graphic with the optimization curve
        
    returns:
        learning_rate_opt  :  optimum learning_rate value
        n_estimators_opt   :  optimum n_estimators value
        rmse_ada           :  RMSE value for optimum parameters
        
    outputs:
        screen output of the prediction RMSE values for each run and optimum values at the end
        txt file output of the screen log messages
        csv file output of the data
        jpg output of the optimization graphic
    
    """ 
    learning_rate_it = np.linspace(learning_rate_min, learning_rate_max, learning_rate_n)
    n_estimators_it = np.linspace(n_estimators_min, n_estimators_max, n_estimators_n)
    
    min_rmse = np.inf
    best_point = []
    
    ct = pd.DataFrame(columns=["learning_rate","n_estimators","rmse_ada"])
    
    # regression training loop
    with open(outfile_log, 'w') as f:
        for learning_rate_i in learning_rate_it:
            for n_estimators_i in n_estimators_it:
                ada_opt = AdaBoostRegressor(learning_rate=learning_rate_i, n_estimators=int(n_estimators_i))
                ada_opt.fit(x_train,y_train)
                ada_pred = ada_opt.predict(x_test)
                rmse_ada = sqrt(mean_squared_error(ada_pred.reshape(-1,1), y_test.to_numpy().reshape(-1,1)))
                print(f"learning_rate = {learning_rate_i}, n_estimators = {n_estimators_i}, rmse_ada = {rmse_ada:.4f}")
                f.write(f"learning_rate = {learning_rate_i}, n_estimators = {n_estimators_i}, rmse_ada = {rmse_ada:.4f}\n")
                ct = ct.append({'learning_rate': learning_rate_i, 'n_estimators': int(n_estimators_i), 'rmse_ada': rmse_ada}, ignore_index = True)
                if rmse_ada < min_rmse:
                    min_rmse = rmse_ada
                    best_point = [learning_rate_i,n_estimators_i] 
    
    # screen and file log output
        print("Optimum point with RMSE = {:.4f} found at {}".format(min_rmse, best_point))
        f.write("Optimum point with RMSE = {:.4f} found at {}\n".format(min_rmse, best_point))
        
    # file output
        ct.to_csv(outfile_dat, index=False)
        
    # graph output
    x = ct.iloc[:,0].to_numpy().reshape(ct.iloc[:,0].unique().shape[0],ct.iloc[:,1].unique().shape[0])
    y = ct.iloc[:,1].to_numpy().reshape(ct.iloc[:,0].unique().shape[0],ct.iloc[:,1].unique().shape[0])
    z = ct.iloc[:,2].to_numpy().reshape(ct.iloc[:,0].unique().shape[0],ct.iloc[:,1].unique().shape[0])
    
    z1 = np.quantile(z,0.0)
    z2 = np.quantile(z,0.95)
    
    fig = plt.figure(figsize=(10,10))
    zl = np.linspace(z1, z2, 20)
    cset = plt.contour(x, y, z, levels=zl, alpha=0.3, linewidths=0.5)
    plt.contourf(x, y, z, levels=zl, cmap="plasma_r")
    plt.plot(best_point[0], best_point[1], marker='o', markersize=20, color='green')
    plt.clabel(cset, inline=1, fontsize=10)
    plt.xlabel(ct.columns[0])
    plt.ylabel(ct.columns[1])
    plt.colorbar(format='%.1f')
    plt.title("Ada Boost Hyperparameter Optimization")
    
    fig.savefig(outfile_fig, dpi=150)
    
    # return values
    return best_point[0], best_point[1], min_rmse


# Support Vector Machine regression
#
def svmhyp(x_train, x_test, y_train, y_test, c_min, c_max, c_n, kernels, outfile_log, outfile_dat, outfile_fig):
    """ 
    Support Vector Machine Regression hyperparameter gridsearch optimizer
    
    parameter:
    
        x_train  :   pandas data frame with features for training
        x_test   :   padas data frame with features for testing
        y_train  :   pandas series with target for training
        y_test   :   pandas series with target for testing
        
        c_min    :  minimum iteration value for c
        c_max    :  maximum iteration value for c
        c_n      :  number of iterations for c
        kernels  :  list of kernel names
        
        outfile_log  :   output text file to save the optimization log
        outfile_dat  :   output csv file to save the RMSE values for each run
        outfile_fig  :   outpit jpg file to save the graphic with the optimization curve
        
    returns:
        c_opt        :  optimum c value
        kernel_opt   :  optimum kernel
        rmse_svm     :  RMSE value for optimum parameters
        
    outputs:
        screen output of the prediction RMSE values for each run and optimum values at the end
        txt file output of the screen log messages
        csv file output of the data
        jpg output of the optimization graphic
    
    """ 
    c_it = np.linspace(c_min, c_max, c_n)
    
    min_rmse = np.inf
    best_point = []
    
    ct = pd.DataFrame(columns=["kernel","C","rmse_svm"])
    
    # regression training loop
    with open(outfile_log, 'w') as f:
        for kernel_i in kernels:
            for c_i in c_it:
                svm_opt = SVR(C=c_i,kernel=kernel_i)
                svm_opt.fit(x_train,y_train)
                svm_pred = svm_opt.predict(x_test)
                rmse_svm = sqrt(mean_squared_error(svm_pred.reshape(-1,1), y_test.to_numpy().reshape(-1,1)))
                print(f"kernel = {kernel_i}, C = {c_i}, rmse_svm = {rmse_svm:.4f}")
                f.write(f"kernel = {kernel_i}, C = {c_i}, rmse_svm = {rmse_svm:.4f}\n")
                ct = ct.append({'kernel': kernel_i, 'C': c_i, 'rmse_svm': rmse_svm}, ignore_index = True)
                if rmse_svm < min_rmse:
                    min_rmse = rmse_svm
                    best_point = [kernel_i,c_i]
    
    # screen and file log output
        print("Optimum point with RMSE = {:.4f} found at {}".format(min_rmse, best_point))
        f.write("Optimum point with RMSE = {:.4f} found at {}\n".format(min_rmse, best_point))
        
    # file output
        ct.to_csv(outfile_dat, index=False)
        
    # graph output
    fig = plt.figure(figsize=(12,8))
    plt.ylim(15, 25)

    for ki in kernels:
        c = ct[ct.iloc[:,0]==ki].iloc[:,1]
        rmse_svm = ct[ct.iloc[:,0]==ki].iloc[:,2]
        plt.plot(c, rmse_svm, label=ki)

    plt.xlabel(ct.columns[0])
    plt.ylabel(ct.columns[1])
    plt.legend()
    plt.title("SVM Hyperparameter Optimization")
    
    fig.savefig(outfile_fig, dpi=150)
    
    # return values
    return best_point[0], best_point[1], min_rmse


# XGBoost regression
#
def xgbhyp(x_train, x_test, y_train, y_test, learning_rate_min, learning_rate_max, learning_rate_n, subsample_min, subsample_max, subsample_n, outfile_log, outfile_dat, outfile_fig):
    """ 
    XGBoost Regression hyperparameter gridsearch optimizer
    
    parameter:
    
        x_train  :   pandas data frame with features for training
        x_test   :   padas data frame with features for testing
        y_train  :   pandas series with target for training
        y_test   :   pandas series with target for testing
        
        learning_rate_min   :  minimum iteration value for learning_rate
        learning_rate_max   :  maximum iteration value for learning_rate
        learning_rate_n     :  number of iterations for learning_rate
        subsample_min       :  minimum iteration value for subsample
        subsample_max       :  maximum iteration value for subsample
        subsample_n         :  number of iterations for subsample
        
        outfile_log  :   output text file to save the optimization log
        outfile_dat  :   output csv file to save the RMSE values for each run
        outfile_fig  :   outpit jpg file to save the graphic with the optimization curve
        
    returns:
        learning_rate_opt  :  optimum learning_rate value
        subsample_opt      :  optimum subsample value
        rmse_xgb           :  RMSE value for optimum parameters
        
    outputs:
        screen output of the prediction RMSE values for each run and optimum values at the end
        txt file output of the screen log messages
        csv file output of the data
        jpg output of the optimization graphic
    
    """ 
    learning_rate_it = np.linspace(learning_rate_min, learning_rate_max, learning_rate_n)
    subsample_it = np.linspace(subsample_min, subsample_max, subsample_n)
    
    min_rmse = np.inf
    best_point = []
    
    ct = pd.DataFrame(columns=["learning_rate","subsample","rmse_xgb"])
    
    # regression training loop
    with open(outfile_log, 'w') as f:
        for learning_rate_i in learning_rate_it:
            for subsample_i in subsample_it:
                xgb_opt = XGBRegressor(learning_rate=learning_rate_i, objective ='reg:squarederror', max_depth=16, subsample=subsample_i, colsample_bytree=0.5, min_child_weight=1, eval_metric='rmse')
                xgb_opt.fit(x_train,y_train)
                xgb_pred = xgb_opt.predict(x_test)
                rmse_xgb = sqrt(mean_squared_error(xgb_pred.reshape(-1,1), y_test.to_numpy().reshape(-1,1)))
                print(f"learning_rate = {learning_rate_i}, subsample = {subsample_i}, rmse_xgb = {rmse_xgb:.4f}")
                f.write(f"learning_rate = {learning_rate_i}, subsample = {subsample_i}, rmse_xgb = {rmse_xgb:.4f}\n")
                ct = ct.append({'learning_rate': learning_rate_i, 'subsample': subsample_i, 'rmse_xgb': rmse_xgb}, ignore_index = True)
                if rmse_xgb < min_rmse:
                    min_rmse = rmse_xgb
                    best_point = [learning_rate_i,subsample_i] 
    
    # screen and file log output
        print("Optimum point with RMSE = {:.4f} found at {}".format(min_rmse, best_point))
        f.write("Optimum point with RMSE = {:.4f} found at {}\n".format(min_rmse, best_point))
        
    # file output
        ct.to_csv(outfile_dat, index=False)
        
    # graph output
    x = ct.iloc[:,0].to_numpy().reshape(ct.iloc[:,0].unique().shape[0],ct.iloc[:,1].unique().shape[0])
    y = ct.iloc[:,1].to_numpy().reshape(ct.iloc[:,0].unique().shape[0],ct.iloc[:,1].unique().shape[0])
    z = ct.iloc[:,2].to_numpy().reshape(ct.iloc[:,0].unique().shape[0],ct.iloc[:,1].unique().shape[0])
    
    z1 = np.quantile(z,0.0)
    z2 = np.quantile(z,0.95)
    
    fig = plt.figure(figsize=(10,10))
    zl = np.linspace(z1, z2, 20)
    cset = plt.contour(x, y, z, levels=zl, alpha=0.3, linewidths=0.5)
    plt.contourf(x, y, z, levels=zl, cmap="plasma_r")
    plt.plot(best_point[0], best_point[1], marker='o', markersize=20, color='green')
    plt.clabel(cset, inline=1, fontsize=10)
    plt.xlabel(ct.columns[0])
    plt.ylabel(ct.columns[1])
    plt.colorbar(format='%.1f')
    plt.title("XGBoost Hyperparameter Optimization")
    
    fig.savefig(outfile_fig, dpi=150)
    
    # return values
    return best_point[0], best_point[1], min_rmse