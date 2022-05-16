# Superconductor critical temperature prediction

This code is predicting the critical temperatures of superconductors and is based on the superconductor dataset from the UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Superconductivty+Data

Following program and function structure exists:

    (for further details see the docstrings and comments of the python code)

<pre>

</pre>
Programs:
==========
<pre>

</pre>
expraw.py
=========
### Exploratory data analysis raw data

- Reading the raw data
- Display of the statistics
- Boxplot of the attributes
- Some histograms
- Some crossplots
- Analysis of instances -> real dups
- Analysis of instances -> attribute dups

tidy.py
=======

### Data cleanup

- Reading of the raw data
- Removal of excess real dups
- Removal of attribute dups
- Normalization of features, except target
- Saving to file

exptidy.py
==========

### Exploratory data analysis cleaned data

- Reading the cleaned data
- Display of the statistics
- Boxplot of the attributes
- Some histograms
- Some crossplots
- Heatmap of the absolute correlation matrix

reduce.py
=========

### Feature reduction

- Reading the cleaned data
- Reduction of the features
- Heatmap of the absolute correlation matrix
- Crossplot matrix
- Save to file

splitfull.py
============

### Split into training and test data

- Reading the raw data
- Normalize features without target
- training- test data split
- Save training data to file
- Save test data to file

splitred.py
===========

### Split into training and test data

- Reading the reduced data
- Training- Test data split
- Save training data to file
- Save test data to file

pca.py
======

### Principal Component Analysis

- Reading the training and test data
- Display PCA variance sum - number of features
- Heatmap of the PCA mapping
- Bar chart PCA components vs. explained variance
- PCA of features with selected threshold on training dataset
- PCA of features with selected threshold on test dataset
- Save training data to file
- Save test data to file

hyper.py
========

### Hyperparameter optimization

- Reading the reduced training and test data
- Regression runs of the 8 models
- Crossplot display
- Barchart RMSE vs. model

regfull.py
==========

### Regression on the raw data set

- Reading in the training and test data
- Regression runs of the 8 models
- Crossplot display
- Barchart RMSE vs. model

regpca.py
=========

### Regression on the PCA data set

- Reading in the training and test data
- Regression runs of the 8 models
- Crossplot display
- Barchart RMSE vs. model

regress.py
=========

### Regression on the feature reduced dataset

- Reading in the training and test data
- Regression runs of the 8 models
- Crossplot representation
- Bar chart RMSE vs. model

<pre>



</pre>
Functions:
===========
<pre>

</pre>
prep.py
=======

### def dedup(infile, outfile):
### Deduplicate instances

- Remove real and attribute duplicates from instances.

    - Input: filename_input, filename_output

        - Remove real duplicates
        - Remove attribute duplicates

    - Return: -

### def norm(infile, outfile):
### Normalize features

- normalize features, leave target

    - Input: filename_input, filename_output, target

        - Normalize dataset

    - Return: -

### def reduce(infile, outfile, cr_exit, cr_tar=0.99, crt=0.0):
### Reduce features

- Reduce features according to the simplified approach

    - Input: filename_input, filename_output, cr_exit, cr_tar=0.99, crt=0.0

        - Reduce features according to the simplified approach

    - Return: -

util.py
=======

### def tsplit(infile, out_train, out_test, test_size=None):
### Split data

- Split data into training data and test data

    - Input: filename_input, filename_train, filename_test, test_size=None

        - Split DataFrame into training and test data
        - Build new DataFrame from x_train, y_train
        - Build new DataFrame from x_test, y_test
        - Save DataFrame_train in filename_train as CSV
        - Save DataFrame_test in filename_test as CSV

    - Return: -

### def read_train_test(in_train, in_test):
### Read training and test data

- read training data and test data each from files into separate DataFrames

    - Input: filename_train, filename_test,

        - read x_train, y_train from filename_train as CSV
        - read x_test, Y_test from filename_test as CSV

    - return: x_train, y_train, x_test, y_test

regr.py
=======

### def linreg(x_train, x_test, y_train, y_test, grd=1):
### Linear Regression

- Perform linear regression

    - Input: x_train, y_train, x_test, y_test, grd=1

        - Set polynomial degree equal to grd
        - Apply PolynomialFeature transform to x_train and x_test
        - Train linear regression on x_train and y_train
        - Create Predictor
        - Calculate RMSE

    - return: lin, lin_pred, rmse_lin

### def knnreg(x_train, x_test, y_train, y_test, k=1):
### kNN Regression

- perform kNN regression

    - Input: x_train, y_train, x_test, y_test, k=1

        - train kNN regression on x_train and y_train
        - Create Predictor
        - calculate RMSE

    - return: knn, knn_pred, rmse_knn

### def treereg(x_train, x_test, y_train, y_test, max_depth=None, min_samples_split=2):
### Decision Tree Regression

- Perform Decision Tree Regression

    - Input: x_train, y_train, x_test, y_test, max_depth ,min_samples_split

        - Train Decision Tree Regression on x_train and y_train
        - Create Predictor
        - Calculate RMSE

    - return: tr, tr_pred, rmse_tr

### def randreg(x_train, x_test, y_train, y_test, max_depth=None, min_samples_split=2):
### Random Forest Regression

- Perform Random Forest Regression

    - Input: x_train, y_train, x_test, y_test, max_depth ,min_samples_split

        - Train Random Forest Regression on x_train and y_train
        - Create Predictor
        - Calculate RMSE

    - Return: tr, tr_pred, rmse_tr

### def adareg(x_train, x_test, y_train, y_test, learning_rate=1.0, n_estimators=50):
### Ada Boost Regression

- Perform Ada Boost Regression

    - Input: x_train, y_train, x_test, y_test, learning_rate , n_estimators

            - Train Ada Boost Regression on x_train and y_train
        - Create Predictor
        - Calculate RMSE

    - return: ada, ada_pred, rmse_ada

### def svmreg(x_train, x_test, y_train, y_test, C=1.0, kernel='rbf'):
### SVM Regression

- Perform SVM regression

    - Input: x_train, y_train, x_test, y_test, C, kernel

        - Train SVM regression on x_train and y_train
        - Create Predictor
        - calculate RMSE

    - return: svm, svm_pred, rmse_svm

### def votreg(x_train, x_test, y_train, y_test, estimators=[]):
### Voting Regression

- Perform Voting Regression

    - Input: x_train, y_train, x_test, y_test, knn, lin, tr, rf

        - Train voting regression on x_train and y_train
        - Create Predictor
        - calculate RMSE

    - return: vot, vot_pred, rmse_vot

### def xgbreg(x_train, x_test, y_train, y_test, learning_rate=0.05, objective ='reg:squarederror', max_depth=16, subsample=1.0, colsample_bytree=0.5, min_child_weight=1, eval_metric='rmse'):
### XGBoost Regression

- Perform XGBoost Regression

    - input: x_train, y_train, x_test, y_test, learning_rate, subsample

        - set default parameters: objective ='reg:squarederror', max_depth=16, colsample_bytree=0.5, min_child_weight=1, eval_metric='rmse'
        - train XGBoost regression on x_train and y_train
        - Create Predictor
        - calculate RMSE

    - return: xgb, xgb_pred, rmse_xgb

hypopt.py
=========

### def knnhyp(x_train, x_test, y_train, y_test, kmin, kmax, outfile_log, outfile_dat, outfile_fig):
### kNN Hyperparameter Optimization

- Perform kNN hyperparameter optimization

    - Input: x_train, y_train, x_test, y_test, kmin, kmax, filename_log, filename_data, filename_fig.

        - Train in a loop from kmin to kmax kNN
        - Calculate the RMSE for each k
        - Find minimum RMSE
        - Calculate the corresponding kopt
        - Create graph with curve and save it under filename_fig

    - Return: kopt

### def treehyp(x_train, x_test, y_train, y_test, max_depth_min, max_depth_max, max_depth_n, min_samples_min, min_samples_max, min_samples_n, outfile_log, outfile_dat, outfile_fig):
### Decision Tree Hyperparameter Optimization

- Perform Decision Tree Hyperparameter Optimization

    - Input: x_train, y_train, x_test, y_test, max_depth_min, max_depth_max, max_deth_n, min_samples_min, min_samples_max, min_samples_n, filename_log, filename_data

        - In a loop from max_depth_min to max_depth_max count max_depth_n:
        - In a loop from min_samples_min to min_samples_max Count min_samples_n:
        - Calculate the RMSE for each counter index pair.
        - Find minimum RMSE
        - Store associated index pair

    - Return: max_depth_opt, min_samples_opt
<pre>

</pre>

### def randhyp(x_train, x_test, y_train, y_test, max_depth_min, max_depth_max, max_depth_n, min_samples_min, min_samples_max, min_samples_n, outfile_log, outfile_dat, outfile_fig):
### Random Forest Hyperparameter Optimization

- Same as Decision Tree Hyperparameter Optimization

<pre>

</pre>
### def adahyp(x_train, x_test, y_train, y_test, learning_rate_min, learning_rate_max, learning_rate_n, n_estimators_min, n_estimators_max, n_estimators_n, outfile_log, outfile_dat, outfile_fig):
### Ada Boost Hyperparameter Optimization

- Same as Decision Tree Hyperparameter Optimization
<pre>

</pre>

### def svmhyp(x_train, x_test, y_train, y_test, c_min, c_max, c_n, kernels, outfile_log, outfile_dat, outfile_fig):
### SVM Hyperparameter Optimization

- Same as Decision Tree Hyperparameter Optimization

<pre>

</pre>
### def xgbhyp(x_train, x_test, y_train, y_test, learning_rate_min, learning_rate_max, learning_rate_n, subsample_min, subsample_max, subsample_n, outfile_log, outfile_dat, outfile_fig):
### XGBoost Hyperparameter Optimization

- Same as Decision Tree Hyperparameter Optimization
