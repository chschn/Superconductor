#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Deduplicate the instaces of the dataset in file "infile" and write results into file "outfile"
#
def dedup(infile, outfile):
    """ 
    De-duplication of instances
    
    1. Eliminating true duplicates (i.e. all attributes incl. target are identical) - keep first instance
        We only want to keep one instance. 
        Otherwise there would be an artificial extra weight for these materials.
        
    2. Eliminating attribute duplicates (i.e. all attributes identical, but target different) - don't keep
        We don't want to consider these instances, since measurements contradict. We don't know the reason.
    
    parameter:
    
        infile  :   name of input csv file with duplicate instances
        outfile :   name of output csv file for cleaned data
        
    returns:
        None
        
    outputs:
        screen output with stats
        
    """        
    inpath = "./" + infile
    outpath = "./" + outfile
    
    # read the contents of infile into dataframe "df" 
    df = pd.read_csv(inpath, sep=',', header=0)
    rawc = df.shape[0]
    
    # Remove true duplicates from the dataframe
    df.drop_duplicates(subset=None, keep='first', inplace=True, ignore_index=True)
    
    # Remove attribute duplicates from the dataframe
    df.drop_duplicates(subset=list(df.columns[:-1]), keep=False, inplace=True, ignore_index=True)
    rawd = df.shape[0]
    
    # export resulting dataframe into outfile
    df.to_csv(outpath, index=False)
    
    # screen message
    print(f"Instances de-duplicated from count = {rawc} to {rawd}")

    
# Normalise the features in file "infile" without normalising the target column. Write results into file "outfile"
#
def norm(infile, outfile):
    """ 
    Normalization of all features but the target
    
    Normalizing the target would make the interpretation of predictions and errors difficult 
    (wouldn't represent temperature in Kelvin anymore)
    
    parameter:
    
        infile  :   name of input csv file to be normalized
        outfile :   name of output csv file for normalized data
        
    returns:
        None
        
    outputs:
        screen output with stats
        
    """        
    inpath = "./" + infile
    outpath = "./" + outfile

    # read the contents of infile into dataframe "df" 
    df = pd.read_csv(inpath, sep=',', header=0)
    
    # evaluate min and max before normalisation for screen message only
    min_raw = df.iloc[:,:-1].to_numpy().ravel().min()
    max_raw = df.iloc[:,:-1].to_numpy().ravel().max()
    
    # normalise the dataset without the target column
    df_s = df.drop(['critical_temp'], axis=1)
    scaler = MinMaxScaler()
    df_s = scaler.fit_transform(df)
    df_s = pd.DataFrame(df_s, columns=df.columns)
    df_s = df_s.assign(critical_temp=df['critical_temp'])
    
    # evaluate min and max after normalisation for screen message only
    min_norm = df_s.iloc[:,:-1].to_numpy().ravel().min()
    max_norm = df_s.iloc[:,:-1].to_numpy().ravel().max()

    # export resulting dataframe into outfile
    df_s.to_csv(outpath, index=False)
    
    # screen output
    print(f"Normalised from min = {min_raw:.3f} and max = {max_raw:.3f} to min = {min_norm:.3f} and max = {max_norm:.3f}")
    
    
# Reduce the number of features according to the simplified approach
#
def reduce(infile, outfile, cr_exit, cr_tar=0.99, crt=0.0):
    """ 
    Feature reduction of the dataset
    
    1. Reducing attributes which have multicolinearity:
        Go from the highest correlation off the diagonal to the lowest until threshold is reached (cr_exit)
        From each pair of correlating attributes, keep the one with higher correlation with the target and discard the other
        Do not discard if the correlation with the target is above second threshold (cr_tar)
        
    2. Discard attributes with very low correlation with target (threshold: crt)
    
    parameter:
    
        infile      :   name of input csv file with full set of features
        outfile     :   name of output csv file for feature reduced data
        cr_exit     :   threshold for absolute value of cross-correlation  
                            -> values below are not considered
        cr_tar=0.99 :   threshold for absolute value of target correlation 
                            -> values above are not considered
        crt=0.0     :   override limit for absolute value of target correlation 
                            -> values below lead to elimination regardless of cross-correlation

    returns:
        None
        
    outputs:
        screen output with stats

    """    
    inpath = "./" + infile
    outpath = "./" + outfile

    # read the contents of infile into dataframe "df" 
    df = pd.read_csv(inpath, sep=',', header=0)
    
    # generate the absolute value correlation matrix
    crmx = df.corr().abs()
    
    # Reducing attributes which have multicolinearity
    # Go from the highest correlation off the diagonal to the lowest until threshold is reached (cr_exit)
    # From each pair of correlating attributes, keep the one with higher correlation with the target and discard the other
    # Do not discard if the correlation with the target is above second threshold (cr_tar)
    cr_ini = 0.0
    cr_lim = 1.0
    coldel = []
    colkeep = []
    cr = []
    while 1 == 1:
        cr_max = np.amax(crmx.to_numpy(), where= crmx.to_numpy()<cr_lim, initial=cr_ini)
        if cr_max < cr_exit:
            break
        cr_lim = cr_max
        cr.append(cr_max)
        mxind = list(crmx.eq(cr_max).any()[crmx.eq(cr_max).any()].index.values)
        if mxind[0] == 'critical_temp' or mxind[1] == 'critical_temp':
            continue
        if crmx[mxind[0]]["critical_temp"] >= crmx[mxind[1]]["critical_temp"] and crmx[mxind[1]]["critical_temp"] < cr_tar:
            coldel.append(mxind[1])
            colkeep.append(mxind[0])
        elif crmx[mxind[1]]["critical_temp"] >= crmx[mxind[0]]["critical_temp"] and crmx[mxind[0]]["critical_temp"] < cr_tar:
            coldel.append(mxind[0])
            colkeep.append(mxind[1])
            
    # Discard attributes with very low correlation with target
    # Threshold: crt
    coldel_crt = []
    cr_crt = []
    for col in df.columns[:-1]:
        if crmx[col]["critical_temp"] < crt:
            coldel_crt.append(col)
            cr_crt.append(crmx[col]["critical_temp"])
            
    # Build the total list of columns to discard
    coldel = coldel + coldel_crt
    
    # Output the number of attributes to be deleted and kept
    coldel = list(dict.fromkeys(coldel))
    print(f"Columns to delete = {len(coldel)}, columns remaining = {len(df.columns) - len(coldel)}")
    
    # Physically drop the columns to be discarded
    df.drop(coldel, axis=1, inplace=True)

    # export resulting dataframe into outfile
    df.to_csv(outpath, index=False)
