"""
=========================================================================== 
                         p r e p r o c e s s . p y
---------------------------------------------------------------------------
This file contains functions to pre-process the data.
Part of the functions are reused from assignment1_main.py.

Author          : Tomoko Ayakawa
Created on      : 1 April 2019
Last modified on: 8 April 2019
===========================================================================
"""

import sys
sys.path.append("./sub")
sys.path.append("../")

# import original libraries
from conf import myVariables as VAR
import load_data as DATA
#import histogram as HST
#import correlation as CRRL
#import classifier as CLS
#import feature_importance as IMP

# -------------------------------------------------------------------------
# Returns the specified number of data samples. When not specified, return 
# the minimal number of samples.
# This function is based on and modified from get_minimal_data in 
# assignment1_main.py.
# -------------------------------------------------------------------------
def get_small_data(X, y):
    from sklearn.model_selection import train_test_split
    X, y=X.values, y.values 
    num_rows=len(X)
    num_cols=len(X[0])
    minimal=n=50+8*num_cols
    
    # the number of samples to be used
    n=input("Enter the number of samples to use (Total: %d): " % num_rows)  
    try:
        n=int(n)
    except:
        n=0
    
    if ((n<=0) or (n>num_rows)): n=minimal
    
    #split the data
    X, features, y, targets = \
            train_test_split(X, y, test_size=n/num_rows)
    
    print (" - Number of features: %d\n" \
           " - Minimul data size : %d\n" \
           " - Specified size    : %d\n" % (num_cols, minimal, n))
    
    return features, targets

# -------------------------------------------------------------------------
# Pre-process the data with a specified scaler.
# When the scaler ID is not specified, MinMaxScaler will be applied.
# -------------------------------------------------------------------------
def pre_processing(X, display_result=True):   
    import pandas as pd
    from sklearn.preprocessing import \
        MinMaxScaler, QuantileTransformer, StandardScaler
    
    # select the scaler
    mode=input("Select the scaler 0 (None), 1 (MinMax), 2 (Quantile), " \
               "3 (Standard): ")  
    try:
        mode=int(mode)
    except:
        mode=1
    
    # fit the scalar
    if mode==0:
        scl=None
        X_nrm=X
    else:
        if mode==1: scl=MinMaxScaler()
        if mode==2: scl=QuantileTransformer(output_distribution="normal")
        if mode==3: scl=StandardScaler()
        scl.fit(X)
        X_nrm=scl.transform(X)
    
    print("Scaler:", scl)
    if display_result==True:
        print("Original data:\n", pd.DataFrame(X).head(5))
        print("Pre-processed data:\n", pd.DataFrame(X_nrm).head(5))
                
    return scl, X_nrm

# -------------------------------------------------------------------------
# Split the data into training dataset and test dataset.
# When the given test size is invalid, 0.2 is used as a default.
# -------------------------------------------------------------------------
def split_data(X, y):
    from sklearn.model_selection import train_test_split
    
    # obtain the test size (0~1)
    test_size=input("Enter the test_size (0<=test_size<1): ")
    try:
        test_size=float(test_size)
    except:
        test_size=VAR.def_test
        
    if (test_size<0) or (test_size>=1): test_size=VAR.def_test
        
    X_tr, X_te, y_tr, y_te=train_test_split(X, y, test_size=test_size)
    print("Training: %d, Test: %d\n" % (len(X_tr), len(X_te)))
    
    return X_tr, X_te, y_tr, y_te
    
# -------------------------------------------------------------------------
# Allow the programme to be ran from the command line.
# -------------------------------------------------------------------------
if __name__ == '__main__':
    data_id = 0
    col_names, features_df, targets_df, data_df, pic_file = \
        DATA.load_data(data_id=data_id)
    unique_labels = DATA.verify_data(data_df, targets_df, False)

    features, classes=get_small_data(features_df, targets_df)
    scl, features_nrm=pre_processing(features, display_result=False)
    X_tr, X_te, y_tr, y_te=split_data(features_nrm, classes)