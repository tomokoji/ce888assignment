"""
=========================================================================== 
                          u t i l i t y . p y
---------------------------------------------------------------------------
This file contains micellenious functions used for the CE888 assignment 2.

Author          : Tomoko Ayakawa
Created on      : 1 April 2019
Last modified on: 8 April 2019
===========================================================================
"""

import sys
sys.path.append("./sub")

# import original libraries
import load_data as DATA
#import histogram as HST
#import correlation as CRRL
#import classifier as CLS
#import feature_importance as IMP

# -------------------------------------------------------------------------
# Returns minimal dataset. When specified, return the specified number of
# samples.
# -------------------------------------------------------------------------
def get_small_data(features_np, targets_np, min_samples=None):
    from sklearn.model_selection import train_test_split
    
    num_rows=len(features_np)
    num_cols=len(features_np[0])
    
    if isinstance(min_samples, int): min_rows=min_samples
    else: min_rows = 50 + 8*num_cols
    
    ratio = (min_rows/num_rows)
    
    print ("ratio:", ratio)
    X, min_features, y, min_targets = \
            train_test_split(features_np, targets_np, test_size=ratio)
    
    print ("Minimul size of the data for %d features: %d" % \
           (num_cols, min_rows))
    print ("The size of the data used for classification: %d" % \
           len(min_features))
    
    return min_features, min_targets
# -------------------------------------------------------------------------
# Allow the programme to be ran from the command line.
# -------------------------------------------------------------------------
if __name__ == '__main__':
    data_id = 0
    col_names, features_df, targets_df, data_df, pic_file = \
        DATA.load_data(data_id=data_id)
    unique_labels = DATA.verify_data(data_df, targets_df, False)

    min_features, min_targets=\
        get_small_data(features_df.values, targets_df.values, 10) 