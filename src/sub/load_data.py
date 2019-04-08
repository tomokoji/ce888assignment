"""
===========================================================================
                          l o a d _ d a t a . p y
---------------------------------------------------------------------------
This code loads datases and converd them into the same DataFrame format.

Author          : Tomoko Ayakawa
Created on      : 17 February 2019
Last modified on: 8 April 2019
===========================================================================
"""
import sys
import pandas as pd
import numpy as np

sys.path.append("../")

import load_human_activity as HUMAN
import load_phishing as PHS
import load_spam as SPAM
from conf import myVariables as VAR

# -------------------------------------------------------------------------
# Get timestamp for picture file name.
# -------------------------------------------------------------------------
def timestamp():
    from time import gmtime, strftime
    
    timestamp = strftime("%Y%m%d%H%M%S", gmtime())
    
    return timestamp

# -------------------------------------------------------------------------
# Read source files.
# -------------------------------------------------------------------------
def load_data(data_id):
    # human activity
    def load_human_activity():
        pic_file = "human_%s" %  timestamp()

        col_names = HUMAN.get_column_names(VAR.human_acty_col_name_file)
        features_df = HUMAN.get_features(VAR.human_acty_feature_file)
        targets_df = HUMAN.get_targets(VAR.human_acty_target_file)

        # concatenate the dataframes
        data_df = pd.concat ([features_df, targets_df], axis = 1)

        # replace default column names with feature names
        data_df.columns = col_names

        # replace string representations into numeric values
        data_df = data_df.convert_objects(convert_numeric=True)

        return col_names, features_df, targets_df, data_df, pic_file

    # spam
    def load_spam():
        pic_file = "spam_%s" %  timestamp()
        col_names = SPAM.get_col_names(VAR.spam_col_name_file)
        features_df, targets_df, data_df = \
            SPAM.get_data(VAR.spam_data_file, col_names)
        
        return col_names, features_df, targets_df, data_df, pic_file
        
    # phishing
    def load_phishing():
        pic_file = "phishing_%s" %  timestamp()
        col_names, features_df, targets_df, data_df = \
            PHS.get_data(VAR.phishing_data_file)

        return col_names, features_df, targets_df, data_df, pic_file

    if data_id == 0:
        return load_human_activity()
    elif data_id == 1:
        return load_spam()
    elif data_id == 2:
        return load_phishing()

# -------------------------------------------------------------------------
# Confirm general information about the data.
# -------------------------------------------------------------------------
def verify_data (data_df, targets_df, dispaly_range=True):
    # count null values
    count_nan = data_df.isnull().values.sum()
    print("Number of NaN: %d" % count_nan)

    if data_df.isnull().values.sum() != 0:
        print ("Count NaN in rows:\n", data_df.isnull().sum(axis=1))
        print ("Count NaN in columns:\n", data_df.isnull().sum())

    # data shape
    print ("Data shape: ", data_df.shape)

    # unique labels and thier ratio in target
    unique_labels, counts = np.unique(targets_df.values, \
                                      return_counts=True)
    ratio=["%.2f" % r for r in counts/sum(counts)]
    print ("Target labels:", unique_labels)
    print ("Class distribution:", ratio)

    # range of features
    if dispaly_range==True:
        min_max =  pd.concat([pd.DataFrame(data_df.max()),\
                              pd.DataFrame(data_df.min())],axis=1)
        min_max.columns=["Max", "Min"]
        print ("The range of features: ", min_max)

    return unique_labels
    
# -------------------------------------------------------------------------
# Allow the programme to be ran from the command line.
# -------------------------------------------------------------------------
if __name__ == '__main__':   
    col_names, features_df, targets_df, data_df, pic_file = \
        load_data(data_id=0)
    unique_labels = verify_data(data_df, targets_df, dispaly_range=False)
    
    print("First five rows in the data\n", data_df.head(5))
