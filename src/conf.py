"""
===========================================================================
                              c o n f . p y
---------------------------------------------------------------------------
This file contains configurable variables used by other python scripts.

Author          : Tomoko Ayakawa
Created on      : 17 February 2019
Last modified on: 17 February 2019
===========================================================================
"""

class myVariables():
    import os
    dir_name = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
    
    # source file path
    data_path = dir_name + "/data/"
    
    # output file path 
    out_path = dir_name + "/output_ass1/"
    
    # source file names
    ## (1) human activity data set
    human_acty_col_name_file = data_path + "human_activity/features.txt"
    human_acty_feature_file = data_path + "human_activity/train/X_train.txt"
    human_acty_target_file = data_path + "human_activity/train/y_train.txt"
    
    ## (2) spambase data set 
    spam_col_name_file = data_path + "spam/spambase.names"
    spam_data_file = data_path + "spam/spambase.data"
    
    ## (3) phishing data set
    phishing_data_file = data_path + "phishing/Training Dataset.txt"
    
    # the number of top features to display
    ## correlation with the target
    crr_top = 20
    
    ## feature importance
    imp_top = 20

    # Class labels
    human_cls={1:"walking", 2:"walking upstairs", 3:"walking downstairs", \
               4:"sitting", 5:"standing", 6:"laying"}
    spam_cla={1:"spam", 0:"non-spam"}
    psh_cla={-1:"phisy",1:"legitimate"}
