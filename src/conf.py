"""
===========================================================================
                              c o n f . p y
---------------------------------------------------------------------------
This file contains configurable variables used by other python scripts.

Author          : Tomoko Ayakawa
Created on      : 17 February 2019
Last modified on: 10 April 2019
===========================================================================
"""

class myVariables():
    import os
    dir_name=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
    
    # --------------------------------------------------------------------
    # Common variables
    # --------------------------------------------------------------------
    # source file path
    data_path=dir_name + "/data/" 
        
    # source file names
    ## (1) human activity data set
    human_acty_col_name_file=data_path + "human_activity/features.txt"
    human_acty_feature_file=data_path + "human_activity/train/X_train.txt"
    human_acty_target_file=data_path + "human_activity/train/y_train.txt"
    
    ## (2) spambase data set 
    spam_col_name_file=data_path + "spam/spambase.names"
    spam_data_file=data_path + "spam/spambase.data"
    
    ## (3) phishing data set
    phishing_data_file=data_path + "phishing/Training Dataset.txt"
    
    # Class labels
    human_cls={1:"walking", 2:"walking upstairs", 3:"walking downstairs", \
               4:"sitting", 5:"standing", 6:"laying"}
    spam_cla={1:"spam", 0:"non-spam"}
    psh_cla={-1:"phisy",1:"legitimate"}

    # --------------------------------------------------------------------
    # Variables used for Assignment 1
    # --------------------------------------------------------------------
    # output file path 
    out_path=dir_name + "/output_ass1/"
    
    # the number of top features to display
    ## correlation with the target
    crr_top=20
    
    ## feature importance
    imp_top=20
    
    # --------------------------------------------------------------------
    # Variables used for Assignment 2
    # --------------------------------------------------------------------
    # output file path 
    out_path=dir_name + "/output_ass2/"
    
    # default values
    ## (1) data preparation
    def_test=0.2    # test data size
    
    ## (2) autoencoder
    #      (dict key is the data_id 0:human acty, 1:spam, 2:phishing)
    ae_epoch=20     # training epoch for autoencoders
    ae_dropout=0    # dropout rate for autoencoders
    ae_lr={0: 0.001, 1: 0.001, 2: 0.001}  # learning rate
    ae_layers={0: [55, 30, 10],  # number of neurons of each layer
               1: [55, 30, 10], 
               2: [55, 30, 10]} 
    ae_act="relu" # activation function
    ae_loss="mse" # loss function
    ae_opt="adam" # optimizer
    