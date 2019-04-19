"""
===========================================================================
                              c o n f . p y
---------------------------------------------------------------------------
This file contains configurable variables used by other python scripts.

Author          : Tomoko Ayakawa
Created on      : 17 February 2019
Last modified on: 19 April 2019
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
    ## (1) general
    def_test=0.2    # test data size
    
    act_list={0:"relu", 1:"sigmoid", 2:"tanh", 3:"softmax"}
    loss_list={0:"mse", 1:"mean_absolute_error", \
              2:"mean_squared_logarithmic_error", \
              3:"categorical_crossentropy"}
    opt_list={0:"adam", 1:"sdg"}
    
    ## (2) autoencoder
    #      (dict key is the data_id 0:human acty, 1:spam, 2:phishing)
    yesno={0:"False", 1:"True"}
    ae_mode_list={0:"Basic", 1:"Stacked"}
    ae_mode=1       # autoencoder type (0:normal, 1:stacked)
    ae_epoch=100    # training epoch for autoencoders
    ae_dropout=0    # dropout rate for autoencoders
    ae_layers={0: [500, 400, 300, 200, 100, 50, 25, 10],  # number of 
               1: [50, 40, 30, 20, 10],                   # neurons of 
               2: [25, 20, 105, 10]}                      # each layer
    ae_act=0            # activation function (index of act_list)
    ae_loss=0           # loss function (index of loss_list)
    ae_opt=0            # optimizer 0:adam 1:sdg
    ae_lr=0.001         # learning rate
    ae_momentum=0.9     # momentum
    ae_verbose=0       # 0:False, 1: True
    ae_summary_display=0  # 0:False, 1: True
    
    ## (3) MLP
    h_num=20               # number of neurons in the hidden layer
    h_act=0                # activation function (index of act_list)
    out_act=3              # activation function (index of act_list)
    mlp_loss=3             # loss function (index of loss_list)
    mlp_opt=0              # optimizer 0:adam 1:sdg
    mlp_lr=0.001            # learning rate
    mlp_momentum=0.9       # momentum
    val_rate=0.2           # validation rate
    mlp_epoch=20           # training epoch for autoencoders
    mlp_verbose=0          # 0:False, 1: True
    mlp_summary_display=0  # 0:False, 1: True
    finetune=0             # 0:False, 1: True
    cv=5                   # number of folds for cross validation
    
    ## (4) Grid Search
    grid_splits=2          # number of grid split
    alpha=1                # reguralization parameter
    