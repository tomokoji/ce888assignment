"""
=========================================================================== 
                a s s i g n m e n t 2 _ m a i n . p y
---------------------------------------------------------------------------
This code is the main function of the programme written for CE888 
assignment 2. It shows menus for a user and calls methods in accordance
with the selected menu.

Author          : Tomoko Ayakawa
Created on      : 10 April 2019
Last modified on: 16 April 2019
===========================================================================
"""

import sys
import numpy as np
sys.path.append("./sub")

# import original libraries
from conf import myVariables as VAR
import load_data as DATA
import preprocess as PREP
import autoencoder as AE
import mlp as MLP
#import histogram as HST
#import correlation as CRRL
#import classifier as CLS
#import feature_importance as IMP

# -------------------------------------------------------------------------
# Repeat showing a menu
# -------------------------------------------------------------------------
def menu():
    err, ready=0, 0
    correct_ans=[str(i) for i in list(range(11))]

    while err<3: 
        print(">>> Select menu (~ No.%d ready to run):\n" \
              "    [Prepare Data]\n" \
              "      0) Load Data\n" \
              "      1) Obtain small data\n" \
              "      2) Pre-process the data\n" \
              "      3) Split the data\n" \
              "    [Train Classifier]\n" \
              "      4) Autoencoder\n" \
              "      5) MLP\n" \
              "      6) Grid search for optimal parameter\n"
              "    [Evaluation]\n" \
              "      -) Decision Tree\n" \
              "      -) Naive Bayes\n" \
              "      -) SVM\n" \
              "    [Quit]\n" \
              "      others) Exit the menu" % ready)

        ans=input(">>> ")
        
        if ans in correct_ans:
            ans=int(ans)
        else: 
            break
        
        if ans==0: # load data
            # select the data to load
            data_err=0
            while data_err<3: 
                data_id=input(">>> Select the data:\n" \
                           "    0) Human Activity\n" \
                           "    1) Spam\n" \
                           "    2) Phishing\n"
                           ">>> ")
                try:
                    data_id=int(data_id)
                    data_err=10
                except:
                    data_id=9
                if data_id not in [0,1,2]: 
                    data_err+=1
                    continue
            if data_err==3: break
            
            col_names, features_df, targets_df, data_df, pic_file = \
                DATA.load_data(data_id=data_id)
            unique_labels=DATA.verify_data(data_df, targets_df, \
                                           dispaly_range=False)
            print (data_df.head(5))
            ready=1
        elif ans==1: # obtain small data
            if ready<ans:
                print("[ ERROR ] Data is not loaded yet.")
                continue
            features, classes=PREP.get_small_data(features_df, targets_df)
            ready=2
        elif ans==2: # pre-process the data
            if ready<ans:
                print("[ ERROR ] Small data are not prepared yet.")
                continue
            scl, features_nrm=PREP.pre_processing(features, \
                                             display_result=True)
            ready=3
        elif ans==3: # split the data into training and test datasets
            if ready<ans:
                print("[ ERROR ] Data is not pre-processed yet.")
                continue
            X_tr, X_te, y_tr, y_te=PREP.split_data(features_nrm, classes)
            X, y=features_nrm, classes
            ready=4
        elif ans==4: # train an autoencoder
            if ready<ans:
                print("[ ERROR ] Data preparation is not completed yet.")
                continue
            
            # select parameters for autoencoder
            ae_layers, mode, act, opt, loss, dropout, epochs, verbose, \
                summary_display=AE.get_parameters(data_id)
            ans=input("Continue? (y/n): ")
            if (ans!="y") and (ans!="Y"): continue
        
            # train an autoencoder
            encoder, histories=AE.autoencoder(X, layers=ae_layers, \
                    mode=mode, act=act, opt=opt, loss=loss, \
                    dropout=dropout, epochs=epochs, verbose=verbose, \
                    summary_display=summary_display)
            
            # display the training loss history
            AE.plot_ae_loss_history(histories, mode, pic_file)
            
            # obtain compressed features
            X_all_cmp=encoder.predict(features_nrm)
            X_tr_cmp=encoder.predict(X_tr)
            X_te_cmp=encoder.predict(X_te)           
            print("The number of compressed features:", len(X_all_cmp[0]))
            ready=5         
        elif ans==5: # train an MLP
            if ready<ans:
                print("[ ERROR ] An autoencoder is not trained yet.")
                continue
            
            finetune, h_num, h_act, out_act, opt, loss, epochs, val_rate, \
                verbose, summary_display=MLP.get_parameters()
            ans=input("Continue? (y/n): ")
            
            if (ans!="y") and (ans!="Y"): continue 
            k=ae_layers[-1]
            n=len(np.unique(y))
            model=MLP.build_mlp(encoder, num_in=k, num_out=n, \
                            finetune=finetune, h_num=h_num, h_act=h_act, \
                            out_act=out_act, opt=opt, loss=loss, \
                            summary_display=summary_display)
            
            histories=MLP.train_mlp(X, y, model, epochs=epochs, \
                            val_rate=val_rate, verbose=verbose)
            MLP.plot_mlp_loss_history(histories, pic_file)
            ready=6            
        elif ans==6: # grid dsearch
            if ready<ans:
                print("[ ERROR ] A classifier is not trained yet.")
                continue

# -------------------------------------------------------------------------
# Allow the programme to be ran from the command line.
# -------------------------------------------------------------------------
if __name__ == '__main__':
    menu()    