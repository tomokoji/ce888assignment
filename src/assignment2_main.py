"""
=========================================================================== 
                a s s i g n m e n t 2 _ m a i n . p y
---------------------------------------------------------------------------
This code is the main function of the programme written for CE888 
assignment 2. It shows menus for a user and calls methods in accordance
with the selected menu.

Author          : Tomoko Ayakawa
Created on      : 10 April 2019
Last modified on: 10 April 2019
===========================================================================
"""

import sys
sys.path.append("./sub")

# import original libraries
from conf import myVariables as VAR
import load_data as DATA
import preprocess as PREP
import autoencoder as AE
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
              "----------------------------------------\n" \
              "    [Evaluation]\n" \
              "      8) Decision Tree\n" \
              "      9) Naive Bayes\n" \
              "     10) SVM\n" \
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
            
            # select type of the autoencoder
            mode=input("Select 0 for Normal Autoencoder, " \
                       "1 for Stacked Autoencoder (default=0): ")
            try:
                mode=int(mode)
            except:
                mode=0
        
            encode, hist=AE.autoencoder(X, layers=VAR.ae_layers[data_id], \
                mode=mode, act=VAR.ae_act, opt=VAR.ae_opt, \
                loss=VAR.ae_loss, dropout=0, \
                epochs=10, verbose=0, summary_display=False)
            
            AE.plot_ae_loss_history(hist, mode, "test", save=True)
            ready=5         
            
            
            
            
        elif ans==6:
            if cr==False: cr_np = CRRL.correlation(data_df)
            CRRL.cr_bar_graph (cr_np, pic_file, col_names)
            cr=True
        elif ans==7:
            clf = CLS.train (features_df, targets_df, 0)[-1]
            IMP.feature_importance(clf, col_names, pic_file, \
                                   pic_file.split("_")[0])
        elif ans>7:
            cls_name={8: "Decision_Tree", 9:"Naive_Bayes", 10: "SVM"}
            title = "%s_%s" % (pic_file.split("_")[0], cls_name[ans])
            X, y = features_df.values, targets_df.values
            min_ans=input("Use minimal dataset? (y/n): ")
            if min_ans.lower()=="y":
                X, y = get_small_data(X, y)
            true_label, pred_label, clf = CLS.train (X, y, ans - 8)
            CLS.plot_confusion_matrix(true_label, pred_label, \
                             unique_labels, pic_file, title, cls_name[ans])

# -------------------------------------------------------------------------
# Allow the programme to be ran from the command line.
# -------------------------------------------------------------------------
if __name__ == '__main__':
    menu()    