"""
=========================================================================== 
                a s s i g n m e n t 1 _ m a i n . p y
---------------------------------------------------------------------------
This code loads datases and converd them into the same DataFrame format.

Author          : Tomoko Ayakawa
Created on      : 17 February 2019
Last modified on: 17 February 2019
===========================================================================
"""

import sys
sys.path.append("./sub")

# import original libraries
import load_data as DATA
import histogram as HST
import correlation as CRRL
import classifier as CLS
import feature_importance as IMP

# -------------------------------------------------------------------------
# Repeat showing a menu
# -------------------------------------------------------------------------
def menu():
    err=0
    data_load=False # data load flag (False: not loaded, True: loaded)
    cr=False # correlation matrix flag (False: not created, True: created)
    correct_ans=[str(i) for i in list(range(3))]
    analysis_menus=""

    while err<3:
        if data_load!=False:
            correct_ans=[str(i) for i in list(range(11))]
            analysis_menus="    [Data Structure]\n" \
                     "      3) Histogram of features\n" \
                     "      4) Histogram of targets\n" \
                     "      5) Heatmap of feature correlation\n" \
                     "    [Feature Importance]\n" \
                     "      6) Bargraph of feature-target correlation\n" \
                     "      7) Feature importance\n" \
                     "    [Classification with Small Data]\n" \
                     "      8) Decision Tree\n" \
                     "      9) Naive Bayes\n" \
                     "     10) SVM\n" \
    
    
        print(">>> Select menu:\n" \
              "    [Load Data]\n" \
              "      0) Human Activity\n" \
              "      1) Spam\n" \
              "      2) Phishing\n" \
              "%s" \
              "    [Quit]\n" \
              "      others) Exit the menu" % analysis_menus)
        ans = input(">>> ")
        
        if ans in correct_ans:
            ans = int (ans)
        else: 
            break
        
        if ans < 3:
            col_names, features_df, targets_df, data_df, pic_file = \
                DATA.load_data(data_id=ans)
            unique_labels = DATA.verify_data(data_df, targets_df)
            print (data_df.head(5))
            data_load=True
        elif ans==3:
            HST.histogram(data_df[data_df.columns[:-1]], \
                                   pic_file, "_features")
        elif ans==4:
            HST.histogram(data_df[data_df.columns[-1:]],\
                                   pic_file, "_targets")
        elif ans==5:
            if cr==False: cr_np = CRRL.correlation(data_df)
            CRRL.cr_heatmap (cr_np, pic_file, col_names)
            cr=True
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
                X, y = get_minimal_data(X, y)
            true_label, pred_label, clf = CLS.train (X, y, ans - 8)
            CLS.plot_confusion_matrix(true_label, pred_label, \
                             unique_labels, pic_file, title, cls_name[ans])
# -------------------------------------------------------------------------
# Minimal dataset.
# -------------------------------------------------------------------------
def get_minimal_data(features_np, targets_np):
    from sklearn.model_selection import train_test_split
    
    num_rows=len(features_np)
    num_cols=len(features_np[0])
    min_rows = 50 + 8*num_cols
    
    ratio = (min_rows/num_rows) * 0.7
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
    menu()    