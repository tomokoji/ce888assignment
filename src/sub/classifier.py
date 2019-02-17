"""
===========================================================================
                 c l a s s i f i e r . p y
---------------------------------------------------------------------------
This code analyses feature importance based on a Decision Tree classifier.

Author          : Tomoko Ayakawa
Created on      : 17 February 2019
Last modified on: 17 February 2019
===========================================================================
"""
import sys
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../")
from conf import myVariables as VAR

# -------------------------------------------------------------------------
# Set the predictors
# -------------------------------------------------------------------------
def set_predictors():
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.naive_bayes import GaussianNB 
    from sklearn import svm
    
    predictors = []
    predictors.append (ExtraTreesClassifier())
    predictors.append (GaussianNB ())
    predictors.append (svm.SVC (kernel = "rbf"))

    return predictors

# -------------------------------------------------------------------------
# Train a classifier and make a prediction
# ------------------------------------------------------------------------- 
def train(X, y, pred_id):
    from sklearn.model_selection import train_test_split
    
    predictors = set_predictors()
    clf = predictors[pred_id]
    X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.3)

    clf.fit(X_tr, y_tr)
    p = clf.predict(X_ts)
       
    return y_ts, p, clf

# -------------------------------------------------------------------------
# Display confusion matrix.
# This method is written in reference to CE888 Lab 3 script.
# ------------------------------------------------------------------------- 
def plot_confusion_matrix(y, p, unique_labels, pic_file, title, clf_type):
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    import itertools

    cm = confusion_matrix(y, p)
    
    fig=plt.figure()

    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(unique_labels))
    plt.xticks(tick_marks, unique_labels, rotation=90)
    plt.yticks(tick_marks, unique_labels)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        t = "%d"%(cm[i, j])
        plt.text(j, i, t,horizontalalignment="center",\
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    plt.show()
    print(classification_report(y, p))
    
    fig.savefig("%s%s_conf_matrix_%s.png" % \
                (VAR.out_path, pic_file, clf_type), bbox_inches='tight')
# -------------------------------------------------------------------------
# Compare the predictors
# ------------------------------------------------------------------------- 
def compare(X, y, predictors):
    from sklearn import metrics
    from sklearn.model_selection import cross_validate
    from sklearn.metrics.scorer import make_scorer
    
    labels = ["Decision Tree",  "Naive Bayse", "SVM"]
    mets = ["precision", "recall", "F-measure", "accuracy"]
    scores = []
    
    for clf in predictors:
        scoring = {"P": "precision_macro", "R": "recall_macro", \
                   "F": "f1_macro", \
                   "A": make_scorer (metrics.accuracy_score)}
        cv_scores = cross_validate (clf, X, y, scoring = scoring, \
                                    cv = 10, return_train_score = True)
    
        score = []
        for test_met in ["test_P", "test_R", "test_F", "test_A"]:
            score.append (np.mean (cv_scores[test_met]))
        scores.append (score)
        
    print (pd.DataFrame (scores, index = labels, columns = mets))
    
# -------------------------------------------------------------------------
# Allow the programme to be ran from the command line.
# -------------------------------------------------------------------------
if __name__ == '__main__':
    from sklearn.datasets import load_iris
    data = load_iris()
    features = data.data
    target= data.target
    unique_labels = np.sort(np.unique(target))
    
    true_label, pred_label, clf = train (features, target, 0)
    cnf_matrix = plot_confusion_matrix(true_label, pred_label, \
                                       unique_labels, "test", \
                                       "test plot", "DT")
                                      