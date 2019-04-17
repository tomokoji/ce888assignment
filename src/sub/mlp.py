"""
===========================================================================
                                  m l p . p y
---------------------------------------------------------------------------
This code is to build and train an malti layer perceotron as a classifier
(discriminative neural network).

Author          : Tomoko Ayakawa
Created on      : 17 April 2019
Last modified on: 17 April 2019
===========================================================================
"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../")
# import original libraries
from conf import myVariables as VAR
import nn_parameters as PARA

# -------------------------------------------------------------------------
# Define the parameters for the mlp. When skipped or an invalid 
# value is given, the default values that are defined in conf.py are used.
# -------------------------------------------------------------------------
def get_parameters():
    num_para=10
    PARA.msg(num_para)
    
    i=1
    # Conduct fine tune of weights between layers of the autoencoder
    finetune=PARA.display(i, num_para, VAR.finetune, \
                          "Finetune autoencoder")
    
    # Number of neurons of the hidden layer
    i+=1
    h_num=PARA.hidden_neurons(i, num_para, VAR.h_num)

    # Number of neurons in each layer
    i+=1
    h_act=PARA.activation(i, num_para, VAR.h_act, " (hidden layer)")

    # Activation function
    i+=1
    out_act=PARA.activation(i, num_para, VAR.out_act, " (hidden layer)")

    # Optimiser
    i+=1
    opt=PARA.optimiser(i, num_para, VAR.mlp_opt, VAR.mlp_lr, \
                       VAR.mlp_momentum)

    # Loss function
    i+=1
    loss=PARA.loss(i, num_para, VAR.mlp_loss)
       
    # Trainig epochs
    i+=1
    epochs=PARA.training_epoch(i, num_para, VAR.mlp_epoch)
    
    # Validation rate
    i+=1
    val_rate=PARA.validation(i, num_para, VAR.val_rate)

    # Verbose
    i+=1
    verbose=PARA.display(i, num_para, VAR.mlp_verbose, "Verbose")
    
    # summary_display
    i+=1
    summary_display=PARA.display(i, num_para, VAR.mlp_summary_display, \
                                 "Summary display")
    
    print("\nParameters for the MLP are")
    print(" 1. Finetune: : %d(%s)" % (finetune, VAR.yesno[finetune]))
    print(" 2. Number of neurons in the hidden layer: %d" % (h_num))
    print(" 3. Hidden layer activation function: %s" % (h_act))
    print(" 4. Output layer activation function: %s" % (out_act))
    print(" 5. Optimiser:", opt)
    print(" 6. Loss function:", loss)
    print(" 7. Epochs: %d" % epochs)
    print(" 8. Validation rate: %f" % val_rate)
    print(" 9. Verbose: %d(%s)" % (verbose, VAR.yesno[verbose]))
    print(" 10. Summary_display: %d(%s)" % (summary_display, \
          VAR.yesno[summary_display]))
    
    return finetune, h_num, h_act, out_act, opt, loss, epochs, val_rate, \
        verbose, summary_display
    
# -------------------------------------------------------------------------
# Build an MLP classifier.
# -------------------------------------------------------------------------
def build_mlp(encoder, num_in, num_out, finetune=VAR.finetune, \
        h_num=VAR.h_num, h_act=VAR.h_act, out_act=VAR.out_act, \
        opt=VAR.mlp_opt, loss=VAR.mlp_loss, \
        summary_display=VAR.mlp_summary_display):
    
    from keras.models import Sequential
    from keras.layers import Dense
    
    # reconstruct the autoencoder
    model=Sequential()
    for e in encoder.layers:
        if finetune==1: e.trainable=True
        elif finetune==0: e.trainable=False
        model.add(e)
        
    model.add(Dense(h_num, input_dim=num_in, activation=h_act))
    model.add(Dense(num_out, activation=out_act))

    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
    
    if summary_display==1: print(model.summary())
    
    return model

# ------------------------------------------------------------------------
# Train the MLP classifier.
# ------------------------------------------------------------------------
def train_mlp(X, y, model, epochs=VAR.mlp_epoch, val_rate=VAR.val_rate, \
              verbose=VAR.mlp_verbose):  
    # Replate the target with dummy values
    y=multi_class(y)
    
    # train the mlp
    loss_hiss=model.fit(X, y, epochs=epochs, validation_split=val_rate, \
                      verbose=verbose)
    
    return loss_hiss
    
# -------------------------------------------------------------------------
# Replace the target with binary matrices (1s and 0s) of shape 
# (samples, classes).
# -------------------------------------------------------------------------
def multi_class(y):
    from keras.utils import np_utils
    from sklearn.preprocessing import LabelEncoder
   
    # encode class values as integers
    label_encoder=LabelEncoder()
    encoded_Y=label_encoder.fit_transform(y)

    # convert integers to dummy variables (i.e. one hot encoded)
    encoded_Y=np_utils.to_categorical(encoded_Y)

    return encoded_Y

# -------------------------------------------------------------------------
# Plot the loss history and save it as a picture.
# -------------------------------------------------------------------------
def plot_mlp_loss_history(histories, pic_file):
    import matplotlib.ticker as ticker
    
    ans=input("Save training loss history as a picture? (y/n): ")
    
    if (ans=="y") or (ans=="Y"): save=True
    else: save=False 
        
    fig=plt.figure(figsize=(15,5))

    metrics=[["loss", "val_loss"], ["acc", "val_acc"]]
    labels=["Loss", "Accuracy"]
    for i, m , l in zip(range(1,3), metrics, labels):
        plt.subplot(1, 2, i)
        plt.plot(histories.history[m[0]], label="training")
        plt.plot(histories.history[m[1]], label="validation")
        plt.title("MLP Training and Validation %s" % l)
        plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator
               (integer=True))
        plt.legend()

    plt.show ()
    
    if save==True:
        fig.savefig("%s%s_ae_loss.png" % (VAR.out_path, pic_file), \
                bbox_inches='tight')  
        
# -------------------------------------------------------------------------
# Cross validation of the model to obrain accuracy with 95% confidence 
# interval.
# -------------------------------------------------------------------------
def cross_validation(X, y, CV):
    from keras.wrappers.scikit_learn import KerasClassifier
    from sklearn.model_selection import cross_val_score

    estimator=KerasClassifier(build_fn=CV, epochs=100, verbose=0)
    y=multi_class(y)
    scores=cross_val_score(estimator, X, y, cv=5)
    
    print("Accuracy (95%% confidence interval): %0.2f (+/- %0.2f)" \
          % (scores.mean(), scores.std() * 2))
    
    return scores
# -------------------------------------------------------------------------
# Allow the programme to be ran from the command line.
# -------------------------------------------------------------------------
if __name__ == "__main__":
    import autoencoder as AE
    
    X=np.random.rand(10,100)
    y=[0,1,2,3,4]*2
    
    finetune, h_num, h_act, out_act, opt, loss, epochs, val_rate, \
        verbose, summary_display=get_parameters()
    
    ans=input("Continue? (y/n): ")
    
    # prepare a mock autoencoder
    ae_layers=[50, 30, 10]
    encoder, histories=AE.autoencoder(X, layers=ae_layers, mode=1, \
                    act="relu", opt="adam", loss="mse", dropout=0, \
                    epochs=10, verbose=0, summary_display=0)
    
    # train mlp
    if (ans=="y") or (ans=="Y"):
        k=ae_layers[-1]
        n=len(np.unique(y))
    
        model=build_mlp(encoder, num_in=k, num_out=n, finetune=finetune, \
            h_num=h_num, h_act=h_act, out_act=out_act, \
            opt=opt, loss=loss, summary_display=summary_display)
        
        histories=train_mlp(X, y, model, epochs=epochs, \
                            val_rate=val_rate, verbose=verbose)
    
        plot_mlp_loss_history(histories, "test")

        #scores=mlp.cross_validation(X, y)
        