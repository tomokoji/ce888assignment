# ce888assignment
This repository contains deliverables of CE888 assignment 1 and 2. <br>

## Contents
1. data: training data set collected for the assignment
    - human_activity (source: [Human Activity data set at UCI](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones))
    - spam (source: [Spambase data set at UCI](https://archive.ics.uci.edu/ml/datasets/spambase))
    - phishing (source: [Phishing website data set at UCI](https://archive.ics.uci.edu/ml/datasets/phishing+websites))
2. src: python scripts developed for the project
    - For Assignment 1
		- **assignment1_main.py**: the main script to read and analyse the data
		- **assignment1_for_demo.ipynb**: jupyter notebook for demo
    - For Assignment 2
		- **assignment2_main.py**: the main script to do build and train an autoencoder and a discriminative neural network
		- **assignment2_for_demo.ipynb**: jupyter notebook for demo
    - **conf.py**: configurable variables used by the python scripts
    - sub: directory containing sub modules called by *assignment1_main.py* and *assignment2_main.py* <br>
	
	|#|File name|Description|Developed for assignment1|Modified for assignment2|Developed for assignment2|
	|-----|-----|-----|-----|-----|-----|
	|1|load_data.py|Load datasets and convert them into the same DataFrame format|Yes|Yes||
	|2|load_human_activity.py|Read source files for human activity dataset|Yes|-|-|
	|3|load_spam.py|Read source files for spam base dataset|Yes|-||
	|4|load_phishing.py|Read source files for phishing website dataset|Yes|Yes|-|
	|5|histogram.py|Display and save histogram of data|Yes|-|-|
	|6|correlation.py|Display and saves correlation of the features|Yes|-|-|
	|7|feature_importance.py|Analyse feature importance based on a Decision Tree classifier|Yes|-|-|
	|8|classifier.py|Build and train an classifiers (DT, Bayes, SVM) and show the results|Yes|Yes|-|
	|9|pca.py|Carry out PCA and plot the data in 3D|Yes|Yes|-|
	|10|preprocess.py|Pre-process the data|-|-|Yes|
	|11|nn_parameters.py|Obtain parameters for neural networks|-|-|Yes|
	|12|autoencoder.py|Build and train an autoencoder for feature learning|-|-|Yes|
	|13|grid_search.py|Carry out grid-search for the optimal parameters|-|-|Yes|
	|14|mlp.py|Build, train and evaluate a discriminative neural network|-|-|Yes|
	|15|comparison.py|Compare the classification results by an autoencoder with other classifier|-|-|Yes|
	
3. output_ass1: png files of data plot that are created for the assignment 1
4. output_ass2: png files of data plot that are created for the assignment 2

## Usage
The programme can be run  in *src* directory from command line as:
 `python3 assignment2_main.py`
<br>
*assignment2_main.py* calls the menus to load the data, build an autoencoder and a classifier, evaluate and compare the results.

## Demo
Demonstration of the programme is available on *ce888_assignment2_for_demo.ipynb* in *src* directory.
