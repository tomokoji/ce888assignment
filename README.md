# ce888assignment
This repository contains deliverables of CE888 assignment1. <br>

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
		- **assignment2_main.py**: the main script to do build and train autoencoder and a discriminative neural network
		- **assignment2_for_demo.ipynb**: jupyter notebook for demo
    - **conf.py**: configurable variables used by the python scripts
    - sub: directory containing sub modules called by *assignment1_main.py* and *assignment2_main.py* <br>
	|file name|Developed for assignment1|Modified for assignment2|Developed for assignment2|
	|-----|-----|-----|-----|
	|load_data.py|Yes|Yes||
	|load_human_activity.py|Yes|-|-|
	|load_spam.py|Yes|-||
	|load_phishing.py|Yes|Yes|-|
	|histogram.py|Yes|-|-|
	|correlation.py|Yes|-|-|
	|feature_importance.py|Yes|-|-|
	|classifier.py|Yes|Yes|-|
	|pca.py|Yes|Yes|-|
	|preprocess.py|-|-|Yes|
	|nn_parameters.py|-|-|Yes|
	|autoencoder.py|-|-|Yes|
	|grid_search.py|-|-|Yes|
	|mlp.py|-|-|Yes|
	|comparison.py|-|-|Yes|
3. output_ass1: png files of data plot that are created for the assignment 1
4. output_ass2: png files of data plot that are created for the assignment 2

## Usage
The programme can be run  in *src* directory from command line as:
 `python3 assignment1_main.py`
<br>
*assignment1_main.py* calls the menus to load and analyse the data.

## Demo
Demonstration of the programme is available on *ce888_assignment1_for_demo.ipynb* in *src* directory.
