# ce888assignment
This repository contains deliverables of CE888 assignment1. <br>

## Contents
1. data: training data set collected for the assignment
    - human_activity (source: [Human Activity data set at UCI](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones))
    - spam (source: [Spambase data set at UCI](https://archive.ics.uci.edu/ml/datasets/spambase))
    - phishing (source: [Phishing website data set at UCI](https://archive.ics.uci.edu/ml/datasets/phishing+websites))
2. src: python scripts to load and analyse the data
    - **assignment1_main.py**: the main script to read and analyse the data
    - **assignment1_for_demo.ipynb**: jupyter notebook for demo
    - **conf.py**: configurable variables used by the python scripts
    - sub: directory containing sub modules called by *assignment1_main.py*
3. output: png files of data plot

## Usage
The programme can be run  in *src* directory from command line as:
 `python3 assignment1_main.py`
<br>
*assignment1_main.py* calls the menus to load and analyse the data.

## Demo
Demonstration of the programme is available on *ce888_assignment1_for_demo.ipynb* in *src* directory.
