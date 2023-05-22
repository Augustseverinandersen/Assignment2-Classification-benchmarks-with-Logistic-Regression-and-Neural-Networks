[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10448918&assignment_repo_type=AssignmentRepo)
# Assignment 2 - Classification benchmarks with Logistic Regression and Neural Networks
 
# Assignment 2 - Classification benchmarks with Logistic Regression and Neural Networks

## 2.1 Assignment Description
Written by Ross:
-	For this assignment, we'll be writing scripts that classify the Cifar10 dataset. You should write code that does the following: Load the CIFAR-10 dataset, and pre-process the data (e.g. greyscale, reshape). Train a classifier on the data and save a classification report. 
-	You should write one script which does this for a logistic regression classifier and one which does it for a neural network classifier. In both cases, you should use the machine learning tools available via scikit-learn.
## 2.2 Machine Specifications and My Usage
All the computation done for this project was performed on the UCloud interactive HPC system, which is managed by the eScience Center at the University of Southern Denmark. Python version 1.73.1. It took nine minutes to run on a 16-CPU machine, with `--tolerance` set at 0.04 in script `logistic_regression_classifier.py`, and `--max_iterations` set at 50 in script ``neural_network_classifier.py``.
### 2.2.1 Prerequisites

## 2.3 Contribution

### 2.3.1 Data


## 2.4 Packages


## 2.5 Repository contents


## 2.6 Methods / What the Code Does
### 2.6.1 Logistic Regression Script:


### 2.6.2 Neural Network Script:


## 2.7 Discussion

### 2.7.1 Logistic Regression vs Neural Network


### 2.7.2 Logistic Regression Tolerance


| Tolerance | Accuracy f1-score |
|-----------|------------------|
| 0.1       | 0.32             |
| 0.9       | 0.31             |
| 0.05      | 0.31             |
| 0.04      | 0.31             |
| 0.01      | 0.30             |



### 2.7.3 Neural Network Max Iteration


## 2.8 Usage


