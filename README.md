[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10448918&assignment_repo_type=AssignmentRepo)
# Assignment 2 - Classification benchmarks with Logistic Regression and Neural Networks
 
# Assignment 2 - Classification benchmarks with Logistic Regression and Neural Networks

## 2.1 Assignment Description
Written by Ross:

For this assignment, we'll be writing scripts that classify the Cifar10 dataset. You should write code that does the following:
- Load the CIFAR-10 dataset and pre-process the data (e.g., greyscale, reshape).
- Train a classifier on the data and save a classification report.

You should write one script that does this for a logistic regression classifier and one that does it for a neural network classifier. In both cases, you should use the machine learning tools available via scikit-learn.

## 2.2 Machine Specifications and My Usage
All the computation done for this project was performed on the UCloud interactive HPC system, which is managed by the eScience Center at the University of Southern Denmark. Python version 1.73.1 was used. It took nine minutes to run on a 16-CPU machine, with `--tolerance` set at 0.04 in the script `logistic_regression_classifier.py`, and `--max_iterations` set at 50 in the script `neural_network_classifier.py`.

### 2.2.1 Prerequisites
To run this script, make sure to have Bash and Python 3 installed on your device. This script has only been tested on UCloud.

## 2.3 Contribution
The code in this repository was made in collaboration with my fellow students. The data used in this assignment was created by the Canadian Institute for Advanced Research (CIFAR).

### 2.3.1 Data
CIFAR-10 consists of 10 classes, each containing 60,000 32x32 pixel color images. The classes are airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. Each class consists of 5,000 training images and 1,000 test images. According to the documentation, the classes do not overlap, meaning that the class automobile and truck do not contain the same types of vehicles. When building the dataset, the following criteria were in place to decide if an image was good enough:
1. The class name should be high on the list of likely answers to the question "What is in this picture?"
2. The image should be photorealistic.
3. The image should contain only one prominent instance of the object to which the class refers.

## 2.4 Packages
These are the packages that I used to make the two scripts:
- `os` is used to navigate file paths on different operating systems.
- `numpy` (version 1.24.2) is used to handle arrays and data munging.
- `cv2` (version 4.7.0.72) is used to convert the images from color to greyscale.
- `TensorFlow` (version 2.11.0) is used to import the CIFAR-10 dataset.
- `scikit-learn` (version 1.2.2) is used to create a train-test split, create a classification report, create a logistic regression, and to create the neural network.
- `argparse` is used to make command-line arguments.

## 2.5 Repository contents
This repository contains the following folders and files:
- `out`: This folder contains the classification reports.
- `src`: This folder contains the two scripts `logistic_regression_classifier.py` and `neural_network_classifier.py`.
- `README.md`: This is the README file for this repository.
- `requirements.txt`: This file contains version-controlled packages that will be installed.
- `setup.sh`: This is the setup file that creates a virtual environment and installs packages from the `requirements.txt` file.

## 2.6 Methods / What the Code Does

### 2.6.1 Logistic Regression Script:
The logistic regression script starts by loading the CIFAR-10 dataset from TensorFlow into four variables: train images and labels, and test images and labels. It also creates 10 label string names for the dataset. The script then performs the following steps:
1. Convert the images into greyscale using `cv2.cvtColor` and store the greyscaled images as a NumPy array.
2. Rescale the data by dividing by 255 (number of pixels) to normalize it between 0-1.
3. Reshape the data using the shape of the data and splitting it into three variables: `nsamples` (number of samples), `nx` (x dimensions), and `ny` (y dimensions). Reshape the data into a tuple by keeping `nsamples` and multiplying `nx` with `ny`, resulting in a new shape of 50,000 and 1024.
4. Create a logistic regression classifier using scikit-learn with the specified arguments.
5. Fit the training data and labels to the logistic regression and train the model.
6. Use the test data to get predictions from the trained logistic regression model.
7. Create a classification report of the predictions and save the model.

### 2.6.2 Neural Network Script:
The neural network script follows a similar structure as the logistic regression script. It loads the CIFAR-10 dataset, converts the images to greyscale, rescales and reshapes the data. The script then performs the following steps:
1. Create a neural network classifier using scikit-learn with the specified parameters.
2. Fit the training data and labels to the neural network and start training.
3. Test the model on the training data and create a classification report of the predictions.

## 2.7 Discussion

### 2.7.1 Logistic Regression vs Neural Network
As expected, the neural network classifier performed better than the logistic regression. The accuracy f1-score for the logistic regression was 0.32, but for the neural network, it was 0.39. The neural network performed better in all ten classes. The logistic regression had the highest f1-score for the class truck with a score of 0.42 and the worst for the class cat with a score of 0.19. The neural network performed best on the class ship with a score of 0.49 and worst on the class cat with a score of 0.23. The reason the neural network performs better than the logistic regression is that the neural network uses hidden layers and weights that are updated on each image, allowing it to capture specific features or patterns from the images.

### 2.7.2 Logistic Regression Tolerance
Different tolerance settings were tested for the logistic regression model. The table below shows the tolerance settings and the corresponding accuracy f1-scores:

| Tolerance | Accuracy f1-score |
|-----------|------------------|
| 0.1       | 0.32             |
| 0.9       | 0.31             |
| 0.05      | 0.31             |
| 0.04      | 0.31             |
| 0.01      | 0.30             |

Lower tolerance values resulted in more epochs for the model, but the accuracy of predictions decreased. The highest accuracy f1-score was achieved with a tolerance setting of 0.1. It can be deduced that lower tolerance levels lead to less accurate predictions.

### 2.7.3 Neural Network Max Iteration
Two different numbers of max iterations were tested for the neural network model. The first run had 20 max iterations, and the second had 50 max iterations. The first run stopped at 20 iterations with an accuracy f1-score of 0.38. The second run stopped at iteration 44 with an accuracy f1-score of 0.40. Early stopping was enabled to prevent overfitting, so the second run stopped before reaching the maximum iterations. The run with 50 iterations had a higher score in all classes except for cat. It can be deduced that 20 iterations were not enough to train the model adequately. However, increasing the max iterations could lead to a decrease in accuracy for the class cat.

## 2.8 Usage
To run the scripts in this repository, follow these steps:
1. Clone the repository.
2. Run `bash setup.sh` in the command line to create a virtual environment and install the specified packages from the `requirements.txt` file.
3. Run `source ./assignment_2/bin/activate` in the command line to activate the virtual environment.
4. To run the logistic regression classifier, execute the following command in the command line: `python3 src/logistic_regression_classifier.py --tolerance 0.1`. The `--tolerance` argument can be changed to adjust the model's performance. The default value is 0.1.
5. To run the neural network classifier, execute the following command in the command line: `python3 src/neural_network_classifier.py --max_iterations 50`. The `--max_iterations` argument can be changed to adjust the model's performance. The default value is 20.

