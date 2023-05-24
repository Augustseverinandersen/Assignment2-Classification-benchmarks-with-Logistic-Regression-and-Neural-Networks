[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10448918&assignment_repo_type=AssignmentRepo)
# Assignment 2 - Classification benchmarks with Logistic Regression and Neural Networks

## 2.1 Assignment Description
Written by Ross:
For this assignment, we'll be writing scripts that classify the Cifar10 dataset. You should write code that does the following: Load the CIFAR-10 dataset, and pre-process the data (e.g. greyscale, reshape). Train a classifier on the data and save a classification report. 

You should write one script which does this for a logistic regression classifier and one which does it for a neural network classifier. In both cases, you should use the machine learning tools available via scikit-learn.
## 2.2 Machine Specifications and My Usage
All the computation done for this project was performed on the UCloud interactive HPC system, which is managed by the eScience Center at the University of Southern Denmark. This script ran on Coder Python 1.73.1 and Python version 3.9.2. It took nine minutes to run on a 16-CPU machine, with `--tolerance` set at 0.04 in script `logistic_regression_classifier.py`, and `--max_iterations` set at 50 in script ``neural_network_classifier.py``.
### 2.2.1 Prerequisites
To run this script, make sure to have Bash and Python 3 installed on your device. This script has only been tested on Ucloud. 
## 2.3 Contribution
The code in this repository was made in collaboration with my fellow students. 
The data used in this assignment was created by the _Canadian Institute for Advanced Research_ (CIFAR).
### 2.3.1 Data
CIFAR-10 consists of 10 classes, each containing 60 000 32x32 pixel colour images. The classes are _airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck_. Each class consists of 5 000 training images and 1 000 test images. According to the [documentation](https://www.cs.toronto.edu/~kriz/cifar.html), the classes do not overlap. Meaning that the class _automobile_ and _truck_ do not contain the same types of vehicles. When building the dataset, the following criteria were in place to decide if an image was good enough - [source](https://paperswithcode.com/dataset/cifar-10):
1. _The class name should be high on the list of likely answers to the question "What is in this picture?"_
2.	_The image should be photorealistic._
3.	_The image should contain only one prominent instance of the object to which the class refers._
## 2.4 Packages
These are the packages that I used to make the two scripts:
-	Os is used to navigate file paths, on different operating systems.
-	Numpy (version 1.24.2) is used to handle arrays and data munging.
-	Cv2 (version 4.7.0.72) is used to convert the images from colour to greyscale.
-	TensorFlow (version 2.11.0) is used to import the _CIFAR-10_ dataset.
-	Scikit-learn (version 1.2.2) is used to create a _train-test split_, create a _classification report_, create a _logistic regression_, and to create the _neural network_.
-	Argparse is used to make command line arguments.
## 2.5 Repository contents
This repository contains the following folders and file:
-	**out** this folder contains the classification reports
-	**src** this folder contains the two scripts ``logistic_regression_classifier.py`` and ``neural_network_classifier.py`` 
-	**README.md** is the README file for this repository
-	**requirements.txt** contains version-controlled packages that will be installed.
-	**setup.sh** the setup file, that creates a virtual environment, and installs packages from the requirements.txt file.
## 2.6 Methods / What the Code Does
### 2.6.1 Logistic Regression Script:
-	The logistic regression script starts by loading the CIFAR-10 dataset from TensorFlow into four variables. The four variables are train images and labels and test images and labels. 10 labels are also created which will be used in the classification report since the current labels are just numbers.
-	Next, the script converts the images into greyscale by using _cv2.cvtColor_ and specifying that it is colour images to greyscale. The grey-scaled images are stored as a NumPy array. 
-	After grey scaling the data is then rescaled by dividing by 255 (number of pixels). By doing so we normalize the data to between 0-1, making it easier to compute. 
-	The data is then reshaped. The function gets the shape of the data and splits it into three variables, nsamples which is the number of samples, nx (x dimensions) which is 32, and ny (y dimensions) which is 32. The data is then reshaped into a tuple by keeping nsamples and by multiplying nx with ny. This gives the data a new shape of 50 000 and 1024. 
-	The logistic regression is then created with the following arguments:
    - _penalty = none_. Keeping all weights
    - _tol = float(args.tolerance)_. The tolerance can be set by you. A default of 0.1 is given. Float transformation is used here, as the argparse takes the input as a string.
    - _verbose = True_. Prints the output to the command line.
     - _solver="saga"_. Is used for multiclass problems.
     - _multi_class="multinomial"_. Specifying a multiclass classification. 
-	The training data and labels are then fitted to the logistic regression, and the logistic regression is trained. 
-	Lastly, the script uses the test data to get predictions from the trained logistic regression model, a classification report of the predictions is created, and the model is saved.

### 2.6.2 Neural Network Script:
-	The neural network script loads the CIFAR-10 dataset into four variables, and creates label string names, for the dataset. 
-	The images are then converted to greyscale, rescaled, and reshaped in the same way as with the logistic regression script. 
-	The next step is the creation of the neural network with the following parameters:

	- _random_state=42_. This sets a seed, so every time the script is run it runs on the same seed. Otherwise, it would run on different seeds, making comparison difficult.
	- _hidden_layer_sizes_. This specifies the neural network’s structure, which is two layers. The first layer has 100 neurons and the second has 10 neurons.
	- _learning_rate="adaptive"_. By choosing adaptive, the learning rate changes during training. It starts quickly but as the model learns it gets slower and makes smaller adjustments. This makes the predictions more    fine-tuned.
	- _early_stopping=True_. By setting it to True the model stops when it is not improving anymore, thereby preventing overfitting the model.
	- _validation_fraction = 0.2_. Creating a validation split of 20% of the training images for each class.
	- _verbose = True_. Output is printed to the command line during training.
	- _max_iter=args.max_iteration_. The max iterations can be chosen by the user but a default of 20 is set. If max_iteration is reached the model stops training. 
-	The training data and labels are then fitted to the created neural network, and it starts training. 
-	Lastly, the model is then tested on the training data, and a classification report on the predictions is created and saved.
## 2.7 Discussion
### 2.7.1 Logistic Regression vs Neural Network
As expected, the neural network classifier performed better than the logistic regression. The _accuracy f1-score_ for the logistic regression was 0.32, but for the neural network, it was 0.39. The neural network performed better in all ten classes. The logistic regression had the highest _f1-score_ for the class truck with a score of 0.42 and the worst for the class cat with a score of 0.19. The neural network performed best on the class ship with a score of 0.49, and the work on class cat with a score of 0.23. The reason that the neural network performs better than the logistic regression is that the logistic regression has pre-made formulas for predicting the correct class. A neural network uses hidden layers and weights that are updated on each image, to get specific features or patterns from the images. Thereby, a neural network has a more precise reason for predicting the correct class when trained on the images, than a logistic regression.
### 2.7.2 Logistic Regression Tolerance
I tried five different tolerance settings when training my logistic regression to see how it would influence my model. Below is a table of the tolerance setting and _accuracy f1-score_. 

| Tolerance | Accuracy f1-score |
|-----------|------------------|
| 0.1       | 0.32             |
| 0.9       | 0.31             |
| 0.05      | 0.31             |
| 0.04      | 0.31             |
| 0.01      | 0.30             |

The lower the tolerance the more epochs the model took, this could mean that the model started overfitting too much. The highest _accuracy f1-score_ is for a tolerance setting of 0.1. It can be deduced that the lower the tolerance level the less accurate the predictions become.
### 2.7.3 Neural Network Max Iteration
I tried two different numbers of max iterations for the neural network. The first run had 20 max iterations, and the second had 50 max iterations. The first run stopped at 20 and had an accuracy f1-score of 0.38. The second run stopped at iteration 44 and had an accuracy f1-score of 0.40. The reason that this run stopped before max iterations were reached is because early stopping was on, to prevent overfitting. The run with 50 iterations had a higher score in all classes except cat. What can be deduced is that 20 iterations are not enough to train the model. However, with the class cat having a f1-score reduction of 0.5 (when max iteration was set at 50), could mean that more iterations could lead to a fall in accuracy.
## 2.8 Usage
To run the scripts in this repository, follow these steps:
-	Clone the repository.
- Navigate to the correct directory.
-	Run ``bash setup.sh`` in the command line. This will create a virtual environment and install the packages specified in the requirements files.
-	Run ``source ./assignment_2/bin/activate`` in the command-line, to activate the virtual environment.
-	In the command line run ``python3 src/logistic_regression_classifier.py --tolerance 0.1``. This will run the logistic regression classifier.
	 - The argparse ``--tolerance`` has a default value of 0.1. You can change it to see how the model performance changes.  The argparse takes a string as input but is changed to a float in the script. 
-	In the command line run ``python3 src/neural_network_classifier.py --max_iterations 50``. This will run the neural network classifier.
	 - The argparse ``--max_iterations`` has a default value of 20. You can change it to see how the model’s performance differs. The argparse takes an integer as input.
