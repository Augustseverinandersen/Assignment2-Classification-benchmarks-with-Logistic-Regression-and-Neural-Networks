[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10448918&assignment_repo_type=AssignmentRepo)
# Assignment 2 - Classification benchmarks with Logistic Regression and Neural Networks
## Contribution
- The code in this repository was made in collaboration with my fellow students. 
- In code comments are made by me.
### Data
- The data used in this assignment was created by the *Canadian Institute for Advanced Research* (CIFAR). CIFAR-10 consists of 10 classes, each contain 60 000 32x32 colour images. The classes are *airplane, automobile, bird, cat, deer, dog, frog, horse, ship,* and *truck.* Each class consists of 5 000 training images and 1 000 test images. According to the [documentation](https://www.cs.toronto.edu/~kriz/cifar.html), the classes do not overlap. Meaning that the class *automobile* and *truck* do not contain the same types of vechiles. When building the dataset the following criterie was in place to decide if an image was good enough [source](https://paperswithcode.com/dataset/cifar-10):
  1. *The class name should be high on the list of likely answers to the question "What is in this picture?"*
  2. *The image should be photo-realistic*
  3. *The image should contain only one prominent instance of the object to which the class refers*
## Packages
- Os
- Numpy 
- Cv2
- Tensorflow 
- Scikit Learn
- 
## Repository contents
## Machine Specifications and my usage
## Assignment Description
For this assignment, we'll be writing scripts which classify the ```Cifar10``` dataset.

You should write code which does the following:

- Load the Cifar10 dataset
- Preprocess the data (e.g. greyscale, reshape)
- Train a classifier on the data
- Save a classification report

You should write one script which does this for a logistic regression classifier **and** one which does it for a neural network classifier. In both cases, you should use the machine learning tools available via ```scikit-learn```.
## Methods / What the code does
## Discussion 
## Usage





## Tips

- You should structure your project by having scripts saved in a folder called ```src```, and have a folder called ```out``` where you save the classification reports.
- Consider using some of the things we've seen in class, such as virtual environments and setup scripts.

## Purpose

- To ensure that you can use ```scikit-learn``` to build simple benchmark classifiers on image classification data
- To demonstrate that you can build reproducible pipelines for machine learning projects
- To make sure that you can structure repos appropriately
