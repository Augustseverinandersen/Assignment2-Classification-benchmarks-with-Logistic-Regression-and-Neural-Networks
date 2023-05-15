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
  - Is used to navigate filepaths 
- Numpy 
  - Is used to handle arrays, and data munging.
- Cv2
  - Is used to convert the images from colour to greyscale.
- Tensorflow 
  - Tensorflow is used to import the CIFAR-10 dataset.
- Scikit Learn
  - Is used to create a train-test split, create a classification report, create a logistic regression, and to create the neural network.
- Argparse 
  - Is used to make command-line arguments.

## Repository contents
- This repository contains the following folders and file:
  - assignment_2
    - This folder contains the virtual environment
  - out 
    - This folder contains the classification reports
  - src
    - This folder contains the two scripts, *logistic_regression_classifier.py* and *neural_network_classifier.py*
  - README.md
    - The README file for this repository
  - requirements.txt
    - Version controlled packages that will be installed.
  - setup.sh
    - The setup file, that creates a virtual environment, and installs the requirements.txt file.

## Machine Specifications and my usage
  - I created this repository on Ucloud. It took nine minutes to run on a 16-CPU machine, with ```--tolerance``` set on 0.04 in script *logistic_regression_classifier.py*, and ```--max_iterations``` set on 50 in script *neural_network_classifier.py.*

## Assignment Description
For this assignment, we'll be writing scripts which classify the ```Cifar10``` dataset.
You should write code which does the following:

- Load the Cifar10 dataset
- Preprocess the data (e.g. greyscale, reshape)
- Train a classifier on the data
- Save a classification report

You should write one script which does this for a logistic regression classifier **and** one which does it for a neural network classifier. In both cases, you should use the machine learning tools available via ```scikit-learn```.

## Methods / What the code does
__Logistic Regression Script:__
- The logistic regression script starts by loading the CIFAR-10 dataset from tensorflow into four variables. The four variables are train images and labels, test images and labels. 10 labels are also created which will be used in the classification report, since the current labels are just numbers. Next, the script converts the images into greyscale by using *cv2.cvtColor,* and specifing that it is colour images to greyscale. The greyscaled image is stored as a NumPy array. After greyscaling the data is than rescaled in size by dividing by 255. By doing so we normalise the data to between 0-1. The next function in the script, is reshaping the grayscale image data. The function gets the shape of the data and splits it into three variables, *nsamples* which is the number of samples, *nx* x dimension which is 32, and *ny* y dimension which is 32. The data is than reshaped into a tuple by keeping *nsamples* and by multiplying *nx* with *ny*. This gives the data a new shape of 50 000 and 1024. The logisitic regression is than created with the following arguments: 
  - *penalty = none*. Keeping all weights 
  - *tol = float(args.tolerance)*. The tolerance can be set by the you. A default of 0.1 is given. Float transformation is used here, as the argparse takes the input as a string.
  - *verbose = True*. Prints the output to the command-line 
  - *solver="saga"*. Is used for multiclass problems.
  - *multi_class="multinomial"*. Specifing a multiclass classification.
The training data and labels are then fitted to the logistic regression, and the logistic regression is trained. Lastly, the script uses the test data to get predictions from the trained logistic regression model, a classification report of the predictions is created, and the model is saved.

__Neural Network Script:__
- The nerual network script loads the CIFAR-10 dataset into four variables, and creates label string names, for the dataset. The images are than  converted to greyscale, rescaled, and reshaped the same way as with the logistic regression script. The next step is the creation of the neural network with the following parameters:
  - *random_state=42*. This sets a seed, so every time the script is run it runs on the same seed. Otherwise, it would run on different seeds, making comparison difficult.
  - *hidden_layer_sizes*. This specifies the neural networks structure, which is two layers. The first layer has 100 neurons and the second has 10 neurons. 
  - *learning_rate="adaptive"*. By choosing *adaptive*, the learning rate changes during training. It starts quick but as the model learns it gets slower and makes smaller adjustments. This makes the predictions more fine-tuned.
  - *early_stopping=True*. By setting it to *True* the model stops when it is not improving anymore, thereby preventing overfitting the model.
  - *validation_fraction = 0.2*. Creating a validation split of 20% of the training images for each class.
  - *verbose = True*. Output is printed to the command-line during training. 
  - *max_iter=args.max_iteration*. The max iterations can be chosen by the user but a default of 20 is set. If *max_iteration* is reached the model stops training. 
The training data and labels are then fitted to the created neural network and it starts training. Lastly, the model is than tested on the training data, a classifiation report on the predictions is created, and saved.
## Discussion 
### Logistic Regression vs Neural Network
- As aspected the neural network classifier performed better than the logistic regression. The accuracy f1-score for the logistic regression was 0.32, but for the neural network it was 0.39. The neural network performed better in all ten classes. The logistic regression had the highest f1-score  for the class *truck* with a score of 0.42 and the worst for class *cat* with a score of 0.19. The neural network performed best on class *ship* with a score of 0.49, and the work on class *cat* with a score of 0.23. The reason that the neural network performences better than the logistic regression is that the logistic regression has pre-made formulas for predicting the correct class. A neural network uses hidden layers and weights that are updated on each images, to get specific features or patterens from the images. Thereby, a neural network has a more precise reason for predicting the correct class, when trained on the images. 
### Logistic Regression Tolerance 
- I tried five different tolerance settings when training my logistic regression to see how it would influence my model. Below is a table of tolerance setting and accuracy f1-score.
|Tolerance| Accuracy F1-score|
|0.1|0.32|
|0.09|0.31|
|0.05|0.31|
|0.04|0.31|
|0.01|0.30|
- The lower the tolerance the more epochs the model took, this could mean that the model started overfitting to mush. The highest accuracy f1-score is for a tolerance setting of 0.1. It can be deduced that the lower the tolerence level the less accurate the predictions become.

### Neural Network Max Iteration
- I tried two different amount of max iterations for the neural network. The first run had 20 max iterations, and the second had 50 max iterations. The first run stopped at 20, and had an accuracy f1-score of 0.38. The second run stopped at iteration 44 and had an accuracy f1-score of 0.40. The reason that this run stopped earlier is because earlier stopping was on, to prevent overfitting. The run with 50 iterations had a higher score in all classes except *cat*. What can be decuded is that 20 iterations leaves the model underfitting. However, with the class *cat* having a f1-score reduction of 0.5, the more iterations could lead to fall in accuracy.


## Usage
To run the scripts in this repository follow these steps:
1. Clone the repository
2. Run ```bash setup.sh``` in the command-line, which will create a virtual environment, and install the requirements. 
3. Run ```source ./assignment_2/bin/activate``` in the command-line, to activate the virtual environment.
4. In the command-line run ```python3 src/logistic_regression_classifier.py --tolerance 0.04```. This will run the logistic regression classifier 
  - ```--tolerance``` has a default value of 0.1. You can change it to see how classification report changes.
5. In the command-line run ```python3 src/neural_network_classifier.py --max_iterations 50```. This will run the neural network classifier.
  ```--max_iterations``` has a default value of 20. You can change it to see how the models performance differs.
