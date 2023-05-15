
# Importing Libraries 

# System tools
import os

# Data munging
import numpy as np
import cv2
import argparse

# Data loader
from tensorflow.keras.datasets import cifar10

# Mchine learning tools
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Neural Network
from sklearn.neural_network import MLPClassifier



def input_parse():
    # initialize the parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_iterations", type=int , default = 20, help = "Set the max amount of iterations for the model")
    # parse the arguments from command line
    args = parser.parse_args()
    return args

# Getting the data 
def load_data():
    print("Getting the data")
    (X_train, y_train), (X_test, y_test) = cifar10.load_data() # Loading the cifar 10 dataset

    # Creating labels to be used later 
    # the current labels are just numbers from 1-9
    labels = ["airplane", 
            "automobile", 
            "bird", 
            "cat", 
            "deer", 
            "dog", 
            "frog", 
            "horse", 
            "ship", 
            "truck"]
    return X_train, y_train, X_test, y_test, labels


# Converting to grey scale 
def greyscale(data_train, data_test):
    print("Converting to greyscale")
    X_train_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in data_train])
    X_test_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in data_test])
    # Using opencvs colour conversion model 
    # Create a np.array which is made from the list we make when converging images to grey scale for every image in X_test

    # Creating smaller weights 
    X_train_scaled = (X_train_grey)/255.0
    X_test_scaled = (X_test_grey)/255.0
    # Scaling numbers down to between 0-1
    return X_train_scaled, X_test_scaled


# Reshaping the training data  
def reshaping(data_train_greyscale, data_test_greyscale):
    print("Reshaping the data")
    nsamples, nx, ny = data_train_greyscale.shape
    # new shape to be two values, 50 000 and 1024 (32 times by 32)  
    X_train_dataset = data_train_greyscale.reshape((nsamples,nx*ny))

    # Reshaping the test data 
    nsamples, nx, ny = data_test_greyscale.shape
    X_test_dataset = data_test_greyscale.reshape((nsamples,nx*ny))
    
    return X_train_dataset, X_test_dataset


# Neural Network
def neural_network_function(X_train_dataset, y_train, args):
    print("Training neural network")
    clf = MLPClassifier(random_state=42, # Keeping it reproducible 
                        hidden_layer_sizes=(100, 10), # Two hidden layers, everytime it goes from one layer to another and a weight changes it goes back and starts agian.
                        learning_rate="adaptive", # Beginning of the model it will just be guessing. we want it to learn quickly. as soon as it learns a bit, we get it to slow down, and think more about how to predict.
                        early_stopping=True, # Stop early if it is not getting better scores. 
                        validation_fraction = 0.2, # Creating a validation split aswell of 20% of the training images
                        verbose=True, # Print the status to the command-line
                        max_iter=args.max_iterations).fit(X_train_dataset, y_train) # max iteratoin of 20. Fitting on training data

    return clf

# Getting the predictions 
def prediction(model, data):
    print("Predictions")
    y_pred = model.predict(data) # Testing the model on test data

    return y_pred


# Creating a report 
def validation(test_labels, test_prediction, my_labels):
    print("Validation") 
    report = classification_report(test_labels, # Creating a classification report
                                test_prediction, 
                                target_names=my_labels) # setting labels 
    return report


# Saving the report 
def saving_report(data_report):
    folder_path = os.path.join("out") # Save folder
    file_name = "neural_network_classifier_metrics.txt" # Save name
    file_path = os.path.join(folder_path, file_name) # Joing the two together

    with open(file_path, "w") as f: # "Writing" the classifier metrics, thereby saving it.
        f.write(data_report)
    print("Reports saved")


def main_function():
    print("Neural Network Script:")
    args = input_parse() # Command line arguments
    X_train, y_train, X_test, y_test, labels = load_data() # Loading the data
    X_train_greyscale, X_test_grayscale = greyscale(X_train, X_test) # Converting to greyscale
    X_train_dataset_done, X_test_dataset_done = reshaping(X_train_greyscale, X_test_grayscale) # Reshaping it
    clf_done = neural_network_function(X_train_dataset_done, y_train, args) # Creating the neural network and training it
    y_prediction = prediction(clf_done, X_test_dataset_done) # Testing it
    calculated_report = validation(y_test, y_prediction, labels) # Creating a classification report
    saving_report(calculated_report) # Saving the report 

if __name__ == "__main__": # If this script is called from the terminal run the main function
    main_function()