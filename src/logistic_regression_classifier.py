
# Importing Libraries 

# Data Munging
import numpy as np
import cv2
import argparse

# System tools
import os

# Data loader
from tensorflow.keras.datasets import cifar10

# Machine learning tools
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Logistic Regression
from sklearn.linear_model import LogisticRegression

def input_parse():
    # initialize the parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--tolerance", type=str , default = 0.1, help = "Set the tolerance level for the logistic regression training")
    # parse the arguments from command line
    args = parser.parse_args()
    return args

# Getting the data 
def load_data():
    print("Getting the data")
    (X_train, y_train), (X_test, y_test) = cifar10.load_data() # From tensorflow I am loading the cifar images, into four variables

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
    nsamples, nx, ny = data_train_greyscale.shape # Getting the shape of the current train greyscale
    # new shape to be two values, 50 000 and 1024
    X_train_dataset = data_train_greyscale.reshape((nsamples,nx*ny))

    # Reshaping the test data 
    nsamples, nx, ny = data_test_greyscale.shape
    X_test_dataset = data_test_greyscale.reshape((nsamples,nx*ny))
    
    return X_train_dataset, X_test_dataset


# Logisitic regression 
def logistic_regression_classifier(X_train_dataset, y_train, args):
    print("Training Logistic Regression")
    clf = LogisticRegression(penalty="none", # Math about logistic regression. Dont use any penalty values. Use all weights 
    # When penaly is set to l1 = if a value is very small set it to zero. Keeping only meaningful weights 
                            tol=float(args.tolerance), # Tolerance, if the weights change by 0.1 or less, stop training.
                            verbose=True, # Print an output about model preformance. 
                            solver="saga", # multiclass dataset 
                            multi_class="multinomial").fit(X_train_dataset, y_train) # multiclass problem. Fitting on train data 
    return clf


# Getting the predictions 
def prediction(model, data):
    print("Predictions")
    y_pred = model.predict(data) # Using test data to get predictions
    return y_pred


# Creating a report
def validation(test_labels, test_prediction, my_labels): 
    print("Validation")
    report = classification_report(test_labels, 
                                test_prediction, 
                                target_names=my_labels) # setting labels (The ones created earlier)
    return report


# Saving the report 
def saving_report(data_report):
    folder_path = os.path.join("out") # Defining folder 
    file_name = "logistic_reg_classifier_metrics.txt" # Giving it a name
    file_path = os.path.join(folder_path, file_name) # Joining the two together 

    with open(file_path, "w") as f: # "Writing" the classifier metrics, thereby saving it.
        f.write(data_report)
    print("Reports saved")


def main_function():
    args = input_parse() # Command line arguments
    X_train, y_train, X_test, y_test, labels = load_data() # Loading the data
    X_train_greyscale, X_test_grayscale = greyscale(X_train, X_test) # Converting to greyscale
    X_train_dataset_done, X_test_dataset_done = reshaping(X_train_greyscale, X_test_grayscale) # Reshaping it
    clf_done = logistic_regression_classifier(X_train_dataset_done, y_train, args) # Training the logistic regression
    y_prediction = prediction(clf_done, X_test_dataset_done) # Getting predictions 
    calculated_report = validation(y_test, y_prediction, labels) # Creating a classification report
    saving_report(calculated_report) # Saving the report


if __name__ == "__main__": # If this script is called from the terminal run the main function
    main_function()