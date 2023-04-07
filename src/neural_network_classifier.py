
# Importing Libraries 

# path tools
import os
import numpy as np
import cv2

# data loader
from tensorflow.keras.datasets import cifar10

# machine learning tools
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# classification models
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Getting the data 

def load_data():
    print("Getting the data")
    (X_train, y_train), (X_test, y_test) = cifar10.load_data() # .load_data # returns 4 objects # () groups them together. # Creating a tuple. 

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
    # new shape to be two values, 50 000 and 1024 (32 times 32)  
    X_train_dataset = data_train_greyscale.reshape((nsamples,nx*ny))

    # Reshaping the test data 
    nsamples, nx, ny = data_test_greyscale.shape
    X_test_dataset = data_test_greyscale.reshape((nsamples,nx*ny))
    
    return X_train_dataset, X_test_dataset

# Neural Network

def neural_network_function(X_train_dataset, y_train):
    print("Training neural network")
    clf = MLPClassifier(random_state=42, 
                        hidden_layer_sizes=(100, 10), #hidden layers, everytime it goes from one layer to another and a weight changes it goes back and starts agian.
                        learning_rate="adaptive", # beginning of the model it will just be guessing. we want it to learn quickly. as soon as it learns a bit, we get it to slow down, and think more about how to predict.
                        early_stopping=True, # Stop early if it is not getting better scores. you can change it by chaning tollerance level
                        verbose=True,
                        max_iter=20).fit(X_train_dataset, y_train) # max iteratoin of 20 times

    return clf

# Getting the predictions 
def prediction(model, data):
    print("Predictions")
    y_pred = model.predict(data)

    return y_pred

# Creating a report 
def validation(test_labels, test_prediction, my_labels):
    print("Validation") 
    report = classification_report(test_labels, 
                                test_prediction, 
                                target_names=my_labels) # setting labels 
    return report

# Saving the report 
def saving_report(data_report):
    folder_path = os.path.join("out")
    file_name = "neural_network_classifier_metrics.txt"
    file_path = os.path.join(folder_path, file_name)

    with open(file_path, "w") as f: # "Writing" the classifier metrics, thereby saving it.
        f.write(data_report)
    print("Reports saved")


def main_function():
    print("Neural Network Script:")
    X_train, y_train, X_test, y_test, labels = load_data()
    X_train_greyscale, X_test_grayscale = greyscale(X_train, X_test)
    X_train_dataset_done, X_test_dataset_done = reshaping(X_train_greyscale, X_test_grayscale)
    clf_done = neural_network_function(X_train_dataset_done, y_train)
    y_prediction = prediction(clf_done, X_test_dataset_done)
    calculated_report = validation(y_test, y_prediction, labels)
    print(calculated_report)
    saving_report(calculated_report)

if __name__ == "__main__":
    main_function()