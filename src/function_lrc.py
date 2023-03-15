
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
print("Getting the data")

def load_data()
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
          return X_train, y_train, X_test, y_test

# Converting to grey scale 
print("Converting to greyscale")
def greyscale(data_train, data_test):
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
print("Reshaping the data")
def reshaping(data_train_greyscale, data_test_greyscale):
    nsamples, nx, ny = data_train_greyscale.shape
    # new shape to be two values, 50 000 and 1024 (32 times 32)  
    X_train_dataset = data_train_greyscale.reshape((nsamples,nx*ny))

    # Reshaping the test data 
    nsamples, nx, ny = data_test_greyscale.shape
    X_test_dataset = data_test_greyscale.reshape((nsamples,nx*ny))
    
    return X_train_dataset, X_test_dataset


# Logesitc regression 
print("Training Logistic Regression")
def logistic_regression_classifier(X_train_dataset, y_train):
    clf = LogisticRegression(penalty="none", # Math about logistic regression. Dont use any penalty values. Use all weights 
    # When penaly is set to l1 = if a value is very small set it to zero. Keeping only meaningful weights 
                            tol=0.1, # by how much weights should be changing. if not changing by this much then just the training. If the model is not improving by this much then just stop
                            verbose=True, # Print an output about model preformance. 
                            solver="saga", # multiclass dataset 
                            multi_class="multinomial").fit(X_train_dataset, y_train) # multiclass problem 
    return clf


# Getting the predictions 
def prediction(model, data):
    y_pred = model.predict(data)
    return y_pred


# Creating a report
def validation(test_labels, test_prediction, my_labels): 
    report = classification_report(test_labels, 
                                test_prediction, 
                                target_names=my_labels) # setting labels 
    return report


# Saving the report 
def saving_report(data_report):
    folder_path = os.path.join("out")
    file_name = "test_logistic_reg_classifier_metrics.txt"
    file_path = os.path.join(folder_path, file_name)

    with open(file_path, "w") as f: # "Writing" the classifier metrics, thereby saving it.
        f.write(data_report)
    print("Reports saved")


def main_function():
    X_train, y_train, X_test, y_test = load_data()
    X_train_greyscale, X_test_grayscale = greyscale(X_train, X_test)
    X_train_dataset_done, X_test_dataset_done = reshaping(X_train_greyscale, X_test_grayscale)
    clf_done = logistic_regression_classifier(X_train_dataset_done, y_train)
    y_prediction = prediction(clf_done, X_test_dataset_done)
    calculated_report = validation(y_test, y_prediction, labels)
    saving_report(calculated_report)


if __name__ == "__main__":
    main_function()
