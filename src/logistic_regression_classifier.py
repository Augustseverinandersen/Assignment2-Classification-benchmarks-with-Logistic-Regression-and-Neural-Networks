
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

# Converting to grey scale 
print("Converting to greyscale")
X_train_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_train])
X_test_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_test])
# Using opencvs colour conversion model 
# Create a np.array which is made from the list we make when converging images to grey scale for every image in X_test

# Creating smaller weights 
X_train_scaled = (X_train_grey)/255.0
X_test_scaled = (X_test_grey)/255.0
# Scaling numbers down to between 0-1

# Reshaping the training data  
print("Reshaping the data")
nsamples, nx, ny = X_train_scaled.shape
# new shape to be two values, 50 000 and 1024 (32 times 32)  
X_train_dataset = X_train_scaled.reshape((nsamples,nx*ny))

# Reshaping the test data 
nsamples, nx, ny = X_test_scaled.shape
X_test_dataset = X_test_scaled.reshape((nsamples,nx*ny))


# Logesitc regression 
print("Training Logistic Regression")
clf = LogisticRegression(penalty="none", # Math about logistic regression. Dont use any penalty values. Use all weights 
# When penaly is set to l1 = if a value is very small set it to zero. Keeping only meaningful weights 
                        tol=0.1, # by how much weights should be changing. if not changing by this much then just the training. If the model is not improving by this much then just stop
                        verbose=True, # Print an output about model preformance. 
                        solver="saga", # multiclass dataset 
                        multi_class="multinomial").fit(X_train_dataset, y_train) # multiclass problem 

# Getting the predictions 
y_pred = clf.predict(X_test_dataset)

# Creating a report 
report = classification_report(y_test, 
                               y_pred, 
                               target_names=labels) # setting labels 

# Saving the report 
folder_path = os.path.join("out")
file_name = "logistic_reg_classifier_metrics.txt"
file_path = os.path.join(folder_path, file_name)

with open(file_path, "w") as f: # "Writing" the classifier metrics, thereby saving it.
    f.write(report)
print("Reports saved")