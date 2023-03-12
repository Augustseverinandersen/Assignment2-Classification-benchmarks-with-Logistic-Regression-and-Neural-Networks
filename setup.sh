#!/usr/bin/env bash

# Create virtual enviroment 
#python3 -m venv assignment_2

# Activate virtual enviroment 
#source ./assignment_2/bin/activate 

# Installing requirements 
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# Run the code 
#python3 src/logistic_regression_classifier.py 
#python3 src/neural_network_classifier.py 
python3 src/function_lrc.py 

# Deactivate the virtual environment.
#deactivate