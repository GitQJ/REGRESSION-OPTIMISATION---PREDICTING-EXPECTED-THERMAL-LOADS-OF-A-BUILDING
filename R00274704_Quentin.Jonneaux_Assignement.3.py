#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 20:27:24 2025

@author: Quentin
"""

# Importing necessary libraries
import pandas as pd # Pandas for data reading and manipulation
import numpy as np # Numpy for mathematical calculation and manipulating arrays

import matplotlib.pyplot as plt # Matplotlib pyplot to provide visuals
import seaborn as sns # Seaborn to provide advanced plots

from sklearn import model_selection # Scikit Learn model_selection to apply K-Fold cross validation

###############################################################################
# Task 1 - Input data
# Create a function that import and preprocess data
def data_input():
    # Store file path of Dataset
    path = '/Users/Quentin/Desktop/Hdip Data Science and Analytics/Year 1/Semester 3/COMP8043 - Machine Learning/Assignment 3/energy_performance.csv'
    
    # Store the data in a data frame
    data = pd.read_csv(path)
    
    # Split features and targets
    features = data.drop(['Heating load', 'Cooling load'], axis=1)
    targets = data[['Heating load', 'Cooling load']]
    
    # Get and Output the minimum and maximum heating loads
    min_heating_load = min(targets['Heating load'])
    print('Minimum Heating Load: ',min_heating_load)
    max_heating_load = max(targets['Heating load'])
    print('Maximum Heating Load: ',max_heating_load)
    
    # Get and Output the minimum and maximum cooling loads
    min_cooling_load = min(targets['Cooling load'])
    print('Minimum Cooling Load: ',min_cooling_load)
    max_cooling_load = max(targets['Cooling load'])
    print('Maximum Cooling Load: ',max_cooling_load)
    
    # Convert features and each target into numpy arrays for algorithm training
    features=np.array(features)
    heating_targets=np.array(targets['Heating load'])
    cooling_targets=np.array(targets['Cooling load'])
    
    # Return arrays for training
    return features,heating_targets,cooling_targets

###############################################################################
# Task 2 - model function
# Create a function that compute a polynomial
def calculate_model(degree, data, p0):
    # We initialize the result vector of zeros with a lengths of the data points
    result = np.zeros(data.shape[0])
    # We initialize the index at 0 for the coefficient access for computation
    a = 0
    # We iterate through 8 loops for 8 variables to calculate the function result
    for i in range(degree+1): # For each loop, we iterate with the degree number
        for j in range(degree+1):
            for k in range(degree+1):
                for l in range(degree+1):
                    for m in range(degree+1):
                        for n in range(degree+1):
                            for o in range(degree+1):
                                for p in range(degree+1):
                                    if i+j+k+l+m+n+o+p<=degree:
                                        # We compute result as long as we haven't reach the degree
                                        result += p0[a] * (data[:, 0] ** i) * (data[:, 1] ** j) * (data[:, 2] ** k) * (data[:, 3] ** l) * (data[:, 4] ** m) * (data[:, 5] ** n) * (data[:, 6] ** o) * (data[:, 7] ** p)
                                        a += 1
    # We return the result of the model function
    return result

# Create a function that compute the vector size from the degree
def num_parameters(degree):
    # initialize size at zero
    t = 0
    # We iterate through 8 loops for 8 variables to calculate the vector size
    for i in range(degree+1):
        for j in range(degree+1):
            for k in range(degree+1):
                for l in range(degree+1):
                    for m in range(degree+1):
                        for n in range(degree+1):
                            for o in range(degree+1):
                                for p in range(degree+1):
                                    if i+j+k+l+m+n+o+p<=degree:
                                        # We add 1 to size as long as we haven't reach degree
                                        t = t + 1
    # We return the size
    return t


###############################################################################
# Task 3 - linearization
# Create a function that compute the target vector and the Jacobian at the linearization point
def linearize(deg, data, p0):
    # Calculating the model function at p0 (to then calculate the Jacobian)
    f0 = calculate_model(deg, data, p0)
    # Initialize the Jacobian vector with the model function vector length)
    J = np.zeros((len(f0), len(p0)))
    # Defining small incrementation value (to calculate the partial derivative)
    epsilon = 1e-6
    # Iterate for each value in the p0 vector
    for i in range(len(p0)):
        # Add the perturbation value to the coefficient
        p0[i] += epsilon
        # Calculate the model function at new point
        fi = calculate_model(deg, data, p0)
        # Revert perturbation
        p0[i] -= epsilon
        # Calculate the partial derivative (difference divided by perturbation)
        di = (fi - f0) / epsilon
        # Store the partial derivative in the Jacobian matrix
        J[:, i] = di
    # We return the target vector and the Jacobian matrix
    return f0, J
###############################################################################
# Task 4 - parameter update
# Create a function that calculates the optimal parameter update
def calculate_update(y, f0, J):
    # We define a step
    l = 1e-2
    # We compute the normal equation matrix, augmented with regularization term by adding a diagonal matrix of size of the Jacobian
    N = J.T @ J + l * np.eye(J.shape[1])
    # We calculate the residual by substracting the estimation to the actual value at data point
    r = y - f0
    # We create the right hand-side of the normal equation system
    n = J.T @ r
    # We solve the parameter update
    dp = np.linalg.solve(N, n)
    # We return the update
    return dp
###############################################################################
# Task 5 - regression
# Create a function that calculates the coefficient vector that best fits the training data
def regression(degree,features,targets):
    # Setting up the maximum number of iteration to prevent converging endlessly
    max_iter = 100
    # initialise the parameter vector of coefficients with zeros (adjusting the size to degrees)
    p0 = np.zeros(num_parameters(degree))
    # Setting up an iterative procedure
    for i in range(max_iter):
        # Linearize
        f0, J = linearize(degree, features, p0)
        # Update parameter
        dp = calculate_update(targets, f0, J)
        # Update paramater vector by adding the update to each coefficient
        p0 += dp
        # print(i)
        # print(dp)
    return p0
###############################################################################
# Task 6 - model selection
# Choosing best parameter through cross-validation
def main():
    # Get and preprocess data
    features,heating_targets,cooling_targets = data_input()
    
    # Creating a K-Fold cross validation procedure (Using 5 folds, shuffling indexes, setting random_state as 42 for reproducibility)
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Heating CV
    # Store absolute differences for each polynomial degree
    heat_abs_diffs = {0:[],
                      1:[],
                      2:[]}
    
    # Split data in train and test set for Heating load target
    for train_index, test_index in kf.split(features, heating_targets):
        # Iterate between degrees ranging from 0 to 2
        for deg in range(3):
            # Caculate the coefficient that best fit training set
            p = regression(deg,features[train_index],heating_targets[train_index])
            # Make predictions on the test set based on those coefficients
            preds = (calculate_model(deg, features[test_index], p))
            # Calculate the metric (difference of estimated value from actual value of target)
            differences = preds-heating_targets[test_index]
            # Calculate the absolute value of the differences
            abs_diff = abs(np.mean(differences))
            # Append the differences to the dictionary
            heat_abs_diffs[deg].append(abs_diff)
    
    # Create a dictionary calculating the mean absolute differences for each degree
    mean_heat_abs_diffs = {0:np.mean(heat_abs_diffs[0]),
                           1:np.mean(heat_abs_diffs[1]),
                           2:np.mean(heat_abs_diffs[2])}
    
    # Get the best value (smallest mean absolute difference)
    heat_min_val = min(mean_heat_abs_diffs.values())

    # Get the best parameter (degree with the best difference metric)
    heat_best_param = [k for k, v in mean_heat_abs_diffs.items() if v == heat_min_val][0]
    print('Heating load estimation best parameter is: ',heat_best_param,'\nfor a mean difference of prediction of : ',heat_min_val)
    
    # Cooling CV
    # Store absolute differences for each polynomial degree
    cool_abs_diffs = {0:[],
                      1:[],
                      2:[]}
    # Split data in train and test set for Cooling load target
    for train_index, test_index in kf.split(features, cooling_targets):
        # Iterate between degrees ranging from 0 to 2
        for deg in range(3):
            # Caculate the coefficient that best fit training set
            p = regression(deg,features[train_index],cooling_targets[train_index])
            # Make predictions on the test set based on those coefficients
            preds = (calculate_model(deg, features[test_index], p))
            # Calculate the metric (difference of estimated value from actual value of target)
            differences = preds-cooling_targets[test_index]
            # Calculate the absolute value of the differences
            abs_diff = abs(np.mean(differences))
            # Append the differences to the dictionary
            cool_abs_diffs[deg].append(abs_diff)
        
    # Create a dictionary calculating the mean absolute differences for each degree
    mean_cool_abs_diffs = {0:np.mean(cool_abs_diffs[0]),
                           1:np.mean(cool_abs_diffs[1]),
                           2:np.mean(cool_abs_diffs[2])}
    
    # Get the best value (smallest mean absolute difference)
    cool_min_val = min(mean_cool_abs_diffs.values())
    
    # Get the best parameter (degree with the best difference metric)
    cool_best_param = [k for k, v in mean_cool_abs_diffs.items() if v == cool_min_val][0]
    print('Cooling load estimation best parameter is: ',cool_best_param,'\nfor a mean difference of prediction of : ',cool_min_val)


# ###############################################################################
# Task 7 - evaluation and visualisation of results

    # Final Evaluation of Heating Loads predictions
    # Train the model by calculating the best parameter vectors using the best degree parameter from CV procedure
    heat_p = regression(heat_best_param,features,heating_targets)
    # Make predictions for all features in the dataset
    heat_preds = (calculate_model(heat_best_param, features, heat_p))
    # Make a dataframe storing predictions, actual values and absolute differences for each observations
    heat_data = pd.DataFrame({'Predictions':heat_preds,'Actual':heating_targets,'Absolute Differences':abs(heat_preds-heating_targets)})
    
    # Clear the plots
    plt.close("all")
    # Create a linear model plot to illustrate relationship between predictions and actual values
    sns.lmplot(x='Predictions',y='Actual',data = heat_data,line_kws={'color': 'orange'})
    # Name visual and axis
    plt.title('Estimate Heating Loads against actual loads')
    plt.xlabel('Estimated loads')
    plt.ylabel('Actual loads')
    # Display plot
    plt.show()
    
    # Compute final mean absolute difference for the target
    final_mean_heat_abs_diff = np.mean(heat_data['Absolute Differences'])
    print('Mean absolute difference between estimated and actual heating loads: ',final_mean_heat_abs_diff)
    print('correlation coefficient between predictions and actual values: ',np.corrcoef(heat_preds, heating_targets)[0][1])
    
    # Final Evaluation of Cooling Loads predictions
    
    # Train the model by calculating the best parameter vectors using the best degree parameter from CV procedure
    cool_p = regression(cool_best_param,features,cooling_targets)
    # Make predictions for all features in the dataset
    cool_preds = (calculate_model(cool_best_param, features, cool_p))
    
    # Make a dataframe storing predictions, actual values and absolute differences for each observations
    cool_data = pd.DataFrame({'Predictions':cool_preds,'Actual':cooling_targets,'Absolute Differences':abs(cool_preds-cooling_targets)})
    
    # Clear the plots
    plt.close("all")
    # Create a linear model plot to illustrate relationship between predictions and actual values
    sns.lmplot(x='Predictions',y='Actual',data = cool_data,line_kws={'color': 'orange'})
    # Name visual and axis
    plt.title('Estimate Cooling Loads against actual loads')
    plt.xlabel('Estimated loads')
    plt.ylabel('Actual loads')
    # Display plot
    plt.show()
    
    # Compute final mean absolute difference for the target
    final_mean_cool_abs_diff = np.mean(cool_data['Absolute Differences'])
    print('Mean absolute difference between estimated and actual cooling loads: ',final_mean_cool_abs_diff)
    print('correlation coefficient between predictions and actual values: ',np.corrcoef(cool_preds, cooling_targets)[0][1])
    


main()