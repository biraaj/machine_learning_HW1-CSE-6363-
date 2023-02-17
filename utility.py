# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 23:41:48 2023

@author: biraaj
"""

import numpy as np

def replace_parenthisis(_string):
    """
        Function to filter out the data string by removing parenthesis
    """
    return _string.replace('(','').replace(')','').strip().split(", ")

def loadTrain_data(train_data_path, remove_feature_from_end=0):
    """
        This function takes in train data path and outputs numpy array of feature and targets.
    """
    _feat = []
    _targ = []
    
    with open(train_data_path) as _file:
        for _row in _file:
            temp_data = replace_parenthisis(_row)
            features = list(map(float,temp_data[:len(temp_data)-1-remove_feature_from_end]))
            targets = list(map(str,temp_data[len(temp_data)-1:]))
            _feat.append(features)
            _targ.append(targets)
    return np.array(_feat),np.array(_targ)

def loadTest_data(test_data_path):
    """
        This function takes in test data path and outputs all test features as numpy array
    """
    _feat = []
    with open(test_data_path) as _file:
        for _row in _file:
            temp_data = replace_parenthisis(_row)
            features = list(map(float,temp_data[:len(temp_data)]))
            _feat.append(features)
    return np.array(_feat)

def cartesian_dist(p,q):
    """
        This function calculates euclidean distance between two cartesian points.
    """
    return np.sqrt(np.sum(np.square(p-q)))

def manhattan_dist(p,q):
    """
        This function calculates manhattan distance.
    """
    return np.sum(np.abs(p-q))

def knn(train_features, train_targets, data_point ,k=1,distance_formula="cartesian"):
    """

    Parameters
    ----------
        train_features : a*b numpy array (a= number of samples, b=number of features in each sample)
        train_targets : c*1 numpy array(c: respective targets for each sample of a)
        data_point : one set of features to test or compare
        k : it's a integer representing number of neighbours to compare with

    Returns
    -------
        distance of k neighbours
        results/targets of nearest neighbours

    """
    
    _neighbour = []
    _dist = []
    for _index in range(len(train_features)):
        if distance_formula == "cartesian":
            _dist.append(cartesian_dist(train_features[_index], data_point))
        elif distance_formula == "manhattan":
            _dist.append(manhattan_dist(train_features[_index], data_point))
            
    sorted_indices = np.argsort(_dist)
    _dist.sort()
    for _k in range(k):
        _neighbour.append(_dist[_k])
    _target = [train_targets[x] for x in sorted_indices[:k]]
    return _neighbour,_target

def leave_out_one_knn(train_features,train_targets,k=1,distance_formula="cartesian"):
    """
        The function chooses one data point from the whole data set and validates it's accuracy by predicting the target for it comparing it's neighbour.
        Parameters
        ----------
            train_features : a*b numpy array (a= number of samples, b=number of features in each sample)
            train_targets : c*1 numpy array(c: respective targets for each sample of a)
            k : it's a integer representing number of neighbours to compare with

        Returns
        -------
            percentage of accurately predicted values
    """
    values_predicted = []
    acuurate_predictions = 0
    for _element_index in range(train_features.shape[0]):
        new_train_features = np.delete(train_features, _element_index,0)
        new_train_targets = np.delete(train_targets, _element_index)
        neighbour,_targets = knn(new_train_features, new_train_targets, train_features[_element_index],k,distance_formula)
        unique_items, item_counts = np.unique(_targets, return_counts = True)
        values_predicted.append(unique_items[np.where(item_counts == item_counts.max())][0])
    
    for _index,_item in enumerate(values_predicted):
        if(_item == train_targets[_index]):
            acuurate_predictions+=1
    
    percentage_of_accuracy = (acuurate_predictions/len(values_predicted))*100
    
    return percentage_of_accuracy
        
        
            
        
        
    


    
    
    
    

            
            
        