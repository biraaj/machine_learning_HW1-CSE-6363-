# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 02:52:32 2023

@author: biraaj
"""
from utility import loadTrain_data,loadTest_data,knn,leave_out_one_knn
import numpy as np

#paths for train and test data
train_data_2b = "train_data_2b.txt"
test_data_2b = "test_data_2b.txt"
program_data_2c_2d_2e = "program_data_2c_2d_2e.txt"

#input for k variations to be used with knn
k_list = [1,3,5]


#2b implementing knn classifier to classify the test points

print("*********** implementing knn without any modifications and using euclidean distance ***********")
loaded_train_features, loaded_train_targets = loadTrain_data(train_data_2b)
print(loaded_train_features.shape[0])
loaded_test_features = loadTest_data(test_data_2b)
# Here we can see predictions for each test point with different k values
for k in k_list:
    neighbour_targets_mapped = [knn(loaded_train_features, loaded_train_targets, data_point,k) for data_point in loaded_test_features]
    print("** when k value is ",k,"**")
    print("*****")
    for _entry in neighbour_targets_mapped:
        unique_items, item_counts = np.unique(_entry[1], return_counts = True)
        print(unique_items[np.where(item_counts == item_counts.max())][0])
    print("*****")
    
# Loading program data for 2c,d,e
loaded_program_data_features, loaded_program_data_targets = loadTrain_data(program_data_2c_2d_2e)

#2c implementing leave one out validation using all the features(it's 4 here) and cartesian distance
print("***********")
print("")
print("*********** Validating knn implementation using leave one out method with 4 features and cartesian distance for k value of 1,3,5 ************")
performance_list = [leave_out_one_knn(loaded_program_data_features,loaded_program_data_targets,k,"cartesian") for k in k_list]
best_performance = [0,0]
for index,k in enumerate(k_list):
    print("Accuracy of knn for k=",k," :",performance_list[index],"%")
    if performance_list[index] > best_performance[0]:
        best_performance = [performance_list[index],k]
print("The Best performance is observed for k=",best_performance[1],"with",best_performance[0],"% accuracy")

#2d implementing leave one out validation using all the features(it's 4 here) and manhattan distance
print("***********")
print("")
print("*********** Validating knn implementation using leave one out method with 4 features and manhattan distance for k value of 1,3,5 ************")
performance_list = [leave_out_one_knn(loaded_program_data_features,loaded_program_data_targets,k,"manhattan") for k in k_list]
best_performance = [0,0]
for index,k in enumerate(k_list):
    print("Accuracy of knn for k=",k," :",performance_list[index],"%")
    if performance_list[index] > best_performance[0]:
        best_performance = [performance_list[index],k]
print("The Best performance is observed for k=",best_performance[1],"with",best_performance[0],"% accuracy")


# Removing the last feature from program data
loaded_program_data_features, loaded_program_data_targets = loadTrain_data(program_data_2c_2d_2e,1)

#2e implementing leave one out validation removing one feature and cartesian distance
print("***********")
print("")
print("*********** Validating knn implementation using leave one out method with 3 features and cartesian distance for k value of 1,3,5 ************")
performance_list = [leave_out_one_knn(loaded_program_data_features,loaded_program_data_targets,k,"cartesian") for k in k_list]
best_performance = [0,0]
for index,k in enumerate(k_list):
    print("Accuracy of knn for k=",k," :",performance_list[index],"%")
    if performance_list[index] > best_performance[0]:
        best_performance = [performance_list[index],k]
print("The Best performance is observed for k=",best_performance[1],"with",best_performance[0],"% accuracy")

    


