#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow import keras
from tensorflow.keras import layers

def run_lgb_model(clf, dataset, labels, folds=5, random_state=42, boost_rounds=1000):
    cv = StratifiedShuffleSplit(n_splits=folds, random_state=random_state)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    predictions = []
    testing = []
    
    params = {
    'metric' : 'binary_logloss',
    'learning_rate' : 0.0002,
    'boosting_type' : 'gbdt',
    'objective' : 'binary',
    'max_depth' : -1,
    'num_leaves' : 31
    }
    
    
    # Formatted output string
    PERF_FORMAT_STRING = "    \nAccuracy: {:>0.{display_precision}f}\nPrecision: {:>0.{display_precision}f}\nRecall: {:>0.{display_precision}f}\nF1: {:>0.{display_precision}f}\nF2: {:>0.{display_precision}f}"
    RESULTS_FORMAT_STRING = "\nTotal predictions: {:4d}\nTrue positives: {:4d}\nFalse positives: {:4d}    \nFalse negatives: {:4d}\nTrue negatives: {:4d}"


    for train_idx, test_idx in cv.split(dataset, labels):
        data_train = []
        data_test = []
        labels_train = []
        labels_test = []
        for ii in train_idx:
            data_train.append(dataset.iloc[ii])
            labels_train.append(labels.iloc[ii])
        for jj in test_idx:
            data_test.append(dataset.iloc[jj])
            labels_test.append(labels.iloc[jj])

            
        #d_train = lgb.Dataset(data_train, label=labels_train)
        #d_train = lgb.Dataset(data_train, label=labels_train)
        #clf = lgb.train(params, train_set=d_train, num_boost_round=boost_rounds, init_model=clf)
        #clf = clf.fit(train_set=data_train, labels_train)
        tmp_predictions = clf.predict(data_test)
        

        predictions.append(tmp_predictions)
        testing.append(labels_test)
        ## Convert prediction probabilities to ints
        for idx, value in enumerate(tmp_predictions):
            if value >= 0.5:
                tmp_predictions[idx] = 1
            else:
                tmp_predictions[idx] = 0


        for prediction, truth in zip(tmp_predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print("Warning: Found a predicted label not == 0 or 1.")
                print("All predictions should take value 0 or 1.")
                print("Evaluating performance for processed predictions:")
                break
    try:
            total_predictions = true_negatives + false_negatives + false_positives + true_positives
            accuracy = 1.0*(true_positives + true_negatives)/total_predictions
            precision = 1.0*true_positives/(true_positives+false_positives)
            recall = 1.0*true_positives/(true_positives+false_negatives)
            f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
            f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
            print(clf)
            print(PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5))
            print(RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives))
            print("")
    except:
            print("Got a divide by zero when trying out:", clf)
            print("Precision or recall may be undefined due to a lack of true positive predicitons.")

    return clf, predictions, testing


CLF_PICKLE_FILENAME = "lgb_classifier.pkl"
DATASET_PICKLE_FILENAME = "my_dataset_modified.pkl"
LABELS_LIST_FILENAME = "labels.pkl"

def load_classifier_and_data():
    #with open(CLF_PICKLE_FILENAME, "r") as clf_infile:
    #    clf = pickle.load(clf_infile)\
    clf = lgb.Booster(model_file=CLF_PICKLE_FILENAME)
    dataset = pd.read_pickle(DATASET_PICKLE_FILENAME)
    label_list = pd.read_pickle(LABELS_LIST_FILENAME)
    return clf, dataset, label_list


def main():
    ### load up student's classifier, dataset, and feature_list
    clf, dataset, labels = load_classifier_and_data()
    
    _, _, _ = run_lgb_model(clf=clf, dataset=dataset, labels=labels, random_state=15, boost_rounds=50000)
    
if __name__ == '__main__':
    main()

