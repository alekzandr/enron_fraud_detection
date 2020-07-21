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

# Convert function to handle dataset if still from python 2 pickle output
def convert(data):
    if isinstance(data, bytes):  return data.decode('ascii')
    if isinstance(data, dict):   return dict(map(convert, data.items()))
    if isinstance(data, tuple):  return map(convert, data)
    return data

# Open dataset and read it in
with open('final_project_dataset.pkl', 'rb') as dataset_infile:
    dataset = pickle.load(dataset_infile, encoding = 'bytes')

# Format and Convert dataset if still python 2 pickle
dataset = convert(dataset)
dataset = pd.DataFrame.from_dict(dataset)
dataset = dataset.T

dataset['salary'] = pd.to_numeric(dataset['salary'], errors='coerce', downcast='float')
dataset['to_messages'] = pd.to_numeric(dataset['to_messages'], errors='coerce', downcast='float')
dataset['deferral_payments'] = pd.to_numeric(dataset['deferral_payments'], errors='coerce', downcast='float')
dataset['total_payments'] = pd.to_numeric(dataset['total_payments'], errors='coerce', downcast='float')
dataset['loan_advances'] = pd.to_numeric(dataset['loan_advances'], errors='coerce', downcast='float')
dataset['bonus'] = pd.to_numeric(dataset['bonus'], errors='coerce', downcast='float')
dataset.drop(['email_address'], axis=1, inplace=True)
dataset['restricted_stock_deferred'] = pd.to_numeric(dataset['restricted_stock_deferred'], errors='coerce', downcast='float')
dataset['deferred_income'] = pd.to_numeric(dataset['deferred_income'], errors='coerce', downcast='float')
dataset['total_stock_value'] = pd.to_numeric(dataset['total_stock_value'], errors='coerce', downcast='float')
dataset['expenses'] = pd.to_numeric(dataset['expenses'], errors='coerce', downcast='float')
dataset['from_poi_to_this_person'] = pd.to_numeric(dataset['from_poi_to_this_person'], errors='coerce', downcast='float')
dataset['exercised_stock_options'] = pd.to_numeric(dataset['exercised_stock_options'], errors='coerce', downcast='float')
dataset['from_messages'] = pd.to_numeric(dataset['from_messages'], errors='coerce', downcast='float')
dataset['other'] = pd.to_numeric(dataset['other'], errors='coerce', downcast='float')
dataset['from_this_person_to_poi'] = pd.to_numeric(dataset['from_this_person_to_poi'], errors='coerce', downcast='float')
dataset['poi'] = dataset['poi'].astype('boolean')
dataset['long_term_incentive'] = pd.to_numeric(dataset['long_term_incentive'], errors='coerce', downcast='float')
dataset['shared_receipt_with_poi'] = pd.to_numeric(dataset['shared_receipt_with_poi'], errors='coerce', downcast='float')
dataset['restricted_stock'] = pd.to_numeric(dataset['restricted_stock'], errors='coerce', downcast='float')
dataset['director_fees'] = pd.to_numeric(dataset['director_fees'], errors='coerce', downcast='float')
dataset['from_non_poi_to_this_person'] = dataset['to_messages'] - dataset['from_poi_to_this_person']
dataset['from_this_person_to_non_poi'] = dataset['from_messages'] - dataset['from_this_person_to_poi']
dataset['total_non_salary'] = dataset['total_payments'] - dataset['salary']

# Save new dataset
dataset.to_pickle('./my_dataset.pkl')

# LightGBM Hyperparameters
params = {
    'metric' : 'binary_logloss',
    'learning_rate' : 0.0002,
    'boosting_type' : 'gbdt',
    'objective' : 'binary',
    'max_depth' : 10,
    'num_leaves' : 10
}

# Create labels data frame and convert boolean values to int
# True is 1. False is 0.
labels = dataset['poi']
labels = labels.astype(int)
labels = labels.astype(int)

# We'll Drop poi since we made a new dataframe
dataset.drop(['poi'], axis=1, inplace=True) 

# Fill missing values with the mean of that column
dataset.fillna(dataset.mean(), inplace=True)

# Scale value between 0 to 1
scaler = MinMaxScaler()
dataset_scaled = scaler.fit_transform(dataset)
dataset_scaled = pd.DataFrame(data=dataset_scaled)

clf = lgb.LGBMClassifier()

# Train and testing parameters for LightGBM Model

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

        d_train = lgb.Dataset(data_train, label=labels_train)
        clf = lgb.train(params, train_set=d_train, num_boost_round=boost_rounds, init_model=clf)
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

# Train and testing parameters for ANN
def run_nn_model(clf, dataset, labels, folds=5, random_state=42, epochs=1000):
    cv = StratifiedShuffleSplit(n_splits=folds, random_state=random_state)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    predictions = []
        
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
            
        data_train = np.array(data_train)
        labels_train = np.array(labels_train)
        data_test = np.array(data_test)
        labels_test = np.array(labels_test)
          

        clf.fit(
            data_train,
            labels_train,
            batch_size=64,
            epochs=epochs,
        )


        tmp_predictions = clf.predict(data_test)
        predictions.append(tmp_predictions)

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

    return clf, predictions

# Create LightGBM Model and traing it.
lgb_clf = lgb.LGBMClassifier()
lgb_clf, lgb_predictions, labels_test = run_lgb_model(clf=lgb_clf, dataset=dataset_scaled, labels=labels, random_state=15, boost_rounds=50000)

# Create neural network architecture 
def create_nn(input_dim, output_dim):    
    model = keras.Sequential(
    [
        layers.Dense(input_dim, activation="relu", name="layer1"),
        layers.Dense(32, activation="relu", 
                     kernel_regularizer=regularizers.l2(0.001),
                     name="layer2"),
        layers.Dropout(.2),
        layers.Dense(output_dim, activation='relu', name="layer6"),
    ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.002),  # Optimizer
        # Loss function to minimize
        loss=keras.losses.BinaryCrossentropy(),
        # List of metrics to monitor
        metrics=[keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.5)],)
    return model

# Create ANN
nn_clf = create_nn(dataset_scaled.shape[1], 1)

# Train ANN
#nn_clf, nn_predictions = run_nn_model(nn_clf, dataset_scaled, labels, folds=5, random_state=15, epochs=5000)


# Train and testing parameters for KMeans Model
def run_kmeans_model(clf, dataset, labels, folds=5, random_state=42):
    cv = StratifiedShuffleSplit(n_splits=folds, random_state=random_state)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    predictions = []
    
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

        clf.fit(data_train, labels_train)
        tmp_predictions = clf.predict(data_test, labels_test)
        
        predictions.append(tmp_predictions)
        
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

    return clf, predictions


# Crate a KMeans model
km_clf = KMeans(n_clusters=75, random_state=15, max_iter=100000)

# Train KMeans model
#km_clf, km_predictions = run_kmeans_model(km_clf, dataset_scaled, labels, folds=5, random_state=15)

true_negatives = 0
false_negatives = 0
true_positives = 0
false_positives = 0

'''   
# Formatted output string
PERF_FORMAT_STRING = "\
\nAccuracy: {:>0.{display_precision}f}\nPrecision: {:>0.{display_precision}f}\nRecall: {:>0.{display_precision}f}\nF1: {:>0.{display_precision}f}\nF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\nTotal predictions: {:4d}\nTrue positives: {:4d}\nFalse positives: {:4d}\
\nFalse negatives: {:4d}\nTrue negatives: {:4d}"

# Ensemble logic that checks predictions in a majority
# vote method
for i, j, k, l in zip(lgb_predictions, nn_predictions, km_predictions, labels_test):
    votes = []
    
    for ii, jj, kk in zip(i, j, k):
        #print(ii)
        if ii+jj+kk > 1:
            votes.append(1)
        else:
            votes.append(0)

    for prediction, truth in zip(votes, l):
        #print('Prediction: ' + (str(predictions)))
        #print('Truth: ' +  str(labels_test))
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
        #print(clf)
        print(PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5))
        print(RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives))
        print("")
except:
        print("Got a divide by zero when trying out:", clf)
        print("Precision or recall may be undefined due to a lack of true positive predicitons.")

'''

#with open("lgb_classifier.pkl", 'w') as clf_outfile:
#    pickle.dump(lgb_clf, clf_outfile)
#with open("nn_classifier.pkl", 'w') as clf_outfile:
#    pickle.dump(lgb_clf, clf_outfile)
#with open("kmeans_classifier.pkl", 'w') as clf_outfile:
#    pickle.dump(lgb_clf, clf_outfile)
lgb_clf.save_model('lgb_classifier.pkl')
dataset_scaled.to_pickle('./my_dataset_modified.pkl')
labels.to_pickle('labels.pkl')

