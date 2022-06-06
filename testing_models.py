from audioop import avg
import cv2
import os
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from training_models import get_results



def test_model(all_f_train, all_f_val, all_f_test, array_idx, mode='val', merge_dataset=True):

    """_Test the model with the selected features: print the acc and confusion matrices of different classifiers_

        Parameters:
        ----------
            ...
            array_idx: (list or array) parameter with the indexes of the selected features after tuning
            mode: (string) select the data in which test the model, either 'val' or 'test'
            merge_dataset: (bool) if true, merge the training and validation set to train the model

        return:
        ------
            labels_pred: ground truth labels
            prob_predictions: probability outputs of the random forest classifier
    """


    # get the labels
    train_labels = all_f_train[:,-1]
    val_labels   = all_f_val[:,-1]
    test_labels  = all_f_test[:,-1]

    # remove labels
    all_f_train = all_f_train[:,:-1]
    all_f_val   = all_f_val[:,:-1]
    all_f_test  = all_f_test[:,:-1]


    # get the selected features

    f_train = all_f_train[:,array_idx]
    f_val   = all_f_val[:,array_idx]
    f_test  = all_f_test[:,array_idx]

    
    # merge dataset
    if merge_dataset:
        f_train = np.concatenate((f_train, f_val), axis=0)
        train_labels = np.concatenate((train_labels, val_labels), axis=0)


   
    # shuffle train data
    # f_train, train_labels = shuffle(f_train, train_labels)


    ## TRAINING
    if mode=='val':
        f_pred = f_val
        labels_pred = val_labels
    elif mode=='test':
        f_pred = f_test
        labels_pred = test_labels
    elif mode=='train':
        f_pred = f_train
        labels_pred = train_labels


    # LINEAR
    print('\n Starting training...')
    model = svm.SVC(kernel='linear')
    get_results(model, f_train, train_labels, f_pred, labels_pred, 'SVC Linear: ')

    # POLY 1
    model = svm.SVC(kernel='poly', degree = 1)
    get_results(model, f_train, train_labels, f_pred, labels_pred, 'SVC Poly 1: ')

    # POLY 4
    model = svm.SVC(kernel='poly', degree = 4)
    get_results(model, f_train, train_labels, f_pred, labels_pred, 'SVC Poly 4: ')

    # SIGMOID
    model = svm.SVC(kernel='sigmoid')
    get_results(model, f_train, train_labels, f_pred, labels_pred, 'SVC Sigmoid: ')

    # rbf (gaussian)
    model = svm.SVC(kernel='rbf')
    get_results(model, f_train, train_labels, f_pred, labels_pred, 'SVC Gaussian: ')


    # random forest
    model = RandomForestClassifier(n_estimators = 1000, max_depth=8, random_state=0)
    prob_predictions = get_results(model, f_train, train_labels, f_pred, labels_pred, 'Random Forest: ', return_prob=True)

    # XGBBoost
    model = XGBClassifier(learning_rate = 0.1, 
#                         max_depth=5, 
                        n_estimators=5000, 
#                         subsample=0.5,
#                         colsample_bytrr=0.5,
                        # tree_method='gpu_hist', 
                        # predictor='gpu_predictor',
                        eval_metric='auc',
                        verbosity=1)

    get_results(model, f_train, train_labels, f_pred, labels_pred, 'XGBBoost: ')

    # naive bayes
    model = GaussianNB()
    get_results(model, f_train, train_labels, f_pred, labels_pred, 'Naive Bayes: ')

    #MLP
    model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20, 10), random_state=1, max_iter=10000)
    get_results(model, f_train, train_labels, f_pred, labels_pred, 'MLP: ')

    return prob_predictions, labels_pred
