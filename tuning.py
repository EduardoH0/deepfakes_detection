from audioop import avg
import cv2
import os
import random 
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold


def load_dataset(folder_path, type_face):

    """_Load dataset from the specified folder_

        Parameters:
        ----------
            folder_path: (string) origin folder
            type_face: (string) type of face, either 'real' or 'fake'

        return:
        ------
            images: list with the loaded images
    """

    images = []

    for number_folder in os.listdir(folder_path + type_face):

        for image_name in os.listdir(folder_path + type_face + number_folder + '/'):
            img = cv2.imread(folder_path + type_face + number_folder + '/' + image_name, 1)
            images.append(img)

    return images


def train_models(model, f_train, labels_train, f_pred, labels_pred, predict_train=False, weigth_train=0.5, weigth_val=0.5):

    """_Train the selected classifier_

        Parameters:
        ----------
            model: (class) selected model/classifier
            ...
            predict_train: (bool) if true, get also the predictions for the data used for training
            weigth_train: (float) weigth of the acc obtained in training for the final acc
            weigth_val: (float) weigth of the acc obtained in validation for the final acc

        return:
        ------
            y_pred: acc of the validation or training + validation set
    """

    model.fit(f_train, labels_train.ravel())
    y_pred = model.predict(f_pred)
    y_pred = np.sum(y_pred==labels_pred.ravel()) / len(labels_pred.ravel())

    if predict_train:
        # predict training acc
        y_pred_train = model.predict(f_train)
        y_pred_train = np.sum(y_pred_train==labels_train.ravel()) / len(labels_train.ravel())

        # compute the average
        y_pred = (y_pred*weigth_val + y_pred_train*weigth_train)

    return y_pred



def tuning_fn(f_train, f_val, LINEAR_MODEL=False, POLY_MODEL=False, GAUSSIAN_MDOEL=True, RF_MODEL=False, save_idx=True, merge_data=True, shuffle_data = True):

    """_Tunning for MODE 0: split the features in two groups. Firstly, start with hald of the features and add one by one the features from the other half.
    If the acc of the model increases, add the features, otherwise drop it. Then get the other half of the features (without the dropped ones) and apply the
    same procedure._

        Parameters:
        ----------
            LINEAR_MODEL: (bool) if true use the SVC liner model for tunning
            POLY_MODEL: (bool) if true use the SVC poly (degree 1) model for tunning
            GAUSSIAN_MODEL: (bool) if true use the SVC Gaussian model for tunning
            RF_MODEL: (bool) if true use the Random Forest model for tunning
            save_idx: (bool) if true, save the indexes with the selected features
            merge_date: (bool) if true, merge the training and validation dataset for tunning (will fake results)
            shuffle_data: (bool) if true, shuffle the training data before training

        return:
        ------
            idx_features: (array) array with the indexes of the selected features
    """
    
    # get the labels
    train_labels = f_train[:,-1]
    val_labels   = f_val[:,-1]

    # remove labels
    f_train = f_train[:,:-1]
    f_val   = f_val[:,:-1]

    if merge_data:
        f_val = np.concatenate((f_train, f_val),axis=0)
        val_labels = np.concatenate((train_labels, val_labels),axis=0)


    # shuffle training data
    if shuffle_data:
        f_train, train_labels = shuffle(f_train, train_labels)
        f_val, val_labels = shuffle(f_val, val_labels)


    # list to store the idx of the selected features
    idx_features = []

    # parameters 
    half_features = f_train.shape[1]//2 # (split features in two)
    # half_features = 90

    max_avg_results = 0

    for i in range(2):

        if i == 0:
            # split features
            half_f_train = f_train[:,:half_features]
            half_f_val   = f_val[:,:half_features]

            # get iterator length
            half_features_it = f_train[:,half_features:].shape[1]
            # half_features_it = half_features


        else:
            half_f_train = half_f_train[:,half_features:]
            half_f_val   = half_f_val[:,half_features:]

            half_features_it = half_features

            y_pred = []

            if LINEAR_MODEL:
                model = svm.SVC(kernel='linear')
                y_pred.append(train_models(model, half_f_train, train_labels, half_f_val, val_labels))

            if POLY_MODEL:
                model = svm.SVC(kernel='poly', degree = 1)
                y_pred.append(train_models(model, half_f_train, train_labels, half_f_val, val_labels))

            if GAUSSIAN_MDOEL:
                model = svm.SVC(kernel='rbf')
                y_pred.append(train_models(model, half_f_train, train_labels, half_f_val, val_labels))

            if RF_MODEL:
                model = RandomForestClassifier(n_estimators = 1000, max_depth=8, random_state=1)
                y_pred.append(train_models(model, half_f_train, train_labels, half_f_val, val_labels))


            max_avg_results = np.mean([y_pred])


        for j in range(half_features_it):

            # try with the next feature
            if i == 0:
                temp_features_train = np.concatenate((half_f_train, f_train[:,half_features+j, None]), axis=1)
                temp_features_val   = np.concatenate((half_f_val, f_val[:,half_features+j, None]), axis=1)
            else:
                temp_features_train = np.concatenate((half_f_train, f_train[:,j, None]), axis=1)
                temp_features_val = np.concatenate((half_f_val, f_val[:,j, None]), axis=1)

            y_pred = []

            if LINEAR_MODEL:
                model = svm.SVC(kernel='linear')
                y_pred.append(train_models(model, temp_features_train, train_labels, temp_features_val, val_labels))
                
            if POLY_MODEL:
                model = svm.SVC(kernel='poly', degree = 1)
                y_pred.append(train_models(model, temp_features_train, train_labels, temp_features_val, val_labels))

            if GAUSSIAN_MDOEL:
                model = svm.SVC(kernel='rbf')
                y_pred.append(train_models(model, temp_features_train, train_labels, temp_features_val, val_labels))

            if RF_MODEL:
                model = RandomForestClassifier(n_estimators = 1000, max_depth=8, random_state=1)
                y_pred.append(train_models(model, temp_features_train, train_labels, temp_features_val, val_labels))
            
            avg_results = np.mean([y_pred]) 


            # if the prediction is better, permanently add that feature to the model
            if max_avg_results < avg_results:
                max_avg_results = avg_results

                if i==0:
                    half_f_train = np.concatenate((half_f_train, f_train[:,half_features+j, None]), axis=1)
                    half_f_val   = np.concatenate((half_f_val, f_val[:,half_features+j, None]), axis=1)

                    idx_features.append(j+half_features)
                else:
                    half_f_train = np.concatenate((half_f_train, f_train[:, j, None]), axis=1)
                    half_f_val   = np.concatenate((half_f_val, f_val[:, j, None]), axis=1)

                    idx_features.append(j)

            print(f'Iteration {j}')
            print(avg_results, max_avg_results)

       

    print("Number of features selected: ",len(idx_features))
    print("Idx features: ", idx_features)


    if save_idx:
        np.save('selected_features.npy', idx_features)

    return idx_features


def tuning_fn_random(f_train, f_val, LINEAR_MODEL=False, POLY_MODEL=False, GAUSSIAN_MDOEL=True, RF_MODEL=False, save_idx=True, 
                      merge_data=True, shuffle_data = True, min_n = 20, max_n = 100, n_iterations=100, weight_train=0.5, weigth_val=0.5):

    """_Tunning for MODE 1: randomly choose different batches of features of different lenghts. Aim to find the features that maximize at the same
    time the predictions for the same data used for training and the validation data. This method tries to improve generalization for a samll dataset
    in which is easy to overfit._

        Parameters:
        ----------
            LINEAR_MODEL: (bool) if true use the SVC liner model for tunning
            POLY_MODEL: (bool) if true use the SVC poly (degree 1) model for tunning
            GAUSSIAN_MODEL: (bool) if true use the SVC Gaussian model for tunning
            RF_MODEL: (bool) if true use the Random Forest model for tunning
            save_idx: (bool) if true, save the indexes with the selected features
            merge_date: (bool) if true, merge the training and validation dataset for tunning (will fake results)
            shuffle_data: (bool) if true, shuffle the training data before training
            min_n: (int) min length of feature batch considered
            max_n: (int) max length of feature batch considered
            n_iterations: (int) number of iterations performed for each of the feature batch lengths considered
            weigth_train: (float) weigth of the acc obtained in training for the final acc
            weigth_val: (float) weigth of the acc obtained in validation for the final acc


        return:
        ------
            best_idxs: (list) list with the best selected features
    """

    # params
    best_results = 0
    best_idxs = []
    iter_record = 0
    
    # get the labels
    train_labels = f_train[:,-1]
    val_labels   = f_val[:,-1]

    # remove labels
    f_train = f_train[:,:-1]
    f_val   = f_val[:,:-1]

    if merge_data:
        f_val = np.concatenate((f_train, f_val),axis=0)
        val_labels = np.concatenate((train_labels, val_labels),axis=0)


    # shuffle training data
    if shuffle_data:
        f_train, train_labels = shuffle(f_train, train_labels)
        f_val, val_labels = shuffle(f_val, val_labels)

    
    for n_features in range(min_n, max_n+1):
        
        for it in range(n_iterations):

            y_pred = []
            # get the random indexes    
            random_idx = random.sample(range(f_train.shape[1]), n_features)

            random_train = f_train[:, random_idx]
            random_val   = f_val[:, random_idx]

            if LINEAR_MODEL:
                model = svm.SVC(kernel='linear')
                y_pred.append(train_models(model, random_train, train_labels, random_val, val_labels, predict_train=True, weigth_train=weight_train, weigth_val=weigth_val))
                
            if POLY_MODEL:
                model = svm.SVC(kernel='poly', degree = 1)
                y_pred.append(train_models(model, random_train, train_labels, random_val, val_labels, predict_train=True, weigth_train=weight_train, weigth_val=weigth_val))

            if GAUSSIAN_MDOEL:
                model = svm.SVC(kernel='rbf')
                y_pred.append(train_models(model, random_train, train_labels, random_val, val_labels, predict_train=True, weigth_train=weight_train, weigth_val=weigth_val))

            if RF_MODEL:
                model = RandomForestClassifier(n_estimators = 1000, max_depth=8, random_state=1)
                y_pred.append(train_models(model, random_train, train_labels, random_val, val_labels, predict_train=True, weigth_train=weight_train, weigth_val=weigth_val))

            avg_results = np.mean([y_pred]) 

            if avg_results>best_results:
                best_results = avg_results
                best_idxs.append(random_idx)

        iter_record+=n_iterations
        print(iter_record, 'iterations done')
    return best_idxs[-5:]
            


def remove_correlated_values(train_features, thres_corr = 0.95, save_idx = False):

    """_Tunning for MODE 2: delete highly correlated features. Delete every feature that has higher correlation than 'thres_corr'_

        Parameters:
        ----------
            thres_corr: (float) correlation threshold 
            save: (bool) if true, save indexes of the selected features


        return:
        ------
            idx_features: (array) array with the indexes of the selected features
    """

    # remove the label
    train_features = train_features[:,:-1]

    # compute the correlation matrix
    corr_coef = np.corrcoef(train_features, rowvar=False)

    # get the upper triangular of the correlation and detect every column that has any value under the threshold
    upper_tri = ((corr_coef * np.triu(np.ones(corr_coef.shape), k=1))<thres_corr).all(axis=0)
    
    # create an array with the selected features
    idx_features = np.linspace(0,len(upper_tri)-1,len(upper_tri), dtype=('int32'))[upper_tri]

    if save_idx:
        np.save('selected_features.npy', idx_features)

    return idx_features




        







    



    







