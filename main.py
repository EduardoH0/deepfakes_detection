import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import skimage.color as skc
import argparse 
import imutils
import time
import json
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from extract_features import *
from detect_faces import *
from utils import load_dataset, plot_single_img, compute_AUC
from training_models import train_model
from tuning import tuning_fn, tuning_fn_random, remove_correlated_values
from testing_models import test_model

# gpu libraries for numpy (not implemented)
from numba import cuda
#cupy library





def main_feature_extraction(img_list, gab_filters):

    """_Extract the features of the cropped faces_

        Parameters:
        ----------
            ...
            gab_filters: (list) batch of gabor filters

        return:
        ------
            features_vector: array with the extracted features
    """

    global gf_lambda

    # initialize features vector
    features_vector = np.zeros([len(img_list), 53])

    for i in range(len(img_list)):

            # gray scale
            img_gray = cv2.cvtColor(img_list[i], cv2.COLOR_BGR2GRAY)

            # Features based on gradients
            features_vector[i, 0:5] = compute_gradient(img_gray)

            # features based on gray level co-ocurrence matrix
            features_vector[i, 5:10] = features_coomatrix(img_gray)
            
            # features based on wavelet transform
            features_vector[i, 10:19] = wavelet_tranform(img_gray, color=False)

            #features based on wavelet transform (multiple levels)
            features_vector[i, 19:35] = wavelet_tranform2(img_gray, levels=5, w_type = 'haar', color=False)

            # features based on gabor filters
            features_vector[i, 35:53] = gabor_filter(img_gray, gab_filters, n_batches=len(gf_lambda))

    return features_vector

def main_feature_extraction_blocks(img_list, gab_filters):

    """_Extract the features of the cropped faces (many of them from the blocks in which the faces are divided)_

        Parameters:
        ----------
            ...
            gab_filters: (list) batch of gabor filters

        return:
        ------
            features_vector: array with the extracted features
    """

    global gf_lambda

    # initialize features vector
    features_vector = np.zeros([len(img_list), 205])

    for i in range(len(img_list)):

            # gray scale
            img_gray = cv2.cvtColor(img_list[i], cv2.COLOR_BGR2GRAY)
            # divide img in slices (grayscale)
            real_gray_slices = divide_image(img_gray, n=3, color=False)
            # divide img in slices (color)
            real_color_slices = divide_image(img_list[i], n=3, color=False)

            for j in range(len(real_gray_slices)):

                # Features based on gradients
                features_vector[i, 0+(j*5):5+(j*5)] = compute_gradient(real_gray_slices[j])

                # features based on gray level co-ocurrence matrix
                features_vector[i,  45+(j*5):50+(j*5)] = features_coomatrix(real_gray_slices[j])
                
                # features based on wavelet transform
                features_vector[i, 90+(j*9):99+(j*9)] = wavelet_tranform(real_color_slices[j], color=True)

            # features based on wavelet transform
            features_vector[i, 171:187] = wavelet_tranform2(img_gray, levels=5, w_type = 'haar', color=False)

            # features based on gabor filters
            features_vector[i, 187:205] = gabor_filter(img_gray, gab_filters, n_batches=len(gf_lambda))

    return features_vector


## PARAMETERS
task_folder = ["Task_1/", "Task_2_3/"]
folder_path = ["development", "evaluation"]
options = ['real/', 'fake/']
width_img  = 200 
height_img = 200
gf_lambda  = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
gf_theta   = 16

F_BLOCKS = False
F_RANDOM_SPLIT = False


# PROCESS   
F_EXTRACT_FEATURES = True
F_NORMALIZATION = True
F_SAVE_FEATURES = True
F_LOAD_FEATURES = False
F_TRAIN_MODEL = True
F_TUNE_MODEL = True
F_TEST_MODEL = True
F_AUC = True


# TUNING MODES
# 0 = Features splitted by half and acc checked adding feature by feature
# 1 = Choose de features that maximize at the same time the acc of the training and validation set
# 2 = Eliminate highly correlated features
F_TUNING_MODE = 1

# TASK SELECTION
# 1 = TASK 1
# 2 = TASK 2
F_TASK = 1




          
        
if __name__ == '__main__':

    """_MAIN_

        Parameters:
        ----------
            ...
            F_BLOCKS _____________ (bool) if true divide the cropped faces into blocks or slices (9, computational cost *9)
            F_RANDOM_SPLIT _______ (bool) if true, randomly split the training and validation set
            F_EXTRACT_FEATURES ___ (bool) if true, extract the features for the training, validation and test set
            F_NORMALIZATION ______ (bool) if true, normalize the extracted features
            F_SAVE_FEATURES ______ (bool) if true, save the extracted features of the input data
            F_LOAD_FEATURES ______ (bool) if true, load the saved features and the idxs of the selected features
            F_TRAIN_MODEL ________ (bool) if true, train the model and plot results for different classifiers (acc and confusion matrices)
            F_TUNE_MODEL _________ (bool) if true, tune the model selecting the best parameters
            F_TEST_MODEL _________ (bool) if true, test the model with the selected parameters
            F_AUC ________________ (bool) if true, comput the ROC curve and calculate the AUC

            F_TUNING_MODE ________ (int) select the tuning mode for the model
                0 _ Features splitted by half and acc checked adding feature by feature
                1 _ Choose the features that maximize at the same time the acc of the training and validation set
                2 _ Eliminate highly correlated features
            
            F_TASK _______________ (int) select the task
                1 _ Task 1 
                2 _ Task 2 (only for evaluation of the task 1 model)

            width_img ____________ (int) resize width
            height_img ___________ (int) resize height
            gf_lambda ____________ (list(floats)) coefficients of lambda of the gabor filters
            gf_theta _____________ (int) number of orientations considered for the gabor filters within {0, 180}
    """

    if F_EXTRACT_FEATURES:

        #0. GABOR FILTERS
        gab_filters = get_gabor_filters(n_theta=gf_theta, lambda_val=gf_lambda)

        for n_main in range(2):


            # 1. LOAD DATASET
            print('1. Loading dataset...')
            real_faces = load_dataset(folder_path = (task_folder[n_main*(F_TASK-1)] + folder_path[n_main] + "/"), type_face = options[0])
            fake_faces = load_dataset(folder_path = (task_folder[n_main*(F_TASK-1)] + folder_path[n_main] + "/"), type_face = options[1])

            
            # 2. CROP THE FACE REGION
            print('2. Cropping faces')
            real_faces_cropped = []
            fake_faces_cropped = []


            for i in range(len(real_faces)):
                img = face_detector_cvlib(real_faces[i], folder_path[n_main], resize=True, x=width_img, y=height_img)
                if isinstance(img, np.ndarray):
                    real_faces_cropped.append(img)

            for i in range(len(fake_faces)):
                img = face_detector_cvlib(fake_faces[i], folder_path[n_main], resize=True, x=width_img, y=height_img)
                if isinstance(img, np.ndarray):
                    fake_faces_cropped.append(img)

    

            
            # 3. FEATURE EXTRACTION
            print('3. Extracting features')


            ## get the resolution of the images
            # get_resolution(real_faces_cropped, fake_faces_cropped, save=True)

            start = time.time()
            if F_BLOCKS:
                features_real = main_feature_extraction_blocks(real_faces_cropped, gab_filters)
            else:
                features_real = main_feature_extraction(real_faces_cropped, gab_filters)
            end = time.time()
            print('Time: %8.10f' % (end-start))

            start = time.time()
            if F_BLOCKS:
                features_fake = main_feature_extraction_blocks(fake_faces_cropped, gab_filters)
            else:
                features_fake = main_feature_extraction(fake_faces_cropped, gab_filters)
            end = time.time()
            print('Time: %8.10f' % (end-start))

            # 4. ADD LABELS (Real=0 ; Fake=1)
            print('4. Adding labels')
            features_real = np.append(features_real, np.zeros([features_real.shape[0],1]), axis=1)
            features_fake = np.append(features_fake, np.ones([features_fake.shape[0],1]), axis=1)


            # 5. SPLIT TRAIN, VALIDATION AND TEST
            print('5. Splitting data')
            if n_main==0:

                if F_RANDOM_SPLIT:
                    features_real = shuffle(features_real)
                    features_fake = shuffle(features_fake)


                n_split = 300

                train_features_real = features_real[:n_split,:]
                train_features_fake = features_fake[:n_split,:]
                val_features_real = features_real[n_split:,:]
                val_features_fake = features_fake[n_split:,:]
                
                

                train_features = np.concatenate((train_features_real, train_features_fake), axis=0)
                val_features = np.concatenate((val_features_real,val_features_fake), axis=0)

            else:
                test_features = np.concatenate((features_real,features_fake), axis=0)


    # 6. NORMALIZATION
    if F_NORMALIZATION:
        print("6. Normalization")
        fnorm_train, fnorm_val, fnorm_test = features_normalization(train_features, val_features, test_features, zero_mean=True)

    # 7. SAVE FEATURES
    if F_SAVE_FEATURES:
        print("7. Saving features")
        np.save('fnrom_train.npy', fnorm_train)
        np.save('fnrom_val.npy', fnorm_val)
        np.save('fnrom_test.npy', fnorm_test)

    if F_LOAD_FEATURES:
        fnorm_train  = np.load('fnrom_train.npy')
        fnorm_val    = np.load('fnrom_val.npy')
        fnorm_test   = np.load('fnrom_test.npy')
        
        ## load array with the index of the selected features (MODE 0 or 2)
        # idx_features = np.load('selected_features.npy')

        ## load array with the index of the selected features (MODE 1)
        with open("idx_list", "r") as fp:
            idx_features = json.load(fp)

        ## CREATE array with the idx of ALL the features, regardless of saved tunning
        # idx_features = np.linspace(0,fnorm_train.shape[1]-2,fnorm_train.shape[1]-1,dtype=int)
        
        

    # 8. TRAIN MODEL
    if F_TRAIN_MODEL:
        print("8. Training model")
        train_model(fnorm_train, fnorm_val, fnorm_test, mode='test')

    # 9. TUNE MODEL
    if F_TUNE_MODEL:
        print("9. Tunning model")
        if F_TUNING_MODE == 0:
            idx_features = tuning_fn(fnorm_train, fnorm_val, LINEAR_MODEL=False, POLY_MODEL=False, GAUSSIAN_MDOEL=True, RF_MODEL=False, save_idx=True, merge_data=False, shuffle_data=False)
        
        if F_TUNING_MODE == 1:
            idx_features = tuning_fn_random(fnorm_train, fnorm_val, LINEAR_MODEL=True, POLY_MODEL=False, GAUSSIAN_MDOEL=True, RF_MODEL=False, save_idx=True, 
                                            merge_data=True, shuffle_data = True, min_n = 3, max_n = 205, n_iterations=25, weight_train=0.5, weigth_val=0.5)

            with open("idx_list", "w") as fp:
                json.dump(idx_features, fp)
        
        if F_TUNING_MODE == 2:
            idx_features = remove_correlated_values(fnorm_train, thres_corr=0.95, save_idx=True)
        
        
        
    # 10. TEST MODEL
    if F_TEST_MODEL:
        print("10. Testing model")

        if F_TUNING_MODE == 1:
            for idfe in idx_features:
                prob_predctions, y_true = test_model(fnorm_train, fnorm_val, fnorm_test, idfe, mode='train', merge_dataset=False)

                if F_AUC:
                    compute_AUC(pred_prob=prob_predctions, y_true=y_true, plot_results=True)

        else:
            prob_predctions, y_true = test_model(fnorm_train, fnorm_val, fnorm_test, idx_features, mode='test', merge_dataset=False)


    # 11. AUC
    if F_AUC:
        compute_AUC(pred_prob=prob_predctions, y_true=y_true, plot_results=True)

                

