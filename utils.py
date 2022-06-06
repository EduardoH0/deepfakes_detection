import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


folder_path = ["Task_1/evaluation/"]
options = ['real/', 'fake/']


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


def visualize_results(y_pred, folder_path=folder_path, options=options):

    """_Visualize the results. Plot the input images with the groundtruth and the prediction_

        Parameters:
        ----------
            ...
            folder_path: (string) origin folder
            options: (list) type of face, either 'real' or 'fake'

    """

    real_img = load_dataset(folder_path, options[0])
    fake_img = load_dataset(folder_path, options[1])

    real_img = real_img[:len(fake_img)]

    label = ["REAL", "FAKE"]


    for idx in range(len(real_img)):

        # display real
        cv2.imshow((str(idx) + ". REAL: " + label[int(y_pred[idx])]), real_img[idx])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # display fake
        cv2.imshow((str(idx) + ". FAKE: " + label[int(y_pred[idx + len(real_img)])]), fake_img[idx])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def plot_single_img(img):
    
    cv2.imshow('1', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def compute_AUC(pred_prob, y_true, plot_results=False):

    """_Compute the ROC and plot it. Print the AUC coefficient._

        Parameters:
        ----------
            ...
            plot_results: (bool) if true, plot ROC and save figure

        return:
        ------
            images: list with the loaded images
    """

    idx_largest_prob = np.argmax(pred_prob, axis=1)

    # range {-1, 1} probabilities of the predictions
    #  - max prob fake class =-1
    #  - max prob real class = 1
    pred_prob[:,1] = pred_prob[:,1] * (-1)

    # invert so 1 is the positive class
    y_true = 1 - y_true

    # get the probability
    y_pred_prob = pred_prob[np.linspace(0,pred_prob.shape[0]-1,pred_prob.shape[0], dtype='int32'),idx_largest_prob]

    # compute the AUC
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred_prob, pos_label=1)
    auc_val = metrics.auc(fpr, tpr)

    if plot_results:

        # print AUC
        print('AUC: ',auc_val)
        fig = plt.figure(figsize=(12,8))
        plt.plot(fpr, tpr, 'g')
        plt.title('AUC = '+str("{:.3f}".format(auc_val)))
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        # plt.figure(tight_layout=True)
        plt.show()

        # save figure
        fig.savefig('AUC_'+str("{:.3f}".format(auc_val))+'.jpg')

    return auc_val

