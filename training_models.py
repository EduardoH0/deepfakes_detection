import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier




def get_results(model, f_train, train_labels, f_pred, labels_pred, model_name, return_prob=False):

    """_Plot confusion matrix and return predicted probabilities_

        Parameters:
        ----------
            ...
            return_prob = (bool) if True, returns predicted probabilites (only for random forest model)
    """

    # train model
    model.fit(f_train, train_labels.ravel())
    # get predictions
    y_pred = model.predict(f_pred)
    # print acc
    print(model_name, np.sum(y_pred==labels_pred.ravel()) / len(labels_pred.ravel()),' acc')
    # print confusion matrix
    CM = confusion_matrix(labels_pred, y_pred)
    print(CM)

    if return_prob:
        predictions = model.predict_proba(f_pred)
        return predictions


def train_model(f_train, f_val, f_test, mode='val'):

    """_Train with different classifiers_

        Parameters:
        ----------
            ...
            mode = (string) select the data in which to run the predictions (default validation)
    """
    # get the labels
    train_labels = f_train[:,-1]
    val_labels   = f_val[:,-1]
    test_labels  = f_test[:,-1]

    # remove labels
    f_train = f_train[:,:-1]
    f_val   = f_val[:,:-1]
    f_test  = f_test[:,:-1]

    # shuffle train data
    f_train, train_labels = shuffle(f_train, train_labels)

    ## TRAINING
    if mode=='val':
        f_pred = f_val
        labels_pred = val_labels
    elif mode=='test':
        f_pred = f_test
        labels_pred = test_labels

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
    get_results(model, f_train, train_labels, f_pred, labels_pred, 'Random Forest: ')

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

