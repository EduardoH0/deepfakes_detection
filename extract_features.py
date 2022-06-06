from ctypes import sizeof
from cv2 import mean
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from scipy.stats import kurtosis, skew
from numba import jit
import pywt
from sklearn.preprocessing import StandardScaler


kernelx = np.array([[ 1,  2,  1],
                    [ 0,  1,  0],
                    [-1, -2, -1]])

kernely = np.array([[-1,  0,  1],
                    [-2,  0,  2],
                    [-1,  0,  1]])



def compute_gradient(gray): #5 features

    """_Returns multiple properties related with the gradients magnitude of the input image_

        return
        ------
            std_gray: standard deviation of the input image (gray scale)
            mean_mag: mean of the gradient magnitude 
            std_mag: standard deviation of the gradient magnitude
            skew_mag: skewness of the gradient magnitude
            kurtosis_mag: kurtosis coefficient of the gradient magnitude
    """


    Gx   = cv2.filter2D(gray, ddepth=cv2.CV_32F, kernel=kernelx) # enough with CV_32F?
    Gy   = cv2.filter2D(gray, ddepth=cv2.CV_32F, kernel=kernely)

    N = gray.size

    # std gray_scale
    std_gray = np.std(gray)

    # magnitude gradient 
    magnitude = np.sqrt(Gx**2 + Gy**2)

    # Normalization 
    # magnitude = (magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude))

    # # mean gradient
    mean_mag = np.mean(magnitude)

    # std gradient
    std_mag = np.std(magnitude)

    # skew => sum(Xi - mean)**3 / ((N-1) * std**3)
    skew_mag = skew(magnitude.flatten())

    # val_t = 0
    # for i in range(gray.shape[0]):
    #     for j in range(gray.shape[1]):
    #         val_t += (magnitude[i,j] - mean_mag)**3
    # val_t = val_t / ((gray.size -1) * std_mag**3)

    # kurtosis => sum(Xi - mean)**4 / (N * std**4) (Pearson's definition)
    kurtosis_mag = kurtosis(magnitude.flatten(), fisher=False)

    # val_t = 0
    # for i in range(gray.shape[0]):
    #     for j in range(gray.shape[1]):
    #         val_t += (magnitude[i,j] - mean_mag)**4 
    # val_t = val_t / (N * std_mag**4)

    return(std_gray, mean_mag, std_mag, skew_mag, kurtosis_mag)


def co_ocurrence_matrix(gray):

    """_Returns the coocurrence matrix of the input image (within spefified range)_
        
        return
        ------
            coom: coocurrence matrix (pixels until level 64)

    """

    coom = np.zeros(shape=([256, 256, 4]))


    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            
            #   0 degrees
            if j != gray.shape[1] - 1:
                coom[gray[i,j], gray[i,j+1], 0] += 1
            #  90 degrees
            if i != gray.shape[0] - 1:
                coom[gray[i,j], gray[i+1,j], 2] += 1

            if i < gray.shape[0] - 1 and j < gray.shape[1] - 1:
                #  45 degrees
                coom[gray[i+1,j], gray[i,j+1], 1] += 1
                # 135 degrees
                coom[gray[i,j], gray[i+1,j+1], 3] += 1

    return coom[0:64, 0:64, :]


def features_coomatrix(img): #5 features

    """_Returns features based on the coomatrix on the input image_
        
        return
        ------
            contrast: richness of the texture details and depth
            energy: uniformity of the gray level
            homogeinity: intensity of the local texture changes
            entropy: amount of information of the local area
            correlation: degree of correlation

    """

    coom = co_ocurrence_matrix(img)

    # avoid 0 in the co-ocurrence matrix
    coom = coom + 1

    f_contrast    = np.zeros([1, coom.shape[2]])
    f_homogeinity = np.zeros([1, coom.shape[2]])
    f_correlation = np.zeros([1, coom.shape[2]])

    # entropy (amount of information of the local area)
    f_entropy = np.sum(-coom*np.log(coom), axis=(0,1))
    
    # energy (uniformity of the gray level)
    f_energy = np.sum(coom**2, axis=(0,1))

    # correlation ecuation elements
    i_matrix, j_matrix = np.mgrid[1:coom.shape[0]+1:1, 1:coom.shape[1]+1:1]
    mean_i  = np.zeros([1, coom.shape[2]])
    mean_j  = np.zeros([1, coom.shape[2]])
    sigma_i = np.zeros([1, coom.shape[2]])
    sigma_j = np.zeros([1, coom.shape[2]])

    

    
    for k in range(coom.shape[2]):
        mean_i[0,k]  = np.sum(coom[:,:,k] * i_matrix)
        mean_j[0,k]  = np.sum(coom[:,:,k] * j_matrix)

        sigma_i[0,k] = np.sqrt(np.sum(coom[:,:,k] * (i_matrix - mean_i[0,k])**2)) 
        sigma_j[0,k] = np.sqrt(np.sum(coom[:,:,k] * (j_matrix - mean_j[0,k])**2))   


    for i in range(coom.shape[0]):
        for j in range(coom.shape[1]):

            # contrast (richness of the texture details and depth)
            f_contrast += coom[i,j,:] * (i+1 - (j+1))**2
            
            # homogeneity (intensity of the local texture changes)
            f_homogeinity += coom[i,j,:] / (1 + np.abs(i-j))

            # correlation (degree of correlation)
            f_correlation += (i+1 - mean_i) * (j+1 - sigma_j) * coom[i,j,:] / (sigma_i * sigma_j)


    # calculate the average of directions (0, 45, 90, 135)
    f_contrast = np.mean(f_contrast)
    f_energy = np.mean(f_energy)
    f_homogeinity = np.mean(f_homogeinity)
    f_entropy = np.mean(f_entropy)
    f_correlation = np.mean(f_correlation)


    return f_contrast, f_energy, f_homogeinity, f_entropy, f_correlation


def wavelet_tranform(img, color=False): #9 features

    """_Returns features based on the wavelet transform (one level)_

        (ca, (ch, cv, cd)) => Approximation, horizontal detail, vertical detail and diagonal detail coefficients respectively
            ca: low + low filter (LL)
            ch: low + high filter (LH)
            cv: high + low filter (HL)
            cd: high + high filter (HH)
        
        Parameters:
        ----------
            color: (bool) if True, coefficients computed on color image

        return
        ------
            avg_value_(x): mean value of each coefficient
            std_value_(x): standard deviation of each coefficient
            energy_value_(x): energy of each coefficient
    """


    if color:
        _, (chR, cvR, cdR) = pywt.dwt2(img[:,:,2], 'haar')
        _, (chG, cvG, cdG) = pywt.dwt2(img[:,:,1], 'haar')
        _, (chB, cvB, cdB) = pywt.dwt2(img[:,:,0], 'haar')

        ch = np.stack([chR,chG,chB], axis=2)
        cv = np.stack([cvR,cvG,cvB], axis=2)
        cd = np.stack([cdR,cdG,cdB], axis=2)


    else:

        _, (ch, cv, cd) = pywt.dwt2(img, 'haar')

    # average value
    # avg_value_ca = np.mean(ca, axis=(0,1))
    avg_value_ch = np.mean(ch) # horizontal
    avg_value_cv = np.mean(cv) # vertical
    avg_value_cd = np.mean(cd) # diagonal

    # standard deviation
    std_value_h = np.std(ch)
    std_value_v = np.std(cv) 
    std_value_d = np.std(cd)

    # energy
    energy_value_h = 1/(ch.size) * np.sum(ch)
    energy_value_v = 1/(cv.size) * np.sum(cv) 
    energy_value_d = 1/(cd.size) * np.sum(cd)

    return  avg_value_ch, avg_value_cv, avg_value_cd, std_value_h, std_value_v, std_value_d, energy_value_h, energy_value_v, energy_value_d

def wavelet_tranform2(img, levels=5, w_type = 'haar', color=False): #9 features

    """_Returns features based on the wavelet transform (one level)_

        (ca, (ch, cv, cd)) => Approximation, horizontal detail, vertical detail and diagonal detail coefficients respectively
            ca: low + low filter (LL)
            ch: low + high filter (LH)
            cv: high + low filter (HL)
            cd: high + high filter (HH)
        
        Parameters:
        ----------
            levels: (int) number of levels of the decomposition
            w_type: (string) type of wavelet transform
            color: (bool) if True, coefficients computed on color image
            

        return
        ------
            feature_vector: vector with the std of the coefficients 
    """

    feature_vector = []

    if color:

        caR = img[:,:,2]
        caG = img[:,:,1]
        caB = img[:,:,0]

        # wavelet transform for number of levels
        for i in range(levels):

            caR, (chR, cvR, cdR) = pywt.dwt2(caR, wavelet=w_type)
            caG, (chG, cvG, cdG) = pywt.dwt2(caG, wavelet=w_type)
            caB, (chB, cvB, cdB) = pywt.dwt2(caB, wavelet=w_type)

            ch = np.stack([chR,chG,chB], axis=2)
            cv = np.stack([cvR,cvG,cvB], axis=2)
            cd = np.stack([cdR,cdG,cdB], axis=2)

            # standard deviation
            feature_vector.append(np.std(ch))
            feature_vector.append(np.std(cv))
            feature_vector.append(np.std(cd))
        
        # finally the coefficients
        ca = np.stack([caR,caG,caB], axis=2)
        feature_vector.append(np.std(ca))

    # gray level
    else:

        # compute the features from the different levels
        for i in range(levels):

            ca, (ch, cv, cd) = pywt.dwt2(img, wavelet=w_type)
            img = ca

            # standard deviation
            feature_vector.append(np.std(ch))
            feature_vector.append(np.std(cv))
            feature_vector.append(np.std(cd))
    
        feature_vector.append(np.std(ca))

    return  np.array(feature_vector)


def get_gabor_filters(k_size=31, n_theta=16, lambda_val=2, sigma_val=4, gamma_val=0.5):

    """_Returns a batch of gabor filters_
        
        Parameters:
        ----------
            k_size: (int) size of the kernel 
            theta : (int) orientation
            sigma : (float) standard deviation of the gaussian envelope
            lambda: (float) wavelength of the sinusoidal factor.
            gamma : (float) spatial aspect ratio
            phi   : (float) pahse offset
            

        return
        ------
            filters: batch of filters
    """

    filters = []
    for lambda_value in lambda_val:
        for theta_val in np.arange(0, np.pi, np.pi / n_theta):
            kern = cv2.getGaborKernel(ksize=(k_size, k_size), sigma=sigma_val, theta=theta_val, lambd=lambda_value, gamma=gamma_val, psi=0, ktype=cv2.CV_32F)
            # kern /= 1.5*kern.sum()
            # kern /= kern.sum()
            filters.append(kern)

    return filters


def gabor_filter(img, filters, n_batches):

    """_Returns features based on gabor filters_
        
        Parameters:
        ----------
            filters: (list) gabor kernels
            n_batches: (int) number of filters that are going to be considered together (batch of n_batches) 
            
        return
        ------
            feat_gabor: array with mean and std features from the filtered images
    """
    
    feat_gabor = []

    # apply filters in batches of orientations
    for i in range(n_batches):

        filtered_img = np.zeros_like(img)
        filter_batch = filters[0+(n_batches*i) : n_batches+(n_batches*i)]

        for kern in filter_batch:
            fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
            np.maximum(filtered_img, fimg, filtered_img) # stay with the max value of each pixel

        # mean value
        feat_gabor.append(np.mean(filtered_img))
        # std value
        feat_gabor.append(np.std(filtered_img))

    return np.array(feat_gabor)


def features_normalization(features_train, features_val, features_test, zero_mean = True):

    """_2 ways of feature normalization_
        
        Parameters:
        ----------
            zero_mean: (bool) select normalization mode
                True => zero mean and unit variance
                False => normalize features within {0, 1}
            
        return
        ------
            normalized features (train, validation and test)
    """

    # transform to: mean=0 std=1
    if zero_mean:
        scaler = StandardScaler()
        features_train[:,:-1] = scaler.fit_transform(features_train[:,:-1])
        features_val[:,:-1]   = scaler.transform(features_val[:,:-1])
        features_test[:,:-1]  = scaler.transform(features_test[:,:-1])

    else:
        for i in range(features_train.shape[1]-1):

            max_f = np.max(features_train[:,i])
            min_f = np.min(features_train[:,i])

            features_train[:,i] = (features_train[:,i] - min_f) / (max_f - min_f) 
            features_val[:,i]   = (features_val[:,i] - min_f) / (max_f - min_f) 
            features_test[:,i]  = (features_test[:,i] - min_f) / (max_f - min_f)


    return features_train, features_val, features_test

# 
def divide_image(img, n, color=False):

    """Creates a grid in the inpute image and returns the different blocks_
        
        Parameters:
        ----------
            n: (int) number of blocks per image
            color: (bool) divide color or grayscale input image
            
        return
        ------
            img_slices: blocks of the input image
    """

    img_slices = []
    rows_img = img.shape[0]//n
    columns_img = img.shape[1]//n
    
    if not color:
        for i in range(0, img.shape[0]//n*n, rows_img):
            for j in range(0, img.shape[1]//n*n, columns_img):
                img_slices.append(img[i:i+rows_img, j:j+columns_img])
        return img_slices

    else:
        for i in range(0, img.shape[0]//n*n, rows_img):
            for j in range(0, img.shape[1]//n*n, columns_img):
                img_slices.append(img[i:i+rows_img, j:j+columns_img,:])
        return img_slices
        






