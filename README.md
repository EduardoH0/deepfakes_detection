# deepfakes_detection
Machine learning Deepfake Detection algorithm tested on the UADFV dataset. 

The extracted features are based on the following papers:
- Textures (gradient domain, wavelet transform, co-ocurrence matrix):
   _DeepFake Videos Detection Based on Texture Features
Bozhi Xu , Jiarui Liu , Jifan Liang , Wei Lu, and Yue Zhang_
- Textures (co-ocurrence matrix):
_An Investigation of the Textural Characteristics Associated with Gray Level 
Cooccurrence Matrix Statistical Parameters
Andrea Baraldi and Flavio Panniggian_
- Textures (co-ocurrence matrix and wavelet transform):
_Identification of vibration level in metal cutting using undecimated wavelet transform 
and gray-level co-occurrence matrix texture features
Khalil Khalili, Mozhgan Danesh_
- Wavelet transform:
_Image-based classidication of paper surface quality using wavelet texture analysis Marco S.Reis and Amir Bauer_
- Gabor filters:
_Detecting Deepfakes with Deep Learning and Gabor Filters Wildan J.Hadi, SUhad M. Kadhem, Ayad R. Abbas_

Additional (not tested but should be considered in the future as improvement):
- _Face Spoofing Detection From Single Images Using Micro-Texture Analysis
Jukka Määttä, Abdenour Hadid, Matti Pietikäinen_
Additional feature: Local binary patterns (it seems that may provide better results that gabor 
filters)
- _Efficient Method of Visual Feature Extraction for Facial Image Detection and Retrieval
Ahmed Abdu Alattab, Sameem Abdul Kareem_
“Additional feature”: Color histograms

## Pipeline
### Preprocessing
Through the Cvlib libray, detection and cropping of the face region within the images. 
Resizing of the cropped images to avoid feature bias (resize size based on the size distribution of our dataset).
### Feature extraction
- **Textures based on the gradient domain**: in the gray-scale domain, the image gradient can represent texture since it characterizes
the changes of each pixel in the neighborhood. The vertical and horizontal gradients of the face region are extracted and the magnitude is obtained.
Then, based on the gradient magnitude, the mean, the variance, the skewness, and kurtosis features are extracted to get some 
statistical properties of the data distribution. Along these 4 features, the variance of the grayscale image is extracted. 
- **Texture features based on the gray level co-ocurrence matrix**: the gray level co-ocurrence matrix gives us the probability of
two adjacent gray level pixels appearing in an image with an specific spatial distribution. This co-ocurrence matrix is computed 
in 4 different directions (0, 45, 90, 135), always with unitary distance (in other words, adjacent pixels). From this co-ocurrence matrix 
some features are extracted:
  - Contrast: reflects the richness of the texture details and the depth of the image.
  - Energy: reflects the uniformity of the gray level distribution of the image.
  - Homogeneity: reflects the intensity of the local texture changes.
  - Entropy: measures the amount of information of the local area.
  - Correlation: level of "correlation"
- **Wavelet transform**: a 2D wavelet transform is computed to extract texture features (horizontal, vertical and diagonal
detail of the image, LH, HL, and HH filtering respectively). One wavelet transfrom is performed on the RGB image, and the energy,
standard deviation and mean are calculated. On the grayscale image, a 5 level wavelet transform decomposition is performed and 
the standard deviation of the output ir returned.
- **Gabor filters**: gabor filters are used to extract different texture features. These filters can be heavily parametrized,
thus filers of different wavelenghts and orientations are used. The filters are organized in bathces of 16 (equal to the number
of orientations _theta_) and each pixel of the images takes the higher value of the batch. 

All this features can be extracted either from the **whole face region** (which will result in a 1D array of 53 features) or from 
different regions of the face. The best results are provided by the second method, in which the face area is divided in 9 different
blocks of equal area, and the features are extracted from each of them (yet somme of the features are extracted from the whole face area).
This second approach would yield a 1D array of 205 features. NOTE that this can be improved performing **FACE ALIGNMENT** (not implemented) before dividing the face region, so the 9 blocks
should consistently represent the same face features from image to image.

### Training/Validation splitting
The split can be randomly performed or not. (With the UADFV dataset it might be good practice to manually select the validation data, 
forcing it to be more different to the training data, and avoiding face repetition).

### Normalization
Key step in any implementation. Two different techniques are proposed, normalizing the data within the range {0,1} or standardizing
the data so it has 0 mean and unit variance (probably more adequate the second proposal). 

### Training
Four different classifiers are proposed: support vector machines with different kernels (linear, polynomial, sigmoid, gaussian),
random forest, XGBoost, Naïve Bayes, and Multilayer Perceptron. (The best results were from random forest). 

### Tunning
In my case, the dataset was quite small hence so many features (specially if I divided the faces in 9 regions) were unnecessary 
and counterproductive. Therefore, 3 tunning techniques were proposed:
- **Half/half evaluation**: half of the features are taken and the model is trained adding one by one the features of the other half.
If the acc improves with the additional feature, it is permanently added to the batch. Then, the same is done with the other half, but starting
only with the features that have been selected in the previous analysis. Acc should be measured in validation, not testing!
In order to reduce overfitting, different models might be used for feature selection than the one used for the final prediction 
(for instance SVM with Gaussian kernel + Random Forest for tunning and Random Forest for the final prediction, however, this is
not a common practice and may lack foundation. 
- **Random selection + max(mean(training+validation acc))**: this approach focuses on reducing the overfitting due to a reduced dataset.
It randomly select a "x" number of features, train the model and compute the acc both for training and validation data. Its aim is
to find the combination of features tha maximize the acc of the trainind and validation data at the same time, thus avoiding overfitting.
Different weights can be assigned to the training and validation accuracy. The number of selected features is gradually increased
within a range. This was the tunning used for the final implementation. 
- **Correlated features removal**: remove those features that are highly correlated (a threshold is set to determine when to remove them).

Note: tunning is crucial, specially with so many features and a small dataset. In my final implementation, 20 of 53 features were selected
and 35 of the 205 for the two different approaches.

### Testing
Provides the acc and the AUC score. 


# Deep learning
A simplified version of the VGG16 architecture is proposed as a deep learning model for DeepFakes Detection. (Suitable for small datasets, hence the 
simplicity of the net).
- 1 2D convolutional layer (10 channels)
- 1 max pooling layer
- 2 fully connected layers
- 1 dropout layer
The perk of this approach against machine learning is that we can use convolutional layers, so more spatial information can be extracted.
Relu as activation function and sigmoid for the final layer. SGD as optimizer, lr=0.01, momentum=0.9, batch_size=64.
Weight decay is introduced looking forward reducing the overfitting (for the same reason the dropout layer is included).

A 12-input channel is provided to the model. Three RGB channels (color information), and the other 9 channels correspond to the image filtered
by batches of Gabor filters. Each batch of Gabor filters is made by 16 filters (16 different orientations _theta_), and each batch has
a different wavelength _lambda_.

As in the machine learning approach, the images that are passed to the network correspond to the face region detected on the images by Cvlib.

