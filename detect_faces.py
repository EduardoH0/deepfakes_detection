import cv2
import dlib
import cvlib as cv
import time

import numpy as np
import matplotlib.pyplot as plt



def get_bb_face(img, rect):

    # extract the starting and ending (x, y)-coordinates of the
	# bounding box
	startX = rect.left()
	startY = rect.top()
	endX   = rect.right()
	endY   = rect.bottom()
	# ensure the bounding box coordinates fall within the spatial
	# dimensions of the image
	startX = max(0, startX)
	startY = max(0, startY)
	endX = min(endX, img.shape[1])
	endY = min(endY, img.shape[0])
	# compute the width and height of the bounding box
	w = endX - startX
	h = endY - startY
	# return our bounding box coordinates
	return (startX, startY, w, h)

def face_detector_dlib(image):

    # load dlib's CNN face detector
    print("[INFO] loading CNN face detector...")
    detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

    # load the input image from disk, resize it, and convert it from
    # BGR to RGB channel ordering (which is what dlib expects)
   
    # image = imutils.resize(image, width=600)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # perform face detection using dlib's face detector
    start = time.time()
    print("[INFO[ performing face detection with dlib...")
    results = detector(rgb, 1)
    end = time.time()
    print("[INFO] face detection took {:.4f} seconds".format(end - start))

    # convert the resulting dlib rectangle objects to bounding boxes,
    # then ensure the bounding boxes are all within the bounds of the
    # input image
    boxes = [get_bb_face(image, r.rect) for r in results]
    # loop over the bounding boxes
    for (x, y, w, h) in boxes:
        # draw the bounding box on our image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # show the output image
    cv2.imshow("Output", image)
    cv2.waitKey(0)


def face_detector_cvlib(img, mode, resize=False, x=150, y=220):

    """_Find a potential face region in the input face image, and return that cropped region.
    Resize the cropped region by bilinear interpolation._

        Parameters:
        ----------
            ...
            x, y: (int, int) desired size for resizing

        return:
        ------
            img: face region, cropped and resized
    """

    # get the faces bb
    faces, _ = cv.detect_face(img)
    
    try:
        
        # Try get the first face prediction
        x1, y1, x2, y2 = faces[0]

        # Check other predictions if some of these statements doesn't hold
        if (x1<0 or y1<0 or x2>img.shape[1] or y2>img.shape[0]):
            for i in range(1, len(faces)):
                x1, y1, x2, y2 = faces[i]
                # break when prediction is within the boundaries of the img
                if (x1>0 and y1>0 and x2<img.shape[1] and y2<img.shape[0]):
                    break
        
        if not resize:
            # return cropped img
            return img[y1:y2, x1:x2]
        else:
            return cv2.resize(img[y1:y2, x1:x2], (x, y)) # bilinear interpolation default
                
    except:
        
        print('No face was detected')

        if mode=="development":
            pass
        else:
            return cv2.resize(img, (x, y)) # bilinear interpolation default

#
def get_resolution(real_img, fake_img, save):
    
    """_Returns a graph with the resolution of all the input images_
    """
    
    x = []
    y = []
    for i in range(len(real_img)):

        x.append(real_img[i].shape[0])
        y.append(real_img[i].shape[1])
        
        x.append(fake_img[i].shape[0])
        y.append(fake_img[i].shape[1])

    print(len(x))
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    points = ax.scatter(np.array(y), np.array(x), color='blue', alpha=0.5)
    ax.set_title("Image Resolution")
    ax.set_xlabel("Width", size=14)
    ax.set_ylabel("Height", size=14)
    plt.xticks(np.arange(0, 500, 20))
    plt.yticks(np.arange(0, 500, 20))
    plt.show()

    if save:
        plt.savefig('Faces_dimension.png')