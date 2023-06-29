import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
from skimage.transform import resize
from skimage.color import rgb2gray
import os 
import sys

if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.utils import rgb_to_hsv,image_difference,resize_image

def find_vector_set(diff_image, new_size):

    i = 0
    j = 0
    vector_set = np.zeros((int(new_size[0] * new_size[1] / 25), 25))
    while i < vector_set.shape[0]:
        while j < new_size[0]:
            k = 0
            while k < new_size[1]:
                block   = diff_image[j:j+5, k:k+5]
                feature = block.ravel()
                vector_set[i, :] = feature
                k = k + 5
            j = j + 5
        i = i + 1

    mean_vec   = np.mean(vector_set, axis = 0)
    vector_set = vector_set - mean_vec
    return vector_set, mean_vec

def find_FVS(EVS, diff_image, mean_vec, new):

    i = 2
    feature_vector_set = []

    while i < new[0] - 2:
        j = 2
        while j < new[1] - 2:
            block = diff_image[i-2:i+3, j-2:j+3]
            feature = block.flatten()
            feature_vector_set.append(feature)
            j = j+1
        i = i+1

    FVS = np.dot(feature_vector_set, EVS)
    FVS = FVS - mean_vec
    return FVS

def clustering(FVS, components, new):

    kmeans = KMeans(components, verbose = 0)
    kmeans.fit(FVS)
    output = kmeans.predict(FVS)
    count  = Counter(output)

    least_index = min(count, key = count.get)
    change_map  = np.reshape(output,(new[0] - 4, new[1] - 4))
    return least_index, change_map

def find_PCAKmeans(image1, image2,type='gray',factor=5):

    #image1 = imread(imagepath1)
    #image2 = imread(imagepath2)
    if type=='gray':
        image1=image1[:,:,[5,3,1]]
        image2=image2[:,:,[5,3,1]]
        image1=rgb2gray(image1)
        image2=rgb2gray(image2)  

        image1,new_size=resize_image(image1,factor)
        image2,new_size=resize_image(image2,factor)

    elif type=='hsv':
        image1=image1[:,:,[5,3,1]]
        image2=image2[:,:,[5,3,1]]
        image1=rgb_to_hsv(image1)
        image2=rgb_to_hsv(image2) 

        image1,new_size=resize_image(image1,factor)
        image2,new_size=resize_image(image2,factor)

    else:
        image1,new_size=resize_image(image1,factor)
        image2,new_size=resize_image(image2,factor)

    if type=='gray':    
        diff_image = abs(image1 - image2)
    else:
        diff_image = image_difference(image1,image2)


    vector_set, mean_vec = find_vector_set(diff_image, new_size)
    pca     = PCA()
    pca.fit(vector_set)
    EVS = pca.components_

    FVS     = find_FVS(EVS, diff_image, mean_vec, new_size)
    components = 3
    least_index, change_map = clustering(FVS, components, new_size)

    change_map[change_map == least_index] = 255
    change_map[change_map != 255] = 0

    change_map = change_map.astype(np.uint8)
    kernel     = np.asarray(((0,0,1,0,0),
                             (0,1,1,1,0),
                             (1,1,1,1,1),
                             (0,1,1,1,0),
                             (0,0,1,0,0)), dtype=np.uint8)
    cleanChangeMap = cv2.erode(change_map,kernel)
    cleanChangeMap  = cleanChangeMap/255
    cleanChangeMap = cleanChangeMap.astype('uint8')
    return cleanChangeMap,diff_image
