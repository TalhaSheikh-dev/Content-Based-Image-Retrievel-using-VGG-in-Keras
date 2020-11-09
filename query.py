# -*- coding: utf-8 -*-
from extract_cnn_vgg16_keras import VGGNet

import numpy as np
import h5py
import os
import matplotlib.image as mpimg
import argparse
import cv2
import shutil



# read in indexed images' feature vectors and corresponding image names
os.environ["CUDA_VISIBLE_DEVICES"] = ""

shutil.rmtree("output")
os.mkdir("output")

h5f = h5py.File('featureCNN_map.h5','r')

feats = h5f['feats'][:]
# print(feats)

imgNames = h5f['names'][:]
h5f.close()


# read and show query image
queryDir = 'database/caltex256/006_0001.jpg'
queryImg = mpimg.imread(queryDir)
save = cv2.cvtColor(queryImg,cv2.COLOR_RGB2BGR)
cv2.imwrite("output/query.jpg",save)

# init VGGNet16 model
model = VGGNet()

# extract query image's feature, compute simlarity score and sort
queryVec = model.extract_feat(queryDir)
scores = np.dot(queryVec, feats.T)
rank_ID = np.argsort(scores)[::-1]
rank_score = scores[rank_ID]
#print rank_ID
#print rank_score


# number of top retrieved images to show
maxres = 10
imlist = [imgNames[index] for i,index in enumerate(rank_ID[0:maxres])]

# show top #maxres retrieved result one by one
for i,im in enumerate(imlist):

    image = mpimg.imread('database/caltex256'+"/"+str(im, 'utf-8'))
    save = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    cv2.imwrite("output/{}.jpg".format(i),save)
