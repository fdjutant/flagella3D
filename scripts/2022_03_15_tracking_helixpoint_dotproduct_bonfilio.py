#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 23:51:29 2022

@author: bonfilio
"""

import numpy as np
from skimage.morphology import skeletonize_3d
from sklearn.decomposition import PCA

data = np.load('0.npy')[0]
points = np.argwhere(data == 1)

#step to get centerline, can be replaced with fitting
center = skeletonize_3d(data) 
c_points = np.argwhere(center == 255)

#Change to float
points = points.astype('float')
c_points = c_points.astype('float')

#Shift to CM
CM = points.mean(axis=0)
points -= CM
c_points -= CM 

#Get the main axis from PCA
pca = PCA(n_components=3)
pca.fit(points)
n0 = pca.components_[0]
n0 = n0/np.linalg.norm(n0) 

#Determine initial point by setting a "score" value
score_target = 0.7

#Calculate scores via dot product
scores = np.array([np.dot(n0, c_points[i])/np.linalg.norm(c_points[i])\
                   for i in range(len(c_points))])
scores -= score_target

#Get candidate target points where the dot product is close to the target score
tol = 1e-3
idx = np.argwhere(abs(scores)<1e-3)
candidate = np.array([c_points[i[0]] for i in idx]) 

#Get center of mass of candidates and use Gram-Schmidt
avg = candidate.mean(axis=0)
n1 = avg - np.dot(avg,n0)*n0
n1 = n1/np.linalg.norm(n1)
n2 = np.cross(n0,n1)






