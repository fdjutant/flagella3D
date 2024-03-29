import numpy as np
from skimage import measure
from skimage.morphology import skeletonize_3d
import cv2 

#%% image processing
class ImPro:
    def __init__(self, volImage, thresValue):
        self.volImage = volImage
        self.thresValue = thresValue

    def thresVol(self):    # threshold volumetric image
        img0 = self.volImage
        # img0 = img0/max(img0.ravel())
        # img = img0>self.thresValue
        img = img0 > np.mean(img0[np.nonzero(img0)]) / self.thresValue
        
        return img
    
    def selectLargest(self): # only keep largest body
        img = self.thresVol()
        blobs = np.uint8(measure.label(img, background=0))
        kernel = np.ones((6,6), np.uint8);
        blobs = cv2.morphologyEx(blobs, cv2.MORPH_CLOSE, kernel, iterations=1)
        labels = np.unique(blobs.ravel())[1:]
        sizes = np.array([np.argwhere(blobs==l).shape[0] for l in labels])

        return sizes, blobs, labels
    
    def BlobAndSkel(self):   # create blob and skel
        sizes, blobs, labels = self.selectLargest()
        keep = labels[np.argwhere((sizes == max(sizes)))[0]]
        blob = blobs == keep
        # blobSkel = skeletonize_3d(blob) # skeletonize image
        
        return blob
    
    def extCoord(self):    # extract coordinates
        blob = self.BlobAndSkel()
        X0 = np.argwhere(blob).astype('float') # coordinates 
        return X0
        
    def computeCM(self):   # compute center of mass
        X0 = self.extCoord()
        CM1 = np.array([sum(X0[:,j]) for j in range(X0.shape[1])])/X0.shape[0]
        return CM1 