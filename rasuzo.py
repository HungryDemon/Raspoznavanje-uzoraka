#!/usr/bin/python3
# -*- coding: utf-8 -*-

import glob
import os
import numpy as np
import cv2
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import shannon_entropy
import pickle
from neupy import algorithms, environment
import matplotlib.pyplot as plt


path = '/home/mirko/FER/Diplomski/RU/Database/Sorted/'
test_names=[]

def main():

    preprocess()
    #normalise()
    
    alldata=[]


#    with open (path+'normalised', 'rb') as fp:
    with open (path+'outfile.1', 'rb') as fp:

        alldata = pickle.load(fp)
    
    for data in alldata:
        plt.scatter(data[0], data[1], color='silver', alpha=0.1)
    
    sofmnet = algorithms.SOFM(
        n_inputs=2,
        n_outputs=5,
    
        step=0.01,
        std=0.1,
        reduce_step_after=200,
        reduce_radius_after=2,
        reduce_std_after=20,
        show_epoch=10,
        shuffle_data=True,
        verbose=True,

        learning_radius=2,
        distance='euclid',
        features_grid=(5, 1),
    )
    
    sofmnet.train(alldata, epochs=10)

    plt.scatter(sofmnet.weight[0:1,:], sofmnet.weight[1:2,:], color='red')
    
    with open (path+'test_output.1', 'rb') as fp:

        testdata = pickle.load(fp)
                
    for data in testdata:
        plt.scatter(data[0], data[1], color='green')
        
    test(sofmnet)
    plt.savefig('graph_with_test.png')
    plt.show()


        
def test(som):
    testdata=[]
#    with open (path+'test_normalised', 'rb') as fp:
    with open (path+'test_output.1', 'rb') as fp:

        testdata = pickle.load(fp)
                
    for i in range(len(testdata)):
        print(test_names[i])
        print(som.predict(testdata[i]))
        print()
        
        
def normalise():
    alldata=[]
    testdata=[]

    contrastMax=0
    homogenityMax=0
    energyMax=0
    entropyMax=0

    with open (path+'outfile', 'rb') as fp:
        alldata = pickle.load(fp)

    with open (path+'test_output', 'rb') as fp:
        testdata = pickle.load(fp)
    
    for data in alldata:
        for i in range(4):
            if data[i]>contrastMax:
                contrastMax=data[i]
                
            if data[i+4]>homogenityMax:
                homogenityMax=data[i+4]

            if data[i+8]>energyMax:
                energyMax=data[i+8]
                
            if data[i+12]>entropyMax:
                entropyMax=data[i+12]
                
    for data in alldata:
        for i in range(4):
            data[i]/=contrastMax
                
            data[i+4]/=homogenityMax

            data[i+8]/=energyMax
                
            data[i+12]/=entropyMax
            
    for data in testdata:
        for i in range(4):
            data[i]/=contrastMax
                
            data[i+4]/=homogenityMax

            data[i+8]/=energyMax
                
            data[i+12]/=entropyMax

    with open(path+'normalised', 'wb') as fp:
        pickle.dump(alldata, fp)
        
    with open(path+'test_normalised', 'wb') as fp:
        pickle.dump(testdata, fp)

def preprocess():
    
    alldata=[]
    
    for filename in glob.glob(os.path.join(path, "**/*.jpg"),recursive=True):
        img = cv2.imread(filename,0)
#        gldm = greycomatrix(img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])
        gldm = greycomatrix(img, [1], [0, np.pi/2])
        data=[]
        data.extend(greycoprops(gldm,prop='contrast')[0])
#        data.extend(greycoprops(gldm,prop='homogeneity')[0])
#        data.extend(greycoprops(gldm,prop='energy')[0])
#        data.append(shannon_entropy(gldm[:,:,0,0]))
#        data.append(shannon_entropy(gldm[:,:,0,1]))
#        data.append(shannon_entropy(gldm[:,:,0,2]))
#        data.append(shannon_entropy(gldm[:,:,0,3]))

        alldata.append(data)
        
    testdata=[]
    for filename in glob.glob(os.path.join(path, "TEST/*.jpg")):
        spstr = filename.split('/')
        test_names.append(spstr[len(spstr)-1])

        img = cv2.imread(filename,0)
#        gldm = greycomatrix(img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])
        gldm = greycomatrix(img, [1], [0, np.pi/2])

        data=[]
        data.extend(greycoprops(gldm,prop='contrast')[0])
#        data.extend(greycoprops(gldm,prop='homogeneity')[0])
#        data.extend(greycoprops(gldm,prop='energy')[0])
#        data.append(shannon_entropy(gldm[:,:,0,0]))
#        data.append(shannon_entropy(gldm[:,:,0,1]))
#        data.append(shannon_entropy(gldm[:,:,0,2]))
#        data.append(shannon_entropy(gldm[:,:,0,3]))
        testdata.append(data)
        
#    print(gldm.shape)
#    print(greycoprops(gldm,prop='contrast'))
#    print(greycoprops(gldm,prop='homogeneity')) 
#    print(greycoprops(gldm,prop='energy'))
#    print(shannon_entropy(gldm[:,:,0,0]))
#    print(shannon_entropy(gldm[:,:,0,1]))
#    print(shannon_entropy(gldm[:,:,0,2]))
#    print(shannon_entropy(gldm[:,:,0,3]))

    with open(path+'outfile.1', 'wb') as fp:
        pickle.dump(alldata, fp)

    with open(path+'test_output.1', 'wb') as fp:
        pickle.dump(testdata, fp)

        
if __name__ == '__main__':
    main()
