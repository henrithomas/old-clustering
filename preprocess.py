#Dakota Crowder
#Henri Thomas
#Henry Thomas
#
#Mar 25, 2018
#
#Dependencies:
#    python 2.x.x
#    scikit
#    numpy

from sklearn.decomposition import PCA
from scipy.signal import periodogram
import numpy as np
import os

#returns a list of list of lists of 2d numpy arrays
#print activities[6][5][29] #7th activity, 6th person, 30th segment
def importData(path):
    filepaths = []
    for x in os.walk(path):
        for i in range(60):
            if len(x[2]) > 0:
                filepaths.append(x[0] + '/' + str(x[2][i])) #filepaths.append(x[0] + '\\' + str(x[2][i]))
    activities = []
    subjects = []
    segments = []
    counter = 0
    for i in range(19):
        for j in range(8):
            for k in range(60):
                #print('processing file ' + str(counter + 1) + ' of ' + str(len(filepaths)))
                data = np.genfromtxt(filepaths[counter], delimiter = ',')
                counter += 1
                segments.append(data)
            subjects.append(segments)
            segments = []            
        activities.append(subjects)
        subjects = []
        segments = []        
    return activities

# takes in the file path for the folders of data, gets the array of
# the all the text file values for each activity, person and segment,
# then iterates through them and creates a periodogram from the 45
# different measurements from the 3 sensors for the 5 units on the person
# being measured, then sends it to pca to be reduced then returned from this
# function
def formatData(importedData):
    #x, y, z = importedData.shape
    x = 19
    y = 8
    z = 60
    formatted = []
    for activity in range(x):
        for person in range(y):
            for segment in range(z):
                file_psd = [activity]
                for i in range(45):
                    array = importedData[activity][person][segment][:, i]
                    freq, pxx = periodogram(np.fft.fft(array), 25)
                    file_psd.extend(pxx)
                formatted.append(file_psd)
    return formatted

#reduces the data set using PCA 
def pca(feature_mat):
    clusterPCA = PCA(300)
    clusterPCA.fit(feature_mat)
    feature_mat = clusterPCA.transform(feature_mat)
    return feature_mat

#writes data to a csv file
def writeData(formatted):
    np.savetxt('formatted.csv', formatted, delimiter = ',')

if __name__ == '__main__':
    print('Importing data...')
    importedData = importData('data') #importedData = importData(path)
    print('Preprocessing data...')
    formatted = formatData(importedData)
    formatted = np.asmatrix(formatted)
    print('Feature set shape:',formatted.shape)
    print('Performing PCA...')
    labels = formatted[:,0]
    formatted = pca(formatted[:,1:formatted.shape[1]])
    pcaformatted = np.concatenate((labels,formatted),axis=1)
    print('Writing data...')
    writeData(pcaformatted)
    print('PCA feature set shape:',formatted.shape)
    print('Done.')
    
    
    
    
    