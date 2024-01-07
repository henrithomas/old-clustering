#Henri Thomas
#K-Means Clustering
#4/1/18

import numpy as np
import math
from scipy.spatial import distance
#Number of clusters:
k = 19
#Use cosine similarity?
withCos = True
#Used for calculating entropy for sedentary, active, and exercising 
SAE = False
#Holds distances from a vector to a centroid:
dist = np.zeros(k)

def euclideanDist(a,b):
    return distance.euclidean(a,b)

def cosineSim(a,b):
   return 1 - distance.cosine(a,b)

def convergenceC(currC,prevC):
    #Computes the cosine similarity between previous centroids prevC
    #and current centroids currC. Returns true if all centroids have moved
    #very little, or not at all. 
    s = 0
    for i in range(k):
        if cosineSim(currC[i],prevC[i]) >= .999:
            s = s + 1
    if s == k:
        return True
    else:
        return False

def recomputeCentroids(S,C):
    #Takes the vector sets S for each cluster and computes its
    #centroid C
    for l in range(k):
        mat = S[l]
        n = mat.shape[0]
        if n != 0:
            C[l] = 1/n * mat.sum(axis = 0)
    return C
    
def clusterEucl(D,C,labels,n):
    #Measures the euclidean distance between the vectors in D
    #and the centroids in C to assign a cluster. 
    for i in range(n):
        for j in range(k):
            dist[j] = euclideanDist(D[i],C[j])
        labels[i] = np.argmin(dist)
    return labels
 
def clusterCos(D,C,labels,n):
    #Measures the cosine similarity between the vectors in D  
    #and the centroids in C to assign a cluster.
    for i in range(n):
        for j in range(k):
            dist[j] = cosineSim(D[i],C[j])
        labels[i] = np.argmax(dist)
    return labels
    
def setClusterSets(D,labels,S):
    #Returns the sets of vectors for each of the clusters.
    for i in range(k):
        S.append(D[np.argwhere(labels == float(i))[:,0],:])
    return S
    
def runClustering(data,centroids,nlabs,plabs,n,mode): 
    convergence = False
    if mode == False:
        print('Running clustering with euclidean distance, k =',k)
    else:
        print('Running clustering with cosine similarity, k =',k)
    clusterSet = []
    while convergence == False:
        #Run clustering with cosine similarity or eucl. distance
        if mode == True:
            nlabs = clusterCos(data,centroids,nlabs,n)
        else:
            nlabs = clusterEucl(data,centroids,nlabs,n)
        #Set the cluster sets for the current iteration
        clusterSet.clear()
        clusterSet = setClusterSets(data,nlabs,clusterSet)   
        centroidsPrev = centroids.copy()
        centroids =  recomputeCentroids(clusterSet,centroids)
        convergence = convergenceC(centroids,centroidsPrev)
    print('Converged.')
    return (centroids, nlabs, clusterSet)
    
def meanEntropy(new,original,mode):
    #Computes the mean entropy for the clustering
    entropies = np.zeros(k)
    temp = original.astype(int)
    cSize = temp.shape[0]   
    #Relabels original activities to Sedentary: 0, Active: 1, Exercising: 2
    if mode == True:
        for j in range(cSize):
            if (temp[j] >= 1 and temp[j] <= 4) or temp[j] == 7 or temp[j] == 8:
                temp[j] = 0
            elif (temp[j] >= 9 and temp[j] <= 11) or temp[j] == 5 or temp[j] == 6:
                temp[j] = 1
            else:
                temp[j] = 2
    
    #Calculates entropy for each cluster
    e = []
    for i in range(k):
        m = np.transpose(temp[np.argwhere(new == i)]).flatten()
        size = len(m)
        counts = np.bincount(np.transpose(m).flatten())
        for n in range(len(counts)):
            if counts[n] != 0:
                e.append((counts[n]/size) * math.log10(counts[n]))
        eSum = -1 * sum(e)
        entropies[i] = (size/cSize) * eSum
        e.clear()
    #Returns mean entropy of clusters
    return sum(entropies)
    
if __name__ == '__main__':
    data = np.genfromtxt('formatted.csv', delimiter = ',')
    numInstances = data.shape[0]
    originalLabels = data[:,0].astype(int)
    data = data[:,1:data.shape[1]]
    newLabels = np.zeros((numInstances,1))
    prevLabels = np.zeros((numInstances,1))
    prevLabels = originalLabels
    
    indices = np.random.randint(data.shape[0], size=k)
    centroids = data[indices,:]
    centroids, newLabels, Clusters = runClustering(data,centroids,newLabels,prevLabels,numInstances,withCos)
    Entropy = meanEntropy(newLabels.copy(),originalLabels.copy(),SAE)
    print('Final mean entropy:',Entropy)
    
    
    