from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist

from keras.models import Sequential
from keras.layers.core import Dense
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

def consufionMatrix(svcl,X_t,y_t):
    y_p = svcl.predict(X_t)
    print(confusion_matrix(y_t,y_p))    
def consufionMatrixKNN(svcl,X_t,y_t):
    y_p = svcl.predict(X_t)
    y_p = pd.DataFrame(y_p).idxmax(axis=1)
    print(confusion_matrix(y_t[:, 0],y_p.to_numpy()))

def crearSVM(kernel,X_train,y_train):
    svcl = SVC(kernel=kernel, gamma=10)
    svcl.fit(X_train,y_train)
    return svcl

def kmeans(k,X):
    kmeans = KMeans(k)
    kmeans.fit(X)
    return kmeans

def crearModelo(X_train, y_train):
    k = 5
    knn = KNeighborsClassifier(k)
    model= knn.fit(X_train,y_train)
    return model

df = pd.read_csv('nuevaData.csv', delimiter=',')
X=df[['cap-shape','cap-surface','cap-color','bruises','odor','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring','ring-type','spore-print-color','population','habitat']].values
y=df[['class']].values

pca = PCA(2)
pca.fit(X)  
#pickle.dump(pca, open('ModeloPCA.h5','wb'))
datospca = pca.transform(X)
X_trainPCA,X_testPCA,y_trainPCA,y_testPCA= train_test_split(datospca,y,random_state = 0)

y_trainPCA_d = np.array(pd.get_dummies(y_trainPCA[:, 0],columns=['class']))
y_testPCA_d = np.array(pd.get_dummies(y_testPCA[:,0],columns=['class']))

kernel = "rbf"
#kernel = "linear"
#kernel = "poly"
#kernel = "sigmoid"
modeloSVM = crearSVM(kernel,X_trainPCA,y_trainPCA[: , 0])
consufionMatrix(modeloSVM,X_trainPCA,y_trainPCA)
consufionMatrix(modeloSVM,X_testPCA,y_testPCA)

filename= 'modeloSVMPCA.h5'
pickle.dump(modeloSVM, open(filename,'wb'))

model=crearModelo(X_trainPCA,y_trainPCA_d)
consufionMatrixKNN(model,X_trainPCA,y_trainPCA)
consufionMatrixKNN(model,X_testPCA,y_testPCA)
filename= 'modeloKNNPCA.h5'
pickle.dump(model, open(filename,'wb'))