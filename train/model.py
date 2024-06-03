#Importacion de librerias
import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn import preprocessing
import matplotlib.pyplot as plt
import sklearn
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from tensorflow.keras.utils import to_categorical, plot_model
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,GRU,Embedding,Dropout,Flatten,Conv1D,MaxPooling1D,LSTM
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import joblib
import seaborn as sns

from sklearn.datasets import load_iris

def train_and_eval(
    #num_epochs = 50,
    test_size=0.2,
    random_state=33,
    model_path=None,
):
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    features = X_train.shape[1]
    nClasses = len(df[' Label'].unique())

    model_dnn_1 = Sequential()
    model_dnn_1.add(Dense(64, input_dim=features, activation='relu')) #Capa de entrada
    model_dnn_1.add(Dense(128, activation='relu'))                    #Capa densa 1
    model_dnn_1.add(Dropout(0.2))                                     #Capa Dropout 1
    model_dnn_1.add(Dense(256, activation='relu'))                    #Capa densa 2
    model_dnn_1.add(Dense(256, activation='relu'))                    #Capa densa 3
    model_dnn_1.add(Dense(256, activation='relu'))                    #Capa densa 4
    model_dnn_1.add(Dropout(0.2))                                     #Capa Dropout 2
    model_dnn_1.add(Dense(128, activation='relu'))                    #Capa densa 5
    model_dnn_1.add(Dense(128, activation='relu'))                    #Capa densa 6
    model_dnn_1.add(Dense(64, activation='relu'))                     #Capa densa 7
    model_dnn_1.add(Dense(nClasses, activation='softmax'))                  #Capa de salida

    model_dnn_1.compile(optimizer='adam', loss= 'sparse_categorical_crossentropy', metrics= ['accuracy'])
    model_dnn_1.fit(X_train, y_train, validation_data=(X_test, y_test))
    
    y_pred = model_dnn_1.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred, average='weighted')
    f1 = metrics.f1_score(y_pred, y_pred, average='weighted')
    results = {
        'accuracy': accuracy,
        'recall': recall,
        'f1': f1,
    }
    if model_path:
        joblib.dump(model_dnn_1, model_path)
    return results
