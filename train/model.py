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
    df_shuffle = pd.read_csv("./df_shuffle_1000.csv")
    features = df_shuffle.drop(" Label", axis=1).values
    labels = df_shuffle[" Label"].values
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)
    
    features = X_train.shape[1]
    nClasses = len(df_shuffle[' Label'].unique())

    classifier = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        leaf_size=leaf_size,
        metric=metric,
        p=p,
        weights=weights,
    )
    classifier.fit(X_train, y_train)
    
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
