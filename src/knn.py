from sklearn.neighbors import *
from sklearn.preprocessing import *
from sklearn.metrics import *
import numpy as np


class KNNModel():
    
    def __init__(self, X_train, Y_train, X_test, Y_test):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        
    def model(self):
        x_train = np.zeros(self.X_train.shape)
        np.copyto(x_train, self.X_train)

        y_train = np.zeros(self.Y_train.shape)
        np.copyto(y_train, self.Y_train)

        x_test = np.zeros(self.X_test.shape)
        np.copyto(x_test, self.X_test)

        y_test = np.zeros(self.Y_test.shape)
        np.copyto(y_test, self.Y_test)
        
        knnModel = KNeighborsClassifier()
        knnModel.fit(x_train, y_train)
        y_pred = knnModel.predict(x_test)
        
        print("Confusion Matrix is as follows:")
        knn_cm = confusion_matrix(y_test, y_pred)
        print(knn_cm)
        print("\nThe accuracy is:")
        knn_acc = accuracy_score(y_test, y_pred) * 100
        print(knn_acc)
        
        p = precision_score(y_test, y_pred)
        
        r = recall_score(y_test, y_pred)
        
        print("Precision is :", p)
        print("Recall is:", r)