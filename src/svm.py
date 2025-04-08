from sklearn.svm import *
from sklearn.preprocessing import *
from sklearn.metrics import *
import numpy as np

class SVM():
    
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


        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        print(x_test.shape)
        x_test = sc.transform(x_test)

        svmModel = SVC(kernel='rbf')
        svmModel.fit(x_train, y_train)
        y_pred = svmModel.predict(x_test)

        print("Confusion Matrix is as follows:")
        svm_cm = confusion_matrix(y_test, y_pred)
        print(svm_cm)
        print("\nThe accuracy is:")
        svm_acc = accuracy_score(y_test, y_pred) * 100
        print(svm_acc)
        # accuracy.append(svm_acc)
        p = precision_score(y_test, y_pred)
        # precision.append(p)
        r = recall_score(y_test, y_pred)
        # recall.append(r)

        print("Precision is :", p)
        print("Recall is:", r)




        
        