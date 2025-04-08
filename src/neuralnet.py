from sklearn.preprocessing import *
from sklearn.metrics import *
import numpy as np

from keras.models import *
from keras.layers import *
from keras.optimizers import *

class NeuralNet():
    
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
        x_test = sc.transform(x_test)
        
        
        nnModel = Sequential()
        nnModel.add(Dense(16, input_dim=9, activation='relu'))
        nnModel.add(Dense(16, activation='relu'))
        nnModel.add(Dense(8, activation='relu'))
        nnModel.add(Dense(1, activation='sigmoid'))
        
        adam = Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999)
        nnModel.compile(optimizer=adam, loss='binary_crossentropy', metrics=["accuracy"])
        nnModel.fit(x_train, y_train, epochs=200, batch_size=100)
        
        y_p = nnModel.predict(x_test)
        y_p = y_p[:, 0]
        y_pred = []
        for z in y_p:
          if z>=0.5:
            y_pred.append(1)
          else:
            y_pred.append(0)
        
        
        print("Confusion Matrix is as follows:")
        nn_cm = confusion_matrix(y_test, y_pred)
        print(nn_cm)
        print("\nThe accuracy is:")
        nn_acc = accuracy_score(y_test, y_pred) * 100
        print(nn_acc)
        
        p = precision_score(y_test, y_pred)
        
        r = recall_score(y_test, y_pred)
        
        print("Precision is :", p)
        print("Recall is:", r)
        
        
        
        