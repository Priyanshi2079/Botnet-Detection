import pickle
from svm import SVM
from logreg import LogReg
from dectree import DecTree
from randomforest import RanFor
from naivebayes import NBModel
from knn import KNNModel
from neuralnet import NeuralNet

dataset_file = open('flowdata.pickle', 'rb')
dataset = pickle.load(dataset_file, encoding='bytes')

X_train, Y_train, X_test, Y_test = dataset[0], dataset[1], dataset[2], dataset[3]

print("The training data contains:", X_train.shape[0], "records and", X_train.shape[1], "columns")
print("The testing data contains:", X_test.shape[0], "records and", X_test.shape[1], "columns")

def call_algo(algo):

    global X_train, X_test, Y_train, Y_test
    
    if algo == 'SVM':
        model = SVM(X_train, Y_train, X_test, Y_test)
        model.model()
        
    if algo == 'LogReg':
        model = LogReg(X_train, Y_train, X_test, Y_test)
        model.model()
        
    if algo == "DecTree":
        model = DecTree(X_train, Y_train, X_test, Y_test)
        model.model()
        
    if algo == "RanFor":
        model = RanFor(X_train, Y_train, X_test, Y_test)
        model.model()
        
    if algo == "NB":
        model = NBModel(X_train, Y_train, X_test, Y_test)
        model.model()
    
    if algo == "KNN":
        model = KNNModel(X_train, Y_train, X_test, Y_test)
        model.model()
        
    if algo == "NeuralNet":
        model = NeuralNet(X_train, Y_train, X_test, Y_test)
        model.model()


#call_algo('NeuralNet')