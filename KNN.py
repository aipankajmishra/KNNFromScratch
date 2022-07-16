from heapq import heappush
import heapq
from unicodedata import name
import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


TEST_SIZE = 0.3

class KNN:

    def __init__(self, neighbors = 5):
        self.neighbors = neighbors

        
    def fit(self,X_train,y_train):
        self.X_train = X 
        self.y_train = y

    
    def _predict(self,xq):
        distances = []
        for x in self.X_train: 
            euc_distance = KNN.calculate_distance(xq,x)
            distances.append(euc_distance)
        indices = np.argsort(distances)[:self.neighbors]
        return np.argmax(np.bincount(self.y_train[indices]))
        

    def predict(self, X_test):
        return [self._predict(x) for x in X_test]

    @staticmethod
    def calculate_distance(x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))
    

def load_dataset():
    iris = load_iris()
    X,y  = iris.data, iris.target
    return X,y


# TODO: Validate the results...

if __name__ == "__main__":
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= TEST_SIZE)

    clf = KNN(neighbors=3)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)   
    accuracy_score = np.mean(y_test == predictions) 
    print(f"KNN from scratch gives accuracy score of {accuracy_score}")


    knn_clf = KNeighborsClassifier(n_neighbors= 7)
    knn_clf.fit(X_train, y_train)
    predictions_sk = knn_clf.predict(X_test)
    accuracy_score_sk = np.mean(y_test == predictions_sk)
    print(f"KNN model accuracy from sklearn classifier {accuracy_score_sk}")

