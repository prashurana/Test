#importing libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, precision_score, f1_score



def wrapper():
    X,Y = data_preprocessing()
    X = encoding_categorical_variable(X)
    X_train, X_test, Y_train, Y_test = split_feature_scale_data(X, Y)
    classifier = build_classifier(X_train, Y_train)
    #score = print_score(classifier, X_train, Y_train, X_test, Y_test, train=True )
    score = print_score(classifier, X_train, Y_train, X_test, Y_test, train=False)
    return score 

def data_preprocessing():
    data = pd.read_csv("Churn_Modelling.csv")
    X = data.iloc[:, 3:13].values
    Y = data.iloc[:, 13].values
    return X, Y


def encoding_categorical_variable(X):
    labelencoder_X_1 = LabelEncoder()
    X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
    labelencoder_X_2 = LabelEncoder()
    X[:, 2] = labelencoder_X_1.fit_transform(X[:, 2])
    columntransformer = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder= 'passthrough')
    X = columntransformer.fit_transform(X)
    X = X.astype(float)
    X = X[:, 1:]
    return X

def split_feature_scale_data(X, Y):
    X_train, X_test, Y_train, Y_test =  train_test_split(X, Y, test_size=0.2, random_state = 0)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)
    return X_train, X_test, Y_train, Y_test

def build_classifier(X_train, Y_train):
    classifier = Sequential()
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation = 'relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )
    classifier.fit(X_train, Y_train, batch_size= 32, epochs = 500)
    return classifier

def print_score(clf, X_train, Y_train, X_test, Y_test, train=True):
    '''
    print the accuracy score, classification report and confusion matrix of classifier
    '''
    if train:
        '''
        training performance
        '''
        print("Train Result:\n")
        print("accuracy score: {0:.4f}\n".format(accuracy_score(Y_train, clf.predict(X_train).round())))
        print("Classification Report: \n {}\n".format(classification_report(Y_train, clf.predict(X_train).round())))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(Y_train, clf.predict(X_train).round())))

        classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
        accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv= 10, n_jobs= -1)
        print("Average Accuracy: \t {0:.4f}".format(np.mean(accuracies)))
        print("Accuracy SD: \t\t {0:.4f}".format(np.std(accuracies)))
        
    elif train==False:
        '''
        test performance
        '''
        print("Test Result:\n")        
        print("accuracy score: {0:.4f}\n".format(accuracy_score(Y_test, clf.predict(X_test).round())))
        print("Classification Report: \n {}\n".format(classification_report(Y_test, clf.predict(X_test).round())))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(Y_test, clf.predict(X_test).round()))) 

if __name__ == "__main__":
    score = wrapper()