# Data Preprocessing
import numpy as np
import pandas as pd

#Reading the Data
data = pd.read_csv("D:/Deep Learning/Volume 1 - Supervised Deep Learning/Part 1 - Artificial Neural Networks (ANN)/Section 4 - Building an ANN/Churn_Modelling.csv")
X = data.iloc[:, 3:13].values
Y = data.iloc[:, 13].values

# Encoding Categorical Variable 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_1.fit_transform(X[:, 2])
columntransformer = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder= 'passthrough')
X = columntransformer.fit_transform(X)
X = X.astype(float)
X = X[:, 1:]


#Splitting the data 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test =  train_test_split(X, Y, test_size=0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Building ANN

# Importing Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
''''
# Intitialising the ANN
classifier = Sequential()

# Adding the input layer and the first Hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation = 'relu', input_dim = 11))
classifier.add(Dropout(p=0.1))
#Adding second hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation = 'relu'))
classifier.add(Dropout(p=0.1))

#Adding output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation = 'sigmoid'))

#compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )

# fitting the ANN to Training set
classifier.fit(X_train, Y_train, batch_size= 10, epochs = 100)

# Making Prediction and evaluating the model
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)

#Predicting a new single observation 
Predict if the customer with the following informations will leave the bank:'
Geography : France 
Credit Score : 600
Gender : Male  
Age: 40
Tenure : 3
Balance : 60000
Number of Products : 2
Has Credit Card : Yes 
Is Active Member : Yes
Estimated Salary : 50000


new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0.0, 600.0, 1.0, 40.0, 3.0, 60000.0, 2.0, 1.0, 1.0]])))



#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
'''

### Evaluating, Improving and Tuning the ANN

## Evaluating 
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation = 'relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv= 10, n_jobs= -1)
mean = accuracies.mean()
variance = accuracies.std()


### Improving the ANN
## Dropout Regularization to reduce overfitting if needed 
#Added that in above comment section p is the percent of neuron you want to disable to reduce overfitting

###Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.layers import Dense
from keras.layers import Dropout
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation = 'relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'] )
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size':[10, 25, 32],
              'epochs': [100, 75, 500],
              'optimizer' : ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
gridsearch = grid_search.fit(X_train, Y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

