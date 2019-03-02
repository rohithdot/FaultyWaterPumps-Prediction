from __future__ import print_function

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier

dataset = pd.read_csv("dataset_processed.csv") 

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,22].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

tuned_Bag_parameters = [{'n_estimators' : [10,5], 'max_features' : [0.5,0.3,1.0], 'max_samples': [1.0,0.3,0.6], 'random_state' : [1] }]
scores = ['accuracy']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(BaggingClassifier(), tuned_Bag_parameters, cv=5,scoring='%s' % score)
      
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Detailed classification report:")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
  
    print("Accuracy Score:")
    print(accuracy_score(y_true, y_pred))

    print()

