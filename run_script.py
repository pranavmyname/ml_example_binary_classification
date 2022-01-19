import numpy as np
import matplotlib.pylab as plt
plt.rcParams['axes.grid'] = True
import scipy.io
from scipy.fft import fft, fftfreq
from scipy.stats import entropy, skew
import pandas as pd
pd.options.display.max_columns = None
import math 
from scipy import signal
import json
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix

train_df = pd.read_csv("data/training_set.csv")
train_df.drop(columns=train_df.columns[0], 
        axis=1, 
        inplace=True)

test_df = pd.read_csv("data/test_set.csv")

test_df.drop(columns=test_df.columns[0], 
        axis=1, 
        inplace=True)


scorers = {
    'precision_score': make_scorer(precision_score),
    'recall_score': make_scorer(recall_score),
    'accuracy_score': make_scorer(accuracy_score)
}


def grid_search_fun(clf, parameters, scorer, X_train, y_train):
    estimators = GridSearchCV(clf, parameters, scoring = scorers[scorer], cv = 5)
    estimators.fit(X_train,y_train )
    clf_best = estimators.best_estimator_
    
    
    return clf_best, estimators.best_params_

clf = LogisticRegression(random_state=0)
parameters = {'C': [0.1, 0.5, 1]}

train_df_norm = (train_df-train_df.mean())/train_df.std()
train_df_norm.head()
y_train = train_df.Y.values
X_train = train_df_norm.drop(["Y"], axis =1)

test_df_norm = (test_df-test_df.mean())/test_df.std()
X_test = test_df_norm.values

clf, best_param = grid_search_fun(clf, parameters, 'accuracy_score', X_train, y_train)

y_pred = clf.predict(X_test)
print(y_pred)