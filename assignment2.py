# This section is devoted to the importation of the needed libraries


import numpy as np
import matplotlib.pyplot as pl
# %matplotlib inline
import pandas as pd
import sklearn
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import learning_curve

try:
    from sklearn.model_selection import GridSearchCV
except ImportError:
    from sklearn.grid_search import GridSearchCV
try: 
    from sklearn.model_selection import KFold, cross_val_score
    legacy = False 
except ImportError:
    from sklearn.cross_validation import KFold, cross_val_score
    legacy = True

# Now we need to load the data we are going to deal with, namely those 
# refering to the Spambase dataset
# Loading data and storing them as training/ test data and training/test targets
test_data = np.loadtxt("test-data.csv",delimiter=",")
train_data = np.loadtxt("train-data.csv",delimiter=",")
test_targets = np.loadtxt("test-targets.csv",delimiter=",")
train_targets = np.loadtxt("train-targets.csv",delimiter=",")
# Here some manipulations are needed in order to deal with suitable arrays
test_targets = test_targets.ravel()
test_targets = np.array(test_targets).astype(int)
train_targets = train_targets.ravel()
train_targets = np.array(train_targets).astype(int)   
 
# The classifier we wish to adopt is the SVC with "rbf" kernel performing the GridSearchCV()
# Hence we declare some possible parameters in order to go through a preliminar model selection
possible_parameters = {
    'C': [ 1e+1, 1e+2, 1e+3, 1e+4, 1e+5],
    'gamma': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
}
# Then we train our model
clf = GridSearchCV(SVC(kernel='rbf'),possible_parameters, n_jobs=3, cv=3)
clf.fit(train_data, train_targets)
# Eventually we can take a look to the optimal parameters
optimal_parameters = clf.best_estimator_
optimal_parameters


# Now we wish to compute and take a look at the average accuracy, precision, recall, 
# and F1 over the test set, thus
prediction_targets = clf.predict(test_data)
accuracy = metrics.accuracy_score(test_targets,prediction_targets)
precision = metrics.precision_score(test_targets,prediction_targets)
recall = metrics.recall_score(test_targets, prediction_targets)
f1 = metrics.f1_score(test_targets, prediction_targets)

accuracy, precision, recall, f1
# We are also required to store the predictions we obtained with our classifier in a txt
# file
np.savetxt("predictions_spambase.txt",prediction_targets,fmt='%i')

# Our next goal is to compute and take a look at the average accuracy, precision, recall, 
# and F1 over the the cross validation folds, and eventually plot the learning curve
# First of all let say that we want to use again a SVC with "rbf" kernel classifier, where
# the parameters are the optimal ones we got before
best_C = 100000
best_gamma = 1e-5

# Our classifier is
clf = SVC(C=best_C, kernel='rbf', gamma=best_gamma)
# and we want to train it using a 5 cross validation strategy, so let  define
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# For the accuracy we will provide the plot of the learning curve:
train_sizes, train_scores, val_scores = learning_curve(clf, train_data, train_targets, scoring='accuracy', cv=5,random_state=42)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)
val_scores_std = np.std(val_scores, axis=1)
# Plotting

pl.figure()
pl.title("Learning curve where C = 100000 and gamma = 1e-5")
pl.xlabel("Training Examples")
pl.ylabel("Score/Accuracy")
pl.grid()


# Plot the mean and std for the training scores
pl.plot(train_sizes, train_scores_mean, 'o-', color="c", label="Training score")
pl.fill_between(train_sizes, train_scores_mean - train_scores_std,
                train_scores_mean + train_scores_std, alpha=0.1, color="c")

# Plot the mean and std for the validation scores
pl.plot(train_sizes, val_scores_mean, 'o-', color="g", label="Cross-Val score")
pl.fill_between(train_sizes, val_scores_mean - val_scores_std,
                 val_scores_mean + val_scores_std, alpha=0.1, color="g")

pl.ylim(0.7,1.1)
pl.legend()
pl.show()

# Now we finish providing the average precision, recall, 
# and F1 over the the cross validation folds
Prec_scores = cross_val_score(clf, train_data, train_targets, cv=kf.split(train_data), scoring='precision')
Rec_scores = cross_val_score(clf, train_data, train_targets, cv=kf.split(train_data), scoring='recall')
f1_scores = cross_val_score(clf,train_data, train_targets, cv=kf.split(train_data), scoring='f1')

Prec_scores

Rec_scores

f1_scores

