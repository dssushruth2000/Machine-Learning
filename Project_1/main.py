import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

#fecting from openml
def load_dataset(dataset_id):
    return fetch_openml(data_id=dataset_id, as_frame=True) 

def prepare_data(dataset):
    X = dataset.data
    y = dataset.target
    y = (y == y.unique()[0]).astype(int)
    return X, y

def perform_grid_search(X, y, criterion):
    param_grid = {'criterion': [criterion], 'min_samples_leaf': [1, 5, 10, 20, 50]}
    dt = DecisionTreeClassifier()
    grid_search = GridSearchCV(dt, param_grid, cv=10, scoring='roc_auc')
    grid_search.fit(X, y)
    return grid_search.best_estimator_

def plot_roc_curve(classifier, X, y, label):
    cv = StratifiedKFold(n_splits=10)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for train, test in cv.split(X, y):
        probas_ = classifier.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])
        fpr, tpr, thresholds = roc_curve(y.iloc[test], probas_[:, 1])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = np.mean(aucs)
    plt.plot(mean_fpr, mean_tpr, label=f'{label} (AUC = {mean_auc:.2f})', lw=2, alpha=.8)

# Loading the datasets
dataset1 = load_dataset(1130)
dataset2 = load_dataset(1134)

# Prepare the data
X1, y1 = prepare_data(dataset1)
X2, y2 = prepare_data(dataset2)


criteria = ['gini', 'entropy']

# Plot for dataset 1
plt.figure(figsize=(10, 8))
for criterion in criteria:
    best_dt1 = perform_grid_search(X1, y1, criterion)
    plot_roc_curve(best_dt1, X1, y1, label=f'{criterion.capitalize()}')
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.title('ROC for Dataset 1130')
plt.legend(loc="lower right")
plt.show()

# Plot for dataset 2
plt.figure(figsize=(10, 8))
for criterion in criteria:
    best_dt2 = perform_grid_search(X2, y2, criterion)
    plot_roc_curve(best_dt2, X2, y2, label=f'{criterion.capitalize()}')
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.title('ROC for Dataset 1134')
plt.legend(loc="lower right")
plt.show()