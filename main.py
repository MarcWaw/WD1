import pandas as pd
import tqdm
import plotly.graph_objects as go
import plotly.offline as pyo
import numpy as np
import matplotlib.pyplot as plt
import Statistic
import display

from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve
from tabulate import tabulate

# ------------------==========================  HERE PROGRAM STARTS  ==========================------------------
# Odczyt danych
dataset = pd.read_csv('Data\\cumulative_prepared.csv')
X = dataset.drop(columns=['koi_disposition'])
y_string = dataset['koi_disposition']
# Zamiana etykiety na wartość binarną
y_list = []
for classification in y_string:
    if classification == 'CONFIRMED':
        y_list.append(True)
    else:
        y_list.append(False)
y = pd.DataFrame(y_list, columns=['koi_disposition'])

clfs_names = ['Drzewa Decyzyjne', 'SVM', 'kNN', 'Naiwny Bayes', 'Regresja logistyczna']
clfs = [BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=10, random_state=0), BaggingClassifier(base_estimator=SVC(), n_estimators=10, random_state=0),
        BaggingClassifier(base_estimator=KNeighborsClassifier(), n_estimators=10, random_state=0), BaggingClassifier(base_estimator=GaussianNB(), n_estimators=10, random_state=0),
        BaggingClassifier(base_estimator=LogisticRegression(solver='lbfgs', max_iter=1000), n_estimators=10, random_state=0)]




times_cross_validation = 2
scores_names = ['Accuracy', 'Precision', 'Recall', 'F1']
scores = np.zeros((len(scores_names), len(clfs), times_cross_validation))

# Balansowanie zbioru przy użyciu metody SMOTE - Synthetic Minority Over-sampling Technique z biblioteki imblearn
sm = SMOTE(random_state=1410)
X_resampled, y_resampled = sm.fit_resample(X, y)

# Skalowanie do normalizacji
min_max_scaler = preprocessing.MinMaxScaler()

# Walidacja krzyżowa
kf = KFold(n_splits=times_cross_validation, shuffle=True, random_state=1410)

# -----------------------------------------------------------------------------------------------------------------
# Główna pętla
for i, (train_index, test_index) in tqdm.tqdm(enumerate(kf.split(X_resampled))):
    # -------------------------------------------------------------------------------------------------------------
    # Tablice do krzywych ROC
    fpr_array = []
    tpr_array = []
    for estimator_index, esti in enumerate(clfs):
        X_train, X_test = X_resampled.iloc[train_index], X_resampled.iloc[test_index]
        y_train, y_test = y_resampled.iloc[train_index], y_resampled.iloc[test_index]
        # Czy występuje normalizacja parametrów?
        if clfs_names[estimator_index] == 'Regresja logistyczna':
            X_train = min_max_scaler.fit_transform(X_train)
            X_test = min_max_scaler.fit_transform(X_test)
        # Iteracja przez estymatory
        clf = esti.fit(X_train, y_train.values.ravel())
        prediction = clf.predict(X_test)
        prediction_roc = clf.predict_proba(X_test)  # Predict do ROC

        # Zapisywanie wyników metryk
        scores[:, estimator_index, i] = [accuracy_score(y_test, prediction), precision_score(y_test, prediction),
                                               recall_score(y_test, prediction), f1_score(y_test, prediction)]

        # ROC
        fpr, tpr, treshold = roc_curve(y_test, prediction_roc[:, 1])
        fpr_array.append(fpr)
        tpr_array.append(tpr)
    # -------------------------------------------------------------------------------------------------------------
    # Stworzenie plotów ROC
    display.plot_roc_curve(fpr_array, tpr_array, clfs_names)
    plt.savefig(f'ROC_plots/ROC_fold{i + 1}.png')
    plt.clf()




display.show_results(clfs, scores)
