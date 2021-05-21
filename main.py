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


def prepare_input_data():
    data = pd.read_csv('Data\\cumulative.csv')
    data = data.drop(columns=['rowid', 'kepid', 'kepoi_name', 'kepler_name', 'koi_pdisposition', 'koi_score',
                              'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec', 'koi_tce_delivname',
                              'koi_teq_err1', 'koi_teq_err2']).copy()

    data.dropna(inplace=True)

    print(data)
    df = pd.DataFrame(data)
    df.to_csv("Data\\cumulative_prepared.csv", index=False)
    dataset = pd.read_csv('Data\\cumulative_prepared.csv')

    X_out = dataset.drop(columns=['koi_disposition'])
    # ----------------------------------------------------------------------------------------
    y_string = dataset['koi_disposition']
    y_list = []
    for classification in y_string:
        if classification == 'CONFIRMED':
            y_list.append(True)
        else:
            y_list.append(False)
    y_out = pd.DataFrame(y_list, columns=['koi_disposition'])

    return X_out, y_out


class Estimator:
    Normalize = False

    def __init__(self, estimator, normalize, name):
        self.Normalize = normalize
        self.Estimator = estimator
        self.Name = name


def make_predictions(estimators):
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1']

    scores_array = []
    for i in range(len(metric_names)):
        scores_array.append(np.zeros((times_cross_validation, len(estimators))))

    # Balansowanie zbioru przy użyciu metody SMOTE - Synthetic Minority Over-sampling Technique z biblioteki imblearn
    sm = SMOTE(random_state=1410)
    X_resampled, y_resampled = sm.fit_resample(X, y)

    # Skalowanie do normalizacji
    min_max_scaler = preprocessing.MinMaxScaler()

    # Walidacja krzyżowa
    kf = KFold(n_splits=times_cross_validation, shuffle=True, random_state=1410)

    # -----------------------------------------------------------------------------------------------------------------
    # Główna pętla
    iterator = 0

    for train_index, test_index in tqdm.tqdm(kf.split(X_resampled)):
        # -------------------------------------------------------------------------------------------------------------
        # Tablice do krzywych ROC
        fpr_array = []
        tpr_array = []
        estimator_index = 0
        for esti in estimators:
            X_train, X_test = X_resampled.iloc[train_index], X_resampled.iloc[test_index]
            y_train, y_test = y_resampled.iloc[train_index], y_resampled.iloc[test_index]
            # Czy występuje normalizacja parametrów?
            if esti.Normalize:
                X_train = min_max_scaler.fit_transform(X_train)
                X_test = min_max_scaler.fit_transform(X_test)
            # Iteracja przez estymatory
            clf = BaggingClassifier(base_estimator=esti.Estimator, n_estimators=10, random_state=0).fit(
                X_train, y_train.values.ravel())
            prediction = clf.predict(X_test)
            prediction_roc = clf.predict_proba(X_test)  # Predict do ROC
            # Zapisywanie wyników metryk
            scores_array[0][iterator][estimator_index] = accuracy_score(y_test, prediction)
            scores_array[1][iterator][estimator_index] = precision_score(y_test, prediction)
            scores_array[2][iterator][estimator_index] = recall_score(y_test, prediction)
            scores_array[3][iterator][estimator_index] = f1_score(y_test, prediction)
            # ROC
            fpr, tpr, treshold = roc_curve(y_test, prediction_roc[:, 1])
            fpr_array.append(fpr)
            tpr_array.append(tpr)

            estimator_index += 1
        # -------------------------------------------------------------------------------------------------------------
        # Stworzenie plotów ROC
        display.plot_roc_curve(fpr_array, tpr_array, names)
        plt.savefig(f'ROC_plots/ROC_fold{iterator + 1}.png')
        plt.clf()

        iterator += 1
    return scores_array


# ------------------==========================  HERE PROGRAM STARTS  ==========================------------------
X, y = prepare_input_data()

estimators = [Estimator(DecisionTreeClassifier(), False, 'Drzewa Decyzyjne'),
              Estimator(SVC(), False, 'SVM'),
              Estimator(KNeighborsClassifier(), False, 'kNN'),
              Estimator(GaussianNB(), False, 'Naiwny Bayes'),
              Estimator(LogisticRegression(solver='lbfgs', max_iter=1000), True, 'Regresja logistyczna')]

names = []
for est in estimators:
    names.append(est.Name)

times_cross_validation = 2

scores = make_predictions(estimators)
scores_accuracy = scores[0]
scores_accuracy = np.transpose(scores_accuracy)

display.show_results(estimators, scores)
