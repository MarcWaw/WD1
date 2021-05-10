import pandas as pd
import tqdm

import plotly.graph_objects as go
import plotly.offline as pyo
import numpy as np
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


def prepare_input_data(remove_incomplete_rows=True):
    data = pd.read_csv('Data\cumulative.csv')
    data = data.drop(
        columns=['rowid', 'kepid', 'kepoi_name', 'kepler_name', 'koi_pdisposition', 'koi_score', 'koi_fpflag_nt',
                 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec', 'koi_tce_delivname', 'koi_teq_err1',
                 'koi_teq_err2']).copy()

    if remove_incomplete_rows is True:
        data.dropna(inplace=True)

    print(data)
    df = pd.DataFrame(data)
    df.to_csv("Data\cumulative_prepared.csv", index=False)
    dataset = pd.read_csv('Data\cumulative_prepared.csv')

    X = dataset.drop(columns=['koi_disposition'])
# ----------------------------------------------------------------------------------------
    y_string = dataset['koi_disposition']
    y_list = []
    for classification in y_string:
        if classification == 'CONFIRMED':
            y_list.append(True)
        else:
            y_list.append(False)
    y = pd.DataFrame(y_list, columns=['koi_disposition'])

    # Wypełnienie pustych rekordów średnimi
    if remove_incomplete_rows is False:
        X.fillna(X.mean(), inplace=True)

    return X, y


class estimator():

    Normalize = False
    Oversample = True

    def __init__(self, estimator, normalize, oversample, name):
        self.Normalize = normalize
        self.Oversample = oversample
        self.Estimator = estimator
        self.Name = name


def make_predictions(new_estimators, times_cross_validation, precision_scores):

    # Balansowanie zbioru przy użyciu metody SMOTE - Synthetic Minority Over-sampling Technique z biblioteki imblearn
    sm = SMOTE(random_state=1410)
    X_resampled, y_resampled = sm.fit_resample(X, y)

    # Skalowanie do normalizacji
    min_max_scaler = preprocessing.MinMaxScaler()

    # Walidacja krzyżowa
    kf = KFold(n_splits=times_cross_validation, shuffle=True, random_state=1410)

    # ---------------------------------------------------------------------------------------------------------------------
    # Główna pętla
    iterator = 0
    for train_index, test_index in tqdm.tqdm(kf.split(X_resampled)):
        X_train, X_test = X_resampled.iloc[train_index], X_resampled.iloc[test_index]
        y_train, y_test = y_resampled.iloc[train_index], y_resampled.iloc[test_index]
        # -----------------------------------------------------------------------------------------------------------------

        estimator_index = 0
        for est in new_estimators:
            if est.Oversample:
                X_train, X_test = X_resampled.iloc[train_index], X_resampled.iloc[test_index]
                y_train, y_test = y_resampled.iloc[train_index], y_resampled.iloc[test_index]
            else:
                X_train, X_test = X.iloc[train_index], X_resampled.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y_resampled.iloc[test_index]
            # Czy występuje normalizacja parametrów?
            if est.Normalize:
                X_train = min_max_scaler.fit_transform(X_train)
                X_test = min_max_scaler.fit_transform(X_test)
            # Iteracja przez estymatory
            clf = BaggingClassifier(base_estimator=est.Estimator, n_estimators=10, random_state=0).fit(
                X_train, y_train.values.ravel())
            prediction = clf.predict(X_test)
            # Zapisywanie wyników metryk
            precision_scores[0][iterator][estimator_index] = accuracy_score(y_test, prediction)
            precision_scores[1][iterator][estimator_index] = precision_score(y_test, prediction)
            precision_scores[2][iterator][estimator_index] = recall_score(y_test, prediction)
            precision_scores[3][iterator][estimator_index] = f1_score(y_test, prediction)
            estimator_index += 1
        # -----------------------------------------------------------------------------------------------------------------
        iterator += 1
    return precision_scores


def show_results():
    # Wyświetlanie wyników
    mean_accuracy = precision_scores[0].mean(0)
    mean_precision = precision_scores[1].mean(0)
    mean_recall = precision_scores[2].mean(0)
    mean_f1 = precision_scores[3].mean(0)
    j = 0
    for metric in [mean_accuracy, mean_precision, mean_recall, mean_f1]:
        print('\nMetryka: ' + metric_names[j])
        for i in range(len(estimators)):
            print(names[i] + ": " + str(format(metric[i] * 100, '.1f')))
        j += 1
    # ---------------------------------------------------------------------------------------------------------------------
    # Wyświetlanie wykresu
    plot_tree_results = [mean_accuracy[0], mean_precision[0], mean_recall[0], mean_f1[0], mean_accuracy[0]]
    plot_svm = [mean_accuracy[1], mean_precision[1], mean_recall[1], mean_f1[1], mean_accuracy[1]]
    plot_knn = [mean_accuracy[2], mean_precision[2], mean_recall[2], mean_f1[2], mean_accuracy[2]]
    plot_nb = [mean_accuracy[3], mean_precision[3], mean_recall[3], mean_f1[3], mean_accuracy[3]]
    plot_lr = [mean_accuracy[4], mean_precision[4], mean_recall[4], mean_f1[4], mean_accuracy[4]]
    plot_names = [*metric_names, metric_names[0]]
    fig = go.Figure(data=[go.Scatterpolar(r=plot_tree_results, theta=plot_names, name='Drzewo decyzyjne'),
                          go.Scatterpolar(r=plot_svm, theta=plot_names, name='SVM'),
                          go.Scatterpolar(r=plot_knn, theta=plot_names, name='kNN'),
                          go.Scatterpolar(r=plot_nb, theta=plot_names, name='Naiwny Bayes'),
                          go.Scatterpolar(r=plot_lr, theta=plot_names, name='Regresja Logistyczna')],
                    layout=go.Layout(title=go.layout.Title(text='Wyniki'),
                                     polar={'radialaxis': {'visible': True}},
                                     showlegend=True
                                     )
                    )
    pyo.plot(fig)


# HERE PROGRAM STARTS

X, y = prepare_input_data()

estimators = [estimator(DecisionTreeClassifier(), False, True, 'Drzewa Decyzyjne'),
              estimator(SVC(), False, True, 'SVM'),
              estimator(KNeighborsClassifier(), False, True, 'kNN'),
              estimator(GaussianNB(), False, True, 'Naiwny Bayes'),
              estimator(LogisticRegression(solver='lbfgs', max_iter=1000), True, True, 'Regresja logistyczna')]

metric_names = ['Accuracy', 'Precision', 'Recall', 'F1']
times_cross_validation = 10

names = []
for est in estimators:
    names.append(est.Name)

precision_scores = []
for i in range(len(metric_names)):
    precision_scores.append(np.zeros((times_cross_validation, len(estimators))))

precision_scores = make_predictions(estimators, times_cross_validation, precision_scores).copy()

show_results(precision_scores, names)
