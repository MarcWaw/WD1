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

dataset = pd.read_csv('Data\ezgo.csv')
X = dataset.drop(columns=['koi_disposition'])
# .drop(
# columns=['rowid', 'kepid', 'kepoi_name', 'kepler_name', 'koi_disposition', 'koi_pdisposition', 'koi_score', 'koi_fpflag_nt',
#          'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec', 'koi_tce_delivname', 'koi_teq_err1', 'koi_teq_err2'])
# Brak w bazie: 'koi_vet_stat', 'koi_vet_date', 'koi_disp_prov', 'koi_comment'
# Dodatkowo: koi_tce_delivname (jakaś nazwa); koi_teq_err1, koi_teq_err2 (puste)

# Zamiana etykiety na wartość binarną, jak się po prostu zamieni CONFIRMED na True to wywala błąd, bo wcześniej był to
# string. Dlatego trochę więcej kodu jest
# ---------------------------------------------------------------------------------------------------------------------
y_string = dataset['koi_disposition']
y_list = []
for classification in y_string:
    if classification == 'CONFIRMED':
        y_list.append(True)
    else:
        y_list.append(False)
y = pd.DataFrame(y_list, columns=['koi_disposition'])
# ---------------------------------------------------------------------------------------------------------------------
estimators = [DecisionTreeClassifier(), SVC(), KNeighborsClassifier(), GaussianNB(),
              LogisticRegression(solver='lbfgs', max_iter=1000)]

names = ['Drzewa Decyzyjne', 'SVM', 'kNN', 'Naiwny Bayes', 'Regresja logistyczna']
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1']

# Wypełnienie pustych rekordów średnimi
# X.fillna(X.mean(), inplace=True)

# Balansowanie zbioru przy użyciu metody SMOTE - Synthetic Minority Over-sampling Technique z biblioteki imblearn
sm = SMOTE(random_state=1410)
X_resampled, y_resampled = sm.fit_resample(X, y)

# Skalowanie do normalizacji
min_max_scaler = preprocessing.MinMaxScaler()

# Krotność walidacji krzyżowej
times_cross_validation = 10
# ---------------------------------------------------------------------------------------------------------------------
# Deklaracja macierzy wyników
results_accuracy = np.zeros((times_cross_validation, len(estimators)))
results_precision = np.zeros((times_cross_validation, len(estimators)))
results_recall = np.zeros((times_cross_validation, len(estimators)))
results_f1 = np.zeros((times_cross_validation, len(estimators)))
# ---------------------------------------------------------------------------------------------------------------------
# Walidacja krzyżowa
kf = KFold(n_splits=times_cross_validation, shuffle=True, random_state=1410)
# ---------------------------------------------------------------------------------------------------------------------
iterator = 0
#                    [T    , SVM  , kNN  , NB   , LR  ]
normalization_flag = [False, False, False, False, True]
oversampling_flag = [True, True, True, True, True]
# ---------------------------------------------------------------------------------------------------------------------
# Główna pętla
for train_index, test_index in tqdm.tqdm(kf.split(X_resampled)):
    X_train, X_test = X_resampled.iloc[train_index], X_resampled.iloc[test_index]
    y_train, y_test = y_resampled.iloc[train_index], y_resampled.iloc[test_index]
    # -----------------------------------------------------------------------------------------------------------------
    for i in range(len(estimators)):
        # Czy wskazany jest oversampling?
        if oversampling_flag[i]:
            X_train, X_test = X_resampled.iloc[train_index], X_resampled.iloc[test_index]
            y_train, y_test = y_resampled.iloc[train_index], y_resampled.iloc[test_index]
        else:
            X_train, X_test = X.iloc[train_index], X_resampled.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y_resampled.iloc[test_index]
        # Czy występuje normalizacja parametrów?
        if normalization_flag[i]:
            X_train = min_max_scaler.fit_transform(X_train)
            X_test = min_max_scaler.fit_transform(X_test)
        # Iteracja przez estymatory
        clf = BaggingClassifier(base_estimator=estimators[i], n_estimators=10, random_state=0).fit(
            X_train, y_train.values.ravel())
        prediction = clf.predict(X_test)
        # Zapisywanie wyników metryk
        results_accuracy[iterator][i] = accuracy_score(y_test, prediction)
        results_precision[iterator][i] = precision_score(y_test, prediction)
        results_recall[iterator][i] = recall_score(y_test, prediction)
        results_f1[iterator][i] = f1_score(y_test, prediction)
    # -----------------------------------------------------------------------------------------------------------------
    iterator += 1
# ---------------------------------------------------------------------------------------------------------------------
# Wyświetl macierze wyników
# j = 0
# for metric in [results_accuracy, results_precision, results_recall, results_f1]:
#     print('\nMacierz metryki ' + metric_names[j])
#     print(metric)
#     j += 1
# ---------------------------------------------------------------------------------------------------------------------
# Wyświetlanie wyników
mean_accuracy = results_accuracy.mean(0)
mean_precision = results_precision.mean(0)
mean_recall = results_recall.mean(0)
mean_f1 = results_f1.mean(0)
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
# ---------------------------------------------------------------------------------------------------------------------