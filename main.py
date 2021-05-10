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
y_string = dataset['koi_disposition']
y_list = []
for classification in y_string:
    if classification == 'CONFIRMED':
        y_list.append(True)
    else:
        y_list.append(False)
y = pd.DataFrame(y_list, columns=['koi_disposition'])

# LogisticRegression() - wywala błędy póki co, może trzeba standaryzacje zrobić
estimators = [DecisionTreeClassifier(), SVC(), KNeighborsClassifier(), GaussianNB(),
              LogisticRegression(solver='lbfgs', max_iter=100000)]
names = ['Drzewa Decyzyjne', 'SVM', 'kNN', 'Naiwny Bayes', 'Regresja logistyczna']

result_accuracy = [0 for i in range(len(estimators))]
result_precision = [0 for i in range(len(estimators))]
result_recall = [0 for i in range(len(estimators))]
result_f1 = [0 for i in range(len(estimators))]

# Wypełnienie pustych rekordów średnimi
# X.fillna(X.mean(), inplace=True)

# Balansowanie zbioru przy użyciu metody SMOTE - Synthetic Minority Over-sampling Technique z biblioteki imblearn
sm = SMOTE(random_state=1410)
X_resampled, y_resampled = sm.fit_resample(X, y)
min_max_scaler = preprocessing.MinMaxScaler()

# Krotność walidacji krzyżowej
times_cross_validation = 10

# Walidacja krzyżowa
kf = KFold(n_splits=times_cross_validation, shuffle=True, random_state=1410)

for train_index, test_index in tqdm.tqdm(kf.split(X_resampled)):
    X_train, X_test = X_resampled.iloc[train_index], X_resampled.iloc[test_index]
    y_train, y_test = y_resampled.iloc[train_index], y_resampled.iloc[test_index]

    # Iteracja po algorytmach uczących
    for i in range(len(estimators) - 1):
        clf = BaggingClassifier(base_estimator=estimators[i], n_estimators=10, random_state=0).fit(
            X_train, y_train.values.ravel())
        prediction = clf.predict(X_test)
        result_accuracy[i] += accuracy_score(y_test, prediction)
        result_precision[i] += precision_score(y_test, prediction)
        result_recall[i] += recall_score(y_test, prediction)
        result_f1[i] += f1_score(y_test, prediction)

    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.fit_transform(X_test)
    clf = BaggingClassifier(base_estimator=estimators[-1], n_estimators=10, random_state=0).fit(
        X_train, y_train.values.ravel())
    prediction = clf.predict(X_test)
    result_accuracy[-1] += accuracy_score(y_test, prediction)
    result_precision[-1] += precision_score(y_test, prediction)
    result_recall[-1] += recall_score(y_test, prediction)
    result_f1[-1] += f1_score(y_test, prediction)

# Wyświetlanie wyników
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1']

j = 0
for metric in [result_accuracy, result_precision, result_recall, result_f1]:
    print('\nMetryka: ' + metric_names[j])
    for i in range(len(estimators)):
        print(f'{names[i]} {metric[i] / times_cross_validation}')
    j += 1

# Wyświetlanie wykresu
for i in range(len(estimators)):
    result_accuracy[i] = result_accuracy[i] / times_cross_validation
    result_precision[i] = result_precision[i] / times_cross_validation
    result_recall[i] = result_recall[i] / times_cross_validation
    result_f1[i] = result_f1[i] / times_cross_validation

plot_accuracy = [*result_accuracy, result_accuracy[0]]
plot_precision = [*result_precision, result_precision[0]]
plot_recall = [*result_recall, result_recall[0]]
plot_f1 = [*result_f1, result_f1[0]]
names = [*names, names[0]]

fig = go.Figure(data=[go.Scatterpolar(r=plot_accuracy, theta=names, name='Accuracy'),
                      go.Scatterpolar(r=plot_precision, theta=names, name='Precision'),
                      go.Scatterpolar(r=plot_recall, theta=names, name='Recall'),
                      go.Scatterpolar(r=plot_f1, theta=names, name='F1')],
                layout=go.Layout(title=go.layout.Title(text='Metrics comparison'),
                                 polar={'radialaxis': {'visible': True}},
                                 showlegend=True
                                 )
                )

pyo.plot(fig)
