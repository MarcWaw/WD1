import pandas as pd
import tqdm
import matplotlib.pyplot as plt
import RadarChart
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

data = pd.read_csv('Data\ezgo.csv')
X = data.drop(columns=['koi_disposition'])
# .drop(
# columns=['rowid', 'kepid', 'kepoi_name', 'kepler_name', 'koi_disposition', 'koi_pdisposition', 'koi_score', 'koi_fpflag_nt',
#          'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec', 'koi_tce_delivname', 'koi_teq_err1', 'koi_teq_err2'])
# Brak w bazie: 'koi_vet_stat', 'koi_vet_date', 'koi_disp_prov', 'koi_comment'
# Dodatkowo: koi_tce_delivname (jakaś nazwa); koi_teq_err1, koi_teq_err2 (puste)

# Zamiana etykiety na wartość binarną, jak się po prostu zamieni CONFIRMED na True to wywala błąd, bo wcześniej był to
# string. Dlatego trochę więcej kodu jest
y_string = data['koi_disposition']
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
result = [0 for i in range(len(estimators))]

# Wypełnienie pustych rekordów średnimi
X.fillna(X.mean(), inplace=True)

# Balansowanie zbioru przy użyciu metody SMOTE - Synthetic Minority Over-sampling Technique z biblioteki imblearn
sm = SMOTE(random_state=1410)
X_resampled, y_resampled = sm.fit_resample(X, y)
min_max_scaler = preprocessing.MinMaxScaler()

# Krotność walidacji krzyżowej
times_cross_validation = 5

# Walidacja krzyżowa
kf = KFold(n_splits=times_cross_validation, shuffle=True, random_state=1410)

for train_index, test_index in tqdm.tqdm(kf.split(X_resampled)):
    X_train, X_test = X_resampled.iloc[train_index], X_resampled.iloc[test_index]
    y_train, y_test = y_resampled.iloc[train_index], y_resampled.iloc[test_index]

    # Skalowanie danych, żeby wszytskie wartości znajdowały się w zakresie (0, 1)
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.fit_transform(X_test)

    # Iteracja po algorytmach uczących
    for i in range(1, len(estimators)):
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
# N = len(metric_names)
# theta = RadarChart.radar_factory(N, frame='polygon')

j = 0
for metric in [result_accuracy, result_precision, result_recall, result_f1]:
    print('\nMetryka: ' + metric_names[j])
    for i in range(len(estimators)):
        print(f'{names[i]} {metric[i] / times_cross_validation}')
    j += 1

# fig, axs = plt.subplots(subplot_kw=dict(projection='radar'))
# colors = ['b', 'r', 'g', 'm', 'y']

# Plot the four cases from the example data on separate axes
# for ax, (title, case_data) in zip(axs.flat, data):
#     ax.set_rgrids([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
#     ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
#                  horizontalalignment='center', verticalalignment='center')
#     for d, color in zip(case_data, colors):
#         ax.plot(theta, d, color=color)
#         ax.fill(theta, d, facecolor=color, alpha=0.25)
#     ax.set_varlabels(spoke_labels)

# add legend relative to top-left plot
# labels = ('Accuracy', 'Precision', 'Recall', 'F1')
# legend = axs[0, 0].legend(labels, loc=(0.9, .95), labelspacing=0.1, fontsize='small')
# fig.text(0.5, 0.965, 'Metryki', horizontalalignment='center', color='black', weight='bold', size='large')
# plt.show()
