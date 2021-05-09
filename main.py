import pandas as pd
import numpy as np
import tqdm
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

data = pd.read_csv('Data\cumulative.csv')
X = data.drop(
    columns=['kepid', 'kepoi_name', 'kepler_name', 'koi_disposition', 'koi_pdisposition', 'koi_score', 'koi_fpflag_nt',
             'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec', 'koi_tce_delivname', 'koi_teq_err1', 'koi_teq_err2'])
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
              LogisticRegression(solver='lbfgs', max_iter=1000)]
names = ['Drzewa Decyzyjne', 'SVM', 'kNN', 'Naiwny Bayes', 'Regresja logistyczna']
result = [0 for i in range(len(estimators))]

# Wypełnienie pustych rekordów średnimi
X.fillna(X.mean(), inplace=True)

# Balansowanie zbioru przy użyciu metody SMOTE - Synthetic Minority Over-sampling Technique z biblioteki imblearn
sm = SMOTE(random_state=1410)
X_resampled, y_resampled = sm.fit_resample(X, y)
min_max_scaler = preprocessing.MinMaxScaler()

times_cross_validation = 10

# Walidacja krzyżowa
kf = KFold(n_splits=times_cross_validation, shuffle=True, random_state=1410)

for train_index, test_index in tqdm.tqdm(kf.split(X_resampled)):
    X_train, X_test = X_resampled.iloc[train_index], X_resampled.iloc[test_index]
    y_train, y_test = y_resampled.iloc[train_index], y_resampled.iloc[test_index]

    # Wykonanie predykcji dla algorytmu Drzew, który nie wymaga skalowania danych
    clf = BaggingClassifier(base_estimator=estimators[0], n_estimators=10, random_state=0).fit(X_train,
                                                                                               y_train.values.ravel())
    predict = clf.predict(X_test)
    result[0] += accuracy_score(y_test, predict)

    # Skalowanie danych, żeby wszytskie wartości znajdowały się w zakresie (0, 1)
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.fit_transform(X_test)

    # Iteracja po algorytmach uczących
    for i in range(1, len(estimators)):
        clf = BaggingClassifier(base_estimator=estimators[i], n_estimators=10, random_state=0).fit(
            X_train, y_train.values.ravel())
        predict = clf.predict(X_test)
        result[i] += accuracy_score(y_test, predict)

for i in range(len(estimators)):
    print(f'{names[i]} {result[i] / times_cross_validation}')
