import pandas as pd
import numpy as np
import tqdm
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score


data = pd.read_csv('Data\cumulative.csv')
X = data.drop(columns=['kepid', 'kepoi_name', 'kepler_name', 'koi_disposition', 'koi_pdisposition', 'koi_score', 'koi_fpflag_nt',
                       'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec', 'koi_tce_delivname', 'koi_teq_err1', 'koi_teq_err2'])
# Brak w bazie: 'koi_vet_stat', 'koi_vet_date', 'koi_disp_prov', 'koi_comment'
# wyjebałem dodatkowo: koi_tce_delivname (jakaś nazwa); koi_teq_err1, koi_teq_err2 (puste)

#Zamiana etykiety na wartość binarną, jak się po prostu zamieni CONFIRMED na True to wywala błąd, bo wcześniej był to string
#Dlatego trochę więcej kodu jest
y_string = data['koi_disposition']
y_list = []
for classification in y_string:
    if classification == 'CONFIRMED':
        y_list.append(True)
    else:
        y_list.append(False)
y = pd.DataFrame(y_list, columns=['koi_disposition'])

estimators = [DecisionTreeClassifier(), SVC(), KNeighborsClassifier(), GaussianNB()] #LogisticRegression() - wywala błędy póki co, może trzeba standaryzacje zrobić
names = ['Drzewa Decyzyjne', 'SVM', 'kNN', 'naiwny Bayes']
result = [0 for i in range(len(estimators))]

#Wypełnienie pustych rekordów średnimi
X.fillna(X.mean(), inplace=True)

times_cross_validation = 10
#Walidacja krzyżowa
kf = KFold(n_splits=times_cross_validation, shuffle=True, random_state=1410)
for train_index, test_index in tqdm.tqdm(kf.split(X)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    #Iteracja po algorytmach uczących
    for i in range(len(estimators)):
        clf = BaggingClassifier(base_estimator=estimators[i], n_estimators=10, random_state=0).fit(X_train, y_train.values.ravel())
        predict = clf.predict(X_test)
        result[i] += accuracy_score(y_test, predict)

for i in range(len(estimators)):
    print(f'{names[i]} {result[i]/times_cross_validation}')

