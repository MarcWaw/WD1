import pandas as pd
import tqdm
import numpy as np
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve

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

clfs_names = ['Tree', 'SVM', 'kNN', 'GNB', 'RegLog', 'B:Tree', 'B:SVM', 'B:kNN', 'B:GNB', 'B:RegLog']
clfs = [DecisionTreeClassifier(), SVC(probability=True), KNeighborsClassifier(), GaussianNB(),
        LogisticRegression(solver='lbfgs', max_iter=1000),
        BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=10, random_state=0),
        BaggingClassifier(base_estimator=SVC(probability=True), n_estimators=10, random_state=0),
        BaggingClassifier(base_estimator=KNeighborsClassifier(), n_estimators=10, random_state=0),
        BaggingClassifier(base_estimator=GaussianNB(), n_estimators=10, random_state=0),
        BaggingClassifier(base_estimator=LogisticRegression(solver='lbfgs', max_iter=1000), n_estimators=10,
                          random_state=0)]

# Balansowanie zbioru przy użyciu metody SMOTE - Synthetic Minority Over-sampling Technique z biblioteki imblearn
sm = SMOTE(random_state=1410)
X_resampled, y_resampled = sm.fit_resample(X, y)

# Skalowanie do normalizacji
min_max_scaler = preprocessing.MinMaxScaler()

# Walidacja krzyżowa
times_cross_validation = 10
kf = KFold(n_splits=times_cross_validation, shuffle=True, random_state=1410)

scores_names = ['Acc', 'Prec', 'Rec', 'F1', 'ROC']
scores = np.zeros((len(scores_names), len(clfs), times_cross_validation))

for i, (train_index, test_index) in tqdm.tqdm(enumerate(kf.split(X_resampled))):
    # -------------------------------------------------------------------------------------------------------------
    # Tablice do krzywych ROC
    fpr_array = []
    tpr_array = []
    for estimator_index, esti in enumerate(clfs):
        X_train, X_test = X_resampled.iloc[train_index], X_resampled.iloc[test_index]
        y_train, y_test = y_resampled.iloc[train_index], y_resampled.iloc[test_index]
        # Czy występuje normalizacja parametrów?
        if clfs_names[estimator_index] == 'RegLog' or clfs_names[estimator_index] == 'B:RegLog':
            min_max_scaler.fit(X_train)
            X_train = min_max_scaler.transform(X_train)
            X_test = min_max_scaler.transform(X_test)
        # Iteracja przez estymatory
        clf = esti.fit(X_train, y_train.values.ravel())
        prediction = clf.predict(X_test)
        prediction_roc = clf.predict_proba(X_test)  # Predict do ROC

        # Zapisywanie wyników metryk
        scores[:, estimator_index, i] = [accuracy_score(y_test, prediction), precision_score(y_test, prediction),
                                         recall_score(y_test, prediction), f1_score(y_test, prediction),
                                         roc_auc_score(y_test, prediction_roc[:, 1])]

        # ROC
        fpr, tpr, treshold = roc_curve(y_test, prediction_roc[:, 1])
        fpr_array.append(fpr)
        tpr_array.append(tpr)
    np.save(rf'Results\fpr_fold{i}.npy', np.array(fpr_array, dtype="object"))
    np.save(rf'Results\tpr_fold{i}.npy', np.array(tpr_array, dtype="object"))

np.save(rf'Results\clfs_names.npy', clfs_names)
np.save(rf'Results\scores_names.npy', scores_names)
np.save(rf'Results\scores.npy', scores)
