import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.ensemble import BaggingClassifier
from sklearn import model_selection

data = pd.read_csv('Data\cumulative.csv')
X = data.drop(columns=['kepid', 'kepoi_name', 'kepler_name', 'koi_disposition', 'koi_pdisposition', 'koi_score',
                       'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec'])
# Brak w bazie: 'koi_vet_stat', 'koi_vet_date', 'koi_disp_prov', 'koi_comment'

y = data['koi_disposition']
for i in range(len(y)):
    if y[i] == 'CONFIRMED':
        y.iloc[i] = 1
    else:
        y.iloc[i] = 0

kf = KFold(n_splits=10, shuffle=True, random_state=1410)
scores = []

# for train_index, test_index in kf.split(X):
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]

base_cls = DecisionTreeClassifier()
clf = BaggingClassifier(base_estimator=base_cls, n_estimators=10, random_state=0)
results = model_selection.cross_val_score(clf, X, y, cv = kf)

print("accuracy :")
print(results.mean())
