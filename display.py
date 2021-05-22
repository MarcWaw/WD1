import plotly.graph_objects as go
import plotly.offline as pyo
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np
import Statistic
import texttable


# Plotowanie ROC
def plot_roc_curve(fpr, tpr, label=None):
    for i in range(len(label)):
        plt.plot(fpr[i], tpr[i], linewidth=2, label=label[i])
    plt.plot([0, 1], [0, 1], 'k--')  # przekątna wykresu
    plt.xlabel('Odsetek fałszywie pozytywnych')
    plt.ylabel('Odsetek prawdziwie pozytywnych (pełność)')
    plt.legend()


def show_results(scores, estimators_names, metric_names):

    mean_accuracy = scores[0].mean(1)
    mean_precision = scores[1].mean(1)
    mean_recall = scores[2].mean(1)
    mean_f1 = scores[3].mean(1)

    plot_tree_results = [estimators_names[0], mean_accuracy[0], mean_precision[0], mean_recall[0], mean_f1[0]]
    plot_svm = [estimators_names[1], mean_accuracy[1], mean_precision[1], mean_recall[1], mean_f1[1]]
    plot_knn = [estimators_names[2], mean_accuracy[2], mean_precision[2], mean_recall[2], mean_f1[2]]
    plot_nb = [estimators_names[3], mean_accuracy[3], mean_precision[3], mean_recall[3], mean_f1[3]]
    plot_lr = [estimators_names[4], mean_accuracy[4], mean_precision[4], mean_recall[4], mean_f1[4]]
    plots = [plot_tree_results, plot_svm, plot_knn, plot_nb, plot_lr]
    print(tabulate(plots, metric_names))

    # -----------------------------------------------------------------------------------------------------------------
    # Wyświetlanie wykresu
    plot_names = [*metric_names, metric_names[0]]
    fig = go.Figure(data=[go.Scatterpolar(r=plot_tree_results, theta=plot_names, name='Drzewo decyzyjne'),
                          go.Scatterpolar(r=plot_svm, theta=plot_names, name='SVM'),
                          go.Scatterpolar(r=plot_knn, theta=plot_names, name='kNN'),
                          go.Scatterpolar(r=plot_nb, theta=plot_names, name='Naiwny Bayes'),
                          go.Scatterpolar(r=plot_lr, theta=plot_names, name='Regresja Logistyczna')],
                    layout=go.Layout(title=go.layout.Title(text='Wyniki'),
                                     polar={'radialaxis': {'visible': True}},
                                     showlegend=True))
    pyo.plot(fig)


def slicer_vectorized(a, start, end):
    b = a.view((str, 1)).reshape(len(a), -1)[:, start:end]
    return np.fromstring(b.tostring(), dtype=(str, end - start))


def prepare_latex_data(t_student, clfs_names):
    latex_array = []
    for t_s in t_student:
        sign_t = [""]
        n_t_s = t_s.T
        for i in range(len(n_t_s)):
            better_than = ""
            check = []
            for j in range(len(n_t_s)):
                if n_t_s[j][i] == 1:
                    check.append(1)
                    better_than += str(j + 1)
                    better_than += ','
            if len(check) == len(clfs_names) - 1:
                better_than = "all"
            if better_than != '-' and better_than != 'all':
                sign_t.append(better_than[:-1])
            else:
                sign_t.append(better_than)

        for u in range(1, len(sign_t)):
            if len(sign_t[u]) == 0 or sign_t[u] is None:
                sign_t[u] = "-"
        latex_array.append(sign_t)
    return latex_array


def GenerateLatexTable(all_scores, dtn, t_student, clfs_names):
    t_s_arr = prepare_latex_data(t_student, clfs_names)
    names = ['Score', 'Tree', 'SVM', 'kNN', 'GNB', 'RegLog', 'B:Tree', 'B:SVM', 'B:kNN', 'B:GNB', 'B:RegLog']
    space_row = np.full(len(names), ' ', dtype=str)
    number_of_vals = all_scores[0].shape[1]  # 9
    number_of_data_sets = all_scores[0].shape[0]  # number of data sets
    number_of_folds = all_scores[0].shape[2]  # 5
    arr = []
    arr_mean = []
    rows = [names]
    for i in range(number_of_data_sets):
        for j in range(number_of_vals):
            for t in range(number_of_folds):
                arr.append(all_scores[0][i][j][t])
            arr_mean.append(np.mean(arr.copy()))
            arr = []
        int_arr = np.round(arr_mean, 10)  # generating U10 array for full dataset name
        str_arr = list(map(str, int_arr))  # int -> str array
        temp = np.insert(str_arr, 0, dtn[i])  # inserting dataset file name
        int_arr = np.array(temp[1:])  # selecting all apart from first
        str_arr = slicer_vectorized(int_arr, 0, 5)  # setting U5 array
        for k in range(1, len(temp)):  # switching items U10 -> U5
            temp[k] = str_arr[k - 1]
        rows.append(temp.copy())  # appending rows to final array
        rows.append(t_s_arr[i])
        rows.append(space_row)
        arr_mean = []

    table = texttable.Texttable()
    table.set_cols_align(["c"] * len(rows))
    table.set_deco(texttable.Texttable.HEADER | texttable.Texttable.VLINES)
    print("\\begin{tabular}{lcccccccccc}")
    print('Tabulate Latex:')
    print(tabulate(rows, headers='firstrow', tablefmt='latex'))
