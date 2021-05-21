import plotly.graph_objects as go
import plotly.offline as pyo
import matplotlib.pyplot as plt
from tabulate import tabulate
import Statistic


# Plotowanie ROC
def plot_roc_curve(fpr, tpr, label=None):
    for i in range(len(label)):
        plt.plot(fpr[i], tpr[i], linewidth=2, label=label[i])
    plt.plot([0, 1], [0, 1], 'k--')  # przekątna wykresu
    plt.xlabel('Odsetek prawdziwie pozytywnych (pełność)')
    plt.ylabel('Odsetek preawdziwie pozytywnych')
    plt.legend()


def show_results(estimators, scores):
    # Wyświetlanie wyników
    estimators_names = ['Drzewa Decyzyjne', 'SVM', 'kNN', 'Naiwny Bayes', 'Log regrsieon']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1']

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
    # estimators_names_arr  ay_in_array = np.expand_dims(np.array(estimators_names), axis=1)
    # print(tabulate(np.concatenate((estimators_names_array_in_array, plots), axis=1), metric_names))
    for j, metric in enumerate([mean_accuracy, mean_precision, mean_recall, mean_f1]):
        print('\nMetryka: ' + metric_names[j] + '\n')
        # for i in range(len(estimators)):
        #     print(names[i] + ": " + str(format(metric[i] * 100, '.1f')))
        print(f'{Statistic.t_student(estimators, estimators_names, scores[j], 0.05, False)}\n\n')

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
