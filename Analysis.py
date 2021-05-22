import display
import numpy as np
import Statistic
import matplotlib.pyplot as plt

# Nazwy klasyfikatorów
clfs_names = np.load(rf'Results\clfs_names.npy')
# Nazwy metryk ['Accuracy', 'Precision', 'Recall', 'F1']
scores_names = np.load(rf'Results\scores_names.npy')
# 3-wymiarowa tablica z wynikami (metryka_dokładności, klasyfikator, fold walidacji krzyżowej)
scores = np.load(rf'Results\scores.npy')

times_cross_validation = scores.shape[2]

# Krzywe ROC
for i in range(times_cross_validation):
    fpr = np.load(rf'Results\fpr_fold{i}.npy', allow_pickle=True).tolist()
    tpr = np.load(rf'Results\tpr_fold{i}.npy', allow_pickle=True).tolist()
    display.plot_roc_curve(fpr, tpr, clfs_names)
    plt.savefig(f'ROC_plots/ROC_fold{i + 1}.png')
    plt.clf()

t_student = []
p_value = []
for i in range(len(scores_names)):
    t_st, p_v = Statistic.t_student(clfs_names, scores[i], 0.05)
    t_student.append(t_st)
    p_value.append(p_v)
    # print(scores[i])


# display.show_results(scores, clfs_names, scores_names)
display.GenerateLatexTable([scores], scores_names, t_student, clfs_names)

