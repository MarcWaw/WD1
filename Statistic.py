import numpy as np
from scipy.stats import ttest_ind


def t_student(headers_array, scores, alfa=.05, print_result=False):
    t_statistic = np.zeros((len(headers_array), len(headers_array)))
    p_value = np.zeros((len(headers_array), len(headers_array)))
    # Wyliczenie t_statystyki i p-value dla wszytskich par
    for i in range(len(headers_array)):
        for j in range(len(headers_array)):
            t_statistic[i, j], p_value[i, j] = ttest_ind(scores[i], scores[j])
    # Wyliczenie przewagi danego algorytmu
    advantage = np.zeros((len(headers_array), len(headers_array)))
    advantage[t_statistic > 0] = 1

    # Wyliczenie które algorytmy sa statystycznie różne
    significance = np.zeros((len(headers_array), len(headers_array)))
    significance[p_value <= alfa] = 1

    # Wymnożenie macieży przewag i macieży znaczności
    stat_better = significance * advantage

    return stat_better, p_value
