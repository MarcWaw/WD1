import numpy as np
from scipy.stats import ttest_ind
from tabulate import tabulate

def t_student(clfs, headers_array, scores, alfa = .05, print_result = False):
    t_statistic = np.zeros((len(clfs), len(clfs)))
    p_value = np.zeros((len(clfs), len(clfs)))
    print('Scores: ')
    print(scores)
    # Wyliczenie t_statystyki i p-value dla wszytskich par
    for i in range(len(clfs)):
        for j in range(len(clfs)):
            t_statistic[i, j], p_value[i, j] = ttest_ind(scores[i], scores[j])
    print(t_statistic)
    # Wyliczenie przewagi danego algorytmu
    advantage = np.zeros((len(clfs), len(clfs)))
    advantage[t_statistic > 0] = 1

    # Wyliczenie które algorytmy sa statystycznie różne
    significance = np.zeros((len(clfs), len(clfs)))
    significance[p_value <= alfa] = 1

    #Wymnożenie macieży przewag i macieży znaczności
    stat_better = significance * advantage
    if print_result == True:
        # Printowanie danych
        headers_array_in_array = []  # Tabela z nazwami w formacie np [["GNB"], ["kNN"], ["CART"]]
        for name in headers_array:
            temp = []
            temp.append(name)
            headers_array_in_array.append(temp)

        t_statistic_table = np.concatenate((headers_array_in_array, t_statistic), axis=1)
        t_statistic_table = tabulate(t_statistic_table, headers_array, floatfmt=".2f")

        p_value_table = np.concatenate((headers_array_in_array, p_value), axis=1)
        p_value_table = tabulate(p_value_table, headers_array, floatfmt=".2f")

        advantage_table = tabulate(np.concatenate((headers_array_in_array, advantage), axis=1), headers_array)

        significance_table = tabulate(np.concatenate((headers_array_in_array, significance), axis=1), headers_array)

        stat_better_table = tabulate(np.concatenate((headers_array_in_array, stat_better), axis=1), headers_array)

        print('------------------------------------------------------------------')
        print(f"t-statistic:\n {t_statistic_table} \n\np-value:\n {p_value_table} \n\nAdvantage:\n {advantage_table} \n\nStatistical "
              f"significance (alpha = {alfa}):\n {significance_table} \n\nStatistically significantly better:\n {stat_better_table}")
        print('------------------------------------------------------------------')
    return stat_better