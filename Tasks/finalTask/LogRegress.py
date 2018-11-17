import datetime

import matplotlib.pyplot as plt
import pandas
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score

# Task #2 - логистическая регрессия **********************************************************************************


features = pandas.read_csv('./input_data/features.csv', index_col='match_id')

y = features['radiant_win']

X = features.drop(columns=['duration',
                           'radiant_win',
                           'tower_status_radiant',
                           'tower_status_dire',
                           'barracks_status_radiant',
                           'barracks_status_dire'])

X.fillna('0', inplace=True)


def show_plot(accuracy, plot_name):
    plt.figure()
    plt.plot(accuracy.index, accuracy.values, 'r', linewidth=2)
    plt.legend(['test', 'train'])
    plt.savefig('../rate_' + str(plot_name) + '.png')


# 1. Оцените качество логистической регрессии (sklearn.linear_model.LogisticRegression с L2-регуляризацией)
# с помощью кросс-валидации по той же схеме, которая использовалась для градиентного бустинга.
# Подберите при этом лучший параметр регуляризации (C). Какое наилучшее качество у вас получилось?
# Как оно соотносится с качеством градиентного бустинга?
# Чем вы можете объяснить эту разницу?
# Быстрее ли работает логистическая регрессия по сравнению с градиентным бустингом?

def test_accuracy_logic_regress(cv, X, y):
    scores = list()
    c_range = [0.01, 0.05, 0.1, 0.5, 1.0]
    for c in c_range:
        print('c=', c)
        model_time_start = datetime.datetime.now()

        model = LogisticRegression(solver='newton-cg', C=c, random_state=42)
        model_accuracy = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        scores.append(model_accuracy)

        print('Time:', datetime.datetime.now() - model_time_start)
        print('model_accuracy=', model_accuracy.mean())
        print()

    return pandas.DataFrame(scores, c_range).mean(axis=1).sort_values(ascending=False)


# total_time_logistic_start = datetime.datetime.now()
#
# cv = KFold(n_splits=5, shuffle=True, random_state=42)
# accuracy = test_accuracy_logic_regress(cv, X, y)
#
# top_accuracy = accuracy.head(1)
# print('top_accuracy.index[0]=', top_accuracy.index[0])
# print('top_accuracy.values[0]=', top_accuracy.values[0])
#
# show_plot(accuracy, 'logistic_plot')
#
# print('_logistic Time elapsed:', datetime.datetime.now() - total_time_logistic_start)

# top_accuracy.index[0]= 0.01
# top_accuracy.values[0]= 0.7165888549784863
# _logistic Time elapsed: 0:27:28.202461


# 2. Среди признаков в выборке есть категориальные, которые мы использовали как числовые, что вряд ли является хорошей идеей.
# Категориальных признаков в этой задаче одиннадцать: lobby_type и r1_hero, r2_hero, ..., r5_hero, d1_hero, d2_hero, ..., d5_hero.
# Уберите их из выборки, и проведите кросс-валидацию для логистической регрессии на новой выборке с подбором лучшего параметра регуляризации.
# Изменилось ли качество? Чем вы можете это объяснить?


# total_time_logistic_start_1 = datetime.datetime.now()
#
# cv_1 = KFold(n_splits=5, shuffle=True, random_state=42)
#
# X = X.drop(columns=['lobby_type',
#            'r1_hero',
#            'r2_hero',
#            'r3_hero',
#            'r4_hero',
#            'r5_hero',
#            'd1_hero',
#            'd2_hero',
#            'd3_hero',
#            'd4_hero',
#            'd5_hero'])
#
# accuracy_1 = test_accuracy_logic_regress(cv_1, X, y)
#
# top_accuracy_1 = accuracy_1.head(1)
# print('top_accuracy_1.index[0]=', top_accuracy_1.index[0])
# print('top_accuracy_1.values[0]=', top_accuracy_1.values[0])
#
# show_plot(accuracy_1, 'logistic_plot_1')
#
# print('_logistic Time elapsed:', datetime.datetime.now() - total_time_logistic_start_1)


# top_accuracy_1.index[0]= 2.0
# top_accuracy_1.values[0]= 0.716602803580008
# _logistic Time elapsed: 0:24:09.310879

# 3. На предыдущем шаге мы исключили из выборки признаки rM_hero и dM_hero, которые показывают, какие именно герои играли за каждую команду.
# Это важные признаки — герои имеют разные характеристики, и некоторые из них выигрывают чаще, чем другие.
# Выясните из данных, сколько различных идентификаторов героев существует в данной игре (вам может пригодиться фукнция unique или value_counts).

hero_features = features[['r1_hero',
                          'r2_hero',
                          'r3_hero',
                          'r4_hero',
                          'r5_hero',
                          'd1_hero',
                          'd2_hero',
                          'd3_hero',
                          'd4_hero',
                          'd5_hero']]

n_len = len(pandas.unique(hero_features.values.ravel('K')))

print('unique counts=', n_len)
# unique counts= 108

# 4. Воспользуемся подходом "мешок слов" для кодирования информации о героях.
# Пусть всего в игре имеет N различных героев.
# Сформируем N признаков, при этом i-й будет равен нулю, если i-й герой не участвовал в матче;
# единице, если i-й герой играл за команду Radiant; минус единице, если i-й герой играл за команду Dire.
# Ниже вы можете найти код, который выполняет данной преобразование.
# Добавьте полученные признаки к числовым, которые вы использовали во втором пункте данного этапа.

print(hero_features)

X_pick = np.zeros((hero_features.shape[0], n_len))

for i, match_id in enumerate(hero_features.index):
    for p in range(5):
        X_pick[i, hero_features.ix[match_id, 'r%d_hero' % (p + 1)] - 1] = 1
        X_pick[i, hero_features.ix[match_id, 'd%d_hero' % (p + 1)] - 1] = -1
