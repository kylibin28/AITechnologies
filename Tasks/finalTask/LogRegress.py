import datetime

import matplotlib.pyplot as plt
import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score





# Task #2 - логистическая регрессия **********************************************************************************


features = pandas.read_csv('./features.csv', index_col='match_id')

y = features['radiant_win']

features = features.drop('duration', 1)
features = features.drop('radiant_win', 1)
features = features.drop('tower_status_radiant', 1)
features = features.drop('tower_status_dire', 1)
features = features.drop('barracks_status_radiant', 1)
features = features.drop('barracks_status_dire', 1)

X = features
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
    c_range = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
    for c in c_range:
        print('c=', c)
        model_time_start = datetime.datetime.now()

        model = LogisticRegression(solver='newton-cg', C=c, random_state=241)
        model_accuracy = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        scores.append(model_accuracy)

        print('Time:', datetime.datetime.now() - model_time_start)
        print('model_accuracy=', model_accuracy.mean())
        print()

    return pandas.DataFrame(scores, c_range).mean(axis=1).sort_values(ascending=False)





total_time_logistic_start = datetime.datetime.now()

cv = KFold(n_splits=5, shuffle=True, random_state=241)
accuracy = test_accuracy_logic_regress(cv, X, y)

top_accuracy = accuracy.head(1)
print('top_accuracy.index[0]=', top_accuracy.index[0])
print('top_accuracy.values[0]=', top_accuracy.values[0])

show_plot(accuracy, 'logistic_plot')

print('_logistic Time elapsed:', datetime.datetime.now() - total_time_logistic_start)

# top_accuracy.index[0]= 0.5
# top_accuracy.values[0]= 0.6549933148205287
# _logistic Time elapsed: 0:34:22.788164





# 2. Среди признаков в выборке есть категориальные, которые мы использовали как числовые, что вряд ли является хорошей идеей.
# Категориальных признаков в этой задаче одиннадцать: lobby_type и r1_hero, r2_hero, ..., r5_hero, d1_hero, d2_hero, ..., d5_hero.
# Уберите их из выборки, и проведите кросс-валидацию для логистической регрессии на новой выборке с подбором лучшего параметра регуляризации.
# Изменилось ли качество? Чем вы можете это объяснить?



total_time_logistic_start_1 = datetime.datetime.now()

cv_1 = KFold(n_splits=5, shuffle=True, random_state=241)

X = X.drop('lobby_type', 1)
X = X.drop('r1_hero', 1)
X = X.drop('r2_hero', 1)
X = X.drop('r3_hero', 1)
X = X.drop('r4_hero', 1)
X = X.drop('r5_hero', 1)
X = X.drop('d1_hero', 1)
X = X.drop('d2_hero', 1)
X = X.drop('d3_hero', 1)
X = X.drop('d4_hero', 1)
X = X.drop('d5_hero', 1)

accuracy_1 = test_accuracy_logic_regress(cv, X, y)

top_accuracy_1 = accuracy_1.head(1)
print('top_accuracy_1.index[0]=', top_accuracy_1.index[0])
print('top_accuracy_1.values[0]=', top_accuracy_1.values[0])

show_plot(accuracy_1, 'logistic_plot_1')

print('_logistic Time elapsed:', datetime.datetime.now() - total_time_logistic_start_1)




# 3. На предыдущем шаге мы исключили из выборки признаки rM_hero и dM_hero, которые показывают, какие именно герои играли за каждую команду.
# Это важные признаки — герои имеют разные характеристики, и некоторые из них выигрывают чаще, чем другие.
# Выясните из данных, сколько различных идентификаторов героев существует в данной игре (вам может пригодиться фукнция unique или value_counts).



