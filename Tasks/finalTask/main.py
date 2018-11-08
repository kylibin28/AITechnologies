import pandas
import time
import datetime
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss

# 1. Считайте таблицу с признаками из файла features.csv с помощью кода, приведенного выше.
# Удалите признаки, связанные с итогами матча (они помечены в описании данных как отсутствующие в тестовой выборке).
features = pandas.read_csv('./features.csv', index_col='match_id')
# features.head()

y = features['radiant_win']

features = features.drop('duration', 1)
features = features.drop('radiant_win', 1)
features = features.drop('tower_status_radiant', 1)
features = features.drop('tower_status_dire', 1)
features = features.drop('barracks_status_radiant', 1)
features = features.drop('barracks_status_dire', 1)

X = features

# 2. Проверьте выборку на наличие пропусков с помощью функции count(),
# которая для каждого столбца показывает число заполненных значений.
# Много ли пропусков в данных? Запишите названия признаков, имеющих пропуски,
# и попробуйте для любых двух из них дать обоснование, почему их значения могут быть пропущены.


# 3. Замените пропуски на нули с помощью функции fillna().
# На самом деле этот способ является предпочтительным для логистической регрессии,
# поскольку он позволит пропущенному значению не вносить никакого вклада в предсказание.
# Для деревьев часто лучшим вариантом оказывается замена пропуска на очень большое или очень маленькое значение —
# в этом случае при построении разбиения вершины можно будет отправить объекты с пропусками в отдельную ветвь дерева.
# Также есть и другие подходы — например, замена пропуска на среднее значение признака.
# Мы не требуем этого в задании, но при желании попробуйте разные подходы к обработке пропусков и сравните их между собой.

features.fillna('0', inplace=True)

# 4. Какой столбец содержит целевую переменную? Запишите его название.

print('radiant_win')


# 5. Забудем, что в выборке есть категориальные признаки,
# и попробуем обучить градиентный бустинг над деревьями на имеющейся матрице "объекты-признаки".
# Зафиксируйте генератор разбиений для кросс-валидации по 5 блокам (KFold),
# не забудьте перемешать при этом выборку (shuffle=True), поскольку данные в таблице отсортированы по времени,
# и без перемешивания можно столкнуться с нежелательными эффектами при оценивании качества.
# Оцените качество градиентного бустинга (GradientBoostingClassifier) с помощью данной кросс-валидации,
# попробуйте при этом разное количество деревьев (как минимум протестируйте следующие значения для количества деревьев: 10, 20, 30).
# Долго ли настраивались классификаторы?
# Достигнут ли оптимум на испытанных значениях параметра n_estimators, или же качество, скорее всего, продолжит расти при дальнейшем его увеличении?


def show_plot(accuracy):
    plt.figure()
    plt.plot(accuracy.index, accuracy.values, 'r', linewidth=2)
    plt.legend(['test', 'train'])
    plt.savefig('../rate.png')


# Оцените качество градиентного бустинга (GradientBoostingClassifier) с помощью данной кросс-валидации,
# попробуйте при этом разное количество деревьев (как минимум протестируйте следующие значения для количества деревьев: 10, 20, 30).
def test_accuracy(cv, X, y):
    scores = list()
    k_range = [10, 30, 50, 70, 100]
    for k in k_range:
        print('k=', k)
        model_time_start = datetime.datetime.now()

        model = RandomForestClassifier(n_estimators=k, random_state=241, n_jobs=-1)
        model_accuracy = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        scores.append(model_accuracy)

        print('Time:', datetime.datetime.now() - model_time_start)
        print('model_accuracy=', model_accuracy.mean())

    return pandas.DataFrame(scores, k_range).mean(axis=1).sort_values(ascending=False)





# ***************************************************************************

total_time_start = datetime.datetime.now()

cv = KFold(n_splits=5, shuffle=True, random_state=241)
accuracy = test_accuracy(cv, X, y)

top_accuracy = accuracy.head(1)
print('top_accuracy.index[0]=', top_accuracy.index[0])
print('top_accuracy.values[0]=', top_accuracy.values[0])

show_plot(accuracy)

print('Time elapsed:', datetime.datetime.now() - total_time_start)

# Results
# top_accuracy.index[0]= 100
# top_accuracy.values[0]= 0.6387740409338682
# Time elapsed: 0:02:18.327551


# В отчете по данному этапу вы должны ответить на следующие вопросы:

# 1. Какие признаки имеют пропуски среди своих значений?
# Что могут означать пропуски в этих признаках (ответьте на этот вопрос для двух любых признаков)?
# radiant_flying_courier_time: время приобретения предмета "flying_courier" - команда может не покупать предмет
# first_blood_time, first_blood_team и first_blood_player1 - игра может закончиться без кровопролития

# 2. Как называется столбец, содержащий целевую переменную?
# radiant_win

# 3. Как долго проводилась кросс-валидация для градиентного бустинга с 30 деревьями?
# Инструкцию по измерению времени можно найти ниже по тексту. Какое качество при этом получилось?
# Напомним, что в данном задании мы используем метрику качества AUC-ROC.
# k= 30
# Time: 0:00:17.476608
# model_accuracy= 0.6215982721382289

# 4. Имеет ли смысл использовать больше 30 деревьев в градиентном бустинге?
# Что бы вы предложили делать, чтобы ускорить его обучение при увеличении количества деревьев?

# При увеличении количеста деревьев точность классификации возрасла незначительно.
# Однако обучеие заняло значительно больше времени.
# Я считаю что быльше 30 деревьев использовать не целесообразно.

# Для ускорения обучения я использовал распараллеливание потоков.
# Градиентный бустинг очень хорошо распараллеливается. Время сокрасилось с 8.5 минут до 2.2