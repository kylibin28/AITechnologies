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

# print("space:\n", features.count())

# 3. Замените пропуски на нули с помощью функции fillna().
# На самом деле этот способ является предпочтительным для логистической регрессии,
# поскольку он позволит пропущенному значению не вносить никакого вклада в предсказание.
# Для деревьев часто лучшим вариантом оказывается замена пропуска на очень большое или очень маленькое значение —
# в этом случае при построении разбиения вершины можно будет отправить объекты с пропусками в отдельную ветвь дерева.
# Также есть и другие подходы — например, замена пропуска на среднее значение признака.
# Мы не требуем этого в задании, но при желании попробуйте разные подходы к обработке пропусков и сравните их между собой.

features.fillna('0', inplace=True)

# print("without space:\n", features.count())

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

#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7)
#
#
# def log_loss_results(model, X, y):
#     # Используйте метод staged_decision_function для предсказания качества на обучающей и тестовой выборке
#     # на каждой итерации.
#     results = []
#     for pred in model.staged_decision_function(X):
#         results.append(log_loss(y, [sigmoid(y_pred) for y_pred in pred]))
#
#     return results
#
#
def show_plot(accuracy):
    plt.figure()
    plt.plot(accuracy.index, accuracy.values, 'r', linewidth=2)
    plt.legend(['test', 'train'])
    plt.savefig('../rate.png')



#
# def model_test(learning_rate):
#     model = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=250, verbose=True, random_state=241)
#     model.fit(X, y)
#
#     train_loss = log_loss_results(model, X_train, y_train)
#     test_loss = log_loss_results(model, X_test, y_test)
#     return plot_loss(learning_rate, test_loss, train_loss)
#
#
# def forest_train(n_estimators_rate):
#     model = RandomForestClassifier(n_estimators=n_estimators_rate, random_state=241)
#     model.fit(X_train, y_train)
#     y_pred = model.predict_proba(X_test)[:, 1]
#     test_loss = log_loss(y_test, y_pred)
#     print(3, test_loss)


# Оцените качество градиентного бустинга (GradientBoostingClassifier) с помощью данной кросс-валидации,
# попробуйте при этом разное количество деревьев (как минимум протестируйте следующие значения для количества деревьев: 10, 20, 30).
def test_accuracy(cv, X, y):
    scores = list()
    k_range = [10, 30, 50, 70, 100]
    for k in k_range:
        print('k=', k)
        model = RandomForestClassifier(n_estimators=k)
        scores.append(cross_val_score(model, X, y, cv=cv, scoring='accuracy'))

    return pandas.DataFrame(scores, k_range).mean(axis=1).sort_values(ascending=False)


# ***************************************************************************

start_time = datetime.datetime.now()

cv = KFold(n_splits=5, shuffle=True)
accuracy = test_accuracy(cv, X, y)


top_accuracy = accuracy.head(1)
print('top_accuracy.index[0]=', top_accuracy.index[0])
print('top_accuracy.values[0]=', top_accuracy.values[0])

show_plot(accuracy)

#top_accuracy.index[0]= 100
# top_accuracy.values[0]= 0.6396585416023861
# Time elapsed: 0:08:41.905862

print('Time elapsed:', datetime.datetime.now() - start_time)
