import pandas
import sklearn
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

import sys

# Загрузите выборку из файла titanic.csv с помощью пакета Pandas.
# Оставьте в выборке четыре признака: класс пассажира (Pclass), цену билета (Fare), возраст пассажира (Age) и его пол (Sex).
# Обратите внимание, что признак Sex имеет строковые значения.
# Выделите целевую переменную — она записана в столбце Survived.
# В данных есть пропущенные значения — например, для некоторых пассажиров неизвестен их возраст. Такие записи при чтении их в pandas принимают значение nan. Найдите все объекты, у которых есть пропущенные признаки, и удалите их из выборки.
# Обучите решающее дерево с параметром random_state=241 и остальными параметрами по умолчанию (речь идет о параметрах конструктора DecisionTreeСlassifier).
# Вычислите важности признаков и найдите два признака с наибольшей важностью. Их названия будут ответами для данной задачи (в качестве ответа укажите названия признаков через запятую или пробел, порядок не важен).

sys.path.append("..")

data = pandas.read_csv('../source/wine.csv', header=None)

y = data[0]
X = data.loc[:, 1:]
# print(y)
# print(X)

kf = KFold(n_splits=5, shuffle=True, random_state=42)


def test_accuracy(kf, X, y):
    scores = list()
    k_range = range(1, 51)
    for k in k_range:
        model = KNeighborsClassifier(n_neighbors=k)
        scores.append(cross_val_score(model, X, y, cv=kf, scoring='accuracy'))

    return pandas.DataFrame(scores, k_range).mean(axis=1).sort_values(ascending=False)


accuracy = test_accuracy(kf, X, y)
top_accuracy = accuracy.head(1)
print(1, top_accuracy.index[0])
print(2, top_accuracy.values[0])

# 1 1
# 2 0.7304761904761905

# Масштабирование
X = sklearn.preprocessing.scale(X)
accuracy = test_accuracy(kf, X, y)

top_accuracy = accuracy.head(1)
print(3, top_accuracy.index[0])
print(4, top_accuracy.values[0])

# 3 29
# 4 0.9776190476190475