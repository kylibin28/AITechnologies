import pandas

# 1. Считайте таблицу с признаками из файла features.csv с помощью кода, приведенного выше.
# Удалите признаки, связанные с итогами матча (они помечены в описании данных как отсутствующие в тестовой выборке).
features = pandas.read_csv('./features.csv', index_col='match_id')
# features.head()

features = features.drop('duration', 1)
features = features.drop('radiant_win', 1)
features = features.drop('tower_status_radiant', 1)
features = features.drop('tower_status_dire', 1)
features = features.drop('barracks_status_radiant', 1)
features = features.drop('barracks_status_dire', 1)

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


# 5. Забудем, что в выборке есть категориальные признаки,
# и попробуем обучить градиентный бустинг над деревьями на имеющейся матрице "объекты-признаки".
# Зафиксируйте генератор разбиений для кросс-валидации по 5 блокам (KFold),
# не забудьте перемешать при этом выборку (shuffle=True), поскольку данные в таблице отсортированы по времени,
# и без перемешивания можно столкнуться с нежелательными эффектами при оценивании качества.
# Оцените качество градиентного бустинга (GradientBoostingClassifier) с помощью данной кросс-валидации,
# попробуйте при этом разное количество деревьев (как минимум протестируйте следующие значения для количества деревьев: 10, 20, 30). Долго ли настраивались классификаторы? Достигнут ли оптимум на испытанных значениях параметра n_estimators, или же качество, скорее всего, продолжит расти при дальнейшем его увеличении?