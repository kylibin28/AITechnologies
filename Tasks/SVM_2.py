import pandas
from sklearn import datasets
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold

# 1. Загрузите объекты из новостного датасета 20 newsgroups,
# относящиеся к категориям "космос" и "атеизм" (инструкция приведена выше).
# Обратите внимание, что загрузка данных может занять несколько минут
newsgroups = datasets.fetch_20newsgroups(
    subset='all',
    categories=['alt.atheism', 'sci.space']
)
X = newsgroups.data
y = newsgroups.target

# 2. Вычислите TF-IDF-признаки для всех текстов. Обратите внимание, что в этом задании мы предлагаем вам вычислить
# TF-IDF по всем данным. При таком подходе получается, что признаки на обучающем множестве используют информацию
# из тестовой выборки — но такая ситуация вполне законна, поскольку мы не используем значения целевой переменной
# из теста. На практике нередко встречаются ситуации, когда признаки объектов тестовой выборки известны на момент обучения,
# и поэтому можно ими пользоваться при обучении алгоритма.

vectorizer = TfidfVectorizer()
vectorizer.fit_transform(X)

# 3. Подберите минимальный лучший параметр C из множества [10^-5, 10^-4, ... 10^4, 10^5] для SVM с линейным ядром
# (kernel='linear') при помощи кросс-валидации по 5 блокам. Укажите параметр random_state=241 и для SVM, и для KFold.
# В качестве меры качества используйте долю верных ответов (accuracy).

feature_mapping = vectorizer.get_feature_names()

grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(y.size, n_folds=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(vectorizer.transform(X), y)

C = gs.best_params_.get('C')

# 4. Обучите SVM по всей выборке с оптимальным параметром C, найденным на предыдущем шаге.

model = SVC(kernel='linear', random_state=241, C=C)
model.fit(vectorizer.transform(X), y)

# 5. Найдите 10 слов с наибольшим по модулю весом. Они являются ответом на это задание. Укажите их через запятую или
# пробел, в нижнем регистре, в лексикографическом порядке.

words = vectorizer.get_feature_names()
coef = pandas.DataFrame(model.coef_.data, model.coef_.indices)
top_words = coef[0].map(lambda w: abs(w)).sort_values(ascending=False).head(10).index.map(lambda i: words[i])
print("top_words=", top_words)
print(1, ','.join(top_words))
top_words.sort()
print(1, ','.join(top_words))