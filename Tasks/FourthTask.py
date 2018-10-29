import numpy as np
import pandas
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# 1. Загрузите обучающую и тестовую выборки из файлов perceptron-train.csv
# и perceptron-test.csv. Целевая переменная записана в первом столбце, признаки — во втором и третьем.
trainData = pandas.read_csv('../source/perceptron-train.csv', header=None)
y_train = trainData[0]
X_train = trainData.loc[:, 1:]

testData = pandas.read_csv('../source/perceptron-test.csv', header=None)
y_test = testData[0]
X_test = testData.loc[:, 1:]

# 2. Обучите персептрон со стандартными параметрами и random_state=241.
clf = Perceptron(random_state=241)
clf.fit(X_train, y_train)

# 3. Подсчитайте качество (долю правильно классифицированных объектов, accuracy) полученного классификатора на тестовой выборке.
accuracy = accuracy_score(y_test, clf.predict(X_test))
print("accuracy without scale=", accuracy)

# 4. Нормализуйте обучающую и тестовую выборку с помощью класса StandardScaler.
scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)

# 5. Обучите персептрон на новой выборке. Найдите долю правильных ответов на тестовой выборке.
clf_scale = Perceptron(random_state=241)
clf_scale.fit(X_train_scale, y_train)

accuracy_scale = accuracy_score(y_test, clf_scale.predict(X_test_scale))
print("accuracy with scale=", accuracy_scale)

# 6. Найдите разность между качеством на тестовой выборке после нормализации и качеством до нее. Это число и будет ответом на задание.
print("different=", accuracy_scale - accuracy)


