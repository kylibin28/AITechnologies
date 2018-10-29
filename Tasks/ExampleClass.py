# import numpy as np
#
# X = np.random.normal(loc=1, scale=10, size=(1000, 50))
# m = np.mean(X, axis=0)
# std = np.std(X, axis=0)
# X_norm = ((X - m) / std)
# print(X_norm)
#
# Z = np.array([[4, 5, 0],
#               [1, 9, 3],
#               [5, 1, 1],
#               [3, 3, 3],
#               [9, 9, 9],
#               [4, 7, 1]])
#
# r = np.sum(Z, axis=1)
# print(np.nonzero(r > 10))
#
# A = np.eye(3)
# B = np.eye(3)
# print(A)
# print(B)
#
# AB = np.vstack((A, B))
# print(AB)
#



import numpy as np
import pandas
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


# 1. Загрузите обучающую и тестовую выборки из файлов perceptron-train.csv и perceptron-test.csv.
# Целевая переменная записана в первом столбце, признаки — во втором и третьем.

df_train = pandas.read_csv('../source/perceptron-train.csv', header=None)
y_train_new = df_train[0]
X_train_new = df_train.loc[:, 1:]

df_test = pandas.read_csv('../source/perceptron-test.csv', header=None)
y_test_new = df_test[0]
X_test_new = df_test.loc[:, 1:]

# 2. Обучите персептрон со стандартными параметрами и random_state=241.
model = Perceptron(random_state=241)
model.fit(X_train_new, y_train_new)

# 3. Подсчитайте качество (долю правильно классифицированных объектов, accuracy) полученного классификатора на тестовой выборке.
acc_before = accuracy_score(y_test_new, model.predict(X_test_new))
print(1, acc_before)
# 4. Нормализуйте обучающую и тестовую выборку с помощью класса StandardScaler.

scaler_new = StandardScaler()
X_train_scaled = scaler_new.fit_transform(X_train_new)
X_test_scaled = scaler_new.transform(X_test_new)

# 5. Обучите персептрон на новых выборках. Найдите долю правильных ответов на тестовой выборке.

model = Perceptron(random_state=241)
model.fit(X_train_scaled, y_train_new)
acc_after = accuracy_score(y_test_new, model.predict(X_test_scaled))
print(2, acc_after)

# 6. Найдите разность между качеством на тестовой выборке после нормализации и качеством до нее.
# Это число и будет ответом на задание.

print(3, acc_after - acc_before)







