import pandas
import re

data = pandas.read_csv('../source/titanic.csv', index_col='PassengerId')

# 1 Какое количество мужчин и женщин ехало на корабле? В качестве ответа приведите два числа через пробел.
sex = data['Sex'].value_counts()
print(sex)
# male      577
# female    314

# 2 Какой части пассажиров удалось выжить? Посчитайте долю выживших пассажиров.
# Ответ приведите в процентах (число в интервале от 0 до 100, знак процента не нужен), округлив до двух знаков.
survived = data['Survived'].value_counts('1')
print(survived)
# 0    0.616162
# 1    0.383838

surv_counts = data['Survived'].value_counts()
surv_percent = 100.0 * surv_counts[1] / surv_counts.sum()
print(2, "{:0.2f}".format(surv_percent))

# 3 Какую долю пассажиры первого класса составляли среди всех пассажиров?
# Ответ приведите в процентах (число в интервале от 0 до 100, знак процента не нужен),
# округлив до двух знаков.
Pclass = data['Pclass'].value_counts('1')
print(Pclass)
# 3    0.551066
# 1    0.242424
# 2    0.206510

pclass_counts = data['Pclass'].value_counts()
pclass_percent = 100.0 * pclass_counts[1] / pclass_counts.sum()
print(3, "{:0.2f}".format(pclass_percent))

# 4 # # # Какого возраста были пассажиры? Посчитайте среднее и медиану возраста пассажиров.
# В качестве ответа приведите два числа через пробел.
midleMean = data['Age'].mean()
midleMedian = data['Age'].median()
print(midleMean)
print(midleMedian)
# 29.69911764705882
# 28.0

# 5 # # # # Коррелируют ли число братьев/сестер/супругов с числом родителей/детей?
# Посчитайте корреляцию Пирсона между признаками SibSp и Parch.
sibSp = data['SibSp']
corrilation = data['Parch'].corr(sibSp)
print(corrilation)
# 0.4148376986201565

# 6 # # # # # Какое самое популярное женское имя на корабле?
# Извлеките из полного имени пассажира (колонка Name) его личное имя (First Name).
# Это задание — типичный пример того, с чем сталкивается специалист по анализу данных.
# Данные очень разнородные и шумные, но из них требуется извлечь необходимую информацию.
# Попробуйте вручную разобрать несколько значений столбца Name и выработать правило для извлечения имен,
# а также разделения их на женские и мужские.
fn = data[data['Sex'] == 'female']['Name']

def extract_first_name(name):

    # первое слово в скобках
    m = re.search(".*\\((.*)\\).*", name)
    if m is not None:
        return m.group(1).split(" ")[0]
    # первое слово после Mrs. or Miss. or else
    m1 = re.search(".*\\. ([A-Za-z]*)", name)
    return m1.group(1)

# получаем имя с максимальной частотой
r = fn.map(lambda full_name: extract_first_name(full_name)).value_counts().idxmax()
print(r)

# Anna