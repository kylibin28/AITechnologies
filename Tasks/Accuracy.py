import pandas
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
import sklearn.metrics as metrics


# 1. Загрузите файл classification.csv. В нем записаны истинные классы объектов выборки
# (колонка true) и ответы некоторого классификатора (колонка pred).
trainData = pandas.read_csv('../source/classification.csv')
true_ansvers = trainData['true']
pred = trainData['pred']

# print(trainData)

# 2. Для этого подсчитайте величины TP, FP, FN и TN согласно их определениям.
# Например, FP — это количество объектов, имеющих класс 0, но отнесенных
# алгоритмом к классу 1. Ответ в данном вопросе — четыре числа через пробел.

clf_table = {'tp': (1, 1), 'fp': (0, 1), 'fn': (1, 0), 'tn': (0, 0)}
for name, res in clf_table.items():
    clf_table[name] = len(trainData[(trainData['true'] == res[0]) & (trainData['pred'] == res[1])])

print(1, '{tp} {fp} {fn} {tn}'.format(**clf_table))

# 3. 3. Посчитайте основные метрики качества классификатора:
#
# Accuracy (доля верно угаданных) — sklearn.metrics.accuracy_score
# Precision (точность) — sklearn.metrics.precision_score
# Recall (полнота) — sklearn.metrics.recall_score
# F-мера — sklearn.metrics.f1_score
# В качестве ответа укажите эти четыре числа через пробел.

print("#2")
print("accuracy_score=", accuracy_score(true_ansvers, pred))
print("precision_score=", precision_score(true_ansvers, pred))
print("recall_score=", recall_score(true_ansvers, pred))
print("f1_score=", f1_score(true_ansvers, pred))

# 4. Имеется четыре обученных классификатора. В файле scores.csv записаны истинные классы
# и значения степени принадлежности положительному классу для каждого классификатора на некоторой выборке:
#
# для логистической регрессии — вероятность положительного класса (колонка score_logreg),
# для SVM — отступ от разделяющей поверхности (колонка score_svm),
# для метрического алгоритма — взвешенная сумма классов соседей (колонка score_knn),
# для решающего дерева — доля положительных объектов в листе (колонка score_tree).
# Загрузите этот файл.

scoresData = pandas.read_csv('../source/scores.csv')
score_true = scoresData['true']
score_logreg = scoresData['score_logreg']
score_svm = scoresData['score_svm']
score_knn = scoresData['score_knn']
score_tree = scoresData['score_tree']

# print("scoresData=", scoresData)
# print("score_true=", score_true)
# print("score_logreg=", score_logreg)

# 5. Посчитайте площадь под ROC-кривой для каждого классификатора.
# Какой классификатор имеет наибольшее значение метрики AUC-ROC (укажите название столбца)?
# Воспользуйтесь функцией sklearn.metrics.roc_auc_score.

scores = {}
for clf in scoresData.columns[1:]:
    scores[clf] = metrics.roc_auc_score(scoresData['true'], scoresData[clf])

print(3, pandas.Series(scores).sort_values(ascending=False).head(1).index[0])

# 6. Какой классификатор достигает наибольшей точности (Precision) при полноте (Recall) не менее 70% ?
# Чтобы получить ответ на этот вопрос, найдите все точки precision-recall-кривой с помощью функции
# sklearn.metrics.precision_recall_curve. Она возвращает три массива: precision, recall, thresholds.
# В них записаны точность и полнота при определенных порогах, указанных в массиве thresholds.
# Найдите максимальной значение точности среди тех записей, для которых полнота не меньше, чем 0.7.

# precision_logreg, recall_logreg, thresholds_logreg = precision_recall_curve(score_true, score_logreg)
# precision_svm, recall_svm, thresholds_svm = precision_recall_curve(score_true, score_svm)
# precision_knn, recall_knn, thresholds_knn = precision_recall_curve(score_true, score_knn)
# precision_tree, recall_tree, thresholds_tree = precision_recall_curve(score_true, score_tree)
#
# print("#6")
# print("recall_logreg=", recall_logreg)
# print("recall_svm=", recall_svm)
# print("recall_knn=", recall_knn)
# print("recall_tree=", recall_tree)

pr_scores = {}
for clf in scoresData.columns[1:]:
    pr_curve = metrics.precision_recall_curve(scoresData['true'], scoresData[clf])
    pr_curve_df = pandas.DataFrame({'precision': pr_curve[0], 'recall': pr_curve[1]})
    pr_scores[clf] = pr_curve_df[pr_curve_df['recall'] >= 0.7]['precision'].max()

print(4, pandas.Series(pr_scores).sort_values(ascending=False).head(1).index[0])