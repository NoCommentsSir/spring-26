# Лабораторная работа №1. Логическая классификация

В рамках лабораторной работы предстоит реализовать алгоритм построения бинарного решающего дерева и сравнить его с эталонной реализацией.

На лекции были рассмотрены следующие алгоритмы:
* алгоритм построения бинарного решающего дерева ID3;
* алгоритм редукции дерева;
* алгоритм бинаризации вещественного признака;

## Задание

1. выбрать датасет для классификации, например на [kaggle](https://www.kaggle.com/datasets?tags=13302-Classification);
   1. датасет должен содержать пропуски;
   2. датасет должен содержать категориальные и количественные признаки;
2. реализовать алгоритм построения дерева ID3 с критерием Джини;
3. реализовать обработку пропущенных значений через оценку вероятности;
4. обучить дерево на выбранном датасете;
5. оценить качество классификации;
6. реализовать алгоритм редукции дерева;
7. сравнить качество классификации и регрессии до и после редукции дерева;
8. сравнить с [эталонной](https://scikit-learn.org/stable/) реализацией бинарного решающего дерева;
    1. сравнить качество работы;
9. подготовить небольшой отчет о проделанной работе.


## Фрагменты кода

Искусственные пропуски (`data/datasets/breast_cancer.py`):

```python
mask = rng.random(X.shape) < miss_rate
X = X.mask(mask)
self.df[X.columns] = X
```

Прунинг своего дерева (`model/tree.py`):

```python
class ReducedErrorPruner:
    def __init__(self, X_val: np.ndarray, y_val: np.ndarray):
        self.X_val = np.asarray(X_val)
        self.y_val = np.asarray(y_val)

    def prune(self, tree: DecisionTree) -> DecisionTree:
        t = _copy_tree(tree)
        t.root = _prune(t.root, self.X_val, self.y_val)
        return t
```

Подбор `ccp_alpha` для sklearn (`main.py`):

```python
path = sk.cost_complexity_pruning_path(X_tr, y_tr)
for a in path.ccp_alphas:
    clf = SklearnTree(**sk_params, ccp_alpha=float(a))
    clf.fit(X_tr, y_tr)
    va_acc = sklearn_accuracy(y_va, clf.predict(X_va))
    ...
sk_p = SklearnTree(**sk_params, ccp_alpha=best_alpha)
sk_p.fit(X_tr, y_tr)
```



## Результаты

```
=== Датасет breast_cancer: 569 x 30 ===
Классы: [212 357], NaN в X (после preprocess): 1029
NaN после масштабирования: train=552, val=272, test=205

=== Своё дерево: глубина 6, листьев 18 ===

Train
--------------------------------------------------
Accuracy:  0.9490
Precision: 0.9310
Recall:    0.9895
F1:        0.9594
Confusion matrix:
 [[109  14]
 [  2 189]]

Test
--------------------------------------------------
Accuracy:  0.8673
Precision: 0.8889
Recall:    0.9014
F1:        0.8951
Confusion matrix:
 [[34  8]
 [ 7 64]]

=== После прунинга: глубина 3, листьев 5 ===

Train
--------------------------------------------------
Accuracy:  0.9268
Precision: 0.9615
Recall:    0.9162
F1:        0.9383
Confusion matrix:
 [[116   7]
 [ 16 175]]

Test
--------------------------------------------------
Accuracy:  0.9469
Precision: 0.9851
Recall:    0.9296
F1:        0.9565
Confusion matrix:
 [[41  1]
 [ 5 66]]

=== Sklearn (без прунинга): глубина 6, листьев 17 ===

Sklearn train
--------------------------------------------------
Accuracy:  1.0000
Precision: 1.0000
Recall:    1.0000
F1:        1.0000
Confusion matrix:
 [[123   0]
 [  0 191]]

Sklearn test
--------------------------------------------------
Accuracy:  0.9381
Precision: 0.9444
Recall:    0.9577
F1:        0.9510
Confusion matrix:
 [[38  4]
 [ 3 68]]

=== Sklearn + CCP (ccp_alpha=0.009908, val acc=0.9225): глубина 3, листьев 7 ===

Sklearn+CCP train
--------------------------------------------------
Accuracy:  0.9682
Precision: 0.9689
Recall:    0.9791
F1:        0.9740
Confusion matrix:
 [[117   6]
 [  4 187]]

Sklearn+CCP test
--------------------------------------------------
Accuracy:  0.9646
Precision: 0.9718
Recall:    0.9718
F1:        0.9718
Confusion matrix:
 [[40  2]
 [ 2 69]]

=== Сводка (test) ===
  Своё (test)            acc=0.8673  F1=0.8951
  Прунинг, своё (test)   acc=0.9469  F1=0.9565
  Sklearn (test)         acc=0.9381  F1=0.9510
  Sklearn+CCP (test)     acc=0.9646  F1=0.9718
```
