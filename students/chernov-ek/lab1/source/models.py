from collections import deque

import numpy as np

from evaluators import calculate_gini_impurity, calculate_information_gain


class Node:
    """
    Хранит данные одного узла дерева решений.

    Attributes:
        value (str | None): Значение узла или метка ребра. По умолчанию: None.
        next (Node | None): Следующий узел для ребра. По умолчанию: None.
        children (list[Node] | None): Дочерние ребра узла. По умолчанию: None.
        feature_id (int | None): Индекс признака для разбиения. По умолчанию: None.
        threshold (float | None): Порог для числового признака. По умолчанию: None.
        is_numeric (bool): Флаг числового признака. По умолчанию: False.
        prediction (int | None): Прогноз большинства классов в узле. По умолчанию: None.

    Fallbacks:
        Узел может быть листом, если children равно None.
    """

    def __init__(self) -> None:
        """
        Инициализирует пустой узел дерева решений.

        Parameters:
            None: Функция не принимает параметры. По умолчанию: None.

        Returns:
            None: Атрибуты узла заполняются начальными значениями.

        Fallbacks:
            Все связи и значения остаются None до построения дерева.
        """
        # Значение хранит имя признака в узле или класс в листе.
        self.value: str | None = None

        # Следующий узел используется для представления ребра дерева.
        self.next: Node | None = None

        # Дочерние элементы содержат ребра от текущего узла.
        self.children: list[Node] | None = None

        # Параметры разбиения заполняются для внутренних узлов.
        self.feature_id: int | None = None
        self.threshold: float | None = None
        self.is_numeric = False

        # Прогноз большинства нужен для неизвестных ветвей при предсказании.
        self.prediction: int | None = None


class DecisionTreeClassifier:
    """
    Реализует классификатор дерева решений с ID3-подобным алгоритмом.

    Attributes:
        features (numpy.ndarray): Матрица признаков обучающей выборки. По умолчанию: нет.
        feature_names (list[str]): Названия признаков. По умолчанию: нет.
        labels (numpy.ndarray): Метки классов обучающей выборки. По умолчанию: нет.
        node (Node | None): Корневой узел дерева. По умолчанию: None.
        gini (float): Неоднородность Джини исходной выборки. По умолчанию: вычисляется.

    Fallbacks:
        При неизвестной ветке используется прогноз большинства классов текущего узла.
    """

    def __init__(
        self,
        features: np.ndarray,
        feature_names: list[str],
        labels: np.ndarray,
    ) -> None:
        """
        Сохраняет обучающую выборку и начальные параметры дерева.

        Parameters:
            features (numpy.ndarray): Матрица признаков. По умолчанию: нет.
            feature_names (list[str]): Названия признаков. По умолчанию: нет.
            labels (numpy.ndarray): Метки классов. По умолчанию: нет.

        Returns:
            None: Экземпляр классификатора получает начальное состояние.

        Fallbacks:
            Корневой узел остается None до вызова id3().
        """
        # Сохраняем обучающие данные в виде массивов NumPy.
        self.features = np.array(features)
        self.feature_names = list(feature_names)
        self.labels = np.array(labels)

        # Корень дерева создается во время обучения.
        self.node: Node | None = None
        self.gini = calculate_gini_impurity(self.labels)

    def _majority_class(self, labels: np.ndarray) -> int:
        """
        Возвращает самый частый класс в наборе меток.

        Parameters:
            labels (numpy.ndarray): Метки классов. По умолчанию: нет.

        Returns:
            int: Метка класса с максимальной частотой.

        Fallbacks:
            При равенстве частот выбирается первый класс после сортировки np.unique.
        """
        # Считаем частоты классов и берем класс с максимальным количеством.
        values, counts = np.unique(labels, return_counts=True)
        return int(values[np.argmax(counts)])

    def _get_feature_max_information_gain(
        self,
        row_ids: list[int],
        feature_ids: list[int],
    ) -> tuple[str, int, float | None, bool] | None:
        """
        Находит признак с максимальным приростом информации.

        Parameters:
            row_ids (list[int]): Индексы объектов текущего узла. По умолчанию: нет.
            feature_ids (list[int]): Индексы доступных признаков. По умолчанию: нет.

        Returns:
            tuple[str, int, float | None, bool] | None: Описание лучшего признака.

        Fallbacks:
            Если список признаков пуст, возвращается None.
        """
        # Инициализируем лучший признак отсутствующим значением.
        best_feature = None
        best_gain = -1

        # Проверяем каждый доступный признак на текущем подмножестве строк.
        for feature_id in feature_ids:
            gain, threshold, is_numeric = calculate_information_gain(
                self.features[row_ids],
                self.labels[row_ids],
                feature_id,
            )

            # Запоминаем признак, который дает максимальное улучшение.
            if gain > best_gain:
                best_gain = gain
                best_feature = (
                    str(self.feature_names[feature_id]),
                    feature_id,
                    threshold,
                    is_numeric,
                )

        return best_feature

    def id3(self) -> None:
        """
        Строит дерево решений по алгоритму ID3 с критерием Джини.

        Parameters:
            None: Функция не принимает параметры. По умолчанию: None.

        Returns:
            None: Корневой узел сохраняется в self.node.

        Fallbacks:
            Если признаки не дают разбиения, дерево завершается листом большинства.
        """
        # Начинаем построение со всех строк и всех признаков.
        row_ids = list(range(len(self.features)))
        feature_ids = list(range(len(self.feature_names)))
        self.node = self._id3_recursive(row_ids, feature_ids, self.node)

    def _id3_recursive(
        self,
        row_ids: list[int],
        feature_ids: list[int],
        node: Node | None,
    ) -> Node:
        """
        Рекурсивно строит поддерево для выбранных объектов и признаков.

        Parameters:
            row_ids (list[int]): Индексы объектов в текущем узле. По умолчанию: нет.
            feature_ids (list[int]): Индексы доступных признаков. По умолчанию: нет.
            node (Node | None): Узел для заполнения. По умолчанию: None.

        Returns:
            Node: Заполненный узел дерева.

        Fallbacks:
            При чистом узле, отсутствии признаков или порога создается лист.
        """
        # Создаем узел, если рекурсивный вызов получил пустую ссылку.
        if node is None:
            node = Node()

        # Запоминаем класс большинства для листа или резервного предсказания.
        labels_in_node = self.labels[row_ids]
        node.prediction = self._majority_class(labels_in_node)

        # Останавливаем построение, если все объекты имеют один класс.
        if len(set(labels_in_node)) == 1:
            node.value = str(labels_in_node[0])
            return node

        # Останавливаем построение, если признаки закончились.
        if len(feature_ids) == 0:
            node.value = str(node.prediction)
            return node

        # Выбираем лучший признак для разбиения текущего узла.
        best_feature = self._get_feature_max_information_gain(row_ids, feature_ids)
        if best_feature is None:
            node.value = str(node.prediction)
            return node

        # Записываем параметры выбранного признака во внутренний узел.
        best_feature_name, best_feature_id, threshold, is_numeric = best_feature
        node.value = best_feature_name
        node.feature_id = best_feature_id
        node.threshold = threshold
        node.is_numeric = is_numeric
        node.children = []

        # Исключаем использованный признак для дочерних разбиений.
        next_feature_ids = [
            feature_id for feature_id in feature_ids if feature_id != best_feature_id
        ]

        # Формируем бинарные ветки для числового признака.
        if is_numeric:
            if threshold is None:
                node.value = str(node.prediction)
                node.children = None
                return node

            numeric_values = self.features[row_ids, best_feature_id].astype(float)
            splits = [
                (
                    f"<= {threshold:.4f}",
                    [
                        row_id
                        for row_id, value in zip(row_ids, numeric_values)
                        if value <= threshold
                    ],
                ),
                (
                    f"> {threshold:.4f}",
                    [
                        row_id
                        for row_id, value in zip(row_ids, numeric_values)
                        if value > threshold
                    ],
                ),
            ]
        else:
            # Формируем ветку для каждого значения категориального признака.
            feature_values = self.features[row_ids, best_feature_id]
            splits = [
                (
                    str(value),
                    [
                        row_id
                        for row_id in row_ids
                        if self.features[row_id][best_feature_id] == value
                    ],
                )
                for value in np.unique(feature_values)
            ]

        # Рекурсивно строим дочерние узлы для каждого разбиения.
        for value, child_row_ids in splits:
            child = Node()
            child.value = str(value)
            node.children.append(child)

            # Пустая ветка получает лист с прогнозом большинства текущего узла.
            if not child_row_ids:
                child.next = Node()
                child.next.value = str(node.prediction)
                child.next.prediction = node.prediction
            else:
                child.next = self._id3_recursive(
                    child_row_ids,
                    next_feature_ids.copy(),
                    child.next,
                )

        return node

    def _predict_one(self, row: np.ndarray, node: Node) -> int | None:
        """
        Предсказывает класс для одного объекта.

        Parameters:
            row (numpy.ndarray): Значения признаков одного объекта. По умолчанию: нет.
            node (Node): Корневой или текущий узел дерева. По умолчанию: нет.

        Returns:
            int | None: Предсказанная метка класса.

        Fallbacks:
            Если подходящая ветка не найдена, возвращается прогноз большинства узла.
        """
        # Спускаемся по дереву, пока текущий узел имеет дочерние ветки.
        while node and node.children:
            if node.feature_id is None:
                return node.prediction

            if node.is_numeric:
                if node.threshold is None:
                    return node.prediction

                value = float(row[node.feature_id])
                branch_value = (
                    node.children[0].value
                    if value <= node.threshold
                    else node.children[1].value
                )
            else:
                branch_value = str(row[node.feature_id])

            # Ищем дочернюю ветку, соответствующую значению признака.
            next_node = None
            for child in node.children:
                if child.value == branch_value:
                    next_node = child.next
                    break

            # Возвращаем резервный прогноз при неизвестной категории.
            if next_node is None:
                return node.prediction

            node = next_node

        return node.prediction if node else None

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Предсказывает классы для одного или нескольких объектов.

        Parameters:
            features (numpy.ndarray): Матрица признаков или один объект. По умолчанию: нет.

        Returns:
            numpy.ndarray: Массив предсказанных меток классов.

        Fallbacks:
            Если дерево не обучено, выбрасывается ValueError.
        """
        # Проверяем, что дерево построено перед предсказанием.
        if self.node is None:
            raise ValueError("The tree has not been trained. Call id3() first.")

        # Приводим один объект к матрице из одной строки.
        features = np.array(features)
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Предсказываем класс независимо для каждой строки.
        return np.array([self._predict_one(row, self.node) for row in features])

    def print_tree(self) -> None:
        """
        Печатает значения узлов дерева в ширину.

        Parameters:
            None: Функция не принимает параметры. По умолчанию: None.

        Returns:
            None: Структура дерева выводится в консоль.

        Fallbacks:
            Если дерево не построено, функция завершает работу без вывода.
        """
        # Пустое дерево не содержит узлов для вывода.
        if not self.node:
            return

        # Обходим дерево в ширину с помощью очереди.
        nodes = deque([self.node])
        while nodes:
            node = nodes.popleft()
            print(node.value)

            # Печатаем значения ребер и добавляем дочерние узлы в очередь.
            if node.children:
                for child in node.children:
                    print(f"({child.value})")
                    nodes.append(child.next)
            elif node.next:
                print(node.next)
