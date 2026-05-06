import numpy as np


def calculate_gini_impurity(labels: np.ndarray) -> float:
    """
    Вычисляет неоднородность Джини для набора меток.

    Parameters:
        labels (numpy.ndarray): Массив меток классов. По умолчанию: нет.

    Returns:
        float: Значение неоднородности Джини.

    Fallbacks:
        Для пустого массива возвращается 0.
    """
    # Пустой узел не содержит неопределенности.
    if len(labels) == 0:
        return 0

    # Считаем долю каждого класса в переданном наборе.
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)

    return 1 - np.sum(probabilities**2)


def is_numeric_column(values: np.ndarray) -> bool:
    """
    Проверяет, можно ли привести значения признака к числовому типу.

    Parameters:
        values (numpy.ndarray): Значения одного признака. По умолчанию: нет.

    Returns:
        bool: True, если признак числовой, иначе False.

    Fallbacks:
        При ошибке приведения типа возвращается False.
    """
    # Пробное приведение отделяет количественные признаки от категориальных.
    try:
        values.astype(float)
    except (ValueError, TypeError):
        return False

    return True


def calculate_gini_from_counts(counts: np.ndarray) -> float:
    """
    Вычисляет неоднородность Джини по количествам классов.

    Parameters:
        counts (numpy.ndarray): Количество объектов каждого класса. По умолчанию: нет.

    Returns:
        float: Значение неоднородности Джини.

    Fallbacks:
        Если суммарное количество объектов равно 0, возвращается 0.
    """
    # Пустое разбиение не влияет на итоговую оценку.
    total = counts.sum()
    if total == 0:
        return 0

    # Переводим количества в вероятности классов.
    probabilities = counts / total

    return 1 - np.sum(probabilities**2)


def calculate_numeric_information_gain(
    feature_values: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float | None]:
    """
    Находит лучший порог и прирост информации для числового признака.

    Parameters:
        feature_values (numpy.ndarray): Значения числового признака. По умолчанию: нет.
        labels (numpy.ndarray): Метки классов для объектов. По умолчанию: нет.

    Returns:
        tuple[float, float | None]: Лучший прирост информации и найденный порог.

    Fallbacks:
        Если признак не дает разбиения, возвращаются 0 и None.
    """
    # Сортируем объекты по значению признака для перебора соседних порогов.
    numeric_values = feature_values.astype(float)
    order = np.argsort(numeric_values)
    sorted_values = numeric_values[order]
    sorted_labels = labels[order]

    # Один уникальный уровень признака не может улучшить дерево.
    if len(np.unique(sorted_values)) <= 1:
        return 0, None

    # Кодируем классы в индексы для быстрых векторных подсчетов.
    _, encoded_labels = np.unique(sorted_labels, return_inverse=True)
    total_counts = np.bincount(encoded_labels)
    left_counts = np.zeros_like(total_counts)
    parent_gini = calculate_gini_from_counts(total_counts)

    # Инициализируем лучший найденный порог.
    best_gain = 0
    best_threshold = None

    # Перебираем границы между разными соседними значениями признака.
    for index in range(len(sorted_values) - 1):
        left_counts[encoded_labels[index]] += 1

        if sorted_values[index] == sorted_values[index + 1]:
            continue

        # Считаем взвешенную неоднородность двух дочерних узлов.
        right_counts = total_counts - left_counts
        left_weight = left_counts.sum() / len(labels)
        right_weight = right_counts.sum() / len(labels)
        weighted_child_gini = (
            left_weight * calculate_gini_from_counts(left_counts)
            + right_weight * calculate_gini_from_counts(right_counts)
        )
        gain = parent_gini - weighted_child_gini

        # Обновляем лучший порог при улучшении прироста информации.
        if gain > best_gain:
            best_gain = gain
            best_threshold = (sorted_values[index] + sorted_values[index + 1]) / 2

    return best_gain, best_threshold


def calculate_information_gain(
    features: np.ndarray,
    labels: np.ndarray,
    feature_index: int,
) -> tuple[float, float | None, bool]:
    """
    Вычисляет прирост информации для выбранного признака.

    Parameters:
        features (numpy.ndarray): Матрица признаков. По умолчанию: нет.
        labels (numpy.ndarray): Метки классов. По умолчанию: нет.
        feature_index (int): Индекс проверяемого признака. По умолчанию: нет.

    Returns:
        tuple[float, float | None, bool]: Прирост, порог и флаг числового признака.

    Fallbacks:
        Для категориальных признаков порог возвращается как None.
    """
    # Извлекаем столбец признака из матрицы объектов.
    feature_values = features[:, feature_index]

    # Для числового признака ищем бинарное разбиение по порогу.
    if is_numeric_column(feature_values):
        gain, threshold = calculate_numeric_information_gain(feature_values, labels)
        return gain, threshold, True

    # Для категориального признака считаем взвешенную сумму по его значениям.
    parent_gini = calculate_gini_impurity(labels)
    values, counts = np.unique(feature_values, return_counts=True)
    weighted_child_gini = 0

    # Суммируем вклад каждого дочернего узла.
    for value, count in zip(values, counts):
        subset_labels = labels[feature_values == value]
        weighted_child_gini += (
            count / len(labels)
        ) * calculate_gini_impurity(subset_labels)

    return parent_gini - weighted_child_gini, None, False
