import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from data_loader import load_archive, load_df
from models import DecisionTreeClassifier


def main() -> None:
    """
    Загружает данные, обучает дерево решений и выводит accuracy.

    Parameters:
        None: Функция не принимает параметры. По умолчанию: None.

    Returns:
        None: Результат оценки выводится в консоль.

    Fallbacks:
        Ошибки загрузки данных и обучения передаются вызывающему коду.
    """
    # Загружаем архив и читаем таблицу с датасетом.
    load_archive()
    dataframe = load_df()

    # Разделяем признаки и целевую переменную по последнему столбцу.
    feature_names = dataframe.columns[:-1]
    label_name = dataframe.columns[-1]

    # Формируем стратифицированные обучающую и тестовую выборки.
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(dataframe[feature_names]),
        np.array(dataframe[label_name]),
        test_size=0.2,
        random_state=42,
        stratify=dataframe[label_name],
    )

    # Обучаем собственную реализацию дерева решений.
    tree_classifier = DecisionTreeClassifier(
        features=x_train,
        feature_names=feature_names,
        labels=y_train,
    )
    tree_classifier.id3()

    # Оцениваем качество классификации на тестовой выборке.
    y_predicted = tree_classifier.predict(x_test)
    print("Accuracy:", accuracy_score(y_test, y_predicted))


# Запускаем сценарий только при прямом вызове файла.
if __name__ == "__main__":
    main()
