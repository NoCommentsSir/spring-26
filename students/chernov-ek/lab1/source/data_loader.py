from pathlib import Path
import zipfile

import pandas as pd
import requests


# Базовые пути проекта и архива с датасетом.
ROOT_PATH = Path(__file__).parent.parent
DATA_PATH = ROOT_PATH / "data"
ARCHIVE_NAME = "diabetes-prediction-dataset.zip"
DATASET_FILE_NAME = "diabetes_prediction_dataset.csv"
DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "iammustafatz/diabetes-prediction-dataset"
)


def load_archive() -> None:
    """
    Загружает архив с датасетом, если он отсутствует локально.

    Parameters:
        None: Функция не принимает параметры. По умолчанию: None.

    Returns:
        None: Архив сохраняется в директорию data.

    Fallbacks:
        Если архив уже существует, загрузка пропускается.
    """
    # Создаем директорию с данными перед сохранением архива.
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    output_file = DATA_PATH / ARCHIVE_NAME

    # Не загружаем файл повторно, если он уже есть на диске.
    if output_file.exists():
        return

    # Получаем архив потоково, чтобы не держать весь файл в памяти.
    response = requests.get(DATASET_URL, stream=True, allow_redirects=True, timeout=60)
    response.raise_for_status()

    # Сохраняем архив порциями фиксированного размера.
    with output_file.open("wb") as archive_file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                archive_file.write(chunk)


def load_df() -> pd.DataFrame:
    """
    Читает CSV-файл с датасетом из локального ZIP-архива.

    Parameters:
        None: Функция не принимает параметры. По умолчанию: None.

    Returns:
        pandas.DataFrame: Таблица с признаками и целевой переменной.

    Fallbacks:
        Если архив отсутствует или поврежден, исключение передается вызывающему коду.
    """
    # Открываем архив и читаем CSV без промежуточной распаковки на диск.
    archive_path = DATA_PATH / ARCHIVE_NAME
    with zipfile.ZipFile(archive_path, "r") as archive:
        with archive.open(DATASET_FILE_NAME) as dataset_file:
            dataframe = pd.read_csv(dataset_file)

    return dataframe
