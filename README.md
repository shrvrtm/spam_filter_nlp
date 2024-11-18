# Спам-Фильтр (NLP)

Этот проект реализует фильтр спама с использованием методов обработки естественного языка (NLP). Программа классифицирует текстовые сообщения на категории спам или не спам (ham). Проект разработан на Python с использованием библиотек scikit-learn, pandas и joblib.

## Возможности:

1. Предобработка данных: Загрузка и подготовка датасетов (например, spam.csv) для задач классификации.
2. Векторизация текста: Преобразование текстовых сообщений в числовые признаки с помощью TfidfVectorizer.
3. Обучение модели: Логистическая регрессия используется для обучения модели классификации.
4. Предсказание: Определение, является ли сообщение спамом или нет.
5. Работа с файлами:
   - Чтение сообщений из файла input_output/input.txt.
   - Запись классифицированных сообщений в файл input_output/output.txt.
6. Сохранение и загрузка модели: Сохранение обученной модели и векторизатора для дальнейшего использования.

## Структора проекта:

  ```plaintext
    spam_filter/
├── data/               # Папка для хранения датасетов (например, spam.csv)
├── input_output/       # Папка для ввода и вывода данных
│   ├── input.txt       # Сообщения для классификации
│   └── output.txt      # Отфильтрованные сообщения
├── models/             # Папка для сохранения модели
│   ├── spam_filter_model.pkl
│   └── tfidf_vectorizer.pkl
├── src/                # Исходный код
│   ├── data_preparation.py  # Загрузка и предобработка данных
│   ├── text_features.py     # Преобразование текста в признаки
│   ├── train_model.py       # Обучение, сохранение и загрузка модели
│   └── predict.py           # Функции предсказания
└── main.py            # Основной файл для запуска проекта
└── README.md          # Описание проекта
   ```
## Установка

1. Клонируйте проект или создайте структуру папок вручную.
2. Установите зависимости:
    ```bash
   pip install numpy pandas scikit-learn joblib
   ```
3. Скачайте датасет (например, SMS Spam Collection Dataset) и сохраните в папке data/.

## Использование

1. В файле input_output/input.txt добавьте сообщения для классификации.
2. Запустите файл main.py:
    ```bash
   python main.py
   ```
3. Результаты будут сохранены в input_output/output.txt.

## Дополнительная информация

- TfidfVectorizer: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
- Логистическая регрессия: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
- Метрики классификации: https://scikit-learn.org/stable/modules/model_evaluation.html