# Подборка вин

Приложение оценивает вероятность экспертной оценки от 7 баллов. Один порог в боковой панели управляет решением для одного вина, метриками и пакетной обработкой CSV.

## Запуск

Проверено на Python 3.12. Выполни команды из папки wine_app.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Для пакетного прогноза нужен CSV в UTF-8 с 12 признаками из test.csv. Столбец target можно убрать. В is_red красное вино кодируется единицей, белое нулём.

## Кэширование

Кэш ресурсов хранит один загруженный пайплайн для всех перезапусков и сессий. Кэш данных хранит результат чтения таблицы и массив тестовых вероятностей, а вызывающая сторона получает копию. При смене порога приложение повторно использует вероятности и пересчитывает метки и метрики. Вызов joblib.load без кэша заново читал бы файл и восстанавливал пайплайн при каждом перезапуске скрипта.

## Как воспроизвести ноутбук

Из корня архива установи зависимости приложения и добавь пакеты для ноутбука в то же окружение.

```bash
python -m pip install -r wine_app/requirements.txt imbalanced-learn==0.13.0 seaborn==0.13.2 notebook
python -m notebook HW3_MMDA.ipynb
```

Датасет расположен в Data/wine_quality.csv. Последняя кодовая ячейка заново сохранит обученный пайплайн, порог и тестовую выборку в wine_app.

Для Streamlit Community Cloud укажи путь wine_app/app.py и выбери Python 3.12 в настройках приложения. Сервис берёт файлы из подключённого GitHub-репозитория. [Инструкция Streamlit](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)

Данные подготовлены из Wine Quality (P. Cortez et al., 2009, CC BY 4.0). [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine+quality)
