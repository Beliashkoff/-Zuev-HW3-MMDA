import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics


APP_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / 'model.joblib')


@st.cache_data
def load_test():
    return pd.read_csv(APP_DIR / 'test.csv')


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


def batch_predictions(model, batch, features, threshold):
    missing = [feature for feature in features if feature not in batch.columns]
    if missing:
        raise ValueError('Не хватает столбцов: ' + ', '.join(missing))
    if batch.empty:
        raise ValueError('В CSV нет строк с данными.')

    inputs = batch.loc[:, features].apply(pd.to_numeric, errors='coerce')
    if not np.isfinite(inputs.to_numpy(dtype=float)).all():
        raise ValueError('Заполни все признаки конечными числами.')
    if not inputs['is_red'].isin([0, 1]).all():
        raise ValueError('В столбце is_red допустимы только 0 (белое) и 1 (красное).')

    scores = model.predict_proba(inputs)[:, 1]
    return inputs.assign(
        probability_good=scores,
        recommendation=np.where(scores >= threshold, 'В подборку', 'Вне подборки'),
    )


def main():
    st.set_page_config(page_title='Подборка вин', layout='wide')
    st.title('Подборка вин')
    st.write('Модель оценивает вероятность того, что эксперты поставят вину 7 баллов или выше.')

    model = load_model()
    test = load_test()
    features = model.feature_names_in_.tolist()
    X_test = test.loc[:, features]
    y_test = test['target']
    scores = test_proba(model, X_test)
    config = json.loads((APP_DIR / 'config.json').read_text(encoding='utf-8'))
    threshold = st.sidebar.slider(
        'Порог рекомендации', 0.0, 1.0, config['threshold'], step=0.0001, format='%.4f'
    )

    single_tab, metrics_tab, batch_tab, weights_tab = st.tabs(
        ['Одно вино', 'Модельные метрики', 'Пакетный прогноз', 'Коэффициенты']
    )

    with single_tab:
        fields = st.columns(3)
        values = {}
        for index, feature in enumerate(features):
            with fields[index % len(fields)]:
                if feature == 'is_red':
                    color = st.radio(
                        'Цвет вина', ['Белое', 'Красное'],
                        index=int(X_test[feature].median()), horizontal=True,
                    )
                    values[feature] = int(color == 'Красное')
                else:
                    column = X_test[feature]
                    values[feature] = st.number_input(
                        feature,
                        min_value=float(column.min()),
                        max_value=float(column.max()),
                        value=float(column.median()),
                        format='%.6f',
                    )

        example = pd.DataFrame([values], columns=features)
        probability = model.predict_proba(example)[0, 1]
        st.metric('Вероятность высокой оценки', f'{probability:.1%}')
        st.write('В подборку' if probability >= threshold else 'Вне подборки')

    with metrics_tab:
        predictions = scores >= threshold
        matrix = metrics.confusion_matrix(y_test, predictions, labels=[0, 1])
        tn, fp, fn, tp = matrix.ravel()
        precision = metrics.precision_score(y_test, predictions, zero_division=0)
        recall = metrics.recall_score(y_test, predictions, zero_division=0)
        f1 = metrics.f1_score(y_test, predictions, zero_division=0)
        p_column, r_column, f_column = st.columns(3)
        p_column.metric('Precision', f'{precision:.3f}')
        r_column.metric('Recall', f'{recall:.3f}')
        f_column.metric('F1', f'{f1:.3f}')
        st.dataframe(pd.DataFrame(
            matrix, index=['Факт 0', 'Факт 1'], columns=['Прогноз 0', 'Прогноз 1']
        ))

        fpr, tpr, _ = metrics.roc_curve(y_test, scores, drop_intermediate=False)
        pr_precision, pr_recall, _ = metrics.precision_recall_curve(y_test, scores)
        fig, (roc_ax, pr_ax) = plt.subplots(1, 2, figsize=(10, 4))
        roc_ax.plot(fpr, tpr)
        roc_ax.plot([0, 1], [0, 1], '--', color='gray')
        roc_ax.scatter(fp / (fp + tn), recall, color='tab:orange', zorder=3)
        roc_ax.set(xlabel='FPR', ylabel='TPR', title='ROC')
        pr_ax.step(pr_recall, pr_precision, where='post')
        pr_ax.axhline(y_test.mean(), linestyle='--', color='gray')
        pr_ax.scatter(recall, precision if tp + fp else 1.0, color='tab:orange', zorder=3)
        pr_ax.set(xlabel='Recall', ylabel='Precision', title='Precision–Recall')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        if tp + fp == 0:
            st.caption('При этом пороге подборка пуста. На PR-кривой отмечена конечная точка (0, 1).')
        st.write(
            'При повышении порога подборка становится строже и в неё попадает меньше вин. '
            'При снижении порога можно найти больше хороших вин, но в подборке могут '
            'чаще встречаться вина с более низкой оценкой.'
        )

    with batch_tab:
        uploaded = st.file_uploader('Загрузи CSV с признаками вин', type='csv')
        st.caption('Столбец target для прогноза не требуется. Для is_red используй 1 или 0.')
        if uploaded is not None:
            try:
                batch = pd.read_csv(uploaded)
                if not isinstance(batch.index, pd.RangeIndex):
                    raise ValueError('В CSV число полей в строке превышает число заголовков.')
                result = batch_predictions(model, batch, features, threshold)
            except pd.errors.EmptyDataError:
                st.error('CSV пуст. Добавь заголовки и хотя бы одну строку.')
            except pd.errors.ParserError:
                st.error('Не удалось разобрать CSV. Проверь разделители и число полей.')
            except UnicodeDecodeError:
                st.error('Сохрани CSV в кодировке UTF-8.')
            except ValueError as error:
                st.error(str(error))
            else:
                st.dataframe(result, hide_index=True)
                st.download_button(
                    'Скачать прогнозы', result.to_csv(index=False).encode('utf-8'),
                    file_name='wine_predictions.csv', mime='text/csv',
                )

    with weights_tab:
        classifier = model.named_steps['logisticregression']
        coefficients = pd.Series(classifier.coef_[0], index=features).sort_values()
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = np.where(coefficients > 0, 'tab:blue', 'tab:orange')
        ax.barh(coefficients.index, coefficients, color=colors)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set(xlabel='Коэффициент после стандартизации')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        for label, selection in (
            ('За высокую оценку', coefficients[coefficients > 0].tail(3).iloc[::-1]),
            ('Против высокой оценки', coefficients[coefficients < 0].head(3)),
        ):
            st.write(label + ' — ' + ', '.join(f'{name} ({value:+.2f})' for name, value in selection.items()))
        st.caption('Коэффициенты описывают вклад признаков в прогноз модели при прочих равных.')


if __name__ == '__main__':
    main()
