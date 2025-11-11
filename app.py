import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_model_resources
import io
import os

# Настройка страницы
st.set_page_config(
    page_title="Essay Grading System - Case Solution",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Загрузка модели
@st.cache_resource
def load_model():
    return load_model_resources()

def main():
    st.title("🎯 Система оценки экзаменационных ответов")
    st.markdown("""
    Аналог Colab решения для автоматической оценки ответов на экзаменационные вопросы.
    Загрузите CSV файл с транскрибациями ответов для получения оценок.
    """)
    
    # Информация о системе
    with st.sidebar:
        st.header("📋 О системе")
        st.markdown("""
        **Критерии оценки:**
        - Вопрос 1: 0-1 балл
        - Вопрос 2: 0-2 балла  
        - Вопрос 3: 0-1 балл
        - Вопрос 4: 0-2 балла
        
        **Требования к файлу:**
        - CSV формат
        - Колонка с транскрибациями ответов
        - Колонка с номерами вопросов (1-4)
        """)
    
    # Загрузка модели
    with st.spinner("Загрузка модели для оценки..."):
        grader = load_model()
    
    if grader is None:
        st.error("❌ Ошибка загрузки модели. Проверьте файлы модели в папке 'my_trained_model_2'")
        return
    
    st.success("✅ Модель успешно загружена!")
    
    # Основной интерфейс
    st.header("📤 Загрузка данных")
    
    uploaded_file = st.file_uploader(
        "Выберите CSV файл с экзаменационными данными", 
        type=['csv'],
        help="Файл должен содержать колонки с транскрибациями ответов и номерами вопросов"
    )
    
    if uploaded_file is not None:
        try:
            # Чтение файла
            df = pd.read_csv(uploaded_file, delimiter=';', encoding='utf-8')
            st.success(f"Файл загружен! Найдено {len(df)} записей")
            
            # Показ структуры данных
            st.subheader("Структура данных")
            st.write(f"**Колонки:** {list(df.columns)}")
            st.write(f"**Размер:** {df.shape}")
            
            # Предпросмотр данных
            with st.expander("Предпросмотр данных"):
                st.dataframe(df.head(10))
            
            # Выбор колонок
            col1, col2 = st.columns(2)
            
            with col1:
                text_column = st.selectbox(
                    "Выберите колонку с транскрибациями ответов",
                    options=df.columns.tolist(),
                    help="Колонка с текстом ответов студентов"
                )
            
            with col2:
                question_column = st.selectbox(
                    "Выберите колонку с номерами вопросов",
                    options=df.columns.tolist(),
                    help="Колонка с номерами вопросов (1-4)"
                )
            
            # Проверка данных
            if text_column and question_column:
                # Показ примеров данных
                st.subheader("Проверка данных")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Пример транскрибации:**")
                    sample_text = df[text_column].iloc[0] if len(df) > 0 else "Нет данных"
                    st.text_area("", value=sample_text[:300] + "..." if len(str(sample_text)) > 300 else sample_text, 
                               height=100, key="text_sample")
                
                with col2:
                    st.write("**Распределение вопросов:**")
                    question_counts = df[question_column].value_counts().sort_index()
                    st.write(question_counts)
                
                # Кнопка запуска оценки
                if st.button("🚀 Запустить оценку ответов", type="primary"):
                    with st.spinner("Оцениваю ответы..."):
                        # Подготовка данных
                        texts = df[text_column].fillna('').astype(str).tolist()
                        question_numbers = df[question_column].astype(int).tolist()
                        
                        # Прогресс бар
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Обработка батчами как в Colab
                        batch_size = 16
                        all_grades = []
                        
                        total_batches = (len(texts) + batch_size - 1) // batch_size
                        
                        for i in range(0, len(texts), batch_size):
                            batch_texts = texts[i:i + batch_size]
                            batch_questions = question_numbers[i:i + batch_size]
                            
                            status_text.text(f"Обработка батча {i//batch_size + 1}/{total_batches}")
                            
                            batch_grades = grader.predict_grades(batch_texts, batch_questions)
                            all_grades.extend(batch_grades)
                            
                            progress = (i + batch_size) / len(texts)
                            progress_bar.progress(min(progress, 1.0))
                        
                        # Добавление оценок в DataFrame
                        df['predicted_score'] = all_grades
                        
                        st.success(f"✅ Оценка завершена! Обработано {len(texts)} ответов")
                        
                        # Результаты
                        st.header("📊 Результаты оценки")
                        
                        # Статистика
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Средний балл", f"{df['predicted_score'].mean():.2f}")
                        with col2:
                            st.metric("Максимальный балл", f"{df['predicted_score'].max()}")
                        with col3:
                            st.metric("Минимальный балл", f"{df['predicted_score'].min()}")
                        with col4:
                            st.metric("Всего ответов", len(df))
                        
                        # Детальная статистика по вопросам
                        st.subheader("Статистика по вопросам")
                        
                        question_stats = df.groupby(question_column)['predicted_score'].agg([
                            ('count', 'count'),
                            ('mean_score', 'mean'),
                            ('max_score', 'max'),
                            ('min_score', 'min')
                        ]).round(2)
                        
                        st.dataframe(question_stats)
                        
                        # Визуализация
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            df['predicted_score'].hist(bins=20, ax=ax, alpha=0.7, color='#4CAF50')
                            ax.set_xlabel('Оценка')
                            ax.set_ylabel('Количество ответов')
                            ax.set_title('Распределение оценок')
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                        
                        with col2:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            score_by_question = df.groupby(question_column)['predicted_score'].mean()
                            score_by_question.plot(kind='bar', ax=ax, alpha=0.7, color='#2196F3')
                            ax.set_xlabel('Номер вопроса')
                            ax.set_ylabel('Средняя оценка')
                            ax.set_title('Средние оценки по вопросам')
                            ax.grid(True, alpha=0.3)
                            plt.xticks(rotation=0)
                            st.pyplot(fig)
                        
                        # Таблица с результатами
                        st.subheader("Детальные результаты")
                        results_df = df[[question_column, text_column, 'predicted_score']].copy()
                        st.dataframe(results_df.head(20))
                        
                        # Скачивание результатов
                        st.subheader("📥 Скачать результаты")
                        
                        csv = df.to_csv(index=False, sep=';', encoding='utf-8')
                        
                        st.download_button(
                            label="Скачать CSV с оценками",
                            data=csv,
                            file_name="exam_results_with_scores.csv",
                            mime="text/csv",
                            help="Файл будет содержать исходные данные + колонку с предсказанными оценками"
                        )
            
        except Exception as e:
            st.error(f"❌ Ошибка обработки файла: {e}")
            st.info("""
            **Возможные причины:**
            - Неправильный разделитель (должен быть точкой с запятой)
            - Неверная кодировка (должна быть UTF-8)
            - Отсутствуют необходимые колонки
            - Неправильный формат номеров вопросов
            """)

if __name__ == "__main__":
    main()
