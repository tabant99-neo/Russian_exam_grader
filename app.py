import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_model_resources
import io
import time
import os

# Настройка страницы
st.set_page_config(
    page_title="Оценка эссе",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Загрузка модели (кэшируется)
@st.cache_resource
def load_model():
    return load_model_resources()

def check_model_files():
    """Проверка наличия файлов модели"""
    model_dir = 'my_trained_model_2'
    required_files = {
        'model.safetensors': 'Основные веса модели',
        'tokenizer.json': 'Токенизатор',
        'tokenizer_config.json': 'Конфигурация токенизатора',
        'special_tokens_map.json': 'Специальные токены',
        'config.json': 'Конфигурация модели'
    }
    
    missing_files = []
    existing_files = []
    
    for file, description in required_files.items():
        file_path = os.path.join(model_dir, file)
        if os.path.exists(file_path):
            existing_files.append((file, description, "✅"))
        else:
            missing_files.append((file, description, "❌"))
    
    return existing_files, missing_files

def main():
    st.title("📝 Система автоматической оценки эссе")
    st.markdown("---")
    
    # Проверка файлов модели
    st.sidebar.title("Проверка модели")
    existing_files, missing_files = check_model_files()
    
    st.sidebar.subheader("Найденные файлы:")
    for file, description, status in existing_files:
        st.sidebar.text(f"{status} {file}")
    
    if missing_files:
        st.sidebar.subheader("Отсутствующие файлы:")
        for file, description, status in missing_files:
            st.sidebar.text(f"{status} {file}")
        st.sidebar.error("Не все файлы модели найдены!")
    else:
        st.sidebar.success("Все файлы модели на месте!")
    
    # Загрузка модели
    with st.spinner("Загрузка модели..."):
        grader = load_model()
    
    if grader is None:
        st.error("""
        ❌ Ошибка загрузки модели. Пожалуйста, проверьте:
        1. Все ли файлы модели находятся в папке `my_trained_model_2/`
        2. Правильные ли названия у файлов
        3. Поддерживает ли модель архитектуру для sequence classification
        """)
        
        st.info("""
        🔍 **Необходимые файлы:**
        - `model.safetensors` - веса модели
        - `tokenizer.json` - токенизатор  
        - `tokenizer_config.json` - конфиг токенизатора
        - `special_tokens_map.json` - специальные токены
        - `config.json` - конфигурация модели
        """)
        return
    
    st.success("✅ Модель успешно загружена!")
    
    # Информация о модели
    st.sidebar.markdown("---")
    st.sidebar.subheader("Информация о модели")
    st.sidebar.text(f"Тип: Transformers")
    st.sidebar.text(f"Архитектура: {type(grader.model).__name__}")
    st.sidebar.text(f"Токенизатор: {type(grader.tokenizer).__name__}")
    
    # Основной интерфейс
    tab1, tab2, tab3 = st.tabs(["📤 Загрузка файла", "✍️ Ручной ввод", "ℹ️ О системе"])
    
    with tab1:
        st.header("Загрузка CSV файла")
        st.markdown("""
        Загрузите CSV файл с эссе. Файл должен содержать колонку с текстом эссе.
        После обработки вы сможете скачать файл с добавленными оценками.
        """)
        
        uploaded_file = st.file_uploader("Выберите CSV файл", type=['csv'])
        
        if uploaded_file is not None:
            try:
                # Чтение файла
                df = pd.read_csv(uploaded_file)
                st.success(f"Файл успешно загружен! Размер: {df.shape}")
                
                # Выбор колонки с текстом
                text_column = st.selectbox(
                    "Выберите колонку с текстом эссе",
                    options=df.columns.tolist(),
                    key="file_text_column"
                )
                
                if st.button("Оценить эссе", key="file_grade"):
                    with st.spinner("Обработка эссе..."):
                        # Прогресс бар
                        progress_bar = st.progress(0)
                        
                        # Получение оценок
                        essays = df[text_column].fillna('').astype(str).tolist()
                        
                        # Обработка чанками для больших файлов
                        batch_size = 32
                        all_grades = []
                        
                        for i in range(0, len(essays), batch_size):
                            batch_essays = essays[i:i + batch_size]
                            batch_grades = grader.predict_grades(batch_essays)
                            all_grades.extend(batch_grades)
                            
                            # Обновление прогресса
                            progress = min((i + batch_size) / len(essays), 1.0)
                            progress_bar.progress(progress)
                        
                        # Добавление оценок в DataFrame
                        df['predicted_grade'] = all_grades
                        
                        st.success(f"Обработано {len(essays)} эссе!")
                        
                        # Показ результатов
                        st.subheader("Результаты оценки")
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.dataframe(df[[text_column, 'predicted_grade']].head(10))
                        
                        with col2:
                            st.metric("Средняя оценка", f"{df['predicted_grade'].mean():.2f}")
                            st.metric("Макс. оценка", f"{df['predicted_grade'].max():.1f}")
                            st.metric("Мин. оценка", f"{df['predicted_grade'].min():.1f}")
                        
                        # Визуализация
                        st.subheader("Визуализация результатов")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            df['predicted_grade'].hist(bins=20, ax=ax, alpha=0.7)
                            ax.set_xlabel('Оценка')
                            ax.set_ylabel('Количество')
                            ax.set_title('Распределение оценок')
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                        
                        with col2:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            grade_counts = df['predicted_grade'].value_counts().sort_index()
                            grade_counts.plot(kind='bar', ax=ax, alpha=0.7)
                            ax.set_xlabel('Оценка')
                            ax.set_ylabel('Количество')
                            ax.set_title('Частоты оценок')
                            ax.grid(True, alpha=0.3)
                            plt.xticks(rotation=45)
                            st.pyplot(fig)
                        
                        # Скачивание результата
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Скачать результат CSV",
                            data=csv,
                            file_name="essays_with_grades.csv",
                            mime="text/csv",
                            key="download_csv"
                        )
                        
            except Exception as e:
                st.error(f"Ошибка обработки файла: {e}")
                st.info("Проверьте формат файла и кодировку (должен быть UTF-8)")
    
    with tab2:
        st.header("Ручной ввод эссе")
        st.markdown("Введите текст эссе для получения оценки")
        
        essay_text = st.text_area(
            "Текст эссе",
            height=300,
            placeholder="Введите текст эссе здесь...",
            key="manual_input"
        )
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if st.button("Оценить эссе", key="manual_grade"):
                if essay_text.strip():
                    with st.spinner("Оценка эссе..."):
                        grade = grader.predict_single_grade(essay_text)
                        
                        st.subheader("Результат оценки")
                        
                        # Красивое отображение оценки
                        st.metric(
                            label="Предсказанная оценка",
                            value=f"{grade:.1f}",
                            delta=f"из 10 баллов"
                        )
                        
                        # Визуализация оценки
                        fig, ax = plt.subplots(figsize=(10, 2))
                        ax.barh([0], [grade], color='#4CAF50', alpha=0.7, height=0.5)
                        ax.set_xlim(0, 10)
                        ax.set_yticks([])
                        ax.set_xlabel('Оценка')
                        ax.set_title('Результат оценки')
                        ax.axvline(x=grade, color='red', linestyle='--', alpha=0.8)
                        ax.text(grade, 0, f' {grade:.1f}', ha='left', va='center', 
                               fontweight='bold', fontsize=12, color='red')
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        
                else:
                    st.warning("Пожалуйста, введите текст эссе")
    
    with tab3:
        st.header("О системе")
        st.markdown("""
        ### 📊 Система автоматической оценки эссе
        
        Эта система использует передовые модели машинного обучения для автоматической 
        оценки текстовых эссе на русском языке.
        
        **Возможности:**
        - 📤 Пакетная обработка CSV файлов
        - ✍️ Индивидуальная оценка эссе
        - 📊 Визуализация результатов
        - 📥 Экспорт результатов
        
        **Технические детали:**
        - Модель: Fine-tuned Transformer
        - Токенизатор: Предобученный BPE токенизатор
        - Диапазон оценок: 1-10 баллов
        - Поддержка: Русский язык
        
        **Использование:**
        1. Загрузите CSV файл с колонкой текстов эссе
        2. Выберите соответствующую колонку
        3. Получите оценки и статистику
        4. Скачайте результат
        """)

if __name__ == "__main__":
    main()