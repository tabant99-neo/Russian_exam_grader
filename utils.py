import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from typing import List, Union
import os

# Скачиваем необходимые данные NLTK
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class TextPreprocessor:
    def __init__(self):
        self.stemmer = SnowballStemmer("russian")
        self.stop_words = set(stopwords.words('russian'))
        self.stop_words.update(['это', 'вот', 'ну', 'бы', 'как', 'так', 'и', 'в', 'над', 'к', 'до', 'не', 'на', 'но', 'за', 'то', 'с', 'ли', 'а', 'во', 'от', 'со', 'для', 'о', 'же', 'ни', 'быть', 'он', 'say', 'said', 'would', 'could', 'should', 'the', 'a', 'an'])
        
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        
        # Приведение к нижнему регистру
        text = text.lower()
        
        # Удаление специальных символов и цифр
        text = re.sub(r'[^а-яёa-z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Удаление стоп-слов и стемминг
        words = text.split()
        words = [self.stemmer.stem(word) for word in words if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(words)

class EssayGrader:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.preprocessor = TextPreprocessor()
        
        # Загрузка токенизатора и модели
        self.load_model()
        
    def load_model(self):
        """Загрузка модели и токенизатора"""
        try:
            # Загрузка токенизатора
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Загрузка модели
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32
            )
            
            # Переводим модель в режим оценки
            self.model.eval()
            
            print("Модель и токенизатор успешно загружены")
            
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            raise
    
    def preprocess_essays(self, essays: List[str]) -> dict:
        """Токенизация эссе"""
        cleaned_essays = [self.preprocessor.clean_text(essay) for essay in essays]
        
        # Токенизация с помощью transformers
        inputs = self.tokenizer(
            cleaned_essays,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        return inputs
    
    def predict_grades(self, essays: List[str]) -> np.ndarray:
        """Предсказание оценок для списка эссе"""
        if len(essays) == 0:
            return np.array([])
        
        # Предобработка
        inputs = self.preprocess_essays(essays)
        
        # Предсказание
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=1)
            
        # Предполагаем, что модель классифицирует на 10 классов (оценки 1-10)
        # Если у вас регрессия, адаптируйте эту часть
        if predictions.shape[1] == 1:
            # Регрессия - нормализуем к 1-10
            grades = (predictions.numpy().flatten() * 9 + 1).round(1)
        else:
            # Классификация - берем argmax + 1 для оценок 1-10
            grades = (predictions.argmax(dim=1).numpy() + 1).astype(float)
        
        return grades
    
    def predict_single_grade(self, essay: str) -> float:
        """Предсказание оценки для одного эссе"""
        return self.predict_grades([essay])[0]

def load_model_resources():
    """Загрузка модели и ресурсов"""
    try:
        model_path = 'my_trained_model_2'
        
        # Проверяем существование файлов модели
        required_files = [
            'model.safetensors',
            'tokenizer.json', 
            'tokenizer_config.json',
            'special_tokens_map.json',
            'config.json'
        ]
        
        missing_files = []
        for file in required_files:
            file_path = os.path.join(model_path, file)
            if not os.path.exists(file_path):
                missing_files.append(file)
        
        if missing_files:
            raise FileNotFoundError(f"Отсутствуют файлы модели: {missing_files}")
        
        grader = EssayGrader(model_path)
        return grader
        
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return None