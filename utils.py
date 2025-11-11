import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EssayGrader:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()
        
    def load_model(self):
        """Загрузка модели и токенизатора как в Colab"""
        try:
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            logger.info("Loading model...")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32
            )
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """Очистка текста как в Colab"""
        if not isinstance(text, str):
            return ""
        
        # Удаление HTML тегов
        text = re.sub(r'<[^>]+>', '', text)
        
        # Удаление специальных символов, оставляем только буквы, цифры и пунктуацию
        text = re.sub(r'[^\w\s\.\,\!\?]', ' ', text)
        
        # Замена множественных пробелов
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip().lower()
    
    def predict_grades(self, texts: List[str], question_numbers: List[int]) -> np.ndarray:
        """Предсказание оценок как в Colab"""
        if len(texts) == 0:
            return np.array([])
        
        try:
            # Очистка текстов
            cleaned_texts = [self.clean_text(text) for text in texts]
            
            # Токенизация
            inputs = self.tokenizer(
                cleaned_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Предсказание
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.softmax(outputs.logits, dim=1)
            
            # Преобразование в оценки в зависимости от номера вопроса
            grades = []
            for i, pred in enumerate(predictions):
                question_num = question_numbers[i]
                if question_num in [1, 3]:  # Вопросы 1 и 3: 0-1 балл
                    grade = pred.argmax().item()
                    grade = min(grade, 1)  # Ограничиваем максимум 1 баллом
                else:  # Вопросы 2 и 4: 0-2 балла
                    grade = pred.argmax().item()
                    grade = min(grade, 2)  # Ограничиваем максимум 2 баллами
                grades.append(grade)
            
            return np.array(grades)
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            # Возвращаем нулевые оценки в случае ошибки
            return np.zeros(len(texts))

def load_model_resources():
    """Загрузка модели"""
    try:
        model_path = 'my_trained_model_2'
        grader = EssayGrader(model_path)
        return grader
    except Exception as e:
        logger.error(f"Error loading model resources: {e}")
        return None
