"""
LLM-эксперт с YandexGPT для анализа и КОРРЕКЦИИ прогноза
Использует OpenAI-совместимый API Yandex Cloud
"""

import os
import requests
import numpy as np
from typing import Dict, List, Optional
import json
import traceback
from datetime import datetime

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  openai не установлен. Установите: pip install openai")


class LLMExpert:
    """LLM-эксперт для анализа временных рядов и коррекции прогноза"""

    def __init__(self):
        # Читаем из переменных окружения
        self.api_key = os.getenv('YANDEX_API_KEY', '')
        self.folder_id = os.getenv('YANDEX_FOLDER_ID', '')
        self.model = os.getenv('YANDEX_MODEL', 'yandexgpt-lite')
        
        # OpenAI-совместимый API endpoint
        self.base_url = "https://ai.api.cloud.yandex.net/v1"
        
        # Инициализация OpenAI клиента
        self.client = None
        if OPENAI_AVAILABLE and self.api_key and self.folder_id:
            try:
                self.client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    # Передаём folder_id как project (совместимость с OpenAI API)
                    default_headers={"x-folder-id": self.folder_id}
                )
                print(f"🔧 LLM Expert инициализирован (OpenAI-совместимый API)")
                print(f"   - API Key: ✓ Установлен")
                print(f"   - Folder ID: {self.folder_id}")
                print(f"   - Model: {self.model}")
            except Exception as e:
                print(f"⚠️  Ошибка инициализации OpenAI клиента: {e}")
                self.client = None
        else:
            print(f"🔧 LLM Expert инициализирован")
            print(f"   - API Key: {'✗ НЕ УСТАНОВЛЕН' if not self.api_key else '✓ Установлен'}")
            print(f"   - Folder ID: {'✗ НЕ УСТАНОВЛЕН' if not self.folder_id else self.folder_id}")
            if not OPENAI_AVAILABLE:
                print(f"   - OpenAI: ✗ Не установлен (pip install openai)")

    def _fetch_web_context(self, urls: List[str]) -> str:
        """Извлечение контекста из веб-ссылок"""
        context = []
        for url in urls:
            try:
                response = requests.get(url, timeout=10, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                if response.status_code == 200:
                    text = response.text[:2000]
                    context.append(f"Источник: {url}\n{text}\n")
            except Exception as e:
                context.append(f"Не удалось загрузить {url}: {str(e)}\n")
        
        return "\n".join(context) if context else ""

    def test_yandex_gpt(self) -> bool:
        """Тестирование подключения к YandexGPT через OpenAI API"""
        
        if not self.client:
            print("❌ OpenAI клиент не инициализирован")
            print("💡 Проверьте:")
            print("   1. pip install openai")
            print("   2. YANDEX_API_KEY установлен")
            print("   3. YANDEX_FOLDER_ID установлен")
            return False
        
        try:
            print("🧪 Тестирую подключение к YandexGPT через OpenAI API...")
            print(f"   - Base URL: {self.base_url}")
            print(f"   - Folder ID: {self.folder_id}")
            print(f"   - Model: {self.model}")
            
            # Формируем model URI в формате Yandex Cloud
            model_uri = f"gpt://{self.folder_id}/{self.model}/latest"
            
            # Используем новый метод responses.create()
            response = self.client.responses.create(
                model=model_uri,
                temperature=0.3,
                instructions="Ты помощник для анализа данных.",
                input="Привет! Как дела?",
                max_output_tokens=100
            )
            
            print("✅ Подключение успешно!")
            print(f"   Ответ: {response.output_text[:100]}...")
            return True
            
        except Exception as e:
            print(f"❌ Тест не прошёл: {e}")
            print(f"   Тип ошибки: {type(e).__name__}")
            traceback.print_exc()
            
            # Проверка прав доступа
            if "403" in str(e) or "Permission denied" in str(e):
                print("\n💡 Ошибка 403 - проблема с доступом!")
                print("   Проверьте:")
                print("   1. Роль 'ai.languageModels.user' назначена на каталог")
                print("   2. Folder ID правильный")
                print("   3. API Key активен")
                print(f"\n📚 См. документацию: docs/YANDEX_GPT_SETUP.md")
            
            return False

    def _call_yandex_gpt(self, prompt: str, instructions: str = "") -> str:
        """Вызов YandexGPT через OpenAI-совместимый API"""
        
        if not self.client:
            print("❌ OpenAI клиент не инициализирован")
            return ""
        
        if not prompt or len(prompt.strip()) == 0:
            print("❌ Пустой промпт")
            return ""
        
        # Обрезаем промпт если слишком длинный
        max_prompt_length = 10000
        if len(prompt) > max_prompt_length:
            prompt = prompt[:max_prompt_length]
            print(f"⚠️  Промпт обрезан до {max_prompt_length} символов")
        
        try:
            print(f"📡 Отправляю запрос к YandexGPT через OpenAI API...")
            print(f"   - Folder ID: {self.folder_id}")
            print(f"   - Model: {self.model}")
            print(f"   - Input size: {len(prompt)} символов")
            
            # Формируем model URI
            model_uri = f"gpt://{self.folder_id}/{self.model}/latest"
            
            # Используем responses.create()
            response = self.client.responses.create(
                model=model_uri,
                temperature=0.3,
                instructions=instructions or "Ты эксперт по анализу временных рядов. Отвечай ТОЛЬКО валидным JSON.",
                input=prompt,
                max_output_tokens=500
            )
            
            result_text = response.output_text
            print(f"✅ Получен ответ: {len(result_text)} символов")
            
            return result_text
            
        except Exception as e:
            error_msg = str(e)
            print(f"\n❌ YandexGPT API Error:")
            print(f"   - Error: {error_msg}")
            print(f"   - Type: {type(e).__name__}")
            
            # Детальная диагностика
            if "403" in error_msg or "Permission denied" in error_msg:
                print("\n   💡 СОВЕТ: Ошибка 403 - проблема с доступом!")
                print("   Проверьте:")
                print(f"   1. Роль 'ai.languageModels.user' назначена на каталог {self.folder_id}")
                print("   2. Folder ID правильный")
                print("   3. API Key активен")
                print(f"\n   📚 Полная инструкция: docs/YANDEX_GPT_SETUP.md")
                print(f"   🔥 Быстрое решение: YANDEX_GPT_QUICK_FIX.md")
            
            elif "401" in error_msg or "Unauthorized" in error_msg:
                print("\n   💡 СОВЕТ: Ошибка 401 - проблема с API ключом!")
                print("   Проверьте:")
                print("   1. YANDEX_API_KEY правильный")
                print("   2. API ключ активен")
                print("   3. API ключ привязан к сервисному аккаунту")
            
            elif "404" in error_msg or "Not Found" in error_msg:
                print("\n   💡 СОВЕТ: Ошибка 404 - неправильный folder ID или модель!")
                print("   Проверьте:")
                print(f"   1. YANDEX_FOLDER_ID: {self.folder_id}")
                print(f"   2. YANDEX_MODEL: {self.model}")
            
            elif "timeout" in error_msg.lower():
                print("\n   💡 СОВЕТ: Timeout!")
                print("   - Попробуйте ещё раз")
                print("   - Проверьте интернет-соединение")
            
            return ""

    def correct_forecast(
        self,
        historical_data: np.ndarray,
        forecast: np.ndarray,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
        web_urls: Optional[List[str]] = None
    ) -> Dict:
        """
        Коррекция прогноза с помощью YandexGPT
        
        Args:
            historical_data: исторические значения
            forecast: прогноз модели
            lower_bound: нижняя граница 95% доверительного интервала
            upper_bound: верхняя граница 95% доверительного интервала
            web_urls: список URL для извлечения контекста
            
        Returns:
            dict с ключами:
                - forecast: скорректированный прогноз
                - lower_bound: скорректированная нижняя граница
                - upper_bound: скорректированная верхняя граница
                - analysis: текстовый анализ от LLM
                - correction_applied: был ли применён LLM (True/False)
        """
        
        # Если клиент не инициализирован - возвращаем базовый анализ
        if not self.client:
            return {
                'forecast': forecast,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'analysis': self._basic_analysis(historical_data, forecast),
                'correction_applied': False
            }
        
        # Статистика исторических данных
        mean_val = np.mean(historical_data)
        std_val = np.std(historical_data)
        trend = np.polyfit(range(len(historical_data)), historical_data, 1)[0]
        
        # Контекст из веб-источников
        web_context = ""
        if web_urls:
            web_context = self._fetch_web_context(web_urls)
        
        # Формируем промпт
        instructions = (
            "Ты эксперт по анализу временных рядов и прогнозированию. "
            "Твоя задача - проанализировать прогноз и вернуть коррекцию в формате JSON.\n\n"
            "ВАЖНО: Отвечай ТОЛЬКО валидным JSON, без дополнительного текста."
        )
        
        # Определяем направление тренда
        trend_direction = "растёт" if trend > 0 else "падает"
        
        # Блок внешнего контекста
        web_context_block = ""
        if web_context:
            web_context_block = "\nВНЕШНИЙ КОНТЕКСТ:\n{}\n".format(web_context)
        
        # Собираем промпт через format()
        prompt = (
            "Проанализируй временной ряд и прогноз:\n\n"
            "ИСТОРИЧЕСКИЕ ДАННЫЕ:\n"
            "- Среднее значение: {mean_val:.2f}\n"
            "- Стандартное отклонение: {std_val:.2f}\n"
            "- Тренд: {trend_direction} ({trend:.2f} в день)\n"
            "- Последние 10 значений: {last_values}\n\n"
            "ПРОГНОЗ МОДЕЛИ:\n"
            "- Значения: {forecast_values}\n"
            "- 95% доверительный интервал:\n"
            "  * Нижняя граница: {lower_values}\n"
            "  * Верхняя граница: {upper_values}\n"
            "{web_context_block}\n"
            "ЗАДАЧА:\n"
            "Верни коэффициенты коррекции для каждой точки прогноза в формате JSON:\n\n"
            "{{\n"
            '  "correction_factors": [1.0, 1.05, 0.95, ...],\n'
            '  "analysis": "краткий анализ тренда и факторов",\n'
            '  "reasoning": "почему применены эти коэффициенты"\n'
            "}}\n\n"
            "Коэффициент 1.0 = без изменений, >1.0 = увеличение, <1.0 = уменьшение."
        ).format(
            mean_val=mean_val,
            std_val=std_val,
            trend_direction=trend_direction,
            trend=trend,
            last_values=historical_data[-10:].tolist(),
            forecast_values=forecast.tolist(),
            lower_values=lower_bound.tolist(),
            upper_values=upper_bound.tolist(),
            web_context_block=web_context_block
        )
        
        # Вызов YandexGPT
        llm_response = self._call_yandex_gpt(prompt, instructions)
        
        if not llm_response:
            # Fallback если LLM недоступен
            return {
                'corrected_forecast': forecast,
                'corrected_lower': lower_bound,
                'corrected_upper': upper_bound,
                'analysis': self._basic_analysis(historical_data, forecast),
                'correction_applied': False
            }
        
        # Парсинг JSON ответа
        try:
            # Извлекаем JSON из ответа (может быть обёрнут в ```json ... ```)
            json_text = llm_response
            if "```json" in json_text:
                json_text = json_text.split("```json")[1].split("```")[0].strip()
            elif "```" in json_text:
                json_text = json_text.split("```")[1].split("```")[0].strip()
            
            llm_data = json.loads(json_text)
            
            correction_factors = llm_data.get('correction_factors', [])
            analysis = llm_data.get('analysis', 'Анализ от YandexGPT')
            
            # Применяем коррекцию
            if len(correction_factors) == len(forecast):
                corrected_forecast = forecast * np.array(correction_factors)
                corrected_lower = lower_bound * np.array(correction_factors)
                corrected_upper = upper_bound * np.array(correction_factors)
                
                print(f"✅ Коррекция применена: {correction_factors}")
                
                return {
                    'corrected_forecast': corrected_forecast,
                    'corrected_lower': corrected_lower,
                    'corrected_upper': corrected_upper,
                    'analysis': analysis,
                    'correction_applied': True
                }
            else:
                print(f"⚠️  Несоответствие длин: {len(correction_factors)} != {len(forecast)}")
                return {
                    'corrected_forecast': forecast,
                    'corrected_lower': lower_bound,
                    'corrected_upper': upper_bound,
                    'analysis': analysis,
                    'correction_applied': False
                }
            
        except json.JSONDecodeError as e:
            print(f"❌ Ошибка парсинга JSON: {e}")
            print(f"   Ответ LLM: {llm_response[:200]}...")
            
            # Fallback
            return {
                'corrected_forecast': forecast,
                'corrected_lower': lower_bound,
                'corrected_upper': upper_bound,
                'analysis': self._basic_analysis(historical_data, forecast),
                'correction_applied': False
            }

    def _basic_analysis(self, historical_data: np.ndarray, forecast: np.ndarray) -> str:
        """Базовый статистический анализ без LLM"""
        mean_hist = np.mean(historical_data)
        mean_forecast = np.mean(forecast)
        
        trend = "растёт" if mean_forecast > mean_hist else "падает"
        change_pct = ((mean_forecast - mean_hist) / mean_hist * 100) if mean_hist != 0 else 0
        
        return f"Прогноз {trend} на {abs(change_pct):.1f}% относительно исторического среднего. " \
               f"Среднее историческое: {mean_hist:.2f}, среднее прогноза: {mean_forecast:.2f}."


# Пример использования
if __name__ == "__main__":
    import sys
    
    print("="*60)
    print("🧪 ТЕСТ YandexGPT Expert (OpenAI API)")
    print("="*60)
    
    # Загрузка .env
    from dotenv import load_dotenv
    load_dotenv('../config/.env')
    
    # Проверка переменных окружения
    api_key = os.getenv('YANDEX_API_KEY')
    folder_id = os.getenv('YANDEX_FOLDER_ID')
    
    if not api_key or not folder_id:
        print("\n❌ Не установлены переменные окружения!")
        print("Создайте config/.env с:")
        print("  YANDEX_API_KEY=your_key")
        print("  YANDEX_FOLDER_ID=your_folder")
        sys.exit(1)
    
    # Инициализация эксперта
    expert = LLMExpert()
    
    # Тест подключения
    print("\n" + "="*60)
    print("1️⃣ Тест подключения")
    print("="*60)
    success = expert.test_yandex_gpt()
    
    if success:
        print("\n" + "="*60)
        print("2️⃣ Тест коррекции прогноза")
        print("="*60)
        
        # Тестовые данные
        historical = np.array([100, 105, 103, 108, 110, 107, 112, 115])
        forecast = np.array([118, 120, 122])
        lower = np.array([115, 117, 119])
        upper = np.array([121, 123, 125])
        
        result = expert.correct_forecast(historical, forecast, lower, upper)
        
        print(f"\n✅ Результат коррекции:")
        print(f"   Прогноз: {result['forecast']}")
        print(f"   Анализ: {result['analysis']}")
        print(f"   Коррекция применена: {result['correction_applied']}")
    else:
        print("\n❌ Тест не прошёл")
        print("📚 См. инструкцию: docs/YANDEX_GPT_SETUP.md")
