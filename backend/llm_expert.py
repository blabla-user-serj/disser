"""
LLM-эксперт с YandexGPT для анализа и КОРРЕКЦИИ прогноза
Работает с OpenAI-совместимым API Yandex Cloud
"""

import os
import requests
import numpy as np
from typing import Dict, List, Optional
import json
import traceback
from datetime import datetime
YANDEX_CLOUD_MODEL = "yandexgpt-lite"

class LLMExpert:
    """LLM-эксперт для анализа временных рядов и коррекции прогноза"""

    def __init__(self):
        # Читаем из переменных окружения
        self.api_key = os.getenv('YANDEX_API_KEY', '')
        self.folder_id = os.getenv('YANDEX_FOLDER_ID', '')
        self.model = os.getenv('YANDEX_MODEL', 'yandexgpt-lite')
        
        # REST API endpoint для YandexGPT
        self.api_url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        
        print(f"🔧 LLM Expert инициализирован")
        print(f"   - API Key: {'✓ Установлен' if self.api_key else '✗ НЕ УСТАНОВЛЕН'}")
        print(f"   - Folder ID: {self.folder_id if self.folder_id else '✗ НЕ УСТАНОВЛЕН'}")

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
        """Тестирование подключения к YandexGPT"""
        
        if not self.api_key or not self.folder_id:
            print("❌ API Key или Folder ID не установлены")
            return False
        
        try:
            print("🧪 Тестирую подключение к YandexGPT...")
            print(f"   - API URL: {self.api_url}")
            print(f"   - Folder ID: {self.folder_id}")
            print(f"   - Model: {self.model}")
            
            headers = {
                "Authorization": f"Api-Key {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "modelUri": f"gpt://{self.folder_id}/{self.model}/latest",
                "completionOptions": {
                    "stream": False,
                    "temperature": 0.3,
                    "maxTokens": 50
                },
                "messages": [
                    {
                        "role": "user",
                        "text": "Привет"
                    }
                ]
            }
            
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                text = result['result']['alternatives'][0]['message']['text']
                print("✅ Подключение успешно!")
                print(f"   Ответ: {text[:100]}...")
                return True
            else:
                print(f"❌ HTTP {response.status_code}: {response.text}")
                return False
            
        except Exception as e:
            print(f"❌ Тест не прошёл: {e}")
            return False

    def _call_yandex_gpt(self, prompt: str) -> str:
        """Вызов YandexGPT REST API без streaming"""
        
        if not self.api_key or not self.folder_id:
            print("❌ Ошибка: не установлены YANDEX_API_KEY или YANDEX_FOLDER_ID")
            return None

        try:
            # Проверка размера промпта
            max_prompt_size = 8000
            if len(prompt) > max_prompt_size:
                print(f"⚠️  Промпт слишком большой ({len(prompt)} символов)")
                print(f"   Обрезаю до {max_prompt_size} символов...")
                prompt = prompt[:max_prompt_size]

            print(f"📡 Отправляю запрос к YandexGPT REST API...")
            print(f"   - Folder ID: {self.folder_id}")
            print(f"   - Model: {self.model}")
            print(f"   - Prompt size: {len(prompt)} символов")
            
            headers = {
                "Authorization": f"Api-Key {self.api_key}",
                "Content-Type": "application/json"
            }
            
            system_prompt = (
                "Ты - эксперт по анализу временных рядов и прогнозированию. "
                "Твоя задача - анализировать данные, учитывать внешние факторы "
                "и КОРРЕКТИРОВАТЬ прогноз на основе найденной информации. "
                "Отвечай на русском языке. ВАЖНО: отвечай ТОЛЬКО валидным JSON без дополнительного текста."
            )
            
            payload = {
                "modelUri": f"gpt://{self.folder_id}/{self.model}/latest",
                "completionOptions": {
                    "stream": False,
                    "temperature": 0.3,
                    "maxTokens": 2000
                },
                "messages": [
                    {
                        "role": "system",
                        "text": system_prompt
                    },
                    {
                        "role": "user",
                        "text": prompt
                    }
                ]
            }
            
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                full_response = result['result']['alternatives'][0]['message']['text']
                print("✅ Успешно получен ответ от YandexGPT")
                print(f"📊 Размер ответа: {len(full_response)} символов")
                return full_response
            else:
                print(f"\n❌ YandexGPT API Error:")
                print(f"   - HTTP Status: {response.status_code}")
                print(f"   - Response: {response.text}")
                
                if response.status_code == 403:
                    print(f"\n   💡 СОВЕТ: Ошибка 403 - проблема с доступом!")
                    print(f"   Проверьте:")
                    print(f"   1. Роль 'ai.languageModels.user' назначена на каталог {self.folder_id}")
                    print(f"   2. Folder ID правильный")
                    print(f"   3. API Key активен")
                elif response.status_code == 429:
                    print(f"\n   💡 СОВЕТ: Слишком много запросов (Rate Limit)")
                
                return None
            
        except requests.exceptions.Timeout:
            print(f"\n❌ Timeout: запрос к YandexGPT превысил 60 секунд")
            return None
            
        except requests.exceptions.RequestException as e:
            print(f"\n❌ Request Error:")
            print(f"   - {e}")
            return None
            
        except Exception as e:
            print(f"\n❌ Неожиданная ошибка:")
            print(f"   - Type: {type(e).__name__}")
            print(f"   - Message: {str(e)}")
            traceback.print_exc()
            return None

    def correct_forecast(
        self,
        historical_data: np.ndarray,
        forecast: np.ndarray,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
        web_urls: Optional[List[str]] = None
    ) -> Dict:
        """
        Анализ и КОРРЕКЦИЯ прогноза с помощью LLM

        Args:
            historical_data: исторические данные
            forecast: прогноз модели
            lower_bound: нижняя граница доверительного интервала
            upper_bound: верхняя граница доверительного интервала
            web_urls: список URL для извлечения контекста

        Returns:
            dict: {
                'corrected_forecast': скорректированный прогноз,
                'corrected_lower': скорректированная нижняя граница,
                'corrected_upper': скорректированная верхняя граница,
                'analysis': текстовый анализ,
                'correction_applied': был ли применён LLM (True/False)
            }
        """

        # Базовая статистика
        mean_hist = np.mean(historical_data)
        std_hist = np.std(historical_data)
        trend = np.polyfit(range(len(historical_data)), historical_data, 1)[0]

        # Извлечение веб-контекста
        web_context = ""
        if web_urls:
            web_context = self._fetch_web_context(web_urls)

        # Формирование промпта
        prompt = f"""Проанализируй данные временного ряда и СКОРРЕКТИРУЙ прогноз:

ИСТОРИЧЕСКИЕ ДАННЫЕ:
- Среднее значение: {mean_hist:.2f}
- Стандартное отклонение: {std_hist:.2f}
- Тренд (наклон): {trend:.4f} {'(рост)' if trend > 0 else '(падение)'}
- Последние 10 значений: {historical_data[-10:].tolist()}

ПРОГНОЗ МОДЕЛИ (следующие {len(forecast)} точек):
- Прогноз: {forecast.tolist()}
- Нижняя граница: {lower_bound.tolist()}
- Верхняя граница: {upper_bound.tolist()}

{'ВНЕШНИЙ КОНТЕКСТ (из веб-источников):' if web_context else 'ВНЕШНИЙ КОНТЕКСТ НЕ ПРЕДОСТАВЛЕН (используй только исторические данные)'}
{web_context}

ТВОЯ ЗАДАЧА:
1. Проанализируй исторические данные и выяви паттерны
2. Если предоставлен внешний контекст - учти его влияние (новости, события, тренды)
3. СКОРРЕКТИРУЙ прогноз модели на основе анализа
4. Укажи процент коррекции для каждой точки прогноза (например: +5%, -10%, 0%)

ФОРМАТ ОТВЕТА (строго JSON):
{{
  "analysis": "Краткий анализ данных и внешних факторов",
  "correction_factors": [список коэффициентов коррекции для каждой точки, например: [1.05, 1.03, 1.0, 0.98, ...]],
  "reasoning": "Объяснение коррекции"
}}

Пример: если прогноз [100, 105, 110], а внешний контекст говорит о росте на 10%, то correction_factors: [1.1, 1.1, 1.1]
"""

        # Вызов YandexGPT
        llm_response = self._call_yandex_gpt(prompt)

        # Обработка ответа
        if llm_response:
            try:
                # Парсинг JSON из ответа
                json_start = llm_response.find('{')
                json_end = llm_response.rfind('}') + 1

                if json_start != -1 and json_end > json_start:
                    json_str = llm_response[json_start:json_end]
                    result = json.loads(json_str)

                    correction_factors = result.get('correction_factors', [])

                    # Применение коррекции
                    if len(correction_factors) == len(forecast):
                        corrected_forecast = forecast * np.array(correction_factors)
                        corrected_lower = lower_bound * np.array(correction_factors)
                        corrected_upper = upper_bound * np.array(correction_factors)

                        analysis = f"{result.get('analysis', '')}\n\n{result.get('reasoning', '')}"

                        return {
                            'corrected_forecast': corrected_forecast,
                            'corrected_lower': corrected_lower,
                            'corrected_upper': corrected_upper,
                            'analysis': analysis,
                            'correction_applied': True
                        }

            except json.JSONDecodeError:
                print(f"Не удалось распарсить JSON из ответа LLM: {llm_response}")
            except Exception as e:
                print(f"Ошибка при обработке ответа LLM: {str(e)}")

        # Fallback: базовый анализ без LLM
        print("\n⚠️  LLM коррекция недоступна, используется базовый анализ")
        print("   Прогноз возвращается без изменений")
        analysis = self._basic_analysis(historical_data, forecast, trend, web_context)

        return {
            'corrected_forecast': forecast,  # Без коррекции
            'corrected_lower': lower_bound,
            'corrected_upper': upper_bound,
            'analysis': analysis,
            'correction_applied': False
        }

    def _basic_analysis(
        self,
        historical_data: np.ndarray,
        forecast: np.ndarray,
        trend: float,
        web_context: str
    ) -> str:
        """Базовый анализ без LLM"""
        mean_hist = np.mean(historical_data)
        std_hist = np.std(historical_data)
        mean_forecast = np.mean(forecast)

        analysis = f"📊 Базовый статистический анализ (без LLM):\n\n"
        analysis += f"📈 Исторические данные:\n"
        analysis += f"   - Среднее значение: {mean_hist:.2f}\n"
        analysis += f"   - Стандартное отклонение: {std_hist:.2f}\n"
        analysis += f"   - Количество точек: {len(historical_data)}\n\n"
        analysis += f"📉 Тренд: {'📈 Растущий' if trend > 0 else '📉 Падающий'} (наклон: {trend:.4f})\n\n"
        analysis += f"🔮 Прогноз:\n"
        analysis += f"   - Среднее значение: {mean_forecast:.2f}\n"
        analysis += f"   - Количество точек: {len(forecast)}\n"
        
        if web_context:
            analysis += f"\n🌐 Внешний контекст:\n"
            analysis += f"   - Загружено {len(web_context)} символов\n"
            analysis += f"   - ⚠️  Контекст учтен частично (LLM недоступен)\n"
        
        analysis += f"\n💡 Примечание: Для полной коррекции прогноза настройте YandexGPT API:\n"
        analysis += f"   1. Установите YANDEX_API_KEY\n"
        analysis += f"   2. Установите YANDEX_FOLDER_ID\n"

        return analysis


# Пример использования:
if __name__ == "__main__":
    # Установите переменные окружения перед запуском:
    # export YANDEX_API_KEY="your_api_key"
    # export YANDEX_FOLDER_ID="your_folder_id"
    
    expert = LLMExpert()
    
    # Сначала проверяем подключение
    print("=" * 60)
    print("ЭТАП 1: ПРОВЕРКА ПОДКЛЮЧЕНИЯ")
    print("=" * 60)
    if not expert.test_yandex_gpt():
        print("\n❌ Не удаётся подключиться к YandexGPT.")
        print("Проверьте:")
        print("  1. Переменные окружения: YANDEX_API_KEY, YANDEX_FOLDER_ID")
        print("  2. Права доступа: роль ai.languageModels.user на уровне каталога")
        print("  3. API Key: не истёк и активен")
        exit(1)
    
    # Если тест прошёл, запускаем основной код
    print("\n" + "=" * 60)
    print("ЭТАП 2: АНАЛИЗ И КОРРЕКЦИЯ ПРОГНОЗА")
    print("=" * 60)
    
    # Пример данных
    historical = np.array([100, 102, 104, 106, 108, 110, 112, 114, 116, 118])
    forecast = np.array([120, 122, 124, 126, 128])
    lower = forecast * 0.95
    upper = forecast * 1.05
    
    # Коррекция без веб-контекста
    result = expert.correct_forecast(
        historical_data=historical,
        forecast=forecast,
        lower_bound=lower,
        upper_bound=upper
    )
    
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ:")
    print("=" * 60)
    print(f"✓ Коррекция применена: {result['correction_applied']}")
    print(f"✓ Исходный прогноз: {forecast}")
    print(f"✓ Скорректированный прогноз: {result['corrected_forecast']}")
    print(f"\n📊 Анализ:\n{result['analysis']}")
