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
from openai import APIError, RateLimitError, APIStatusError

YANDEX_CLOUD_MODEL = "yandexgpt-lite"

class LLMExpert:
    """LLM-эксперт для анализа временных рядов и коррекции прогноза"""

    def __init__(self):
        # Читаем из переменных окружения
        self.api_key = os.getenv('YANDEX_API_KEY', '')
        self.folder_id = os.getenv('YANDEX_FOLDER_ID', '')
        self.model = os.getenv('YANDEX_MODEL', 'yandexgpt-lite')
        
        # OpenAI-совместимый endpoint Yandex Cloud
        self.base_url = "https://llm.api.cloud.yandex.net/v1"
        
        # Инициализируем OpenAI клиент с параметрами Yandex Cloud
        try:
            import openai
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                project=self.folder_id
            )
        except Exception as e:
            print(f"❌ Ошибка инициализации OpenAI клиента: {e}")
            self.client = None

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
        
        if not self.client:
            print("❌ OpenAI клиент не инициализирован")
            return False
        
        try:
            print("🧪 Тестирую подключение к YandexGPT...")
            print(f"   - Base URL: {self.base_url}")
            print(f"   - Folder ID: {self.folder_id}")
            print(f"   - API Key: {self.api_key[:10]}..." if self.api_key else "   - API Key: НЕ УСТАНОВЛЕН")
            
            response = self.client.chat.completions.create(
                model=f"gpt://{self.folder_id}/yandexgpt-lite/latest",
                messages=[
                    {"role": "user", "content": "Привет"}
                ],
                max_tokens=50,
                temperature=0.3,
            )
            
            print("✅ Подключение успешно!")
            print(f"   Ответ: {response.choices[0].message.content[:100]}...")
            return True
            
        except Exception as e:
            print(f"❌ Тест не прошёл: {e}")
            return False

    def _call_yandex_gpt(self, prompt: str) -> str:
        """Вызов YandexGPT API через OpenAI SDK с полным логированием"""
        
        if not self.api_key or not self.folder_id:
            print("❌ Ошибка: не установлены YANDEX_API_KEY или YANDEX_FOLDER_ID")
            return None

        if not self.client:
            print("❌ OpenAI клиент не инициализирован")
            return None

        try:
            # Проверка размера промпта
            max_prompt_size = 10000
            if len(prompt) > max_prompt_size:
                print(f"⚠️  Промпт слишком большой ({len(prompt)} символов)")
                print(f"   Обрезаю до {max_prompt_size} символов...")
                prompt = prompt[:max_prompt_size]

            print(f"📡 Отправляю запрос к YandexGPT...")
            print(f"   - Base URL: {self.base_url}")
            print(f"   - Folder ID: {self.folder_id}")
            print(f"   - Model URI: gpt://{self.folder_id}/yandexgpt-lite/latest")
            print(f"   - Prompt size: {len(prompt)} символов")
            
            # Используем streaming для получения ответа
            response = self.client.chat.completions.create(
                model=f"gpt://{self.folder_id}/yandexgpt-lite/latest",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Ты - эксперт по анализу временных рядов и прогнозированию. "
                            "Твоя задача - анализировать данные, учитывать внешние факторы "
                            "и КОРРЕКТИРОВАТЬ прогноз на основе найденной информации. "
                            "Отвечай на русском языке. ВАЖНО: отвечай ТОЛЬКО валидным JSON без дополнительного текста."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=2000,
                temperature=0.3,
                stream=True
            )
            
            # Собираем streaming ответ по частям
            full_response = ""
            print("📥 Получаю ответ от YandexGPT (streaming)...")
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    print(".", end="", flush=True)  # Индикатор прогресса
            
            print("\n✅ Успешно получен ответ от YandexGPT")
            print(f"📊 Размер ответа: {len(full_response)} символов")
            return full_response
            
        except APIStatusError as e:
            # Ошибка с HTTP статусом
            print(f"\n❌ YandexGPT APIStatusError:")
            print(f"   - HTTP Status: {e.status_code}")
            print(f"   - Message: {e.message}")
            print(f"   - Type: {e.type}")
            
            # Извлекаем полные заголовки ответа
            if hasattr(e, 'response'):
                headers = e.response.headers
                print(f"   - Request ID: {headers.get('x-request-id', 'N/A')}")
                print(f"   - Trace ID: {headers.get('x-trace-id', 'N/A')}")
                print(f"   - Server Request ID: {headers.get('server-request-id', 'N/A')}")
            
            print(f"\n   📋 Полная ошибка для отправки в support:")
            error_dict = {
                'status_code': e.status_code,
                'message': e.message,
                'type': str(e.type),
                'timestamp': datetime.now().isoformat()
            }
            print(f"   {json.dumps(error_dict, ensure_ascii=False, indent=2)}")
            
            if "403" in str(e.status_code) or "Permission" in e.message:
                print(f"\n   💡 СОВЕТ: Ошибка 403 - проблема с доступом!")
                print(f"   Проверьте:")
                print(f"   1. Роль 'ai.languageModels.user' назначена на каталог {self.folder_id}")
                print(f"   2. Folder ID совпадает с каталогом, где выданы права")
                print(f"   3. API Key не истёк и активен")
            
            return None
            
        except RateLimitError as e:
            print(f"\n❌ YandexGPT RateLimitError (слишком много запросов):")
            print(f"   - {e}")
            return None
            
        except APIError as e:
            print(f"\n❌ YandexGPT APIError:")
            print(f"   - {e}")
            print(f"\n   Traceback:")
            traceback.print_exc()
            return None
            
        except Exception as e:
            print(f"\n❌ Неожиданная ошибка:")
            print(f"   - Type: {type(e).__name__}")
            print(f"   - Message: {str(e)}")
            print(f"\n   Полный traceback:")
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

        analysis = f"Анализ без LLM:\n"
        analysis += f"- Исторические данные: среднее={mean_hist:.2f}, σ={std_hist:.2f}\n"
        analysis += f"- Тренд: {'растущий' if trend > 0 else 'падающий'} ({trend:.4f})\n"
        analysis += f"- Прогноз: среднее={mean_forecast:.2f}\n"
        
        if web_context:
            analysis += f"- Внешний контекст учтен: {len(web_context)} символов\n"

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
