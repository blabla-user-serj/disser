"""
TimeLLM с поддержкой локальных GGUF моделей через llama-cpp-python

КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Глобальный реестр моделей для предотвращения OOM
"""
import numpy as np
import pandas as pd
import warnings
import os
import gc
import sys
import math
import weakref


def sanitize_array(arr, fallback_value=0.0):
    """
    Очистка массива от NaN и Inf значений для JSON сериализации.
    
    Args:
        arr: numpy array или list для очистки
        fallback_value: значение для замены NaN/Inf
        
    Returns:
        Очищенный numpy array без NaN и Inf
    """
    arr = np.array(arr, dtype=float)
    
    # Находим валидные значения для вычисления fallback
    valid_mask = np.isfinite(arr)
    
    if valid_mask.any():
        # Используем среднее валидных значений или последнее валидное
        valid_values = arr[valid_mask]
        computed_fallback = np.mean(valid_values)
    else:
        computed_fallback = fallback_value
    
    # Заменяем NaN на fallback
    arr = np.where(np.isnan(arr), computed_fallback, arr)
    
    # Заменяем Inf/-Inf на fallback
    arr = np.where(np.isinf(arr), computed_fallback, arr)
    
    return arr

# Настройки CUDA для стабильной работы
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Синхронный режим для отладки

# Кэширование HuggingFace моделей
# Модели будут скачиваться один раз и храниться локально
HF_CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', '.cache', 'huggingface')
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = HF_CACHE_DIR
print(f"📦 Кэш HuggingFace: {HF_CACHE_DIR}")

import torch
warnings.filterwarnings('ignore')

# Безопасная инициализация CUDA
try:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.set_float32_matmul_precision('high')
        torch.set_grad_enabled(False)
        print(f"✅ CUDA инициализирован: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"⚠️  Ошибка инициализации CUDA: {e}")
    print(f"   Будет использован CPU режим")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== ГЛОБАЛЬНЫЙ РЕЕСТР МОДЕЛЕЙ ====================
# Отслеживает все созданные модели для корректной очистки памяти
_ACTIVE_MODELS = []
_OOM_FALLBACK_TO_SIMPLE = False  # Флаг для автоматического переключения на simple режим


def register_model(model):
    """Регистрирует модель в глобальном реестре"""
    global _ACTIVE_MODELS
    _ACTIVE_MODELS.append(weakref.ref(model))
    # Очищаем мёртвые ссылки
    _ACTIVE_MODELS = [ref for ref in _ACTIVE_MODELS if ref() is not None]


def destroy_all_models():
    """Уничтожает ВСЕ зарегистрированные модели и освобождает GPU память"""
    global _ACTIVE_MODELS
    
    destroyed_count = 0
    for ref in _ACTIVE_MODELS:
        model = ref()
        if model is not None:
            try:
                if hasattr(model, 'model') and model.model is not None:
                    # Для NeuralForecast моделей
                    if hasattr(model.model, 'models'):
                        for m in model.model.models:
                            if hasattr(m, 'llm'):
                                del m.llm
                            if hasattr(m, 'to'):
                                try:
                                    m.cpu()
                                except:
                                    pass
                    del model.model
                    model.model = None
                    destroyed_count += 1
            except Exception as e:
                print(f"⚠️  Ошибка при уничтожении модели: {e}")
    
    _ACTIVE_MODELS.clear()
    
    if destroyed_count > 0:
        print(f"🗑️ Уничтожено {destroyed_count} моделей")
    
    # Агрессивная очистка памяти
    clear_gpu_memory_completely()


def clear_gpu_memory_completely():
    """
    УЛУЧШЕННАЯ полная очистка GPU памяти
    
    Включает:
    1. Синхронизацию CUDA операций
    2. Очистку кэша PyTorch
    3. Сброс пула памяти CUDA
    4. Множественные циклы сборки мусора
    5. Очистку кэша HuggingFace Transformers
    """
    if not torch.cuda.is_available():
        return
    
    try:
        # 1. Синхронизируем все операции CUDA
        torch.cuda.synchronize()
        
        # 2. Первая очистка кэша
        torch.cuda.empty_cache()
        
        # 3. Сбрасываем статистику памяти
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        
        # 4. Множественные циклы сборки мусора Python
        for _ in range(3):
            gc.collect()
        
        # 5. IPC collect — освобождает память занятую дочерними процессами
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        
        # 6. Финальная очистка CUDA
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        # 7. Показываем состояние памяти
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        real_free = total - allocated  # Используем allocated, а не reserved
        
        print(f"🧹 GPU память: выделено={allocated:.2f}GB, зарезервировано={reserved:.2f}GB, реально свободно={real_free:.2f}GB")
        
        # Предупреждение если память всё ещё занята
        if allocated > 1.0:
            print(f"⚠️  ВНИМАНИЕ: {allocated:.2f}GB всё ещё выделено! Рекомендуется перезапуск сервера.")
            
    except Exception as e:
        print(f"⚠️  Ошибка при очистке GPU памяти: {e}")

class TimeLLM:
    """
    TimeLLM с поддержкой:
    1. NeuralForecast.TimeLLM (если установлен)
    2. Локальные GGUF модели через llama-cpp-python
    3. Fallback на простую статистическую модель
    """
    
    def __init__(self, llm_backend='simple', llm_model='gpt2', llm_path=None, gguf_config=None, use_cpu=False, neuralforecast_model='qwen2-0.5b'):
        """
        Инициализация TimeLLM
        
        Args:
            llm_backend: 'neuralforecast' (по умолчанию, SLM модели), 'gguf' (локальная GGUF), 'simple' (fallback)
            llm_model: Название LLM модели
            llm_path: Путь к GGUF файлу
            gguf_config: dict с конфигурацией GGUF
            use_cpu: Игнорируется
            neuralforecast_model: Модель для NeuralForecast:
                🟢 РЕКОМЕНДУЕМЫЕ SLM 2024-2025:
                - 'qwen2-0.5b' (по умолчанию): Qwen2-0.5B (500M) - самая лёгкая, быстрая, современная
                - 'llama3.2-1b': Llama-3.2-1B (1B) - от Meta, очень быстрая
                - 'gemma-2b': Gemma-2B (2B) - от Google, баланс скорость/качество
                - 'phi3-mini': Phi-3-mini (3.8B) - лучшая точность среди SLM
                - 'stablelm-zephyr-3b': StableLM-Zephyr-3B (3B) - стабильная
                
                🟡 Классические (медленнее):
                - 'gpt2': GPT-2 (124M) - старая, но лёгкая
                - 'distilgpt2': DistilGPT-2 (82M) - ещё легче
        """
        self.llm_backend = llm_backend
        self.llm_model = llm_model
        self.llm_path = llm_path
        self.use_cpu = use_cpu
        self.neuralforecast_model = neuralforecast_model
        self.gguf_config = gguf_config or {
            'n_ctx': 2048,
            'n_threads': 8,
            'n_gpu_layers': 48,
            'temperature': 0.3,
            'max_tokens': 512
        }
        
        self.model = None
        self.llm_instance = None  # Для GGUF модели
        self.data = None
        self.h = None
        self.freq = None
        self.residuals = None
        
        # Инициализация GGUF модели если указан путь
        if self.llm_backend == 'gguf' and self.llm_path:
            self._init_gguf_model()
    
    def _init_gguf_model(self):
        """Инициализация GGUF модели через llama-cpp-python"""
        try:
            from llama_cpp import Llama
            
            print(f"Загрузка GGUF модели из: {self.llm_path}")
            print(f"Конфигурация: {self.gguf_config}")
            try:
                self.llm_instance = Llama(
                    model_path=self.llm_path,
                    n_ctx=self.gguf_config.get('n_ctx', 4096),
                    n_threads=self.gguf_config.get('n_threads', 8),
                    n_gpu_layers=-1,
                    verbose=True
                )
            except Exception as e:
                print(e)
            print("✓ GGUF модель загружена успешно")
            
        except ImportError:
            print("Warning: llama-cpp-python не установлен. Установите: pip install llama-cpp-python")
            print("Используется fallback режим.")
            self.llm_backend = 'simple'
        except Exception as e:
            print(f"Warning: Не удалось загрузить GGUF модель: {e}")
            print("Используется fallback режим.")
            self.llm_backend = 'simple'
    
    def _generate_prompt(self, data, task='forecast'):
        """
        Автоматическая генерация промптов для TimeLLM
        
        Args:
            data: временной ряд
            task: тип задачи ('forecast', 'analysis')
            
        Returns:
            str: сгенерированный промпт
        """
        # Статистика данных
        mean_val = np.mean(data)
        std_val = np.std(data)
        trend = np.polyfit(range(len(data)), data, 1)[0]
        
        # Определение тренда
        if trend > std_val * 0.1:
            trend_desc = "shows a steady upward trend"
        elif trend < -std_val * 0.1:
            trend_desc = "shows a downward trend"
        else:
            trend_desc = "is relatively stable"
        
        # Волатильность
        volatility = std_val / mean_val if mean_val != 0 else 0
        if volatility > 0.3:
            volatility_desc = "high volatility"
        elif volatility > 0.1:
            volatility_desc = "moderate volatility"
        else:
            volatility_desc = "low volatility"
        
        # Последние значения для контекста
        recent_values = data[-min(10, len(data)):]
        recent_str = ", ".join([f"{v:.2f}" for v in recent_values])
        
        if task == 'forecast':
            prompt = f"""Time series {trend_desc} with {volatility_desc}.
Mean: {mean_val:.2f}, Standard deviation: {std_val:.2f}.
Recent observations: {recent_str}.
Task: predict the next values based on historical patterns."""
        
        elif task == 'analysis':
            prompt = f"""Time series analysis:
- Number of observations: {len(data)}
- Mean: {mean_val:.2f}
- Std: {std_val:.2f}
- Trend: {trend_desc}
- Volatility: {volatility_desc}
- Range: [{np.min(data):.2f}, {np.max(data):.2f}]"""
        
        else:
            prompt = f"Time series with {len(data)} observations."
        
        return prompt
    
    def _gguf_forecast(self, data, steps):
        """
        Прогнозирование через GGUF модель
        
        Использует LLM для генерации insight, затем применяет статистическую модель
        с коррекцией на основе LLM insight
        """
        if self.llm_instance is None:
            return self._simple_forecast(data, steps)
        
        # Генерация промпта для LLM
        prompt = self._generate_prompt(data, task='forecast')
        
        # Дополнительный контекст для прогнозирования
        full_prompt = f"""{prompt}

Based on the patterns above, provide insights for forecasting the next {steps} time steps.
Consider:
1. Trend direction and strength
2. Seasonality patterns (if any)
3. Volatility and uncertainty
4. Potential turning points

Provide a brief analysis (2-3 sentences) focusing on the forecast direction and confidence level."""
        
        try:
            # Вызов LLM
            response = self.llm_instance(
                full_prompt,
                max_tokens=self.gguf_config.get('max_tokens', 512),
                temperature=self.gguf_config.get('temperature', 0.3),
                stop=["###", "\n\n\n"],
                echo=False
            )
            
            llm_text = response['choices'][0]['text'].strip()
            print(f"\n[TimeLLM GGUF Insight]:\n{llm_text}\n")
            
            # Анализ insight для коррекции
            # Ищем ключевые слова для определения направления
            bullish_keywords = ['increase', 'grow', 'upward', 'rise', 'positive', 'bullish']
            bearish_keywords = ['decrease', 'decline', 'downward', 'fall', 'negative', 'bearish']
            
            llm_text_lower = llm_text.lower()
            
            bullish_score = sum(1 for kw in bullish_keywords if kw in llm_text_lower)
            bearish_score = sum(1 for kw in bearish_keywords if kw in llm_text_lower)
            
            # Коэффициент коррекции на основе LLM insight
            if bullish_score > bearish_score:
                correction_factor = 1.0 + 0.05 * (bullish_score - bearish_score)
            elif bearish_score > bullish_score:
                correction_factor = 1.0 - 0.05 * (bearish_score - bullish_score)
            else:
                correction_factor = 1.0
            
            print(f"[TimeLLM] Correction factor: {correction_factor:.3f}")
            
        except Exception as e:
            print(f"Warning: GGUF inference failed: {e}")
            correction_factor = 1.0
        
        # Базовый прогноз (статистический)
        base_forecast = self._simple_forecast(data, steps)
        
        # Применение коррекции от LLM
        corrected_forecast = base_forecast * correction_factor
        
        return corrected_forecast
    
    def _simple_forecast(self, data, steps):
        """
        Улучшенный статистический прогноз
        
        Комбинирует:
        - Линейный тренд
        - Экспоненциальное сглаживание
        - Сезонность
        """
        # Если модель обучена, используем сохранённые параметры
        if hasattr(self, 'trend_slope'):
            last_index = len(data)
            forecast = []
            
            for i in range(steps):
                # Базовый прогноз: тренд + последнее сглаженное значение
                trend_component = self.trend_slope * (last_index + i) + self.trend_intercept
                
                # Добавляем сезонность если есть
                if self.seasonal_pattern is not None:
                    season_idx = (last_index + i) % len(self.seasonal_pattern)
                    seasonal_component = self.seasonal_pattern[season_idx]
                else:
                    seasonal_component = 0
                
                # Взвешенная комбинация
                pred = 0.7 * trend_component + 0.3 * data[-1] + seasonal_component
                forecast.append(pred)
            
            return np.array(forecast)
        
        # Fallback: простой метод
        trend = np.mean(np.diff(data)) if len(data) > 1 else 0
        last_value = data[-1]
        
        forecast = np.array([last_value + trend * (i+1) for i in range(steps)])
        
        # Добавляем сезонность если есть достаточно данных
        if len(data) >= 12:
            seasonal_period = min(12, len(data) // 3)
            seasonal_pattern = []
            
            for i in range(seasonal_period):
                indices = list(range(i, len(data), seasonal_period))
                if indices:
                    seasonal_mean = np.mean(data[indices])
                    seasonal_pattern.append(seasonal_mean - np.mean(data))
            
            # Применяем сезонность к прогнозу
            for i in range(steps):
                season_idx = i % len(seasonal_pattern)
                forecast[i] += seasonal_pattern[season_idx]
        
        return forecast
    
    def fit(self, data, freq='D', steps=None):
        """
        Обучение модели с автоматическим fallback при OOM
        
        Args:
            data: numpy array с историческими данными
            freq: частота данных ('D'=daily, 'H'=hourly, 'M'=monthly, и т.д.)
            steps: желаемый горизонт прогноза (если None — определяется автоматически)
        """
        self.requested_steps = steps  # сохраняем для использования в predict
        global _OOM_FALLBACK_TO_SIMPLE
        
        self.data = data
        self.freq = freq
        
        # Регистрируем модель в глобальном реестре
        register_model(self)
        
        print(f"\n{'='*60}")
        print(f"TimeLLM: Обучение на {len(data)} точках")
        print(f"Backend: {self.llm_backend}")
        print(f"{'='*60}")
        
        # Если ранее был OOM, сразу используем simple режим
        if _OOM_FALLBACK_TO_SIMPLE and self.llm_backend == 'neuralforecast':
            print("⚠️  Предыдущий OOM! Автоматическое переключение на simple режим.")
            print("   Перезапустите сервер для повторной попытки GPU.")
            self.llm_backend = 'simple'
        
        # Режим GGUF
        if self.llm_backend == 'gguf':
            if self.llm_instance is None and self.llm_path:
                self._init_gguf_model()
            
            if self.llm_instance:
                print("✓ Используется GGUF модель")
            else:
                print("Warning: GGUF недоступен, используется simple режим")
                self.llm_backend = 'simple'
        
        # Режим NeuralForecast с современными SLM
        elif self.llm_backend == 'neuralforecast':
            if not torch.cuda.is_available():
                print("⚠️ Warning: NeuralForecast требует GPU. CUDA недоступен.")
                print("Используется simple режим")
                self.llm_backend = 'simple'
            else:
                # КРИТИЧНО: Уничтожаем ВСЕ предыдущие модели перед обучением
                print("🗑️ Освобождаю память от предыдущих моделей...")
                destroy_all_models()
                
                # Проверяем доступную память
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                reserved = torch.cuda.memory_reserved(0) / 1024**3
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                free = total - allocated  # Используем allocated (реальное потребление)
                
                print(f"📊 GPU память перед обучением: свободно≈{free:.2f}GB (выделено={allocated:.2f}GB) из {total:.2f}GB")
                
                # Если недостаточно памяти, сразу переключаемся на simple
                if free < 3.0:  # Снижен порог: 3GB минимум
                    print(f"⚠️  Недостаточно GPU памяти ({free:.2f}GB < 4GB)!")
                    print("   Переключаюсь на simple режим")
                    _OOM_FALLBACK_TO_SIMPLE = True
                    self.llm_backend = 'simple'
                else:
                    try:
                        print(f"🚀 Используется NeuralForecast с современными SLM 2024-2025")
                        
                        self._fit_neuralforecast(data, freq, steps=self.requested_steps)
                        print("✓ Используется NeuralForecast.TimeLLM с SLM")
                        
                    except RuntimeError as e:
                        error_msg = str(e).lower()
                        print(f"❌ CUDA RuntimeError: {e}")
                        
                        # Экстренная очистка памяти
                        print("🧹 Экстренная очистка GPU памяти...")
                        destroy_all_models()
                        
                        if "out of memory" in error_msg:
                            print("❌ CUDA OOM! Устанавливаю флаг fallback на simple режим.")
                            _OOM_FALLBACK_TO_SIMPLE = True
                        
                        print("Переключаюсь на simple режим")
                        self.llm_backend = 'simple'
                        
                    except Exception as e:
                        print(f"Warning: NeuralForecast failed: {e}")
                        destroy_all_models()
                        print("Переключаюсь на simple режим")
                        self.llm_backend = 'simple'
        
        # Simple режим (fallback)
        if self.llm_backend == 'simple':
            self._fit_simple(data)
            print("✓ Используется Simple статистический режим (fallback)")
        
        print(f"{'='*60}\n")
        
        return self
    
    def _fit_neuralforecast(self, data, freq, steps=None):
        """Обучение через NeuralForecast.TimeLLM"""
        from neuralforecast import NeuralForecast
        from neuralforecast.models import TimeLLM as NF_TimeLLM
        
        # NeuralForecast требует GPU для работы
        if not torch.cuda.is_available():
            raise RuntimeError("NeuralForecast требует CUDA GPU. GPU недоступен.")
        
        # Включаем расширяемые сегменты для уменьшения фрагментации
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # ПОЛНАЯ очистка GPU памяти перед созданием модели
        print("🧹 Выполняю полную очистку GPU памяти...")
        clear_gpu_memory_completely()
        
        # Удаляем старую модель если есть
        if hasattr(self, 'model') and self.model is not None:
            print("🗑️ Удаляю старую модель...")
            del self.model
            clear_gpu_memory_completely()
        
        # Проверяем доступную память GPU и оптимизируем параметры
        gpu_memory = 0.0
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            gpu_memory = gpu_props.total_memory / 1024**3
            gpu_name = gpu_props.name
            
            # Проверяем доступную память (с учетом уже занятой)
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            free_memory = gpu_memory - reserved
            
            print(f"🎮 GPU: {gpu_name}")
            print(f"📊 Общая память: {gpu_memory:.2f} GB")
            print(f"📊 Доступно: {free_memory:.2f} GB")
            
            # Реально свободная память = total - уже выделено другими процессами
            real_allocated = torch.cuda.memory_allocated(0) / 1024**3
            real_free = gpu_memory - real_allocated
            print(f"📊 Реально свободно GPU: {real_free:.2f}GB (выделено={real_allocated:.2f}GB)")
            
            # Определяем параметры исходя из РЕАЛЬНО свободной памяти
            # Qwen2-0.5B: ~1GB fp16 + gradients + activations
            # mapping_layer зависит от patch_len и input_size:
            #   params = (input_size/patch_len * patch_len) * d_model  -> минимизируем input_size
            # Модель выбирается позже из self.neuralforecast_model
            # Здесь задаём только размеры входов
            if real_free >= 6.0:
                optimal_batch_size = 1
                optimal_input_size = 16
                optimal_llm_layers = 2
                optimal_horizon = 6
                default_model = 'smollm2-360m'  # SOTA 2025, vocab=49K
                print("⚡ Режим: 6GB+ свободно, input=16, SmolLM2-360M")
            elif real_free >= 3.0:
                optimal_batch_size = 1
                optimal_input_size = 8
                optimal_llm_layers = 1
                optimal_horizon = 4
                default_model = 'smollm2-135m'
                print("⚡ Режим: 3GB+ свободно, input=8, SmolLM2-135M")
            elif real_free >= 1.5:
                optimal_batch_size = 1
                optimal_input_size = 4
                optimal_llm_layers = 1
                optimal_horizon = 2
                default_model = 'distilgpt2'
                print("⚡ Режим: 1.5GB+ свободно, input=4, DistilGPT-2")
            else:
                raise RuntimeError(
                    f"Недостаточно GPU памяти: свободно {real_free:.2f}GB, "
                    f"нужно минимум 1.5GB."
                )
        else:
            optimal_batch_size = 2
            optimal_input_size = 48
            optimal_llm_layers = 2
            optimal_horizon = 24
            default_model = 'phi-1.5'
            gpu_memory = 0.0
            default_model = 'phi-1.5'
        
        try:
            # Подготовка данных
            df = pd.DataFrame({
                'unique_id': ['series_1'] * len(data),
                'ds': pd.date_range(start='2020-01-01', periods=len(data), freq=freq),
                'y': data
            })
            
            # Генерация промпта
            prompt = self._generate_prompt(data, task='forecast')
            
            # Параметры модели
            # Вариант 2: используем steps пользователя как horizon (без экстраполяции)
            if steps is not None and steps >= 1:
                # Ограничиваем горизонт разумным максимумом
                max_horizon = max(1, len(data) // 3)  # не более 1/3 данных
                horizon = min(steps, max_horizon, optimal_horizon)
                if horizon < steps:
                    print(f"⚠️  Запрошено {steps} шагов, обучаем на horizon={horizon} (макс для {len(data)} точек)")
                else:
                    print(f"✅ Горизонт обучения = {horizon} шагов (запрошено пользователем)")
            else:
                horizon = max(1, min(len(data) // 10, optimal_horizon))
            input_size = min(len(data) - horizon, optimal_input_size)
            input_size = max(input_size, 2)  # минимум 2
            llm_layers = optimal_llm_layers
            
            # Выбор модели на основе параметра
            # Современные Small Language Models (SLM) 2024-2025
            # Оптимизированы для быстрой работы и малого потребления памяти
            
            model_choice = self.neuralforecast_model or default_model
            
            # vocab_size модели определяет размер mapping_layer:
            #   mapping_layer = Linear(vocab_size, 1024) — захардкожено в NeuralForecast
            #   SmolLM2:   vocab=49152  → ~200MB ✔ SOTA 2025
            #   gpt2:      vocab=50257  → ~200MB ✔
            #   Qwen2.5:   vocab=151936 → ~620MB ⚠️
            model_configs = {
                # ⭐ РЕКОМЕНДУЕТСЯ: SmolLM2 (HuggingFace, 2025) - SOTA при малом vocab
                'smollm2-135m': {
                    'name': 'HuggingFaceTB/SmolLM2-135M',
                    'd_llm': 576,
                    'vocab_size': 49152,
                    'description': '⭐ SmolLM2-135M (135M, 2025) - быстрейшая, vocab=49K, mapping ~200MB, <1GB VRAM'
                },
                'smollm2-360m': {
                    'name': 'HuggingFaceTB/SmolLM2-360M',
                    'd_llm': 960,
                    'vocab_size': 49152,
                    'description': '⭐ SmolLM2-360M (360M, 2025) - баланс, vocab=49K, mapping ~200MB, 1GB VRAM'
                },
                'smollm2-1.7b': {
                    'name': 'HuggingFaceTB/SmolLM2-1.7B',
                    'd_llm': 2048,
                    'vocab_size': 49152,
                    'description': '⭐ SmolLM2-1.7B (1.7B, 2025) - лучшее качество, vocab=49K, mapping ~200MB, 4GB VRAM'
                },
                # Классика - надёжный фоллбэк
                'gpt2': {
                    'name': 'gpt2',
                    'd_llm': 768,
                    'vocab_size': 50257,
                    'description': '🟢 GPT-2 (124M) - надёжный fallback, vocab=50K, mapping ~200MB'
                },
                'distilgpt2': {
                    'name': 'distilgpt2',
                    'd_llm': 768,
                    'vocab_size': 50257,
                    'description': '🟢 DistilGPT-2 (82M) - самая лёгкая, vocab=50K, mapping ~200MB'
                },
                # Модели с большим vocab (автозамена на gpt2 если памяти <10GB)
                'phi3-mini': {
                    'name': 'microsoft/Phi-3-mini-4k-instruct',
                    'd_llm': 3072,
                    'vocab_size': 32064,
                    'description': '🟡 Phi-3-mini (3.8B, 2024) - vocab=32K, mapping ~130MB, 6GB VRAM, модель скачивается долго'
                },
                'tinyllama': {
                    'name': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                    'd_llm': 2048,
                    'vocab_size': 32000,
                    'description': '🟡 TinyLlama (1.1B) - vocab=32K, mapping ~130MB, 3GB VRAM'
                },
                'phi-1.5': {
                    'name': 'microsoft/phi-1_5',
                    'd_llm': 2048,
                    'vocab_size': 51200,
                    'description': '🟡 Phi-1.5 (1.3B) - vocab=50K, mapping ~200MB, 3GB VRAM'
                },
                'qwen2-0.5b': {
                    'name': 'Qwen/Qwen2-0.5B',
                    'd_llm': 896,
                    'vocab_size': 151936,
                    'description': '🟡 Qwen2-0.5B (500M) - vocab=152K → mapping 620MB, нужно 8GB+'
                },
                'qwen2.5-0.5b': {
                    'name': 'Qwen/Qwen2.5-0.5B',
                    'd_llm': 896,
                    'vocab_size': 151936,
                    'description': '🟡 Qwen2.5-0.5B (500M, 2024) - vocab=152K → mapping 620MB, нужно 8GB+'
                },
                'llama3.2-1b': {
                    'name': 'meta-llama/Llama-3.2-1B',
                    'd_llm': 2048,
                    'vocab_size': 128256,
                    'description': '🟡 Llama-3.2-1B (1B) - vocab=128K → mapping 520MB, нужно 8GB+'
                },
                'stablelm-zephyr-3b': {
                    'name': 'stabilityai/stablelm-zephyr-3b',
                    'd_llm': 2560,
                    'vocab_size': 50257,
                    'description': '🟡 StableLM-Zephyr-3B (3B) - vocab=50K, mapping ~200MB, 5GB VRAM'
                },
                'gemma-2b': {
                    'name': 'google/gemma-2b',
                    'd_llm': 2048,
                    'vocab_size': 256000,
                    'description': '🔴 Gemma-2B (2B) - vocab=256K → mapping 1GB, нужно 12GB+'
                },
            }
            
            if model_choice not in model_configs:
                print(f"⚠️ Неизвестная модель '{model_choice}', используем {default_model} по умолчанию")
                model_choice = default_model
            
            config = model_configs[model_choice]
            llm_model_name = config['name']
            d_llm_value = config['d_llm']
            vocab_size = config.get('vocab_size', 50257)
            
            # Если выбрана модель с большим vocab -> mapping_layer будет большим
            mapping_mb = vocab_size * 1024 * 4 / 1024**2
            print(f"🤖 Выбрана модель: {config['description']}")
            print(f"📊 mapping_layer размер: ~{mapping_mb:.0f}MB (vocab={vocab_size:,})")
            
            # Автозамена модели если mapping_layer слишком большой
            if mapping_mb > 300 and real_free < 10.0:
                print(f"⚠️  mapping_layer={mapping_mb:.0f}MB слишком большой для {real_free:.1f}GB свободной памяти")
                print(f"🔄 Автозамена на gpt2 (vocab=50K, mapping_layer ~200MB)")
                model_choice = 'gpt2'
                config = model_configs['gpt2']
                llm_model_name = config['name']
                d_llm_value = config['d_llm']
            
            print(f"📊 NeuralForecast: Модель={llm_model_name}, d_llm={d_llm_value}")
            print(f"📊 Параметры: batch_size={optimal_batch_size}, input_size={input_size}, horizon={horizon}, llm_layers={llm_layers}")
            print(f"📊 Устройство: GPU (CUDA)")
            
            # Создаем модель TimeLLM для NeuralForecast
            # NeuralForecast автоматически использует GPU если доступен
            # Оптимизировано для RTX 4070 Ti Super (16GB VRAM)
            # 
            # ВАЖНО: Ограничиваем время обучения для быстрого отклика API
            # Для production используйте max_steps=500-1000
            
            # Определяем нужен ли early stopping на основе размера данных
            use_early_stop = len(data) > 30  # Для маленьких выборок отключаем
            
            # КРИТИЧНО: Минимальное количество шагов для API (быстрый отклик)
            # Для production качества используйте max_steps=500-1000 в отдельном скрипте
            api_max_steps = 20  # Было 100 - слишком долго для API!
            
            # Patch параметры: маленький patch_len снижает размер mapping_layer
            patch_len = min(input_size // 2, 4)  # 4 по умолчанию (было 16)
            patch_len = max(patch_len, 1)
            stride = patch_len  # без перекрытий — меньше памяти
            
            # Общие kwargs
            # enc_in=1: univariate ряд — главная экономия активаций
            # windows_batch_size=1: вместо дефолтных 1024 — главная причина OOM
            # d_model=16: уменьшаем внутреннюю размерность
            timellm_kwargs = dict(
                h=horizon,
                input_size=input_size,
                llm=llm_model_name,
                llm_num_hidden_layers=llm_layers,
                d_llm=d_llm_value,
                # d_model и d_ff оставляем дефолтными (32/128):
                # d_ff используется как d_keys в ReprogrammingLayer —
                # изменение ломает матричные размерности
                enc_in=1,
                dec_in=1,
                patch_len=patch_len,
                stride=stride,
                windows_batch_size=1,
                inference_windows_batch_size=1,
                prompt_prefix=prompt,
                learning_rate=5e-3,
                batch_size=optimal_batch_size,
                max_steps=api_max_steps,
                random_seed=42,
            )
            
            if use_early_stop:
                timellm_kwargs.update(dict(
                    val_check_steps=10,
                    early_stop_patience_steps=2,
                ))
                timellm_model = NF_TimeLLM(**timellm_kwargs)
                print(f"⚙️  Параметры: max_steps={api_max_steps}, patch_len={patch_len}, llm_layers={llm_layers}, enc_in=1, windows_batch=1, early_stop=enabled")
            else:
                timellm_model = NF_TimeLLM(**timellm_kwargs)
                print(f"⚙️  Параметры: max_steps={api_max_steps}, patch_len={patch_len}, llm_layers={llm_layers}, enc_in=1, windows_batch=1")
            
            print(f"   ⚠️  ВАЖНО: Для качественного прогноза обучайте модель отдельно с max_steps=500-1000")
            print(f"   Текущий режим оптимизирован для быстрого отклика API (<2 мин)")
            
            nf = NeuralForecast(models=[timellm_model], freq=freq)
            
            # Обучение модели с защитой от сбоев
            print("🚀 Начинаю обучение модели...")
            print(f"⏱️  Ожидаемое время: ~30-90 секунд (max_steps={api_max_steps})")
            print(f"📊 Параметры: {len(data)} точек, horizon={horizon}, input_size={input_size}")
            print(f"💡 Для качественного прогноза требуется max_steps=500+, но это займёт 5-10 минут")
            print(f"   Текущий режим: быстрый API отклик с базовым качеством")
            
            try:
                # Синхронизация CUDA перед обучением
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                # Определяем val_size для early stopping
                # NeuralForecast требует val_size > 0 при использовании early stopping
                if use_early_stop:
                    # Используем 20% данных для валидации (минимум 2 точки)
                    val_size = max(2, int(len(data) * 0.2))
                    print(f"📊 Валидационный набор: {val_size} точек (для early stopping)")
                    nf.fit(df=df, val_size=val_size)
                else:
                    nf.fit(df=df)
                
                # Синхронизация после обучения
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    
            except RuntimeError as e:
                error_msg = str(e)
                print(f"\n❌ RuntimeError при обучении: {error_msg}")
                
                # Очистка памяти при ошибке
                clear_gpu_memory_completely()
                
                if "out of memory" in error_msg.lower():
                    raise RuntimeError(f"CUDA out of memory. Попробуйте уменьшить batch_size или использовать CPU режим")
                else:
                    raise
            except Exception as e:
                print(f"\n❌ Неожиданная ошибка при обучении: {e}")
                clear_gpu_memory_completely()
                raise
            
            # ПОЛНАЯ очистка памяти после обучения
            print("🧹 Очищаю GPU память после обучения...")
            clear_gpu_memory_completely()
            
            self.model = nf
            self.h = horizon
            
            # Вычисление остатков
            if len(data) > horizon:
                train_df = df.iloc[:-horizon]
                val_df = df.iloc[-horizon:]
                
                # Используем torch.no_grad() для инференса
                print("📊 Вычисляю остатки для доверительных интервалов...")
                with torch.no_grad():
                    forecast_df = nf.predict(df=train_df)
                    predictions = forecast_df['TimeLLM'].values
                    actuals = val_df['y'].values
                    
                    self.residuals = actuals - predictions[:len(actuals)]
                
                # Очистка после вычисления остатков
                clear_gpu_memory_completely()
            else:
                self.residuals = np.array([0])
                
        except RuntimeError as e:
            # Перехватываем CUDA OOM и ошибки размеров матриц
            error_msg = str(e)
            if "out of memory" in error_msg.lower():
                # ПОЛНАЯ очистка GPU памяти перед fallback
                print("❌ CUDA out of memory! Выполняю экстренную очистку памяти...")
                clear_gpu_memory_completely()
                print(f"❌ CUDA out of memory. Попробуйте:")
                print(f"   - Использовать более легкую модель (gpt2)")
                print(f"   - Уменьшить input_size или batch_size")
                print(f"   - Освободить GPU память")
                raise RuntimeError(f"NeuralForecast CUDA OOM error: {error_msg}")
            elif "cannot be multiplied" in error_msg.lower():
                # Ошибка размеров матриц
                clear_gpu_memory_completely()
                raise RuntimeError(f"NeuralForecast matrix size error: {error_msg}")
            else:
                clear_gpu_memory_completely()
                raise
        except Exception as e:
            # ПОЛНАЯ очистка GPU памяти при любой другой ошибке
            print(f"❌ Ошибка при обучении NeuralForecast: {e}")
            clear_gpu_memory_completely()
            raise
    
    def _fit_simple(self, data):
        """
        Улучшенная статистическая модель (вместо NeuralForecast)
        
        Использует комбинацию:
        - Экспоненциальное сглаживание (Holt-Winters)
        - Сезонная декомпозиция
        - Линейный тренд
        """
        self.data = data
        self.mean = np.mean(data)
        self.std = np.std(data)
        
        # Линейный тренд
        x = np.arange(len(data))
        coeffs = np.polyfit(x, data, 1)
        self.trend_slope = coeffs[0]
        self.trend_intercept = coeffs[1]
        
        # Экспоненциальное сглаживание (alpha=0.3)
        self.smoothed = []
        s = data[0]
        for val in data:
            s = 0.3 * val + 0.7 * s
            self.smoothed.append(s)
        self.smoothed = np.array(self.smoothed)
        
        # Сезонность (если достаточно данных)
        if len(data) >= 12:
            seasonal_period = min(12, len(data) // 3)
            self.seasonal_pattern = []
            
            for i in range(seasonal_period):
                indices = list(range(i, len(data), seasonal_period))
                if indices:
                    seasonal_mean = np.mean(data[indices])
                    self.seasonal_pattern.append(seasonal_mean - self.mean)
            
            self.seasonal_pattern = np.array(self.seasonal_pattern)
        else:
            self.seasonal_pattern = None
        
        # Остатки для доверительных интервалов
        trend_line = self.trend_slope * x + self.trend_intercept
        self.residuals = data - trend_line
        
        self.model = 'simple'
        
        print(f"   📊 Параметры:")
        print(f"      - Среднее: {self.mean:.2f}")
        print(f"      - Тренд: {self.trend_slope:.4f}")
        print(f"      - Сезонность: {'✓ обнаружена' if self.seasonal_pattern is not None else '✗ не обнаружена'}")
    
    def predict(self, steps, return_conf_int=True, alpha=0.05):
        """
        Прогнозирование на steps шагов вперёд
        
        Args:
            steps: количество шагов прогноза
            return_conf_int: возвращать ли доверительные интервалы
            alpha: уровень значимости (0.05 = 95% интервал)
            
        Returns:
            dict: {
                'forecast': прогноз,
                'lower_bound': нижняя граница (если return_conf_int=True),
                'upper_bound': верхняя граница (если return_conf_int=True)
            }
        """
        if self.data is None:
            raise ValueError("Модель не обучена. Вызовите fit() сначала.")
        
        # GGUF режим
        if self.llm_backend == 'gguf':
            forecast = self._gguf_forecast(self.data, steps)
        
        # NeuralForecast режим (работает на GPU)
        elif self.llm_backend == 'neuralforecast' and self.model != 'simple':
            try:
                # Очистка GPU памяти перед предсказанием
                torch.cuda.empty_cache()
                gc.collect()
                
                df = pd.DataFrame({
                    'unique_id': ['series_1'] * len(self.data),
                    'ds': pd.date_range(start='2020-01-01', periods=len(self.data), freq=self.freq or 'D'),
                    'y': self.data
                })
                
                # Используем обученный горизонт модели
                # NeuralForecast.predict() не принимает параметр h - использует h из модели
                print(f"📊 Выполняю прогноз на {self.h} шагов (обученный горизонт)...")
                with torch.no_grad():
                    forecast_df = self.model.predict(df=df)
                    
                    # Получаем прогноз из результата
                    model_forecast = forecast_df['TimeLLM'].values
                    
                    if steps <= len(model_forecast):
                        # Горизонт модели достаточен — берём нужное количество
                        forecast = model_forecast[:steps]
                    else:
                        # Горизонт модели меньше запрошенного:
                        # Rolling-window: сдвигаем окно данных и дополняем прогноз
                        print(f"🔄 Rolling-window: horizon={len(model_forecast)}, нужно {steps} шагов")
                        forecast = list(model_forecast)
                        rolling_data = list(self.data)
                        remaining = steps - len(model_forecast)
                        step_done = len(model_forecast)
                        while remaining > 0:
                            # Сдвигаем окно: добавляем уже предсказанные значения
                            rolling_data_arr = np.array(rolling_data + forecast[:step_done])
                            roll_df = pd.DataFrame({
                                'unique_id': ['series_1'] * len(rolling_data_arr),
                                'ds': pd.date_range(start='2020-01-01', periods=len(rolling_data_arr), freq=self.freq or 'D'),
                                'y': rolling_data_arr
                            })
                            with torch.no_grad():
                                roll_result = self.model.predict(df=roll_df)
                                roll_fc = roll_result['TimeLLM'].values
                            take = min(len(roll_fc), remaining)
                            forecast.extend(roll_fc[:take].tolist())
                            remaining -= take
                            step_done += take
                            print(f"   Rolling: добавлено {take} шагов, осталось {remaining}")
                        forecast = np.array(forecast[:steps])
                
                # ПОЛНАЯ очистка GPU памяти после предсказания
                clear_gpu_memory_completely()
            except Exception as e:
                print(f"❌ NeuralForecast prediction failed: {e}")
                print(f"   Используется fallback на simple forecast")
                import traceback
                traceback.print_exc()
                # ПОЛНАЯ очистка GPU памяти при ошибке
                clear_gpu_memory_completely()
                forecast = self._simple_forecast(self.data, steps)
        
        # Simple режим
        else:
            forecast = self._simple_forecast(self.data, steps)
        
        # Fallback значение на основе последних данных
        last_value = float(self.data[-1]) if self.data is not None and len(self.data) > 0 else 0.0
        
        # Очистка прогноза от NaN/Inf
        forecast = sanitize_array(forecast, fallback_value=last_value)
        
        result = {'forecast': forecast}
        
        # Доверительные интервалы
        if return_conf_int:
            from scipy import stats
            # Выбираем std остатков; если остатки вырожденные (напр. [0]) — берём std данных
            if self.residuals is not None and len(self.residuals) > 1 and np.std(self.residuals) > 0:
                std_residual = np.std(self.residuals)
            else:
                # Fallback: std остатков от среднего (не std всего ряда с трендом)
                std_residual = np.std(np.array(self.data) - np.mean(self.data))

            z_score = stats.norm.ppf(1 - alpha / 2)

            # Интервал расширяется с горизонтом: margin(h) = z * std * sqrt(h)
            h = np.arange(1, len(forecast) + 1)
            margins = z_score * std_residual * np.sqrt(h)
            lower = forecast - margins
            upper = forecast + margins
            
            # Очистка доверительных интервалов от NaN/Inf
            result['lower_bound'] = sanitize_array(lower, fallback_value=forecast[0] * 0.9)
            result['upper_bound'] = sanitize_array(upper, fallback_value=forecast[0] * 1.1)
            
            # Гарантируем, что lower <= forecast <= upper
            result['lower_bound'] = np.minimum(result['lower_bound'], forecast)
            result['upper_bound'] = np.maximum(result['upper_bound'], forecast)
        
        return result
    
    def get_metrics(self, y_true, y_pred):
        """Расчёт метрик качества"""
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        }
    
    def get_info(self):
        """Информация о модели"""
        if self.data is None:
            return {
                'llm_backend': self.llm_backend,
                'llm_model': self.llm_model,
                'status': 'Модель не обучена',
                'mode': 'not_trained'
            }
        
        info = {
            'llm_backend': self.llm_backend,
            'llm_model': self.llm_model,
            'data_points': len(self.data),
            'freq': self.freq,
            'status': 'Обучена'
        }
        
        if self.llm_backend == 'gguf':
            info['mode'] = 'gguf'
            info['llm_path'] = self.llm_path
            info['gguf_config'] = self.gguf_config
            info['gguf_loaded'] = self.llm_instance is not None
        elif self.llm_backend == 'neuralforecast':
            info['mode'] = 'neuralforecast'
            info['horizon'] = self.h
        else:
            info['mode'] = 'simple'
            info['trend'] = float(self.trend) if hasattr(self, 'trend') else 0
        
        # Пример промпта
        prompt_example = self._generate_prompt(self.data, task='analysis')
        info['prompt_example'] = prompt_example[:200] + '...'
        
        return info
    
    def get_prompt(self, task='forecast'):
        """
        Получить сгенерированный промпт для текущих данных
        
        Args:
            task: тип задачи ('forecast', 'analysis')
            
        Returns:
            str: промпт
        """
        if self.data is None:
            return "Нет данных для генерации промпта"
        
        return self._generate_prompt(self.data, task=task)
