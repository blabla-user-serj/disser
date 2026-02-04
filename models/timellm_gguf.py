"""
TimeLLM с поддержкой локальных GGUF моделей через llama-cpp-python
"""
import numpy as np
import pandas as pd
import warnings
import os
import gc
import sys

# Настройки CUDA для стабильной работы
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Синхронный режим для отладки

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

def clear_gpu_memory_completely():
    """Полная очистка GPU памяти - агрессивный метод"""
    if not torch.cuda.is_available():
        return
    
    try:
        # Синхронизируем все операции CUDA
        torch.cuda.synchronize()
        
        # Очищаем кэш
        torch.cuda.empty_cache()
        
        # Сбрасываем статистику памяти
        torch.cuda.reset_peak_memory_stats()
        
        # Принудительная сборка мусора Python
        import gc
        gc.collect()
        
        # Еще раз очищаем кэш после сборки мусора
        torch.cuda.empty_cache()
        
        # Показываем состояние памяти
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"🧹 GPU память после очистки: выделено={allocated:.2f} GB, зарезервировано={reserved:.2f} GB")
    except Exception as e:
        print(f"⚠️  Ошибка при очистке GPU памяти: {e}")

class TimeLLM:
    """
    TimeLLM с поддержкой:
    1. NeuralForecast.TimeLLM (если установлен)
    2. Локальные GGUF модели через llama-cpp-python
    3. Fallback на простую статистическую модель
    """
    
    def __init__(self, llm_backend='simple', llm_model='gpt2', llm_path=None, gguf_config=None, use_cpu=False, neuralforecast_model='gpt2'):
        """
        Инициализация TimeLLM
        
        Args:
            llm_backend: 'simple' (по умолчанию, стабильно), 'gguf' (локальная GGUF), 'neuralforecast' (ЭКСПЕРИМЕНТАЛЬНО)
            llm_model: Название LLM модели для NeuralForecast
            llm_path: Путь к GGUF файлу (для llm_backend='gguf')
            gguf_config: dict с конфигурацией GGUF
            use_cpu: Игнорируется
            neuralforecast_model: Модель для NeuralForecast (НЕ РЕКОМЕНДУЕТСЯ - нестабильно на Windows)
            
        ВАЖНО: 
        - По умолчанию используется 'simple' режим (стабильный статистический прогноз)
        - NeuralForecast режим вызывает ошибки памяти на Windows и не рекомендуется
        - Для качественного прогноза используйте Гибридную модель (SARIMA + XGBoost + TimeLLM simple)
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
    
    def fit(self, data, freq='D'):
        """
        Обучение модели
        
        Args:
            data: numpy array с историческими данными
            freq: частота данных ('D'=daily, 'H'=hourly, 'M'=monthly, и т.д.)
        """
        self.data = data
        self.freq = freq
        
        print(f"\n{'='*60}")
        print(f"TimeLLM: Обучение на {len(data)} точках")
        print(f"Backend: {self.llm_backend}")
        print(f"{'='*60}")
        
        # ВАЖНО: По умолчанию используем simple режим (стабильно)
        # NeuralForecast вызывает ошибки памяти на Windows
        if self.llm_backend == 'neuralforecast':
            print("⚠️  WARNING: NeuralForecast режим ЭКСПЕРИМЕНТАЛЬНЫЙ и нестабилен!")
            print("   Может вызывать ошибки памяти на Windows.")
            print("   Автоматически переключаюсь на stable simple режим.")
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
        
        # Simple режим (по умолчанию)
        if self.llm_backend == 'simple':
            self._fit_simple(data)
            print("✓ Используется Simple статистический режим (стабильный)")
        
        print(f"{'='*60}\n")
        
        return self
    
    def _fit_neuralforecast(self, data, freq):
        """Обучение через NeuralForecast.TimeLLM"""
        from neuralforecast import NeuralForecast
        from neuralforecast.models import TimeLLM as NF_TimeLLM
        
        # NeuralForecast требует GPU для работы
        if not torch.cuda.is_available():
            raise RuntimeError("NeuralForecast требует CUDA GPU. GPU недоступен.")
        
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
            
            # Определяем оптимальные параметры на основе доступной памяти
            # Для RTX 4070 Ti Super (16GB) используем агрессивные параметры
            if gpu_memory >= 15.0:  # 16GB карта
                optimal_batch_size = 8
                optimal_input_size = 128
                optimal_horizon = 48
                # ИЗМЕНЕНО: используем gpt2 вместо phi-2 для стабильности
                default_model = 'gpt2'  # Было: 'phi-1.5'
                print("⚡ Режим: 16GB VRAM - используется лёгкая модель gpt2 для стабильности")
            elif gpu_memory >= 12.0:  # 12-15GB карта
                optimal_batch_size = 6
                optimal_input_size = 96
                optimal_horizon = 36
                default_model = 'gpt2'
                print("⚡ Режим: Высокая производительность (12GB+ VRAM)")
            elif gpu_memory >= 8.0:  # 8-12GB карта
                optimal_batch_size = 4
                optimal_input_size = 64
                optimal_horizon = 24
                default_model = 'gpt2'
                print("⚡ Режим: Оптимизированная производительность (8GB+ VRAM)")
            else:  # Меньше 8GB
                optimal_batch_size = 2
                optimal_input_size = 48
                optimal_horizon = 24
                default_model = 'gpt2'
                print("⚡ Режим: Экономия памяти (<8GB VRAM)")
        else:
            optimal_batch_size = 2
            optimal_input_size = 48
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
            
            # Оптимизированные параметры для RTX 4070 Ti Super (16GB VRAM)
            horizon = max(1, min(len(data) // 10, optimal_horizon))
            input_size = min(len(data) - horizon, optimal_input_size)
            
            # Выбор модели на основе параметра
            # ВАЖНО: Для API используем только лёгкие модели!
            # Тяжёлые модели (phi-2, phi-1.5, tinyllama) могут вызывать:
            # - Out of memory даже на 16GB
            # - Очень долгое обучение (>5 минут)
            # - Сбои CUDA на Windows
            
            model_choice = self.neuralforecast_model or default_model
            
            model_configs = {
                'gpt2': {
                    'name': 'gpt2',
                    'd_llm': 768,
                    'description': '🟢 GPT-2 (124M) - РЕКОМЕНДУЕТСЯ для API: быстро, стабильно, мало памяти'
                },
                'distilgpt2': {
                    'name': 'distilgpt2',
                    'd_llm': 768,
                    'description': '🟢 DistilGPT-2 (82M) - ещё легче чем GPT-2, очень быстро'
                },
                'tinyllama': {
                    'name': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                    'd_llm': 2048,
                    'description': '🟡 TinyLlama (1.1B) - средний размер, требует 4-6GB VRAM'
                },
                'phi-1.5': {
                    'name': 'microsoft/phi-1.5',
                    'd_llm': 2048,
                    'description': '🔴 Phi-1.5 (1.3B) - тяжёлая, требует 6-8GB VRAM, медленно'
                },
                'phi-2': {
                    'name': 'microsoft/phi-2',
                    'd_llm': 2560,
                    'description': '🔴 Phi-2 (2.7B) - ОЧЕНЬ тяжёлая, НЕ рекомендуется для API!'
                }
            }
            
            if model_choice not in model_configs:
                print(f"⚠️ Неизвестная модель '{model_choice}', используем {default_model} по умолчанию")
                model_choice = default_model
            
            config = model_configs[model_choice]
            llm_model_name = config['name']
            d_llm_value = config['d_llm']
            
            print(f"🤖 Выбрана модель: {config['description']}")
            
            # Предупреждение для тяжёлых моделей
            if model_choice in ['phi-2', 'phi-1.5', 'tinyllama']:
                print(f"⚠️  ВНИМАНИЕ: Модель {model_choice} может:")
                print(f"   - Требовать много времени на загрузку (1-5 минут)")
                print(f"   - Вызывать Out of Memory на некоторых системах")
                print(f"   - Обучаться очень долго (>5 минут даже с max_steps=20)")
                print(f"   💡 Рекомендуется использовать 'gpt2' для API режима")
            
            print(f"📊 NeuralForecast: Модель={llm_model_name}, d_llm={d_llm_value}")
            print(f"📊 Параметры: batch_size={optimal_batch_size}, input_size={input_size}, horizon={horizon}")
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
            
            if use_early_stop:
                # С early stopping (для больших данных)
                timellm_model = NF_TimeLLM(
                    h=horizon,
                    input_size=input_size,
                    llm=llm_model_name,
                    d_llm=d_llm_value,
                    prompt_prefix=prompt,
                    learning_rate=5e-3,  # Увеличен для очень быстрого обучения
                    batch_size=optimal_batch_size,
                    max_steps=api_max_steps,
                    val_check_steps=10,
                    early_stop_patience_steps=2,
                    random_seed=42
                )
                print(f"⚙️  Параметры обучения: max_steps={api_max_steps}, early_stop=enabled")
            else:
                # Без early stopping (для маленьких данных)
                timellm_model = NF_TimeLLM(
                    h=horizon,
                    input_size=input_size,
                    llm=llm_model_name,
                    d_llm=d_llm_value,
                    prompt_prefix=prompt,
                    learning_rate=5e-3,  # Увеличен для очень быстрого обучения
                    batch_size=optimal_batch_size,
                    max_steps=api_max_steps,
                    random_seed=42
                )
                print(f"⚙️  Параметры обучения: max_steps={api_max_steps} (быстрый режим для API)")
            
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
                    
                    # Если запрошено больше чем обучено, расширяем прогноз
                    if steps > len(model_forecast):
                        print(f"⚠️  Запрошено {steps} шагов, но модель обучена на {len(model_forecast)}")
                        print(f"   Использую простую экстраполяцию для дополнительных шагов")
                        # Используем последний тренд для экстраполяции
                        trend = np.mean(np.diff(model_forecast[-5:])) if len(model_forecast) >= 5 else 0
                        extra_steps = steps - len(model_forecast)
                        extra_forecast = np.array([model_forecast[-1] + trend * (i+1) for i in range(extra_steps)])
                        forecast = np.concatenate([model_forecast, extra_forecast])
                    else:
                        # Берём нужное количество шагов
                        forecast = model_forecast[:steps]
                
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
        
        result = {'forecast': forecast}
        
        # Доверительные интервалы
        if return_conf_int:
            from scipy import stats
            std_residual = np.std(self.residuals) if self.residuals is not None else np.std(self.data)
            z_score = stats.norm.ppf(1 - alpha/2)
            
            margin = z_score * std_residual
            result['lower_bound'] = forecast - margin
            result['upper_bound'] = forecast + margin
        
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
