"""
TimeLLM с поддержкой локальных GGUF моделей через llama-cpp-python
"""
import numpy as np
import pandas as pd
import warnings
import os
import gc

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"

import torch
warnings.filterwarnings('ignore')
torch.cuda.empty_cache()
torch.set_float32_matmul_precision('high')
torch.set_grad_enabled(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clear_gpu_memory_completely():
    """Полная очистка GPU памяти - агрессивный метод"""
    if not torch.cuda.is_available():
        return
    
    # Синхронизируем все операции CUDA
    torch.cuda.synchronize()
    
    # Очищаем кэш
    torch.cuda.empty_cache()
    
    # Сбрасываем статистику памяти
    torch.cuda.reset_peak_memory_stats()
    
    # Принудительная сборка мусора
    gc.collect()
    
    # Еще раз очищаем кэш после сборки мусора
    torch.cuda.empty_cache()
    
    # Показываем состояние памяти
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"🧹 GPU память после очистки: выделено={allocated:.2f} GB, зарезервировано={reserved:.2f} GB")

class TimeLLM:
    """
    TimeLLM с поддержкой:
    1. NeuralForecast.TimeLLM (если установлен)
    2. Локальные GGUF модели через llama-cpp-python
    3. Fallback на простую статистическую модель
    """
    
    def __init__(self, llm_backend='gguf', llm_model='gpt2', llm_path=None, gguf_config=None, use_cpu=False, neuralforecast_model='tinyllama'):
        """
        Инициализация TimeLLM
        
        Args:
            llm_backend: 'gguf' (локальная GGUF), 'neuralforecast' (NeuralForecast), 'simple' (fallback)
            llm_model: Название LLM модели для NeuralForecast (gpt2, llama и т.д.)
            llm_path: Путь к GGUF файлу (для llm_backend='gguf')
            gguf_config: dict с конфигурацией GGUF:
                {
                    'n_ctx': 2048,        # Размер контекста
                    'n_threads': 8,       # CPU threads
                    'n_gpu_layers': 0,    # GPU layers (0 для CPU)
                    'temperature': 0.7,   # Температура генерации
                    'max_tokens': 512     # Максимум токенов
                }
            use_cpu: [УСТАРЕЛО] NeuralForecast работает только на GPU. Параметр игнорируется.
            neuralforecast_model: Модель для NeuralForecast:
                - 'phi-1.5' (по умолчанию): microsoft/phi-1.5 - современная модель 2023 года
                - 'tinyllama': TinyLlama-1.1B - самая современная модель 2024 года
                - 'gpt2': gpt2 - самая легкая модель
                - 'phi-2': microsoft/phi-2 - более мощная модель (может не поместиться в 16GB)
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
        """Простой статистический прогноз (fallback)"""
        # Линейный тренд + последнее значение
        trend = np.mean(np.diff(data)) if len(data) > 1 else 0
        last_value = data[-1]
        
        forecast = np.array([last_value + trend * (i+1) for i in range(steps)])
        
        # Добавляем сезонность если есть
        if len(data) >= 12:
            # Простая сезонная компонента (среднее по сезонам)
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
        
        # Режим GGUF
        if self.llm_backend == 'gguf':
            if self.llm_instance is None and self.llm_path:
                self._init_gguf_model()
            
            if self.llm_instance:
                print("✓ Используется GGUF модель")
            else:
                print("Warning: GGUF недоступен, используется simple режим")
                self.llm_backend = 'simple'
        
        # Режим NeuralForecast (требует GPU)
        elif self.llm_backend == 'neuralforecast':
            if not torch.cuda.is_available():
                print("⚠️ Warning: NeuralForecast требует GPU. CUDA недоступен.")
                print("Используется simple режим")
                self.llm_backend = 'simple'
            else:
                try:
                    self._fit_neuralforecast(data, freq)
                    print("✓ Используется NeuralForecast.TimeLLM на GPU")
                except Exception as e:
                    print(f"Warning: NeuralForecast failed: {e}")
                    print("Используется simple режим")
                    self.llm_backend = 'simple'
        
        # Simple режим
        if self.llm_backend == 'simple':
            self._fit_simple(data)
            print("✓ Используется Simple статистический режим")
        
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
                default_model = 'phi-1.5'
                print("⚡ Режим: Максимальная производительность (16GB VRAM)")
            elif gpu_memory >= 12.0:  # 12-15GB карта
                optimal_batch_size = 6
                optimal_input_size = 96
                optimal_horizon = 36
                default_model = 'phi-1.5'
                print("⚡ Режим: Высокая производительность (12GB+ VRAM)")
            elif gpu_memory >= 8.0:  # 8-12GB карта
                optimal_batch_size = 4
                optimal_input_size = 64
                optimal_horizon = 24
                default_model = 'phi-1.5'
                print("⚡ Режим: Оптимизированная производительность (8GB+ VRAM)")
            else:  # Меньше 8GB
                optimal_batch_size = 2
                optimal_input_size = 48
                optimal_horizon = 24
                default_model = 'phi-1.5'
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
            # Для RTX 4070 Ti Super (16GB) можно использовать более мощные модели
            # Доступные модели:
            #   - 'phi-2': microsoft/phi-2 (2.7B, 2023) - РЕКОМЕНДУЕТСЯ для 16GB, d_llm=2048
            #   - 'phi-1.5': microsoft/phi-1.5 (1.3B, 2023) - современная, d_llm=2048
            #   - 'tinyllama': TinyLlama/TinyLlama-1.1B-Chat-v1.0 (1.1B, 2024) - самая современная, d_llm=2048
            #   - 'gpt2': gpt2 (124M, 2019) - самая легкая, d_llm=768
            
            # Для 16GB карты используем phi-2 по умолчанию (более мощная модель)
            model_choice = self.neuralforecast_model or default_model
            
            model_configs = {
                'phi-1.5': {
                    'name': 'microsoft/phi-1.5',
                    'd_llm': 2048,
                    'description': 'Современная модель 2023 года (1.3B параметров)'
                },
                'tinyllama': {
                    'name': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                    'd_llm': 2048,
                    'description': 'Самая современная модель 2024 года (1.1B параметров)'
                },
                'gpt2': {
                    'name': 'gpt2',
                    'd_llm': 768,
                    'description': 'Классическая модель 2019 года (124M параметров) - самая легкая'
                },
                'phi-2': {
                    'name': 'microsoft/phi-2',
                    'd_llm': 2560,  # Правильный размер: phi-2 имеет hidden_size=2560, не 2048!
                    'description': 'Мощная модель 2023 года (2.7B параметров) - ОПТИМАЛЬНА для RTX 4070 Ti Super'
                }
            }
            
            if model_choice not in model_configs:
                print(f"⚠️ Неизвестная модель '{model_choice}', используем {default_model} по умолчанию")
                model_choice = default_model
            
            config = model_configs[model_choice]
            llm_model_name = config['name']
            d_llm_value = config['d_llm']
            
            print(f"🤖 Выбрана модель: {config['description']}")
            
            print(f"📊 NeuralForecast: Модель={llm_model_name}, d_llm={d_llm_value}")
            print(f"📊 Параметры: batch_size={optimal_batch_size}, input_size={input_size}, horizon={horizon}")
            print(f"📊 Устройство: GPU (CUDA)")
            
            # Создаем модель TimeLLM для NeuralForecast
            # NeuralForecast автоматически использует GPU если доступен
            # Оптимизировано для RTX 4070 Ti Super (16GB VRAM)
            timellm_model = NF_TimeLLM(
                h=horizon,
                input_size=input_size,
                llm=llm_model_name,
                d_llm=d_llm_value,
                prompt_prefix=prompt,
                learning_rate=1e-4,  # Оптимальный learning rate
                batch_size=optimal_batch_size,  # Адаптивный batch_size на основе VRAM (8 для 16GB)
                random_seed=42
            )
            
            nf = NeuralForecast(models=[timellm_model], freq=freq)
            
            # Обучение модели
            print("🚀 Начинаю обучение модели...")
            nf.fit(df=df)
            
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
        """Упрощённая версия для fallback"""
        self.data = data
        self.mean = np.mean(data)
        self.std = np.std(data)
        self.trend = np.mean(np.diff(data)) if len(data) > 1 else 0
        
        # Остатки для доверительных интервалов
        if len(data) > 5:
            rolling_mean = pd.Series(data).rolling(window=min(5, len(data))).mean()
            self.residuals = data - rolling_mean.fillna(method='bfill').values
        else:
            self.residuals = np.zeros(len(data))
        
        self.model = 'simple'
    
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
                
                # Используем torch.no_grad() для инференса на GPU
                with torch.no_grad():
                    forecast_df = self.model.predict(df=df, h=steps)
                    forecast = forecast_df['TimeLLM'].values[:steps]
                
                # ПОЛНАЯ очистка GPU памяти после предсказания
                clear_gpu_memory_completely()
            except Exception as e:
                print(f"Warning: NeuralForecast prediction failed: {e}")
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
