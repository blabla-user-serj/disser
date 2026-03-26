"""
Гибридная модель прогнозирования экстремально коротких временных рядов

Реализует математическую модель из Главы 2 диссертации:
- Формула 2.5, 2.11: Динамические веса EWA с β=1.2
- Формула 2.6: Корректирующий коэффициент α(n) = 1 + ln(30/n) при n < 30
- Формула 2.7: Взвешенная ошибка ER_i(t) с λ=0.8
- Формула 2.8, 2.9: Коррекция смещения малой выборки Δ_bias(n,h)
- Формула 2.10: Ансамблевый прогноз
- Формула 2.13, 2.14: Робастная обработка выбросов через MAD
- Формула 2.17: Адаптивный вес LLM w_LLM(n)
- Формула 2.18: Финальная формула прогноза
"""
import numpy as np
import gc
import math
import torch
from scipy import stats
from .sarima_xs import SARIMAXS
from .xgboost_model import XGBoostTS
from .timellm_gguf import TimeLLM, clear_gpu_memory_completely


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


class HybridModel:
    """
    Гибридная модель с адаптивным взвешиванием для экстремально коротких рядов (n=5-20)
    
    Научная новизна:
    1. Адаптивные веса EWA с коррекцией на малую выборку
    2. Коррекция смещения Δ_bias(n,h)
    3. Робастная обработка выбросов через MAD
    4. Адаптивный вес LLM-эксперта w_LLM(n)
    """
    
    # ==================== КОНСТАНТЫ ИЗ ДИССЕРТАЦИИ ====================
    BETA = 1.2          # Формула 2.5: коэффициент чувствительности
    LAMBDA = 0.8        # Формула 2.7: коэффициент затухания для коротких рядов
    GAMMA_BIAS = 0.15   # Формула 2.9: коэффициент компенсации смещения
    T_NORMAL = 100      # Формула 2.9: теоретическая длина нормального ряда (T=100 по диссертации, стр. 677)
    MAD_THRESHOLD = 2.5 # Формула 2.13: порог для обнаружения выбросов
    MAD_SCALE = 0.6745  # Формула 2.14: масштаб MAD для нормального распределения
    
    def __init__(self, decay_factor=None, use_cv=True, n_splits=3, use_slm=True, slm_model='qwen2-0.5b'):
        """
        Args:
            decay_factor: коэффициент экспоненциального сглаживания (по умолчанию λ=0.8 из Формулы 2.7)
            use_cv: использовать ли CV в базовых моделях
            n_splits: количество сплитов для CV
            use_slm: использовать ли SLM модели (NeuralForecast)
            slm_model: какую SLM модель использовать
        """
        # Сохраняем параметры для отложенной инициализации TimeLLM
        self.use_cv = use_cv
        self.n_splits = n_splits
        self.use_slm = use_slm
        self.slm_model = slm_model
        
        self.sarima = SARIMAXS(use_cv=use_cv, n_splits=n_splits)
        self.xgboost = XGBoostTS(use_cv=use_cv, n_splits=n_splits)
        
        # TimeLLM создаётся позже при обучении
        self.timellm = None
        
        # Формула 2.7: λ = 0.8 для коротких рядов
        self.decay_factor = decay_factor if decay_factor is not None else self.LAMBDA
        
        # Инициализация весов с приоритетом SARIMA-XS для коротких рядов
        # SARIMA-XS показывает лучшую устойчивость на экстремально коротких рядах (n=5-20)
        self.weights = {
            'sarima': 0.6,    # Больший начальный вес для SARIMA-XS
            'xgboost': 0.3,
            'timellm': 0.1
        }
        
        self.error_history = {
            'sarima': 0,
            'xgboost': 0,
            'timellm': 0
        }
        
        self.data = None
        self.data_robust = None  # Данные после робастной обработки
        self.std_estimate = None  # Оценка σ для Δ_bias
    
    # ==================== РОБАСТНАЯ ОБРАБОТКА ВЫБРОСОВ ====================
    # Формулы 2.13, 2.14
    
    def _robust_preprocess(self, data):
        """
        Робастная обработка выбросов на основе MAD (Формулы 2.13, 2.14)
        
        X_robust(t) = X(t) если |X(t) - μ̂| ≤ 2.5σ̂, иначе обрезка
        μ̂ = median(X)
        σ̂ = MAD / 0.6745
        MAD = median(|X - μ̂|)
        """
        # Формула 2.14: Робастные оценки
        mu_robust = np.median(data)
        mad = np.median(np.abs(data - mu_robust))
        sigma_robust = mad / self.MAD_SCALE if mad > 0 else np.std(data)
        
        # Сохраняем для Δ_bias
        self.std_estimate = sigma_robust
        
        # Формула 2.13: Обработка выбросов
        threshold = self.MAD_THRESHOLD * sigma_robust
        
        data_robust = np.copy(data)
        outlier_count = 0
        
        for i in range(len(data)):
            deviation = np.abs(data[i] - mu_robust)
            if deviation > threshold:
                # Winsorization: ограничиваем значение
                sign = np.sign(data[i] - mu_robust)
                data_robust[i] = mu_robust + sign * threshold
                outlier_count += 1
        
        if outlier_count > 0:
            print(f"🔧 Робастная обработка: обнаружено {outlier_count} выбросов из {len(data)} точек")
            print(f"   μ̂ = {mu_robust:.4f}, σ̂ = {sigma_robust:.4f}")
        
        return data_robust
    
    # ==================== КОЭФФИЦИЕНТЫ АДАПТАЦИИ ====================
    # Формулы 2.6, 2.12
    
    def _alpha_correction(self, n):
        """
        Формула 2.6: Корректирующий коэффициент α(n)

        α(n) = 1 + ln(30 / n)  для n < 30
        α(n) = 1.0             для n >= 30

        Примечание: формула использует фиксированное значение 30 (порог КВР),
        не T_NORMAL из Δ_bias. Значения:
          При n=5:  α ≈ 2.79
          При n=10: α ≈ 2.10
          При n=15: α ≈ 1.69
          При n=20: α ≈ 1.41
          При n=30: α = 1.0
        """
        # Формула 2.6: порог 30 (КВР), не T_NORMAL=100 (который используется только в Δ_bias)
        ALPHA_THRESHOLD = 30
        if n < ALPHA_THRESHOLD:
            return 1 + np.log(ALPHA_THRESHOLD / n)
        return 1.0
    
    def _kappa_correction(self, n):
        """
        Формула 2.12: Коэффициент κ(n) для весов ансамбля
        
        κ(n) = 1 + 0.5 × (20-n)/15
        
        При n=5: κ = 1.5 (максимальная коррекция)
        При n=20: κ = 1.0 (без коррекции)
        """
        if 5 <= n <= 20:
            return 1 + 0.5 * (20 - n) / 15
        elif n < 5:
            return 1.5  # Максимальная коррекция
        return 1.0
    
    # ==================== АДАПТИВНЫЙ ВЕС LLM ====================
    # Формулы 2.16, 2.17
    
    def _confidence_score(self, n):
        """
        Формула 2.16: Оценка достоверности
        
        Confidence_Score = 1 - (n/30) × 0.8   # диссертация, стр. 700
        
        Чем короче ряд, тем выше неопределенность.
        При n=30: CS = 1 - 0.8 = 0.2
        При n=5:  CS = 1 - (5/30)*0.8 = 0.867
        """
        return max(1 - (n / 30) * 0.8, 0.0)
    
    def _w_llm(self, n):
        """
        Формула 2.17: Адаптивный вес LLM-коррекции
        
        w_LLM(n) = 0.15 + 0.35 × (n-5)/15  для 5 ≤ n ≤ 30   # диссертация, стр. 703
        w_LLM(n) = 0.5                      для n > 30
        w_LLM(n) = 0.1                      для n < 5
        
        При n=5:  w=0.15 (минимальное доверие)
        При n=20: w=0.50
        При n=30: w=0.15+0.35*25/15≈0.733 → ограничено 0.5 (cap)
        """
        if 5 <= n <= 30:
            return min(0.15 + 0.35 * (n - 5) / 15, 0.5)
        elif n > 30:
            return 0.5
        else:
            return 0.1
    
    # ==================== КОРРЕКЦИЯ СМЕЩЕНИЯ ====================
    # Формулы 2.8, 2.9
    
    def _delta_bias(self, n, h):
        """
        Формулы 2.8, 2.9: Коррекция смещения малой выборки
        
        Δ_bias(n, h) = γ × h × σ̂ × √(1/n + 1/T)

        Где:
        - γ = 0.15 (коэффициент компенсации, Гл. 2)
        - h = горизонт прогноза
        - σ̂ = робастная оценка стандартного отклонения
        - n = длина ряда
        - T = 30 (теоретическая длина нормального ряда, Гл. 2)
        """
        if self.std_estimate is None:
            sigma = np.std(self.data) if self.data is not None else 1.0
        else:
            sigma = self.std_estimate
        
        # Формула 2.9: T=100 (теоретическая длина нормального ряда по диссертации)
        uncertainty_factor = np.sqrt(1/n + 1/self.T_NORMAL)  # T_NORMAL=100
        delta = self.GAMMA_BIAS * h * sigma * uncertainty_factor
        
        return delta
    
    # ==================== ДИНАМИЧЕСКИЕ ВЕСА EWA ====================
    # Формулы 2.5, 2.7, 2.11
    
    def _update_weights(self, errors):
        """
        Обновление весов моделей по формулам 2.5, 2.7, 2.11
        
        Формула 2.7: ER_i(t) = λ × ER_i(t-1) + (1-λ) × NormMAE_i
        Формула 2.5/2.11: w_i(t) = exp(-β × ER_i(t) × α(n)) / Σ exp(...)
        
        ВАЖНО: Используются НОРМАЛИЗОВАННЫЕ ошибки (деление на масштаб данных),
        чтобы exp(-β × ER) не обращался в ноль при больших абсолютных MAE.
        
        NormMAE = MAE / scale, где scale = max(std, |mean|*0.1, 1.0)
        
        ХОЛОДНЫЙ СТАРТ: при первом вызове (error_history = 0) обычный шаг EWA
        даёт ER_i = (1-λ)·NormMAE = 0.2·NormMAE — разброс ошибок сжимается в 5 раз,
        и модели с кратно разными MAE получают почти одинаковые веса.
        Исправление: при холодном старте ER инициализируется напрямую значением NormMAE
        (эквивалентно сходившемуся EWA с бесконечной историей одинаковых ошибок).
        
        С параметрами:
        - β = 1.2 (чувствительность)
        - λ = 0.8 (затухание)
        - α(n) = 1 + ln(30/n) (коррекция на малую выборку)
        """
        n = len(self.data)
        alpha = self._alpha_correction(n)
        
        # Вычисляем масштаб данных для нормализации ошибок
        data_std = np.std(self.data)
        data_mean_abs = np.abs(np.mean(self.data))
        scale = max(data_std, data_mean_abs * 0.1, 1.0)  # Защита от деления на ноль
        
        print(f"📐 Коэффициенты: α(n={n}) = {alpha:.4f}")
        print(f"📏 Масштаб для нормализации ошибок: {scale:.4f}")
        
        # Нормализуем ошибки
        normalized_errors = {}
        for model in ['sarima', 'xgboost', 'timellm']:
            raw_error = errors.get(model, 0)
            if raw_error == float('inf'):
                normalized_errors[model] = 10.0  # Штраф за неработающую модель
            else:
                normalized_errors[model] = raw_error / scale
            print(f"   {model}: MAE={raw_error:.4f} → NormMAE={normalized_errors[model]:.4f}")
        
        # Формула 2.7: Обновление истории ошибок с экспоненциальным сглаживанием.
        # Холодный старт: если история нулевая, инициализируем напрямую (без λ-демпфирования),
        # чтобы разброс ошибок полностью отразился в весах.
        cold_start = all(v == 0.0 for v in self.error_history.values())
        for model in ['sarima', 'xgboost', 'timellm']:
            if cold_start:
                self.error_history[model] = normalized_errors[model]
            else:
                self.error_history[model] = (
                    self.decay_factor * self.error_history[model] +
                    (1 - self.decay_factor) * normalized_errors[model]
                )
        
        # Формула 2.5/2.11: Вычисление весов
        # w_i = exp(-β × ER_i × α(n)) / Σ  — по диссертации формула 2.5
        # κ(n) в диссертации не входит в формулу весов, поэтому исключён
        exp_weights = {}
        for model in ['sarima', 'xgboost', 'timellm']:
            exponent = -self.BETA * self.error_history[model] * alpha
            # Ограничиваем экспоненту для численной стабильности
            exponent = max(exponent, -50)  # exp(-50) ≈ 1.9e-22, достаточно малое
            exp_weights[model] = np.exp(exponent)
            print(f"   {model}: exp({exponent:.4f}) = {exp_weights[model]:.6f}")
        
        # Нормализация
        total = sum(exp_weights.values())
        
        if total > 0:
            for model in ['sarima', 'xgboost', 'timellm']:
                self.weights[model] = exp_weights[model] / total
        else:
            # Fallback: веса с приоритетом SARIMA-XS
            print("⚠️ Сумма весов = 0, используем веса по умолчанию (приоритет SARIMA-XS)")
            self.weights = {
                'sarima': 0.5,
                'xgboost': 0.3,
                'timellm': 0.2
            }
        
        # Проверка: минимальный вес не должен быть меньше 0.05
        MIN_WEIGHT = 0.05
        needs_redistribution = any(w < MIN_WEIGHT and w > 0 for w in self.weights.values())
        
        if needs_redistribution:
            print(f"🔄 Применяем минимальный порог веса {MIN_WEIGHT}")
            # Устанавливаем минимальные веса и перенормируем
            for model in self.weights:
                if self.weights[model] < MIN_WEIGHT:
                    self.weights[model] = MIN_WEIGHT
            
            # Перенормируем
            total = sum(self.weights.values())
            for model in self.weights:
                self.weights[model] /= total
    
    # ==================== СОЗДАНИЕ/УДАЛЕНИЕ TimeLLM ====================
    
    def _create_timellm(self):
        """Создание TimeLLM модели с предварительной очисткой GPU памяти"""
        if torch.cuda.is_available():
            print("🧹 Очистка GPU памяти перед созданием TimeLLM...")
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            gc.collect()
            torch.cuda.empty_cache()
            
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"📊 GPU память: выделено={allocated:.2f} GB, зарезервировано={reserved:.2f} GB")
        
        if self.use_slm:
            print(f"🤖 Создание TimeLLM с NeuralForecast + SLM '{self.slm_model}'")
            return TimeLLM(
                llm_backend="neuralforecast", 
                neuralforecast_model=self.slm_model
            )
        else:
            print("⚡ Создание TimeLLM в Simple режиме (без GPU)")
            return TimeLLM(llm_backend="simple")
    
    def _destroy_timellm(self):
        """Полное удаление TimeLLM и освобождение GPU памяти"""
        if self.timellm is not None:
            print("🗑️ Удаление TimeLLM модели...")
            
            if hasattr(self.timellm, 'model') and self.timellm.model is not None:
                del self.timellm.model
                self.timellm.model = None
            
            del self.timellm
            self.timellm = None
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
            
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"🧹 GPU память после удаления: выделено={allocated:.2f} GB, зарезервировано={reserved:.2f} GB")
    
    # ==================== ОБУЧЕНИЕ ====================
    
    def fit(self, data, steps=None):
        """
        Обучение гибридной модели с робастной предобработкой
        
        Этапы:
        1. Робастная обработка выбросов (Формулы 2.13, 2.14)
        2. Обучение базовых моделей (SARIMA, XGBoost, TimeLLM)
        3. Валидация и расчёт ошибок
        4. Обновление весов EWA (Формулы 2.5, 2.7, 2.11)
        """
        self._fit_steps = steps
        self.data = data
        n = len(data)
        
        print("\n" + "="*60)
        print(f"🚀 ГИБРИДНАЯ МОДЕЛЬ: Обучение на {n} точках")
        print(f"   Диапазон экстремально коротких рядов: 5 ≤ n ≤ 20")
        print("="*60)
        
        # ==================== Шаг 1: Робастная предобработка ====================
        print("\n📊 Шаг 1: Робастная обработка выбросов (MAD)...")
        self.data_robust = self._robust_preprocess(data)
        
        # Определяем нужна ли валидация для весов
        # НС-2: граничное условие n >= 10 (диссертация требует обновления весов при n=10)
        need_validation = n >= 10
        
        if need_validation:
            split = int(0.8 * n)
            train_data = self.data_robust[:split]
            val_data = self.data_robust[split:]
            val_steps = len(val_data)
        
        errors = {}
        
        # ==================== Шаг 2: SARIMA ====================
        print("\n" + "="*50)
        print("📊 Шаг 2a: Обучение SARIMA-XS...")
        print("="*50)
        
        try:
            if need_validation:
                temp_sarima = SARIMAXS(use_cv=False)
                temp_sarima.fit(train_data)
                pred = temp_sarima.predict(val_steps, return_conf_int=False)['forecast']
                errors['sarima'] = np.mean(np.abs(val_data - pred))
                print(f"✅ SARIMA валидация: MAE = {errors['sarima']:.4f}")
                del temp_sarima
            
            self.sarima.fit(self.data_robust)
            print("✅ SARIMA обучена на полных данных")
            
        except Exception as e:
            print(f"⚠️ SARIMA ошибка: {e}")
            errors['sarima'] = float('inf')
        
        # ==================== Шаг 3: XGBoost ====================
        print("\n" + "="*50)
        print("📊 Шаг 2b: Обучение XGBoost...")
        print("="*50)
        
        try:
            if need_validation:
                temp_xgb = XGBoostTS(use_cv=False)
                temp_xgb.fit(train_data)
                pred = temp_xgb.predict(val_steps, return_conf_int=False)['forecast']
                errors['xgboost'] = np.mean(np.abs(val_data - pred))
                print(f"✅ XGBoost валидация: MAE = {errors['xgboost']:.4f}")
                del temp_xgb
            
            self.xgboost.fit(self.data_robust)
            print("✅ XGBoost обучена на полных данных")
            
        except Exception as e:
            print(f"⚠️ XGBoost ошибка: {e}")
            errors['xgboost'] = float('inf')
        
        # ==================== Шаг 4: TimeLLM ====================
        print("\n" + "="*50)
        print("📊 Шаг 2c: Обучение TimeLLM...")
        print("="*50)
        
        try:
            if need_validation and self.use_slm:
                print("🔄 Этап 1: Валидация TimeLLM на train_data...")
                self._destroy_timellm()
                
                temp_timellm = self._create_timellm()
                temp_timellm.fit(train_data)
                pred = temp_timellm.predict(val_steps, return_conf_int=False)['forecast']
                
                if len(pred) >= val_steps:
                    errors['timellm'] = np.mean(np.abs(val_data - pred[:val_steps]))
                else:
                    errors['timellm'] = np.mean(np.abs(val_data[:len(pred)] - pred))
                
                print(f"✅ TimeLLM валидация: MAE = {errors['timellm']:.4f}")
                
                del temp_timellm
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    allocated = torch.cuda.memory_allocated(0) / 1024**3
                    print(f"🧹 GPU память после валидации: {allocated:.2f} GB")
                
                print("🔄 Этап 2: Обучение TimeLLM на полных данных...")
            
            self.timellm = self._create_timellm()
            self.timellm.fit(self.data_robust, steps=self._fit_steps)
            print("✅ TimeLLM обучена на полных данных")
            
        except Exception as e:
            print(f"⚠️ TimeLLM ошибка: {e}")
            errors['timellm'] = float('inf')
            self.timellm = TimeLLM(llm_backend="simple")
            self.timellm.fit(self.data_robust, steps=self._fit_steps)
        
        # ==================== Шаг 5: Обновление весов ====================
        if need_validation:
            print("\n" + "="*50)
            print("📊 Шаг 3: Обновление весов ансамбля (EWA)...")
            print("="*50)
            print(f"   SARIMA MAE:  {errors.get('sarima', 'N/A')}")
            print(f"   XGBoost MAE: {errors.get('xgboost', 'N/A')}")
            print(f"   TimeLLM MAE: {errors.get('timellm', 'N/A')}")
            
            self._update_weights(errors)
            
            print(f"\n📊 Итоговые веса (Формула 2.11):")
            print(f"   SARIMA:  {self.weights['sarima']:.4f}")
            print(f"   XGBoost: {self.weights['xgboost']:.4f}")
            print(f"   TimeLLM: {self.weights['timellm']:.4f}")
        
        # Информация об адаптивных коэффициентах
        print(f"\n📐 Адаптивные коэффициенты для n={n}:")
        print(f"   α(n) = {self._alpha_correction(n):.4f} (Формула 2.6)")
        print(f"   κ(n) = {self._kappa_correction(n):.4f} (Формула 2.12)")
        print(f"   w_LLM(n) = {self._w_llm(n):.4f} (Формула 2.17)")
        print(f"   Confidence = {self._confidence_score(n):.4f} (Формула 2.16)")
        
        print("\n" + "="*60)
        print("✅ Гибридная модель обучена")
        print("="*60 + "\n")
        
        return self
    
    # ==================== ПРОГНОЗИРОВАНИЕ ====================
    
    def predict(self, steps, return_conf_int=True, alpha=0.05, llm_correction=None):
        """
        Прогнозирование по Формуле 2.18:
        
        Ŷ_final(t+h) = Ŷ_ensemble(t+h) + Δ_bias(n,h) + w_LLM(n) × Δ_LLM(t+h)
        
        Args:
            steps: количество шагов прогноза (h)
            return_conf_int: возвращать ли доверительные интервалы
            alpha: уровень значимости (0.05 = 95% интервал)
            llm_correction: внешняя коррекция от LLM-эксперта (Δ_LLM)
            
        Returns:
            dict с прогнозом и метаданными
        """
        n = len(self.data)
        
        forecasts = {}
        lower_bounds = {}
        upper_bounds = {}
        
        # ==================== Прогнозы базовых моделей ====================
        
        # SARIMA
        try:
            result = self.sarima.predict(steps, return_conf_int=return_conf_int, alpha=alpha)
            forecasts['sarima'] = np.array(result['forecast'])
            if return_conf_int:
                lower_bounds['sarima'] = np.array(result['lower_bound'])
                upper_bounds['sarima'] = np.array(result['upper_bound'])
        except Exception as e:
            print(f"Warning: SARIMA prediction failed: {e}")
            forecasts['sarima'] = np.zeros(steps)
            if return_conf_int:
                lower_bounds['sarima'] = np.zeros(steps)
                upper_bounds['sarima'] = np.zeros(steps)
        
        # XGBoost
        try:
            result = self.xgboost.predict(steps, return_conf_int=return_conf_int, alpha=alpha)
            forecasts['xgboost'] = np.array(result['forecast'])
            if return_conf_int:
                lower_bounds['xgboost'] = np.array(result['lower_bound'])
                upper_bounds['xgboost'] = np.array(result['upper_bound'])
        except Exception as e:
            print(f"Warning: XGBoost prediction failed: {e}")
            forecasts['xgboost'] = np.zeros(steps)
            if return_conf_int:
                lower_bounds['xgboost'] = np.zeros(steps)
                upper_bounds['xgboost'] = np.zeros(steps)
        
        # TimeLLM
        try:
            result = self.timellm.predict(steps, return_conf_int=return_conf_int, alpha=alpha)
            forecasts['timellm'] = np.array(result['forecast'])
            if return_conf_int:
                lower_bounds['timellm'] = np.array(result['lower_bound'])
                upper_bounds['timellm'] = np.array(result['upper_bound'])
        except Exception as e:
            print(f"Warning: TimeLLM prediction failed: {e}")
            forecasts['timellm'] = np.zeros(steps)
            if return_conf_int:
                lower_bounds['timellm'] = np.zeros(steps)
                upper_bounds['timellm'] = np.zeros(steps)
        
        # ==================== Формула 2.10: Ансамблевый прогноз ====================
        ensemble_forecast = (
            self.weights['sarima'] * forecasts['sarima'] +
            self.weights['xgboost'] * forecasts['xgboost'] +
            self.weights['timellm'] * forecasts['timellm']
        )
        
        # ==================== Формула 2.8, 2.9: Коррекция смещения ====================
        # Δ_bias применяется к каждому шагу прогноза
        bias_corrections = np.array([self._delta_bias(n, h+1) for h in range(steps)])
        
        # ==================== Формула 2.17: Адаптивный вес LLM ====================
        w_llm = self._w_llm(n)
        
        # ==================== Формула 2.18: Финальный прогноз ====================
        # Ŷ_final = Ŷ_ensemble + Δ_bias + w_LLM × Δ_LLM
        
        if llm_correction is not None:
            # Внешняя LLM коррекция передана
            delta_llm = np.array(llm_correction)
            if len(delta_llm) != steps:
                delta_llm = np.resize(delta_llm, steps)
        else:
            # Без LLM коррекции
            delta_llm = np.zeros(steps)
        
        final_forecast = ensemble_forecast + bias_corrections + w_llm * delta_llm
        
        print(f"\n📊 Прогноз (Формула 2.18):")
        print(f"   Ŷ_ensemble: {ensemble_forecast[:3]}...")
        print(f"   Δ_bias: {bias_corrections[:3]}...")
        print(f"   w_LLM × Δ_LLM: {(w_llm * delta_llm)[:3]}...")
        print(f"   Ŷ_final: {final_forecast[:3]}...")
        
        # Fallback значение на основе последних данных
        last_value = float(self.data[-1]) if self.data is not None and len(self.data) > 0 else 0.0
        
        # Очистка финального прогноза от NaN/Inf
        final_forecast = sanitize_array(final_forecast, fallback_value=last_value)
        ensemble_forecast = sanitize_array(ensemble_forecast, fallback_value=last_value)
        bias_corrections = sanitize_array(bias_corrections, fallback_value=0.0)
        
        result = {
            'forecast': final_forecast,
            'ensemble_forecast': ensemble_forecast,
            'bias_correction': bias_corrections,
            'llm_weight': w_llm,
            'llm_correction': sanitize_array(w_llm * delta_llm, fallback_value=0.0),
            'weights': self.weights.copy(),
            'individual_forecasts': forecasts
        }
        
        if return_conf_int:
            # Взвешенная комбинация доверительных интервалов
            hybrid_lower = (
                self.weights['sarima'] * lower_bounds['sarima'] +
                self.weights['xgboost'] * lower_bounds['xgboost'] +
                self.weights['timellm'] * lower_bounds['timellm']
            )
            
            hybrid_upper = (
                self.weights['sarima'] * upper_bounds['sarima'] +
                self.weights['xgboost'] * upper_bounds['xgboost'] +
                self.weights['timellm'] * upper_bounds['timellm']
            )
            
            # Добавляем Δ_bias к доверительным интервалам
            lower = hybrid_lower + bias_corrections
            upper = hybrid_upper + bias_corrections
            
            # Очистка доверительных интервалов от NaN/Inf
            result['lower_bound'] = sanitize_array(lower, fallback_value=final_forecast[0] * 0.9)
            result['upper_bound'] = sanitize_array(upper, fallback_value=final_forecast[0] * 1.1)
            
            # Гарантируем, что lower <= forecast <= upper
            result['lower_bound'] = np.minimum(result['lower_bound'], final_forecast)
            result['upper_bound'] = np.maximum(result['upper_bound'], final_forecast)
        
        return result
    
    # ==================== МЕТРИКИ ====================
    
    def get_metrics(self, y_true, y_pred):
        """Расчёт метрик качества, включая NMAE (Формула 2.21)"""
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        # R²
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Формула 2.21: Нормализованная MAE
        sigma_real = np.std(y_true)
        nmae = mae / sigma_real if sigma_real > 0 else mae
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2,
            'NMAE': nmae  # Формула 2.21
        }
    
    def get_info(self):
        """Информация о модели с параметрами из диссертации"""
        n = len(self.data) if self.data is not None else 0
        
        info = {
            'weights': self.weights,
            'error_history': self.error_history,
            'parameters': {
                'beta': self.BETA,
                'lambda': self.decay_factor,
                'gamma_bias': self.GAMMA_BIAS,
                'T_normal': self.T_NORMAL,
                'MAD_threshold': self.MAD_THRESHOLD
            },
            'adaptive_coefficients': {
                'alpha_n': self._alpha_correction(n) if n > 0 else None,
                'kappa_n': self._kappa_correction(n) if n > 0 else None,
                'w_llm_n': self._w_llm(n) if n > 0 else None,
                'confidence_score': self._confidence_score(n) if n > 0 else None
            }
        }
        
        # Информация о базовых моделях
        try:
            info['sarima_info'] = self.sarima.get_info()
        except:
            info['sarima_info'] = {'status': 'Ошибка'}
        
        try:
            info['xgboost_info'] = self.xgboost.get_info()
        except:
            info['xgboost_info'] = {'status': 'Ошибка'}
        
        try:
            info['timellm_info'] = self.timellm.get_info() if self.timellm else {'status': 'Не инициализирована'}
        except:
            info['timellm_info'] = {'status': 'Ошибка'}
        
        return info
