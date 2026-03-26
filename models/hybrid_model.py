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

ИСПРАВЛЕНИЯ v2 (2026-03-26):
1. Walk-forward LOO валидация вместо разового 80/20 hold-out:
   нестабильность весов при n=10 и 2 валидационных точках устранена.
2. Знак Δ_bias адаптивный: коррекция соответствует направлению тренда.
3. Адаптивная схема "best-wins": если лучшая одиночная модель превосходит
   ансамбль более чем на порог DOMINANCE_THRESH — используется она одна.
4. EWA-весовое обновление происходит по средним LOO-ошибкам (агрегация
   по всем fold'ам), а не по единственной точке.
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
    2. Коррекция смещения Δ_bias(n,h) с адаптивным знаком
    3. Робастная обработка выбросов через MAD
    4. Адаптивный вес LLM-эксперта w_LLM(n)
    5. Walk-forward LOO валидация для надёжной оценки весов
    6. Best-wins защита: ансамбль используется только если он лучше
    """
    
    # ==================== КОНСТАНТЫ ИЗ ДИССЕРТАЦИИ ====================
    BETA = 1.2          # Формула 2.5: коэффициент чувствительности
    LAMBDA = 0.8        # Формула 2.7: коэффициент затухания для коротких рядов
    GAMMA_BIAS = 0.15   # Формула 2.9: коэффициент компенсации смещения
    T_NORMAL = 100      # Формула 2.9: теоретическая длина нормального ряда (T=100 по диссертации, стр. 677)
    MAD_THRESHOLD = 2.5 # Формула 2.13: порог для обнаружения выбросов
    MAD_SCALE = 0.6745  # Формула 2.14: масштаб MAD для нормального распределения
    
    # Порог доминирования: если лучшая одиночная модель обеспечивает MAE
    # на DOMINANCE_THRESH (25%) лучше, чем взвешенный ансамбль — используем её.
    # v3: повышен с 0.10 до 0.25 (БАГ-2: слишком раннее срабатывание).
    DOMINANCE_THRESH = 0.25
    
    # Порог отсечения плохих моделей (БАГ-4):
    # Модель с RelMAE > TRIM_RATIO * min_MAE исключается из ансамбля (вес=0).
    # При TRIM_RATIO=3.0: если лучшая MAE=0.1, то модели с MAE>0.3 отсекаются.
    TRIM_RATIO = 3.0
    
    # Минимальный вес (снижен с 0.05 до 0.01 — только числовая стабильность).
    # v3: 0.01 вместо 0.05 (БАГ-3: принудительное включение плохих моделей).
    MIN_WEIGHT = 0.01
    
    # Порог "близких ошибок" — если max/min LOO ошибок < SIMILAR_THRESH,
    # модели считаются равноценными и используются равные веса (снижает риск
    # случайного перевзвешивания при шумных LOO-оценках при n=10).
    SIMILAR_ERR_THRESH = 1.3  # < 30% разброса → равный ансамбль
    
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
        
        # Признак доминирования одиночной модели
        self._dominant_model = None   # None → используем ансамбль
        self._loo_errors = {}         # LOO-ошибки по каждой модели
        
        self.data = None
        self.data_robust = None  # Данные после робастной обработки
        self.std_estimate = None  # Оценка σ для Δ_bias
        self._trend_sign = 0.0    # Знак тренда для адаптивной коррекции Δ_bias
    
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
        outlier_indices = []
        
        for i in range(len(data)):
            deviation = np.abs(data[i] - mu_robust)
            if deviation > threshold:
                outlier_indices.append(i)
                outlier_count += 1
        
        # ИСПРАВЛЕНИЕ v2: защита от чрезмерного клиппинга.
        # Если "выбросов" больше 20% данных, значит это не выбросы, а
        # нормальный диапазон волатильного ряда. В этом случае MAD-порог
        # повышаем до 4σ, чтобы обрезать только истинные выбросы.
        max_outlier_frac = 0.20
        if len(data) > 0 and outlier_count / len(data) > max_outlier_frac:
            threshold_4s = 4.0 * sigma_robust
            outlier_indices_strict = [i for i in outlier_indices
                                       if abs(data[i] - mu_robust) > threshold_4s]
            print(f"ℹ️  MAD: {outlier_count} потенц. выбросов ({outlier_count/len(data)*100:.0f}% > {max_outlier_frac*100:.0f}%). "
                  f"Повышаем порог до 4σ: {len(outlier_indices_strict)} реальных выбросов.")
            outlier_indices = outlier_indices_strict
        
        for i in outlier_indices:
            sign = np.sign(data[i] - mu_robust)
            effective_threshold = max(abs(data[i] - mu_robust) * 0.9, threshold)
            # Используем 4σ-порог для окончательного клиппинга при широком диапазоне
            clip_thresh = 4.0 * sigma_robust if len(outlier_indices) < outlier_count else threshold
            data_robust[i] = mu_robust + sign * clip_thresh
        
        outlier_count = len(outlier_indices)
        if outlier_count > 0:
            print(f"🔧 Робастная обработка: обрезано {outlier_count} выбросов из {len(data)} точек")
            print(f"   μ̂ = {mu_robust:.4f}, σ̂ = {sigma_robust:.4f}")
        
        # Оцениваем знак тренда для адаптивной коррекции смещения
        if len(data_robust) >= 3:
            slope = np.polyfit(np.arange(len(data_robust)), data_robust, 1)[0]
            # Нормализуем знак: 0 если |тренд| мал относительно масштаба
            if sigma_robust > 0 and abs(slope) > 0.05 * sigma_robust:
                self._trend_sign = np.sign(slope)
            else:
                self._trend_sign = 0.0
        else:
            self._trend_sign = 0.0
        
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
        
        Δ_bias(n, h) = γ × h × σ̂ × √(1/n + 1/T) × sign(trend)

        Где:
        - γ = 0.15 (коэффициент компенсации, Гл. 2)
        - h = горизонт прогноза
        - σ̂ = робастная оценка стандартного отклонения
        - n = длина ряда
        - T = 100 (теоретическая длина нормального ряда, Гл. 2)
        - sign(trend) — знак тренда ряда (±1); 0 если тренд незначим.

        ИСПРАВЛЕНИЕ v2: знак коррекции адаптивный (по направлению тренда),
        а не всегда положительный. Это предотвращает систематическое
        смещение прогноза в неверную сторону.
        """
        if self.std_estimate is None:
            sigma = np.std(self.data) if self.data is not None else 1.0
        else:
            sigma = self.std_estimate
        
        # Формула 2.9: T=100 (теоретическая длина нормального ряда по диссертации)
        uncertainty_factor = np.sqrt(1/n + 1/self.T_NORMAL)  # T_NORMAL=100
        magnitude = self.GAMMA_BIAS * h * sigma * uncertainty_factor
        
        # Адаптивный знак: коррекция направлена по тренду
        delta = magnitude * self._trend_sign
        
        return delta
    
    # ==================== WALK-FORWARD LOO ВАЛИДАЦИЯ ====================
    
    def _walkforward_errors(self, data, model_factory_fn, forecast_horizon=1):
        """
        Walk-forward оценка ошибки модели на горизонте forecast_horizon шагов.

        Алгоритм:
          Для каждого t от min_train до n-forecast_horizon:
            1. Обучить model_factory_fn() на data[:t]
            2. Предсказать forecast_horizon шагов вперёд
            3. Усреднить |pred_h - data[t+h-1]| по h=1..horizon

        Возвращает среднюю абсолютную ошибку по всем фолдам.

        ИСПРАВЛЕНИЕ v2: заменяет единственный 80/20 split.
        ИСПРАВЛЕНИЕ v2b: использует реальный горизонт прогноза (forecast_horizon),
        а не только 1-step-ahead. Это устраняет ситуацию, когда модель хорошо
        предсказывает 1 шаг вперёд, но плохо — реальный горизонт теста.
        """
        n = len(data)
        min_train = max(5, n // 2)   # минимальный размер train для надёжности
        
        fold_errors = []
        for t in range(min_train, n - forecast_horizon + 1):
            train_fold = data[:t]
            # Реальные значения для горизонта [t, t+horizon)
            true_vals = data[t:t + forecast_horizon]
            try:
                m = model_factory_fn()
                m.fit(train_fold)
                pred_result = m.predict(forecast_horizon, return_conf_int=False)
                preds = np.array(pred_result['forecast']).flat[:forecast_horizon]
                if len(preds) < len(true_vals):
                    true_vals = true_vals[:len(preds)]
                fold_mae = float(np.mean(np.abs(true_vals - preds)))
                fold_errors.append(fold_mae)
            except Exception as e:
                # Неудачный fold: штрафная ошибка = σ данных
                fold_errors.append(float(np.std(data)) if np.std(data) > 0 else float(np.mean(np.abs(true_vals))))
        
        if not fold_errors:
            return float('inf')
        return float(np.mean(fold_errors))
    
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
        
        # БАГ-1 FIX: нормализуем MAE на σ данных, а не на min_MAE.
        # Прежний подход (MAE_i/min_MAE) давал RelMAE в [1, cap≈1.19] при n=10,
        # все веса exp() были почти одинаковы. Теперь: NormMAE_i = MAE_i / σ_data.
        raw_errors = {}
        for model in ['sarima', 'xgboost', 'timellm']:
            raw_errors[model] = errors.get(model, 0)

        # σ_data = std(данных). Защита от нулевого σ.
        if self.data is not None and len(self.data) > 1:
            sigma_data = float(np.std(self.data))
            mean_abs = float(np.abs(np.mean(self.data)))
            scale = max(sigma_data, mean_abs * 0.1, 1.0)
        else:
            scale = 1.0

        finite_errors = [v for v in raw_errors.values() if v != float('inf') and v > 0]
        if not finite_errors:
            finite_errors = [scale]
        min_err = min(finite_errors)

        # БАГ-4 FIX: Trimming — исключаем модели хуже TRIM_RATIO × min_MAE
        trim_threshold = self.TRIM_RATIO * min_err
        trimmed_models = set()
        for model in ['sarima', 'xgboost', 'timellm']:
            raw_mae = raw_errors[model]
            if raw_mae == float('inf') or raw_mae > trim_threshold:
                trimmed_models.add(model)
        if trimmed_models:
            print(f"✂️  Trimming: исключаем {trimmed_models} (MAE > {trim_threshold:.4f} = {self.TRIM_RATIO}×min)")

        # --- БАГ-6 (новый): Равный ансамбль при близких ошибках ---
        # При n=10 LOO-оценки шумные. Если все активные модели в пределах
        # SIMILAR_ERR_THRESH от лучшей, перевзвешивание ненадёжно.
        non_trimmed = [m for m in ['sarima', 'xgboost', 'timellm']
                       if m not in trimmed_models
                       and raw_errors.get(m, float('inf')) != float('inf')
                       and raw_errors.get(m, 0) > 0]
        if len(non_trimmed) >= 2:
            errs_nt = [raw_errors[m] for m in non_trimmed]
            spread = max(errs_nt) / min(errs_nt)
            if spread < self.SIMILAR_ERR_THRESH:
                print(f"🤝 Ошибки близкие (max/min={spread:.3f} < {self.SIMILAR_ERR_THRESH}): "
                      f"равный ансамбль для {non_trimmed}")
                # Равные веса для активных (не-trimmed) моделей
                eq_weight = 1.0 / len(non_trimmed)
                for model in ['sarima', 'xgboost', 'timellm']:
                    self.weights[model] = eq_weight if model in non_trimmed else 0.0
                total_w = sum(self.weights.values())
                if total_w > 0:
                    for m in self.weights:
                        self.weights[m] /= total_w
                return  # Равный ансамбль готов, EWA не применяется

        # Нормализуем на σ данных
        normalized_errors = {}
        for model in ['sarima', 'xgboost', 'timellm']:
            if model in trimmed_models or raw_errors[model] <= 0:
                # Trimmed или invalid: максимальный штраф → exp(-β·10·α) ≈ 0
                normalized_errors[model] = 10.0
            else:
                normalized_errors[model] = raw_errors[model] / scale
        
        print(f"📐 Коэффициенты: α(n={n}) = {alpha:.4f}, σ_scale={scale:.4f}")
        print(f"📏 min MAE = {min_err:.4f}, trim_threshold={trim_threshold:.4f}")
        for model in ['sarima', 'xgboost', 'timellm']:
            tag = " [trimmed]" if model in trimmed_models else ""
            print(f"   {model}: MAE={raw_errors[model]:.4f} → NormMAE={normalized_errors[model]:.4f}{tag}")
        
        # Формула 2.7: Обновление истории ошибок с экспоненциальным сглаживанием.
        # Холодный старт: если история нулевая, инициализируем напрямую (без λ-демпфирования),
        # чтобы разброс ошибок полностью отразился в весах с первого обновления.
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
            exponent = max(exponent, -100)  # exp(-100) ≈ 3.7e-44, достаточно малое
            exp_weights[model] = np.exp(exponent)
            print(f"   {model}: ER_hist={self.error_history[model]:.4f}, exp({exponent:.3f}) = {exp_weights[model]:.6f}")
        
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
        
        # БАГ-3 FIX: Снижен MIN_WEIGHT с 0.05 до 0.01 (только числовая стабильность,
        # не принудительное включение плохих моделей)
        needs_redistribution = any(w < self.MIN_WEIGHT and w > 0 for w in self.weights.values())
        
        if needs_redistribution:
            print(f"🔄 Применяем минимальный порог веса {self.MIN_WEIGHT}")
            for model in self.weights:
                if self.weights[model] < self.MIN_WEIGHT:
                    self.weights[model] = self.MIN_WEIGHT
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
        2. Walk-forward LOO оценка ошибок базовых моделей (ИСПРАВЛЕНИЕ v2)
        3. Обучение базовых моделей на полных данных
        4. Обновление весов EWA (Формулы 2.5, 2.7, 2.11)
        5. Проверка доминирования лучшей модели (ИСПРАВЛЕНИЕ v2)
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
        print(f"   Знак тренда (для Δ_bias): {self._trend_sign:+.1f}")
        
        # ==================== Шаг 2: Walk-forward LOO оценка ошибок ====================
        # Используем walk-forward LOO вместо единственного 80/20 split.
        # При n=10: min_train=5, даём 4-5 фолдов → стабильная оценка весов.
        need_validation = n >= 8  # Минимум 8 точек для LOO с min_train=5
        
        errors = {}
        
        # Горизонт для LOO: используем реальный горизонт, но не более n//4
        # чтобы иметь достаточно фолдов. При n=10, steps=4: horizon=min(4, 2)=2
        loo_horizon = max(1, min(steps if steps else 1, max(1, n // 4)))
        
        if need_validation:
            print("\n" + "="*50)
            print(f"📊 Шаг 2: Walk-forward LOO валидация (горизонт={loo_horizon})...")
            print("="*50)
            
            # SARIMA LOO
            print("🔍 SARIMA LOO...")
            try:
                errors['sarima'] = self._walkforward_errors(
                    self.data_robust,
                    lambda: SARIMAXS(use_cv=False),
                    forecast_horizon=loo_horizon
                )
                print(f"✅ SARIMA LOO MAE = {errors['sarima']:.4f}")
            except Exception as e:
                print(f"⚠️ SARIMA LOO ошибка: {e}")
                errors['sarima'] = float('inf')
            
            # XGBoost LOO
            print("🔍 XGBoost LOO...")
            try:
                errors['xgboost'] = self._walkforward_errors(
                    self.data_robust,
                    lambda: XGBoostTS(use_cv=False),
                    forecast_horizon=loo_horizon
                )
                print(f"✅ XGBoost LOO MAE = {errors['xgboost']:.4f}")
            except Exception as e:
                print(f"⚠️ XGBoost LOO ошибка: {e}")
                errors['xgboost'] = float('inf')
            
            # TimeLLM LOO
            if self.use_slm:
                print("🔍 TimeLLM LOO (SLM режим, 70/30 split)...")
                try:
                    # TimeLLM дорог — используем упрощённый LOO: single 70/30 split
                    split = int(0.7 * n)
                    train_fold = self.data_robust[:split]
                    val_fold = self.data_robust[split:]
                    self._destroy_timellm()
                    tmp_llm = self._create_timellm()
                    tmp_llm.fit(train_fold)
                    pred_llm = tmp_llm.predict(len(val_fold), return_conf_int=False)['forecast']
                    errors['timellm'] = float(np.mean(np.abs(val_fold - np.array(pred_llm)[:len(val_fold)])))
                    print(f"✅ TimeLLM split MAE = {errors['timellm']:.4f}")
                    del tmp_llm
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                    gc.collect()
                except Exception as e:
                    print(f"⚠️ TimeLLM LOO ошибка: {e}")
                    errors['timellm'] = float('inf')
            else:
                # Simple-режим: TimeLLM — статистический fallback.
                # Оцениваем его через walk-forward LOO, как и остальные.
                print("🔍 TimeLLM (simple) LOO...")
                try:
                    errors['timellm'] = self._walkforward_errors(
                        self.data_robust,
                        lambda: TimeLLM(llm_backend="simple"),
                        forecast_horizon=loo_horizon
                    )
                    print(f"✅ TimeLLM simple LOO MAE = {errors['timellm']:.4f}")
                except Exception as e:
                    print(f"⚠️ TimeLLM simple LOO ошибка: {e}")
                    errors['timellm'] = float('inf')
            
            # Сохраняем LOO-ошибки для диагностики
            self._loo_errors = dict(errors)
            
            print(f"\n📊 LOO-ошибки базовых моделей:")
            for k, v in errors.items():
                tag = " ← лучшая" if v == min(errors.values()) else ""
                print(f"   {k}: {v:.4f}{tag}")
        
        # ==================== Шаг 3: Обучение финальных моделей на полных данных ====================
        print("\n" + "="*50)
        print("📊 Шаг 3: Обучение на полных данных...")
        print("="*50)
        
        # SARIMA
        try:
            self.sarima.fit(self.data_robust)
            print("✅ SARIMA обучена на полных данных")
        except Exception as e:
            print(f"⚠️ SARIMA ошибка: {e}")
            if 'sarima' not in errors:
                errors['sarima'] = float('inf')
        
        # XGBoost
        try:
            self.xgboost.fit(self.data_robust)
            print("✅ XGBoost обучена на полных данных")
        except Exception as e:
            print(f"⚠️ XGBoost ошибка: {e}")
            if 'xgboost' not in errors:
                errors['xgboost'] = float('inf')
        
        # TimeLLM
        try:
            self.timellm = self._create_timellm()
            self.timellm.fit(self.data_robust, steps=self._fit_steps)
            print("✅ TimeLLM обучена на полных данных")
        except Exception as e:
            print(f"⚠️ TimeLLM ошибка: {e}")
            if 'timellm' not in errors:
                errors['timellm'] = float('inf')
            self.timellm = TimeLLM(llm_backend="simple")
            self.timellm.fit(self.data_robust, steps=self._fit_steps)
        
        # ==================== Шаг 4: Обновление весов ====================
        if need_validation and errors:
            print("\n" + "="*50)
            print("📊 Шаг 4: Обновление весов ансамбля (EWA)...")
            print("="*50)
            
            self._update_weights(errors)
            
            print(f"\n📊 Итоговые веса (Формула 2.11):")
            print(f"   SARIMA:  {self.weights['sarima']:.4f}")
            print(f"   XGBoost: {self.weights['xgboost']:.4f}")
            print(f"   TimeLLM: {self.weights['timellm']:.4f}")
        
        # ==================== Шаг 5: Проверка доминирования ====================
        # Если одна модель значительно лучше всех остальных, используем только её.
        # Это предотвращает ухудшение ансамбля слабыми участниками.
        #
        # ВАЖНОЕ ОГРАНИЧЕНИЕ: TimeLLM в режиме "simple" является линейным
        # трендовым экстраполятором. Его LOO MAE внутри тренировочного окна
        # может быть низким, но он катастрофически ошибается при смене тренда.
        # Поэтому TimeLLM(simple) не может стать доминирующей моделью.
        # (Реальный TimeLLM с LLM-бэкендом этого ограничения не имеет.)
        self._dominant_model = None
        if need_validation and errors:
            finite_errs = {k: v for k, v in errors.items() if v != float('inf') and v > 0}
            if len(finite_errs) >= 2:
                sorted_errs = sorted(finite_errs.items(), key=lambda x: x[1])
                best_name, best_err = sorted_errs[0]
                
                # TimeLLM(simple) не допускается к доминированию — см. комментарий выше
                if best_name == 'timellm' and not self.use_slm:
                    # Переключаемся на следующую лучшую допустимую модель
                    for k, v in sorted_errs[1:]:
                        if k != 'timellm' or self.use_slm:
                            best_name, best_err = k, v
                            break
                    print(f"ℹ️  TimeLLM(simple) исключён из претендентов на доминирование.")
                
                # БАГ-5 FIX: реальная взвешенная ошибка EWA-ансамбля
                # (используем текущие self.weights, уже рассчитанные в _update_weights)
                weighted_ensemble_err = sum(
                    self.weights.get(k, 0) * v
                    for k, v in finite_errs.items()
                )
                
                # БАГ-2 FIX: порог 0.25 (вместо 0.10) + сравнение с реальным (не proxy) ансамблем
                improvement = (weighted_ensemble_err - best_err) / (weighted_ensemble_err + 1e-12)
                if improvement > self.DOMINANCE_THRESH:
                    self._dominant_model = best_name
                    print(f"\n🏆 Доминирование: {best_name} (LOO MAE={best_err:.4f}) лучше "
                          f"взвешенного ансамбля (MAE={weighted_ensemble_err:.4f}) "
                          f"на {improvement*100:.1f}% > {self.DOMINANCE_THRESH*100:.0f}%")
                    print(f"   → Ансамблевые веса заменяются на 100% {best_name}")
                    # Обнуляем веса остальных моделей
                    for m in self.weights:
                        self.weights[m] = 1.0 if m == best_name else 0.0
                else:
                    print(f"\n📊 Ансамбль используется: best={best_name}({best_err:.4f}), "
                          f"weighted_ens={weighted_ensemble_err:.4f}, "
                          f"улучшение={improvement*100:.1f}% ≤ {self.DOMINANCE_THRESH*100:.0f}%")
                    # В simple-режиме TimeLLM — линейный тренд-экстраполятор,
                    # идентичный SARIMA по характеру. Ограничиваем его вес до 15%,
                    # чтобы не давать тренду двойного веса (SARIMA + TimeLLM simple).
                    # Полное исключение (вес=0) не применяется — TimeLLM simple
                    # может иметь хорошие LOO-ошибки и давать полезный сигнал.
                    SIMPLE_TLM_CAP = 0.15
                    if not self.use_slm and self.weights.get('timellm', 0) > SIMPLE_TLM_CAP:
                        excess = self.weights['timellm'] - SIMPLE_TLM_CAP
                        self.weights['timellm'] = SIMPLE_TLM_CAP
                        # Перераспределяем избыток на SARIMA и XGBoost
                        rest = {k: self.weights[k] for k in ('sarima', 'xgboost')}
                        rest_sum = sum(rest.values())
                        if rest_sum > 0:
                            for k in rest:
                                self.weights[k] += excess * rest[k] / rest_sum
                        else:
                            self.weights['sarima'] = 0.425
                            self.weights['xgboost'] = 0.425
                        print(f"   ↳ TimeLLM(simple) ограничен до {SIMPLE_TLM_CAP} (перераспределено +{excess:.3f})")
        
        # Информация об адаптивных коэффициентах
        print(f"\n📐 Адаптивные коэффициенты для n={n}:")
        print(f"   α(n) = {self._alpha_correction(n):.4f} (Формула 2.6)")
        print(f"   κ(n) = {self._kappa_correction(n):.4f} (Формула 2.12)")
        print(f"   w_LLM(n) = {self._w_llm(n):.4f} (Формула 2.17)")
        print(f"   Confidence = {self._confidence_score(n):.4f} (Формула 2.16)")
        print(f"   Δ_bias sign = {self._trend_sign:+.1f}")
        
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
        
        # Защита от вырожденного прогноза: если dominant_model = xgboost и прогноз
        # плоский (std < 1% от диапазона данных), а SARIMA доступна — добавить SARIMA
        # с весом 0.5 (ансамбль SARIMA + XGBoost).
        if (self._dominant_model == 'xgboost' and steps > 1
                and 'sarima' in forecasts
                and np.std(self.data) > 0):
            xgb_fc = forecasts['xgboost']
            data_range = np.max(self.data) - np.min(self.data)
            fc_std = np.std(xgb_fc)
            if fc_std < 0.01 * (abs(data_range) + 1e-10):
                # XGBoost прогноз вырожден (плоский) — смешиваем с SARIMA
                print(f"⚠️  XGBoost прогноз вырожден (std={fc_std:.4f} < 1% диапазона). "
                      f"Добавляем SARIMA (50/50).")
                ensemble_forecast = 0.5 * forecasts['sarima'] + 0.5 * forecasts['xgboost']
                self.weights = {'sarima': 0.5, 'xgboost': 0.5, 'timellm': 0.0}
                self._dominant_model = None  # теперь это ансамбль
        
        # ==================== Формула 2.8, 2.9: Коррекция смещения ====================
        # Δ_bias применяется к каждому шагу прогноза
        # Адаптивный знак: коррекция по направлению тренда (ИСПРАВЛЕНИЕ v2)
        #
        # ИСПРАВЛЕНИЕ v2b: Δ_bias подавляется когда ансамбль уже учитывает тренд.
        # Условия подавления:
        #   1. Доминирующая модель выбрана (она явно обучена на правильный тренд)
        #   2. Нет явного структурного смещения (тренд незначим _trend_sign == 0)
        #   3. Прогнозируемое изменение ансамбля уже согласуется с трендом
        # Если условия выполнены, Δ_bias обнуляется.
        raw_bias = np.array([self._delta_bias(n, h+1) for h in range(steps)])
        
        # Проверяем: нужна ли коррекция?
        apply_bias = True
        if self._dominant_model is not None:
            # Доминирующая модель уже обучена на всех данных — bias не нужен
            apply_bias = False
        elif self._trend_sign == 0.0:
            # Тренд незначим — Δ_bias будет нулевым
            apply_bias = False
        else:
            # Проверяем: ансамблевый прогноз уже движется в сторону тренда?
            if steps >= 2:
                ensemble_trend = float(ensemble_forecast[-1] - ensemble_forecast[0])
                expected_sign = self._trend_sign
                if ensemble_trend * expected_sign > 0:
                    # Ансамбль уже учитывает тренд — уменьшаем коррекцию вдвое
                    raw_bias = raw_bias * 0.5
        
        bias_corrections = raw_bias if apply_bias else np.zeros(steps)
        
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
        print(f"   dominant_model: {self._dominant_model or 'ансамбль'}")
        print(f"   Ŷ_ensemble: {ensemble_forecast[:3]}...")
        print(f"   Δ_bias (sign={self._trend_sign:+.1f}): {bias_corrections[:3]}...")
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
            'dominant_model': self._dominant_model,
            'loo_errors': {k: float(v) for k, v in self._loo_errors.items()},
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
            'dominant_model': self._dominant_model,
            'loo_errors': {k: float(v) for k, v in self._loo_errors.items()},
            'error_history': self.error_history,
            'trend_sign': self._trend_sign,
            'parameters': {
                'beta': self.BETA,
                'lambda': self.decay_factor,
                'gamma_bias': self.GAMMA_BIAS,
                'T_normal': self.T_NORMAL,
                'MAD_threshold': self.MAD_THRESHOLD,
                'dominance_thresh': self.DOMINANCE_THRESH
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
