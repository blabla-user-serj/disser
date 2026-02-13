"""
SARIMA-XS модель с адаптивными ограничениями, доверительными интервалами
и Grid Search с TimeSeriesSplit CV
"""
import numpy as np
import pandas as pd
import math
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')


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


class SARIMAXS:
    """SARIMA-XS с адаптивными ограничениями параметров и кросс-валидацией"""
    
    def __init__(self, use_cv=True, n_splits=3):
        """
        Args:
            use_cv: использовать ли кросс-валидацию для подбора параметров
            n_splits: количество сплитов для TimeSeriesSplit (если use_cv=True)
        """
        self.model = None
        self.fitted_model = None
        self.data = None
        self.order = None
        self.seasonal_order = None
        self.use_cv = use_cv
        self.n_splits = n_splits
        self.cv_scores = {}  # История CV скоров
        
    def _adaptive_constraints(self, n):
        """
        Адаптивные ограничения на основе размера выборки.
        
        КРИТИЧНО для коротких рядов (n < 30):
        - Ограничиваем d <= 1 (двойное дифференцирование нестабильно)
        - Ограничиваем общее число параметров p+q <= min(4, n//4)
        - Для очень коротких рядов (n < 15) используем минимальные параметры
        """
        # Для очень коротких рядов - минимальные параметры
        if n < 10:
            return {'p': 1, 'd': 1, 'q': 1, 'P': 0, 'D': 0, 'Q': 0}
        
        # Максимальное общее число параметров (правило: n/4 - 1)
        max_total_params = max(2, n // 4 - 1)
        
        # p_max: AR компоненты
        p_max = min(2, max_total_params)
        
        # d_max: порядок дифференцирования
        # КРИТИЧНО: для коротких рядов (n < 30) d=2 часто приводит к нестабильности
        if n < 30:
            d_max = 1  # Только первое дифференцирование для коротких рядов
        else:
            d_max = min(2, int(n / 10))
        
        # q_max: MA компоненты (ограничиваем сильнее чем p)
        q_max = min(2, max_total_params - 1)
        
        # Сезонные параметры (только для длинных рядов)
        if n >= 24:
            P_max = min(1, n // 24)
            D_max = 1
            Q_max = min(1, n // 24)
        else:
            P_max = 0
            D_max = 0
            Q_max = 0
        
        return {
            'p': p_max, 'd': d_max, 'q': q_max,
            'P': P_max, 'D': D_max, 'Q': Q_max
        }
    
    def _check_stationarity(self, series):
        """Проверка стационарности ряда"""
        try:
            result = adfuller(series.dropna())
            return result[1] < 0.05  # p-value < 0.05 => стационарен
        except:
            return False
    
    def _time_series_split(self, data, n_splits):
        """
        Генератор TimeSeriesSplit для временных рядов
        
        Args:
            data: временной ряд
            n_splits: количество сплитов
            
        Yields:
            (train_indices, test_indices) для каждого сплита
        """
        n = len(data)
        
        # Минимальный размер train
        min_train_size = max(10, n // (n_splits + 1))
        
        # Размер test на каждом сплите
        test_size = max(1, (n - min_train_size) // n_splits)
        
        for i in range(n_splits):
            train_end = min_train_size + i * test_size
            test_end = min(train_end + test_size, n)
            
            if train_end >= n or test_end > n:
                break
            
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(train_end, test_end)
            
            yield train_idx, test_idx
    
    def _cross_validate_params(self, data, params_list, seasonal_period):
        """
        Кросс-валидация для списка параметров
        
        Args:
            data: временной ряд
            params_list: список кортежей (p, d, q, P, D, Q)
            seasonal_period: сезонный период
            
        Returns:
            best_params, cv_scores
        """
        best_params = None
        best_score = np.inf
        cv_scores = {}
        
        for params in params_list:
            p, d, q, P, D, Q = params
            
            # Скоры для каждого сплита
            fold_scores = []
            
            try:
                # TimeSeriesSplit
                for train_idx, test_idx in self._time_series_split(data, self.n_splits):
                    train_data = data[train_idx]
                    test_data = data[test_idx]
                    
                    if len(train_data) < 10 or len(test_data) < 1:
                        continue
                    
                    try:
                        # Обучение модели на train
                        model = SARIMAX(
                            train_data,
                            order=(p, d, q),
                            seasonal_order=(P, D, Q, seasonal_period),
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                        fitted = model.fit(disp=False, maxiter=50)
                        
                        # Прогноз на test
                        forecast = fitted.forecast(steps=len(test_data))
                        
                        # MAE на test
                        mae = np.mean(np.abs(test_data - forecast))
                        fold_scores.append(mae)
                        
                    except:
                        continue
                
                # Средний MAE по всем фолдам
                if fold_scores:
                    mean_score = np.mean(fold_scores)
                    cv_scores[params] = {
                        'mean_mae': mean_score,
                        'std_mae': np.std(fold_scores),
                        'n_folds': len(fold_scores)
                    }
                    
                    if mean_score < best_score:
                        best_score = mean_score
                        best_params = params
                        
            except Exception as e:
                continue
        
        return best_params, cv_scores
    
    def _find_best_order(self, data, constraints):
        """Подбор оптимальных параметров SARIMA с Grid Search"""
        # Определение сезонности
        seasonal_period = self._detect_seasonality(data)
        
        # Генерация сетки параметров
        params_list = []
        
        for p in range(0, constraints['p'] + 1):
            for d in range(0, constraints['d'] + 1):
                for q in range(0, constraints['q'] + 1):
                    if p + q == 0:  # Минимум один параметр
                        continue
                    
                    for P in range(0, constraints['P'] + 1):
                        for D in range(0, constraints['D'] + 1):
                            for Q in range(0, constraints['Q'] + 1):
                                params_list.append((p, d, q, P, D, Q))
        
        print(f"SARIMA Grid Search: {len(params_list)} комбинаций параметров")
        
        # Выбор метода подбора
        if self.use_cv and len(data) >= 20:
            # Grid Search с TimeSeriesSplit CV
            print(f"Используется TimeSeriesSplit CV с {self.n_splits} сплитами")
            
            best_params, cv_scores = self._cross_validate_params(
                data, params_list, seasonal_period
            )
            
            self.cv_scores = cv_scores
            
            if best_params:
                p, d, q, P, D, Q = best_params
                best_order = (p, d, q)
                best_seasonal = (P, D, Q, seasonal_period)
                
                print(f"Лучшие параметры (CV): order={best_order}, seasonal={best_seasonal}")
                print(f"CV MAE: {cv_scores[best_params]['mean_mae']:.4f} "
                      f"± {cv_scores[best_params]['std_mae']:.4f}")
            else:
                # Fallback: если CV не дал результатов
                print("CV не дал результатов, используется AIC")
                best_order, best_seasonal = self._find_best_order_aic(
                    data, constraints, seasonal_period
                )
        else:
            # Fallback: AIC для малых данных
            print(f"Данных мало ({len(data)} точек), используется Grid Search по AIC")
            best_order, best_seasonal = self._find_best_order_aic(
                data, constraints, seasonal_period
            )
        
        return best_order, best_seasonal
    
    def _is_forecast_stable(self, fitted_model, data, steps=5):
        """
        Проверка стабильности прогноза.
        
        Модель считается нестабильной если:
        - Прогноз содержит NaN/Inf
        - Прогноз выходит за разумные границы (> 10x от диапазона данных)
        - Прогноз монотонно уходит в бесконечность
        
        Returns:
            bool: True если прогноз стабилен
        """
        try:
            forecast = fitted_model.forecast(steps=steps)
            
            # Проверка на NaN/Inf
            if not np.isfinite(forecast).all():
                return False
            
            # Диапазон исходных данных
            data_range = np.max(data) - np.min(data)
            data_mean = np.mean(data)
            
            # Прогноз не должен выходить за 10x от диапазона данных
            max_allowed = data_mean + 10 * data_range
            min_allowed = data_mean - 10 * data_range
            
            if np.any(forecast > max_allowed) or np.any(forecast < min_allowed):
                return False
            
            # Проверка на экспоненциальный рост/падение
            # Если последовательные разности растут экспоненциально - нестабильно
            if len(forecast) >= 3:
                diffs = np.abs(np.diff(forecast))
                if len(diffs) >= 2:
                    growth_rate = diffs[1:] / (diffs[:-1] + 1e-10)
                    if np.any(growth_rate > 5):  # Рост более чем в 5 раз
                        return False
            
            return True
            
        except:
            return False
    
    def _find_best_order_aic(self, data, constraints, seasonal_period):
        """
        Подбор параметров по AIC с проверкой стабильности прогноза.
        
        КРИТИЧНО: AIC может выбрать модель с хорошей подгонкой, но нестабильным прогнозом.
        Поэтому добавляем проверку стабильности прогноза.
        """
        best_aic = np.inf
        best_order = (1, 1, 1)
        best_seasonal = (0, 0, 0, seasonal_period)
        
        candidates = []  # Список кандидатов с их AIC и стабильностью
        
        for p in range(0, constraints['p'] + 1):
            for d in range(0, constraints['d'] + 1):
                for q in range(0, constraints['q'] + 1):
                    if p + q == 0:
                        continue
                    
                    # Ограничение на общее число параметров для коротких рядов
                    n = len(data)
                    if n < 30 and (p + d + q) > max(3, n // 5):
                        continue
                    
                    for P in range(0, constraints['P'] + 1):
                        for D in range(0, constraints['D'] + 1):
                            for Q in range(0, constraints['Q'] + 1):
                                try:
                                    model = SARIMAX(
                                        data,
                                        order=(p, d, q),
                                        seasonal_order=(P, D, Q, seasonal_period),
                                        enforce_stationarity=False,
                                        enforce_invertibility=False
                                    )
                                    fitted = model.fit(disp=False, maxiter=50)
                                    
                                    # Проверка стабильности прогноза
                                    is_stable = self._is_forecast_stable(fitted, data)
                                    
                                    candidates.append({
                                        'order': (p, d, q),
                                        'seasonal': (P, D, Q, seasonal_period),
                                        'aic': fitted.aic,
                                        'stable': is_stable,
                                        'fitted': fitted
                                    })
                                    
                                except:
                                    continue
        
        # Сортируем: сначала стабильные по AIC, потом нестабильные
        stable_candidates = [c for c in candidates if c['stable']]
        unstable_candidates = [c for c in candidates if not c['stable']]
        
        stable_candidates.sort(key=lambda x: x['aic'])
        unstable_candidates.sort(key=lambda x: x['aic'])
        
        if stable_candidates:
            best = stable_candidates[0]
            best_order = best['order']
            best_seasonal = best['seasonal']
            best_aic = best['aic']
            print(f"Лучшие параметры (AIC, стабильный): order={best_order}, seasonal={best_seasonal}, AIC={best_aic:.2f}")
            
            if unstable_candidates:
                worst_unstable = unstable_candidates[0]
                print(f"  (отклонено как нестабильное: order={worst_unstable['order']}, AIC={worst_unstable['aic']:.2f})")
        elif unstable_candidates:
            # Если нет стабильных, берём простейшую модель
            print(f"⚠️ Все модели нестабильны! Используем fallback (0,1,1)")
            best_order = (0, 1, 1)
            best_seasonal = (0, 0, 0, seasonal_period)
            best_aic = float('inf')
        else:
            print(f"⚠️ Не удалось подобрать параметры! Используем (1,1,1)")
        
        return best_order, best_seasonal
    
    def _detect_seasonality(self, data, max_lag=None):
        """Определение сезонного периода"""
        n = len(data)
        
        if max_lag is None:
            max_lag = min(n // 2, 52)
        
        if n < 20:
            return 0
        
        try:
            from statsmodels.tsa.stattools import acf
            acf_values = acf(data, nlags=max_lag, fft=True)
            
            peaks = []
            for i in range(2, len(acf_values) - 1):
                if acf_values[i] > acf_values[i-1] and acf_values[i] > acf_values[i+1]:
                    if acf_values[i] > 0.3:
                        peaks.append((i, acf_values[i]))
            
            if peaks:
                return peaks[0][0]
            else:
                if n >= 365:
                    return 12
                elif n >= 52:
                    return 7
                else:
                    return 0
        except:
            return 0
    
    def fit(self, data):
        """Обучение модели"""
        self.data = data
        n = len(data)
        
        print(f"\n{'='*60}")
        print(f"SARIMA-XS: Обучение на {n} точках")
        print(f"{'='*60}")
        
        # Получение адаптивных ограничений
        constraints = self._adaptive_constraints(n)
        print(f"Ограничения: p≤{constraints['p']}, d≤{constraints['d']}, q≤{constraints['q']}, "
              f"P≤{constraints['P']}, D≤{constraints['D']}, Q≤{constraints['Q']}")
        
        # Подбор оптимальных параметров
        self.order, self.seasonal_order = self._find_best_order(data, constraints)
        
        # Обучение финальной модели на всех данных
        print(f"\nОбучение финальной модели на всех {n} точках...")
        self.model = SARIMAX(
            data,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        self.fitted_model = self.model.fit(disp=False, maxiter=100)
        
        print(f"Финальная модель: AIC={self.fitted_model.aic:.2f}, BIC={self.fitted_model.bic:.2f}")
        print(f"{'='*60}\n")
        
        return self
    
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
        if self.fitted_model is None:
            raise ValueError("Модель не обучена. Вызовите fit() сначала.")
        
        # Получение прогноза
        forecast_result = self.fitted_model.get_forecast(steps=steps)
        forecast = forecast_result.predicted_mean
        
        # Fallback значение на основе последних данных
        last_value = float(self.data[-1]) if self.data is not None and len(self.data) > 0 else 0.0
        
        # Очистка прогноза от NaN/Inf
        forecast = sanitize_array(forecast, fallback_value=last_value)
        
        result = {'forecast': forecast}
        
        if return_conf_int:
            # Доверительные интервалы
            conf_int = forecast_result.conf_int(alpha=alpha)
            # conf_int может быть DataFrame или ndarray
            if hasattr(conf_int, 'iloc'):
                lower = conf_int.iloc[:, 0].values
                upper = conf_int.iloc[:, 1].values
            else:
                lower = conf_int[:, 0]
                upper = conf_int[:, 1]
            
            # Очистка доверительных интервалов от NaN/Inf
            # Используем прогноз как базу для fallback
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
        
        # R²
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
        if self.fitted_model is None:
            return {
                'order': None,
                'seasonal_order': None,
                'aic': None,
                'bic': None,
                'status': 'Модель не обучена',
                'cv_used': self.use_cv
            }
        
        try:
            params_dict = {str(k): float(v) for k, v in self.fitted_model.params.items()}
        except:
            params_dict = {}
        
        info = {
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'aic': float(self.fitted_model.aic),
            'bic': float(self.fitted_model.bic),
            'params': params_dict,
            'status': 'Обучена',
            'cv_used': self.use_cv,
            'n_splits': self.n_splits if self.use_cv else None
        }
        
        # Добавляем CV скоры если есть
        if self.cv_scores and self.order:
            best_params = self.order + self.seasonal_order[:3]
            if best_params in self.cv_scores:
                info['cv_score'] = self.cv_scores[best_params]
        
        return info
