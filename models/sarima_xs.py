"""
SARIMA-XS модель с адаптивными ограничениями, доверительными интервалами
и Grid Search с TimeSeriesSplit CV
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')


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
        """Адаптивные ограничения на основе размера выборки"""
        p_max = min(3, n - 3)
        d_max = min(2, int(n / 4))
        q_max = min(3, n - 3)
        
        # Сезонные параметры (если данных достаточно)
        P_max = min(2, max(0, n // 20))
        D_max = 1 if n >= 24 else 0
        Q_max = min(2, max(0, n // 20))
        
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
    
    def _find_best_order_aic(self, data, constraints, seasonal_period):
        """Подбор параметров по AIC (fallback для малых данных)"""
        best_aic = np.inf
        best_order = (1, 1, 1)
        best_seasonal = (0, 0, 0, seasonal_period)
        
        for p in range(0, constraints['p'] + 1):
            for d in range(0, constraints['d'] + 1):
                for q in range(0, constraints['q'] + 1):
                    if p + q == 0:
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
                                    
                                    if fitted.aic < best_aic:
                                        best_aic = fitted.aic
                                        best_order = (p, d, q)
                                        best_seasonal = (P, D, Q, seasonal_period)
                                except:
                                    continue
        
        print(f"Лучшие параметры (AIC): order={best_order}, seasonal={best_seasonal}, AIC={best_aic:.2f}")
        
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
        
        result = {'forecast': forecast}
        
        if return_conf_int:
            # Доверительные интервалы
            conf_int = forecast_result.conf_int(alpha=alpha)
            # conf_int может быть DataFrame или ndarray
            if hasattr(conf_int, 'iloc'):
                result['lower_bound'] = conf_int.iloc[:, 0].values
                result['upper_bound'] = conf_int.iloc[:, 1].values
            else:
                result['lower_bound'] = conf_int[:, 0]
                result['upper_bound'] = conf_int[:, 1]
        
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
