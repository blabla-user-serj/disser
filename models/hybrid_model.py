"""
Гибридная модель с адаптивным взвешиванием и доверительными интервалами
"""
import numpy as np
from .sarima_xs import SARIMAXS
from .xgboost_model import XGBoostTS
from .timellm_gguf import TimeLLM


class HybridModel:
    """Гибридная модель с адаптивным взвешиванием"""
    
    def __init__(self, decay_factor=0.9, use_cv=True, n_splits=3, use_slm=True, slm_model='qwen2-0.5b'):
        """
        Args:
            decay_factor: коэффициент экспоненциального сглаживания для весов
            use_cv: использовать ли CV в базовых моделях
            n_splits: количество сплитов для CV
            use_slm: использовать ли SLM модели (NeuralForecast)
            slm_model: какую SLM модель использовать (qwen2-0.5b, llama3.2-1b, gemma-2b и т.д.)
        """
        self.sarima = SARIMAXS(use_cv=use_cv, n_splits=n_splits)
        self.xgboost = XGBoostTS(use_cv=use_cv, n_splits=n_splits)
        
        # TimeLLM с выбором режима
        if use_slm:
            # Используем NeuralForecast с современными SLM 2024-2025
            # По умолчанию: Qwen2-0.5B (500M) - самая лёгкая и быстрая
            print(f"🤖 HybridModel: используется TimeLLM с NeuralForecast + SLM '{slm_model}'")
            self.timellm = TimeLLM(
                llm_backend="neuralforecast", 
                neuralforecast_model=slm_model
            )
        else:
            # Используем Simple режим (быстро, без GPU)
            print("⚡ HybridModel: используется TimeLLM в Simple режиме (без GPU)")
            self.timellm = TimeLLM(llm_backend="simple")
        
        self.decay_factor = decay_factor  # λ для экспоненциального сглаживания
        
        self.weights = {
            'sarima': 1/3,
            'xgboost': 1/3,
            'timellm': 1/3
        }
        
        self.error_history = {
            'sarima': 0,
            'xgboost': 0,
            'timellm': 0
        }
        
        self.data = None
        
    def _correction_factor(self, n):
        """Коэффициент коррекции на основе размера выборки"""
        return 1 + np.log(30 / n) if n < 30 else 1.0
    
    def _update_weights(self, errors):
        """
        Обновление весов моделей на основе ошибок
        
        errors: dict {'sarima': error, 'xgboost': error, 'timellm': error}
        """
        n = len(self.data)
        alpha = self._correction_factor(n)
        
        # Обновление истории ошибок с экспоненциальным сглаживанием
        for model in ['sarima', 'xgboost', 'timellm']:
            self.error_history[model] = (
                self.decay_factor * self.error_history[model] + 
                (1 - self.decay_factor) * errors.get(model, 0)
            )
        
        # Вычисление весов: w_i = exp(-β * ER_i * α) / Σ
        beta = 1.0  # Параметр чувствительности
        
        exp_weights = {}
        for model in ['sarima', 'xgboost', 'timellm']:
            exp_weights[model] = np.exp(-beta * self.error_history[model] * alpha)
        
        # Нормализация
        total = sum(exp_weights.values())
        
        if total > 0:
            for model in ['sarima', 'xgboost', 'timellm']:
                self.weights[model] = exp_weights[model] / total
        else:
            # Fallback: равные веса
            for model in ['sarima', 'xgboost', 'timellm']:
                self.weights[model] = 1/3
    
    def fit(self, data):
        """Обучение всех моделей"""
        self.data = data
        
        # Обучение каждой модели
        try:
            self.sarima.fit(data)
        except Exception as e:
            print(f"Warning: SARIMA не обучена: {e}")
        
        try:
            self.xgboost.fit(data)
        except Exception as e:
            print(f"Warning: XGBoost не обучена: {e}")
        
        try:
            self.timellm.fit(data)
        except Exception as e:
            print(f"Warning: TimeLLM не обучена: {e}")
        
        # Вычисление начальных ошибок (на обучающих данных)
        n = len(data)
        if n > 10:
            # Используем последние 20% для валидации
            split = int(0.8 * n)
            train_data = data[:split]
            val_data = data[split:]
            val_steps = len(val_data)
            
            errors = {}
            
            # SARIMA
            try:
                temp_sarima = SARIMAXS(use_cv=False)  # Без CV для скорости
                temp_sarima.fit(train_data)
                pred = temp_sarima.predict(val_steps, return_conf_int=False)['forecast']
                errors['sarima'] = np.mean(np.abs(val_data - pred))
            except:
                errors['sarima'] = 0
            
            # XGBoost
            try:
                temp_xgb = XGBoostTS(use_cv=False)  # Без CV для скорости
                temp_xgb.fit(train_data)
                pred = temp_xgb.predict(val_steps, return_conf_int=False)['forecast']
                errors['xgboost'] = np.mean(np.abs(val_data - pred))
            except:
                errors['xgboost'] = 0
            
            # TimeLLM
            try:
                temp_llm = TimeLLM()
                temp_llm.fit(train_data)
                pred = temp_llm.predict(val_steps, return_conf_int=False)['forecast']
                errors['timellm'] = np.mean(np.abs(val_data - pred))
            except:
                errors['timellm'] = 0
            
            # Обновление весов
            self._update_weights(errors)
        
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
                'forecast': взвешенный прогноз,
                'lower_bound': нижняя граница (если return_conf_int=True),
                'upper_bound': верхняя граница (если return_conf_int=True),
                'weights': веса моделей,
                'individual_forecasts': прогнозы отдельных моделей
            }
        """
        forecasts = {}
        lower_bounds = {}
        upper_bounds = {}
        
        # SARIMA
        try:
            result = self.sarima.predict(steps, return_conf_int=return_conf_int, alpha=alpha)
            forecasts['sarima'] = result['forecast']
            if return_conf_int:
                lower_bounds['sarima'] = result['lower_bound']
                upper_bounds['sarima'] = result['upper_bound']
        except Exception as e:
            print(f"Warning: SARIMA prediction failed: {e}")
            forecasts['sarima'] = np.zeros(steps)
            if return_conf_int:
                lower_bounds['sarima'] = np.zeros(steps)
                upper_bounds['sarima'] = np.zeros(steps)
        
        # XGBoost
        try:
            result = self.xgboost.predict(steps, return_conf_int=return_conf_int, alpha=alpha)
            forecasts['xgboost'] = result['forecast']
            if return_conf_int:
                lower_bounds['xgboost'] = result['lower_bound']
                upper_bounds['xgboost'] = result['upper_bound']
        except Exception as e:
            print(f"Warning: XGBoost prediction failed: {e}")
            forecasts['xgboost'] = np.zeros(steps)
            if return_conf_int:
                lower_bounds['xgboost'] = np.zeros(steps)
                upper_bounds['xgboost'] = np.zeros(steps)
        
        # TimeLLM
        try:
            result = self.timellm.predict(steps, return_conf_int=return_conf_int, alpha=alpha)
            forecasts['timellm'] = result['forecast']
            if return_conf_int:
                lower_bounds['timellm'] = result['lower_bound']
                upper_bounds['timellm'] = result['upper_bound']
        except Exception as e:
            print(f"Warning: TimeLLM prediction failed: {e}")
            forecasts['timellm'] = np.zeros(steps)
            if return_conf_int:
                lower_bounds['timellm'] = np.zeros(steps)
                upper_bounds['timellm'] = np.zeros(steps)
        
        # Взвешенная комбинация
        hybrid_forecast = (
            self.weights['sarima'] * forecasts['sarima'] +
            self.weights['xgboost'] * forecasts['xgboost'] +
            self.weights['timellm'] * forecasts['timellm']
        )
        
        result = {
            'forecast': hybrid_forecast,
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
            
            result['lower_bound'] = hybrid_lower
            result['upper_bound'] = hybrid_upper
        
        return result
    
    def get_metrics(self, y_true, y_pred):
        """Расчёт метрик качества"""
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        # R²
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2
        }
    
    def get_info(self):
        """Информация о модели"""
        info = {
            'weights': self.weights,
            'error_history': self.error_history
        }
        
        # Безопасное получение информации о моделях
        try:
            info['sarima_info'] = self.sarima.get_info()
        except:
            info['sarima_info'] = {'status': 'Ошибка получения информации'}
        
        try:
            info['xgboost_info'] = self.xgboost.get_info()
        except:
            info['xgboost_info'] = {'status': 'Ошибка получения информации'}
        
        try:
            info['timellm_info'] = self.timellm.get_info()
        except:
            info['timellm_info'] = {'status': 'Ошибка получения информации'}
        
        return info
