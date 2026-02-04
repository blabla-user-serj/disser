"""
Гибридная модель с адаптивным взвешиванием и доверительными интервалами
"""
import numpy as np
import gc
import torch
from .sarima_xs import SARIMAXS
from .xgboost_model import XGBoostTS
from .timellm_gguf import TimeLLM, clear_gpu_memory_completely


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
        # Сохраняем параметры для отложенной инициализации TimeLLM
        self.use_cv = use_cv
        self.n_splits = n_splits
        self.use_slm = use_slm
        self.slm_model = slm_model
        
        self.sarima = SARIMAXS(use_cv=use_cv, n_splits=n_splits)
        self.xgboost = XGBoostTS(use_cv=use_cv, n_splits=n_splits)
        
        # TimeLLM создаётся позже при обучении, чтобы контролировать память GPU
        self.timellm = None
        
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
    
    def _create_timellm(self):
        """Создание TimeLLM модели с предварительной очисткой GPU памяти"""
        # КРИТИЧНО: Полная очистка GPU памяти перед созданием TimeLLM
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
            
            # Удаляем внутреннюю модель NeuralForecast
            if hasattr(self.timellm, 'model') and self.timellm.model is not None:
                del self.timellm.model
                self.timellm.model = None
            
            del self.timellm
            self.timellm = None
        
        # Полная очистка GPU памяти
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
            
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"🧹 GPU память после удаления: выделено={allocated:.2f} GB, зарезервировано={reserved:.2f} GB")
        
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
        """
        Обучение всех моделей с последовательной валидацией.
        
        Для честной оценки весов TimeLLM обучается дважды:
        1. На train_data для валидации (затем удаляется)
        2. На полных данных для финального использования
        
        Это обеспечивает корректную оценку ошибок для взвешивания ансамбля.
        """
        self.data = data
        n = len(data)
        
        # Определяем нужна ли валидация для весов
        need_validation = n > 10
        
        if need_validation:
            split = int(0.8 * n)
            train_data = data[:split]
            val_data = data[split:]
            val_steps = len(val_data)
        
        errors = {}
        
        # ==================== SARIMA ====================
        print("\n" + "="*50)
        print("📊 Обучение SARIMA...")
        print("="*50)
        
        try:
            if need_validation:
                # Валидация на train/val split
                temp_sarima = SARIMAXS(use_cv=False)
                temp_sarima.fit(train_data)
                pred = temp_sarima.predict(val_steps, return_conf_int=False)['forecast']
                errors['sarima'] = np.mean(np.abs(val_data - pred))
                print(f"✅ SARIMA валидация: MAE = {errors['sarima']:.4f}")
                del temp_sarima
            
            # Обучение на полных данных
            self.sarima.fit(data)
            print("✅ SARIMA обучена на полных данных")
            
        except Exception as e:
            print(f"⚠️ SARIMA ошибка: {e}")
            errors['sarima'] = float('inf')
        
        # ==================== XGBoost ====================
        print("\n" + "="*50)
        print("📊 Обучение XGBoost...")
        print("="*50)
        
        try:
            if need_validation:
                # Валидация на train/val split
                temp_xgb = XGBoostTS(use_cv=False)
                temp_xgb.fit(train_data)
                pred = temp_xgb.predict(val_steps, return_conf_int=False)['forecast']
                errors['xgboost'] = np.mean(np.abs(val_data - pred))
                print(f"✅ XGBoost валидация: MAE = {errors['xgboost']:.4f}")
                del temp_xgb
            
            # Обучение на полных данных
            self.xgboost.fit(data)
            print("✅ XGBoost обучена на полных данных")
            
        except Exception as e:
            print(f"⚠️ XGBoost ошибка: {e}")
            errors['xgboost'] = float('inf')
        
        # ==================== TimeLLM ====================
        print("\n" + "="*50)
        print("📊 Обучение TimeLLM...")
        print("="*50)
        
        try:
            if need_validation and self.use_slm:
                # ШАГ 1: Валидация - обучаем TimeLLM на train_data
                print("🔄 Этап 1: Валидация TimeLLM на train_data...")
                self._destroy_timellm()  # Убедимся что память свободна
                
                temp_timellm = self._create_timellm()
                temp_timellm.fit(train_data)
                pred = temp_timellm.predict(val_steps, return_conf_int=False)['forecast']
                
                if len(pred) >= val_steps:
                    errors['timellm'] = np.mean(np.abs(val_data - pred[:val_steps]))
                else:
                    errors['timellm'] = np.mean(np.abs(val_data[:len(pred)] - pred))
                
                print(f"✅ TimeLLM валидация: MAE = {errors['timellm']:.4f}")
                
                # Удаляем временную модель и освобождаем GPU память
                del temp_timellm
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    allocated = torch.cuda.memory_allocated(0) / 1024**3
                    print(f"🧹 GPU память после валидации: {allocated:.2f} GB")
                
                # ШАГ 2: Обучение на полных данных
                print("🔄 Этап 2: Обучение TimeLLM на полных данных...")
            
            # Создаём и обучаем финальную модель
            self.timellm = self._create_timellm()
            self.timellm.fit(data)
            print("✅ TimeLLM обучена на полных данных")
            
        except Exception as e:
            print(f"⚠️ TimeLLM ошибка: {e}")
            errors['timellm'] = float('inf')
            # Создаём fallback Simple модель
            self.timellm = TimeLLM(llm_backend="simple")
            self.timellm.fit(data)
        
        # ==================== Обновление весов ====================
        if need_validation:
            print("\n" + "="*50)
            print("📊 Обновление весов ансамбля...")
            print("="*50)
            print(f"   SARIMA MAE:  {errors.get('sarima', 'N/A')}")
            print(f"   XGBoost MAE: {errors.get('xgboost', 'N/A')}")
            print(f"   TimeLLM MAE: {errors.get('timellm', 'N/A')}")
            
            self._update_weights(errors)
            
            print(f"\n📊 Итоговые веса:")
            print(f"   SARIMA:  {self.weights['sarima']:.4f}")
            print(f"   XGBoost: {self.weights['xgboost']:.4f}")
            print(f"   TimeLLM: {self.weights['timellm']:.4f}")
        
        print("\n" + "="*50)
        print("✅ Гибридная модель обучена")
        print("="*50 + "\n")
        
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
