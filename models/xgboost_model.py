import numpy as np
import pandas as pd
import math
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
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

class XGBoostTS:
    """XGBoost для временных рядов с автоматическим подбором параметров"""

    def __init__(self, use_cv=True, n_splits=3, n_estimators=100, max_depth=5, learning_rate=0.1):
        """
        Args:
            use_cv: использовать ли кросс-валидацию для подбора параметров
            n_splits: количество сплитов для TimeSeriesSplit
            n_estimators, max_depth, learning_rate: параметры по умолчанию (если use_cv=False)
        """
        self.use_cv = use_cv
        self.n_splits = n_splits

        # Параметры по умолчанию
        self.default_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate
        }

        # Модель будет создана после подбора параметров
        self.model = None
        self.best_params = None
        self.cv_scores = {}

        # Для feature engineering
        self.scaler_X = None
        self.scaler_y = None
        self.feature_lags = None
        self.feature_columns = None  # СОХРАНЯЕМ ИМЕНА СТОЛБЦОВ
        self.data = None

        # Для доверительных интервалов
        self.residuals = None

        # Режим работы
        self.simple_mode = False
        self.last_value = None
        self.trend = None

    def _create_features(self, data, lags=None):
        """Создание признаков из временного ряда"""
        if lags is None:
            # Адаптивные лаги на основе длины ряда
            n = len(data)
            lags = []
            if n >= 7:
                lags.extend([1, 2, 3, 7])
            elif n >= 3:
                lags.extend([1, 2, 3])
            else:
                lags = [1]

            # Добавляем сезонные лаги
            if n >= 30:
                lags.append(30)
            if n >= 365:
                lags.append(365)

            self.feature_lags = lags # Store lags used in this call if not already stored

        df = pd.DataFrame({'value': data})

        # Лаговые признаки
        for lag in lags:
            if lag < len(data):
                df[f'lag_{lag}'] = df['value'].shift(lag)

        # Скользящие средние
        for window in [3, 7, 14]:
            if window < len(data):
                df[f'ma_{window}'] = df['value'].rolling(window=window).mean()

        # Тренд (индекс как признак)
        df['trend'] = np.arange(len(data))

        # Стандартное отклонение
        for window in [7, 14]:
            if window < len(data):
                df[f'std_{window}'] = df['value'].rolling(window=window).std()

        # Разница (momentum)
        df['diff_1'] = df['value'].diff(1)

        return df

    def _time_series_split(self, data, n_splits):
        """
        Генератор сплитов для временных рядов с сохранением порядка

        Args:
            data: временной ряд
            n_splits: количество сплитов

        Yields:
            (train_data, test_data): кортежи данных для обучения и теста
        """
        n = len(data)
        min_train_size = max(10, n // (n_splits + 1))

        for i in range(1, n_splits + 1):
            # Размер train увеличивается с каждым сплитом
            train_end = min_train_size + (i - 1) * (n - min_train_size) // n_splits
            test_end = min(train_end + (n - min_train_size) // n_splits, n)

            if test_end - train_end < 3:  # Минимум 3 точки в test
                continue

            train_data = data[:train_end]
            test_data = data[train_end:test_end]

            yield train_data, test_data

    def _evaluate_params(self, train_data, test_data, params):
        """
        Оценка качества модели с данными параметрами на одном сплите
        ИСПРАВЛЕННАЯ ВЕРСИЯ - устранена ошибка несоответствия признаков
        """
        try:
            # Создание признаков
            df_train = self._create_features(train_data)
            df_train = df_train.dropna()

            if len(df_train) < 3:
                return None

            X_train = df_train.drop('value', axis=1)
            y_train = df_train['value']

            # --- CRITICAL FIX: Determine lags and columns based on this specific training data ---
            # If self.feature_lags wasn't set globally yet (e.g., during CV before main fit),
            # calculate them based on the length of train_data for this fold.
            temp_feature_lags = []
            n = len(train_data)
            if n >= 7:
                temp_feature_lags.extend([1, 2, 3, 7])
            elif n >= 3:
                temp_feature_lags.extend([1, 2, 3])
            else:
                temp_feature_lags = [1]
            if n >= 30:
                temp_feature_lags.append(30)
            if n >= 365:
                temp_feature_lags.append(365)

            temp_feature_columns = X_train.columns.tolist() # Use columns from this specific train set

            # Нормализация
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()

            X_train_scaled = scaler_X.fit_transform(X_train)
            y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

            # Создание и обучение модели
            model = XGBRegressor(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                random_state=42,
                n_jobs=-1
            )

            model.fit(X_train_scaled, y_train_scaled)

            # Прогноз на тестовой выборке
            predictions = []
            current_data = list(train_data)

            for _ in range(len(test_data)):
                df_test = self._create_features(pd.Series(current_data), lags=temp_feature_lags) # Use the lags from this fold's training
                X_test = df_test.drop('value', axis=1).iloc[-1:].copy()

                # --- CRITICAL FIX: Align features with the training set ---
                # Add missing columns
                missing_cols = [col for col in temp_feature_columns if col not in X_test.columns]
                for col in missing_cols:
                    X_test[col] = 0.0

                # Select in the correct order
                X_test = X_test[temp_feature_columns].values # Use the specific columns list from this fold's training

                if np.isnan(X_test).any():
                    X_test = np.nan_to_num(X_test, nan=np.nanmean(current_data))

                # Transform using the scaler fitted on this split's training data
                X_test_scaled = scaler_X.transform(X_test)

                pred_scaled = model.predict(X_test_scaled)[0]
                pred = scaler_y.inverse_transform([[pred_scaled]])[0][0]

                predictions.append(pred)
                current_data.append(pred)

            # Расчёт MAE
            mae = np.mean(np.abs(test_data - np.array(predictions)))

            return mae

        except Exception as e:
            print(f"Ошибка при оценке параметров {params}: {e}")
            # Consider logging traceback for deeper debugging
            # import traceback
            # traceback.print_exc()
            return None


    def _cross_validate_params(self, data, params_list):
        """
        Кросс-валидация параметров на всех сплитах

        Args:
            data: временной ряд
            params_list: список словарей с параметрами для проверки

        Returns:
            Лучшие параметры (словарь)
        """
        best_params = None
        best_score = float('inf')

        print(f"\n🔍 Grid Search с TimeSeriesSplit CV ({self.n_splits} сплитов)")
        print(f"📊 Проверяется {len(params_list)} комбинаций параметров...")

        for params in params_list:
            scores = []

            # Проверка на всех сплитах
            for fold_idx, (train_data, test_data) in enumerate(self._time_series_split(data, self.n_splits)):
                mae = self._evaluate_params(train_data, test_data, params)

                if mae is not None:
                    scores.append(mae)

            if not scores:
                continue

            # Средний MAE по всем сплитам
            mean_mae = np.mean(scores)

            # Сохраняем результаты
            params_key = f"n={params['n_estimators']}, d={params['max_depth']}, lr={params['learning_rate']}"
            self.cv_scores[params_key] = {
                'mean_mae': mean_mae,
                'std_mae': np.std(scores),
                'scores': scores
            }

            # Обновление лучших параметров
            if mean_mae < best_score:
                best_score = mean_mae
                best_params = params

        if best_params is None:
            print("⚠️ Не удалось найти параметры с CV, используем параметры по умолчанию")
            return self.default_params

        print(f"✅ Лучшие параметры: {best_params}")
        print(f"📈 CV MAE: {best_score:.4f}")

        return best_params

    def _find_best_params_grid_search(self, data):
        """
        Grid Search параметров с TimeSeriesSplit CV

        Args:
            data: временной ряд

        Returns:
            Лучшие параметры (словарь)
        """
        # Сетка параметров для поиска
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3]
        }

        # Генерация всех комбинаций
        params_list = []
        for n_est in param_grid['n_estimators']:
            for max_d in param_grid['max_depth']:
                for lr in param_grid['learning_rate']:
                    params_list.append({
                        'n_estimators': n_est,
                        'max_depth': max_d,
                        'learning_rate': lr
                    })

        # Кросс-валидация
        return self._cross_validate_params(data, params_list)

    def fit(self, data):
        """Обучение модели с автоматическим подбором параметров"""
        self.data = data

        # Минимальное количество данных
        if len(data) < 5:
            # Простой режим для малых данных
            self.simple_mode = True
            self.last_value = data[-1]
            self.trend = np.mean(np.diff(data)) if len(data) > 1 else 0
            return self

        # Подбор параметров
        if self.use_cv and len(data) >= 15:
            # Grid Search с TimeSeriesSplit CV (для достаточных данных)
            print("🎯 Используется Grid Search с TimeSeriesSplit CV")
            self.best_params = self._find_best_params_grid_search(data)
        else:
            # Параметры по умолчанию (для малых данных или если CV отключена)
            print("⚡ Используются параметры по умолчанию (данных мало для CV)")
            self.best_params = self.default_params

        # Создание модели с лучшими параметрами
        self.model = XGBRegressor(
            n_estimators=self.best_params['n_estimators'],
            max_depth=self.best_params['max_depth'],
            learning_rate=self.best_params['learning_rate'],
            random_state=42,
            n_jobs=-1
        )

        # Создание признаков
        df = self._create_features(data)

        # Удаление NaN и подготовка данных
        df = df.dropna()

        if len(df) < 3:
            # Если после dropna очень мало данных
            self.simple_mode = True
            self.last_value = data[-1]
            self.trend = np.mean(np.diff(data)) if len(data) > 1 else 0
            return self

        X = df.drop('value', axis=1)
        y = df['value']

        # ✅ ИСПРАВЛЕНИЕ 1: СОХРАНЯЕМ ИМЕНА СТОЛБЦОВ И ИХ ПОРЯДОК
        self.feature_columns = X.columns.tolist()
        print(f"💾 Сохранены столбцы признаков: {self.feature_columns}")
        print(f"   Количество признаков: {len(self.feature_columns)}")

        # Нормализация
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

        # Обучение финальной модели на всех данных
        self.model.fit(X_scaled, y_scaled)

        # Вычисляем остатки для доверительных интервалов
        y_pred_scaled = self.model.predict(X_scaled)
        self.residuals = y_scaled - y_pred_scaled

        self.simple_mode = False

        print(f"✅ XGBoost обучен с параметрами: {self.best_params}")

        return self

    def predict(self, steps, return_conf_int=True, alpha=0.05):
        """
        Прогнозирование на steps шагов вперёд
        ИСПРАВЛЕННАЯ ВЕРСИЯ - гарантирует правильное количество признаков
        """
        if self.data is None:
            raise ValueError("Модель не обучена. Вызовите fit() сначала.")

        # Простой режим для малых данных
        if self.simple_mode:
            forecast = np.array([self.last_value + self.trend * (i+1) for i in range(steps)])
            result = {'forecast': forecast}

            if return_conf_int:
                std_data = np.std(self.data)
                result['lower_bound'] = forecast - 1.96 * std_data
                result['upper_bound'] = forecast + 1.96 * std_data

            return result

        forecast = []
        lower_bounds = []
        upper_bounds = []

        # Текущие данные для построения признаков
        current_data = list(self.data)

        for step_idx in range(steps):
            # Создание признаков для следующего шага
            df = self._create_features(pd.Series(current_data), lags=self.feature_lags) # Use the lags determined during main fit

            # Берём последнюю строку (последние признаки)
            X_new = df.drop('value', axis=1).iloc[-1:].copy()

            # ✅ ГЛАВНОЕ ИСПРАВЛЕНИЕ: ГАРАНТИРУЕМ ПРАВИЛЬНОЕ КОЛИЧЕСТВО СТОЛБЦОВ

            # Находим недостающие столбцы
            missing_cols = [col for col in self.feature_columns if col not in X_new.columns]
            existing_cols = [col for col in self.feature_columns if col in X_new.columns]

            # Добавляем недостающие столбцы с нулями
            for col in missing_cols:
                X_new[col] = 0.0

            # Выбираем ВСЕ столбцы в правильном порядке (только те, что были при обучении)
            X_new = X_new[self.feature_columns].values

            # Проверка размера (для отладки)
            assert X_new.shape[1] == len(self.feature_columns), \
                f"Ошибка: X_new имеет {X_new.shape[1]} столбцов, " \
                f"но ожидается {len(self.feature_columns)}"

            # Проверка на NaN
            if np.isnan(X_new).any():
                X_new = np.nan_to_num(X_new, nan=np.nanmean(current_data))

            # Нормализация - используем transform (scaler уже обучен в fit)
            X_new_scaled = self.scaler_X.transform(X_new)

            # Прогноз
            pred_scaled = self.model.predict(X_new_scaled)[0]
            pred = self.scaler_y.inverse_transform([[pred_scaled]])[0][0]

            forecast.append(pred)

            # Доверительные интервалы на основе остатков
            if return_conf_int and self.residuals is not None:
                from scipy import stats

                std_residual = np.std(self.residuals)
                z_score = stats.norm.ppf(1 - alpha/2)

                # Интервал в масштабированном пространстве
                margin_scaled = z_score * std_residual

                # Обратное преобразование
                lower_scaled = pred_scaled - margin_scaled
                upper_scaled = pred_scaled + margin_scaled

                lower = self.scaler_y.inverse_transform([[lower_scaled]])[0][0]
                upper = self.scaler_y.inverse_transform([[upper_scaled]])[0][0]

                lower_bounds.append(lower)
                upper_bounds.append(upper)

            # Добавляем прогноз к текущим данным
            current_data.append(pred)

        # Очистка прогноза от NaN/Inf
        last_value = float(self.data[-1]) if self.data is not None and len(self.data) > 0 else 0.0
        forecast_clean = sanitize_array(forecast, fallback_value=last_value)
        
        result = {'forecast': forecast_clean}

        if return_conf_int:
            lower_clean = sanitize_array(lower_bounds, fallback_value=forecast_clean[0] * 0.9)
            upper_clean = sanitize_array(upper_bounds, fallback_value=forecast_clean[0] * 1.1)
            
            # Гарантируем, что lower <= forecast <= upper
            result['lower_bound'] = np.minimum(lower_clean, forecast_clean)
            result['upper_bound'] = np.maximum(upper_clean, forecast_clean)

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
        if self.simple_mode:
            return {
                'mode': 'simple',
                'status': 'Простой режим (мало данных)',
                'trend': float(self.trend) if self.trend is not None else 0
            }

        info = {
            'mode': 'xgboost',
            'status': 'Обучена'
        }

        # Добавляем параметры модели
        if self.best_params:
            info['best_params'] = self.best_params
            info['n_estimators'] = self.best_params['n_estimators']
            info['max_depth'] = self.best_params['max_depth']
            info['learning_rate'] = self.best_params['learning_rate']

        # Добавляем информацию о CV
        if self.cv_scores:
            info['cv_used'] = True
            info['cv_splits'] = self.n_splits

            # Находим лучший результат
            best_key = min(self.cv_scores.keys(), key=lambda k: self.cv_scores[k]['mean_mae'])
            info['best_cv_mae'] = float(self.cv_scores[best_key]['mean_mae'])
        else:
            info['cv_used'] = False

        if self.feature_lags:
            info['feature_lags'] = self.feature_lags

        # ✅ ИСПРАВЛЕНИЕ 3: ДОБАВЛЯЕМ ИНФОРМАЦИЮ О КОЛИЧЕСТВЕ ПРИЗНАКОВ
        if self.feature_columns:
            info['n_features'] = len(self.feature_columns)
            info['feature_names'] = self.feature_columns

        return info
