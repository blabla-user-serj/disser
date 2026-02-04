"""
FastAPI сервер для прогнозирования временных рядов
"""
import sys
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"

import torch
import gc

# Добавляем родительскую директорию в путь для импорта models
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import pandas as pd
import numpy as np
from io import BytesIO, StringIO
from datetime import datetime, timedelta
import json

from models import SARIMAXS, XGBoostTS, TimeLLM, HybridModel
from backend.llm_expert import LLMExpert

# Загрузка переменных окружения из .env
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'config', '.env'))

torch.cuda.empty_cache()
gc.collect()
torch.set_float32_matmul_precision('high')
torch.set_grad_enabled(False)

app = FastAPI(title="Hybrid Forecast API", version="2.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальное хранилище для последнего прогноза (для экспорта)
last_forecast_data = {}


def infer_frequency(dates: pd.Series) -> str:
    """
    Определение частоты временных рядов
    
    Returns:
        str: 'hourly', 'daily', 'weekly', 'monthly', 'yearly', 'unknown'
    """
    if len(dates) < 2:
        return 'unknown'
    
    # Вычисляем медианную разницу между датами
    diffs = pd.Series(dates).diff().dropna()
    median_diff = diffs.median()
    
    # Определяем частоту
    if median_diff <= timedelta(hours=1):
        return 'часовая'
    elif median_diff <= timedelta(days=1, hours=12):
        return 'дневная'
    elif median_diff <= timedelta(days=8):
        return 'недельная'
    elif median_diff <= timedelta(days=35):
        return 'месячная'
    elif median_diff <= timedelta(days=400):
        return 'годовая'
    else:
        return 'unknown'


def generate_forecast_dates(last_date: datetime, steps: int, frequency: str) -> pd.DatetimeIndex:
    """
    Генерация дат для прогноза с правильной частотой
    
    Args:
        last_date: последняя дата в исторических данных
        steps: количество шагов прогноза
        frequency: частота ('hourly', 'daily', 'weekly', 'monthly', 'yearly')
        
    Returns:
        pd.DatetimeIndex: даты прогноза
    """
    if frequency == 'hourly':
        return pd.date_range(start=last_date + timedelta(hours=1), periods=steps, freq='H')
    elif frequency == 'daily':
        return pd.date_range(start=last_date + timedelta(days=1), periods=steps, freq='D')
    elif frequency == 'weekly':
        return pd.date_range(start=last_date + timedelta(weeks=1), periods=steps, freq='W')
    elif frequency == 'monthly':
        return pd.date_range(start=last_date + pd.DateOffset(months=1), periods=steps, freq='MS')
    elif frequency == 'yearly':
        return pd.date_range(start=last_date + pd.DateOffset(years=1), periods=steps, freq='YS')
    else:
        # Fallback: дневная частота
        return pd.date_range(start=last_date + timedelta(days=1), periods=steps, freq='D')


def parse_csv(file_content: bytes):
    """Парсинг CSV с автоопределением разделителя и колонок"""
    # Пробуем разные кодировки
    for encoding in ['utf-8', 'cp1251', 'latin1']:
        try:
            content = file_content.decode(encoding)
            break
        except:
            continue
    else:
        raise ValueError("Не удалось определить кодировку файла")
    
    # Пробуем разные разделители
    for sep in [',', ';', '\t']:
        try:
            df = pd.read_csv(StringIO(content), sep=sep)
            
            if df.shape[1] >= 2:  # Минимум 2 колонки
                # Автоопределение колонок
                date_col = None
                value_col = None
                
                # Поиск колонки с датами
                for col in df.columns:
                    try:
                        pd.to_datetime(df[col].iloc[0])
                        date_col = col
                        break
                    except:
                        continue
                
                # Поиск колонки с числовыми значениями
                for col in df.columns:
                    if col != date_col:
                        try:
                            pd.to_numeric(df[col])
                            value_col = col
                            break
                        except:
                            continue
                
                if date_col and value_col:
                    return df, date_col, value_col
        except:
            continue
    
    # Если не получилось автоопределить, берём первые две колонки
    df = pd.read_csv(StringIO(content))
    return df, df.columns[0], df.columns[1]


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Загрузка CSV/XLSX файла"""
    try:
        content = await file.read()
        
        if file.filename.endswith('.csv'):
            df, date_col, value_col = parse_csv(content)
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(BytesIO(content))
            date_col, value_col = df.columns[0], df.columns[1]
        else:
            raise HTTPException(400, "Поддерживаются только CSV и XLSX файлы")
        
        # Обработка данных
        df[date_col] = pd.to_datetime(df[date_col])
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        
        # Удаление NaN
        df = df.dropna()
        
        # Сортировка по дате
        df = df.sort_values(by=date_col)
        
        # Определение частоты
        frequency = infer_frequency(df[date_col])
        
        return {
            "status": "success",
            "rows": len(df),
            "date_column": date_col,
            "value_column": value_col,
            "frequency": frequency,
            "data": {
                "dates": df[date_col].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                "values": df[value_col].tolist()
            }
        }
    except Exception as e:
        raise HTTPException(500, f"Ошибка обработки файла: {str(e)}")


@app.post("/forecast")
async def forecast(
    dates: str = Form(...),
    values: str = Form(...),
    model_type: str = Form(...),
    steps: int = Form(...),
    web_urls: str = Form(None)
):
    """Прогнозирование временных рядов"""
    global last_forecast_data
    
    try:
        # Парсинг данных
        dates_list = json.loads(dates)
        values_list = json.loads(values)
        
        dates_array = pd.to_datetime(dates_list)
        values_array = np.array(values_list, dtype=float)
        
        # Определение частоты
        frequency = infer_frequency(dates_array)
        
        # ПОЛНАЯ очистка GPU памяти перед обучением моделей
        if torch.cuda.is_available():
            print("🧹 Выполняю полную очистку GPU памяти перед обучением...")
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            gc.collect()
            torch.cuda.empty_cache()
            
            # Показываем состояние памяти
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"📊 GPU память: выделено={allocated:.2f} GB, зарезервировано={reserved:.2f} GB")
        
        # Обучение модели
        if model_type == 'sarima':
            model = SARIMAXS()
        elif model_type == 'xgboost':
            model = XGBoostTS()
        elif model_type == 'timellm':
            # Параметр llm_model можно передать для выбора конкретной SLM
            llm_model = data.get('llm_model', 'qwen2-0.5b')  # По умолчанию Qwen2-0.5B
            model = TimeLLM(
                llm_backend='neuralforecast',  # Используем NeuralForecast с SLM
                neuralforecast_model=llm_model
            )
        elif model_type == 'hybrid':
            model = HybridModel()
        else:
            raise HTTPException(400, f"Неизвестная модель: {model_type}")
        
        try:
            model.fit(values_array)
        except Exception as e:
            print(f"\n❌ Ошибка при обучении модели {model_type}:")
            print(f"   {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Специальное сообщение для TimeLLM
            if model_type == 'timellm':
                error_msg = f"Ошибка обучения TimeLLM: {str(e)}\n\n"
                error_msg += "💡 Рекомендации:\n"
                error_msg += "- Попробуйте модель SARIMA или XGBoost (быстрее и стабильнее)\n"
                error_msg += "- Используйте Гибридную модель для лучшего качества\n"
                error_msg += "- Убедитесь что GPU драйверы установлены корректно\n"
                error_msg += "- TimeLLM требует CUDA и может быть нестабильным на Windows"
                raise HTTPException(500, error_msg)
            else:
                raise HTTPException(500, f"Ошибка обучения модели {model_type}: {str(e)}")
        
        # Прогнозирование с доверительными интервалами
        try:
            result = model.predict(steps, return_conf_int=True, alpha=0.05)
        except Exception as e:
            print(f"\n❌ Ошибка при прогнозировании модели {model_type}:")
            print(f"   {str(e)}")
            import traceback
            traceback.print_exc()
            raise HTTPException(500, f"Ошибка прогнозирования модели {model_type}: {str(e)}")
        
        forecast_values = result['forecast']
        lower_bound = result.get('lower_bound', forecast_values * 0.95)
        upper_bound = result.get('upper_bound', forecast_values * 1.05)
        
        # Генерация дат прогноза с правильной частотой
        last_date = dates_array[-1]
        forecast_dates = generate_forecast_dates(last_date, steps, frequency)
        
        # LLM коррекция прогноза
        try:
            llm_expert = LLMExpert()
            
            # Парсинг веб-ссылок
            web_urls_list = []
            if web_urls:
                web_urls_list = [url.strip() for url in web_urls.split('\n') if url.strip()]
            
            # Коррекция прогноза
            correction_result = llm_expert.correct_forecast(
                historical_data=values_array,
                forecast=forecast_values,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                web_urls=web_urls_list
            )
            
            corrected_forecast = correction_result['corrected_forecast']
            corrected_lower = correction_result['corrected_lower']
            corrected_upper = correction_result['corrected_upper']
            llm_analysis = correction_result['analysis']
            correction_applied = correction_result['correction_applied']
        except Exception as e:
            print(f"\n⚠️  Ошибка при LLM коррекции: {str(e)}")
            print(f"   Используется прогноз без коррекции")
            # Fallback: используем прогноз без коррекции
            corrected_forecast = forecast_values
            corrected_lower = lower_bound
            corrected_upper = upper_bound
            llm_analysis = f"⚠️ LLM коррекция недоступна: {str(e)}\n\nИспользуется базовый прогноз модели."
            correction_applied = False
        
        # Сохранение для экспорта
        last_forecast_data = {
            'historical_dates': dates_array,
            'historical_values': values_array,
            'forecast_dates': forecast_dates,
            'forecast_values': corrected_forecast,
            'lower_bound': corrected_lower,
            'upper_bound': corrected_upper,
            'model_type': model_type,
            'frequency': frequency
        }
        
        # Метрики (на исторических данных)
        if len(values_array) > 10:
            train_size = int(0.8 * len(values_array))
            y_true = values_array[train_size:]
            
            # Прогноз на валидационном наборе
            temp_model = model.__class__()
            temp_model.fit(values_array[:train_size])
            y_pred = temp_model.predict(len(y_true), return_conf_int=False)['forecast']
            
            metrics = model.get_metrics(y_true, y_pred)
        else:
            metrics = {'MAE': 0, 'RMSE': 0, 'R2': 0}
        
        # Информация о модели
        model_info = model.get_info()
        
        # Веса (для гибридной модели)
        weights = model_info.get('weights', {}) if model_type == 'hybrid' else {}
        
        return {
            "status": "success",
            "historical": {
                "dates": dates_array.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                "values": values_array.tolist()
            },
            "forecast": {
                "dates": forecast_dates.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                "values": corrected_forecast.tolist(),
                "lower_bound": corrected_lower.tolist(),
                "upper_bound": corrected_upper.tolist()
            },
            "frequency": frequency,
            "metrics": {
                "MAE": float(metrics['MAE']),
                "RMSE": float(metrics['RMSE']),
                "R2": float(metrics['R2'])
            },
            "llm_analysis": llm_analysis,
            "correction_applied": correction_applied,
            "weights": weights,
            "model_info": model_info
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Ошибка прогнозирования: {str(e)}")


@app.get("/export")
async def export_forecast(format: str = 'csv'):
    """Экспорт результатов прогноза в CSV или XLSX"""
    global last_forecast_data
    
    if not last_forecast_data:
        raise HTTPException(400, "Нет данных для экспорта. Сначала выполните прогнозирование.")
    
    try:
        # Создание DataFrame для экспорта
        # Исторические данные
        hist_df = pd.DataFrame({
            'date': last_forecast_data['historical_dates'],
            'actual': last_forecast_data['historical_values'],
            'forecast': [None] * len(last_forecast_data['historical_values']),
            'lower_bound': [None] * len(last_forecast_data['historical_values']),
            'upper_bound': [None] * len(last_forecast_data['historical_values']),
            'type': ['historical'] * len(last_forecast_data['historical_values'])
        })
        
        # Прогнозные данные
        fore_df = pd.DataFrame({
            'date': last_forecast_data['forecast_dates'],
            'actual': [None] * len(last_forecast_data['forecast_dates']),
            'forecast': last_forecast_data['forecast_values'],
            'lower_bound': last_forecast_data['lower_bound'],
            'upper_bound': last_forecast_data['upper_bound'],
            'type': ['forecast'] * len(last_forecast_data['forecast_dates'])
        })
        
        # Объединение
        export_df = pd.concat([hist_df, fore_df], ignore_index=True)
        
        # Добавление метаданных
        export_df['model'] = last_forecast_data['model_type']
        export_df['frequency'] = last_forecast_data['frequency']
        
        # Экспорт в зависимости от формата
        if format == 'csv':
            output = StringIO()
            export_df.to_csv(output, index=False, encoding='utf-8')
            output.seek(0)
            
            return StreamingResponse(
                iter([output.getvalue()]),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"}
            )
        
        elif format == 'xlsx':
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                export_df.to_excel(writer, index=False, sheet_name='Forecast')
            output.seek(0)
            
            return StreamingResponse(
                output,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": f"attachment; filename=forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"}
            )
        
        else:
            raise HTTPException(400, f"Неподдерживаемый формат: {format}")
        
    except Exception as e:
        raise HTTPException(500, f"Ошибка экспорта: {str(e)}")


@app.get("/")
async def root():
    """Информация об API"""
    return {
        "name": "Hybrid Forecast API",
        "version": "2.0",
        "endpoints": {
            "/upload": "POST - Загрузка CSV/XLSX файла",
            "/forecast": "POST - Прогнозирование временных рядов",
            "/export": "GET - Экспорт результатов (параметр: format=csv|xlsx)",
            "/health": "GET - Проверка здоровья сервиса",
            "/models": "GET - Описание моделей"
        },
        "features": [
            "4 модели прогнозирования (SARIMA-XS, XGBoost, TimeLLM, Hybrid)",
            "Адаптивное взвешивание моделей",
            "Доверительные интервалы",
            "LLM-эксперт с коррекцией прогноза (YandexGPT)",
            "Автоматическое определение частоты данных",
            "Экспорт результатов в CSV/XLSX"
        ]
    }


@app.get("/health")
async def health():
    """Проверка состояния сервиса"""
    return {
        "status": "healthy",
        "yandex_api_configured": bool(os.getenv('YANDEX_API_KEY') and os.getenv('YANDEX_FOLDER_ID'))
    }


@app.get("/models")
async def models_info():
    """Описание доступных моделей"""
    return {
        "sarima": {
            "name": "SARIMA-XS",
            "description": "SARIMA с адаптивными ограничениями параметров",
            "features": ["Автоподбор параметров", "Сезонность", "Доверительные интервалы"]
        },
        "xgboost": {
            "name": "XGBoost",
            "description": "Gradient boosting для временных рядов",
            "features": ["Feature engineering", "Лаговые признаки", "Доверительные интервалы"]
        },
        "timellm": {
            "name": "TimeLLM",
            "description": "Трансформер с патчингом на базе современных SLM 2024-2025",
            "features": ["Patch encoding", "Скрытые представления", "Доверительные интервалы"],
            "available_slm": {
                "qwen2-0.5b": "🟢 Qwen2-0.5B (500M) - Рекомендуется! Топ SLM 2024, самая быстрая, 2GB VRAM",
                "llama3.2-1b": "🟢 Llama-3.2-1B (1B) - Meta SLM 2024, быстрая, 3GB VRAM",
                "gemma-2b": "🟢 Gemma-2B (2B) - Google SLM 2024, баланс скорость/качество, 4GB VRAM",
                "phi3-mini": "🟡 Phi-3-mini (3.8B) - Лучшая точность SLM 2024, 6GB VRAM",
                "stablelm-zephyr-3b": "🟡 StableLM-Zephyr-3B (3B) - Стабильная SLM, 5GB VRAM",
                "gpt2": "🟡 GPT-2 (124M) - Классика 2019, очень быстро, 1GB VRAM",
                "distilgpt2": "🟡 DistilGPT-2 (82M) - Ещё легче GPT-2, <1GB VRAM"
            },
            "default_slm": "qwen2-0.5b",
            "note": "Используйте параметр 'llm_model' в запросе для выбора конкретной SLM"
        },
        "hybrid": {
            "name": "Гибридная модель",
            "description": "Комбинация всех моделей с адаптивным взвешиванием",
            "features": ["Адаптивные веса", "Экспоненциальное сглаживание", "Доверительные интервалы"]
        }
    }


if __name__ == "__main__":


    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
