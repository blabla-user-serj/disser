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
import math
from io import BytesIO, StringIO
from datetime import datetime, timedelta
import json
import re
from typing import List

# Библиотеки для парсинга документов
try:
    import docx
    import fitz  # PyMuPDF
except ImportError:
    pass


def sanitize_for_json(value):
    """
    Очистка значения для JSON сериализации.
    Заменяет NaN, Inf, -Inf на None или 0.
    """
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return 0.0
        return value
    elif isinstance(value, (np.floating, np.float64, np.float32)):
        if np.isnan(value) or np.isinf(value):
            return 0.0
        return float(value)
    elif isinstance(value, np.ndarray):
        return sanitize_array(value).tolist()
    elif isinstance(value, list):
        return [sanitize_for_json(v) for v in value]
    elif isinstance(value, dict):
        return {k: sanitize_for_json(v) for k, v in value.items()}
    return value


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
        valid_values = arr[valid_mask]
        computed_fallback = np.mean(valid_values)
    else:
        computed_fallback = fallback_value
    
    # Заменяем NaN и Inf на fallback
    arr = np.where(np.isnan(arr), computed_fallback, arr)
    arr = np.where(np.isinf(arr), computed_fallback, arr)
    
    return arr

from models import SARIMAXS, XGBoostTS, TimeLLM, HybridModel
from models.timellm_gguf import destroy_all_models, clear_gpu_memory_completely
from backend.llm_expert import LLMExpert

# Загрузка переменных окружения из .env
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'config', '.env'))

torch.cuda.empty_cache()
gc.collect()
torch.set_float32_matmul_precision('high')
torch.set_grad_enabled(False)

app = FastAPI(title="Hybrid Forecast API", version="2.0")

# Тестовый роутер (отдельный файл, не изменяет основную логику)
from backend.test_routes import router as test_router
app.include_router(test_router)

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


# Словарь для русских названий месяцев
RUSSIAN_MONTHS = {
    'январь': 1, 'января': 1, 'янв': 1,
    'февраль': 2, 'февраля': 2, 'фев': 2,
    'март': 3, 'марта': 3, 'мар': 3,
    'апрель': 4, 'апреля': 4, 'апр': 4,
    'май': 5, 'мая': 5,
    'июнь': 6, 'июня': 6, 'июн': 6,
    'июль': 7, 'июля': 7, 'июл': 7,
    'август': 8, 'августа': 8, 'авг': 8,
    'сентябрь': 9, 'сентября': 9, 'сен': 9,
    'октябрь': 10, 'октября': 10, 'окт': 10,
    'ноябрь': 11, 'ноября': 11, 'ноя': 11,
    'декабрь': 12, 'декабря': 12, 'дек': 12
}


def parse_russian_date(date_str):
    """
    Парсинг даты с русскими названиями месяцев.
    Поддерживаемые форматы:
    - "январь 2023", "янв 2023"
    - "01 января 2023", "1 янв 2023"
    - "2023 январь", "2023 янв"
    """
    if pd.isna(date_str):
        return pd.NaT
    
    date_str = str(date_str).lower().strip()
    
    # Ищем русский месяц
    for month_name, month_num in RUSSIAN_MONTHS.items():
        if month_name in date_str:
            # Извлекаем год (4 цифры)
            import re
            year_match = re.search(r'\b(19\d{2}|20\d{2})\b', date_str)
            if year_match:
                year = int(year_match.group(1))
                
                # Извлекаем день (1-2 цифры, не год)
                day_match = re.search(r'\b([1-9]|[12]\d|3[01])\b(?!\d)', date_str)
                day = int(day_match.group(1)) if day_match else 1
                
                try:
                    return pd.Timestamp(year=year, month=month_num, day=day)
                except:
                    return pd.NaT
    
    return None  # Не русская дата


def parse_excel_serial_date(value):
    """
    Парсинг Excel serial date (числовые даты).
    Excel serial date: количество дней с 30.12.1899
    """
    if pd.isna(value):
        return pd.NaT
    
    try:
        # Если это число в диапазоне Excel дат (примерно 1900-2100)
        num_value = float(value)
        if 1 <= num_value <= 100000:  # Диапазон Excel дат
            # Excel serial date конвертация
            return pd.Timestamp('1899-12-30') + pd.Timedelta(days=num_value)
    except (ValueError, TypeError):
        pass
    
    return None


def smart_parse_date(value):
    """
    Умный парсинг даты с поддержкой различных форматов.
    Приоритет:
    1. Русские даты (январь 2023)
    2. Excel serial dates (44927)
    3. Стандартные форматы pandas
    """
    if pd.isna(value):
        return pd.NaT
    
    # Попробуем русскую дату
    result = parse_russian_date(value)
    if result is not None:
        return result
    
    # Попробуем Excel serial date
    result = parse_excel_serial_date(value)
    if result is not None:
        return result
    
    # Стандартный pandas парсинг
    try:
        return pd.to_datetime(value)
    except:
        return pd.NaT


def smart_parse_numeric(value):
    """
    Умный парсинг числовых значений.
    Поддержка:
    - Запятая как десятичный разделитель (10,5 -> 10.5)
    - Пробелы как разделители тысяч (1 000 -> 1000)
    - Проценты (20% -> 20)
    """
    if pd.isna(value):
        return np.nan
    
    # Уже число
    if isinstance(value, (int, float)):
        return float(value)
    
    # Строка
    str_value = str(value).strip()
    
    # Удаляем проценты
    str_value = str_value.replace('%', '')
    
    # Удаляем пробелы (разделители тысяч)
    str_value = str_value.replace(' ', '').replace('\u00a0', '')  # включая неразрывный пробел
    
    # Заменяем запятую на точку (русская локаль)
    str_value = str_value.replace(',', '.')
    
    try:
        return float(str_value)
    except ValueError:
        return np.nan


def find_date_column(df):
    """
    Автоматическое определение колонки с датами.
    Пробует разные колонки и проверяет успешность парсинга.
    """
    date_keywords = ['дата', 'date', 'время', 'time', 'period', 'период', 'год', 'year', 'месяц', 'month']
    
    # Сначала ищем по ключевым словам в названии
    for col in df.columns:
        col_lower = str(col).lower()
        for keyword in date_keywords:
            if keyword in col_lower:
                # Проверяем, что хотя бы одно значение парсится
                test_value = smart_parse_date(df[col].iloc[0])
                if not pd.isna(test_value):
                    return col
    
    # Если не нашли по названию, пробуем каждую колонку
    for col in df.columns:
        # Берём первые 3 непустых значения
        sample = df[col].dropna().head(3)
        if len(sample) == 0:
            continue
        
        parsed_count = 0
        for val in sample:
            result = smart_parse_date(val)
            if not pd.isna(result):
                parsed_count += 1
        
        # Если хотя бы 2 из 3 распарсились - это колонка с датами
        if parsed_count >= 2:
            return col
    
    # Fallback: первая колонка
    return df.columns[0]


def find_value_column(df, date_col):
    """
    Автоматическое определение колонки со значениями.
    """
    value_keywords = ['значение', 'value', 'сумма', 'amount', 'показатель', 'индекс', 'index', 'rate', 'ставка', 'цена', 'price']
    
    # Сначала ищем по ключевым словам
    for col in df.columns:
        if col == date_col:
            continue
        col_lower = str(col).lower()
        for keyword in value_keywords:
            if keyword in col_lower:
                return col
    
    # Fallback: первая числовая колонка, не являющаяся датой
    for col in df.columns:
        if col == date_col:
            continue
        
        # Пробуем распарсить как числа
        sample = df[col].dropna().head(5)
        if len(sample) == 0:
            continue
        
        parsed_count = 0
        for val in sample:
            result = smart_parse_numeric(val)
            if not np.isnan(result):
                parsed_count += 1
        
        if parsed_count >= 3:
            return col
    
    # Fallback: вторая колонка
    return df.columns[1] if len(df.columns) > 1 else df.columns[0]


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
                date_col = find_date_column(df)
                value_col = find_value_column(df, date_col)
                
                if date_col and value_col:
                    return df, date_col, value_col
        except:
            continue
    
    # Если не получилось автоопределить, берём первые две колонки
    df = pd.read_csv(StringIO(content))
    return df, df.columns[0], df.columns[1]


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Загрузка CSV/XLSX файла с улучшенным парсингом.
    
    Поддержка:
    - Русские названия месяцев (январь 2023, янв 2023)
    - Excel serial dates (44927 -> 2023-01-01)
    - Числа с запятой (10,5 -> 10.5)
    - Различные кодировки (UTF-8, CP1251, Latin1)
    - Автоопределение колонок с датами и значениями
    """
    try:
        content = await file.read()
        filename = file.filename.lower() if file.filename else ""
        
        # Валидация расширения файла
        if not (filename.endswith('.csv') or filename.endswith('.xlsx') or filename.endswith('.xls')):
            raise HTTPException(400, "Поддерживаются только CSV и XLSX файлы")
        
        # Парсинг файла
        if filename.endswith('.csv'):
            df, date_col, value_col = parse_csv(content)
        else:
            # XLSX/XLS
            try:
                df = pd.read_excel(BytesIO(content))
            except Exception as e:
                raise HTTPException(400, f"Ошибка чтения Excel файла: {str(e)}. Убедитесь, что файл не повреждён.")
            
            if len(df.columns) < 2:
                raise HTTPException(400, "Файл должен содержать минимум 2 колонки (дата и значение)")
            
            date_col = find_date_column(df)
            value_col = find_value_column(df, date_col)
        
        # Сохраняем исходное количество строк для отчёта
        original_rows = len(df)
        
        # Умный парсинг дат
        df['_parsed_date'] = df[date_col].apply(smart_parse_date)
        
        # Проверяем успешность парсинга дат
        date_success_count = df['_parsed_date'].notna().sum()
        if date_success_count == 0:
            # Детальная диагностика
            sample_values = df[date_col].head(3).tolist()
            raise HTTPException(
                400, 
                f"Не удалось распознать даты в колонке '{date_col}'. "
                f"Примеры значений: {sample_values}. "
                f"Поддерживаемые форматы: 2023-01-15, 15.01.2023, январь 2023, Jan 2023"
            )
        
        # Умный парсинг числовых значений
        df['_parsed_value'] = df[value_col].apply(smart_parse_numeric)
        
        # Проверяем успешность парсинга значений
        value_success_count = df['_parsed_value'].notna().sum()
        if value_success_count == 0:
            sample_values = df[value_col].head(3).tolist()
            raise HTTPException(
                400,
                f"Не удалось распознать числовые значения в колонке '{value_col}'. "
                f"Примеры значений: {sample_values}. "
                f"Поддерживаемые форматы: 10.5, 10,5, 10 000, 20%"
            )
        
        # Используем распарсенные данные
        df[date_col] = df['_parsed_date']
        df[value_col] = df['_parsed_value']
        
        # Удаление строк с NaN (после парсинга)
        df = df.dropna(subset=[date_col, value_col])
        
        # Проверка на пустой результат
        if len(df) == 0:
            raise HTTPException(
                400,
                f"После обработки не осталось валидных данных. "
                f"Исходных строк: {original_rows}. "
                f"Успешно распознано дат: {date_success_count}, значений: {value_success_count}. "
                f"Проверьте формат данных в файле."
            )
        
        # Сортировка по дате
        df = df.sort_values(by=date_col)
        
        # Определение частоты
        frequency = infer_frequency(df[date_col])
        
        # Формируем предупреждения если часть данных была отброшена
        warnings = []
        skipped_rows = original_rows - len(df)
        if skipped_rows > 0:
            warnings.append(f"Пропущено строк с некорректными данными: {skipped_rows}")
        
        if date_success_count < original_rows:
            warnings.append(f"Не распознано дат: {original_rows - date_success_count}")
        
        if value_success_count < original_rows:
            warnings.append(f"Не распознано значений: {original_rows - value_success_count}")
        
        result = {
            "status": "success",
            "rows": len(df),
            "original_rows": original_rows,
            "date_column": date_col,
            "value_column": value_col,
            "frequency": frequency,
            "data": {
                "dates": df[date_col].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                "values": df[value_col].tolist()
            }
        }
        
        if warnings:
            result["warnings"] = warnings
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Ошибка обработки файла: {str(e)}")


# Методы извлечения текста теперь находятся в LLMExpert.extract_text_from_file

@app.post("/forecast")
async def forecast(
    dates: str = Form(...),
    values: str = Form(...),
    model_type: str = Form(...),
    steps: int = Form(...),
    web_urls: str = Form(None),
    dataset_description: str = Form(None),
    llm_model: str = Form('qwen2-0.5b'),
    doc_files: List[UploadFile] = File(None)
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
        
        # КРИТИЧНО: Полная очистка GPU памяти перед обучением моделей
        # Это необходимо, т.к. между запросами FastAPI память может накапливаться
        print("\n" + "="*60)
        print("🧹 ГЛОБАЛЬНАЯ ОЧИСТКА ПЕРЕД НОВЫМ ЗАПРОСОМ")
        print("="*60)
        
        # ШАГИ ОЧИСТКИ:
        # 1. Уничтожаем ВСЕ зарегистрированные модели
        destroy_all_models()
        
        if torch.cuda.is_available():
            # Показываем состояние ДО дополнительной очистки
            allocated_before = torch.cuda.memory_allocated(0) / 1024**3
            reserved_before = torch.cuda.memory_reserved(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            print(f"📊 Состояние GPU памяти:")
            print(f"   Выделено: {allocated_before:.2f} GB")
            print(f"   Зарезервировано: {reserved_before:.2f} GB")
            print(f"   Всего: {total:.2f} GB")
            print(f"   Свободно: {total - reserved_before:.2f} GB")
            
            # Предупреждение если память всё ещё занята
            if allocated_before > 1.0:
                print(f"\n⚠️  ВНИМАНИЕ: {allocated_before:.2f}GB всё ещё выделено!")
                print(f"   💡 Рекомендация: перезапустите сервер для полной очистки.")
                print(f"   Модель может автоматически переключиться на Simple режим.")
        
        print("="*60 + "\n")
        
        # Обучение модели
        if model_type == 'sarima':
            model = SARIMAXS()
        elif model_type == 'xgboost':
            model = XGBoostTS()
        elif model_type == 'timellm':
            # Параметр llm_model можно передать для выбора конкретной SLM
            print(f"🤖 TimeLLM: выбрана SLM модель '{llm_model}'")
            
            # По умолчанию используем simple режим (быстро, без GPU)
            # Для NeuralForecast нужно явно указать llm_backend='neuralforecast'
            backend = 'simple'  # Безопасный режим по умолчанию
            
            # Если пользователь явно выбрал NeuralForecast модель - используем neuralforecast
            if llm_model in ['smollm2-135m', 'smollm2-360m', 'smollm2-1.7b',
                             'qwen2-0.5b', 'qwen2.5-0.5b', 'llama3.2-1b',
                             'gemma-2b', 'phi3-mini', 'stablelm-zephyr-3b',
                             'tinyllama', 'phi-1.5', 'gpt2', 'distilgpt2',
                             'qwen3-1.7b']:
                if torch.cuda.is_available():
                    backend = 'neuralforecast'
                    print(f"✅ Используется NeuralForecast с {llm_model}")
                else:
                    print(f"⚠️  GPU недоступен, используется simple режим вместо NeuralForecast")
            
            model = TimeLLM(
                llm_backend=backend,
                neuralforecast_model=llm_model
            )
        elif model_type == 'hybrid':
            # Гибридная модель: SARIMA + XGBoost + TimeLLM
            # Используем SLM для TimeLLM если доступен GPU
            use_slm = torch.cuda.is_available()
            
            if use_slm:
                print(f"🤖 HybridModel: будет использовать TimeLLM с NeuralForecast + SLM '{llm_model}'")
                print(f"   GPU доступен, используем современную SLM модель")
            else:
                print(f"⚡ HybridModel: будет использовать TimeLLM в Simple режиме")
                print(f"   GPU недоступен, используем статистический режим")
            
            model = HybridModel(
                use_slm=use_slm,
                slm_model=llm_model  # Передаём выбранную пользователем SLM
            )
        else:
            raise HTTPException(400, f"Неизвестная модель: {model_type}")
        
        try:
            # Передаём steps в fit для TimeLLM и Hybrid (обучение на нужном горизонте)
            if model_type in ('timellm', 'hybrid'):
                model.fit(values_array, steps=steps)
            else:
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
        
        # Генерация дат прогноза с правильной частотой
        last_date = dates_array[-1]
        forecast_dates = generate_forecast_dates(last_date, steps, frequency)
        
        # LLM-эксперт используется ТОЛЬКО для гибридной модели
        llm_correction = None  # Δ_LLM для Формулы 2.18
        correction_factors = None
        llm_analysis = ""
        correction_applied = False
        
        # LLM-коррекция применяется только для гибридной модели
        if model_type == 'hybrid':
            try:
                llm_expert = LLMExpert()
                
                # Парсинг веб-ссылок
                web_urls_list = []
                if web_urls:
                    web_urls_list = [url.strip() for url in web_urls.split('\n') if url.strip()]
                
                # Обработка загруженных документов
                doc_context = ""
                if doc_files:
                    print(f"\n📂 Обработка {len(doc_files)} загруженных документов...")
                    for doc_file in doc_files:
                        try:
                            print(f"   📄 Чтение файла: {doc_file.filename}")
                            content = await doc_file.read()
                            text = llm_expert.extract_text_from_file(content, doc_file.filename)
                            if text.strip():
                                print(f"   ✅ Успешно извлечено {len(text)} символов из {doc_file.filename}")
                                doc_context += f"\n--- Содержимое файла {doc_file.filename} ---\n"
                                doc_context += text
                            else:
                                print(f"   ⚠️ Файл {doc_file.filename} пуст или текст не извлечен")
                        except Exception as doc_err:
                            print(f"   ❌ Ошибка при чтении документа {doc_file.filename}: {doc_err}")
                
                # Объединяем описание и контекст из документов
                full_extra_context = dataset_description if dataset_description else ""
                if doc_context:
                    full_extra_context += "\n\n[ДОПОЛНИТЕЛЬНЫЙ КОНТЕКСТ ИЗ ДОКУМЕНТОВ]:\n" + doc_context
                
                # Сначала делаем предварительный прогноз для LLM-анализа
                preliminary_result = model.predict(steps, return_conf_int=True, alpha=0.05)
                preliminary_forecast = preliminary_result['forecast']
                preliminary_lower = preliminary_result.get('lower_bound', preliminary_forecast * 0.95)
                preliminary_upper = preliminary_result.get('upper_bound', preliminary_forecast * 1.05)
                
                # LLM анализирует предварительный прогноз
                correction_result = llm_expert.correct_forecast(
                    historical_data=values_array,
                    forecast=preliminary_forecast,
                    lower_bound=preliminary_lower,
                    upper_bound=preliminary_upper,
                    web_urls=web_urls_list,
                    extra_context=full_extra_context
                )
                
                correction_factors = correction_result.get('correction_factors', [1.0] * steps)
                llm_analysis = correction_result['analysis']
                correction_applied = correction_result['correction_applied']
                
                # Вычисляем Δ_LLM = (corrected - original) для Формулы 2.18
                if correction_applied:
                    corrected_forecast_llm = correction_result['corrected_forecast']
                    llm_correction = corrected_forecast_llm - preliminary_forecast
                    print(f"\n🧠 LLM коррекция:")
                    print(f"   Коэффициенты: {correction_factors}")
                    print(f"   Δ_LLM: {llm_correction}")
                
            except Exception as e:
                print(f"\n⚠️  Ошибка при LLM коррекции: {str(e)}")
                llm_analysis = f"⚠️ LLM коррекция недоступна: {str(e)}"
                correction_applied = False
        else:
            # Для базовых моделей (SARIMA, XGBoost, TimeLLM) LLM-коррекция не применяется
            llm_analysis = f"ℹ️ LLM-эксперт не используется для модели {model_type}.\n\n"
            llm_analysis += "💡 Для получения экспертной коррекции прогноза выберите 'Гибридную модель'."
        
        # Прогнозирование с доверительными интервалами
        try:
            # Для гибридной модели передаём Δ_LLM (Формула 2.18)
            if model_type == 'hybrid' and llm_correction is not None:
                result = model.predict(steps, return_conf_int=True, alpha=0.05, llm_correction=llm_correction)
            else:
                result = model.predict(steps, return_conf_int=True, alpha=0.05)
        except Exception as e:
            print(f"\n❌ Ошибка при прогнозировании модели {model_type}:")
            print(f"   {str(e)}")
            import traceback
            traceback.print_exc()
            raise HTTPException(500, f"Ошибка прогнозирования модели {model_type}: {str(e)}")
        

        forecast_values = result['forecast']
        forecast_values = np.round(forecast_values, decimals = 3)
        lower_bound = result.get('lower_bound', forecast_values * 0.95)
        upper_bound = result.get('upper_bound', forecast_values * 1.05)
        
        # Для гибридной модели коррекция уже применена в predict()
        corrected_forecast = forecast_values
        corrected_lower = lower_bound
        corrected_upper = upper_bound
        
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
        print("!"*60)
        print(corrected_forecast)
        # Метрики (на исторических данных)
        # Для HybridModel метрики уже рассчитаны внутри модели при валидации
        # Для остальных моделей делаем честную валидацию
        if len(values_array) > 10:
            train_size = int(0.8 * len(values_array))
            y_true = values_array[train_size:]
            
            if model_type == 'hybrid':
                # HybridModel уже имеет метрики из валидации
                # Используем прогноз модели на валидационной части
                print("📊 Расчёт метрик для Hybrid: используем внутренние веса модели")
                y_pred = model.predict(len(y_true), return_conf_int=False)['forecast']
                metrics = model.get_metrics(y_true, y_pred[:len(y_true)])
            elif model_type == 'timellm':
                # Для TimeLLM используем уже обученную модель
                # Валидация происходила внутри при обучении
                print("📊 Расчёт метрик для TimeLLM")
                y_pred = model.predict(len(y_true), return_conf_int=False)['forecast']
                if len(y_pred) < len(y_true):
                    y_pred = np.concatenate([y_pred, np.full(len(y_true) - len(y_pred), y_pred[-1])])
                metrics = model.get_metrics(y_true, y_pred[:len(y_true)])
            else:
                # Для лёгких моделей (SARIMA, XGBoost) делаем честную валидацию
                temp_model = model.__class__()
                temp_model.fit(values_array[:train_size])
                y_pred = temp_model.predict(len(y_true), return_conf_int=False)['forecast']
                metrics = model.get_metrics(y_true, y_pred)
                del temp_model  # Освобождаем память
        else:
            metrics = {'MAE': 0, 'RMSE': 0, 'R2': 0}
        
        # Информация о модели
        model_info = model.get_info()
        
        # Веса (для гибридной модели)
        weights = model_info.get('weights', {}) if model_type == 'hybrid' else {}
        
        # Финальная очистка всех числовых значений для JSON
        last_value = float(values_array[-1]) if len(values_array) > 0 else 0.0
        
        corrected_forecast = sanitize_array(corrected_forecast, fallback_value=last_value)
        corrected_lower = sanitize_array(corrected_lower, fallback_value=corrected_forecast[0] * 0.9 if len(corrected_forecast) > 0 else last_value * 0.9)
        corrected_upper = sanitize_array(corrected_upper, fallback_value=corrected_forecast[0] * 1.1 if len(corrected_forecast) > 0 else last_value * 1.1)
        
        # Гарантируем, что lower <= forecast <= upper
        corrected_lower = np.minimum(corrected_lower, corrected_forecast)
        corrected_upper = np.maximum(corrected_upper, corrected_forecast)
        
        # Очищаем метрики
        clean_metrics = {
            "MAE":  sanitize_for_json(float(metrics.get('MAE',  0))),
            "RMSE": sanitize_for_json(float(metrics.get('RMSE', 0))),
            "R2":   sanitize_for_json(float(metrics.get('R2',   0))),
        }
        # MAPE / NMAE если возвращены моделью
        if 'MAPE' in metrics:
            clean_metrics['MAPE'] = sanitize_for_json(float(metrics['MAPE']))
        if 'NMAE' in metrics:
            clean_metrics['NMAE'] = sanitize_for_json(float(metrics['NMAE']))
        
        # Добавляем тип модели в model_info
        model_info['model_type'] = model_type

        # Добавляем компоненты прогноза (ensemble/bias/llm_correction) для Hybrid
        if model_type == 'hybrid' and isinstance(result, dict):
            model_info['forecast_components'] = {
                'ensemble_forecast': sanitize_for_json(result.get('ensemble_forecast', [0.0])),
                'bias_correction':   sanitize_for_json(result.get('bias_correction',   [0.0])),
                'llm_correction':    sanitize_for_json(result.get('llm_correction',    [0.0])),
                'llm_weight':        sanitize_for_json(result.get('llm_weight', 0.0)),
            }

        # Очищаем model_info
        clean_model_info = sanitize_for_json(model_info)
        clean_weights = sanitize_for_json(weights)
        
        # Формируем ответ, совместимый с фронтендом (ожидающим history_dates и history_values)
        return {
            "status": "success",
            "history_dates": dates_array.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            "history_values": sanitize_array(values_array).tolist(),
            "forecast_dates": forecast_dates.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            "forecast_values": corrected_forecast.tolist(),
            "forecast_lower": corrected_lower.tolist(),
            "forecast_upper": corrected_upper.tolist(),
            "frequency": frequency,
            "metrics": clean_metrics,
            "llm_analysis": llm_analysis,
            "correction_applied": correction_applied,
            "weights": clean_weights,
            "model_info": clean_model_info
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
                "smollm2-135m": "⭐ SmolLM2-135M (135M, 2025) - быстрейшая, vocab=49K, <1GB VRAM",
                "smollm2-360m": "⭐ SmolLM2-360M (360M, 2025) - РЕКОМЕНДУЕТСЯ, vocab=49K, 1GB VRAM",
                "smollm2-1.7b": "⭐ SmolLM2-1.7B (1.7B, 2025) - лучшее качество, vocab=49K, 4GB VRAM",
                "gpt2": "🟢 GPT-2 (124M) - надёжный fallback, vocab=50K, 1GB VRAM",
                "distilgpt2": "🟢 DistilGPT-2 (82M) - самая лёгкая, vocab=50K, <1GB VRAM",
                "phi3-mini": "🟡 Phi-3-mini (3.8B, 2024) - vocab=32K, 6GB VRAM, медленная загрузка",
                "tinyllama": "🟡 TinyLlama (1.1B) - vocab=32K, 3GB VRAM",
                "phi-1.5": "🟡 Phi-1.5 (1.3B) - vocab=50K, 3GB VRAM",
                "qwen2-0.5b": "🟡 Qwen2-0.5B (500M) - vocab=152K, автозамена на smollm2 при <10GB",
                "qwen2.5-0.5b": "🟡 Qwen2.5-0.5B (500M, 2024) - vocab=152K, автозамена на smollm2 при <10GB",
                "llama3.2-1b": "🟡 Llama-3.2-1B (1B) - vocab=128K, автозамена на smollm2 при <10GB",
                "gemma-2b": "🔴 Gemma-2B (2B) - vocab=256K, автозамена на smollm2 при <10GB",
                "stablelm-zephyr-3b": "🟡 StableLM-Zephyr-3B (3B) - vocab=50K, 5GB VRAM",
                "qwen3-1.7b": "latest qwen"
            },
            "default_slm": "smollm2-360m",
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
