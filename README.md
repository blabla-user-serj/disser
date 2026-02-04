# 🚀 Гибридная Система Прогнозирования Временных Рядов

**Time-LLM (GGUF) + SARIMA-XS + XGBoost + YandexGPT Expert**

> ⚠️ **ВАЖНО:** Для работы с YandexGPT необходимо настроить API ключи в `config/.env`  
> См. раздел [Конфигурация](#-конфигурация) для подробных инструкций

## 📋 Содержание

- [Описание](#-описание)
- [Архитектура](#-архитектура)
- [Модели](#-модели)
- [Установка](#-установка)
- [Быстрый Старт](#-быстрый-старт)
- [Конфигурация](#-конфигурация)
- [API](#-api)
- [Использование](#-использование)
- [Математические Формулы](#-математические-формулы)

---

## 📖 Описание

Полнофункциональная система для прогнозирования временных рядов, объединяющая:

1. **Time-LLM с GGUF** - глубокое обучение с локальными LLM (Llama, Mistral, и т.д.)
2. **SARIMA-XS** - статистическая модель с Grid Search + Cross-Validation
3. **XGBoost** - gradient boosting с временными признаками
4. **YandexGPT Expert** - LLM-эксперт для анализа и коррекции прогноза
5. **Гибридная модель** - адаптивное взвешивание с экспоненциальным сглаживанием

### ✨ Ключевые Особенности

- ✅ **Локальные LLM** через GGUF (без интернета, полная приватность)
- ✅ **YandexGPT интеграция** для коррекции с учётом внешних факторов
- ✅ **Адаптивные веса** моделей на основе ошибок
- ✅ **Cross-Validation** в SARIMA и XGBoost
- ✅ **Доверительные интервалы** для всех моделей
- ✅ **Веб-интерфейс** на FastAPI + HTML/CSS/JS
- ✅ **Production-ready** код

---

## 🏗️ Архитектура

```
                    Input Time Series
                            |
            +---------------+---------------+
            |               |               |
        Time-LLM        SARIMA-XS       XGBoost
      (GGUF Local)    (Grid Search)  (Features)
            |               |               |
      Forecast_1       Forecast_2      Forecast_3
            |               |               |
            +-----> Hybrid Model <----------+
                 (Adaptive Weighting)
                  w_i = exp(-β*ER_i*α) / Σ
                            |
                   Combined Forecast
                            |
                    YandexGPT Expert
                  (Correction + Analysis)
                            |
                   Final Forecast + CI
```

---

## 🧩 Модели

### 1️⃣ Time-LLM с GGUF

**Локальные LLM модели через llama-cpp-python**

#### Поддерживаемые Модели

**Для API режима (рекомендуется):**
- **GPT-2** (124M) - ✅ РЕКОМЕНДУЕТСЯ: быстро, стабильно, мало памяти
- **DistilGPT-2** (82M) - ✅ Ещё легче, очень быстро

**Для offline обучения (требуют много времени):**
- **TinyLlama** (1.1B) - ⚠️ Средняя модель, 4-6GB VRAM
- **Phi-1.5** (1.3B) - ⚠️ Тяжёлая, 6-8GB VRAM, медленно
- **Phi-2** (2.7B) - ❌ Очень тяжёлая, НЕ для API!

#### Квантизация

- `Q4_K_M` - рекомендуется (баланс скорость/качество)
- `Q5_K_M` - выше качество
- `Q8_0` - максимальное качество

#### Конфигурация

```python
TimeLLM(
    llm_backend='gguf',
    llm_path='/path/to/model.gguf',
    gguf_config={
        'n_ctx': 2048,        # Размер контекста
        'n_threads': 8,       # CPU threads
        'n_gpu_layers': 0,    # GPU layers (0 для CPU)
        'temperature': 0.3,   # Температура
        'max_tokens': 512     # Максимум токенов
    }
)
```

#### Как Работает

1. **Генерация промпта** с статистиками временного ряда
2. **Вызов GGUF модели** для получения insight
3. **Анализ insight** (bullish/bearish keywords)
4. **Коррекция прогноза** на основе LLM insight
5. **Статистический базовый прогноз** + LLM коррекция

---

### 2️⃣ SARIMA-XS

**SARIMA с адаптивными ограничениями и Grid Search + CV**

#### Формула

```
SARIMA(p,d,q)(P,D,Q,s):
Φ(B^s) * φ(B) * ∇^d ∇_s^D y_t = Θ(B^s) * θ(B) * ε_t
```

#### Адаптивные Ограничения

```python
def _adaptive_constraints(n):
    """Ограничения на основе размера выборки"""
    p_max = min(3, n - 3)
    d_max = min(2, n / 4)
    q_max = min(3, n - 3)
    
    P_max = min(2, n // 20)
    D_max = 1 if n >= 24 else 0
    Q_max = min(2, n // 20)
    
    return {'p': p_max, 'd': d_max, 'q': q_max,
            'P': P_max, 'D': D_max, 'Q': Q_max}
```

#### Grid Search с TimeSeriesSplit

- **n_splits=3** (по умолчанию)
- Метрика: MAE на валидации
- Автоопределение сезонности через ACF

---

### 3️⃣ XGBoost для Временных Рядов

**Gradient Boosting с автоматическими временными признаками**

#### Признаки

1. **Лаги**: y_{t-1}, y_{t-2}, ..., y_{t-14}
2. **Скользящие окна**: mean, std, min, max (окна 3, 7, 14)
3. **Сезонные**: sin/cos преобразования для периодов
4. **Временные**: hour, day, day_of_week, month, year

#### Objective Function

```
L(φ) = Σᵢ(ŷᵢ - yᵢ)² + Σₖ[γTₖ + (λ/2)||wₖ||²]
```

#### Cross-Validation

- TimeSeriesSplit с n_splits
- Оптимизация гиперпараметров

---

### 4️⃣ Гибридная Модель (из Диссертации)

**Адаптивное взвешивание с экспоненциальным сглаживанием**

#### Формула Весов

```
w_i = exp(-β * ER_i * α) / Σⱼ exp(-β * ER_j * α)
```

**Где:**
- `ER_i` - экспоненциально сглаженная история ошибок модели i
- `α` - correction factor = `1 + log(30/n)` для малых выборок (n < 30)
- `β` - параметр чувствительности (1.0)

#### Обновление Истории Ошибок

```
ER_i(t) = λ * ER_i(t-1) + (1-λ) * error_i(t)
```

где `λ = 0.9` (decay_factor)

#### Комбинированный Прогноз

```
Ŷ = w_sarima * Ŷ_sarima + w_xgboost * Ŷ_xgboost + w_timellm * Ŷ_timellm
```

---

### 5️⃣ YandexGPT Expert

**LLM-эксперт для анализа и КОРРЕКЦИИ прогноза**

#### Функционал

1. **Анализ временного ряда**
   - Статистики (mean, std, trend)
   - Паттерны и аномалии

2. **Извлечение веб-контекста**
   - Парсинг URL с новостями/событиями
   - Контекст для коррекции

3. **Коррекция прогноза**
   - Применение correction_factors
   - Учёт внешних факторов

#### API Вызов

```python
llm_expert = LLMExpert()

result = llm_expert.correct_forecast(
    historical_data=data,
    forecast=forecast,
    lower_bound=lower,
    upper_bound=upper,
    web_urls=['https://news.example.com/article']
)

# result = {
#     'corrected_forecast': ...,
#     'corrected_lower': ...,
#     'corrected_upper': ...,
#     'analysis': "Текстовый анализ",
#     'correction_applied': True/False
# }
```

#### Prompt Structure

```
ИСТОРИЧЕСКИЕ ДАННЫЕ: [статистики]
ПРОГНОЗ МОДЕЛИ: [прогноз + CI]
ВНЕШНИЙ КОНТЕКСТ: [веб-контекст]

ЗАДАЧА:
1. Анализ паттернов
2. Учёт внешних факторов
3. КОРРЕКЦИЯ прогноза
4. Коэффициенты коррекции

ОТВЕТ (JSON):
{
  "analysis": "...",
  "correction_factors": [1.05, 1.03, ...],
  "reasoning": "..."
}
```

---

## 💾 Установка

### Шаг 1: Клонирование

```bash
tar -xzf hybrid_forecast_complete.tar.gz
cd hybrid_forecast_complete
```

### Шаг 2: Виртуальное Окружение

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows
```

### Шаг 3: Установка Зависимостей

```bash
pip install -r requirements.txt
```

**Примечание:** Если нужен NeuralForecast режим для TimeLLM:
```bash
pip install neuralforecast
```

### Шаг 4: Скачивание GGUF Модели

Рекомендуемые модели:

**Llama 3 8B Q4_K_M:**
```bash
# Скачать с HuggingFace
wget https://huggingface.co/TheBloke/Llama-3-8B-GGUF/resolve/main/llama-3-8b.Q4_K_M.gguf
mv llama-3-8b.Q4_K_M.gguf data/models/
```

**Mistral 7B Q4_K_M:**
```bash
wget https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q4_K_M.gguf
mv mistral-7b-v0.1.Q4_K_M.gguf data/models/
```

**Qwen 7B Q4_K_M:**
```bash
wget https://huggingface.co/Qwen/Qwen-7B-Chat-GGUF/resolve/main/qwen-7b-chat.Q4_K_M.gguf
mv qwen-7b-chat.Q4_K_M.gguf data/models/
```

### Шаг 5: Конфигурация

```bash
cp config/.env.example config/.env
nano config/.env  # Отредактируйте конфигурацию
```

**config/.env:**
```bash
# YandexGPT (опционально)
YANDEX_API_KEY=your_api_key
YANDEX_FOLDER_ID=your_folder_id
YANDEX_MODEL=yandexgpt-lite

# TimeLLM GGUF
TIMELLM_GGUF_PATH=/path/to/hybrid_forecast_complete/data/models/llama-3-8b.Q4_K_M.gguf
TIMELLM_N_CTX=2048
TIMELLM_N_THREADS=8
TIMELLM_N_GPU_LAYERS=0
TIMELLM_TEMPERATURE=0.3

# Server
HOST=0.0.0.0
PORT=8000
```

---

## 🚀 Быстрый Старт

### Вариант 1: Веб-Интерфейс

```bash
# Запуск сервера
cd backend
python main.py
```

Откройте в браузере: `http://localhost:8000`

### Вариант 2: Python API

```python
import numpy as np
from models import HybridModel

# Загрузка данных
data = np.loadtxt('your_data.csv')

# Создание модели
model = HybridModel(
    decay_factor=0.9,
    use_cv=True,
    n_splits=3
)

# Обучение
model.fit(data)

# Прогноз
result = model.predict(
    steps=24,
    return_conf_int=True,
    alpha=0.05
)

print(f"Forecast: {result['forecast']}")
print(f"Weights: {result['weights']}")
print(f"Lower: {result['lower_bound']}")
print(f"Upper: {result['upper_bound']}")
```

### Вариант 3: С YandexGPT Коррекцией

```python
from backend.llm_expert import LLMExpert

# ... обучение модели как выше ...

# Прогноз без коррекции
result = model.predict(steps=24, return_conf_int=True)

# LLM коррекция
llm_expert = LLMExpert()
corrected = llm_expert.correct_forecast(
    historical_data=data,
    forecast=result['forecast'],
    lower_bound=result['lower_bound'],
    upper_bound=result['upper_bound'],
    web_urls=['https://news.example.com/relevant-article']
)

print(f"Corrected Forecast: {corrected['corrected_forecast']}")
print(f"Analysis: {corrected['analysis']}")
```

---

## ⚙️ Конфигурация

### TimeLLM GGUF Параметры

```python
gguf_config = {
    'n_ctx': 2048,        # Размер контекста (больше = больше памяти)
    'n_threads': 8,       # CPU threads (по количеству ядер)
    'n_gpu_layers': 0,    # GPU layers (>0 если есть GPU)
    'temperature': 0.3,   # 0.0-1.0 (ниже = более детерминированно)
    'max_tokens': 512     # Максимум токенов в ответе
}
```

**Для GPU:**
```python
gguf_config = {
    'n_ctx': 4096,
    'n_threads': 4,
    'n_gpu_layers': 35,  # Зависит от VRAM
    'temperature': 0.3,
    'max_tokens': 1024
}
```

### Гибридная Модель Параметры

```python
HybridModel(
    decay_factor=0.9,  # λ для exp. сглаживания (0.7-0.95)
    use_cv=True,       # Использовать CV в базовых моделях
    n_splits=3         # Количество сплитов для CV
)
```

### SARIMA-XS Параметры

```python
SARIMAXS(
    use_cv=True,     # Grid Search с TimeSeriesSplit
    n_splits=3       # Количество сплитов
)
```

### XGBoost Параметры

```python
XGBoostTS(
    use_cv=True,
    n_splits=3,
    xgb_params={
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1,
        'subsample': 0.8
    }
)
```

---

## 📡 API

### FastAPI Endpoints

#### 1. Upload & Forecast

**POST** `/forecast`

```bash
curl -X POST "http://localhost:8000/forecast" \
  -F "file=@data.csv" \
  -F "steps=24" \
  -F "use_llm_expert=true" \
  -F "web_urls=https://news.example.com/article"
```

**Response:**
```json
{
  "forecast": [...],
  "lower_bound": [...],
  "upper_bound": [...],
  "weights": {
    "sarima": 0.35,
    "xgboost": 0.32,
    "timellm": 0.33
  },
  "individual_forecasts": {
    "sarima": [...],
    "xgboost": [...],
    "timellm": [...]
  },
  "llm_analysis": "Анализ от YandexGPT...",
  "correction_applied": true
}
```

#### 2. Model Info

**GET** `/model-info`

```bash
curl "http://localhost:8000/model-info"
```

#### 3. Export Results

**GET** `/export/{format}`

```bash
curl "http://localhost:8000/export/csv" -o results.csv
curl "http://localhost:8000/export/json" -o results.json
```

---

## 📐 Математические Формулы

### Гибридная Модель

**Веса:**
```
w_i = exp(-β * ER_i * α) / Σⱼ exp(-β * ER_j * α)
```

**Error History:**
```
ER_i(t) = λ * ER_i(t-1) + (1-λ) * MAE_i(t)
```

**Correction Factor:**
```
α = 1 + log(30/n)  if n < 30 else 1.0
```

**Combined Forecast:**
```
Ŷ = Σᵢ w_i * Ŷᵢ
```

### SARIMA-XS

```
Φ(Bˢ) φ(B) ∇ᵈ ∇ₛᴰ yₜ = Θ(Bˢ) θ(B) εₜ
```

### XGBoost

```
L(φ) = Σᵢ(ŷᵢ - yᵢ)² + γT + (λ/2)Σⱼwⱼ²
```

### Доверительные Интервалы

```
CI = ŷ ± z_{α/2} * σ_residual
```

где `z_{0.025} = 1.96` для 95% CI

---

## 🎯 Использование

### Пример 1: Базовый прогноз

```python
from models import HybridModel
import numpy as np

# Данные
data = np.random.randn(200)

# Модель
model = HybridModel()
model.fit(data)

# Прогноз
result = model.predict(steps=10)
print(result['forecast'])
```

### Пример 2: С GGUF моделью

```python
from models import TimeLLM

# TimeLLM с GGUF
timellm = TimeLLM(
    llm_backend='gguf',
    llm_path='data/models/llama-3-8b.Q4_K_M.gguf',
    gguf_config={
        'n_ctx': 2048,
        'n_threads': 8,
        'n_gpu_layers': 0
    }
)

timellm.fit(data)
result = timellm.predict(steps=10)
```

### Пример 3: Только SARIMA-XS

```python
from models import SARIMAXS

sarima = SARIMAXS(use_cv=True, n_splits=3)
sarima.fit(data)

result = sarima.predict(steps=10, return_conf_int=True)
print(f"Forecast: {result['forecast']}")
print(f"CI: [{result['lower_bound']}, {result['upper_bound']}]")
```

### Пример 4: Только XGBoost

```python
from models import XGBoostTS

xgb = XGBoostTS(use_cv=True, n_splits=3)
xgb.fit(data)

result = xgb.predict(steps=10, return_conf_int=True)
```

---

## 📊 Веб-Интерфейс

### Функции

1. **Загрузка CSV** - drag & drop или выбор файла
2. **Настройки прогноза** - steps, α для CI
3. **Выбор моделей** - включить/отключить модели
4. **YandexGPT коррекция** - с URL для контекста
5. **Визуализация** - интерактивные графики Plotly
6. **Экспорт** - CSV, JSON, PNG

### Скриншоты

(В разработке - будут добавлены)

---

## 🧪 Тестирование

```bash
# Запуск тестов
pytest tests/ -v

# С покрытием
pytest tests/ --cov=models --cov-report=html
```

---

## 📚 Документация

- `README.md` - Этот файл
- `docs/API.md` - Полная API документация
- `docs/MODELS.md` - Детальное описание моделей
- `docs/FORMULAS.md` - Все математические формулы
- `docs/EXAMPLES.md` - Примеры использования

---

## 🐛 Troubleshooting

### ❌ Ошибка "failed to fetch" при обучении модели

**Причина:** Проблемы с YandexGPT API или долгое обучение модели

**Решение:**
1. **Проверьте настройки YandexGPT:**
   ```bash
   # Скопируйте пример конфигурации
   cp config/.env.example config/.env
   
   # Отредактируйте файл и укажите ваши ключи
   nano config/.env
   ```

2. **Убедитесь что указаны правильные значения:**
   - `YANDEX_API_KEY` - API ключ из консоли Yandex Cloud
   - `YANDEX_FOLDER_ID` - ID каталога (folder ID)
   - `YANDEX_MODEL` - yandexgpt-lite (рекомендуется)

3. **Проверьте права доступа:**
   - В Yandex Cloud назначьте роль `ai.languageModels.user` на сервисный аккаунт
   - Убедитесь что API ключ не истёк

4. **Если YandexGPT недоступен:**
   - Система автоматически переключится на базовый анализ без LLM
   - Прогноз будет работать, но без коррекции от YandexGPT

### ❌ YandexGPT не работает

```
Warning: YandexGPT API key not found
```

**Решение:**
Добавьте в `config/.env`:
```
YANDEX_API_KEY=your_key
YANDEX_FOLDER_ID=your_folder_id
```

**Получение ключей:**
1. Перейдите в [консоль Yandex Cloud](https://console.cloud.yandex.ru/)
2. Создайте сервисный аккаунт с ролью `ai.languageModels.user`
3. Создайте API-ключ в разделе "API-ключи"
4. Скопируйте ID каталога из URL консоли

### GGUF модель не загружается

```
Error: llama-cpp-python not found
```

**Решение:**
```bash
pip install llama-cpp-python
```

Для GPU (CUDA):
```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### YandexGPT не работает

```
Warning: YandexGPT API key not found
```

**Решение:**
Добавьте в `config/.env`:
```
YANDEX_API_KEY=your_key
YANDEX_FOLDER_ID=your_folder_id
```

### Недостаточно памяти для GGUF

```
Error: Out of memory
```

**Решение:**
1. Используйте меньшую модель (Q4 вместо Q8)
2. Уменьшите `n_ctx` в конфигурации
3. Используйте квантизованную версию

---

## 🎉 Заключение

Полная система для прогнозирования временных рядов с:

✅ **Локальными LLM** (GGUF)  
✅ **YandexGPT** коррекцией  
✅ **3 мощными моделями** (SARIMA-XS, XGBoost, Time-LLM)  
✅ **Гибридным ансамблем** с адаптивными весами  
✅ **Веб-интерфейсом**  
✅ **Production-ready**  

**Готово к использованию! 🚀**

---

**Версия:** 3.0.0 COMPLETE  
**Дата:** 2024-11-16  
**Статус:** ✅ Production Ready
