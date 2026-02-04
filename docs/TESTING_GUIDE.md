# 🧪 Руководство по Тестированию TimeLLM с SLM

## 📋 Цель

Протестировать работу TimeLLM с современными Small Language Models (SLM) 2024-2025 на вашем сервере.

## 🔧 Предварительные требования

### Система
- Linux/Ubuntu (sandbox environment)
- Python 3.8+
- CUDA 11.8+ (опционально, для GPU)
- RTX 4070 Ti Super 16GB (или аналогичный GPU)

### Зависимости

```bash
pip install numpy pandas scipy statsmodels xgboost
pip install fastapi uvicorn python-multipart python-dotenv requests scikit-learn

# Для TimeLLM с SLM (требуется PyTorch + CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install neuralforecast transformers accelerate
```

## 🚀 Быстрый старт

### 1. Запуск сервера

```bash
cd /home/user/webapp/backend
python main.py
```

Ожидаемый вывод:
```
INFO:     Started server process [PID]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:3000
```

### 2. Проверка API

```bash
# Проверка health
curl http://localhost:3000/health

# Проверка доступных моделей и SLM
curl http://localhost:3000/models
```

Ожидаемый ответ `/models`:
```json
{
  "timellm": {
    "name": "TimeLLM",
    "available_slm": {
      "qwen2-0.5b": "🟢 Qwen2-0.5B (500M) - Рекомендуется!",
      "llama3.2-1b": "🟢 Llama-3.2-1B (1B) - Meta SLM 2024",
      ...
    },
    "default_slm": "qwen2-0.5b"
  }
}
```

### 3. Подготовка тестовых данных

Создайте файл `test_data.csv`:

```csv
date,value
2024-01-01,100
2024-01-02,102
2024-01-03,105
2024-01-04,103
2024-01-05,107
2024-01-06,110
2024-01-07,108
2024-01-08,112
2024-01-09,115
2024-01-10,113
2024-01-11,118
2024-01-12,120
2024-01-13,119
2024-01-14,123
2024-01-15,125
2024-01-16,124
2024-01-17,128
2024-01-18,130
2024-01-19,132
2024-01-20,135
```

### 4. Тестирование через cURL

#### Тест 1: Qwen2-0.5B (по умолчанию)

```bash
curl -X POST http://localhost:3000/forecast \
  -F 'dates=["2024-01-01","2024-01-02","2024-01-03","2024-01-04","2024-01-05","2024-01-06","2024-01-07","2024-01-08","2024-01-09","2024-01-10","2024-01-11","2024-01-12","2024-01-13","2024-01-14","2024-01-15","2024-01-16","2024-01-17","2024-01-18","2024-01-19","2024-01-20"]' \
  -F 'values=[100,102,105,103,107,110,108,112,115,113,118,120,119,123,125,124,128,130,132,135]' \
  -F 'model_type=timellm' \
  -F 'steps=5'
```

**Ожидаемое время**: 30-90 секунд (зависит от GPU)

**Ожидаемый ответ**:
```json
{
  "status": "success",
  "forecast": {
    "values": [137.5, 139.2, 141.0, 142.8, 144.5],
    "dates": ["2024-01-21", "2024-01-22", ...]
  },
  "model_info": "TimeLLM с SLM: Qwen2-0.5B"
}
```

#### Тест 2: Llama-3.2-1B

```bash
curl -X POST http://localhost:3000/forecast \
  -F 'dates=["2024-01-01",...]' \
  -F 'values=[100,102,...]' \
  -F 'model_type=timellm' \
  -F 'llm_model=llama3.2-1b' \
  -F 'steps=5'
```

**Ожидаемое время**: 40-120 секунд

#### Тест 3: Phi-3-mini (максимальная точность)

```bash
curl -X POST http://localhost:3000/forecast \
  -F 'dates=["2024-01-01",...]' \
  -F 'values=[100,102,...]' \
  -F 'model_type=timellm' \
  -F 'llm_model=phi3-mini' \
  -F 'steps=5'
```

**Ожидаемое время**: 120-180 секунд

### 5. Тестирование через веб-интерфейс

1. Откройте `http://localhost:3000/` в браузере
2. Загрузите `test_data.csv`
3. Выберите модель "TimeLLM (SLM 2024-2025)"
4. Выберите SLM модель из выпадающего списка
5. Установите шагов: 5-10
6. Нажмите "🔮 Выполнить Прогноз"

**Ожидаемый результат**:
- График с историческими данными (синий)
- Прогноз (красный)
- Доверительные интервалы (зелёные области)
- Информация о модели

## 📊 Проверка производительности

### Python скрипт для бенчмарка

```python
import time
import requests
import json

API_URL = "http://localhost:3000"

# Тестовые данные
dates = [f"2024-01-{i:02d}" for i in range(1, 21)]
values = [100 + i*2 for i in range(20)]

# Список SLM для тестирования
slm_models = [
    'qwen2-0.5b',
    'llama3.2-1b',
    'gpt2',
    'phi3-mini'
]

results = []

for slm in slm_models:
    print(f"\n🧪 Тестирование {slm}...")
    
    start_time = time.time()
    
    response = requests.post(
        f"{API_URL}/forecast",
        data={
            'dates': json.dumps(dates),
            'values': json.dumps(values),
            'model_type': 'timellm',
            'llm_model': slm,
            'steps': 5
        }
    )
    
    elapsed = time.time() - start_time
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Успех за {elapsed:.1f}s")
        print(f"   Прогноз: {result['forecast']['values'][:3]}...")
        results.append({
            'model': slm,
            'time': elapsed,
            'success': True
        })
    else:
        print(f"❌ Ошибка: {response.status_code}")
        results.append({
            'model': slm,
            'time': elapsed,
            'success': False
        })

# Итоги
print("\n📊 Результаты:")
print("-" * 50)
for r in results:
    status = "✅" if r['success'] else "❌"
    print(f"{status} {r['model']:20s} {r['time']:6.1f}s")
```

Сохраните как `benchmark_slm.py` и запустите:

```bash
python benchmark_slm.py
```

## 🎯 Критерии успеха

### ✅ Успешный тест

- [ ] Сервер запускается без ошибок
- [ ] API `/health` возвращает `"status": "healthy"`
- [ ] API `/models` возвращает список SLM
- [ ] Qwen2-0.5B: прогноз за 30-90s
- [ ] Llama-3.2-1B: прогноз за 40-120s
- [ ] Phi-3-mini: прогноз за 120-180s
- [ ] Веб-интерфейс отображает выбор SLM
- [ ] График строится корректно
- [ ] Доверительные интервалы присутствуют

### ⚠️ Возможные проблемы

#### 1. CUDA Out of Memory

**Симптомы**: RuntimeError: CUDA out of memory

**Решение**:
```bash
# Используйте более лёгкую модель
llm_model=qwen2-0.5b  # или gpt2
```

#### 2. Модель не загружается

**Симптомы**: Ошибка загрузки из HuggingFace

**Решение**:
```bash
# Проверьте интернет-соединение
# Модели кэшируются в ~/.cache/huggingface/
# Удалите кэш и попробуйте снова
rm -rf ~/.cache/huggingface/hub/models--*
```

#### 3. NeuralForecast не установлен

**Симптомы**: ModuleNotFoundError: No module named 'neuralforecast'

**Решение**:
```bash
pip install neuralforecast
```

#### 4. Fallback на simple режим

**Симптомы**: "Используется simple режим" в логах

**Причина**: GPU недоступен или NeuralForecast не установлен

**Поведение**: Система работает в упрощённом статистическом режиме (тренд + сезонность)

## 📝 Логирование

### Важные логи для проверки

```bash
# Успешная инициализация GPU
✅ CUDA инициализирован: NVIDIA GeForce RTX 4070 Ti SUPER

# Загрузка SLM модели
🤖 Выбрана SLM: 🟢 Qwen2-0.5B (500M) - Топ SLM 2024!
📊 NeuralForecast: Модель=Qwen/Qwen2-0.5B, d_llm=896

# Обучение
🚀 Начинаю обучение модели...
Epoch 0: 100%|██████████| 20/20 [00:45<00:00]

# Прогноз готов
✓ Прогноз выполнен успешно
```

### Ошибки для внимания

```bash
# CUDA недоступен
⚠️ Warning: NeuralForecast требует GPU. CUDA недоступен.
Используется simple режим

# Out of Memory
❌ CUDA RuntimeError: CUDA out of memory
Переключаюсь на simple режим
```

## 🔍 Дополнительная диагностика

### Проверка CUDA

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

### Проверка NeuralForecast

```python
try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import TimeLLM
    print("✅ NeuralForecast установлен")
except ImportError:
    print("❌ NeuralForecast не установлен")
```

### Проверка доступности моделей

```python
from transformers import AutoModel, AutoTokenizer

models = ['Qwen/Qwen2-0.5B', 'meta-llama/Llama-3.2-1B', 'gpt2']

for model_name in models:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"✅ {model_name} доступна")
    except Exception as e:
        print(f"❌ {model_name}: {e}")
```

## 📈 Ожидаемая производительность

На RTX 4070 Ti Super (16GB VRAM), 20 точек данных, 5 шагов прогноза:

| Этап | Qwen2-0.5B | Llama-3.2-1B | Phi-3-mini |
|------|------------|--------------|------------|
| Загрузка модели | 5-10s | 10-20s | 30-60s |
| Обучение (20 steps) | 30-50s | 50-80s | 100-150s |
| Прогноз | 1-3s | 2-5s | 3-8s |
| **Общее** | **40-65s** | **65-110s** | **135-220s** |

## ✅ Чек-лист тестирования

- [ ] Сервер запустился
- [ ] `/health` работает
- [ ] `/models` возвращает SLM список
- [ ] Qwen2-0.5B тест пройден
- [ ] Llama-3.2-1B тест пройден
- [ ] Веб-интерфейс работает
- [ ] Выбор SLM отображается
- [ ] График строится
- [ ] Прогноз корректный
- [ ] Доверительные интервалы есть
- [ ] Производительность приемлемая

## 🎓 Дополнительные тесты

### Тест с малыми данными (<30 точек)

```bash
# Только 10 точек - должен работать без early stopping
curl -X POST http://localhost:3000/forecast \
  -F 'dates=["2024-01-01",...,"2024-01-10"]' \
  -F 'values=[100,102,105,103,107,110,108,112,115,113]' \
  -F 'model_type=timellm' \
  -F 'steps=3'
```

### Тест с большими данными (>100 точек)

```bash
# 100+ точек - проверка масштабируемости
# Генерируйте данные программно
```

### Стресс-тест: несколько моделей подряд

```bash
for model in qwen2-0.5b llama3.2-1b gpt2; do
  echo "Testing $model..."
  curl -X POST http://localhost:3000/forecast \
    -F 'dates=[...]' -F 'values=[...]' \
    -F 'model_type=timellm' -F "llm_model=$model" -F 'steps=5'
  sleep 5
done
```

---

**Дата**: 2025-02-04  
**Версия**: 1.0  
**Статус**: Ready for Testing 🧪
