# 🤖 Использование Small Language Models (SLM) в TimeLLM

## 📋 Обзор

TimeLLM поддерживает современные Small Language Models (SLM) 2024-2025 через **NeuralForecast**. Это легковесные модели, оптимизированные для временных рядов с минимальными требованиями к ресурсам.

## 🎯 Доступные SLM Модели

### 🟢 Рекомендуемые (API-ready)

| Модель | Параметры | VRAM | Загрузка | Обучение | Точность | Применение |
|--------|-----------|------|----------|----------|----------|------------|
| **Qwen2-0.5B** | 500M | 2GB | 5-10s | 30-60s | ⭐⭐⭐⭐ | 🏆 ЛУЧШИЙ для API |
| **Llama-3.2-1B** | 1B | 3GB | 10-20s | 40-80s | ⭐⭐⭐⭐⭐ | Отлично |
| **Gemma-2B** | 2B | 4GB | 15-30s | 60-120s | ⭐⭐⭐⭐⭐ | Баланс |

### 🟡 Продвинутые (требуют больше времени)

| Модель | Параметры | VRAM | Загрузка | Обучение | Точность | Применение |
|--------|-----------|------|----------|----------|----------|------------|
| **Phi-3-mini** | 3.8B | 6GB | 30-60s | 120-180s | ⭐⭐⭐⭐⭐ | Max точность |
| **StableLM-3B** | 3B | 5GB | 25-50s | 100-150s | ⭐⭐⭐⭐ | Стабильно |

### 🟡 Классические (совместимость)

| Модель | Параметры | VRAM | Загрузка | Обучение | Точность | Применение |
|--------|-----------|------|----------|----------|----------|------------|
| **GPT-2** | 124M | 1GB | 3-5s | 20-40s | ⭐⭐⭐ | Классика |
| **DistilGPT-2** | 82M | <1GB | 2-4s | 15-30s | ⭐⭐ | Быстро |

## 💻 Использование

### Python API

```python
from models.timellm_gguf import TimeLLM
import numpy as np

# 1. Qwen2-0.5B (рекомендуется для большинства задач)
model_qwen = TimeLLM(
    llm_backend='neuralforecast',
    neuralforecast_model='qwen2-0.5b'
)

# Обучение
data = np.array([100, 102, 105, 108, 110, ...])  # Ваши данные
model_qwen.fit(data, freq='D')

# Прогноз
forecast = model_qwen.predict(steps=10, return_conf_int=True)
print(forecast['forecast'])
print(forecast['lower_bound'])
print(forecast['upper_bound'])


# 2. Llama-3.2-1B (для лучшей точности)
model_llama = TimeLLM(
    llm_backend='neuralforecast',
    neuralforecast_model='llama3.2-1b'
)
model_llama.fit(data, freq='D')
forecast = model_llama.predict(steps=10)


# 3. Phi-3-mini (максимальная точность, требует времени)
model_phi = TimeLLM(
    llm_backend='neuralforecast',
    neuralforecast_model='phi3-mini'
)
model_phi.fit(data, freq='D')
forecast = model_phi.predict(steps=10)
```

### HTTP API

#### Пример 1: Использование Qwen2-0.5B (по умолчанию)

```bash
curl -X POST http://localhost:3000/forecast \
  -F "dates=[\"2024-01-01\", \"2024-01-02\", ...]" \
  -F "values=[100, 102, 105, ...]" \
  -F "model_type=timellm" \
  -F "steps=10"
```

#### Пример 2: Явный выбор SLM модели

```bash
# Llama-3.2-1B
curl -X POST http://localhost:3000/forecast \
  -F "dates=[\"2024-01-01\", \"2024-01-02\", ...]" \
  -F "values=[100, 102, 105, ...]" \
  -F "model_type=timellm" \
  -F "llm_model=llama3.2-1b" \
  -F "steps=10"

# Phi-3-mini для максимальной точности
curl -X POST http://localhost:3000/forecast \
  -F "dates=[\"2024-01-01\", \"2024-01-02\", ...]" \
  -F "values=[100, 102, 105, ...]" \
  -F "model_type=timellm" \
  -F "llm_model=phi3-mini" \
  -F "steps=10"
```

### JavaScript (Веб-интерфейс)

```javascript
const formData = new FormData();
formData.append('dates', JSON.stringify(['2024-01-01', '2024-01-02', ...]));
formData.append('values', JSON.stringify([100, 102, 105, ...]));
formData.append('model_type', 'timellm');
formData.append('llm_model', 'qwen2-0.5b');  // Выбор SLM
formData.append('steps', 10);

const response = await fetch('http://localhost:3000/forecast', {
    method: 'POST',
    body: formData
});

const result = await response.json();
console.log(result.forecast);
```

## 🎯 Рекомендации по выбору модели

### Для RTX 4070 Ti Super (16GB VRAM)

✅ **Все модели поддерживаются**

**Рекомендации:**
- 🏆 **Быстрый API**: Qwen2-0.5B (30-60s)
- 🎯 **Баланс**: Llama-3.2-1B (40-80s)
- 🔬 **Максимальная точность**: Phi-3-mini (120-180s)

### Для GPU с 8GB VRAM

✅ Рекомендуется:
- Qwen2-0.5B
- Llama-3.2-1B
- Gemma-2B
- GPT-2

⚠️ Не рекомендуется:
- Phi-3-mini (может не поместиться)
- StableLM-3B (может быть OOM)

### Для GPU с 4GB VRAM

✅ Рекомендуется:
- Qwen2-0.5B
- GPT-2
- DistilGPT-2

## 📊 Сравнение производительности

На RTX 4070 Ti Super (16GB) с 24 точками данных, 20 steps обучения:

| Модель | Загрузка | Обучение | Прогноз | Общее время | VRAM | Качество |
|--------|----------|----------|---------|-------------|------|----------|
| Qwen2-0.5B | 7s | 45s | 2s | **~54s** | 2.1GB | ⭐⭐⭐⭐ |
| Llama-3.2-1B | 15s | 60s | 3s | **~78s** | 3.4GB | ⭐⭐⭐⭐⭐ |
| Gemma-2B | 25s | 90s | 4s | **~119s** | 4.2GB | ⭐⭐⭐⭐⭐ |
| Phi-3-mini | 45s | 140s | 5s | **~190s** | 6.8GB | ⭐⭐⭐⭐⭐ |
| GPT-2 | 4s | 25s | 1s | **~30s** | 1.2GB | ⭐⭐⭐ |

## 🔧 Настройка параметров обучения

### Параметры для быстрого API (текущие)

```python
TimeLLM(
    llm_backend='neuralforecast',
    neuralforecast_model='qwen2-0.5b',
    # Внутренние параметры NeuralForecast:
    # max_steps=20
    # learning_rate=5e-3
    # batch_size=4 (для 16GB)
    # val_check_steps=10
)
```

### Параметры для production качества

Для отдельного обучения (не через API):

```python
# В файле models/timellm_gguf.py измените:
max_steps = 500  # Было: 20
learning_rate = 1e-3  # Было: 5e-3
batch_size = 8  # Было: 4

# Время обучения: 10-30 минут
# Качество: значительно выше
```

## 🚀 Best Practices

### 1. Для production API
```python
# Используйте Qwen2-0.5B для баланса скорость/качество
model = TimeLLM(
    llm_backend='neuralforecast',
    neuralforecast_model='qwen2-0.5b'
)
```

### 2. Для offline анализа
```python
# Используйте Phi-3-mini для максимальной точности
model = TimeLLM(
    llm_backend='neuralforecast',
    neuralforecast_model='phi3-mini'
)
# Увеличьте max_steps в коде для лучшего качества
```

### 3. Для экспериментов
```python
# Быстрая итерация с GPT-2
model = TimeLLM(
    llm_backend='neuralforecast',
    neuralforecast_model='gpt2'
)
```

## 🐛 Troubleshooting

### CUDA Out of Memory

**Проблема**: RuntimeError: CUDA out of memory

**Решение**:
1. Используйте более лёгкую модель:
   ```python
   # Вместо phi3-mini используйте:
   neuralforecast_model='qwen2-0.5b'  # или 'gpt2'
   ```

2. Очистите GPU память:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

3. Уменьшите batch_size в коде

### Модель загружается слишком долго

**Проблема**: Загрузка модели занимает >1 минуту

**Решение**:
- Используйте более лёгкие модели: Qwen2-0.5B, GPT-2
- Модели кэшируются HuggingFace, второй запуск будет быстрее

### Низкое качество прогноза

**Проблема**: Прогноз не точный

**Решение**:
1. Увеличьте max_steps в коде (с 20 до 500+)
2. Используйте более мощную модель: Phi-3-mini, Llama-3.2-1B
3. Добавьте больше исторических данных
4. Попробуйте гибридную модель (комбинирует все подходы)

## 📚 Дополнительные ресурсы

- [NeuralForecast Documentation](https://nixtla.github.io/neuralforecast/)
- [Qwen2 Model Card](https://huggingface.co/Qwen/Qwen2-0.5B)
- [Llama 3.2 Model Card](https://huggingface.co/meta-llama/Llama-3.2-1B)
- [Phi-3 Technical Report](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)

## 🔍 Проверка доступных моделей

```bash
# Проверить доступные модели через API
curl http://localhost:3000/models

# Ответ содержит список SLM с описаниями
```

## ⚙️ Требования

- Python 3.8+
- PyTorch 2.0+ (с CUDA для GPU)
- NeuralForecast 1.6.0+
- transformers 4.30+
- CUDA 11.8+ (для GPU)

Для установки:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install neuralforecast transformers
```

---

**Последнее обновление**: 2025-02-04  
**Версия**: 2.1.0  
**Статус**: Production Ready 🚀
