# 🔧 Настройка YandexGPT API - Пошаговая инструкция

## 📋 Проблема: 403 Permission denied

```json
{"error": {
  "grpcCode": 7,
  "httpCode": 403,
  "message": "Permission denied",
  "httpStatus": "Forbidden"
}}
```

Это означает, что **API ключ не имеет прав** на использование YandexGPT в указанном каталоге.

---

## ✅ Решение: Пошаговая настройка

### Шаг 1: Войдите в Yandex Cloud Console

1. Откройте: https://console.cloud.yandex.ru/
2. Войдите с вашей учетной записью

### Шаг 2: Создайте API ключ (если еще не создан)

#### Вариант А: Через интерфейс консоли

1. Перейдите в **Cloud Console**: https://console.cloud.yandex.ru/
2. В верхнем меню выберите нужное **облако**
3. Перейдите в раздел **"Сервисные аккаунты"** (Service Accounts):
   ```
   Меню → IAM → Сервисные аккаунты
   ```
4. Создайте новый сервисный аккаунт или выберите существующий:
   - Нажмите **"Создать сервисный аккаунт"**
   - Имя: `yandexgpt-service-account`
   - Описание: `API ключ для YandexGPT`
5. После создания откройте сервисный аккаунт и создайте **API ключ**:
   - Нажмите **"Создать новый ключ"**
   - Выберите **"API ключ"**
   - Скопируйте ключ (показывается один раз!)

#### Вариант Б: Через Yandex Cloud CLI

```bash
# Установка CLI (если еще не установлен)
curl https://storage.yandexcloud.net/yandexcloud-yc/install.sh | bash

# Авторизация
yc init

# Создание сервисного аккаунта
yc iam service-account create --name yandexgpt-sa

# Получение ID сервисного аккаунта
SA_ID=$(yc iam service-account get yandexgpt-sa --format json | jq -r .id)

# Создание API ключа
yc iam api-key create --service-account-id $SA_ID --description "YandexGPT API key"
```

### Шаг 3: Назначьте роли на каталог (КРИТИЧНО!)

Это **самый важный шаг**, без которого будет ошибка 403.

#### Вариант А: Через интерфейс консоли

1. Перейдите в нужный **каталог** (folder):
   ```
   Console → Выберите облако → Выберите каталог
   ```
   
2. Ваш Folder ID: `b1gpv7fps36gl28jtfqm`
   - Проверьте его в верхней части страницы каталога

3. Перейдите в раздел **"Права доступа"** (Access Bindings):
   ```
   Меню каталога → Права доступа (Access bindings)
   ```

4. Нажмите **"Назначить роли"** (Assign roles)

5. Выберите созданный сервисный аккаунт: `yandexgpt-service-account`

6. Назначьте роль **`ai.languageModels.user`**:
   - Найдите роль в списке
   - Отметьте галочку
   - Нажмите **"Сохранить"**

7. **ВАЖНО**: Также назначьте роль **`ai.viewer`** (опционально, для просмотра):
   - Повторите шаги выше для роли `ai.viewer`

#### Вариант Б: Через Yandex Cloud CLI

```bash
# Получите ID сервисного аккаунта
SA_ID=$(yc iam service-account get yandexgpt-sa --format json | jq -r .id)

# Ваш Folder ID
FOLDER_ID="b1gpv7fps36gl28jtfqm"

# Назначение роли ai.languageModels.user
yc resource-manager folder add-access-binding $FOLDER_ID \
  --role ai.languageModels.user \
  --subject serviceAccount:$SA_ID

# Опционально: роль ai.viewer
yc resource-manager folder add-access-binding $FOLDER_ID \
  --role ai.viewer \
  --subject serviceAccount:$SA_ID

# Проверка назначенных ролей
yc resource-manager folder list-access-bindings $FOLDER_ID \
  --filter "subject.id='$SA_ID'"
```

### Шаг 4: Получите Folder ID

Если вы не знаете Folder ID:

#### Вариант А: Через консоль

1. Откройте: https://console.cloud.yandex.ru/
2. Выберите нужное облако
3. Выберите каталог
4. **Folder ID** отображается в верхней части страницы под названием каталога

#### Вариант Б: Через CLI

```bash
# Список всех каталогов
yc resource-manager folder list

# Получить ID каталога по имени
yc resource-manager folder get <folder-name> --format json | jq -r .id
```

### Шаг 5: Настройте .env файл

Создайте или отредактируйте файл `config/.env`:

```bash
# YandexGPT API
YANDEX_API_KEY=your_api_key_here  # Ваш API ключ из Шага 2
YANDEX_FOLDER_ID=b1gpv7fps36gl28jtfqm  # Ваш Folder ID
YANDEX_MODEL=yandexgpt-lite  # Или yandexgpt (более мощная)

# Опционально: Timeout для запросов
YANDEX_API_TIMEOUT=60
```

### Шаг 6: Проверьте настройки

Запустите тестовый скрипт:

```bash
cd /home/user/webapp
python -c "
import os
import sys
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv('config/.env')

print('=' * 60)
print('🔍 Проверка настроек YandexGPT')
print('=' * 60)

api_key = os.getenv('YANDEX_API_KEY')
folder_id = os.getenv('YANDEX_FOLDER_ID')
model = os.getenv('YANDEX_MODEL', 'yandexgpt-lite')

print(f'API Key: {'✓ Установлен' if api_key else '✗ Не установлен'}')
print(f'Folder ID: {folder_id or '✗ Не установлен'}')
print(f'Model: {model}')
print('=' * 60)

if not api_key or not folder_id:
    print('❌ Настройки неполные!')
    exit(1)

# Тест подключения
print('\\n🧪 Тестирую подключение к YandexGPT...')
from backend.llm_expert import LLMExpert

try:
    expert = LLMExpert()
    expert.test_yandex_gpt()
    print('\\n✅ Подключение успешно!')
except Exception as e:
    print(f'\\n❌ Ошибка: {e}')
    print('\\n💡 Проверьте:')
    print('   1. API ключ правильный')
    print('   2. Folder ID правильный')
    print('   3. Роль ai.languageModels.user назначена')
    exit(1)
"
```

---

## 🔍 Диагностика проблем

### Проблема 1: 403 Permission denied

**Причины**:
- ✗ Роль `ai.languageModels.user` не назначена
- ✗ Неправильный Folder ID
- ✗ API ключ не привязан к сервисному аккаунту

**Решение**:
1. Проверьте роли через CLI:
   ```bash
   yc resource-manager folder list-access-bindings b1gpv7fps36gl28jtfqm
   ```
2. Убедитесь, что сервисный аккаунт имеет роль `ai.languageModels.user`
3. Переназначьте роль, если нужно (см. Шаг 3)

### Проблема 2: 401 Unauthorized

**Причины**:
- ✗ Неправильный API ключ
- ✗ API ключ истек

**Решение**:
1. Проверьте API ключ в `.env` файле
2. Создайте новый API ключ (см. Шаг 2)

### Проблема 3: 404 Not Found

**Причины**:
- ✗ Неправильный Folder ID
- ✗ Каталог удален

**Решение**:
1. Проверьте Folder ID:
   ```bash
   yc resource-manager folder get b1gpv7fps36gl28jtfqm
   ```
2. Используйте правильный Folder ID из консоли

---

## 📊 Альтернативные роли

Если роль `ai.languageModels.user` недостаточна, попробуйте более широкие роли:

| Роль | Описание | Права |
|------|----------|-------|
| `ai.languageModels.user` | Пользователь LLM | Только использование YandexGPT |
| `ai.editor` | Редактор AI | Использование + управление |
| `editor` | Редактор каталога | Полный доступ к каталогу |

**Команда для назначения более широкой роли**:
```bash
yc resource-manager folder add-access-binding b1gpv7fps36gl28jtfqm \
  --role ai.editor \
  --subject serviceAccount:$SA_ID
```

---

## 🧪 Тестовые команды

### Проверка API ключа

```bash
curl -X POST \
  https://llm.api.cloud.yandex.net/foundationModels/v1/completion \
  -H "Authorization: Api-Key $YANDEX_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "modelUri": "gpt://b1gpv7fps36gl28jtfqm/yandexgpt-lite/latest",
    "completionOptions": {
      "stream": false,
      "temperature": 0.3,
      "maxTokens": 100
    },
    "messages": [
      {
        "role": "user",
        "text": "Привет"
      }
    ]
  }'
```

**Ожидаемый успешный ответ**:
```json
{
  "result": {
    "alternatives": [{
      "message": {
        "role": "assistant",
        "text": "Здравствуйте! Чем могу помочь?"
      },
      "status": "ALTERNATIVE_STATUS_FINAL"
    }],
    "usage": {...},
    "modelVersion": "..."
  }
}
```

---

## 📚 Полезные ссылки

- **Yandex Cloud Console**: https://console.cloud.yandex.ru/
- **Документация YandexGPT**: https://cloud.yandex.ru/docs/foundation-models/
- **IAM роли**: https://cloud.yandex.ru/docs/iam/concepts/access-control/roles
- **API ключи**: https://cloud.yandex.ru/docs/iam/concepts/authorization/api-key
- **Yandex Cloud CLI**: https://cloud.yandex.ru/docs/cli/quickstart

---

## ⚠️ Важные замечания

1. **API ключ показывается один раз** при создании - скопируйте его сразу!
2. **Роли назначаются на уровне каталога** (folder), а не облака (cloud)
3. **Изменения прав доступа** могут занять 1-2 минуты для применения
4. **Система работает БЕЗ YandexGPT** - если не настроен, используется простой статистический анализ

---

## 🎯 Краткий чеклист

- [ ] Создан сервисный аккаунт
- [ ] Создан API ключ
- [ ] Роль `ai.languageModels.user` назначена на каталог
- [ ] Folder ID правильный: `b1gpv7fps36gl28jtfqm`
- [ ] API ключ добавлен в `config/.env`
- [ ] Folder ID добавлен в `config/.env`
- [ ] Тестовый запрос выполнен успешно

---

## 💡 Если ничего не помогает

**Используйте систему БЕЗ YandexGPT**:

1. Удалите или закомментируйте в `config/.env`:
   ```bash
   # YANDEX_API_KEY=...
   # YANDEX_FOLDER_ID=...
   ```

2. Система автоматически переключится на **базовый статистический анализ**

3. Модели SARIMA-XS, XGBoost, TimeLLM (simple) и Hybrid **работают отлично без YandexGPT**!

**Рекомендуется использовать Hybrid модель для лучшего качества** - она не зависит от YandexGPT! 🚀
