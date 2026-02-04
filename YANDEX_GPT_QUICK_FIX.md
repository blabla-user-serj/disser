# 🔥 YandexGPT 403 Error - Быстрое решение

## ❌ Ошибка
```
HTTP 403: Permission denied
```

## ✅ Решение за 3 минуты

### 0️⃣ Установите openai (если ещё не установлен)
```bash
pip install openai>=1.0.0
```

### 1️⃣ Откройте Yandex Cloud Console
https://console.cloud.yandex.ru/

### 2️⃣ Назначьте роль (КРИТИЧНО!)

```
1. Выберите каталог → Права доступа
2. Найдите ваш сервисный аккаунт
3. Назначьте роль: ai.languageModels.user
4. Сохраните
```

**Через CLI**:
```bash
yc resource-manager folder add-access-binding b1gpv7fps36gl28jtfqm \
  --role ai.languageModels.user \
  --subject serviceAccount:YOUR_SA_ID
```

### 3️⃣ Проверьте .env

```bash
# config/.env
YANDEX_API_KEY=your_api_key
YANDEX_FOLDER_ID=b1gpv7fps36gl28jtfqm
YANDEX_MODEL=yandexgpt-lite
```

### 4️⃣ Перезапустите сервер

```bash
cd backend
python main.py
```

## 🆘 Все еще не работает?

**Используйте систему БЕЗ YandexGPT**:
- Закомментируйте `YANDEX_API_KEY` в `.env`
- Система автоматически переключится на базовый анализ
- Модели работают отлично без YandexGPT!

## 📚 Полная инструкция
См. `docs/YANDEX_GPT_SETUP.md`

## 🔧 Используется новый API
Теперь используется OpenAI-совместимый API Yandex Cloud:
- Base URL: `https://ai.api.cloud.yandex.net/v1`
- Метод: `responses.create()`
- Требуется: `pip install openai>=1.0.0`
