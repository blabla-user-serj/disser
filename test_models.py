#!/usr/bin/env python3
"""
Тест работоспособности моделей TimeLLM, SARIMA-XS, XGBoost
Без зависимостей от внешних библиотек - только встроенные модули
"""
import sys
import os

# Добавляем текущую директорию в путь
sys.path.insert(0, os.path.dirname(__file__))

print("="*70)
print("🧪 ТЕСТ РАБОТОСПОСОБНОСТИ МОДЕЛЕЙ")
print("="*70)

# Генерируем простые тестовые данные
try:
    import numpy as np
    data = np.array([100.0, 102.0, 104.0, 103.0, 105.0, 107.0, 106.0, 108.0, 
                     110.0, 112.0, 111.0, 113.0, 115.0, 114.0, 116.0, 118.0,
                     120.0, 119.0, 121.0, 123.0, 125.0, 124.0, 126.0, 128.0])
    print(f"\n✅ Сгенерированы тестовые данные: {len(data)} точек")
    print(f"   Диапазон: [{data.min():.2f}, {data.max():.2f}]")
except ImportError:
    print("❌ numpy не установлен, пропускаем тесты")
    sys.exit(1)

# Тест 1: SARIMA-XS
print("\n" + "="*70)
print("📊 ТЕСТ 1: SARIMA-XS")
print("="*70)
try:
    from models.sarima_xs import SARIMAXS
    
    model = SARIMAXS()
    print("✅ SARIMA-XS импортирован")
    
    print("🔄 Обучение SARIMA-XS...")
    model.fit(data)
    print("✅ SARIMA-XS обучен")
    
    print("🔮 Выполнение прогноза...")
    forecast = model.predict(steps=10, return_conf_int=True)
    print(f"✅ SARIMA-XS прогноз: {len(forecast['forecast'])} шагов")
    print(f"   Прогноз: [{forecast['forecast'][0]:.2f}, {forecast['forecast'][1]:.2f}, {forecast['forecast'][2]:.2f}, ...]")
    
    # Получаем информацию о модели
    info = model.get_info()
    print(f"✅ SARIMA-XS инфо:")
    print(f"   Статус: {info.get('status', 'N/A')}")
    if 'best_params' in info:
        print(f"   Лучшие параметры: {info['best_params']}")
    
except Exception as e:
    print(f"❌ SARIMA-XS ошибка: {e}")
    import traceback
    traceback.print_exc()

# Тест 2: XGBoost
print("\n" + "="*70)
print("📊 ТЕСТ 2: XGBoost")
print("="*70)
try:
    from models.xgboost_model import XGBoostTS
    
    model = XGBoostTS()
    print("✅ XGBoost импортирован")
    
    print("🔄 Обучение XGBoost...")
    model.fit(data)
    print("✅ XGBoost обучен")
    
    print("🔮 Выполнение прогноза...")
    forecast = model.predict(steps=10, return_conf_int=True)
    print(f"✅ XGBoost прогноз: {len(forecast['forecast'])} шагов")
    print(f"   Прогноз: [{forecast['forecast'][0]:.2f}, {forecast['forecast'][1]:.2f}, {forecast['forecast'][2]:.2f}, ...]")
    
except Exception as e:
    print(f"❌ XGBoost ошибка: {e}")
    import traceback
    traceback.print_exc()

# Тест 3: TimeLLM (simple режим)
print("\n" + "="*70)
print("📊 ТЕСТ 3: TimeLLM (Simple режим - без GPU)")
print("="*70)
try:
    from models.timellm_gguf import TimeLLM
    
    # Принудительно используем simple режим (без GPU/LLM)
    model = TimeLLM(llm_backend='simple')
    print("✅ TimeLLM импортирован (simple режим)")
    
    print("🔄 Обучение TimeLLM...")
    model.fit(data)
    print("✅ TimeLLM обучен")
    
    print("🔮 Выполнение прогноза...")
    forecast = model.predict(steps=10, return_conf_int=True)
    print(f"✅ TimeLLM прогноз: {len(forecast['forecast'])} шагов")
    print(f"   Прогноз: [{forecast['forecast'][0]:.2f}, {forecast['forecast'][1]:.2f}, {forecast['forecast'][2]:.2f}, ...]")
    
    info = model.get_info()
    print(f"✅ TimeLLM инфо:")
    print(f"   Backend: {info.get('llm_backend', 'N/A')}")
    print(f"   Режим: {info.get('mode', 'N/A')}")
    print(f"   Точек данных: {info.get('data_points', 'N/A')}")
    
except Exception as e:
    print(f"❌ TimeLLM ошибка: {e}")
    import traceback
    traceback.print_exc()

# Тест 4: Гибридная модель
print("\n" + "="*70)
print("📊 ТЕСТ 4: Гибридная модель (комбинирует все модели)")
print("="*70)
try:
    from models.hybrid_model import HybridModel
    
    model = HybridModel()
    print("✅ HybridModel импортирован")
    
    print("🔄 Обучение HybridModel (обучает все 3 модели)...")
    model.fit(data)
    print("✅ HybridModel обучен")
    
    print("🔮 Выполнение прогноза...")
    forecast = model.predict(steps=10, return_conf_int=True)
    print(f"✅ HybridModel прогноз: {len(forecast['forecast'])} шагов")
    print(f"   Прогноз: [{forecast['forecast'][0]:.2f}, {forecast['forecast'][1]:.2f}, {forecast['forecast'][2]:.2f}, ...]")
    
    if 'weights' in forecast:
        print(f"✅ Веса моделей (адаптивное взвешивание):")
        for name, weight in forecast['weights'].items():
            print(f"   {name}: {weight:.4f}")
    
except Exception as e:
    print(f"❌ HybridModel ошибка: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("✅ ТЕСТ ЗАВЕРШЕН")
print("="*70)
print("\n📋 РЕЗЮМЕ:")
print("   • SARIMA-XS: классическая статистическая модель с Grid Search CV")
print("   • XGBoost: gradient boosting для временных рядов")
print("   • TimeLLM: гибридная модель (LLM + статистика)")
print("   • Hybrid: комбинирует все модели с адаптивным взвешиванием")
print("\n💡 РЕКОМЕНДАЦИЯ: Используйте Hybrid для лучшего качества!")
print("="*70)
