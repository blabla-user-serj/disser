#!/usr/bin/env python3
"""
Бенчмарк: сравнение SARIMA, XGBoost, TimeLLM (simple) и Hybrid
на 5 датасетах социальных процессов РФ.

Запуск:
    cd /home/user/webapp
    python benchmark_datasets.py

Выводит таблицу MAE для каждой модели и датасета,
а также итоговый счёт (сколько раз каждая модель лучшая).
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.sarima_xs import SARIMAXS
from models.xgboost_model import XGBoostTS
from models.timellm_gguf import TimeLLM
from models.hybrid_model import HybridModel

# ─── Датасеты ────────────────────────────────────────────────────────────────
DATASETS = {
    "Безработица (%)": (
        np.array([7.4, 6.5, 5.5, 5.5, 5.2, 5.6, 5.5, 5.2, 4.8, 4.6]),
        np.array([5.8, 4.8, 3.9, 3.2]),
    ),
    "Естеств. прирост (тыс.)": (
        np.array([-239.6, 0.0, -4.3, 22.9, -33.7, 32.7, -2.3, -135.8, -224.6, -316.2]),
        np.array([-688.7, -1039.2, -594.7, -598.5]),
    ),
    "Пенсионеры (млн)": (
        np.array([38.6, 39.1, 40.2, 40.6, 41.0, 41.5, 42.7, 43.5, 43.9, 43.1]),
        np.array([43.3, 43.1, 42.8, 42.4]),
    ),
    "Бедность (%)": (
        np.array([12.5, 12.7, 10.7, 10.8, 11.2, 13.4, 13.3, 13.2, 12.9, 12.3]),
        np.array([12.1, 11.0, 10.5, 9.3]),
    ),
    "Ожид. продолж. жизни": (
        np.array([68.8, 69.8, 70.2, 70.8, 70.9, 71.4, 71.9, 72.7, 73.3, 73.3]),
        np.array([71.5, 70.1, 72.8, 73.4]),
    ),
}

MODEL_KEYS = ["sarima", "xgboost", "timellm_simple", "hybrid"]
MODEL_LABELS = {
    "sarima": "SARIMA-XS",
    "xgboost": "XGBoost",
    "timellm_simple": "TimeLLM(simple)",
    "hybrid": "Hybrid v2",
}

# ─── Запуск ──────────────────────────────────────────────────────────────────
results = {k: {} for k in MODEL_KEYS}

for ds_name, (train, test) in DATASETS.items():
    steps = len(test)
    print(f"\n{'='*60}")
    print(f"Датасет: {ds_name}  (train={len(train)}, test={steps})")
    print(f"{'='*60}")

    # SARIMA
    print("\n[SARIMA]")
    try:
        m = SARIMAXS(use_cv=False)
        m.fit(train)
        fc = np.array(m.predict(steps, return_conf_int=False)["forecast"])[:steps]
        results["sarima"][ds_name] = float(np.mean(np.abs(test - fc)))
    except Exception as e:
        print(f"  Ошибка: {e}")
        results["sarima"][ds_name] = None

    # XGBoost
    print("\n[XGBoost]")
    try:
        m = XGBoostTS(use_cv=False)
        m.fit(train)
        fc = np.array(m.predict(steps, return_conf_int=False)["forecast"])[:steps]
        results["xgboost"][ds_name] = float(np.mean(np.abs(test - fc)))
    except Exception as e:
        print(f"  Ошибка: {e}")
        results["xgboost"][ds_name] = None

    # TimeLLM simple
    print("\n[TimeLLM simple]")
    try:
        m = TimeLLM(llm_backend="simple")
        m.fit(train, steps=steps)
        fc = np.array(m.predict(steps, return_conf_int=False)["forecast"])[:steps]
        results["timellm_simple"][ds_name] = float(np.mean(np.abs(test - fc)))
    except Exception as e:
        print(f"  Ошибка: {e}")
        results["timellm_simple"][ds_name] = None

    # Hybrid v2
    print("\n[Hybrid v2]")
    try:
        m = HybridModel(use_slm=False)   # без GPU для чистоты бенчмарка
        m.fit(train, steps=steps)
        fc = np.array(m.predict(steps, return_conf_int=False)["forecast"])[:steps]
        results["hybrid"][ds_name] = float(np.mean(np.abs(test - fc)))
        print(f"  dominant_model={m._dominant_model}, weights={m.weights}")
    except Exception as e:
        import traceback; traceback.print_exc()
        results["hybrid"][ds_name] = None

# ─── Итоговая таблица ────────────────────────────────────────────────────────
print("\n\n" + "="*80)
print("ИТОГОВАЯ ТАБЛИЦА MAE (меньше — лучше)")
print("="*80)

col_w = 25
header = f"{'Датасет':<{col_w}}" + "".join(f"{MODEL_LABELS[k]:>18}" for k in MODEL_KEYS)
print(header)
print("-"*80)

wins = {k: 0 for k in MODEL_KEYS}

for ds_name in DATASETS:
    row_vals = {k: results[k].get(ds_name) for k in MODEL_KEYS}
    valid_vals = {k: v for k, v in row_vals.items() if v is not None}
    
    if valid_vals:
        best_key = min(valid_vals, key=valid_vals.get)
        wins[best_key] += 1
    else:
        best_key = None
    
    row = f"{ds_name:<{col_w}}"
    for k in MODEL_KEYS:
        v = row_vals[k]
        if v is None:
            cell = "  —"
        else:
            marker = " ←" if k == best_key else ""
            cell = f"{v:>14.4f}{marker:>4}"
        row += cell
    print(row)

print("-"*80)
wins_row = f"{'Побед (лучший MAE)':<{col_w}}" + "".join(f"{wins[k]:>18}" for k in MODEL_KEYS)
print(wins_row)
print("="*80)

# Naive baseline
print("\nBaseline (наивный — последнее значение train):")
for ds_name, (train, test) in DATASETS.items():
    naive_mae = float(np.mean(np.abs(test - train[-1])))
    print(f"  {ds_name:<{col_w}} naive MAE = {naive_mae:.4f}")
