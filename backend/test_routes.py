"""
Тестовый режим: прогон всех моделей с расчётом метрик на реальных данных.
Не изменяет основной main.py — подключается через APIRouter.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import math
import json
from io import BytesIO, StringIO
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from models import SARIMAXS, XGBoostTS, TimeLLM, HybridModel

try:
    from models.timellm_gguf import destroy_all_models
except Exception:
    def destroy_all_models():
        pass

router = APIRouter(prefix="/test", tags=["testing"])

# ─────────────────────────────────────────────
# Переиспользуем утилиты из main (импортируем)
# ─────────────────────────────────────────────
from backend.main import (
    sanitize_for_json,
    sanitize_array,
    infer_frequency,
    parse_csv,
    find_date_column,
    find_value_column,
    smart_parse_date,
    smart_parse_numeric,
)

try:
    from backend.llm_expert import LLMExpert
    _LLM_AVAILABLE = True
except Exception:
    _LLM_AVAILABLE = False


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE, MAPE, RMSE на тестовой выборке."""
    n = len(y_true)
    if n == 0:
        return {"MAE": 0.0, "MAPE": 0.0, "RMSE": 0.0}

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # MAPE: пропускаем нули в знаменателе
    mask = y_true != 0
    if mask.sum() > 0:
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    else:
        mape = 0.0

    return {"MAE": round(mae, 4), "MAPE": round(mape, 4), "RMSE": round(rmse, 4)}


def _align_pred(pred: np.ndarray, n: int) -> np.ndarray:
    """Обрезает или дополняет прогноз до длины n."""
    if len(pred) >= n:
        return pred[:n]
    return np.concatenate([pred, np.full(n - len(pred), pred[-1])])


@router.post("/forecast")
async def test_forecast(
    train_file: UploadFile = File(...),
    test_file: UploadFile = File(...),
    slm_model: str = Form("smollm2-360m"),
    web_urls: str = Form(""),
    dataset_description: str = Form(""),
    context_files: List[UploadFile] = File(default=[]),
):
    """
    Тестовый прогноз: обучение на train_file, сравнение с test_file.

    Прогоняет все 4 модели: SARIMA, XGBoost, TimeLLM, Hybrid.
    Возвращает прогнозы и метрики (MAE, MAPE, RMSE) для каждой.
    """
    # ── 1. Парсинг файлов ──────────────────────────────────────────────
    async def _parse(upload: UploadFile):
        content = await upload.read()
        fname = (upload.filename or "").lower()
        if fname.endswith(".csv"):
            df, date_col, val_col = parse_csv(content)
        elif fname.endswith((".xlsx", ".xls")):
            df = pd.read_excel(BytesIO(content))
            date_col = find_date_column(df)
            val_col = find_value_column(df, date_col)
        else:
            raise HTTPException(400, f"Неподдерживаемый формат файла: {upload.filename}")

        df["_d"] = df[date_col].apply(smart_parse_date)
        df["_v"] = df[val_col].apply(smart_parse_numeric)
        df = df.dropna(subset=["_d", "_v"]).sort_values("_d")
        if len(df) == 0:
            raise HTTPException(400, f"Файл '{upload.filename}' не содержит валидных данных")
        dates = pd.to_datetime(df["_d"].values)
        values = df["_v"].values.astype(float)
        return dates, values

    train_dates, train_values = await _parse(train_file)
    test_dates, test_values = await _parse(test_file)

    steps = len(test_values)
    if steps == 0:
        raise HTTPException(400, "Тестовый файл пуст")

    frequency = infer_frequency(train_dates)

    # ── 2. Очистка GPU ────────────────────────────────────────────────
    destroy_all_models()
    if _TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── 3. Прогон моделей ─────────────────────────────────────────────
    MODEL_CONFIGS = [
        ("sarima",   "SARIMA-XS"),
        ("xgboost",  "XGBoost"),
        ("timellm",  "TimeLLM"),
        ("hybrid",   "Hybrid"),
    ]

    # ── 3а. Парсинг веб-ссылок для LLM-эксперта ────────────────────
    web_urls_list = [u.strip() for u in web_urls.splitlines() if u.strip()]

    # ── 3б. Извлечение текста из контекстных файлов (PDF/DOCX) ────────
    extra_context_parts = []
    for cf in context_files:
        if not cf or not cf.filename:
            continue
        fname = cf.filename.lower()
        if not (fname.endswith(".pdf") or fname.endswith(".docx")):
            print(f"[TEST] Пропущен неподдерживаемый файл: {cf.filename}")
            continue
        raw = await cf.read()
        text = LLMExpert.extract_text_from_file(raw, cf.filename)
        if text.strip():
            extra_context_parts.append(f"--- {cf.filename} ---\n{text}")
            print(f"[TEST] Прочитан файл '{cf.filename}': {len(text)} символов")

    # Добавляем описание датасета в начало контекста
    if dataset_description and dataset_description.strip():
        extra_context_parts.insert(0, f"--- ОПИСАНИЕ ДАТАСЕТА ---\n{dataset_description}")
        
    extra_context = "\n\n".join(extra_context_parts)

    results = {}

    for model_key, model_name in MODEL_CONFIGS:
        print(f"\n{'='*50}")
        print(f"[TEST] Обучение модели: {model_name}")
        print(f"{'='*50}")

        try:
            # Инициализация
            if model_key == "sarima":
                model = SARIMAXS()
            elif model_key == "xgboost":
                model = XGBoostTS()
            elif model_key == "timellm":
                cuda_ok = _TORCH_AVAILABLE and torch.cuda.is_available()
                backend = "neuralforecast" if cuda_ok else "simple"
                model = TimeLLM(llm_backend=backend, neuralforecast_model=slm_model)
            else:  # hybrid
                use_slm = _TORCH_AVAILABLE and torch.cuda.is_available()
                model = HybridModel(use_slm=use_slm, slm_model=slm_model)

            # Обучение
            if model_key in ("timellm", "hybrid"):
                model.fit(train_values, steps=steps)
            else:
                model.fit(train_values)

            # ── LLM-коррекция (только для hybrid) ────────────────────
            llm_correction = None
            llm_analysis = ""
            correction_applied = False

            if model_key == "hybrid" and _LLM_AVAILABLE:
                try:
                    llm_expert = LLMExpert()
                    pre = model.predict(steps, return_conf_int=True, alpha=0.05)
                    pre_fc  = pre["forecast"]
                    pre_lo  = pre.get("lower_bound",  pre_fc * 0.95)
                    pre_hi  = pre.get("upper_bound",  pre_fc * 1.05)

                    corr = llm_expert.correct_forecast(
                        historical_data=train_values,
                        forecast=pre_fc,
                        lower_bound=pre_lo,
                        upper_bound=pre_hi,
                        web_urls=web_urls_list,
                        extra_context=extra_context,
                    )
                    correction_applied = corr["correction_applied"]
                    llm_analysis = corr["analysis"]
                    if correction_applied:
                        llm_correction = corr["corrected_forecast"] - pre_fc
                        print(f"[TEST] Hybrid LLM коррекция применена, Δ_LLM[0]={llm_correction[0]:.4f}")
                except Exception as llm_err:
                    print(f"[TEST] Hybrid LLM коррекция недоступна: {llm_err}")
                    llm_analysis = f"⚠️ LLM коррекция недоступна: {llm_err}"

            # Прогноз
            if model_key == "hybrid" and llm_correction is not None:
                pred_result = model.predict(steps, return_conf_int=True, alpha=0.05,
                                            llm_correction=llm_correction)
            else:
                pred_result = model.predict(steps, return_conf_int=True, alpha=0.05)

            forecast = sanitize_array(pred_result["forecast"])
            lower = sanitize_array(pred_result.get("lower_bound", forecast * 0.95))
            upper = sanitize_array(pred_result.get("upper_bound", forecast * 1.05))

            # Выравниваем по длине теста
            forecast = _align_pred(forecast, steps)
            lower    = _align_pred(lower,    steps)
            upper    = _align_pred(upper,    steps)

            # Метрики
            metrics = compute_metrics(test_values, forecast)

            results[model_key] = {
                "name": model_name,
                "status": "ok",
                "forecast": sanitize_for_json(forecast.tolist()),
                "lower_bound": sanitize_for_json(lower.tolist()),
                "upper_bound": sanitize_for_json(upper.tolist()),
                "metrics": metrics,
                "llm_analysis": llm_analysis,
                "llm_correction_applied": correction_applied,
            }

            print(f"[TEST] {model_name}: MAE={metrics['MAE']:.4f}, "
                  f"MAPE={metrics['MAPE']:.2f}%, RMSE={metrics['RMSE']:.4f}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            results[model_key] = {
                "name": model_name,
                "status": "error",
                "error": str(e),
                "forecast": [],
                "lower_bound": [],
                "upper_bound": [],
                "metrics": {"MAE": None, "MAPE": None, "RMSE": None},
            }
            print(f"[TEST] {model_name}: ОШИБКА — {e}")

        finally:
            # Освобождаем GPU между моделями
            destroy_all_models()
            if _TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── 4. Формируем ответ ────────────────────────────────────────────
    return {
        "status": "success",
        "frequency": frequency,
        "steps": steps,
        "train": {
            "dates": train_dates.strftime("%Y-%m-%d %H:%M:%S").tolist(),
            "values": sanitize_for_json(train_values.tolist()),
        },
        "test": {
            "dates": test_dates.strftime("%Y-%m-%d %H:%M:%S").tolist(),
            "values": sanitize_for_json(test_values.tolist()),
        },
        "models": results,
    }
