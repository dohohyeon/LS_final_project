# -*- coding: utf-8 -*-
"""
HGBR + (시간/달력/공휴일±1/주기 포리에) 기반 전기요금 예측
- VALIDATION_MODE: "tscv" | "holdout"
- tscv 모드에서도 TRAIN_MONTHS_USED로 학습기간(월) 제한 가능
- Optuna로 하이퍼파라미터 튜닝 → 최적 파라미터로 재학습 → 제출 파일 생성

입출력
- 입력: ./data/raw/{train.csv, test.csv, sample_submission.csv}
- 출력: ./data/raw/out/{validation_metrics.csv, submission_hgbr_l1_enhanced.csv}
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings("ignore")

try:
    import optuna
except Exception:
    optuna = None

# =========================
# 0) 사용자 설정
# =========================
DATA_DIR = "./data/raw"
OUT_DIR  = os.path.join(DATA_DIR, "out")
os.makedirs(OUT_DIR, exist_ok=True)

# 검증 방법: "tscv" 또는 "holdout"
VALIDATION_MODE = "holdout"   # "tscv" | "holdout"

# --- TS-CV 모드에서 사용할 학습 월 제한 ---
TRAIN_MONTHS_USED = [3,4,5,6,7,8,9,10,11]
TSCV_N_SPLITS = 5  # 고정

# --- Holdout 모드 설정 ---
TRAIN_MONTHS = [3,4,5,6,7,8,9,10]   # 학습 월
VALID_MONTHS = [11]               # 검증 월
REFIT_INCLUDE_VALID = True           # 최종학습에 valid 포함할지

# Target / 칼럼명 후보
TARGET_CANDIDATES = ["전기요금(원)", "target", "TARGET", "price", "Price", "y"]

# Optuna
USE_OPTUNA = True
N_TRIALS   = 50
RANDOM_STATE = 42

# 공휴일: (요청대로) 아래 목록만 반영 + ±1일 윈도우
KR_HOLIDAYS = pd.to_datetime([
    '2024-01-01','2024-02-14','2024-02-15','2024-02-16','2024-02-17','2024-02-18',
    '2024-03-01','2024-05-01','2024-05-05','2024-05-22','2024-06-06',
    '2024-06-13','2024-08-01','2024-08-02','2024-08-03','2024-08-15',
    '2024-09-22','2024-09-23','2024-09-24','2024-09-25','2024-09-26',
    '2024-10-03','2024-10-09','2024-12-25','2024-12-31'
]).normalize()

# =========================
# 1) 유틸
# =========================
def detect_datetime_col(df: pd.DataFrame) -> str:
    candidates = ["측정일시","datetime","timestamp","ds","date","Date","DATE","time","Time","TIMESTAMP"]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        try:
            pd.to_datetime(df[c]); return c
        except Exception:
            pass
    raise ValueError("Datetime column not found.")

def detect_target_col(df: pd.DataFrame) -> str:
    for c in TARGET_CANDIDATES:
        if c in df.columns:
            return c
    nums = df.select_dtypes(include="number").columns
    if len(nums):
        return nums[-1]
    raise ValueError("Target column not found.")

def encode_job_type(train_df: pd.DataFrame, test_df: pd.DataFrame):
    tr = train_df["작업유형"].astype(str) if "작업유형" in train_df.columns else pd.Series([""]*len(train_df), index=train_df.index)
    te = test_df["작업유형"].astype(str)  if "작업유형"  in test_df.columns  else pd.Series([""]*len(test_df),  index=test_df.index)
    cats = pd.Index(tr.unique()).union(pd.Index(te.unique()))
    mp = {c:i for i,c in enumerate(cats)}
    return tr.map(mp).astype("int32"), te.map(mp).astype("int32")

def numeric_only(dfX: pd.DataFrame) -> pd.DataFrame:
    """datetime 제거, bool→int8, 숫자형만 유지 (pandas nullable 안전)."""
    d = dfX.copy()
    dt_cols = d.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns
    d.drop(columns=list(dt_cols), inplace=True, errors="ignore")
    for c in d.columns:
        if pd.api.types.is_bool_dtype(d[c]):
            d[c] = d[c].astype("int8")
    d = d.select_dtypes(include=["number"])
    return d

def to_float32(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype(np.float32)

def align_and_impute_train_test(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    테스트를 훈련과 '동일한 컬럼 집합/순서'로 맞추고,
    훈련 중앙값으로 결측을 채운 뒤 둘 다 반환.
    """
    cols = list(X_train.columns)  # 순서 고정
    Xtr = X_train[cols].copy()
    Xte = X_test.copy()
    for c in cols:
        if c not in Xte.columns:
            Xte[c] = np.nan
    Xte = Xte[cols]
    med = Xtr.median(numeric_only=True)
    Xtr = Xtr.fillna(med)
    Xte = Xte.fillna(med)
    return to_float32(Xtr), to_float32(Xte)

def save_submission(sample_df: pd.DataFrame, test_df: pd.DataFrame, preds: np.ndarray, out_path: str):
    sub = sample_df.copy()
    target_out = sub.columns[-1] if len(sub.columns)>=2 else sub.columns[0]
    if len(sub) == len(test_df):
        sub[target_out] = preds
    else:
        n = min(len(sub), len(preds))
        sub.loc[:n-1, target_out] = preds[:n]
    sub[target_out] = sub[target_out].clip(lower=0)
    sub.to_csv(out_path, index=False)

# =========================
# 2) 시간/달력/공휴일(±1) 포처
# =========================
def add_time_features(df: pd.DataFrame, dtc: str) -> pd.DataFrame:
    ts = pd.to_datetime(df[dtc], errors="coerce")
    X = pd.DataFrame(index=df.index)
    X["_ts"]          = ts

    # 기본 달력 (is_weekend 제거 → is_monday / is_sunday 추가)
    X["slot_15m"]     = (ts.dt.hour*60 + ts.dt.minute)//15
    X["hour"]         = ts.dt.hour
    X["weekday"]      = ts.dt.weekday
    X["is_monday"]    = (ts.dt.weekday == 0).astype("int8")
    X["is_sunday"]    = (ts.dt.weekday == 6).astype("int8")
    X["weekofmonth"]  = ((ts.dt.day - 1)//7 + 1)
    X["day"]          = ts.dt.day
    X["month"]        = ts.dt.month
    X["quarter"]      = ts.dt.quarter
    X["day_of_year"]  = ts.dt.dayofyear
    X["hour_of_week"] = ts.dt.weekday*24 + ts.dt.hour
    X["is_month_start"] = ts.dt.is_month_start.astype("int8")
    X["is_month_end"]   = ts.dt.is_month_end.astype("int8")

    # Daily Fourier (period=96, k=1..3)
    period_day = 96.0
    idx_day = X["slot_15m"].astype(float)
    for k in (1,2,3):
        X[f"sin_day_{k}"] = np.sin(2*np.pi*k*idx_day/period_day)
        X[f"cos_day_{k}"] = np.cos(2*np.pi*k*idx_day/period_day)

    # Weekly Fourier (period=672, k=1..2)
    week_slot = (X["weekday"]*96 + X["slot_15m"]).astype(float)
    period_week = 96.0 * 7.0
    for k in (1,2):
        X[f"sin_week_{k}"] = np.sin(2*np.pi*k*week_slot/period_week)
        X[f"cos_week_{k}"] = np.cos(2*np.pi*k*week_slot/period_week)

    # Yearly Fourier (rough; 365일 기준, k=1)
    year_pos = X["day_of_year"].astype(float)
    period_year = 365.0
    X["sin_year_1"] = np.sin(2*np.pi*year_pos/period_year)
    X["cos_year_1"] = np.cos(2*np.pi*year_pos/period_year)

    return X

def make_kr_holiday_flags_pm1(ts_series: pd.Series) -> pd.DataFrame:
    """지정된 공휴일 + 공휴일의 ±1일을 반영. (is_weekend은 생성하지 않음)"""
    ts = pd.to_datetime(ts_series, errors="coerce")
    day = ts.dt.normalize()

    hol_exact = set(pd.to_datetime(KR_HOLIDAYS).normalize())
    hol_pm1 = set(hol_exact)
    for d in hol_exact:
        hol_pm1.add(d - pd.Timedelta(days=1))
        hol_pm1.add(d + pd.Timedelta(days=1))

    out = pd.DataFrame(index=ts.index)
    out["is_holiday_exact"] = day.isin(hol_exact).astype(int)
    out["is_holiday_pm1"]   = day.isin(hol_pm1).astype(int)
    return out

# =========================
# 3) 모델/튜닝 로직
# =========================
def build_model(params: dict) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=params.get("learning_rate", 0.05),
        max_iter=params.get("max_iter", 300),
        max_depth=params.get("max_depth", None),
        max_leaf_nodes=params.get("max_leaf_nodes", 31),
        min_samples_leaf=params.get("min_samples_leaf", 50),
        l2_regularization=params.get("l2_regularization", 0.0),
        max_bins=params.get("max_bins", 255),
        random_state=RANDOM_STATE
    )

def suggest_params(trial):
    return dict(
        learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        max_iter          = trial.suggest_int("max_iter", 200, 800),
        max_depth         = trial.suggest_int("max_depth", 3, 12),
        max_leaf_nodes    = trial.suggest_int("max_leaf_nodes", 15, 63),
        min_samples_leaf  = trial.suggest_int("min_samples_leaf", 20, 300),
        l2_regularization = trial.suggest_float("l2_regularization", 1e-6, 10.0, log=True),
        max_bins          = trial.suggest_int("max_bins", 128, 255),
    )

def tscv_optimize(X_df: pd.DataFrame, y: pd.Series):
    """학습 데이터(월 제한 반영 후 전체)로 TSCV 튜닝."""
    tscv = TimeSeriesSplit(n_splits=TSCV_N_SPLITS)

    def eval_with_params(params):
        maes = []
        for tr_idx, va_idx in tscv.split(X_df, y):
            Xtr, Xva = X_df.iloc[tr_idx], X_df.iloc[va_idx]
            ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
            med = Xtr.median(numeric_only=True)
            Xtr2 = to_float32(Xtr.fillna(med))
            Xva2 = to_float32(Xva.fillna(med))
            mdl = build_model(params)
            mdl.fit(Xtr2, ytr)
            pred = mdl.predict(Xva2)
            maes.append(mean_absolute_error(yva, pred))
        return float(np.mean(maes))

    if USE_OPTUNA and (optuna is not None):
        def objective(trial):
            params = suggest_params(trial)
            return eval_with_params(params)
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
        best_params = study.best_params
        best_score  = study.best_value
    else:
        best_params = dict(
            learning_rate=0.05, max_iter=400, max_depth=8,
            max_leaf_nodes=31, min_samples_leaf=50, l2_regularization=0.0, max_bins=255
        )
        best_score = eval_with_params(best_params)

    print(f"[TS-CV] best MAE={best_score:.6f} | params={best_params}")
    return best_params, best_score

def holdout_optimize(X_tr: pd.DataFrame, y_tr: pd.Series, X_va: pd.DataFrame, y_va: pd.Series):
    """Holdout 튜닝: Train 월로 학습, Valid 월로 평가."""
    def eval_with_params(params):
        med = X_tr.median(numeric_only=True)
        Xtr2 = to_float32(X_tr.fillna(med))
        Xva2 = to_float32(X_va.fillna(med))
        mdl = build_model(params)
        mdl.fit(Xtr2, y_tr)
        pred = mdl.predict(Xva2)
        return float(mean_absolute_error(y_va, pred))

    if USE_OPTUNA and (optuna is not None):
        def objective(trial):
            params = suggest_params(trial)
            return eval_with_params(params)
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
        best_params = study.best_params
        best_score  = study.best_value
    else:
        best_params = dict(
            learning_rate=0.05, max_iter=400, max_depth=8,
            max_leaf_nodes=31, min_samples_leaf=50, l2_regularization=0.0, max_bins=255
        )
        best_score = eval_with_params(best_params)

    print(f"[Holdout] Val MAE={best_score:.6f} | params={best_params}")
    return best_params, best_score

# =========================
# 4) 메인
# =========================
def main():
    train  = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test   = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    sample = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

    dtc = detect_datetime_col(train)
    tgt = detect_target_col(train)

    # 시간/달력/주기 포처
    X_trn = add_time_features(train, dtc)
    X_tst = add_time_features(test,  dtc)

    # 작업유형 인코딩(없으면 공백)
    tr_job, te_job = encode_job_type(train, test)
    X_trn["작업유형_enc"] = tr_job
    X_tst["작업유형_enc"] = te_job

    # 공휴일 플래그(정확히 리스트 + ±1일) — weekend 관련 컬럼 생성 안 함
    H_tr = make_kr_holiday_flags_pm1(X_trn["_ts"])
    H_te = make_kr_holiday_flags_pm1(X_tst["_ts"])
    X_trn = pd.concat([X_trn, H_tr], axis=1)
    X_tst = pd.concat([X_tst, H_te], axis=1)

    # 숫자형만 사용
    X_trn = numeric_only(X_trn)
    X_tst = numeric_only(X_tst)

    # 타깃/타임스탬프
    y  = pd.to_numeric(train[tgt], errors="coerce")
    ts = pd.to_datetime(train[dtc], errors="coerce")

    # ===== 검증 모드 분기 =====
    if VALIDATION_MODE.lower() == "tscv":
        # (1) 월 제한 적용
        mask_month = ts.dt.month.isin(TRAIN_MONTHS_USED)
        X_used = X_trn.loc[mask_month].reset_index(drop=True)
        y_used = y.loc[mask_month].reset_index(drop=True)

        # (2) TSCV 튜닝
        best_params, cv_mae = tscv_optimize(X_used, y_used)

        # (3) 최종 재학습: 제한 월 전체로 학습
        med = X_used.median(numeric_only=True)
        X_used2 = to_float32(X_used.fillna(med))
        mdl = build_model(best_params)
        mdl.fit(X_used2, y_used)

        # (4) Test 예측 (훈련 컬럼 순서 고정)
        _, X_tst2 = align_and_impute_train_test(X_used2, X_tst)
        test_pred = mdl.predict(X_tst2)

        pd.DataFrame([{"mode":"tscv", "cv_mae":cv_mae}]).to_csv(
            os.path.join(OUT_DIR, "validation_metrics.csv"), index=False, encoding="utf-8-sig"
        )

    else:  # HOLDOUT
        # (1) 월 분할
        mask_tr = ts.dt.month.isin(TRAIN_MONTHS)
        mask_va = ts.dt.month.isin(VALID_MONTHS)

        X_tr, y_tr = X_trn.loc[mask_tr].reset_index(drop=True), y.loc[mask_tr].reset_index(drop=True)
        X_va, y_va = X_trn.loc[mask_va].reset_index(drop=True), y.loc[mask_va].reset_index(drop=True)

        # (2) Holdout 튜닝
        best_params, val_mae = holdout_optimize(X_tr, y_tr, X_va, y_va)

        # (3) 최종 재학습 범위
        if REFIT_INCLUDE_VALID:
            X_refit = pd.concat([X_tr, X_va], axis=0).reset_index(drop=True)
            y_refit = pd.concat([y_tr, y_va], axis=0).reset_index(drop=True)
        else:
            X_refit, y_refit = X_tr, y_tr

        med = X_refit.median(numeric_only=True)
        X_refit2 = to_float32(X_refit.fillna(med))
        mdl = build_model(best_params)
        mdl.fit(X_refit2, y_refit)

        # (4) Test 예측 (훈련 컬럼 순서 고정)
        _, X_tst2 = align_and_impute_train_test(X_refit2, X_tst)
        test_pred = mdl.predict(X_tst2)

        pd.DataFrame([{"mode":"holdout", "val_mae":val_mae}]).to_csv(
            os.path.join(OUT_DIR, "validation_metrics.csv"), index=False, encoding="utf-8-sig"
        )

    # ===== 제출 저장 =====
    save_submission(sample, test, test_pred, os.path.join(OUT_DIR, "submission_hgbr_l1_enhanced.csv"))
    print("Saved:", os.path.join(OUT_DIR, "submission_hgbr_l1_enhanced.csv"))

if __name__ == "__main__":
    main()
