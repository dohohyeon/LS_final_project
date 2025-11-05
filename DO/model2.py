# -*- coding: utf-8 -*-
"""
Stage-1 only with lag/rolling + walk-forward prediction
- 6개 타깃(전력/무효전력/탄소/역률) 각각 Optuna 튜닝(LGB/XGB 중 선택)
- 시간 파생(월/일/시/요일 + sin/cos) + 휴일 + 월초/월말 + (일/월 플래그) + 피크/근접피크/야간저부하
- 라그/롤링: (자동창: 1h/6h/1d) ∪ (고정창: 2,3,6,12,24) 포함
- 검증 MAE 타깃별 출력
- 저장: 기존 test.csv 컬럼 유지 + 타깃명 그대로 새 컬럼을 오른쪽에 추가
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import optuna
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
import xgboost as xgb
from collections import deque

# ===================== 경로/기간/설정 =====================
DATA_TRAIN = "./data/raw/train.csv"
DATA_TEST  = "./data/raw/test.csv"

# (여기만 바꿔가며 실험) — 시계열 분할
TRAIN_START = "2024-01-01"
TRAIN_END   = "2024-10-31"
VALID_START = "2024-11-01"
VALID_END   = "2024-11-30"

# Stage-1 기반 모델: "lgb" 또는 "xgb"
STAGE1_BASE = "lgb"
N_TRIALS_S1 = 40
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ===================== 타깃/컬럼 정의 =====================
DT_COL   = "측정일시"
ID_COL   = "id"
CAT_FEATS = ["작업유형"]

S1_TARGETS = [
    "전력사용량(kWh)",
    "지상무효전력량(kVarh)",
    "진상무효전력량(kVarh)",
    "탄소배출량(tCO2)",
    "지상역률(%)",
    "진상역률(%)"
]

# --- 피크시간대(단일 표, weekday/weekend 불분리) ---
PEAK_HOURS = [8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20]  # 요청 맥락에서 사용해온 검증 피크
NIGHT_LOW  = {22, 23, 0, 1, 2, 3, 4, 5}

# 시간/카테고리 기반 피처(기본 + 주기성 + 휴일/월초/월말 + 일/월 플래그 + 피크)
TIME_FEATS = [
    "month","day","hour","dow",
    "hour_sin","hour_cos",
    "dow_sin","dow_cos",
    "month_sin","month_cos",
    "is_holiday","is_hol_prev","is_hol_next",
    "is_month_start","is_month_end",
    "is_sunday","is_monday",
    "is_peak",# "near_peak","is_night_low"
]

# 고정 공휴일(요청 명단: 바꾸지 않음)
HOLIDAYS_2024 = pd.to_datetime([
    '2024-01-01','2024-02-14','2024-02-15','2024-02-16','2024-02-17','2024-02-18',
    '2024-03-01','2024-05-01','2024-05-05','2024-05-22','2024-06-06',
    '2024-06-13','2024-08-01','2024-08-02','2024-08-03','2024-08-15',
    '2024-09-22','2024-09-23','2024-09-24','2024-09-25','2024-09-26',
    '2024-10-03','2024-10-09','2024-12-25','2024-12-31'
]).date

# ===================== 유틸: 시간 파서/파생 =====================
def _fix_24h_token(s: str) -> str:
    s = str(s).strip()
    if " 24:" in s:
        try:
            d, t = s.split()
            d0 = pd.to_datetime(d)
            return f"{(d0 + pd.Timedelta(days=1)).date()} 00:{t.split(':',1)[1]}"
        except Exception:
            return s
    return s

def parse_dt_mixed(series: pd.Series) -> pd.Series:
    s = series.astype(str).map(_fix_24h_token)
    x = pd.to_datetime(s, errors="coerce")
    return x.ffill().bfill()

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.copy()
    g[DT_COL] = parse_dt_mixed(g[DT_COL])
    g = g.sort_values(DT_COL).reset_index(drop=True)

    # 숫자 캐스팅(타깃들만)
    for c in S1_TARGETS:
        if c in g.columns:
            g[c] = pd.to_numeric(g[c].astype(str).str.replace(",",""), errors="coerce")

    # 시간 파생
    g["date"]  = g[DT_COL].dt.date
    g["month"] = g[DT_COL].dt.month
    g["day"]   = g[DT_COL].dt.day
    g["hour"]  = g[DT_COL].dt.hour
    g["dow"]   = g[DT_COL].dt.dayofweek  # 0=월, 6=일

    # 주기성
    g["hour_sin"]  = np.sin(2*np.pi*g["hour"]/24);  g["hour_cos"]  = np.cos(2*np.pi*g["hour"]/24)
    g["dow_sin"]   = np.sin(2*np.pi*g["dow"]/7);    g["dow_cos"]   = np.cos(2*np.pi*g["dow"]/7)
    g["month_sin"] = np.sin(2*np.pi*g["month"]/12); g["month_cos"] = np.cos(2*np.pi*g["month"]/12)

    # 휴일/월초/월말/전후일 + 요일 플래그
    g["is_holiday"]     = g["date"].isin(HOLIDAYS_2024).astype(int)
    g["is_hol_prev"]    = (pd.to_datetime(g["date"]) - pd.Timedelta(days=1)).dt.date.isin(HOLIDAYS_2024).astype(int)
    g["is_hol_next"]    = (pd.to_datetime(g["date"]) + pd.Timedelta(days=1)).dt.date.isin(HOLIDAYS_2024).astype(int)
    g["is_month_start"] = g[DT_COL].dt.is_month_start.astype(int)
    g["is_month_end"]   = g[DT_COL].dt.is_month_end.astype(int)
    g["is_sunday"]      = (g["dow"]==6).astype(int)
    g["is_monday"]      = (g["dow"]==0).astype(int)

    # 피크/근접피크/야간저부하
    hour_vals = g["hour"].values
    is_peak = np.isin(hour_vals, PEAK_HOURS).astype(int)
    near_set = set()
    for h in PEAK_HOURS:
        near_set.add((h-1) % 24); near_set.add(h); near_set.add((h+1) % 24)
    near_peak = np.isin(hour_vals, list(near_set)).astype(int)
    is_night = np.isin(hour_vals, list(NIGHT_LOW)).astype(int)
    g["is_peak"] = is_peak
    # g["near_peak"] = near_peak
    # g["is_night_low"] = is_night

    # 작업유형/ID 보강
    if "작업유형" not in g.columns:
        g["작업유형"] = "UNK"
    if ID_COL not in g.columns:
        g[ID_COL] = np.arange(len(g))

    return g

def date_mask(df, start, end):
    s = pd.to_datetime(start); e = pd.to_datetime(end)
    return (df[DT_COL] >= s) & (df[DT_COL] <= e)

# ===================== 라그/롤링 설계 & 생성 =====================
def estimate_step_seconds(dt_series):
    d = dt_series.sort_values().diff().dropna().dt.total_seconds()
    return int(round(d.mode().iloc[0]))

def plan_lag_windows(step_sec):
    steps_per_hour = max(1, int(round(3600/step_sec)))
    steps_per_6h   = steps_per_hour * 6
    steps_per_day  = steps_per_hour * 24
    # 자동창
    auto_L = {1, 2, 4, steps_per_hour, steps_per_6h, steps_per_day}
    auto_R = {3, 6, 12, steps_per_hour, steps_per_6h, steps_per_day}
    # 고정창(요청)
    fixed_L = {2, 3, 6, 12, 24}
    fixed_R = {3, 6, 12, 24}
    LAGS = sorted(auto_L.union(fixed_L))
    RWS  = sorted(auto_R.union(fixed_R))
    return LAGS, RWS

def add_lag_roll_train(df, target, LAGS, RWS):
    """
    타깃 변수에 대해: lag(l) + rolling mean/std/max/min(window) 생성 (훈련/검증용; shift(1)로 누수 차단)
    """
    g = df.copy()
    s = pd.to_numeric(g[target], errors="coerce")

    # Lag
    for l in LAGS:
        g[f"{target}_lag_{l}"] = s.shift(l)

    # Rolling (mean/std/max/min) with shift(1)
    for w in RWS:
        r = s.rolling(window=w, min_periods=1)
        g[f"{target}_rm_{w}"]   = r.mean().shift(1)
        g[f"{target}_rs_{w}"]   = r.std(ddof=1).shift(1)
        g[f"{target}_rmax_{w}"] = r.max().shift(1)
        g[f"{target}_rmin_{w}"] = r.min().shift(1)

    lag_cols = [c for c in g.columns if c.startswith(f"{target}_lag_")
                                   or c.startswith(f"{target}_rm_")
                                   or c.startswith(f"{target}_rs_")
                                   or c.startswith(f"{target}_rmax_")
                                   or c.startswith(f"{target}_rmin_")]
    g[lag_cols] = g[lag_cols].fillna(0.0)
    return g, lag_cols

# ===================== 전처리/OHE =====================
def make_ohe(handle_unknown="ignore"):
    try:
        return OneHotEncoder(handle_unknown=handle_unknown, sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown=handle_unknown, sparse=False)

def make_preprocessor(num_cols, cat_cols):
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median"))])
    ohe = make_ohe()
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", ohe, cat_cols)
    ], remainder="drop")

# ===================== XGB 헬퍼(조용히 학습) =====================
def xgb_fit_silent(model, X_tr, y_tr, X_va, y_va):
    try:
        model.fit(X_tr, y_tr,
                  eval_set=[(X_va, y_va)],
                  eval_metric="mae",
                  early_stopping_rounds=100,
                  verbose=False)
    except TypeError:
        model.fit(X_tr, y_tr,
                  eval_set=[(X_va, y_va)],
                  eval_metric="mae",
                  early_stopping_rounds=100)
    return model

def xgb_predict_best(model, X):
    try:
        if hasattr(model, "best_iteration_") and model.best_iteration_ is not None:
            return model.predict(X, iteration_range=(0, model.best_iteration_ + 1))
        return model.predict(X, ntree_limit=model.best_ntree_limit)
    except Exception:
        return model.predict(X)

# ===================== Stage-1 튜닝(LGB/XGB) =====================
def optimize_stage1_lgb(X_tr, y_tr, X_va, y_va, n_trials=N_TRIALS_S1):
    def objective(trial: optuna.trial.Trial):
        params = {
            "objective": "mae",
            "random_state": RANDOM_STATE,
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 200),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "n_jobs": -1,
            "verbosity": -1
        }
        model = LGBMRegressor(**params)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_va, y_va)],
                  callbacks=[early_stopping(100), log_evaluation(0)])
        pred = model.predict(X_va)
        return mean_absolute_error(y_va, pred)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value

def optimize_stage1_xgb(X_tr, y_tr, X_va, y_va, n_trials=N_TRIALS_S1):
    def objective(trial: optuna.trial.Trial):
        params = {
            "random_state": RANDOM_STATE,
            "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-4, 20.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "tree_method": "hist",
            "n_jobs": -1
        }
        model = xgb.XGBRegressor(**params)
        xgb_fit_silent(model, X_tr, y_tr, X_va, y_va)
        pred = xgb_predict_best(model, X_va)
        return mean_absolute_error(y_va, pred)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value

# ===================== 워크-포워드 예측 =====================
def walkforward_predict_test(model, preproc, base_cols, cat_cols,
                             test_sorted, target, LAGS, RWS, seed_hist):
    """
    테스트 구간을 시간순으로 한 스텝씩 예측.
    - dq(최근값 버퍼)로 라그/롤링(평균/표준편차/최대/최소) 생성
    - 역률은 [0,100], 나머지는 하한 0으로 클램핑
    """
    maxlen = max(max(LAGS), max(RWS))
    dq = deque(seed_hist, maxlen=maxlen)
    preds = []

    for _, row in test_sorted.iterrows():
        feat = row[base_cols + cat_cols].to_frame().T.copy()

        # lag
        for l in LAGS:
            feat[f"{target}_lag_{l}"] = dq[-l] if len(dq) >= l else 0.0

        # rolling
        arr = np.array(dq, dtype=float)
        for w in RWS:
            if len(arr) == 0:
                m = s = x = n = 0.0
            else:
                m = float(np.mean(arr[-w:]))
                s = float(np.std(arr[-w:], ddof=1)) if len(arr) >= 2 else 0.0
                x = float(np.max(arr[-w:]))
                n = float(np.min(arr[-w:]))

            feat[f"{target}_rm_{w}"]   = m
            feat[f"{target}_rs_{w}"]   = s
            feat[f"{target}_rmax_{w}"] = x
            feat[f"{target}_rmin_{w}"] = n

        X = preproc.transform(feat)
        yhat = float(model.predict(X)[0])

        # 클램핑
        if target in ("지상역률(%)", "진상역률(%)"):
            yhat = min(max(yhat, 0.0), 100.0)
        else:
            yhat = max(yhat, 0.0)

        preds.append(yhat)
        dq.append(yhat)

    return np.array(preds)

# ===================== 메인(Stage-1만) =====================
def main():
    # 1) 로드 & 파생
    train0 = pd.read_csv(DATA_TRAIN)
    test0  = pd.read_csv(DATA_TEST)
    train_b = add_basic_features(train0)
    test_b  = add_basic_features(test0)

    # 2) 기간 분할
    tr_mask = date_mask(train_b, TRAIN_START, TRAIN_END)
    va_mask = date_mask(train_b, VALID_START, VALID_END)
    assert tr_mask.any() and va_mask.any(), "기간 설정을 확인하세요."

    # 3) 스텝 간격 추정 → 라그/롤링 창 설계
    step_sec = estimate_step_seconds(train_b[DT_COL])
    LAGS, RWS = plan_lag_windows(step_sec)

    # 4) 테스트 시간순 정렬
    test_sorted = test_b.sort_values(DT_COL).reset_index(drop=True)

    # 5) 타깃별 학습/튜닝 및 예측
    preds_dict = {}
    report = {}

    for tgt in S1_TARGETS:
        if tgt not in train_b.columns:
            continue

        # 라그/롤링 추가(훈련/검증) — shift(1)로 누수 차단
        train_aug, lag_cols = add_lag_roll_train(train_b, tgt, LAGS, RWS)

        num_cols = TIME_FEATS + lag_cols
        base_cols = TIME_FEATS.copy()
        cat_cols  = CAT_FEATS.copy()

        # 전처리(훈련/검증)
        pre = make_preprocessor(num_cols=num_cols, cat_cols=cat_cols)
        X_tr0 = train_aug.loc[tr_mask, num_cols + cat_cols]
        X_va0 = train_aug.loc[va_mask, num_cols + cat_cols]
        y_tr  = train_aug.loc[tr_mask, tgt].values
        y_va  = train_aug.loc[va_mask, tgt].values
        pre.fit(X_tr0)
        X_tr = pre.transform(X_tr0)
        X_va = pre.transform(X_va0)

        # 튜닝 + 전구간 리핏
        if STAGE1_BASE == "xgb":
            best_params, best_mae = optimize_stage1_xgb(X_tr, y_tr, X_va, y_va, n_trials=N_TRIALS_S1)
            final = xgb.XGBRegressor(**{**best_params, "random_state":RANDOM_STATE, "n_jobs":-1, "tree_method":"hist"})
        else:
            best_params, best_mae = optimize_stage1_lgb(X_tr, y_tr, X_va, y_va, n_trials=N_TRIALS_S1)
            final = LGBMRegressor(**{**best_params, "objective":"mae", "random_state":RANDOM_STATE, "n_jobs":-1, "verbosity":-1})

        # 전훈련 리핏(라그 포함 전체)
        pre_full = make_preprocessor(num_cols=num_cols, cat_cols=cat_cols)
        X_all0 = train_aug[num_cols + cat_cols]
        pre_full.fit(X_all0)
        X_full = pre_full.transform(X_all0)
        final.fit(X_full, train_aug[tgt].values)

        # 검증 MAE 출력(최적 스코어)
        print(f"[Stage-1] {tgt}: valid MAE = {best_mae:.4f}")
        report[tgt] = best_mae

        # 테스트 워크-포워드 예측
        max_win = max(max(LAGS), max(RWS))
        seed_hist = train_b[tgt].tail(max_win).tolist()

        pred_te = walkforward_predict_test(
            final, pre_full, base_cols, cat_cols, test_sorted, tgt, LAGS, RWS, seed_hist
        )
        preds_dict[tgt] = pred_te

    # 6) 저장: test.csv 원래 컬럼 보존 + 오른쪽에 타깃명 그대로 붙이기
    out = test0.copy()  # 원본 test 구조 유지
    for tgt, arr in preds_dict.items():
        out[tgt] = arr
    out.to_csv("stage1_test_predictions.csv", index=False, encoding="utf-8-sig")
    print("\nSaved: stage1_test_predictions.csv")

    # 7) 요약
    print("\n[Stage-1 Validation MAE by target]")
    for k, v in report.items():
        print(f"- {k}: {v:.4f}")

if __name__ == "__main__":
    main()
