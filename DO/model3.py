# -*- coding: utf-8 -*-
import warnings, os, re
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor

try:
    import optuna
except Exception:
    optuna = None

# =========================
# 0) 사용자 설정
# =========================
TRAIN_PATH = "./data/raw/train.csv"
TEST_PATH  = "./data/raw/test.csv"

STAGE1_PRED_PATH       = "./stage1_test_predictions.csv"   # Test 예측 저장(원 컬럼명만 출력)
STAGE1_VAL_METRICS_CSV = "./stage1_valid_metrics_nov.csv"  # 11월 성능 저장

DT_CANDS = ["측정일시","date","datetime","timestamp"]

# Stage-1 타깃(표준명)
STAGE1_TARGETS = [
    "전력사용량(kWh)",
    "지상무효전력량(kvarh)",
    "진상무효전력량(kvarh)",
    "지상역률(%)",
    "진상역률(%)",
    "탄소배출량(tCO2)",
]
PF_TARGETS = {"지상역률(%)", "진상역률(%)"}  # 0~100으로 클리핑

# 학습/검증 달(홀드아웃=11월로 ‘러프 튜닝’)
TRAIN_MONTHS = list(range(3, 11))   # 3~10월 = 학습 (튜닝 및 재학습용)
VALID_MONTHS = [11]                 # 11월  = 홀드아웃(튜닝용)

# 튜닝/고정 하이퍼파라미터
USE_OPTUNA   = True
N_TRIALS     = 20
RANDOM_STATE = 42

# 재학습 범위
REFIT_USE_VALID = True

# 공휴일(±1일 포함, 2024)
HOLIDAYS_2024 = pd.to_datetime([
    '2024-01-01','2024-02-14','2024-02-15','2024-02-16','2024-02-17','2024-02-18',
    '2024-03-01','2024-05-01','2024-05-05','2024-05-22','2024-06-06',
    '2024-06-13','2024-08-01','2024-08-02','2024-08-03','2024-08-15',
    '2024-09-22','2024-09-23','2024-09-24','2024-09-25','2024-09-26',
    '2024-10-03','2024-10-09','2024-12-25','2024-12-31'
]).normalize()
HOLIDAY_WINDOW_2024 = (
    set(HOLIDAYS_2024)
    | {d + pd.Timedelta(days=1) for d in HOLIDAYS_2024}
    | {d - pd.Timedelta(days=1) for d in HOLIDAYS_2024}
)

# 피크 시간대(예시)
VERIFIED_PEAK_HOURS = [8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20]

# =========================
# 1) 유틸
# =========================
def read_csv_smart(path):
    for enc in ("cp949","utf-8-sig","utf-8"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)

def find_dt_col(df):
    for c in DT_CANDS:
        if c in df.columns:
            return c
    raise KeyError(f"날짜 컬럼을 찾을 수 없습니다. 후보={DT_CANDS}")

def normalize_name(s: str) -> str:
    if s is None: return ""
    s = s.lower().replace(" ", "")
    s = s.replace("[", "(").replace("]", ")").replace("％", "%")
    s = re.sub(r"[^0-9a-z가-힣()%]", "", s)
    s = s.replace("kw h","kwh").replace("kvar h","kvarh")
    return s

ALIASES = {
    "전력사용량(kWh)": ["전력사용량(kwh)", "전력사용량", "kwh", "usage(kwh)", "usagekwh", "전력사용량kwh"],
    "지상무효전력량(kvarh)": ["지상무효전력량(kvarh)", "지상무효전력량", "lagkvarh", "kvarh_lag", "lag(kvarh)"],
    "진상무효전력량(kvarh)": ["진상무효전력량(kvarh)", "진상무효전력량", "leadkvarh", "kvarh_lead", "lead(kvarh)"],
    "지상역률(%)": ["지상역률(%)", "지상역률", "lagpf(%)", "lagpf", "pf_lag"],
    "진상역률(%)": ["진상역률(%)", "진상역률", "leadpf(%)", "leadpf", "pf_lead"],
    "탄소배출량(tCO2)": ["탄소배출량(tco2)", "탄소배출량", "tco2", "co2(tco2)"],
}

def resolve_column(df, canonical_key: str):
    norm_map = {normalize_name(c): c for c in df.columns}
    std_norm = normalize_name(canonical_key)
    if std_norm in norm_map:
        return norm_map[std_norm]
    for alias in ALIASES.get(canonical_key, []):
        n = normalize_name(alias)
        if n in norm_map:
            return norm_map[n]
    for alias in ALIASES.get(canonical_key, []):
        n = normalize_name(alias)
        cand = [norm_map[k] for k in norm_map.keys() if n in k]
        if cand:
            return cand[0]
    return None

def rmse_safe(y_true, y_pred):
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def clip_pred_by_target(tgt_name: str, yhat: np.ndarray) -> np.ndarray:
    """요청 제약: 전체 ≥0, 역률은 0~100%."""
    if tgt_name in PF_TARGETS or ("역률" in tgt_name):
        return np.clip(yhat, 0.0, 100.0)
    return np.clip(yhat, 0.0, None)

# =========================
# 2) 캘린더(날짜/작업유형 기반만 사용)
# =========================
def month_to_season(m):
    if m in (12, 1, 2):  return "winter"
    if m in (3, 4, 5):   return "spring"
    if m in (6, 7, 8):   return "summer"
    return "autumn"

def add_calendar_features(df, dt_col):
    out = df.copy()
    out[dt_col] = pd.to_datetime(out[dt_col], errors="coerce")
    valid = out[dt_col].notna()
    out = out.sort_values(dt_col)

    # 작업유형 보정(없으면 공백)
    if "작업유형" not in out.columns:
        out["작업유형"] = ""

    t = out[dt_col]
    out["year"]   = np.where(valid, t.dt.year, np.nan)
    out["month"]  = np.where(valid, t.dt.month, np.nan)
    out["day"]    = np.where(valid, t.dt.day, np.nan)
    out["hour"]   = np.where(valid, t.dt.hour, np.nan)
    out["minute"] = np.where(valid, t.dt.minute, np.nan)
    out["dow"]    = np.where(valid, t.dt.dayofweek, np.nan)

    # Daily/Weekly Fourier
    minutes_in_day = (t.dt.hour*60 + t.dt.minute).astype(float)
    out["sin_day"] = np.where(valid, np.sin(2*np.pi*minutes_in_day/1440), np.nan)
    out["cos_day"] = np.where(valid, np.cos(2*np.pi*minutes_in_day/1440), np.nan)
    out["sin_dow"] = np.where(valid, np.sin(2*np.pi*t.dt.dayofweek/7), np.nan)
    out["cos_dow"] = np.where(valid, np.cos(2*np.pi*t.dt.dayofweek/7), np.nan)

    # 요일/주말 플래그
    out["is_weekend"] = np.where(valid, (t.dt.dayofweek>=5).astype(int), np.nan)
    out["is_monday"]  = np.where(valid, (t.dt.dayofweek==0).astype(int), np.nan)
    out["is_sunday"]  = np.where(valid, (t.dt.dayofweek==6).astype(int), np.nan)

    # 피크/비피크
    out["is_peak"]    = np.where(valid, t.dt.hour.isin(VERIFIED_PEAK_HOURS).astype(int), np.nan)
    out["is_offpeak"] = np.where(valid, 1 - t.dt.hour.isin(VERIFIED_PEAK_HOURS).astype(int), np.nan)

    # 공휴일(±1일)
    dnorm = t.dt.normalize()
    out["is_holiday"] = np.where(valid, dnorm.isin(HOLIDAY_WINDOW_2024).astype(int), np.nan)

    # 계절
    season = np.where(valid, t.dt.month.map(month_to_season), "none")
    out["season"] = season
    out["is_summer"] = np.where(valid, (out["season"]=="summer").astype(int), np.nan)
    out["is_winter"] = np.where(valid, (out["season"]=="winter").astype(int), np.nan)

    out["dt_invalid"] = (~valid).astype(int)

    # 문자열 범주 준비
    out["작업유형"] = out["작업유형"].fillna("").astype(str)
    out["season"]   = out["season"].astype(str)
    return out

# =========================
# 3) 11월 홀드아웃 튜닝(+재학습)
# =========================
def holdout_tune_params(X_train_df, y_train, X_hold_df, y_hold, cat_cols, use_optuna=USE_OPTUNA, n_trials=N_TRIALS):
    num_cols = [c for c in X_train_df.columns if c not in cat_cols]

    def eval_params(params):
        pre = ColumnTransformer(
            transformers=[
                ("num", SimpleImputer(strategy="median"), num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ],
            remainder="drop"
        )
        Xtr_t = pre.fit_transform(X_train_df)
        Xho_t = pre.transform(X_hold_df)

        mdl = LGBMRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1)
        mdl.fit(Xtr_t, y_train)
        pred = mdl.predict(Xho_t)
        pred = clip_pred_by_target(current_target_name, pred)  # ← 홀드아웃도 제약 반영
        return float(mean_absolute_error(y_hold, pred))

    # Optuna 대상 함수에서 현재 타깃명 참조할 수 있도록 전역 변수 사용
    global current_target_name

    if use_optuna and (optuna is not None):
        def objective(trial):
            params = dict(
                n_estimators      = trial.suggest_int("n_estimators", 400, 1400),
                learning_rate     = trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
                num_leaves        = trial.suggest_int("num_leaves", 31, 255),
                max_depth         = trial.suggest_int("max_depth", 3, 12),
                min_child_samples = trial.suggest_int("min_child_samples", 10, 200),
                subsample         = trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree  = trial.suggest_float("colsample_bytree", 0.6, 1.0),
                reg_alpha         = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                reg_lambda        = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            )
            return eval_params(params)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params = study.best_params
        best_score  = study.best_value
    else:
        best_params = dict(
            n_estimators=900, learning_rate=0.05, num_leaves=63,
            subsample=0.9, colsample_bytree=0.9,
            reg_alpha=0.0, reg_lambda=1.0, max_depth=8, min_child_samples=50
        )
        best_score = eval_params(best_params)

    print(f"[Holdout-Nov TUNE] {current_target_name} best MAE={best_score:.5f} | params={best_params}")
    return best_params, best_score

# =========================
# 4) 메인(Stage-1) — Holdout 튜닝 + 재학습 + Test 예측(제약/출력형식 적용)
# =========================
def run_stage1_only():
    # ----- 데이터 로드 -----
    tr_raw = read_csv_smart(TRAIN_PATH)
    te_raw = read_csv_smart(TEST_PATH)

    # test에 작업유형 없으면 빈 문자열로 생성
    if "작업유형" not in tr_raw.columns:
        tr_raw["작업유형"] = ""
    if "작업유형" not in te_raw.columns:
        te_raw["작업유형"] = ""

    if "id" not in te_raw.columns:
        te_raw = te_raw.copy()
        te_raw["id"] = np.arange(len(te_raw))
        print("[알림] test.csv에 'id'가 없어 임시 id(0..n-1)를 생성했습니다.")

    te_raw["_orig_order"] = np.arange(len(te_raw))

    # 날짜 컬럼
    dt_col = find_dt_col(tr_raw)
    tr_raw[dt_col] = pd.to_datetime(tr_raw[dt_col], errors="coerce")
    te_raw[dt_col] = pd.to_datetime(te_raw[dt_col], errors="coerce")

    tr = tr_raw.dropna(subset=[dt_col]).sort_values(dt_col).reset_index(drop=True)
    te = te_raw.sort_values(dt_col).reset_index(drop=True)

    # ----- (날짜/작업유형 기반) 공통 파생 -----
    tr_base = add_calendar_features(tr, dt_col)
    te_base = add_calendar_features(te, dt_col)

    # align
    all_cols = sorted(set(tr_base.columns) | set(te_base.columns))
    tr_base = tr_base.reindex(columns=all_cols)
    te_base = te_base.reindex(columns=all_cols)

    # 합치기 + 플래그
    tr_base["is_test"] = 0
    te_base["is_test"] = 1
    base = pd.concat([tr_base, te_base], axis=0, ignore_index=True).sort_values(dt_col).reset_index(drop=True)

    # 마스크
    base["month"] = base[dt_col].dt.month
    is_test    = base["is_test"].astype(bool)
    train_mask = (~is_test) & (base["month"].isin(TRAIN_MONTHS))
    valid_mask = (~is_test) & (base["month"].isin(VALID_MONTHS))  # 11월
    test_mask  = is_test

    print(f"[Rows] train={train_mask.sum()}, valid(holdout)={valid_mask.sum()}, test={test_mask.sum()}")

    # ===== 리포트 저장용 =====
    val_metrics = []

    # 공통 피처 구성(숫자/범주)
    base_num_cols = [
        "year","month","day","hour","minute","dow",
        "sin_day","cos_day","sin_dow","cos_dow",
        "is_weekend","is_monday","is_sunday",
        "is_holiday","is_peak","is_offpeak",
        "is_summer","is_winter","dt_invalid",
    ]
    cat_cols  = ["season","작업유형"]

    # ===== Stage-1: 타깃별 학습/예측 =====
    global current_target_name
    preds_by_target = {}  # test 예측 보관(원 컬럼명으로)

    for tgt_std in STAGE1_TARGETS:
        current_target_name = tgt_std  # 튜닝 시 참조
        tgt_col = resolve_column(base, tgt_std) or tgt_std
        if tgt_col not in base.columns:
            base[tgt_col] = np.nan
            print(f"[경고] 타깃 '{tgt_std}'(실제:'{tgt_col}') 컬럼을 찾지 못했습니다. NaN 처리.")

        # 분할
        feat_cols = [c for c in base_num_cols if c in base.columns]
        X_all = base[feat_cols + cat_cols].copy()
        y_all = pd.to_numeric(base[tgt_col], errors="coerce")

        X_train, y_train = X_all.loc[train_mask], y_all.loc[train_mask]
        X_hold,  y_hold  = X_all.loc[valid_mask], y_all.loc[valid_mask]
        X_test           = X_all.loc[test_mask]

        tr_idx = y_train.dropna().index
        ho_idx = y_hold.dropna().index
        X_train, y_train = X_train.loc[tr_idx], y_train.loc[tr_idx]
        X_hold,  y_hold  = X_hold.loc[ho_idx], y_hold.loc[ho_idx]

        if len(X_train) == 0 or len(X_hold) == 0:
            print(f"[Stage-1] {tgt_std}: 학습/홀드아웃 샘플 부족 → 스킵")
            preds_by_target[tgt_std] = np.full(X_test.shape[0], np.nan)
            continue

        # ---- (1) 11월 홀드아웃으로 러프 튜닝 ----
        best_params, ho_mae = holdout_tune_params(
            X_train_df=X_train, y_train=y_train,
            X_hold_df=X_hold,   y_hold=y_hold,
            cat_cols=cat_cols, use_optuna=USE_OPTUNA, n_trials=N_TRIALS
        )

        # ---- (2) 최종 재학습 범위 결정 ----
        if REFIT_USE_VALID and len(X_hold) > 0:
            X_refit = pd.concat([X_train, X_hold], axis=0).sort_index()
            y_refit = pd.concat([y_train, y_hold], axis=0).sort_index()
        else:
            X_refit, y_refit = X_train, y_train

        # 전처리(최종) + 모델(최적파라미터) 재학습
        num_cols = [c for c in X_refit.columns if c not in cat_cols]
        pre_full = ColumnTransformer(
            transformers=[
                ("num", SimpleImputer(strategy="median"), num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
            ],
            remainder="drop"
        )
        Xref_t = pre_full.fit_transform(X_refit)
        mdl = LGBMRegressor(**best_params, random_state=RANDOM_STATE, n_jobs=-1)
        mdl.fit(Xref_t, y_refit)

        # ---- (3) 11월 홀드아웃 성능(제약 반영해서 평가)
        if len(X_hold) > 0:
            Xho_t = pre_full.transform(X_hold)
            pred_ho = mdl.predict(Xho_t)
            pred_ho = clip_pred_by_target(tgt_std, pred_ho)  # 0~ / 0~100
            mae = mean_absolute_error(y_hold, pred_ho)
            rmse = rmse_safe(y_hold, pred_ho)
            print(f"[HOLDOUT-Nov][Stage-1] {tgt_std}  MAE={mae:.4f}  RMSE={rmse:.4f}")
            val_metrics.append({"target": tgt_std, "MAE": float(mae), "RMSE": float(rmse),
                                "CV_MAE": float(np.nan), "n_valid": int(len(y_hold))})

        # ---- (4) 테스트 예측(외생변수만) + 제약 반영
        Xtst_t = pre_full.transform(X_test)
        test_pred = mdl.predict(Xtst_t)
        test_pred = clip_pred_by_target(tgt_std, test_pred)  # 0~ / 0~100
        preds_by_target[tgt_std] = test_pred

    # ===== Test 예측 파일 저장: 원 컬럼명만 출력 =====
    out = te.copy()
    keep_cols = []
    if "id" in out.columns:
        keep_cols.append("id")
    keep_cols += [c for c in [DT_CANDS[0] if DT_CANDS[0] in out.columns else None, "작업유형"] if c in out.columns]
    out = out[keep_cols].copy()

    # 각 타깃의 예측을 "원 컬럼명"으로만 추가 (_pred_s1 미포함)
    for tgt_std in STAGE1_TARGETS:
        if tgt_std in preds_by_target:
            out[tgt_std] = preds_by_target[tgt_std]
        else:
            # 예측 못했으면 NaN 유지
            out[tgt_std] = np.nan

    # 정렬 및 저장
    out = out.reset_index(drop=True)
    out.to_csv(STAGE1_PRED_PATH, index=False, encoding="utf-8-sig")
    print(f"[DONE] Saved Stage-1 test predictions (original column names only): {STAGE1_PRED_PATH}  (rows={len(out)})")

    # ===== 11월 홀드아웃 성능 저장 =====
    if len(val_metrics) > 0:
        mdf = pd.DataFrame(val_metrics).sort_values("target")
        mdf.to_csv(STAGE1_VAL_METRICS_CSV, index=False, encoding="utf-8-sig")
        print(f"[DONE] Saved Stage-1 Nov metrics: {STAGE1_VAL_METRICS_CSV}")
    else:
        print("[INFO] 11월 홀드아웃 성능이 없습니다.")

    return dict(
        stage1_pred_path=STAGE1_PRED_PATH,
        stage1_val_metrics_path=STAGE1_VAL_METRICS_CSV
    )

# =========================
# 실행
# =========================
if __name__ == "__main__":
    artifacts = run_stage1_only()
