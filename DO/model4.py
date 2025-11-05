# -*- coding: utf-8 -*-
import warnings, re
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMRegressor

try:
    import optuna
except Exception:
    optuna = None

# =========================
# 0) 경로/설정
# =========================
TRAIN_PATH         = "./data/raw/train.csv"           # 학습(실측)
STAGE1_PRED_PATH   = "./stage1_test_predictions.csv"  # ★ 1단계 예측 포함 테스트 입력
SUBMISSION_PATH    = "./submission_stage2.csv"        # Stage-2 결과 제출 파일
TEST_FEATURES_PATH = "./test_with_features.csv"       # 테스트에 사용한 전 피처를 붙인 파일
TRAIN_FEATURES_PATH= "./train_with_features.csv"      # ★ 학습 데이터에도 사용 피처를 붙인 파일

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

# Stage-2 최종 타깃
STAGE2_TARGET = "전기요금(원)"

# 학습 달(튜닝용 CV 구간): 3~10월
TRAIN_MONTHS = list(range(3, 11))
# (선택) 재학습 시 11월 포함 여부
REFIT_INCLUDE_VALID_MONTHS = True  # True면 3~11월로 재학습

# 모델/튜닝
USE_OPTUNA   = True
N_TRIALS     = 3
RANDOM_STATE = 42

# TS-CV 설정
TS_N_SPLITS  = 5

# 피크/공휴일(±1일)
VERIFIED_PEAK_HOURS = [8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20]
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

# =========================
# 1) 유틸/정규화/매핑
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
        if c in df.columns: return c
    raise KeyError(f"날짜 컬럼을 찾을 수 없습니다. 후보={DT_CANDS}")

def normalize_name(s: str) -> str:
    if s is None: return ""
    s = s.lower()
    s = s.replace(" ", "")
    s = s.replace("[", "(").replace("]", ")")
    s = s.replace("％", "%")
    s = re.sub(r"[^0-9a-z가-힣()%]", "", s)
    return s

ALIASES = {
    # Stage-1 표준 타깃
    "전력사용량(kWh)": ["전력사용량(kwh)","전력사용량","kwh","usagekwh","usage(kwh)","전력사용량kwh"],
    "지상무효전력량(kvarh)": ["지상무효전력량(kvarh)","지상무효전력량","lagkvarh","kvarh_lag"],
    "진상무효전력량(kvarh)": ["진상무효전력량(kvarh)","진상무효전력량","leadkvarh","kvarh_lead"],
    "지상역률(%)": ["지상역률(%)","지상역률","lagpf(%)","lagpf","pf_lag"],
    "진상역률(%)": ["진상역률(%)","진상역률","leadpf(%)","leadpf","pf_lead"],
    "탄소배출량(tCO2)": ["탄소배출량(tco2)","탄소배출량","tco2","co2(tco2)"],

    # Stage-2 타깃
    "전기요금(원)": ["전기요금(원)","전기요금","요금","cost","price"],

    # PF 파생 입력
    "_PF_LAG":  ["지상역률(%)","지상역률","lagpf(%)","lagpf","pf_lag"],
    "_PF_LEAD": ["진상역률(%)","진상역률","leadpf(%)","leadpf","pf_lead"],
    "_Q_LAG":   ["지상무효전력량(kvarh)","지상무효전력량","lagkvarh","kvarh_lag"],
    "_Q_LEAD":  ["진상무효전력량(kvarh)","진상무효전력량","leadkvarh","kvarh_lead"],
}

def resolve_column(df, canonical_key: str):
    norm_map = {normalize_name(c): c for c in df.columns}
    std_norm = normalize_name(canonical_key)
    if std_norm in norm_map:
        return norm_map[std_norm]
    for alias in ALIASES.get(canonical_key, []):
        n = normalize_name(alias)
        if n in norm_map: return norm_map[n]
    for alias in ALIASES.get(canonical_key, []):
        n = normalize_name(alias)
        cand = [norm_map[k] for k in norm_map.keys() if n in k]
        if cand: return cand[0]
    return None

def rmse_safe(y_true, y_pred):
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

# =========================
# 2) 캘린더/역률 파생
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

    t = out[dt_col]
    out["year"]   = np.where(valid, t.dt.year, np.nan)
    out["month"]  = np.where(valid, t.dt.month, np.nan)
    out["day"]    = np.where(valid, t.dt.day, np.nan)
    out["hour"]   = np.where(valid, t.dt.hour, np.nan)
    out["minute"] = np.where(valid, t.dt.minute, np.nan)
    out["dow"]    = np.where(valid, t.dt.dayofweek, np.nan)

    minutes_in_day = (t.dt.hour*60 + t.dt.minute).astype(float)
    out["sin_day"] = np.where(valid, np.sin(2*np.pi*minutes_in_day/1440), np.nan)
    out["cos_day"] = np.where(valid, np.cos(2*np.pi*minutes_in_day/1440), np.nan)
    out["sin_dow"] = np.where(valid, np.sin(2*np.pi*t.dt.dayofweek/7), np.nan)
    out["cos_dow"] = np.where(valid, np.cos(2*np.pi*t.dt.dayofweek/7), np.nan)

    out["is_weekend"] = np.where(valid, (t.dt.dayofweek>=5).astype(int), np.nan)
    out["is_monday"]  = np.where(valid, (t.dt.dayofweek==0).astype(int), np.nan)
    out["is_sunday"]  = np.where(valid, (t.dt.dayofweek==6).astype(int), np.nan)

    out["is_peak"]    = np.where(valid, t.dt.hour.isin(VERIFIED_PEAK_HOURS).astype(int), np.nan)
    out["is_offpeak"] = np.where(valid, 1 - t.dt.hour.isin(VERIFIED_PEAK_HOURS).astype(int), np.nan)

    dnorm = t.dt.normalize()
    out["is_holiday"] = np.where(valid, dnorm.isin(HOLIDAY_WINDOW_2024).astype(int), np.nan)

    season = np.where(valid, t.dt.month.map(month_to_season), "none")
    out["season"] = season
    out["is_summer"] = np.where(valid, (out["season"]=="summer").astype(int), np.nan)
    out["is_winter"] = np.where(valid, (out["season"]=="winter").astype(int), np.nan)

    out["dt_invalid"] = (~valid).astype(int)
    return out

def add_pf_features_with_resolve(df, dt_col):
    out = df.copy()
    out[dt_col] = pd.to_datetime(out[dt_col], errors="coerce")
    out = out.sort_values(dt_col)

    PF_LAG  = resolve_column(out, "_PF_LAG")
    PF_LEAD = resolve_column(out, "_PF_LEAD")
    Q_LAG   = resolve_column(out, "_Q_LAG")
    Q_LEAD  = resolve_column(out, "_Q_LEAD")

    def _num_series(colname, clip_low=None, clip_high=None, pct=False):
        if (colname is not None) and (colname in out.columns):
            s = pd.to_numeric(out[colname], errors="coerce")
        else:
            s = pd.Series(np.nan, index=out.index, dtype=float)
        if pct:
            s = s.clip(lower=0.0, upper=100.0)
        else:
            lo = -np.inf if clip_low is None else clip_low
            hi =  np.inf if clip_high is None else clip_high
            s = s.clip(lower=lo, upper=hi)
        return s.fillna(0.0)

    pf_lag  = _num_series(PF_LAG,  pct=True)
    pf_lead = _num_series(PF_LEAD, pct=True)
    q_lag   = _num_series(Q_LAG,   clip_low=0.0)
    q_lead  = _num_series(Q_LEAD,  clip_low=0.0)

    pf_mag = np.maximum(pf_lag, pf_lead) / 100.0
    side = np.where(pf_lead > pf_lag, "lead",
            np.where(pf_lag > pf_lead, "lag", "none"))
    sign = np.where(side=="lead", 1.0, np.where(side=="lag", -1.0, 0.0))

    out["pf_mag"] = pf_mag
    out["pf_side"] = side
    out["pf_signed"] = pf_mag * sign
    out["pf_dev_from_unity"] = 1.0 - np.clip(pf_mag, 0.0, 1.0)
    out["pf_angle_rad"] = np.arccos(np.clip(pf_mag, 0.0, 1.0)) * sign

    out["q_signed"] = q_lead - q_lag
    q_sum = q_lead + q_lag
    out["q_ratio"] = out["q_signed"] / (q_sum + 1e-9)
    out["q_side"]  = np.where(out["q_signed"]>0,"lead", np.where(out["q_signed"]<0,"lag","none"))

    out["pf_below90_flag"] = (pf_mag < 0.90).astype(int)

    print("[컬럼 매핑] PF_LAG:", PF_LAG, "| PF_LEAD:", PF_LEAD, "| Q_LAG:", Q_LAG, "| Q_LEAD:", Q_LEAD)
    return out

# =========================
# 3) TS-CV 튜닝 도우미
# =========================
def tscv_tune_lgbm(X_df, y_ser, cat_cols,
                   use_optuna=USE_OPTUNA, n_trials=N_TRIALS, n_splits=TS_N_SPLITS):
    num_cols = [c for c in X_df.columns if c not in cat_cols]
    tscv = TimeSeriesSplit(n_splits=n_splits)

    def eval_params(params):
        fold_maes = []
        for tr_idx, va_idx in tscv.split(X_df, y_ser):
            X_tr_f, X_va_f = X_df.iloc[tr_idx], X_df.iloc[va_idx]
            y_tr_f, y_va_f = y_ser.iloc[tr_idx], y_ser.iloc[va_idx]

            pre_f = ColumnTransformer(
                transformers=[
                    ("num", SimpleImputer(strategy="median"), num_cols),
                    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                ],
                remainder="drop"
            )
            Xtr_t = pre_f.fit_transform(X_tr_f)
            Xva_t = pre_f.transform(X_va_f)

            mdl = LGBMRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)
            mdl.fit(Xtr_t, y_tr_f)
            pred = mdl.predict(Xva_t)
            fold_maes.append(mean_absolute_error(y_va_f, pred))
        return float(np.mean(fold_maes))

    if use_optuna and (optuna is not None):
        def objective(trial):
            params = dict(
                n_estimators      = trial.suggest_int("n_estimators", 400, 1800),
                learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
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
        cv_mae = study.best_value
    else:
        best_params = dict(
            n_estimators=1200, learning_rate=0.05, num_leaves=63,
            subsample=0.9, colsample_bytree=0.9,
            reg_alpha=0.1, reg_lambda=1.0
        )
        cv_mae = eval_params(best_params)

    print(f"[Stage-2][TS-CV] best CV-MAE={cv_mae:.6f} | params={best_params}")
    return best_params, cv_mae

# =========================
# 4) 2단계 (TS-CV + 재학습 + Test/Train 피처 덤프)
# =========================
def run_stage2_tscv_refit():
    # ----- 데이터 로드 -----
    tr  = read_csv_smart(TRAIN_PATH)
    te1 = read_csv_smart(STAGE1_PRED_PATH)  # ★ 1단계 예측 포함 테스트 파일

    if "id" not in te1.columns:
        te1 = te1.copy()
        te1["id"] = np.arange(len(te1))
        print("[알림] stage1_test_predictions.csv에 'id'가 없어 임시 id(0..n-1)를 생성했습니다.")

    # 날짜
    dt_col = find_dt_col(tr)
    tr[dt_col]  = pd.to_datetime(tr[dt_col], errors="coerce")
    te1[dt_col] = pd.to_datetime(te1[dt_col], errors="coerce")
    tr  = tr.dropna(subset=[dt_col]).sort_values(dt_col).reset_index(drop=True)
    te1 = te1.sort_values(dt_col).reset_index(drop=True)

    # ----- 공통 파생 -----
    tr_base = add_calendar_features(tr, dt_col)
    tr_base = add_pf_features_with_resolve(tr_base, dt_col)

    te_base = add_calendar_features(te1, dt_col)
    te_base = add_pf_features_with_resolve(te_base, dt_col)

    # align & 합치기
    all_cols = sorted(set(tr_base.columns) | set(te_base.columns))
    tr_base = tr_base.reindex(columns=all_cols)
    te_base = te_base.reindex(columns=all_cols)
    tr_base["is_test"] = 0
    te_base["is_test"] = 1
    base = pd.concat([tr_base, te_base], axis=0, ignore_index=True)\
             .sort_values(dt_col).reset_index(drop=True)

    # 마스크
    base["month"]  = base[dt_col].dt.month
    is_test        = base["is_test"].astype(bool)
    train_cv_mask  = (~is_test) & (base["month"].isin(TRAIN_MONTHS))           # 3~10월: CV용
    if REFIT_INCLUDE_VALID_MONTHS:
        refit_mask = (~is_test) & (base["month"].isin(TRAIN_MONTHS + [11]))    # 3~11월: 재학습용
    else:
        refit_mask = train_cv_mask                                             # 3~10월: 재학습용
    test_mask      = is_test

    for c in ["pf_side","q_side","season"]:
        if c not in base.columns: base[c] = "none"

    # ----- 피처 구성 (lag/rolling 제외) -----
    s2_base_num = [
        "year","month","day","hour","minute","dow",
        "sin_day","cos_day","sin_dow","cos_dow",
        "is_weekend","is_monday","is_sunday",
        "is_holiday","is_peak","is_offpeak",
        "is_summer","is_winter","dt_invalid",
        "pf_mag","pf_signed","pf_dev_from_unity","pf_angle_rad",
        "q_ratio","pf_below90_flag"
    ]
    cat_cols = ["pf_side","q_side","season"]

    base2 = base.copy()
    s2_feat_cols = [c for c in s2_base_num if c in base2.columns]

    # Stage-1 타깃을 2단계 입력으로: 학습=실측, 테스트=Stage-1 예측(원 컬럼명으로 가정)
    for tstd in STAGE1_TARGETS:
        t_train_col = resolve_column(base2, tstd) or tstd
        if t_train_col not in base2.columns:
            base2[t_train_col] = np.nan

        if tstd not in te1.columns:
            te1[tstd] = np.nan

        newc = f"{tstd}_s2feat"
        base2[newc] = np.nan
        base2.loc[~is_test, newc] = pd.to_numeric(base2.loc[~is_test, t_train_col], errors="coerce").values
        base2.loc[test_mask,  newc] = pd.to_numeric(te1.loc[:, tstd], errors="coerce").values

        s2_feat_cols.append(newc)

    # ----- X/y 구성 -----
    y_col = resolve_column(base2, STAGE2_TARGET) or STAGE2_TARGET
    if y_col not in base2.columns:
        raise KeyError(f"Stage-2 타깃 '{STAGE2_TARGET}'(실제:'{y_col}')가 train에 없습니다.")

    X_all = base2[s2_feat_cols + cat_cols].copy()
    y_all = pd.to_numeric(base2[y_col], errors="coerce")

    # 결측 제거 인덱스
    cv_idx    = (train_cv_mask) & y_all.notna()
    refit_idx = (refit_mask)    & y_all.notna()
    test_idx  = test_mask

    X_cv, y_cv       = X_all.loc[cv_idx],    y_all.loc[cv_idx]
    X_refit, y_refit = X_all.loc[refit_idx], y_all.loc[refit_idx]
    X_test           = X_all.loc[test_idx]

    # ----- (1) TS-CV 튜닝 -----
    best_params, cv_mae = tscv_tune_lgbm(
        X_cv, y_cv, cat_cols,
        use_optuna=USE_OPTUNA, n_trials=N_TRIALS, n_splits=TS_N_SPLITS
    )

    # ----- (2) 최종 재학습 -----
    num_cols = [c for c in X_refit.columns if c not in cat_cols]
    pre_full = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop"
    )
    Xref_t = pre_full.fit_transform(X_refit)
    mdl = LGBMRegressor(**best_params, random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)
    mdl.fit(Xref_t, y_refit)
    print(f"[Stage-2] Refit done on rows={len(X_refit)} (REFIT_INCLUDE_VALID_MONTHS={REFIT_INCLUDE_VALID_MONTHS})")

    # ----- (3) 테스트 예측 -----
    Xte_t = pre_full.transform(X_test)
    pred_te = mdl.predict(Xte_t)
    pred_te = np.clip(pred_te, 0.0, None)  # 전기요금은 음수 불가 → 0 하한

    # 제출 파일 저장
    sub = te1[["id"]].copy()
    sub["target"] = pred_te
    sub.to_csv(SUBMISSION_PATH, index=False, encoding="utf-8-sig")
    print(f"[DONE] Saved submission: {SUBMISSION_PATH} (rows={len(sub)}) | CV-MAE={cv_mae:.6f}")

    # ----- (4) 테스트/학습용 피처 전체 덤프 저장 -----
    calendar_pf_cols = [
        "year","month","day","hour","minute","dow",
        "sin_day","cos_day","sin_dow","cos_dow",
        "is_weekend","is_monday","is_sunday",
        "is_holiday","is_peak","is_offpeak",
        "is_summer","is_winter","dt_invalid",
        "pf_mag","pf_signed","pf_dev_from_unity","pf_angle_rad",
        "q_ratio","pf_below90_flag","pf_side","q_side","season"
    ]

    # --- TEST ---
    test_features = base2.loc[test_idx, :].copy()
    test_out = te1.copy()  # 원본 test(=stage1_test_predictions.csv) 열 보존
    add_cols_test = [c for c in (s2_feat_cols + calendar_pf_cols)
                     if c not in test_out.columns and c in test_features.columns]
    test_out = pd.concat([test_out.reset_index(drop=True),
                          test_features[add_cols_test].reset_index(drop=True)], axis=1)
    test_out.to_csv(TEST_FEATURES_PATH, index=False, encoding="utf-8-sig")
    print(f"[DONE] Saved test features dump: {TEST_FEATURES_PATH} (rows={len(test_out)}, cols={len(test_out.columns)})")

    # --- TRAIN (전체 train 행 기준으로, 사용 피처 옆에 붙이기) ---
    train_features = base2.loc[~is_test, :].copy()
    train_out = tr.copy()  # 원본 train 열 보존(정렬후 인덱스 reset된 상태)
    add_cols_train = [c for c in (s2_feat_cols + calendar_pf_cols)
                      if c not in train_out.columns and c in train_features.columns]
    train_out = pd.concat([train_out.reset_index(drop=True),
                           train_features[add_cols_train].reset_index(drop=True)], axis=1)
    train_out.to_csv(TRAIN_FEATURES_PATH, index=False, encoding="utf-8-sig")
    print(f"[DONE] Saved train features dump: {TRAIN_FEATURES_PATH} (rows={len(train_out)}, cols={len(train_out.columns)})")

    return dict(
        submission_path=SUBMISSION_PATH,
        test_features_path=TEST_FEATURES_PATH,
        train_features_path=TRAIN_FEATURES_PATH,
        cv_mae=float(cv_mae),
        n_refit=len(X_refit), n_test=len(X_test),
        n_features=len(s2_feat_cols)
    )

# =========================
# 실행
# =========================
if __name__ == "__main__":
    artifacts = run_stage2_tscv_refit()
