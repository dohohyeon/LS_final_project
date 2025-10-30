# shared.py
import streamlit as st
import pandas as pd
import numpy as np

# =========================
# 고정 컬럼명 (제공 스키마)
# =========================
COL_TIME = "측정일시"
COL_USAGE = "전력사용량(kWh)"
COL_COST = "전기요금(원)"
COL_JOB = "작업유형"
COL_DEMAND = "수요전력(kW)"
COL_PF = "지상역률(%)"

# 상수 정의
PEAK_DEMAND_THRESHOLD = 30.0
POWER_FACTOR_THRESHOLD = 90.0

@st.cache_data
def load_train(path="./data/raw/train.csv"):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"⚠️ {path} 파일을 찾을 수 없습니다.")
        return pd.DataFrame()

    df.columns = df.columns.str.strip()
    df[COL_TIME] = pd.to_datetime(df[COL_TIME], errors="coerce")
    df = df.dropna(subset=[COL_TIME]).sort_values(COL_TIME)
    df["월"] = df[COL_TIME].dt.to_period("M").astype(str)
    weekday_map = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}
    df["요일"] = df[COL_TIME].dt.weekday.map(weekday_map)
    df["시간"] = df[COL_TIME].dt.hour
    df[COL_DEMAND] = df[COL_USAGE] * 4

    if COL_PF not in df.columns:
        df[COL_PF] = np.random.uniform(88, 99, len(df)).round(2)
    else:
        df[COL_PF] = pd.to_numeric(df[COL_PF], errors='coerce').fillna(95.0)

    if COL_COST not in df.columns:
        df[COL_COST] = df[COL_USAGE] * 150
    else:
        df[COL_COST] = pd.to_numeric(df[COL_COST], errors='coerce').fillna(0)

    df[COL_JOB] = df.get(COL_JOB, "미지정").fillna("미지정")
    return df

def apply_filters(df, jobs_selected, date_range):
    out = df.copy()
    if jobs_selected:
        out = out[out[COL_JOB].isin(jobs_selected)]
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        out = out[(out[COL_TIME] >= start_date) & (out[COL_TIME] <= end_date)]
    return out

def metric_label(col):
    labels = {
        COL_USAGE: "전력사용량(kWh)",
        COL_COST: "전기요금(원)",
        COL_DEMAND: "수요전력(kW)",
        COL_PF: "역률(%)"
    }
    return labels.get(col, col)

def get_agg_func(metric_col):
    if metric_col in [COL_USAGE, COL_COST]:
        return "sum"
    elif metric_col == COL_DEMAND:
        return "max"
    elif metric_col == COL_PF:
        return "mean"
    return "sum"
