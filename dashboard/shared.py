# shared.py
import streamlit as st
import pandas as pd
import numpy as np

# =========================
# 고정 컬럼명 (원본 CSV)
# =========================
# 이 상수들은 모든 탭에서 공통으로 사용하므로 shared.py에 둡니다.
COL_TIME = "측정일시"
COL_USAGE = "전력사용량(kWh)"
COL_COST = "전기요금(원)"
COL_JOB = "작업유형"

# --- 파생 컬럼명도 공통으로 정의 ---
COL_DEMAND = "수요전력(kW)"
COL_LAG_PF = "지상역률(%)"
COL_LEAD_PF = "진상역률(%)"

# =========================
# 데이터 로드 및 전처리
# =========================
@st.cache_data
def load_test(path="./data/raw/test_with_features.csv"):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"⚠️ {path} 파일을 찾을 수 없습니다.")
        return pd.DataFrame()

    df.columns = df.columns.str.strip()
    
    # --- 필수 컬럼 확인 ---
    if COL_TIME not in df.columns or COL_USAGE not in df.columns:
        st.error(f"필수 컬럼({COL_TIME}, {COL_USAGE})이 없습니다.")
        return pd.DataFrame()

    df[COL_TIME] = pd.to_datetime(df[COL_TIME], errors="coerce")
    df = df.dropna(subset=[COL_TIME]).sort_values(COL_TIME)
    
    # --- 파생 변수 (시간/요일/수요전력) ---
    df["월"] = df[COL_TIME].dt.to_period("M").astype(str)
    weekday_map = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}
    df["요일"] = df[COL_TIME].dt.weekday.map(weekday_map)
    df["시간"] = df[COL_TIME].dt.hour
    df[COL_DEMAND] = df[COL_USAGE] * 4 # 15분 데이터 기준

    # --- 역률 컬럼 처리 ---
    # 지상역률 (COL_PF -> COL_LAG_PF)
    if "COL_PF" in df.columns and COL_LAG_PF not in df.columns:
         df.rename(columns={"COL_PF": COL_LAG_PF}, inplace=True)

    if COL_LAG_PF not in df.columns:
        df[COL_LAG_PF] = np.random.uniform(88, 99, len(df)).round(2)
    else:
        df[COL_LAG_PF] = pd.to_numeric(df[COL_LAG_PF], errors='coerce').fillna(95.0)
        
    # 진상역률 처리
    if COL_LEAD_PF not in df.columns:
        df[COL_LEAD_PF] = np.where(df[COL_LAG_PF] < 98, 100.0, np.random.uniform(40, 90, len(df)).round(2))
        df[COL_LAG_PF] = np.where(df[COL_LEAD_PF] < 100, 100.0, df[COL_LAG_PF])
    else:
        df[COL_LEAD_PF] = pd.to_numeric(df[COL_LEAD_PF], errors='coerce').fillna(100.0)
    
    # --- 기타 컬럼 처리 ---
    if COL_COST not in df.columns:
        df[COL_COST] = df[COL_USAGE] * 150 # 임의 단가
    else:
        df[COL_COST] = pd.to_numeric(df[COL_COST], errors='coerce').fillna(0)

    df[COL_JOB] = df.get(COL_JOB, "미지정").fillna("미지정")
    
    # load_train이 전처리된 모든 데이터를 반환
    return df

# --- 탭별 헬퍼 함수 (apply_filters, metric_label, get_agg_func) 제거 ---
# --- 탭별 상수 (THRESHOLD) 제거 ---

def load_train(path="./data/raw/train_with_features.csv"):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"⚠️ {path} 파일을 찾을 수 없습니다.")
        return pd.DataFrame()

    df.columns = df.columns.str.strip()
    
    # --- 필수 컬럼 확인 ---
    if COL_TIME not in df.columns or COL_USAGE not in df.columns:
        st.error(f"필수 컬럼({COL_TIME}, {COL_USAGE})이 없습니다.")
        return pd.DataFrame()

    df[COL_TIME] = pd.to_datetime(df[COL_TIME], errors="coerce")
    df = df.dropna(subset=[COL_TIME]).sort_values(COL_TIME)
    
    # --- 파생 변수 (시간/요일/수요전력) ---
    df["월"] = df[COL_TIME].dt.to_period("M").astype(str)
    weekday_map = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}
    df["요일"] = df[COL_TIME].dt.weekday.map(weekday_map)
    df["시간"] = df[COL_TIME].dt.hour
    df[COL_DEMAND] = df[COL_USAGE] * 4 # 15분 데이터 기준

    # --- 역률 컬럼 처리 ---
    # 지상역률 (COL_PF -> COL_LAG_PF)
    if "COL_PF" in df.columns and COL_LAG_PF not in df.columns:
         df.rename(columns={"COL_PF": COL_LAG_PF}, inplace=True)

    if COL_LAG_PF not in df.columns:
        df[COL_LAG_PF] = np.random.uniform(88, 99, len(df)).round(2)
    else:
        df[COL_LAG_PF] = pd.to_numeric(df[COL_LAG_PF], errors='coerce').fillna(95.0)
        
    # 진상역률 처리
    if COL_LEAD_PF not in df.columns:
        df[COL_LEAD_PF] = np.where(df[COL_LAG_PF] < 98, 100.0, np.random.uniform(40, 90, len(df)).round(2))
        df[COL_LAG_PF] = np.where(df[COL_LEAD_PF] < 100, 100.0, df[COL_LAG_PF])
    else:
        df[COL_LEAD_PF] = pd.to_numeric(df[COL_LEAD_PF], errors='coerce').fillna(100.0)
    
    # --- 기타 컬럼 처리 ---
    if COL_COST not in df.columns:
        df[COL_COST] = df[COL_USAGE] * 150 # 임의 단가
    else:
        df[COL_COST] = pd.to_numeric(df[COL_COST], errors='coerce').fillna(0)

    df[COL_JOB] = df.get(COL_JOB, "미지정").fillna("미지정")
    
    # load_train이 전처리된 모든 데이터를 반환
    return df