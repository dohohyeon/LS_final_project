# tabs/tab_analysis.py
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from shared import * # shared.py에서 컬럼명(COL_...)과 load_train만 가져옴

# =========================
# 탭 2 (통계) 전용 헬퍼 함수
# =========================

def apply_filters(df, jobs_selected, date_range):
    """필터링된 데이터프레임 반환"""
    out = df.copy()
    if jobs_selected:
        out = out[out[COL_JOB].isin(jobs_selected)]
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        out = out[(out[COL_TIME] >= start_date) & (out[COL_TIME] <= end_date)]
    return out

def metric_label(col):
    """지표 컬럼명에 맞는 한글 라벨 반환"""
    labels = {
        COL_USAGE: "전력사용량(kWh)",
        COL_COST: "전기요금(원)",
        COL_DEMAND: "수요전력(kW)",
        COL_LAG_PF: "지상역률(%)", # 수정
        COL_LEAD_PF: "진상역률(%)" # 추가
    }
    return labels.get(col, col)

def get_agg_func(metric_col):
    """지표에 따라 적절한 집계 함수 반환"""
    if metric_col in [COL_USAGE, COL_COST]:
        return "sum"
    elif metric_col == COL_DEMAND:
        return "max"
    elif metric_col in [COL_LAG_PF, COL_LEAD_PF]: # 수정
        return "mean"
    return "sum"

# =========================
# 탭 2 메인 함수
# =========================

def show_tab_analysis(train):
    st.title("📈 전력 데이터 통계 분석")
    st.markdown("#### 🔎 필터")

    c1, c2, c3, c4 = st.columns([2, 3, 2, 2])
    with c1:
        data_source = st.radio("데이터 소스", ["스트리밍 누적", "전체 데이터"], index=1, horizontal=True)

    min_d = train[COL_TIME].min().date()
    max_d = train[COL_TIME].max().date()
    with c2:
        date_range = st.date_input("기간 선택", (min_d, max_d))
        if len(date_range) != 2: # 날짜 범위가 올바르게 선택되었는지 확인
            st.warning("기간을 선택해주세요.")
            st.stop()

    with c3:
        job_values = sorted(train[COL_JOB].dropna().unique().tolist())
        jobs_selected = st.multiselect("작업유형", job_values, default=job_values)

    with c4:
        # [수정] metric_options에 지상/진상 역률 추가
        metric_options = [COL_USAGE, COL_DEMAND, COL_LAG_PF, COL_LEAD_PF, COL_COST]
        metric_col = st.radio("지표 선택", metric_options, format_func=metric_label, horizontal=True)

    # 데이터 소스 선택
    if data_source == "스트리밍 누적" and "stream_df" in st.session_state and not st.session_state.stream_df.empty:
        base_df = st.session_state.stream_df.copy()
    else:
        base_df = train.copy()
        
    # 필터 적용
    df_f = apply_filters(base_df, jobs_selected, date_range)

    if df_f.empty:
        st.info("선택한 조건에 해당하는 데이터가 없습니다.")
        st.stop()

    # 집계 함수 가져오기
    agg_func = get_agg_func(metric_col)
    
    # --- 섹션 1: 월/요일별 분석 ---
    st.markdown(f"### 📅 월·요일별 {metric_label(metric_col)} (집계: {agg_func})")
    c1_viz, c2_viz = st.columns(2)

    with c1_viz:
        monthly = df_f.groupby("월", as_index=False)[metric_col].agg(agg_func)
        fig_m = px.bar(monthly, x="월", y=metric_col, title=f"월별 {metric_label(metric_col)} ({agg_func})")
        st.plotly_chart(fig_m, use_container_width=True)

    with c2_viz:
        order = ["월", "화", "수", "목", "금", "토", "일"]
        weekly = df_f.groupby("요일", as_index=False)[metric_col].agg(agg_func)
        weekly["요일"] = pd.Categorical(weekly["요일"], categories=order, ordered=True)
        weekly = weekly.sort_values("요일")
        fig_w = px.bar(weekly, x="요일", y=metric_col, title=f"요일별 {metric_label(metric_col)} ({agg_func})")
        st.plotly_chart(fig_w, use_container_width=True)

    # --- 섹션 2: 시간대별 분석 ---
    st.markdown(f"### ⏱ 시간대별 작업유형별 {metric_label(metric_col)} (집계: {agg_func})")

    hour_job = (
        df_f
        .groupby(["시간", COL_JOB], dropna=False)[metric_col]
        .agg(agg_func)
        .reset_index()
    )
    all_hours = pd.DataFrame({"시간": np.arange(24)})
    hour_job = all_hours.merge(hour_job, on="시간", how="left")
    hour_job[COL_JOB] = hour_job[COL_JOB].fillna("미지정")
    hour_job[metric_col] = hour_job[metric_col].fillna(0)

    fig_stack = px.bar(
        hour_job, x="시간", y=metric_col, color=COL_JOB,
        barmode="stack", title=f"시간대별 작업유형별 {metric_label(metric_col)} 현황"
    )
    st.plotly_chart(fig_stack, use_container_width=True)
