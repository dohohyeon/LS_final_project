# tabs/tab_analysis.py
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from shared import *

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

    with c3:
        job_values = sorted(train[COL_JOB].dropna().unique().tolist())
        jobs_selected = st.multiselect("작업유형", job_values, default=job_values)

    with c4:
        metric_options = [COL_USAGE, COL_DEMAND, COL_PF, COL_COST]
        metric_col = st.radio("지표 선택", metric_options, format_func=metric_label, horizontal=True)

    base_df = st.session_state.stream_df if (data_source == "스트리밍 누적" and not st.session_state.stream_df.empty) else train
    df_f = apply_filters(base_df, jobs_selected, date_range)
    if df_f.empty:
        st.info("선택한 조건에 해당하는 데이터가 없습니다.")
        st.stop()

    agg_func = get_agg_func(metric_col)
    st.markdown(f"### 📅 월·요일별 {metric_label(metric_col)} (집계: {agg_func})")
    c1_viz, c2_viz = st.columns(2)

    monthly = df_f.groupby("월", as_index=False)[metric_col].agg(agg_func)
    fig_m = px.bar(monthly, x="월", y=metric_col, title=f"월별 {metric_label(metric_col)} ({agg_func})")
    c1_viz.plotly_chart(fig_m, use_container_width=True)

    order = ["월", "화", "수", "목", "금", "토", "일"]
    weekly = df_f.groupby("요일", as_index=False)[metric_col].agg(agg_func)
    weekly["요일"] = pd.Categorical(weekly["요일"], categories=order, ordered=True)
    weekly = weekly.sort_values("요일")
    fig_w = px.bar(weekly, x="요일", y=metric_col, title=f"요일별 {metric_label(metric_col)} ({agg_func})")
    c2_viz.plotly_chart(fig_w, use_container_width=True)

    st.markdown(f"### ⏱ 시간대별 작업유형별 {metric_label(metric_col)}")
    hour_job = df_f.groupby(["시간", COL_JOB])[metric_col].agg(agg_func).reset_index()
    fig_stack = px.bar(hour_job, x="시간", y=metric_col, color=COL_JOB, barmode="stack")
    st.plotly_chart(fig_stack, use_container_width=True)

    st.markdown(f"### 📆 일별 {metric_label(metric_col)} 추이")
    df_day = df_f.copy()
    df_day["date"] = df_day[COL_TIME].dt.date
    daily = df_day.groupby("date", as_index=False)[metric_col].agg(agg_func)
    fig_daily = px.line(daily, x="date", y=metric_col, markers=True)
    st.plotly_chart(fig_daily, use_container_width=True)
