# tabs/tab_realtime.py
import streamlit as st
import plotly.express as px
import pandas as pd
import time
import numpy as np
from shared import *

def show_tab_realtime(train):
    st.title("⚡ 실시간 전력 모니터링")

    st.session_state.setdefault("running", False)
    st.session_state.setdefault("index", 0)
    st.session_state.setdefault("stream_df", pd.DataFrame(columns=train.columns))

    kpi_ph = st.empty()
    chart_demand_ph = st.empty()
    chart_pf_ph = st.empty()
    table_ph = st.empty()

    # 사이드바 제어
    st.sidebar.header("⚙️ 스트리밍 제어")
    speed = st.sidebar.slider("업데이트 간격(초)", 0.1, 5.0, 0.5, 0.1)
    col1, col2 = st.sidebar.columns(2)

    if not st.session_state.running:
        if col1.button("▶ 시작"):
            st.session_state.running = True
            st.rerun()
    else:
        if col1.button("⏸ 정지"):
            st.session_state.running = False
            st.rerun()

    if col2.button("🔄 초기화"):
        st.session_state.index = 0
        st.session_state.stream_df = pd.DataFrame(columns=train.columns)
        st.session_state.running = False
        st.rerun()

    st.sidebar.write("상태:", "🟢 실행 중" if st.session_state.running else "🔴 정지됨")

    def update_dashboard(df_partial):
        if df_partial.empty:
            kpi_ph.info("데이터가 아직 없습니다. '시작' 버튼을 눌러주세요.")
            return

        latest_row = df_partial.iloc[-1]
        latest_time = latest_row[COL_TIME]
        today_data = df_partial[df_partial[COL_TIME].dt.date == latest_time.date()]
        month_data = df_partial[df_partial[COL_TIME].dt.month == latest_time.month]

        current_demand = latest_row[COL_DEMAND]
        current_pf = latest_row[COL_PF]
        today_peak = today_data[COL_DEMAND].max()
        month_peak = month_data[COL_DEMAND].max()
        month_usage = month_data[COL_USAGE].sum()
        month_cost = month_data[COL_COST].sum()

        demand_delta = current_demand - PEAK_DEMAND_THRESHOLD
        pf_delta = current_pf - POWER_FACTOR_THRESHOLD

        with kpi_ph.container():
            k = st.columns(6)
            k[0].metric("실시간 수요전력 (kW)", f"{current_demand:,.1f}", f"{demand_delta:,.1f} (목표: {PEAK_DEMAND_THRESHOLD})", delta_color="inverse")
            k[1].metric("금일 최대 피크 (kW)", f"{today_peak:,.1f}")
            k[2].metric("당월 최대 피크 (kW)", f"{month_peak:,.1f}")
            k[3].metric("실시간 역률 (%)", f"{current_pf:,.1f}", f"{pf_delta:,.1f} (한계: {POWER_FACTOR_THRESHOLD})")
            k[4].metric("당월 누적 사용량 (kWh)", f"{month_usage:,.0f}")
            k[5].metric("당월 누적 요금 (원)", f"{month_cost:,.0f}")

        # 그래프
        fig1 = px.line(df_partial, x=COL_TIME, y=COL_DEMAND, title="실시간 수요전력(kW) 추이", markers=True)
        fig1.add_hline(y=PEAK_DEMAND_THRESHOLD, line_dash="dash", line_color="red", annotation_text="목표 피크")
        chart_demand_ph.plotly_chart(fig1, use_container_width=True)

        fig2 = px.line(df_partial, x=COL_TIME, y=COL_PF, title="실시간 역률(%) 추이", markers=True)
        fig2.add_hline(y=POWER_FACTOR_THRESHOLD, line_dash="dash", line_color="red", annotation_text="역률 한계")
        chart_pf_ph.plotly_chart(fig2, use_container_width=True)

        table_ph.dataframe(df_partial[[COL_TIME, COL_DEMAND, COL_USAGE, COL_PF, COL_COST]].tail(10), use_container_width=True, hide_index=True)

    if st.session_state.running:
        for i in range(st.session_state.index, len(train)):
            if not st.session_state.running:
                break
            row = train.iloc[i:i+1]
            st.session_state.stream_df = pd.concat([st.session_state.stream_df, row], ignore_index=True)
            st.session_state.index = i + 1
            time.sleep(max(0.05, speed))
            update_dashboard(st.session_state.stream_df)

        if st.session_state.index >= len(train):
            st.session_state.running = False
            st.success("✅ 모든 데이터 스트리밍 완료.")
            st.rerun()
    else:
        update_dashboard(st.session_state.stream_df)
