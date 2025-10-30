import streamlit as st
import plotly.express as px
import pandas as pd
import time
import numpy as np
from shared import * # shared.py에서 컬럼명(COL_...)과 load_train만 가져옴

# =========================
# 탭 1 (실시간) 전용 상수
# =========================
PEAK_DEMAND_THRESHOLD = 30.0  # 목표 피크 (kW)
PF_LAG_THRESHOLD = 90.0     # 지상역률 한계 (90%)
PF_LEAD_THRESHOLD = 95.0    # 진상역률 한계 (95%)


def show_tab_realtime(train, speed):
    st.title("⚡ 실시간 전력 모니터링")

    # --- 1. 플레이스홀더 생성 (변경 없음) ---
    # 이 빈 상자들은 스크립트가 실행될 때마다 한 번씩 생성됩니다.
    kpi_ph = st.empty()
    chart_demand_ph = st.empty()
    
    st.markdown("##### 📊 역률 차트 선택")
    pf_choice = st.radio(
        "표시할 역률 차트 선택",
        [COL_LAG_PF, COL_LEAD_PF], 
        horizontal=True,
        key="pf_chart_select",
        format_func=lambda col_name: "지상 역률" if col_name == COL_LAG_PF else "진상 역률",
        label_visibility="collapsed"
    )
    
    chart_pf_ph = st.empty()
    table_ph = st.empty()

    # --- 2. 대시보드 업데이트 함수 (변경 없음) ---
    # 이 함수는 플레이스홀더의 *내용물*만 교체합니다.
    def update_dashboard(df_partial):
        if df_partial.empty:
            kpi_ph.info("데이터가 아직 없습니다. '시작' 버튼을 눌러주세요.")
            return

        # --- KPI 계산 ---
        latest_row = df_partial.iloc[-1]
        latest_time = latest_row[COL_TIME]
        today_data = df_partial[df_partial[COL_TIME].dt.date == latest_time.date()]
        month_data = df_partial[df_partial[COL_TIME].dt.month == latest_time.month]

        current_demand = latest_row[COL_DEMAND]
        current_lag_pf = latest_row[COL_LAG_PF]
        current_lead_pf = latest_row[COL_LEAD_PF]
        
        today_peak = today_data[COL_DEMAND].max()
        month_peak = month_data[COL_DEMAND].max()
        month_usage = month_data[COL_USAGE].sum()
        month_cost = month_data[COL_COST].sum()

        demand_delta = current_demand - PEAK_DEMAND_THRESHOLD
        lag_pf_delta = current_lag_pf - PF_LAG_THRESHOLD
        lag_pf_color = "inverse" if lag_pf_delta < 0 else "normal"
        lead_pf_delta = current_lead_pf - PF_LEAD_THRESHOLD
        lead_pf_color = "inverse" if current_lead_pf < 100 and lead_pf_delta < 0 else "normal"
        
        # --- KPI 업데이트 ---
        with kpi_ph.container():
            k = st.columns(6)
            k[0].metric(
                "실시간 수요전력 (kW)", f"{current_demand:,.1f}", 
                f"{demand_delta:,.1f} (목표: {PEAK_DEMAND_THRESHOLD})", delta_color="inverse"
            )
            k[1].metric("당월 최대 피크 (kW)", f"{month_peak:,.1f}")
            k[2].metric(
                label="실시간 지상역률 (%)", value=f"{current_lag_pf:,.1f} %",
                delta=f"{lag_pf_delta:,.1f} % (한계: {PF_LAG_THRESHOLD})", delta_color=lag_pf_color
            )
            k[3].metric(
                label="실시간 진상역률 (%)", value=f"{current_lead_pf:,.1f} %",
                delta=f"{lead_pf_delta:,.1f} % (한계: {PF_LEAD_THRESHOLD})", delta_color=lead_pf_color
            )
            k[4].metric("당월 누적 사용량 (kWh)", f"{month_usage:,.0f}")
            k[5].metric("당월 누적 요금 (원)", f"{month_cost:,.0f}")

        # --- 수요전력(kW) 차트 업데이트 ---
        fig1 = px.line(df_partial, x=COL_TIME, y=COL_DEMAND, title="실시간 수요전력(kW) 추이", markers=True)
        fig1.add_hline(y=PEAK_DEMAND_THRESHOLD, line_dash="dash", line_color="red", annotation_text="목표 피크")
        fig1.update_yaxes(rangemode="tozero")
        chart_demand_ph.plotly_chart(fig1, use_container_width=True)

        # --- 역률(%) 차트 업데이트 ---
        if pf_choice == COL_LAG_PF:
            fig_pf = px.line(df_partial, x=COL_TIME, y=COL_LAG_PF,
                             title="실시간 지상역률(%) 추이", markers=True, color_discrete_sequence=['#ff7f0e'])
            fig_pf.add_hline(y=PF_LAG_THRESHOLD, line_dash="dash", line_color="red", annotation_text="지상 한계 (90%)", annotation_position="bottom right")
            y_min_val = min(40, df_partial[COL_LAG_PF].min() - 2) if not df_partial.empty else 40
            fig_pf.update_yaxes(range=[y_min_val, 101])
        else:
            fig_pf = px.line(df_partial, x=COL_TIME, y=COL_LEAD_PF,
                             title="실시간 진상역률(%) 추이", markers=True, color_discrete_sequence=['#2ca02c'])
            fig_pf.add_hline(y=PF_LEAD_THRESHOLD, line_dash="dot", line_color="blue", annotation_text="진상 한계 (95%)", annotation_position="top right")
            y_min_val = min(40, df_partial[COL_LEAD_PF].min() - 2) if not df_partial.empty else 40
            fig_pf.update_yaxes(range=[y_min_val, 101])
            
        chart_pf_ph.plotly_chart(fig_pf, use_container_width=True)

        # --- 데이터 테이블 업데이트 ---
        cols_to_show = [col for col in [
            COL_TIME, COL_DEMAND, COL_USAGE, 
            COL_LAG_PF, COL_LEAD_PF, 
            COL_COST, COL_JOB
        ] if col in df_partial.columns]

        table_ph.dataframe(
            df_partial[cols_to_show].sort_values(COL_TIME, ascending=False).head(10),
            use_container_width=True, 
            hide_index=True,
            column_config={
                COL_TIME: st.column_config.DatetimeColumn("측정일시", format="MM-DD HH:mm:ss"),
                COL_DEMAND: st.column_config.NumberColumn("수요전력(kW)", format="%.2f"),
                COL_USAGE: st.column_config.NumberColumn("사용량(kWh)", format="%.2f"),
                COL_LAG_PF: st.column_config.NumberColumn("지상역률(%)", format="%.1f %%"),
                COL_LEAD_PF: st.column_config.NumberColumn("진상역률(%)", format="%.1f %%"),
                COL_COST: st.column_config.NumberColumn("전기요금(원)", format="%d 원"),
            }
        )

    # --- 3. [핵심 수정] 실시간 업데이트 루프 ---
    # `if st.session_state.running:` 대신 `while st.session_state.running:` 사용
    
    # while 루프는 '시작' 버튼이 눌려 running=True가 되면 
    # '정지' 버튼이 눌려 running=False가 될 때까지 이 안에서 계속 반복됩니다.
    # st.rerun()을 호출하지 않기 때문에 페이지 스크롤이 유지됩니다.
    while st.session_state.running:
        if st.session_state.index < len(train):
            # 다음 1개 행(row)을 가져와 세션 상태에 추가
            row = train.iloc[st.session_state.index : st.session_state.index + 1]
            st.session_state.stream_df = pd.concat([st.session_state.stream_df, row], ignore_index=True)
            st.session_state.index += 1
            
            # 플레이스홀더 내용만 업데이트
            update_dashboard(st.session_state.stream_df)
            
            # --- 반응형 sleep ---
            # '정지' 버튼을 눌렀을 때 즉각 반응하도록, 
            # 긴 sleep을 0.1초 단위로 잘라서 체크합니다.
            sleep_duration = max(0.01, speed)
            check_interval = 0.1 # 0.1초마다 '정지' 상태 확인
            
            steps = int(sleep_duration / check_interval)
            for _ in range(steps):
                if not st.session_state.running: # '정지'가 눌렸는지 확인
                    break
                time.sleep(check_interval)
            
            # 남은 시간 마저 대기
            if st.session_state.running:
                 remainder = sleep_duration % check_interval
                 time.sleep(remainder)

        else: # 모든 데이터 스트리밍 완료
            st.session_state.running = False
            st.success("✅ 모든 데이터 스트리밍 완료.")
            break # while 루프 탈출
            
    # --- 루프 종료 후 ---
    # '정지' 상태이거나, 스트리밍이 완료되었을 때
    # 현재까지의 최종 데이터로 대시보드를 한 번 더 그려줍니다.
    update_dashboard(st.session_state.stream_df)

