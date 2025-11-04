import streamlit as st
import plotly.express as px
import pandas as pd
import time
import numpy as np
from shared import * # shared.py에서 컬럼명(COL_...)과 load_train만 가져옴

# =========================
# 탭 1 (실시간) 전용 상수
# =========================
DEFAULT_PEAK_DEMAND_THRESHOLD = 30.0  # 데이터 부족 시 기본값
TIME_SLOT_DEFINITIONS = [
    ("새벽", "새벽(00:00~08:00)", 0, 8),
    ("아침", "아침(08:00~18:00)", 8, 18),
    ("저녁", "저녁(18:00~24:00)", 18, 24),
]
TIME_SLOT_DISPLAY_MAP = {label: display for label, display, _, _ in TIME_SLOT_DEFINITIONS}
PF_LAG_THRESHOLD = 90.0     # 지상역률 한계 (90%)
PF_LEAD_THRESHOLD = 95.0    # 진상역률 한계 (95%)


def _assign_time_slot_label(timestamp):
    if pd.isna(timestamp):
        return None
    hour = timestamp.hour
    for label, _, start, end in TIME_SLOT_DEFINITIONS:
        if start <= hour < end:
            return label
    return TIME_SLOT_DEFINITIONS[-1][0]


def _calc_iqr_threshold(series):
    if series is None:
        return None
    cleaned = series.dropna()
    if cleaned.empty:
        return None
    q1 = cleaned.quantile(0.25)
    q3 = cleaned.quantile(0.75)
    return float(q3 + 1.5 * (q3 - q1))

def _compute_time_slot_thresholds(train_df):
    thresholds = {label: DEFAULT_PEAK_DEMAND_THRESHOLD for label, _, _, _ in TIME_SLOT_DEFINITIONS}
    fallback = DEFAULT_PEAK_DEMAND_THRESHOLD

    if train_df is None or len(train_df) == 0:
        return thresholds, fallback

    if COL_TIME not in train_df.columns or COL_DEMAND not in train_df.columns:
        return thresholds, fallback

    df = train_df[[COL_TIME, COL_DEMAND]].dropna()

    if df.empty:
        return thresholds, fallback

    df = df.copy()
    df[COL_TIME] = pd.to_datetime(df[COL_TIME], errors="coerce")
    df = df.dropna(subset=[COL_TIME])
    if df.empty:
        return thresholds, fallback

    fallback_candidate = _calc_iqr_threshold(df[COL_DEMAND])
    if fallback_candidate is not None:
        fallback = fallback_candidate

    df["_time_slot_label"] = df[COL_TIME].apply(_assign_time_slot_label)

    for slot_label, _, _, _ in TIME_SLOT_DEFINITIONS:
        slot_series = df.loc[df["_time_slot_label"] == slot_label, COL_DEMAND]
        slot_threshold = _calc_iqr_threshold(slot_series)
        thresholds[slot_label] = slot_threshold if slot_threshold is not None else fallback

    return thresholds, fallback

def show_tab_realtime(train, base_interval_sec, playback_factor):
    
    # --- CSS 스타일 추가 (다크모드 네온 스타일) ---
    st.markdown("""
        <style>
        /* KPI 카드 컨테이너 고정 */
        #kpi-sticky-container {
            position: sticky !important;
            top: 0px !important;
            z-index: 999 !important;
            background-color: white !important;
            padding: 10px 0 !important;
        }
        
        /* 다크모드 KPI 카드 스타일 */
        .kpi-card {
            background: #ffffff;
            border: 2px solid #1a1a2e;
            padding: 24px;
            border-radius: 16px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
            min-height: 230px;
            height: 100%;
            position: relative;
            overflow: hidden;
            text-align: center;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        
        .kpi-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, transparent, #1a1a2e, transparent);
            opacity: 0.6;
        }
        
        .kpi-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 48px rgba(0, 0, 0, 0.2);
            border-color: #1a1a2e;
        }
        
        .kpi-label {
            font-size: 16px;
            color: #1a1a2e;
            font-weight: 600;
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 1.5px;
        }
        
        .kpi-value {
            font-size: 32px;
            font-weight: 800;
            color: #1a1a2e;
            margin-bottom: 8px;
            font-family: 'Segoe UI', sans-serif;
        }
        
        .kpi-unit {
            font-size: 16px;
            font-weight: 500;
            color: #1a1a2e;
            margin-left: 4px;
        }
        
        .kpi-delta {
            font-size: 12px;
            font-weight: 600;
            padding: 6px 12px;
            border-radius: 20px;
            display: inline-block;
            backdrop-filter: blur(10px);
            border: 1px solid;
        }
        
        .kpi-delta.positive {
            color: #059669;
            background: rgba(5, 150, 105, 0.1);
            border-color: rgba(5, 150, 105, 0.3);
        }
        
        .kpi-delta.negative {
            color: #dc2626;
            background: rgba(220, 38, 38, 0.1);
            border-color: rgba(220, 38, 38, 0.3);
        }
        
        .kpi-delta.neutral {
            color: #2563eb;
            background: rgba(37, 99, 235, 0.1);
            border-color: rgba(37, 99, 235, 0.3);
        }
        
        .kpi-delta.waiting {
            color: #64748b;
            background: rgba(100, 116, 139, 0.1);
            border-color: rgba(100, 116, 139, 0.3);
        }
        
        /* Expander 헤더 배경색 변경 */
        div[data-testid="stExpander"] details summary {
            background-color: #0E2841 !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 12px !important;
            font-weight: 600 !important;
        }
        
        div[data-testid="stExpander"] details summary:hover {
            background-color: #1a3a5c !important;
            color: white !important;
        }
        
        /* 추가 선택자들 */
        [data-testid="stExpander"] summary {
            background-color: #0E2841 !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 12px !important;
            font-weight: 600 !important;
        }
        
        [data-testid="stExpander"] summary:hover {
            background-color: #1a3a5c !important;
            color: white !important;
        }
        
        [data-testid="stExpander"] > div > div > button {
            background-color: #0E2841 !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 12px !important;
            font-weight: 600 !important;
        }
        
        [data-testid="stExpander"] > div > div > button:hover {
            background-color: #1a3a5c !important;
            color: white !important;
        }
        
        /* 화살표 아이콘도 흰색으로 */
        [data-testid="stExpander"] svg {
            fill: white !important;
            color: white !important;
        }
        </style>
    """, unsafe_allow_html=True)

    time_slot_thresholds, fallback_peak_threshold = _compute_time_slot_thresholds(train)

    # --- 1. 플레이스홀더 생성 ---
    # KPI 컨테이너를 sticky로 만들기 위해 div로 감싸기
    st.markdown('<div id="kpi-sticky-container">', unsafe_allow_html=True)
    kpi_ph = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)
    
    chart_demand_ph = st.empty()
    chart_pf_ph = st.empty()
    table_ph = st.empty()

    # --- 2. 초기 KPI 카드 표시 함수 ---
    def show_initial_kpi():
        with kpi_ph.container():
            cols = st.columns(6)
            
            # KPI 1: 실시간 수요전력    
            with cols[0]:
                st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-label">실시간 수요전력</div>
                        <div class="kpi-value">--<span class="kpi-unit">kW</span></div>
                        <div class="kpi-delta waiting">데이터 대기 중</div>
                    </div>
                """, unsafe_allow_html=True)
            
            # KPI 2: 당월 최대 피크
            with cols[1]:
                st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-label">당월 최대 피크</div>
                        <div class="kpi-value">--<span class="kpi-unit">kW</span></div>
                        <div class="kpi-delta waiting">데이터 대기 중</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with cols[2]:
                st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-label">실시간 지상역률</div>
                        <div class="kpi-value">--<span class="kpi-unit">%</span></div>
                        <div class="kpi-delta waiting">데이터 대기 중</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with cols[3]:
                st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-label">실시간 진상역률</div>
                        <div class="kpi-value">--<span class="kpi-unit">%</span></div>
                        <div class="kpi-delta waiting">데이터 대기 중</div>
                    </div>
                """, unsafe_allow_html=True)
            # KPI 5: 당월 누적 사용량
            with cols[4]:
                st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-label">당월 누적 사용량</div>
                        <div class="kpi-value">--<span class="kpi-unit">kWh</span></div>
                        <div class="kpi-delta waiting">데이터 대기 중</div>
                    </div>
                """, unsafe_allow_html=True)
            
            # KPI 6: 당월 누적 요금
            with cols[5]:
                st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-label">당월 누적 요금</div>
                        <div class="kpi-value">--<span class="kpi-unit">원</span></div>
                        <div class="kpi-delta waiting">데이터 대기 중</div>
                    </div>
                """, unsafe_allow_html=True)

    # --- 3. 대시보드 업데이트 함수 ---
    def update_dashboard(df_partial, pf_choice):
        if df_partial.empty:
            show_initial_kpi()
            return pf_choice

        # --- 그래프용 데이터는 최근 10개만 사용 ---
        df_chart = df_partial.tail(10).copy()
        df_chart.sort_values(COL_TIME, inplace=True)
        df_chart["_time_slot_label"] = df_chart[COL_TIME].apply(_assign_time_slot_label)
        df_chart["_slot_threshold"] = df_chart["_time_slot_label"].map(time_slot_thresholds).fillna(fallback_peak_threshold)
        df_chart["_threshold_display"] = df_chart["_time_slot_label"].map(TIME_SLOT_DISPLAY_MAP).fillna("전체 시간대")
        df_chart["_exceed_flag"] = df_chart[COL_DEMAND] > df_chart["_slot_threshold"]

        # --- KPI 계산 (전체 누적 데이터 사용) ---
        latest_row = df_partial.iloc[-1]
        latest_time = pd.to_datetime(latest_row[COL_TIME], errors="coerce")

        if pd.notna(latest_time):
            today_mask = df_partial[COL_TIME].dt.date == latest_time.date()
            month_mask = df_partial[COL_TIME].dt.month == latest_time.month
            today_data = df_partial.loc[today_mask]
            month_data = df_partial.loc[month_mask]
        else:
            today_data = df_partial.iloc[0:0]
            month_data = df_partial.iloc[0:0]

        current_demand = latest_row[COL_DEMAND]
        current_lag_pf = latest_row[COL_LAG_PF]
        current_lead_pf = latest_row[COL_LEAD_PF]

        current_slot = _assign_time_slot_label(latest_time)
        current_threshold = time_slot_thresholds.get(current_slot, fallback_peak_threshold)
        slot_display = TIME_SLOT_DISPLAY_MAP.get(current_slot, "전체 시간대")
        today_peak = today_data[COL_DEMAND].max() if not today_data.empty else np.nan
        month_peak = month_data[COL_DEMAND].max() if not month_data.empty else np.nan
        month_usage = month_data[COL_USAGE].sum() if not month_data.empty else 0
        month_cost = month_data[COL_COST].sum() if not month_data.empty else 0

        demand_delta = current_demand - current_threshold
        lag_pf_delta = current_lag_pf - PF_LAG_THRESHOLD
        lead_pf_delta = current_lead_pf - PF_LEAD_THRESHOLD
        
        # Delta 색상 결정
        demand_delta_class = "negative" if demand_delta > 0 else "positive"
        lag_pf_delta_class = "positive" if lag_pf_delta >= 0 else "negative"
        lead_pf_delta_class = "positive" if lead_pf_delta >= 0 else "negative"
        
        # --- KPI 업데이트 (HTML 커스텀 카드) ---
        with kpi_ph.container():
            cols = st.columns(6)
            
            # KPI 1: 실시간 수요전력
            with cols[0]:
                current_card_style = "background-color: #FFECEC;" if current_demand > current_threshold else ""
                current_label_style = "color: #B91C1C;" if current_demand > current_threshold else ""
                st.markdown(
                    f"""
                    <div class="kpi-card" style="{current_card_style}">
                        <div class="kpi-label">실시간 수요전력</div>
                        <div class="kpi-value">{current_demand:,.1f}<span class="kpi-unit">kW</span></div>
                        <div class="kpi-delta {demand_delta_class}">
                            {demand_delta:+.1f} kW<br/>
                            <span style="font-weight: 400;">{slot_display} 기준 {current_threshold:,.1f} kW</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            
            # KPI 2: 당월 최대 피크
            with cols[1]:
                st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-label">당월 최대 피크</div>
                        <div class="kpi-value">{month_peak:,.1f}<span class="kpi-unit">kW</span></div>
                        <div class="kpi-delta neutral">월간 최고값</div>
                    </div>
                """, unsafe_allow_html=True)
            
            # KPI 3: 실시간 지상역률
            with cols[2]:
                lag_card_style = "background-color: #FFECEC;" if current_lag_pf < PF_LAG_THRESHOLD else ""
                st.markdown(
                    f"""
                    <div class="kpi-card" style="{lag_card_style}">
                        <div class="kpi-label">실시간 지상역률</div>
                        <div class="kpi-value">{current_lag_pf:,.1f}<span class="kpi-unit">%</span></div>
                        <div class="kpi-delta {lag_pf_delta_class}">
                            {lag_pf_delta:+.1f}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            
            # KPI 4: 실시간 진상역률
            with cols[3]:
                lead_card_style = "background-color: #FFECEC;" if current_lead_pf < PF_LEAD_THRESHOLD else ""
                st.markdown(
                    f"""
                    <div class="kpi-card" style="{lead_card_style}">
                        <div class="kpi-label">실시간 진상역률</div>
                        <div class="kpi-value">{current_lead_pf:,.1f}<span class="kpi-unit">%</span></div>
                        <div class="kpi-delta {lead_pf_delta_class}">
                            {lead_pf_delta:+.1f}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            
            # KPI 5: 당월 누적 사용량
            with cols[4]:
                st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-label">당월 누적 사용량</div>
                        <div class="kpi-value">{month_usage:,.0f}<span class="kpi-unit">kWh</span></div>
                        <div class="kpi-delta neutral">월간 누적</div>
                    </div>
                """, unsafe_allow_html=True)
            
            # KPI 6: 당월 누적 요금
            with cols[5]:
                st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-label">당월 누적 요금</div>
                        <div class="kpi-value">{month_cost:,.0f}<span class="kpi-unit">원</span></div>
                        <div class="kpi-delta neutral">월간 누적</div>
                    </div>
                """, unsafe_allow_html=True)

        # --- 수요전력(kW) 차트 업데이트 (최근 10개만) - 토글 상자로 변경 ---
        with chart_demand_ph.container():
            st.markdown("<br><br>", unsafe_allow_html=True)  # KPI와 차트 사이 여백
            with st.expander("실시간 수요전력(kW) 추이", expanded=True):
                fig1 = px.line(df_chart, x=COL_TIME, y=COL_DEMAND, title="", markers=True)
                if not df_chart.empty:
                    fig1.add_scatter(
                        x=df_chart[COL_TIME],
                        y=df_chart["_slot_threshold"],
                        mode="lines",
                        name="시간대 관리기준선",
                        line=dict(color="red", dash="dash"),
                        text=df_chart["_threshold_display"],
                        hovertemplate="%{text}<br>%{x}<br>%{y:.1f} kW<extra></extra>",
                    )
                    exceed_points = df_chart[df_chart["_exceed_flag"]]
                    if not exceed_points.empty:
                        fig1.add_scatter(
                            x=exceed_points[COL_TIME],
                            y=exceed_points[COL_DEMAND],
                            mode="markers",
                            name="목표 피크 초과",
                            marker=dict(color="#FF4D4F", size=10, symbol="circle"),
                            hovertemplate="목표 초과<br>%{x}<br>%{y:.1f} kW<extra></extra>",
                        )
                fig1.update_layout(
                    plot_bgcolor="#FFECEC" if current_demand > current_threshold else "white",
                    paper_bgcolor="white",
                )
                fig1.update_yaxes(rangemode="tozero")
                st.plotly_chart(fig1, use_container_width=True)

        # --- 역률(%) 차트 업데이트 (최근 10개만) - 토글 상자로 변경 ---
        with chart_pf_ph.container():
            st.markdown("<br><br>", unsafe_allow_html=True)  # 차트 간격 조정
            expander_title = "실시간 지상역률(%) 추이" if pf_choice == COL_LAG_PF else "실시간 진상역률(%) 추이"
            with st.expander(expander_title, expanded=True):
                # 역률 차트 선택 라디오 버튼 (토글 안에 위치)
                pf_choice = st.radio(
                    "역률 차트 선택",
                    [COL_LAG_PF, COL_LEAD_PF], 
                    horizontal=True,
                    key=f"pf_chart_select_{st.session_state.index}",  # 고유 키 생성
                    format_func=lambda col_name: "지상 역률" if col_name == COL_LAG_PF else "진상 역률",
                    index=0 if pf_choice == COL_LAG_PF else 1
                )
                
                if pf_choice == COL_LAG_PF:
                    fig_pf = px.line(df_chart, x=COL_TIME, y=COL_LAG_PF,
                                     title="", markers=True, color_discrete_sequence=['#ff7f0e'])
                    fig_pf.add_hline(y=PF_LAG_THRESHOLD, line_dash="dash", line_color="red", annotation_text="지상 한계 (90%)", annotation_position="bottom right")
                    if not df_chart.empty:
                        below_lag = df_chart[df_chart[COL_LAG_PF] < PF_LAG_THRESHOLD]
                        if not below_lag.empty:
                            fig_pf.add_scatter(
                                x=below_lag[COL_TIME],
                                y=below_lag[COL_LAG_PF],
                                mode="markers",
                                name="지상 한계 미달",
                                marker=dict(color="#FF4D4F", size=10, symbol="triangle-down"),
                                hovertemplate="지상 한계 미달<br>%{x}<br>%{y:.1f} %<extra></extra>",
                            )
                        lag_bg_color = "white" if below_lag.empty or (below_lag[COL_LAG_PF] >= PF_LAG_THRESHOLD).all() else "#FFECEC"
                    else:
                        lag_bg_color = "white"
                    fig_pf.update_layout(
                        plot_bgcolor=lag_bg_color,
                        paper_bgcolor="white",
                    )
                    y_min_val = min(40, df_chart[COL_LAG_PF].min() - 2) if not df_chart.empty else 40
                    fig_pf.update_yaxes(range=[y_min_val, 101])
                else:
                    fig_pf = px.line(df_chart, x=COL_TIME, y=COL_LEAD_PF,
                                     title="", markers=True, color_discrete_sequence=['#2ca02c'])
                    fig_pf.add_hline(y=PF_LEAD_THRESHOLD, line_dash="dot", line_color="blue", annotation_text="진상 한계 (95%)", annotation_position="top right")
                    if not df_chart.empty:
                        below_lead = df_chart[df_chart[COL_LEAD_PF] < PF_LEAD_THRESHOLD]
                        if not below_lead.empty:
                            fig_pf.add_scatter(
                                x=below_lead[COL_TIME],
                                y=below_lead[COL_LEAD_PF],
                                mode="markers",
                                name="진상 한계 미달",
                                marker=dict(color="#FF4D4F", size=10, symbol="triangle-down"),
                                hovertemplate="진상 한계 미달<br>%{x}<br>%{y:.1f} %<extra></extra>",
                            )
                        lead_bg_color = "white" if below_lead.empty or (below_lead[COL_LEAD_PF] >= PF_LEAD_THRESHOLD).all() else "#FFECEC"
                    else:
                        lead_bg_color = "white"
                    fig_pf.update_layout(
                        plot_bgcolor=lead_bg_color,
                        paper_bgcolor="white",
                    )
                    y_min_val = min(40, df_chart[COL_LEAD_PF].min() - 2) if not df_chart.empty else 40
                    fig_pf.update_yaxes(range=[y_min_val, 101])

                st.plotly_chart(fig_pf, use_container_width=True)
        
        return pf_choice

    # --- 데이터 테이블 업데이트 함수 (별도 분리) ---
    def update_table(df_partial, pf_choice):
        # --- 데이터 테이블 업데이트 ---
        pf_columns = [COL_LAG_PF] if pf_choice == COL_LAG_PF else [COL_LEAD_PF]
        base_columns = [COL_TIME, COL_DEMAND, COL_USAGE] + pf_columns + [COL_COST, COL_JOB]
        cols_to_show = [col for col in base_columns if col in df_partial.columns]

        log_limit = 50
        df_display = df_partial[cols_to_show].sort_values(COL_TIME, ascending=False).head(log_limit).copy()

        if df_display.empty:
            styled_display = df_display
        else:
            if COL_DEMAND in df_display.columns and COL_TIME in df_display.columns:
                thresholds_for_rows = df_display[COL_TIME].apply(
                    lambda ts: time_slot_thresholds.get(_assign_time_slot_label(ts), fallback_peak_threshold)
                )
                exceed_mask = (df_display[COL_DEMAND] > thresholds_for_rows).fillna(False)
            else:
                exceed_mask = pd.Series(False, index=df_display.index)

            pf_column = pf_columns[0] if pf_columns else None
            pf_below_mask = pd.Series(False, index=df_display.index)
            if pf_column and pf_column in df_display.columns:
                pf_threshold = PF_LAG_THRESHOLD if pf_choice == COL_LAG_PF else PF_LEAD_THRESHOLD
                pf_below_mask = (df_display[pf_column] < pf_threshold).fillna(False)

            def highlight_rows(row):
                demand_exceed = exceed_mask.loc[row.name] if row.name in exceed_mask.index else False
                pf_below = pf_below_mask.loc[row.name] if row.name in pf_below_mask.index else False

                if demand_exceed:
                    color = "background-color: #DDC987"
                elif pf_below:
                    color = "background-color: #FCDEDC"
                else:
                    color = ""
                return [color] * len(row)

            styled_display = df_display.style.apply(highlight_rows, axis=1)

        column_config = {}
        if COL_TIME in df_display.columns:
            column_config[COL_TIME] = st.column_config.DatetimeColumn("측정일시", format="MM-DD HH:mm:ss")
        if COL_DEMAND in df_display.columns:
            column_config[COL_DEMAND] = st.column_config.NumberColumn("수요전력(kW)", format="%.2f")
        if COL_USAGE in df_display.columns:
            column_config[COL_USAGE] = st.column_config.NumberColumn("사용량(kWh)", format="%.2f")
        if pf_choice == COL_LAG_PF and COL_LAG_PF in df_display.columns:
            column_config[COL_LAG_PF] = st.column_config.NumberColumn("지상역률(%)", format="%.1f %%")
        if pf_choice == COL_LEAD_PF and COL_LEAD_PF in df_display.columns:
            column_config[COL_LEAD_PF] = st.column_config.NumberColumn("진상역률(%)", format="%.1f %%")
        if COL_COST in df_display.columns:
            column_config[COL_COST] = st.column_config.NumberColumn("전기요금(원)", format="%d 원")
        table_ph.dataframe(
            styled_display,
            use_container_width=True,
            hide_index=True,
            column_config=column_config,
        )

    # --- 4. 초기 상태 표시 ---
    show_initial_kpi()
    
    # 초기 pf_choice 설정
    current_pf_choice = COL_LAG_PF

    # --- 5. 실시간 업데이트 루프 ---
    while st.session_state.running:
        if st.session_state.index < len(train):
            rows = train.iloc[st.session_state.index : st.session_state.index + 1]
            st.session_state.stream_df = pd.concat(
                [st.session_state.stream_df, rows], ignore_index=True
            )
            st.session_state.index += 1

            current_pf_choice = update_dashboard(st.session_state.stream_df, current_pf_choice)
            update_table(st.session_state.stream_df, current_pf_choice)

            effective_factor = max(playback_factor, 0.01)
            sleep_duration = max(0.01, base_interval_sec / effective_factor)
            check_interval = min(0.1, sleep_duration)
            steps = int(sleep_duration / check_interval)
            for _ in range(steps):
                if not st.session_state.running:
                    break
                time.sleep(check_interval)

            if st.session_state.running:
                remainder = sleep_duration % check_interval
                if remainder > 1e-9:
                    time.sleep(remainder)

        else:
            st.session_state.running = False
            st.success("✅ 모든 데이터 스트리밍 완료.")
            break

    current_pf_choice = update_dashboard(st.session_state.stream_df, current_pf_choice)
    update_table(st.session_state.stream_df, current_pf_choice)