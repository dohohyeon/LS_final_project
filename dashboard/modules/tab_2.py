# tabs/tab_2.py
# ---------------------------------------------------
#  전력 데이터 통합 분석 (완성형)
# - 주요지표 카드 (상승🔴 / 하락🟢 색상 일관)
# - 날짜/월별 선택 및 전월 비교 (연도 롤오버 포함)
# - 요일·시간대별 평균 전력사용 패턴
# - 피크 수요 및 역률 분석 (원형 마커, 범례명 수정)
# - 시계열 분석 (Range Slider)
# ---------------------------------------------------

import base64
import os
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from shared import (
    COL_USAGE, COL_COST, COL_DEMAND,
    COL_LAG_PF, COL_LEAD_PF, COL_TIME
)

# =========================
# 헬퍼 함수
# =========================
def prepare_source_df(df):
    """시간 관련 컬럼 생성 및 타입 보정"""
    out = df.copy()
    out[COL_TIME] = pd.to_datetime(out[COL_TIME], errors="coerce")
    out = out.dropna(subset=[COL_TIME])
    out["연"] = out[COL_TIME].dt.year
    out["월"] = out[COL_TIME].dt.month
    out["요일"] = out[COL_TIME].dt.day_name()
    out["시간"] = out[COL_TIME].dt.hour
    out["일"] = out[COL_TIME].dt.date
    return out


def calc_carbon_emission(kwh):
    """탄소배출량 계산 (단위: tCO₂)"""
    return kwh * 0.000331


def pct_change(curr, prev):
    """전월 대비 증감률 (%)"""
    return ((curr - prev) / prev * 100) if prev != 0 else 0


def metric_color_scheme(diff):
    """증감률 방향에 따른 색상"""
    if diff is None:
        return {"value": "#222", "border": "#ddd", "bg": "#fff"}
    if diff > 0:
        # 🔴 상승
        return {"value": "#d9534f", "border": "#d9534f", "bg": "#ffeaea"}
    elif diff < 0:
        # 🟢 하락
        return {"value": "#28a745", "border": "#28a745", "bg": "#eaf8ea"}
    else:
        return {"value": "#555", "border": "#ccc", "bg": "#f6f6f6"}


def metric_card(title, value, diff=None, value_color="#222", border_color="#ddd", bg_color="#fff"):
    """지표 카드 HTML"""
    diff_html = ""
    if diff is not None:
        sign = "▲" if diff > 0 else "▼" if diff < 0 else "–"
        color = "#d9534f" if diff > 0 else "#28a745" if diff < 0 else "#888"
        diff_html = f"<div style='color:{color}; font-size:13px; margin-top:4px;'>{sign} {abs(diff):.1f}%</div>"

    return f"""
    <div style="
        flex:1; border:2px solid {border_color}; border-radius:10px;
        padding:15px; margin:5px; text-align:center;
        background-color:{bg_color};
        box-shadow:0 1px 3px rgba(0,0,0,0.05);
    ">
        <div style="font-size:15px; color:#555; font-weight:600;">{title}</div>
        <div style="font-size:20px; font-weight:700; color:{value_color};">{value}</div>
        {diff_html}
    </div>
    """

# ------------------------------------------------
# 내부 헬퍼 함수 정의 (메트릭 카드 렌더링)
# ------------------------------------------------
def render_metric_cards(curr_df, prev_df=None, title="주요 지표"):
    total_usage = curr_df[COL_USAGE].sum()
    total_cost  = curr_df[COL_COST].sum()
    avg_price   = total_cost / total_usage if total_usage > 0 else 0
    carbon      = calc_carbon_emission(total_usage)

    usage_diff = cost_diff = price_diff = carbon_diff = None
    if prev_df is not None and not prev_df.empty:
        prev_usage  = prev_df[COL_USAGE].sum()
        prev_cost   = prev_df[COL_COST].sum()
        prev_price  = prev_cost / prev_usage if prev_usage > 0 else 0
        prev_carbon = calc_carbon_emission(prev_usage)
        usage_diff  = pct_change(total_usage, prev_usage)
        cost_diff   = pct_change(total_cost, prev_cost)
        price_diff  = pct_change(avg_price, prev_price)
        carbon_diff = pct_change(carbon, prev_carbon)

    c_usage  = metric_color_scheme(usage_diff)
    c_cost   = metric_color_scheme(cost_diff)
    c_price  = metric_color_scheme(price_diff)
    c_carbon = metric_color_scheme(carbon_diff)

    cards_html = "".join([
        metric_card("전력사용량", f"{total_usage:,.1f} kWh", usage_diff,
                    value_color=c_usage["value"], border_color=c_usage["border"], bg_color=c_usage["bg"]),
        metric_card("전기요금", f"{total_cost:,.0f} 원", cost_diff,
                    value_color=c_cost["value"], border_color=c_cost["border"], bg_color=c_cost["bg"]),
        metric_card("평균 단가", f"{avg_price:,.1f} 원/kWh", price_diff,
                    value_color=c_price["value"], border_color=c_price["border"], bg_color=c_price["bg"]),
        metric_card("탄소배출량", f"{carbon:,.2f} tCO₂", carbon_diff,
                    value_color=c_carbon["value"], border_color=c_carbon["border"], bg_color=c_carbon["bg"])
    ])

    components.html(f"""
        <div style="border:1.5px solid #ddd; border-radius:12px;
                    background-color:#fafafa; padding:25px; margin-top:10px;">
            <h3 style="text-align:center; font-weight:700; margin-bottom:20px;">{title}</h3>
            <div style="display:flex; justify-content:space-between;">{cards_html}</div>
        </div>
    """, height=280)

# =========================
# 메인 함수
# =========================
def show_tab_analysis(train):
    """전력 데이터 통합 분석 탭"""
    df = prepare_source_df(train)
    min_date, max_date = df[COL_TIME].min().date(), df[COL_TIME].max().date()

    col1, col2 = st.columns([1, 1])
    with col1:
        mode = st.selectbox("분석 기준", ["일별", "월별"])

    # ==================================================
    # 1️⃣ 주요지표 카드
    # ==================================================
    st.markdown("### 주요 지표")

    filtered_df = None

    if mode == "일별":
        # 기본값: 최근 7일
        default_start = (pd.Timestamp(max_date) - pd.DateOffset(days=6)).date()
        default_end   = max_date
        with col2:
            date_range = st.date_input(
                "기간 선택",
                value=(default_start, default_end),
                min_value=min_date,
                max_value=max_date
            )

        # 날짜 유효성 검사
        if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
            st.warning("📅 날짜 범위를 선택해주세요.")
            return

        start_date, end_date = date_range
        if start_date > end_date:
            start_date, end_date = end_date, start_date

        start_date = max(start_date, min_date)
        end_date   = min(end_date, max_date)

        # 현재 기간 데이터
        period_df = df[(df[COL_TIME].dt.date >= start_date) & (df[COL_TIME].dt.date <= end_date)]
        if period_df.empty:
            st.info(f"📭 {start_date} ~ {end_date} 구간에는 데이터가 없습니다.")
            return
        filtered_df = period_df

        # ==================================================
        # ✅ 전월 동일기간 계산 (연도 롤오버 + 예외처리)
        # ==================================================
        curr_start = pd.Timestamp(start_date)
        curr_end   = pd.Timestamp(end_date)

        # 현재 기간 기준 전월 동일 일자 계산
        prev_start = (curr_start - pd.DateOffset(months=1))
        prev_end   = (curr_end - pd.DateOffset(months=1))

        # 전월 일수가 더 짧을 경우 보정 (예: 31일 → 30일)
        prev_start = prev_start.replace(day=min(prev_start.days_in_month, curr_start.day))
        prev_end   = prev_end.replace(day=min(prev_end.days_in_month, curr_end.day))

<<<<<<< Updated upstream
        prev_df = df[(df[COL_TIME].dt.date >= prev_start.date()) & (df[COL_TIME].dt.date <= prev_end.date())]
=======
        render_metric_cards(period_df, prev_df, f"📆 {start_date} ~ {end_date} 기간 주요 지표")
        st.caption(f" 비교 구간: 전월 동일 기간 {prev_start} ~ {prev_end}")
>>>>>>> Stashed changes

        # ==================================================
        # ✅ 카드 렌더링 및 캡션 처리
        # ==================================================
        render_metric_cards(
            period_df,
            prev_df if not prev_df.empty else None,
            f"📆 {start_date} ~ {end_date} 기간 주요 지표"
        )

        if not prev_df.empty:
            st.caption(f"📊 비교 구간: 전월 동일 기간 {prev_start.date()} ~ {prev_end.date()}")
        else:
            st.caption(
                f"📊 {start_date} ~ {end_date} 구간의 전월 동일 기간({prev_start.date()} ~ {prev_end.date()}) 데이터가 "
                "존재하지 않아 증감률을 표시하지 않습니다."
            )
    else:
        with col2:
            sel_month = st.selectbox(
                "월 선택",
                sorted(df["월"].unique()),
                index=len(sorted(df["월"].unique())) - 1
            )

        curr_df = df[df["월"] == sel_month]
        if curr_df.empty:
            st.info(f"📭 {sel_month}월 데이터가 없습니다.")
            return
        filtered_df = curr_df

        # ==================================================
        # ✅ 전월 계산 (연도 롤오버 + 예외처리)
        # ==================================================
        if sel_month > 1:
            prev_month = sel_month - 1
            prev_df = df[df["월"] == prev_month]
            prev_label = f"{prev_month}월"
        else:
            # 1월 → 전년도 12월 데이터 탐색
            curr_year = df["연"].max()
            prev_year = curr_year - 1
            prev_df = df[(df["연"] == prev_year) & (df["월"] == 12)]
            prev_label = f"{prev_year}년 12월"

        # render
        render_metric_cards(
            curr_df,
            prev_df if not prev_df.empty else None,
            f"📆 {sel_month}월 주요 지표"
        )

        # ✅ 캡션 처리
        if not prev_df.empty:
            st.caption(f"📊 비교 구간: 전월({prev_label}) 대비 변화율")
        else:
            st.caption(f"📊 {sel_month}월은 전월 데이터가 존재하지 않아 증감률을 표시하지 않습니다.")
            
    # ---- Tab2 전용 wrapper 시작 ----
    st.markdown('<div class="tab2-scope">', unsafe_allow_html=True)

    # 세션 상태
    if "report_path_tab2" not in st.session_state:
        st.session_state["report_path_tab2"] = None

    # 보고서 생성 버튼
    if st.button("보고서 생성", key="report_generate_btn", use_container_width=True):
        from report_generator import generate_analysis_report
        with st.spinner("보고서를 생성 중입니다..."):
            file_name = f"./reports/electricity_report_{datetime.now().strftime('%Y%m%d_%H%M')}.docx"
            report_path = generate_analysis_report(df, filtered_df, output_path=file_name)
        st.session_state["report_path_tab2"] = report_path
        st.success("보고서 생성이 완료되었습니다.")

    # 다운로드 버튼
    if st.session_state["report_path_tab2"] and os.path.exists(st.session_state["report_path_tab2"]):
        with open(st.session_state["report_path_tab2"], "rb") as f:
            st.download_button(
                "보고서 다운로드",
                f,
                file_name=os.path.basename(st.session_state["report_path_tab2"]),
                key="report_download_btn",
                use_container_width=True
            )

    # ---- wrapper 종료 ----
    st.markdown("</div>", unsafe_allow_html=True)
    # ==================================================
    # 2️⃣ 요일·시간대별 평균 전력 사용량
    # ==================================================
    st.markdown("### 요일·시간대별 전력 사용 패턴")
    if filtered_df is None or filtered_df.empty:
        st.info("⚠️ 선택된 기간의 데이터가 없습니다.")
    else:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        # ✅ 영어 → 한국어 매핑
        weekday_map = {
            "Monday": "월요일",
            "Tuesday": "화요일",
            "Wednesday": "수요일",
            "Thursday": "목요일",
            "Friday": "금요일",
            "Saturday": "토요일",
            "Sunday": "일요일"
        }

        weekday_avg = (
            filtered_df.groupby("요일")[COL_USAGE]
            .mean()
            .reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
            .reset_index()
        )
        weekday_avg["요일(한글)"] = weekday_avg["요일"].map(weekday_map)

        hour_avg = filtered_df.groupby("시간")[COL_USAGE].mean().reset_index()

        fig_pattern = make_subplots(
            rows=1, cols=2,
            subplot_titles=("요일별 평균 전력사용량", "시간대별 평균 전력사용량"),
            horizontal_spacing=0.15
        )

        # ✅ 한글 요일 적용
        fig_pattern.add_trace(
            go.Bar(x=weekday_avg["요일(한글)"], y=weekday_avg[COL_USAGE],
                marker_color="#3A86FF"),
            row=1, col=1
        )

        fig_pattern.add_trace(
            go.Scatter(x=hour_avg["시간"], y=hour_avg[COL_USAGE],
                    mode="lines+markers",
                    line=dict(color="#FF006E", width=2)),
            row=1, col=2
        )

        fig_pattern.update_layout(
            showlegend=False,
            hovermode="x unified",
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig_pattern, use_container_width=True)

    # ==================================================
    # 3️⃣ 피크 수요 및 역률 분석
    # ==================================================
    st.markdown("### 피크 수요 및 역률 분석")

    if filtered_df is None or filtered_df.empty:
        st.info("⚠️ 선택된 기간의 데이터가 없습니다.")
    else:
        peak_row = filtered_df.loc[filtered_df[COL_DEMAND].idxmax()] if COL_DEMAND in filtered_df.columns else None
        peak_power = peak_row[COL_DEMAND] if peak_row is not None else np.nan
        peak_time = peak_row[COL_TIME] if peak_row is not None else None
        avg_lag_pf = filtered_df[COL_LAG_PF].mean() if COL_LAG_PF in filtered_df.columns else np.nan
        avg_lead_pf = filtered_df[COL_LEAD_PF].mean() if COL_LEAD_PF in filtered_df.columns else np.nan
        avg_pf = np.nanmean([avg_lag_pf, avg_lead_pf])

        if COL_DEMAND in filtered_df.columns:
            fig_peak = px.line(
                filtered_df, x=COL_TIME, y=COL_DEMAND,
                title="기간 내 전력 사용량 추이 (상위 피크 3개 강조)",
                labels={COL_TIME: "측정일시", COL_DEMAND: "수요전력(kW)"}
            )

            top3 = filtered_df.nlargest(3, COL_DEMAND)
            fig_peak.add_scatter(
                x=top3[COL_TIME], y=top3[COL_DEMAND],
                mode="markers+text",
                text=[f"피크{i+1}" for i in range(len(top3))],
                textposition="top center",
                marker=dict(size=12, symbol="circle", line=dict(width=1), opacity=1.0),
                name="상위 피크 (Top 3)"
            )

            fig_peak.update_layout(
                hovermode="x unified",
                template="plotly_white",
                legend_title="범례",
                plot_bgcolor="#fff",
                paper_bgcolor="#fff"
            )
            st.plotly_chart(fig_peak, use_container_width=True)
        
        if pd.isna(avg_pf):
            eff_label = "데이터 없음"
            color_code = {"value": "#999", "border": "#ccc", "bg": "#f6f6f6"}
        elif avg_pf >= 95:
            eff_label = "양호"
            color_code = {"value": "#28a745", "border": "#28a745", "bg": "#eaf8ea"}
        elif avg_pf >= 90:
            eff_label = "주의"
            color_code = {"value": "#ff9800", "border": "#ff9800", "bg": "#fff4e0"}
        else:
            eff_label = "개선 필요"
            color_code = {"value": "#dc3545", "border": "#dc3545", "bg": "#ffe8e8"}

        if peak_time is not None:
            if isinstance(peak_time, str):
                peak_time_str = peak_time
            else:
                try:
                    peak_time_str = peak_time.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    peak_time_str = str(peak_time)
        else:
            peak_time_str = "-"
            
        cards_html = "".join([
            metric_card("피크전력", f"{peak_power:,.1f} kW" if not np.isnan(peak_power) else "-"),
            metric_card("피크발생 시각", peak_time_str),
            metric_card("평균 지상역률", f"{avg_lag_pf:.1f} %" if not np.isnan(avg_lag_pf) else "-",
                        value_color=color_code["value"], border_color=color_code["border"], bg_color=color_code["bg"]),
            metric_card("평균 진상역률", f"{avg_lead_pf:.1f} %" if not np.isnan(avg_lead_pf) else "-",
                        value_color=color_code["value"], border_color=color_code["border"], bg_color=color_code["bg"]),
            metric_card("효율 등급", eff_label,
                        value_color=color_code["value"], border_color=color_code["border"], bg_color=color_code["bg"])
        ])

        components.html(f"""
            <div style="border:1.5px solid #ddd; border-radius:12px;
                        background-color:#fafafa; padding:20px; margin-top:10px;">
                <div style="display:flex; justify-content:space-between;">{cards_html}</div>
            </div>
        """, height=240)

    # ==================================================
    # ⚙️ 역률 시계열 그래프 (지상 / 진상 + 할인/할증 구간 표시)
    # ==================================================
    if COL_LAG_PF in filtered_df.columns or COL_LEAD_PF in filtered_df.columns:
        import plotly.graph_objects as go

        pf_df = filtered_df.copy().sort_values(COL_TIME)

        fig_pf = go.Figure()

        # ✅ 지상역률 (Lagging PF)
        if COL_LAG_PF in pf_df.columns:
            fig_pf.add_trace(go.Scatter(
                x=pf_df[COL_TIME],
                y=pf_df[COL_LAG_PF],
                mode="lines",
                name="지상역률(%)",
                line=dict(color="#1f77b4", width=2)
            ))

        # ✅ 진상역률 (Leading PF)
        if COL_LEAD_PF in pf_df.columns:
            fig_pf.add_trace(go.Scatter(
                x=pf_df[COL_TIME],
                y=pf_df[COL_LEAD_PF],
                mode="lines",
                name="진상역률(%)",
                line=dict(color="#ff7f0e", width=2)
            ))

        # 🎯 요금 영향 구간 (배경 표시)
        fig_pf.add_hrect(
            y0=95, y1=100,
            fillcolor="rgba(40,167,69,0.15)",  # 초록색 반투명 (할인 구간)
            layer="below", line_width=0,
            annotation_text="감면 구간 (95% 이상)", annotation_position="top left"
        )
        fig_pf.add_hrect(
            y0=0, y1=90,
            fillcolor="rgba(220,53,69,0.12)",  # 빨간색 반투명 (할증 구간)
            layer="below", line_width=0,
            annotation_text="할증 구간 (90% 미만)", annotation_position="bottom left"
        )

        # 🎯 기준선 (95%, 90%)
        fig_pf.add_hline(y=95, line_dash="dash", line_color="#28a745",
                            annotation_text="95%", annotation_position="top right")
        fig_pf.add_hline(y=90, line_dash="dash", line_color="#dc3545",
                            annotation_text="90%", annotation_position="bottom right")

        # 🎨 레이아웃
        fig_pf.update_layout(
            title="역률 변화 추이 (지상 / 진상)",
            xaxis_title="시간",
            yaxis_title="역률(%)",
            hovermode="x unified",
            template="plotly_white",
            legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
            height=420,
            plot_bgcolor="#fff",
            paper_bgcolor="#fff",
            margin=dict(t=80, b=60)
        )

        st.plotly_chart(fig_pf, use_container_width=True)
        st.caption(
            "💡 역률이 95% 이상이면 전기요금 감면, "
            "90% 미만이면 할증이 적용됩니다. "
            "그래프의 초록색 영역은 감면, 빨간색 영역은 할증 구간을 나타냅니다."
        )

        # ==================================================
        # ⚙️ 역률 적용 시간대별 작업유형 구성비 (지상 / 진상)
        # ==================================================
        st.markdown("#### 역률 적용 시간대별 작업유형 구성비")

        pf_select_ratio = st.selectbox(
            "역률 구분 선택",
            ["지상역률 적용 구간", "진상역률 적용 구간"],
            key="pf_ratio_select"
        )

        pf_df = filtered_df.copy()

        # ✅ 역률 구간 정의
        if pf_select_ratio == "지상역률 적용 구간":
            applied_df = pf_df[(pf_df["시간"] >= 9) & (pf_df["시간"] < 23)].copy()
            pf_label = "지상역률 적용 구간 (09시~23시)"
        else:
            applied_df = pf_df[(pf_df["시간"] >= 23) | (pf_df["시간"] < 9)].copy()
            pf_label = "진상역률 적용 구간 (23시~09시)"

        if applied_df.empty or "작업유형" not in applied_df.columns:
            st.warning(f"ℹ️ {pf_label} 데이터가 부족합니다.")
        else:
            # ✅ 시간대별 작업유형 비율 계산
            ratio_df = (
                applied_df.groupby(["시간", "작업유형"])
                .size().reset_index(name="건수")
            )
            ratio_df["비율(%)"] = ratio_df.groupby("시간")["건수"].transform(lambda x: x / x.sum() * 100)

            # 🎨 작업유형 색상 고정
            color_map = {
                "Light_Load": "#2ecc71",   # 초록
                "Medium_Load": "#f39c12",  # 주황
                "Maximum_Load": "#e74c3c"  # 빨강
            }

            # --------------------------------------------------
            # 🧩 두 그래프를 한 줄(row)에 배치
            # --------------------------------------------------
            col1, col2 = st.columns([2, 1])

            with col1:
                # 누적 막대 그래프
                fig_ratio = px.bar(
                    ratio_df,
                    x="시간",
                    y="비율(%)",
                    color="작업유형",
                    color_discrete_map=color_map,
                    title=f"{pf_label} 시간대별 작업유형 비율 (누적 막대)",
                    labels={"비율(%)": "작업유형 비율(%)"}
                )
                fig_ratio.update_layout(
                    barmode="stack",
                    template="plotly_white",
                    height=450,
                    xaxis=dict(dtick=1),
                    hovermode="x unified",
                    legend_title="작업유형"
                )
                st.plotly_chart(fig_ratio, use_container_width=True)

            with col2:
                # 전체 비중 도넛 차트
                total_ratio = (
                    applied_df["작업유형"].value_counts(normalize=True).mul(100).reset_index()
                )
                total_ratio.columns = ["작업유형", "비율(%)"]

                fig_donut = px.pie(
                    total_ratio,
                    values="비율(%)",
                    names="작업유형",
                    title=f"{pf_label} 전체 구성비",
                    color="작업유형",
                    color_discrete_map=color_map,
                    hole=0.45
                )
                fig_donut.update_layout(
                    template="plotly_white",
                    legend_title="작업유형",
                    height=450,
                    margin=dict(t=80, b=40, l=10, r=10)
                )
                st.plotly_chart(fig_donut, use_container_width=True)

            # --------------------------------------------------
            # 캡션
            # --------------------------------------------------
            st.caption(
                f"💡 {pf_label} 구간에서 시간대별로 수행된 작업유형의 비율을 나타냅니다. "
                "빨간색(Maximum_Load)은 피크 부하, 주황색(Medium_Load)은 중간 부하, "
                "초록색(Light_Load)은 저부하를 의미합니다. "
                "피크 부하 비율이 높은 구간은 역률 저하 가능성이 높습니다."
            )

    # ==================================================
    # 3.5️⃣ 시간대별 작업유형별 전기요금 현황 (누적 막대)
    # ==================================================
    st.markdown("### 시간대별 작업유형별 전기요금 현황")

    if filtered_df is None or filtered_df.empty:
        st.info("⚠️ 선택된 기간의 데이터가 없습니다.")
    elif "작업유형" not in filtered_df.columns:
        st.warning("ℹ️ '작업유형' 컬럼이 없어 그래프를 표시할 수 없습니다.")
    else:
        cost_by_type = (
            filtered_df.groupby(["시간", "작업유형"])[COL_COST]
            .sum()
            .reset_index()
        )

        fig_cost = px.bar(
            cost_by_type,
            x="시간",
            y=COL_COST,
            color="작업유형",
            title="시간대별 작업유형별 전기요금 현황 (누적 막대)",
            labels={COL_COST: "전기요금(원)", "시간": "시간대"},
            text_auto=".2s"
        )
        fig_cost.update_layout(
            barmode="stack",
            template="plotly_white",
            hovermode="x unified",
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            legend_title="작업유형",
            xaxis=dict(dtick=1),
            height=500
        )
        st.plotly_chart(fig_cost, use_container_width=True)

    # ==================================================
    # 4️⃣ 시계열 분석 (Range Slider)
    # ==================================================
    st.markdown("### 시계열 분석")

    metric_options = {
        "전력사용량(kWh)": COL_USAGE,
        "탄소배출량(tCO₂)": "탄소배출량",
        "지상역률(%)": COL_LAG_PF,
        "진상역률(%)": COL_LEAD_PF,
        "전기요금(원)": COL_COST
    }
    selected_label = st.selectbox("표시할 지표 선택", list(metric_options.keys()))
    selected_metric = metric_options[selected_label]

    ts_df = df.copy()
    if selected_metric == "탄소배출량":
        ts_df["탄소배출량"] = ts_df[COL_USAGE] * 0.000331
    ts_agg = ts_df.groupby(COL_TIME)[selected_metric].mean().reset_index()

    fig_ts = px.line(ts_agg, x=COL_TIME, y=selected_metric,
                     title=f"{selected_label} 시계열 추이")
    fig_ts.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="1주", step="day", stepmode="backward"),
                    dict(count=30, label="1개월", step="day", stepmode="backward"),
                    dict(count=90, label="3개월", step="day", stepmode="backward"),
                    dict(step="all", label="전체")
                ])
            ),
            rangeslider=dict(visible=True, bgcolor="#f0f0f0", bordercolor="#aaa", borderwidth=2, thickness=0.1),
            type="date"
        ),
        hovermode="x unified",
        template="plotly_white",
        plot_bgcolor="#fff",
        paper_bgcolor="#fff"
    )
    fig_ts.update_traces(line=dict(width=1.6, color="#007bff"))
    st.plotly_chart(fig_ts, use_container_width=True)

