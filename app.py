import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time

# =========================
# 고정 컬럼명 (제공 스키마)
# =========================
COL_TIME = "측정일시"
COL_USAGE = "전력사용량(kWh)"
COL_COST = "전기요금(원)"
COL_JOB = "작업유형"
# --- 신규 컬럼 ---
COL_DEMAND = "수요전력(kW)" # kWh * 4 (15분 데이터 기준)
COL_PF = "지상역률(%)"

# =========================
# 상수 정의
# =========================
# 공장 관리자가 설정해야 하는 목표치 (예시)
PEAK_DEMAND_THRESHOLD = 30.0  # 목표 피크 (kW) - 이 값을 넘지 않도록 관리
POWER_FACTOR_THRESHOLD = 90.0 # 역률 한계선 (%) - 이 값 미만 시 패널티

# =========================
# 기본 설정
# =========================
st.set_page_config(page_title="실시간 전력 모니터링 대시보드", layout="wide")

# =========================
# 데이터 적재/전처리
# =========================
@st.cache_data
def load_train(path="./data/raw/train.csv"):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"오류: {path} 파일을 찾을 수 없습니다. 스크립트와 동일한 위치에 'data/raw/train.csv' 파일이 있는지 확인하세요.")
        return pd.DataFrame() # 빈 데이터프레임 반환

    df.columns = df.columns.str.strip()

    # 필수 컬럼 확인
    for c in [COL_TIME, COL_USAGE]:
        if c not in df.columns:
            st.error(f"필수 컬럼 누락: {c}. CSV 파일을 확인하세요.")
            return pd.DataFrame()

    # 시간 파싱 및 정렬
    df[COL_TIME] = pd.to_datetime(df[COL_TIME], errors="coerce")
    df = df.dropna(subset=[COL_TIME]).sort_values(COL_TIME)

    # 파생: 월/요일/시간
    df["월"] = df[COL_TIME].dt.to_period("M").astype(str)  # YYYY-MM
    weekday_map = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}
    df["요일"] = df[COL_TIME].dt.weekday.map(weekday_map)
    df["시간"] = df[COL_TIME].dt.hour

    # --- 핵심 수정 ---
    # 1. 수요전력(kW) 계산: 15분 단위 kWh * 4 = 15분 평균 kW
    df[COL_DEMAND] = df[COL_USAGE] * 4

    # 2. 역률(%) 안전 처리 (실제 데이터가 있으면 이 부분은 무시됨)
    if COL_PF not in df.columns:
        st.warning(f"'{COL_PF}' 컬럼이 없어 임의의 샘플 데이터를 생성합니다. (88% ~ 99%)")
        df[COL_PF] = np.random.uniform(88.0, 99.5, size=len(df)).round(2)
    else:
        df[COL_PF] = pd.to_numeric(df[COL_PF], errors='coerce').fillna(95.0) # 기본값 95

    # 3. 전기요금(원) 안전 처리
    if COL_COST not in df.columns:
        st.warning(f"'{COL_COST}' 컬럼이 없어 임의 계산합니다. (kWh * 150원 가정)")
        df[COL_COST] = df[COL_USAGE] * 150 # 임의의 단가
    else:
        df[COL_COST] = pd.to_numeric(df[COL_COST], errors='coerce').fillna(0)

    # 4. 작업유형 NaN -> '미지정'
    if COL_JOB not in df.columns:
        df[COL_JOB] = "미지정"
    df[COL_JOB] = df[COL_JOB].fillna("미지정")

    return df

# 데이터 로드
train = load_train()

if train.empty:
    st.error("데이터 로드에 실패하여 앱을 중지합니다.")
    st.stop()

# =========================
# 스트리밍 상태(탭1에서 사용)
# =========================
st.session_state.setdefault("running", False)
st.session_state.setdefault("index", 0)
st.session_state.setdefault("stream_df", pd.DataFrame(columns=train.columns))

# =========================
# 공용 함수
# =========================
def apply_filters(df, jobs_selected, date_range):
    out = df.copy()
    if jobs_selected:
        out = out[out[COL_JOB].isin(jobs_selected)]
    # 기간 필터
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
        COL_PF: "역률(%)"
    }
    return labels.get(col, col)

def get_agg_func(metric_col):
    """지표에 따라 적절한 집계 함수 반환 (합계, 최대, 평균)"""
    if metric_col in [COL_USAGE, COL_COST]:
        return "sum"
    elif metric_col == COL_DEMAND:
        return "max" # 수요전력은 '최대 피크'가 중요
    elif metric_col == COL_PF:
        return "mean" # 역률은 '평균'이 중요
    return "sum"

# =========================
# 탭 구성
# =========================
tab_rt, tab_viz = st.tabs(["📡 실시간 모니터링", "📈 통계 분석"])

# -------------------------
# 탭1: 실시간 스트리밍
# -------------------------
with tab_rt:
    st.title("⚡ 실시간 전력 모니터링")

    # 탭1 전용 placeholder
    kpi_ph = st.empty()
    chart_demand_ph = st.empty()
    chart_pf_ph = st.empty()
    table_ph = st.empty()

    # 사이드바: 스트리밍 제어
    st.sidebar.header("⚙️ 스트리밍 제어")
    speed = st.sidebar.slider("업데이트 간격(초)", 0.1, 5.0, 0.5, 0.1)
    col1, col2 = st.sidebar.columns(2)

    if not st.session_state.running:
        if col1.button("▶ 시작"):
            st.session_state.running = True
            st.rerun() # 즉시 반영
    else:
        if col1.button("⏸ 정지"):
            st.session_state.running = False
            st.rerun() # 즉시 반영

    if col2.button("🔄 초기화"):
        st.session_state.index = 0
        st.session_state.stream_df = pd.DataFrame(columns=train.columns)
        st.session_state.running = False
        st.rerun() # 즉시 반영

    st.sidebar.write("상태:", "🟢 실행 중" if st.session_state.running else "🔴 정지됨")

    # 실시간 업데이트 함수 (KPI + 차트 2개 + 테이블)
    def update_realtime_dashboard(df_partial: pd.DataFrame):
        if df_partial.empty:
            kpi_ph.info("데이터가 아직 없습니다. '시작' 버튼을 눌러주세요.")
            return

        # --- 1. KPI 업데이트 ---
        latest_row = df_partial.iloc[-1]
        latest_time = latest_row[COL_TIME]

        # 현재 월/일 데이터 필터링
        today_data = df_partial[df_partial[COL_TIME].dt.date == latest_time.date()]
        month_data = df_partial[df_partial[COL_TIME].dt.month == latest_time.month]

        # KPI 계산
        current_demand = latest_row[COL_DEMAND]
        current_pf = latest_row[COL_PF]
        today_peak = today_data[COL_DEMAND].max()
        month_peak = month_data[COL_DEMAND].max()
        month_usage = month_data[COL_USAGE].sum()
        month_cost = month_data[COL_COST].sum()

        # KPI 델타 계산 (목표치/한계선 대비)
        demand_delta = current_demand - PEAK_DEMAND_THRESHOLD
        pf_delta = current_pf - POWER_FACTOR_THRESHOLD

        with kpi_ph.container():
            kpi_cols = st.columns(6)
            kpi_cols[0].metric(
                label="실시간 수요전력 (kW)",
                value=f"{current_demand:,.1f}",
                delta=f"{demand_delta:,.1f} (목표: {PEAK_DEMAND_THRESHOLD})",
                delta_color="inverse" # 높으면 안좋음
            )
            kpi_cols[1].metric(
                label="금일 최대 피크 (kW)",
                value=f"{today_peak:,.1f}"
            )
            kpi_cols[2].metric(
                label="당월 최대 피크 (kW)",
                value=f"{month_peak:,.1f}"
            )
            kpi_cols[3].metric(
                label="실시간 역률 (%)",
                value=f"{current_pf:,.1f}",
                delta=f"{pf_delta:,.1f} (한계: {POWER_FACTOR_THRESHOLD})",
                delta_color="normal" # 높으면 좋음 (한계선 미만 시 inverse)
            )
            kpi_cols[4].metric(
                label="당월 누적 사용량 (kWh)",
                value=f"{month_usage:,.0f}"
            )
            kpi_cols[5].metric(
                label="당월 누적 요금 (원)",
                value=f"{month_cost:,.0f}"
            )

        # --- 2. 수요전력(kW) 차트 ---
        fig_demand = px.line(df_partial, x=COL_TIME, y=COL_DEMAND,
                            title="실시간 수요전력(kW) 추이", markers=True)
        # 목표 피크 한계선 추가
        fig_demand.add_hline(
            y=PEAK_DEMAND_THRESHOLD,
            line_dash="dash", line_color="red",
            annotation_text="목표 피크",
            annotation_position="bottom right"
        )
        fig_demand.update_yaxes(rangemode="tozero") # Y축을 0부터 시작
        chart_demand_ph.plotly_chart(fig_demand, use_container_width=True)

        # --- 3. 역률(%) 차트 ---
        fig_pf = px.line(df_partial, x=COL_TIME, y=COL_PF,
                         title="실시간 역률(%) 추이", markers=True)
        # 역률 한계선 추가
        fig_pf.add_hline(
            y=POWER_FACTOR_THRESHOLD,
            line_dash="dash", line_color="red",
            annotation_text="역률 한계",
            annotation_position="bottom right"
        )
        fig_pf.update_yaxes(range=[min(80, df_partial[COL_PF].min() - 2), 101]) # Y축 범위 지정
        chart_pf_ph.plotly_chart(fig_pf, use_container_width=True)


        # --- 4. 데이터 테이블 ---
        cols_to_show = [COL_TIME, COL_DEMAND, COL_USAGE, COL_PF, COL_COST]
        # 작업유형이 의미있는 데이터인 경우만 포함
        if df_partial[COL_JOB].nunique() > 1:
            cols_to_show.append(COL_JOB)

        table_ph.dataframe(
            df_partial[cols_to_show].sort_values(COL_TIME, ascending=False).head(10), # 최신순 10개
            use_container_width=True,
            hide_index=True,
            column_config={
                COL_TIME: st.column_config.DatetimeColumn("측정일시", format="YYYY-MM-DD HH:mm:ss"),
                COL_DEMAND: st.column_config.NumberColumn("수요전력(kW)", format="%.2f kW"),
                COL_USAGE: st.column_config.NumberColumn("전력사용량(kWh)", format="%.2f kWh"),
                COL_PF: st.column_config.NumberColumn("역률(%)", format="%.1f %%"),
                COL_COST: st.column_config.NumberColumn("전기요금(원)", format="%d 원"),
            }
        )

    # 스트리밍 루프
    if st.session_state.running:
        for i in range(st.session_state.index, len(train)):
            if not st.session_state.running: # 정지 버튼 체크
                break
            row = train.iloc[i:i+1] # 데이터프레임 형태 유지
            st.session_state.stream_df = pd.concat(
                [st.session_state.stream_df, row],
                ignore_index=True
            )
            st.session_state.index = i + 1
            
            # 너무 빠르면 차트가 깜빡이므로 최소 딜레이 보장
            time.sleep(max(0.05, speed)) 
            
            # 마지막 행에서만 차트 업데이트 (성능 최적화)
            if (i % 1 == 0) or (i == len(train) - 1): # 지금은 매번 업데이트 (speed로 조절)
                update_realtime_dashboard(st.session_state.stream_df)

        if st.session_state.index >= len(train):
            st.session_state.running = False
            st.success("✅ 모든 데이터 스트리밍 완료.")
            st.rerun()
    else:
        # 정지 상태일 때도 현재까지의 데이터로 대시보드 표시
        update_realtime_dashboard(st.session_state.stream_df)


# -------------------------
# 탭2: 통계 분석
# -------------------------
with tab_viz:
    st.title("📈 전력 데이터 통계 분석")

    # ---- 상단 필터 UI (그래프 영역) ----
    st.markdown("#### 🔎 필터")
    c1, c2, c3, c4 = st.columns([2, 3, 2, 2])

    # 데이터 소스
    with c1:
        data_source = st.radio("데이터 소스", ["스트리밍 누적", "전체 데이터"], index=1, horizontal=True)

    # 기간
    min_d = train[COL_TIME].min().date()
    max_d = train[COL_TIME].max().date()
    with c2:
        date_range = st.date_input("기간 선택", (min_d, max_d))
        if len(date_range) != 2: # 초기 로드 시 예외 처리
            st.stop()

    # 작업유형
    with c3:
        job_values = sorted(train[COL_JOB].dropna().unique().tolist())
        jobs_selected = st.multiselect("작업유형", job_values, default=job_values)

    # 지표 선택(전력사용량 / 수요전력 / 역률 / 전기요금)
    with c4:
        # 사용 가능한 모든 지표
        metric_options = [COL_USAGE, COL_DEMAND, COL_PF, COL_COST]
        metric_col = st.radio("지표 선택", metric_options,
                                format_func=metric_label, horizontal=True)

    # 소스 데이터 선택
    if data_source == "스트리밍 누적" and not st.session_state.stream_df.empty:
        base_df = st.session_state.stream_df.copy()
    else:
        base_df = train.copy()

    # 필터 적용
    df_f = apply_filters(base_df, jobs_selected, date_range)

    if df_f.empty:
        st.info("선택한 조건에 해당하는 데이터가 없습니다.")
        st.stop()

    # --- 지표별 집계 함수/단위 설정 ---
    agg_func = get_agg_func(metric_col)
    kpi_title = f"선택 구간 {metric_label(metric_col)} ({agg_func})"
    
    unit_map = {
        COL_USAGE: "kWh",
        COL_DEMAND: "kW",
        COL_PF: "%",
        COL_COST: "원"
    }
    kpi_unit = unit_map.get(metric_col, "")
    
    # ========================
    # 섹션 1: 월/요일 요약 (단일 지표)
    # ========================
    st.markdown(f"### 📅 월·요일별 {metric_label(metric_col)} (집계: {agg_func})")

    c1_viz, c2_viz = st.columns(2)
    
    # 월별 집계
    with c1_viz:
        monthly = df_f.groupby("월", as_index=False)[metric_col].agg(agg_func)
        fig_m = px.bar(monthly, x="월", y=metric_col, title=f"월별 {metric_label(metric_col)} ({agg_func})")
        st.plotly_chart(fig_m, use_container_width=True)

    # 요일별 집계 (요일 순서 고정)
    with c2_viz:
        order = ["월", "화", "수", "목", "금", "토", "일"]
        weekly = df_f.groupby("요일", as_index=False)[metric_col].agg(agg_func)
        weekly["요일"] = pd.Categorical(weekly["요일"], categories=order, ordered=True)
        weekly = weekly.sort_values("요일")
        fig_w = px.bar(weekly, x="요일", y=metric_col, title=f"요일별 {metric_label(metric_col)} ({agg_func})")
        st.plotly_chart(fig_w, use_container_width=True)

    # ========================
    # 섹션 2: 시간대 분포 (작업유형별 누적막대)
    # ========================
    st.markdown(f"### ⏱ 시간대별 작업유형별 {metric_label(metric_col)} (집계: {agg_func})")

    hour_job = (
        df_f
        .groupby(["시간", COL_JOB], dropna=False)[metric_col]
        .agg(agg_func)
        .reset_index()
    )
    # 시간 0~23 전체 보장
    all_hours = pd.DataFrame({"시간": np.arange(24)})
    hour_job = all_hours.merge(hour_job, on="시간", how="left")
    hour_job[COL_JOB] = hour_job[COL_JOB].fillna("미지정")
    hour_job[metric_col] = hour_job[metric_col].fillna(0)

    fig_stack = px.bar(
        hour_job, x="시간", y=metric_col, color=COL_JOB,
        barmode="stack", title=f"시간대별 작업유형별 {metric_label(metric_col)} 현황 ({agg_func})"
    )
    st.plotly_chart(fig_stack, use_container_width=True)

    # ========================
    # 섹션 3: 일별 추이 (선 그래프, 단일 지표)
    # ========================
    st.markdown(f"### 📆 일별 {metric_label(metric_col)} 추이 (집계: {agg_func})")

    df_day = df_f.copy()
    df_day["date"] = df_day[COL_TIME].dt.date
    daily = df_day.groupby("date", as_index=False)[metric_col].agg(agg_func)
    fig_daily = px.line(daily, x="date", y=metric_col, markers=True,
                        title=f"일별 {metric_label(metric_col)} 추이 ({agg_func})")
    st.plotly_chart(fig_daily, use_container_width=True)

    # 최종 요약 KPI
    if agg_func == "sum":
        total_val = daily[metric_col].sum()
        kpi_title = f"선택 구간 총 {metric_label(metric_col)}"
    elif agg_func == "max":
        total_val = daily[metric_col].max()
        kpi_title = f"선택 구간 최고 {metric_label(metric_col)}"
    elif agg_func == "mean":
        total_val = daily[metric_col].mean()
        kpi_title = f"선택 구간 평균 {metric_label(metric_col)}"

    st.metric(kpi_title, f"{total_val:,.2f} {kpi_unit}")
