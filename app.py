import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time

# =========================
# 고정 컬럼명 (제공 스키마)
# =========================
COL_TIME   = "측정일시"
COL_USAGE  = "전력사용량(kWh)"
COL_COST   = "전기요금(원)"
COL_JOB    = "작업유형"

# =========================
# 기본 설정
# =========================
st.set_page_config(page_title="전기요금 실시간 스트리밍 & 요약 대시보드", layout="wide")

# =========================
# 데이터 적재/전처리
# =========================
@st.cache_data
def load_train(path="./data/raw/train.csv"):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # 필수 컬럼 확인
    for c in [COL_TIME, COL_USAGE]:
        if c not in df.columns:
            raise ValueError(f"필수 컬럼 누락: {c}")

    # 시간 파싱 및 정렬
    df[COL_TIME] = pd.to_datetime(df[COL_TIME], errors="coerce")
    df = df.dropna(subset=[COL_TIME]).sort_values(COL_TIME)

    # 파생: 월/요일/시간
    df["월"] = df[COL_TIME].dt.to_period("M").astype(str)  # YYYY-MM
    weekday_map = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}
    df["요일"] = df[COL_TIME].dt.weekday.map(weekday_map)
    df["시간"] = df[COL_TIME].dt.hour

    # 누락 가능 컬럼 안전 처리
    if COL_COST not in df.columns:
        df[COL_COST] = np.nan
    if COL_JOB not in df.columns:
        df[COL_JOB] = "미지정"

    # 작업유형 NaN → '미지정'
    df[COL_JOB] = df[COL_JOB].fillna("미지정")

    return df

try:
    train = load_train()
except Exception as e:
    st.error(f"⚠️ 데이터 로드 오류: {e}")
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
        end_date   = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        out = out[(out[COL_TIME] >= start_date) & (out[COL_TIME] <= end_date)]
    return out

def metric_label(col):
    return "전력사용량(kWh)" if col == COL_USAGE else "전기요금(원)"

# =========================
# 탭 구성
# =========================
tab_rt, tab_viz = st.tabs(["📡 실시간 스트리밍", "📈 분석·시각화"])

# -------------------------
# 탭1: 실시간 스트리밍
# -------------------------
with tab_rt:
    st.title("⚡ 전기요금 실시간 스트리밍")

    # 탭1 전용 placeholder
    chart_ph = st.empty()
    table_ph = st.empty()

    # 사이드바: 스트리밍 제어만 유지 (필터는 탭2에 배치)
    st.sidebar.header("⚙️ 스트리밍 제어")
    speed = st.sidebar.slider("업데이트 간격(초)", 0.2, 10.0, 1.0, 0.2)
    col1, col2 = st.sidebar.columns(2)

    if not st.session_state.running:
        if col1.button("▶ 시작"):
            st.session_state.running = True
    else:
        if col1.button("⏸ 정지"):
            st.session_state.running = False

    if col2.button("🔄 초기화"):
        st.session_state.index = 0
        st.session_state.stream_df = pd.DataFrame(columns=train.columns)
        st.session_state.running = False

    st.sidebar.write("상태:", "🟢 실행 중" if st.session_state.running else "🔴 정지됨")

    # 실시간 업데이트 함수
    def update_realtime_chart(df_partial: pd.DataFrame):
        if df_partial.empty:
            chart_ph.info("데이터가 아직 없습니다.")
            return
        fig = px.line(df_partial, x=COL_TIME, y=COL_USAGE,
                      title="실시간 전력사용량(kWh) 추이", markers=True)
        chart_ph.plotly_chart(fig, use_container_width=True)

        cols_to_show = [COL_TIME, COL_USAGE]
        if COL_COST in df_partial.columns:
            cols_to_show.append(COL_COST)

        table_ph.dataframe(
            df_partial[cols_to_show].tail(10),
            use_container_width=True, hide_index=True
        )

    # 스트리밍 루프
    if st.session_state.running:
        for i in range(st.session_state.index, len(train)):
            if not st.session_state.running:
                break
            row = train.iloc[i]
            st.session_state.stream_df = pd.concat(
                [st.session_state.stream_df, pd.DataFrame([row])],
                ignore_index=True
            )
            st.session_state.index = i + 1
            update_realtime_chart(st.session_state.stream_df)
            time.sleep(speed)
        if st.session_state.index >= len(train):
            st.session_state.running = False
            st.success("✅ 모든 데이터 표시 완료.")
    else:
        update_realtime_chart(st.session_state.stream_df)

# -------------------------
# 탭2: 분석·시각화
# -------------------------
with tab_viz:
    st.title("📈 전력사용량·전기요금 분석 / 시각화")

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

    # 작업유형
    with c3:
        job_values = sorted(train[COL_JOB].dropna().unique().tolist())
        jobs_selected = st.multiselect("작업유형", job_values, default=job_values)

    # 지표 선택(전력사용량 / 전기요금)
    with c4:
        metric_col = st.radio("지표 선택", [COL_USAGE, COL_COST],
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

    # ========================
    # 섹션 1: 월/요일 요약 (단일 지표)
    # ========================
    st.markdown(f"### 📅 월·요일 요약 — **{metric_label(metric_col)}**")

    # 월별 합계
    monthly = df_f.groupby("월", as_index=False)[metric_col].sum()
    fig_m = px.bar(monthly, x="월", y=metric_col, title=f"월별 합계 — {metric_label(metric_col)}")
    st.plotly_chart(fig_m, use_container_width=True)

    # 요일별 합계 (요일 순서 고정)
    order = ["월", "화", "수", "목", "금", "토", "일"]
    weekly = df_f.groupby("요일", as_index=False)[metric_col].sum()
    weekly["요일"] = pd.Categorical(weekly["요일"], categories=order, ordered=True)
    weekly = weekly.sort_values("요일")
    fig_w = px.bar(weekly, x="요일", y=metric_col, title=f"요일별 합계 — {metric_label(metric_col)}")
    st.plotly_chart(fig_w, use_container_width=True)

    # ========================
    # 섹션 2: 시간대 분포 (작업유형별 누적막대)
    # ========================
    st.markdown(f"### ⏱ 시간대별 작업유형별 {metric_label(metric_col)} — 누적막대")

    hour_job = (
        df_f
        .groupby(["시간", COL_JOB], dropna=False)[metric_col]
        .sum()
        .reset_index()
    )
    # 시간 0~23 전체 보장
    all_hours = pd.DataFrame({"시간": np.arange(24)})
    hour_job = all_hours.merge(hour_job, on="시간", how="left")
    hour_job[COL_JOB] = hour_job[COL_JOB].fillna("미지정")
    hour_job[metric_col] = hour_job[metric_col].fillna(0)

    fig_stack = px.bar(
        hour_job, x="시간", y=metric_col, color=COL_JOB,
        barmode="stack", title=f"시간대별 작업유형별 {metric_label(metric_col)} 현황"
    )
    st.plotly_chart(fig_stack, use_container_width=True)

    # ========================
    # 섹션 3: 일별 추이 (선 그래프, 단일 지표)
    # ========================
    st.markdown(f"### 📆 일별 합계 추이 — {metric_label(metric_col)}")

    df_day = df_f.copy()
    df_day["date"] = df_day[COL_TIME].dt.date
    daily = df_day.groupby("date", as_index=False)[metric_col].sum()
    fig_daily = px.line(daily, x="date", y=metric_col, markers=True,
                        title=f"일별 합계 추이 — {metric_label(metric_col)}")
    st.plotly_chart(fig_daily, use_container_width=True)

    # KPI
    total_val = daily[metric_col].sum()
    kpi_unit = "kWh" if metric_col == COL_USAGE else "원"
    st.metric(f"선택 구간 총 {metric_label(metric_col)}", f"{total_val:,.2f} {kpi_unit}" if kpi_unit=="kWh" else f"{total_val:,.0f} {kpi_unit}")
