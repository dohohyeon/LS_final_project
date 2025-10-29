import streamlit as st
import pandas as pd
import plotly.express as px
import time

st.set_page_config(page_title="전기요금 실시간 스트리밍", layout="wide")
st.title("⚡ Train 데이터 기반 전기요금 실시간 스트리밍 시뮬레이션")

@st.cache_data
def load_train():
    df = pd.read_csv("./data/raw/train.csv")
    df.columns = df.columns.str.strip()
    if "측정일시" in df.columns:
        df["측정일시"] = pd.to_datetime(df["측정일시"], errors="coerce")
    return df

train = load_train()
usage_col = next((c for c in train.columns if "전력" in c and "사용" in c), None)
if usage_col is None:
    st.error("⚠️ 전력 사용량 관련 컬럼을 찾을 수 없습니다.")
    st.stop()

st.session_state.setdefault("running", False)
st.session_state.setdefault("index", 0)
st.session_state.setdefault("stream_df", pd.DataFrame())
st.session_state.setdefault("chart_placeholder", st.empty())
st.session_state.setdefault("table_placeholder", st.empty())

st.sidebar.header("⚙️ 스트리밍 제어")
speed = st.sidebar.slider("업데이트 간격(초)", 0.2, 10.0, 1.0, 0.2)
col1, col2 = st.sidebar.columns(2)

# ✅ 한 번만 눌러도 바로 반응
if not st.session_state.running:
    if col1.button("▶ 시작"):
        st.session_state.running = True
else:
    if col1.button("⏸ 정지"):
        st.session_state.running = False

if col2.button("🔄 초기화"):
    st.session_state.index = 0
    st.session_state.stream_df = pd.DataFrame()
    st.session_state.running = False

st.sidebar.write("상태:", "🟢 실행 중" if st.session_state.running else "🔴 정지됨")

def update_dashboard(df_partial: pd.DataFrame):
    if df_partial.empty:
        st.session_state.chart_placeholder.info("데이터가 아직 없습니다.")
        return
    fig = px.line(df_partial, x="측정일시", y=usage_col, title="실시간 전력 사용량 추이", markers=True)
    st.session_state.chart_placeholder.plotly_chart(fig, use_container_width=True)
    st.session_state.table_placeholder.dataframe(df_partial.tail(10), use_container_width=True, hide_index=True)

if st.session_state.running:
    for i in range(st.session_state.index, len(train)):
        if not st.session_state.running:
            break
        row = train.iloc[i]
        st.session_state.stream_df = pd.concat([st.session_state.stream_df, pd.DataFrame([row])], ignore_index=True)
        st.session_state.index = i + 1
        update_dashboard(st.session_state.stream_df)
        time.sleep(speed)
    if st.session_state.index >= len(train):
        st.session_state.running = False
        st.success("✅ 모든 데이터 표시 완료.")
else:
    update_dashboard(st.session_state.stream_df)
    if st.session_state.stream_df.empty:
        st.info("스트리밍 대기 중입니다. ▶ 시작 버튼을 눌러주세요.")
