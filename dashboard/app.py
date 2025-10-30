# app.py
import streamlit as st
from shared import load_train
from modules.tab_0 import show_tab_home
from modules.tab_1 import show_tab_realtime
from modules.tab_2 import show_tab_analysis
from modules.tab_3 import show_tab_appendix

st.set_page_config(page_title="전력 모니터링 대시보드", layout="wide")

# -----------------------------
# 데이터 로드
# -----------------------------
train = load_train()
if train.empty:
    st.stop()

# -----------------------------
# ✅ 사이드바 (tab_realtime.py에서 옮김)
# -----------------------------
st.sidebar.header("⚙️ 스트리밍 제어")

# 세션 상태 초기화
st.session_state.setdefault("running", False)
st.session_state.setdefault("index", 0)
st.session_state.setdefault("stream_df", train.iloc[0:0].copy())

# 사이드바 위젯
speed = st.sidebar.slider("업데이트 간격(초)", 0.1, 5.0, 0.5, 0.1)
col1, col2 = st.sidebar.columns(2)

# 버튼 제어
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
    st.session_state.stream_df = train.iloc[0:0].copy()
    st.session_state.running = False
    st.rerun()

# 상태 표시
st.sidebar.write("상태:", "🟢 실행 중" if st.session_state.running else "🔴 정지됨")

# -----------------------------
# 탭 구성 (HOME → 실시간 → 통계 → 부록)
# -----------------------------
tab_pred, tab_rt, tab_viz, tab_appendix = st.tabs([
    " HOME",
    " 실시간 모니터링",
    " 통계 분석",
    " 부록"
])

with tab_pred:
    show_tab_home(train)

with tab_rt:
    # ✅ 사이드바 변수 전달
    show_tab_realtime(train, speed)

with tab_viz:
    show_tab_analysis(train)

with tab_appendix:
    show_tab_appendix(train)
