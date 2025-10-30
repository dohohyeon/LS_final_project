# app.py
import json
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from modules.tab_0 import show_tab_home
from modules.tab_1 import show_tab_realtime
from modules.tab_2 import show_tab_analysis
from modules.tab_3 import show_tab_appendix
from shared import load_train

st.set_page_config(page_title="전력 모니터링 대시보드", layout="wide")

# -----------------------------
# ✅ 탭 중앙 정렬 및 로고 위치 CSS
# -----------------------------
st.markdown("""
    <style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');

    html, body, [class*="css"] {
        font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
    }
    
    /* 메인 배경색 변경 */
    .main {
        background-color: #ffffff !important;
    }
    
    /* 전체 앱 배경색 */
    .stApp {
        background-color: #ffffff !important;
    }
    
    /* 메인 컨텐츠 영역 배경색 */
    [data-testid="stAppViewContainer"] {
        background-color: #ffffff !important;
    }
    
    /* 헤더 영역 배경색 */
    [data-testid="stHeader"] {
        background-color: #ffffff !important;
    }
    
    /* 블록 컨테이너 배경색 */
    .block-container {
        background-color: #ffffff !important;
    }

    /* 사이드바 배경색 변경 */
    [data-testid="stSidebar"] {
        background-color: #0a1f32 !important;
    }
    
    /* 사이드바 텍스트 색상 (가독성을 위해 흰색으로) */
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    /* 사이드바 헤더 가운데 정렬 + 구분선 */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        text-align: center !important;
        padding: 10px 0 15px 0 !important;
        border-bottom: 2px solid #ffffff !important;
        margin-bottom: 20px !important;
    }
    
    /* 사이드바 selectbox 스타일 */
    [data-testid="stSidebar"] .stSelectbox > label {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        margin-bottom: 8px !important;
    }
    
    /* selectbox 드롭다운 */
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox > div > div:hover {
        background: rgba(255, 255, 255, 0.15) !important;
        border-color: rgba(255, 255, 255, 0.5) !important;
    }
    
    /* selectbox 선택된 값 텍스트 */
    [data-testid="stSidebar"] .stSelectbox input {
        color: #ffffff !important;
    }
    
    /* selectbox 화살표 아이콘 */
    [data-testid="stSidebar"] .stSelectbox svg {
        fill: #ffffff !important;
    }
    
    /* 시작 버튼 (secondary) - 밝은 회색 + 검정 글자 */
    [data-testid="stSidebar"] button[kind="secondary"],
    [data-testid="stSidebar"] .stButton button[kind="secondary"],
    [data-testid="stSidebar"] div[data-testid="stVerticalBlock"] button[kind="secondary"] {
        background: #DCDCD5 !important;
        color: #000000 !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 10px 16px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2) !important;
    }

    [data-testid="stSidebar"] button[kind="secondary"]:hover,
    [data-testid="stSidebar"] .stButton button[kind="secondary"]:hover,
    [data-testid="stSidebar"] div[data-testid="stVerticalBlock"] button[kind="secondary"]:hover {
        background: #c8c8c1 !important;
        color: #000000 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3) !important;
    }

    /* 정지 버튼 (primary) - 진한 회색 + 검정 글자 */
    [data-testid="stSidebar"] button[kind="primary"],
    [data-testid="stSidebar"] .stButton button[kind="primary"],
    [data-testid="stSidebar"] div[data-testid="stVerticalBlock"] button[kind="primary"] {
        background: #8B8B8B !important;
        color: #000000 !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 10px 16px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2) !important;
    }

    [data-testid="stSidebar"] button[kind="primary"]:hover,
    [data-testid="stSidebar"] .stButton button[kind="primary"]:hover,
    [data-testid="stSidebar"] div[data-testid="stVerticalBlock"] button[kind="primary"]:hover {
        background: #757575 !important;
        color: #000000 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3) !important;
    }

    /* 버튼 내부 텍스트도 강제로 검정색 */
    [data-testid="stSidebar"] button[kind="secondary"] p,
    [data-testid="stSidebar"] button[kind="secondary"] span,
    [data-testid="stSidebar"] button[kind="secondary"] div,
    [data-testid="stSidebar"] button[kind="primary"] p,
    [data-testid="stSidebar"] button[kind="primary"] span,
    [data-testid="stSidebar"] button[kind="primary"] div {
        color: #000000 !important;
    }

    /* 탭 컨테이너를 위로 강제 이동 */
    div[data-baseweb="tab-list"] {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-top: -80px !important;
        position: relative;
        z-index: 10;
    }
    
    /* 탭 버튼 스타일 개선 - 더 강력한 선택자 */
    button[data-baseweb="tab"] {
        font-size: 18px !important;
        font-weight: 600 !important;
        padding: 12px 24px !important;
    }
    
    /* 탭 버튼 내부 텍스트 크기 강제 적용 */
    button[data-baseweb="tab"] > div {
        font-size: 18px !important;
        font-weight: 600 !important;
    }
    
    /* 탭 버튼 내부 모든 요소 크기 강제 적용 */
    button[data-baseweb="tab"] * {
        font-size: 18px !important;
        font-weight: 600 !important;
    }
    
    /* 활성 탭 텍스트 색상 */
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #0066cc !important;
        font-weight: 700 !important;
    }
    
    /* 활성 탭 내부 요소도 두껍게 */
    button[data-baseweb="tab"][aria-selected="true"] * {
        font-weight: 700 !important;
    }
    
    /* 활성 탭 밑줄 색상 및 굵기 변경 */
    div[data-baseweb="tab-highlight"] {
        background-color: #0066cc !important;
        height: 4px !important;
    }
    
    /* 로고 위치 조정 */
    .logo-container {
        margin-top: 0px;
        margin-bottom: 0px;
    }
    
    /* 로고 다음 요소와의 간격 제거 */
    .logo-container + div {
        margin-top: -20px;
    }
    
    /* 검색창 스타일 */
    .search-container {
        display: flex;
        align-items: center;
        justify-content: flex-end;
        gap: 10px;
        width: 100%;
        max-width: 260px;
        margin-left: auto;
    }
    
    /* 검색 입력창 - 미니멀 플랫 스타일 */
    .stTextInput > div > div > input {
        border-radius: 8px !important;
        padding: 14px 20px !important;
        border: 2px solid #020202 !important;
        font-size: 14px !important;
        width: 100% !important;
        background-color: #ffffff !important;
        transition: all 0.2s ease !important;
        box-shadow: none !important;
    }

    .stTextInput > div > div > input:hover {
        border-color: #0066cc !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: #0066cc !important;
        box-shadow: 0 0 0 3px rgba(0, 102, 204, 0.1) !important;
    }

    /* 검색 버튼 스타일 - 강력한 선택자 */
    button[kind="secondary"][key="header_search_button"] {
        width: 100% !important;
        height: 44px !important;
        border-radius: 8px !important;
        border: 2px solid #020202 !important;
        background-color: #0A1F32 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.15) !important;
    }

    button[kind="secondary"][key="header_search_button"]:hover {
        background-color: #143d60 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2) !important;
    }

    /* 추가 선택자 */
    div[data-testid="column"]:has(button[key="header_search_button"]) button {
        width: 100% !important;
        height: 44px !important;
        border-radius: 8px !important;
        border: 2px solid #020202 !important;
        background-color: #0A1F32 !important;
        background: #0A1F32 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }

    div[data-testid="column"]:has(button[key="header_search_button"]) button:hover {
        background-color: #143d60 !important;
        background: #143d60 !important;
    }

    mark.search-hit {
        background: #ffe58f !important;
        color: #222 !important;
        padding: 0 2px;
        border-radius: 3px;
    }
    
    .viewerBadge_container__1QSob,
    .viewerBadge_link__1S137 {
        display: none !important;
    }

    .stDeployButton {
        visibility: hidden;
    }

    /* ===== 사이드바 닫기 버튼을 항상 보이게 만들기 ===== */
    
    /* 모든 상태에서 버튼 컨테이너를 보이게 */
    section[data-testid="stSidebar"] > div:first-child > div:first-child {
        opacity: 1 !important;
    }
    
    /* 닫기 버튼 자체를 항상 보이게 */
    section[data-testid="stSidebar"] > div:first-child > div:first-child > button {
        opacity: 1 !important;
        visibility: visible !important;
        display: flex !important;
        background: rgba(255, 255, 255, 0.2) !important;
        border: 2px solid rgba(255, 255, 255, 0.7) !important;
        border-radius: 50% !important;
        width: 42px !important;
        height: 42px !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
    }
    
    /* 호버 효과 */
    section[data-testid="stSidebar"] > div:first-child > div:first-child > button:hover {
        background: rgba(255, 255, 255, 0.35) !important;
        border-color: rgba(255, 255, 255, 0.9) !important;
        transform: scale(1.08) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* 화살표 아이콘 색상 */
    section[data-testid="stSidebar"] > div:first-child > div:first-child > button svg {
        fill: #ffffff !important;
        stroke: #ffffff !important;
        color: #ffffff !important;
        width: 20px !important;
        height: 20px !important;
    }
    </style>
""", unsafe_allow_html=True) 

# -----------------------------
# ✅ 로고 및 검색창 표시
# -----------------------------
LOGO_DIR = Path(__file__).parent / "assets" / "banner_image" / "logo_image"

# logo_image 폴더 안의 이미지 파일 찾기
logo_files = []
if LOGO_DIR.exists():
    logo_files = list(LOGO_DIR.glob("*.png")) + list(LOGO_DIR.glob("*.jpg")) + list(LOGO_DIR.glob("*.jpeg"))

if logo_files:
    st.markdown('<div class="logo-container">', unsafe_allow_html=True)
    
    # 로고(왼쪽)와 검색창(오른쪽) 배치
    col_logo, col_search = st.columns([3, 1])
    
    with col_logo:
        st.image(str(logo_files[0]), width=220)
    
    with col_search:
        st.markdown('<div class="search-container">', unsafe_allow_html=True)
        input_col, button_col = st.columns([0.68, 0.32])

        with input_col:
            user_query = st.text_input(
                "검색",
                value=st.session_state.get("search_query", ""),
                placeholder="키워드를 입력하세요...",
                label_visibility="collapsed",
                key="header_search"
            )

        with button_col:
            trigger_search = st.button("검색", key="header_search_button")

        st.markdown('</div>', unsafe_allow_html=True)

        cleaned_query = (user_query or "").strip()
        if trigger_search:
            st.session_state["search_query"] = cleaned_query
        else:
            st.session_state["search_query"] = cleaned_query
    
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.warning(f"⚠️ 로고 이미지를 찾을 수 없습니다: {LOGO_DIR}")
    st.info("dashboard/assets/banner_image/logo_image/ 폴더에 로고 이미지를 배치해 주세요.")

# -----------------------------
# 데이터 로드
# -----------------------------
train = load_train()
if train.empty:
    st.stop()

# -----------------------------
# ✅ 사이드바 (selectbox 드롭다운)
# -----------------------------
st.sidebar.markdown("<div style='height:60px;'></div>", unsafe_allow_html=True)
st.sidebar.header("실시간 전력 모니터링 제어 시스템")

# 세션 상태 초기화
st.session_state.setdefault("running", False)
st.session_state.setdefault("index", 0)
st.session_state.setdefault("stream_df", train.iloc[0:0].copy())

st.sidebar.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)

# 드롭다운으로 업데이트 간격 선택 (1초 단위)
speed_option = st.sidebar.selectbox(
    "업데이트 간격 설정",
    ["1초", "2초", "3초", "4초", "5초"],
    index=0  # 기본값: 1초
)

# 선택된 값을 숫자로 변환
speed = float(speed_option.replace("초", ""))

# 시작/정지 버튼 (type으로 구분)
if not st.session_state.running:
    if st.sidebar.button("▶ 시작", use_container_width=True, key="start_btn", type="secondary"):
        st.session_state.running = True
        st.rerun()
else:
    if st.sidebar.button("⏸ 정지", use_container_width=True, key="stop_btn", type="primary"):
        st.session_state.running = False
        st.rerun()

# 초기화 버튼 (전체 너비)
if st.sidebar.button("초기화", use_container_width=True, key="reset_btn", type="secondary"):
    st.session_state.index = 0
    st.session_state.stream_df = train.iloc[0:0].copy()
    st.session_state.running = False
    st.rerun()

# 상태 표시
st.sidebar.write("🟢 실행 중" if st.session_state.running else "🔴 정지")

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

search_value = (st.session_state.get("search_query", "") or "").strip()
highlight_script = f"""
<script>
(function() {{
  const query = {json.dumps(search_value)};
  const markClass = "search-hit";
  const doc = (window.parent && window.parent.document) ? window.parent.document : document;
  const root = doc.querySelector('[data-testid="stAppViewContainer"]') || doc.body;
  if (!root) return;

  const clearMarks = () => {{
    root.querySelectorAll('mark.' + markClass).forEach(mark => {{
      const textNode = doc.createTextNode(mark.textContent);
      const parent = mark.parentNode;
      if (!parent) return;
      parent.replaceChild(textNode, mark);
      parent.normalize();
    }});
  }};

  const applyHighlight = () => {{
    if (!query) return;
    const escaped = query.replace(/[.*+?^${{}}()|[\\]\\\\]/g, '\\\\$&');
    if (!escaped) return;
    const walker = doc.createTreeWalker(root, NodeFilter.SHOW_TEXT, null);
    const nodes = [];
    while (walker.nextNode()) {{
      const current = walker.currentNode;
      if (!current || !current.parentElement) continue;
      const parentTag = current.parentElement.tagName;
      if (['SCRIPT', 'STYLE', 'NOSCRIPT', 'MARK'].includes(parentTag)) continue;
      if (current.parentElement.closest('button, select, textarea, input')) continue;
      nodes.push(current);
    }}
    nodes.forEach(node => {{
      const text = node.textContent;
      if (!text) return;
      const testRegex = new RegExp(escaped, 'i');
      if (!testRegex.test(text)) return;
      const replaceRegex = new RegExp(escaped, 'gi');
      const span = doc.createElement('span');
      span.innerHTML = text.replace(replaceRegex, match => `<mark class="${{markClass}}">${{match}}</mark>`);
      node.parentNode.replaceChild(span, node);
    }});
  }};

  window.requestAnimationFrame(() => {{
    clearMarks();
    applyHighlight();
  }});
}})();
</script>
"""
components.html(highlight_script, height=0, scrolling=False)