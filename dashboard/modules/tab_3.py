# modules/tab_4.py
import streamlit as st
import pandas as pd
from shared import * # 컬럼명 상수를 가져오기 위해 import

def show_tab_appendix(train):

    # --- 데이터 컬럼 설명 (토글) ---
    with st.expander("### 데이터 컬럼 설명", expanded=True):
        
        # --- 컬럼 설명 테이블 ---
        desc_data = {
            "컬럼": [
                "측정일시",
                "전력사용량(kWh)",
                "지상무효전력량(kVarh)",
                "진상무효전력량(kVarh)",
                "탄소배출량(tCO2)",
                "지상역률(%)",
                "진상역률(%)",
                "작업유형",
                "전기요금(원)"
            ],
            "설명": [
                "측정이 이루어진 날짜와 시간(15분 간격)",
                "실제 전력 사용량(예측의 기초 입력)",
                "무효 전력량(지상 역률에서 발생)",
                "무효 전력량(진상 역률에서 발생)",
                "전력 사용으로 인한 탄소 배출량",
                "지상 방향의 역률(%)",
                "진상 방향의 역률(%)",
                "해당 시점의 부하 유형(예: Light_Load 등)",
                "예측 대상(전력사용량 × 단가 등으로 계산된 실제 요금)"
            ],
            "전기요금과의 관계(쉬운 설명)": [
                "시간대·요일·계절에 따라 단가(TOU)가 달라져서 같은 사용량이라도 요금이 달라짐.",
                "요금의 핵심 구성: 보통 요금 ≈ 사용량 × 단가 라 사용량이 늘면 요금도 커짐.",
                "무효전력이 많으면 역률이 낮아져 산업용 계약에서 패널티/추가요금이 붙을 수 있음.",
                "마찬가지로 역률 악화 요인이 되어 가산요금이 발생할 수 있음.",
                "보통 직접 청구되진 않지만, 사용량과 같이 움직여 요금이 높을 때 배출도 큰 경향이 있음. (탄소비용 계약 시 반영될 수 있음)",
                "역률이 낮을수록 설비 효율이 떨어져 역률요금/패널티가 붙을 수 있어 요금이 늘음.",
                "목표 범위에서 벗어나면 역시 추가요금이 발생하거나 불리함.",
                "어떤 작업이냐에 따라 사용 패턴과 시간대가 달라져서 피크시간 사용 비중이 커지면 요금이 올라감.",
                "타깃 값. 보통 사용량×단가 + (기본요금 + 피크/역률 패널티 등)으로 결정."
            ]
        }
        
        # train 데이터에 실제 있는 컬럼만 필터링해서 보여주기
        valid_cols = [col for col in desc_data["컬럼"] if col in train.columns]
        valid_indices = [desc_data["컬럼"].index(col) for col in valid_cols]
        
        final_desc = {
            "컬럼": [desc_data["컬럼"][i] for i in valid_indices],
            "설명": [desc_data["설명"][i] for i in valid_indices],
            "전기요금과의 관계(쉬운 설명)": [desc_data["전기요금과의 관계(쉬운 설명)"][i] for i in valid_indices]
        }
        
        st.dataframe(pd.DataFrame(final_desc), use_container_width=True, hide_index=True)

    # --- EDA 과정 (토글) ---
    with st.expander("### EDA 과정", expanded=False):
        
        # 1. 데이터 품질 검증 및 오류 발견 상자
        with st.container(border=True):
            st.markdown("##### 1. 데이터 품질 검증 및 오류 발견")
            st.markdown("**결측값 오류 발견**: id 29855번 행에서 전체 컬럼이 0으로 기록된 오류 발견 및 처리")
            
            # id 29855 행 데이터 생성 (원본 오류 데이터)
            error_row_data = {
                "id": [29855],
                "측정일시": ["2024-11-08 00:00:00"],
                "전력사용량(kWh)": ["0"],
                "지상무효전력량(kVarh)": ["0"],
                "진상무효전력량(kVarh)": ["0"],
                "탄소배출량(tCO2)": ["0"],
                "지상역률(%)": ["0"],
                "진상역률(%)": ["0"],
                "작업유형": ["Light_Load"],
                "전기요금(원)": ["0"]
            }
            error_df = pd.DataFrame(error_row_data)
            st.dataframe(error_df, use_container_width=True, hide_index=True)
            
            st.markdown("**시계열 오류 발견**: train과 test 데이터 간 시간 흐름의 불연속성 확인")
        
        st.markdown("---")
        
        # 2. 휴일 설정 상자
        with st.container(border=True):
            st.markdown("##### 2. 휴가 설정")
            st.markdown("- 그래프를 통해 주말이 일요일과 월요일로 설정되어 있음을 확인")
            st.markdown("- **휴가 데이터 보정**: 실제 공장 가동 데이터와 휴가 매칭 오류 발견")
            st.markdown("  - LS 그룹 휴가 등 추가 반영")
            
            # 요일별 전력 사용 패턴 그래프
            st.markdown("**요일별 평균 전력사용량**")
            
            # train 데이터 전처리
            if train is not None and not train.empty:
                import plotly.graph_objects as go
                
                # 날짜 컬럼 확인 및 전처리
                filtered_df = train.copy()
                
                # COL_TIME 컬럼을 datetime으로 변환
                if COL_TIME in filtered_df.columns:
                    filtered_df[COL_TIME] = pd.to_datetime(filtered_df[COL_TIME], errors="coerce")
                    filtered_df = filtered_df.dropna(subset=[COL_TIME])
                    filtered_df["요일"] = filtered_df[COL_TIME].dt.day_name()
                
                # 필요한 컬럼이 있는지 확인
                if "요일" in filtered_df.columns and COL_USAGE in filtered_df.columns:
                    # 영어 → 한국어 매핑
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
                    
                    # 일요일과 월요일은 #E1746C, 나머지는 #99BDB3
                    colors = ['#E1746C' if day in ['일요일', '월요일'] else '#99BDB3' 
                             for day in weekday_avg["요일(한글)"]]
                    
                    # 요일별 그래프만 생성
                    fig_pattern = go.Figure()
                    
                    fig_pattern.add_trace(
                        go.Bar(
                            x=weekday_avg["요일(한글)"],
                            y=weekday_avg[COL_USAGE],
                            marker_color=colors,
                            name="요일별 평균"
                        )
                    )
                    
                    fig_pattern.update_layout(
                        showlegend=False,
                        hovermode="x unified",
                        template="plotly_white",
                        height=400,
                        xaxis_title="요일",
                        yaxis_title="평균 전력사용량(kWh)"
                    )
                    
                    st.plotly_chart(fig_pattern, use_container_width=True)
                    st.caption("💡 빨간색으로 표시된 일요일과 월요일이 주말로 설정되어 있습니다.")
                else:
                    st.info("⚠️ 그래프를 표시하기 위한 데이터 컬럼이 부족합니다.")
            else:
                st.info("⚠️ 데이터가 없습니다.")
        
        st.markdown("---")
        
        # 3. 작업유형별 상관관계 분석 상자
        with st.container(border=True):
            st.markdown("##### 3. 작업유형별 상관관계 분석")
            st.markdown("- **작업유형별 상관관계 차이 발견**: 작업유형(Light_Load, Medium_Load, Maximum_Load)에 따라 변수 간 상관관계가 상이함을 확인")
            st.markdown("- **분석 결과**: 각 작업유형별로 전력사용량과 전기요금 간의 관계 패턴이 다르게 나타남")
            st.markdown("- **분리 학습**: 작업유형별로 모델을 분리하여 학습하는 것이 효과적일 것으로 판단")
            
            # 작업유형별 피어슨 상관관계 히트맵
            st.markdown("**작업유형별 피어슨 상관관계 히트맵**")
            
            if train is not None and not train.empty and "작업유형" in train.columns:
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                
                # 상관관계 분석에 사용할 수치형 컬럼 선택
                numeric_cols = [
                    COL_USAGE,  # "전력사용량(kWh)"
                    "지상무효전력량(kVarh)", 
                    "진상무효전력량(kVarh)",
                    "탄소배출량(tCO2)",
                    COL_LAG_PF,  # "지상역률(%)"
                    COL_LEAD_PF,  # "진상역률(%)"
                    COL_COST  # "전기요금(원)"
                ]
                
                # 실제 존재하는 컬럼만 필터링
                available_cols = [col for col in numeric_cols if col in train.columns]
                
                if len(available_cols) >= 3:
                    # 작업유형 목록
                    work_types = sorted(train["작업유형"].unique())
                    
                    # 서브플롯 생성 (1행 3열)
                    fig = make_subplots(
                        rows=1, cols=len(work_types),
                        subplot_titles=work_types,
                        horizontal_spacing=0.08
                    )
                    
                    # 각 작업유형별로 상관관계 계산 및 히트맵 생성
                    for idx, work_type in enumerate(work_types, 1):
                        # 해당 작업유형 데이터 필터링
                        work_data = train[train["작업유형"] == work_type][available_cols]
                        
                        # 상관관계 행렬 계산
                        corr_matrix = work_data.corr()
                        
                        # 히트맵 추가
                        fig.add_trace(
                            go.Heatmap(
                                z=corr_matrix.values,
                                x=corr_matrix.columns,
                                y=corr_matrix.columns,
                                colorscale=[
                                    [0.0, '#08519c'],    # 진한 파랑 (강한 음의 상관)
                                    [0.25, '#6baed6'],   # 밝은 파랑
                                    [0.5, '#ffffff'],    # 흰색 (상관없음)
                                    [0.75, '#fc8d59'],   # 밝은 빨강
                                    [1.0, '#d7301f']     # 진한 빨강 (강한 양의 상관)
                                ],
                                zmid=0,
                                zmin=-1,
                                zmax=1,
                                text=corr_matrix.values.round(2),
                                texttemplate='%{text}',
                                textfont={"size": 8},
                                colorbar=dict(
                                    title="상관계수",
                                    len=0.7,
                                    x=1.02 if idx == len(work_types) else None,
                                    xanchor='left' if idx == len(work_types) else None
                                ) if idx == len(work_types) else None,
                                showscale=(idx == len(work_types)),
                                hovertemplate='%{y} vs %{x}<br>상관계수: %{z:.3f}<extra></extra>'
                            ),
                            row=1, col=idx
                        )
                        
                        # x축, y축 레이블 설정
                        fig.update_xaxes(
                            tickangle=-45,
                            side='bottom',
                            row=1, col=idx
                        )
                        fig.update_yaxes(
                            row=1, col=idx
                        )
                    
                    # 전체 레이아웃 설정
                    fig.update_layout(
                        height=500,
                        showlegend=False,
                        template="plotly_white",
                        title_text="작업유형별 피어슨 상관관계 분석",
                        title_x=0.5
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("💡 작업유형에 따라 변수 간 상관관계가 다르게 나타남을 확인할 수 있습니다.")
                else:
                    st.info("⚠️ 상관관계 분석을 위한 충분한 수치형 데이터가 없습니다.")
            else:
                st.info("⚠️ 데이터가 없거나 작업유형 컬럼이 없습니다.")
        
        st.markdown("---")
        
        # 4. 요일별 전력 사용 패턴 분석 상자
        with st.container(border=True):
            st.markdown("##### 4. 요일별 전력 사용 패턴 분석")
            st.markdown("- **월별 요일 패턴 비교**: 1월부터 11월까지 각 월별로 요일에 따른 전력 사용량 패턴 분석")
            st.markdown("- **주말 효과 확인**: 일요일과 월요일(주말)에 전력 사용량이 급격히 감소하는 패턴 발견")
            st.markdown("- **월별 차이 발견**: 계절과 월에 따라 요일별 전력 사용 패턴이 다르게 나타남")
            st.markdown("- **1월 특이 패턴**: 1월의 경우 다른 월들에 비해 전반적으로 높은 전력 사용량을 보임")
            
            # 요일별 전력 사용 패턴 그래프
            st.markdown("**2024년 요일별 전력사용량 평균 - 월별 오버레이 (1-11월)**")
            
            if train is not None and not train.empty and COL_TIME in train.columns and COL_USAGE in train.columns:
                import plotly.graph_objects as go
                
                # 데이터 전처리
                df_plot = train.copy()
                df_plot[COL_TIME] = pd.to_datetime(df_plot[COL_TIME], errors="coerce")
                df_plot = df_plot.dropna(subset=[COL_TIME])
                
                # 월과 요일 추출
                df_plot["month"] = df_plot[COL_TIME].dt.month
                df_plot["weekday"] = df_plot[COL_TIME].dt.day_name()
                
                # 요일 순서 정의 (월-일)
                weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                weekday_map_kr = {
                    "Monday": "월",
                    "Tuesday": "화",
                    "Wednesday": "수",
                    "Thursday": "목",
                    "Friday": "금",
                    "Saturday": "토",
                    "Sunday": "일"
                }
                
                # 월별 색상 정의 (이미지와 유사하게)
                month_colors = {
                    1: "#17BECF",   # 청록색 (1월)
                    2: "#FFFF00",   # 노란색 (2월)
                    3: "#9467BD",   # 보라색 (3월)
                    4: "#FF6B6B",   # 빨간색 (4월)
                    5: "#4A90E2",   # 파란색 (5월)
                    6: "#FFA500",   # 주황색 (6월)
                    7: "#90EE90",   # 연두색 (7월)
                    8: "#FFD700",   # 금색 (8월)
                    9: "#D3D3D3",   # 회색 (9월)
                    10: "#DDA0DD",  # 자주색 (10월)
                    11: "#98FB98"   # 연한 초록 (11월)
                }
                
                # 그래프 생성
                fig = go.Figure()
                
                # 각 월별로 요일 평균 계산 및 라인 추가
                for month in range(1, 12):  # 1월부터 11월까지
                    month_data = df_plot[df_plot["month"] == month]
                    
                    if not month_data.empty:
                        # 요일별 평균 계산
                        weekday_avg = (
                            month_data.groupby("weekday")[COL_USAGE]
                            .mean()
                            .reindex(weekday_order)
                            .reset_index()
                        )
                        
                        # 한글 요일로 변환
                        weekday_avg["weekday_kr"] = weekday_avg["weekday"].map(weekday_map_kr)
                        
                        # 라인 추가
                        fig.add_trace(
                            go.Scatter(
                                x=weekday_avg["weekday_kr"],
                                y=weekday_avg[COL_USAGE],
                                mode='lines+markers',
                                name=f"{month}월",
                                line=dict(color=month_colors.get(month, "#808080"), width=2),
                                marker=dict(size=8)
                            )
                        )
                
                # 레이아웃 설정
                fig.update_layout(
                    title="2024년 요일별 전력사용량 평균 - 월별 오버레이 (1-11월)",
                    xaxis_title="요일",
                    yaxis_title="전력사용량 (kWh)",
                    template="plotly_white",
                    height=500,
                    hovermode="x unified",
                    legend=dict(
                        title="월",
                        orientation="v",
                        yanchor="top",
                        y=1,
                        xanchor="left",
                        x=1.02
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                st.caption("💡 일요일과 월요일(주말)에 전력 사용량이 급격히 감소하며, 1월은 전반적으로 높은 사용량을 보입니다.")
            else:
                st.info("⚠️ 그래프를 표시하기 위한 데이터가 부족합니다.")

        st.markdown("---")
        
        # 5. 역률 변수 특성 분석 상자
        with st.container(border=True):
            st.markdown("##### 5. 역률 변수 특성 분석")
            st.markdown("- **역률-전기요금 관계 분석**: 이론적으로 역률이 전기요금과 밀접한 관련이 있으나, 전체 데이터에서는 상관성이 다소 낮게 나타남")
            st.markdown("- **그래프 분석**: 역률과 전기요금 간의 산점도를 통해 비선형적 관계 확인")
            st.markdown("- **임계값 기반 구간 분리**: 특정 임계값을 기준으로 데이터를 구간별로 나누면 더 정확한 예측이 가능할 것으로 판단")
            st.markdown("- **구간별 모델링**: 역률 수준에 따라 다른 예측 모델을 적용하는 것이 효과적")
            
            # 역률 vs 전기요금 산점도
            st.markdown("**역률과 전기요금의 관계**")
            
            if train is not None and not train.empty:
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                
                # 지상역률과 진상역률이 있는지 확인
                if COL_LAG_PF in train.columns and COL_LEAD_PF in train.columns and COL_COST in train.columns:
                    
                    # 서브플롯 생성
                    fig_pf = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=("지상역률(%) vs 전기요금", "진상역률(%) vs 전기요금"),
                        horizontal_spacing=0.12
                    )
                    
                    # 지상역률 산점도
                    fig_pf.add_trace(
                        go.Scatter(
                            x=train[COL_LAG_PF],
                            y=train[COL_COST],
                            mode='markers',
                            marker=dict(size=4, opacity=0.5, color='#E76F51'),
                            name="지상역률"
                        ),
                        row=1, col=1
                    )
                    
                    # 진상역률 산점도
                    fig_pf.add_trace(
                        go.Scatter(
                            x=train[COL_LEAD_PF],
                            y=train[COL_COST],
                            mode='markers',
                            marker=dict(size=4, opacity=0.5, color='#2A9D8F'),
                            name="진상역률"
                        ),
                        row=1, col=2
                    )
                    
                    # 레이아웃 설정
                    fig_pf.update_xaxes(title_text="지상역률(%)", row=1, col=1)
                    fig_pf.update_xaxes(title_text="진상역률(%)", row=1, col=2)
                    fig_pf.update_yaxes(title_text="전기요금(원)", row=1, col=1)
                    fig_pf.update_yaxes(title_text="전기요금(원)", row=1, col=2)
                    
                    fig_pf.update_layout(
                        showlegend=False,
                        template="plotly_white",
                        height=450
                    )
                    
                    st.plotly_chart(fig_pf, use_container_width=True)
                    st.caption("💡 역률과 전기요금 간의 관계가 명확한 선형관계를 보이지 않아, 임계값 기반 구간 분리가 필요합니다.")
                else:
                    st.info("⚠️ 역률 데이터가 없습니다.")
            else:
                st.info("⚠️ 데이터가 없습니다.")
        
        st.markdown("---")

    # --- 모델링 학습 과정 (토글) --- 여기부터 들여쓰기가 EDA 밖으로 나옴
    with st.expander("### 모델링 학습 과정", expanded=False):
        
        # 1. 구현된 베이스라인 모델 상자
        with st.container(border=True):
            st.markdown("##### 1.베이스라인 모델")
            st.markdown("- **모델**: HistGradientBoostingRegressor (L1 Loss)")
            st.markdown("- **손실 함수**: Absolute Error (MAE 최적화)")
            st.markdown("- **주요 하이퍼파라미터**:")
            
            # 하이퍼파라미터 테이블
            params_data = {
                "파라미터": ["loss", "max_depth", "learning_rate", "max_iter", "random_state"],
                "값": ["absolute_error", "8", "0.05", "300", "42"],
                "설명": [
                    "MAE를 직접 최적화하는 손실 함수",
                    "트리의 최대 깊이 제한",
                    "학습률 (보수적 설정으로 과적합 방지)",
                    "부스팅 반복 횟수",
                    "재현 가능성을 위한 시드값"
                ]
            }
            
            st.dataframe(pd.DataFrame(params_data), use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # 2. 시계열 피처 엔지니어링 상자
        with st.container(border=True):
            st.markdown("##### 2. 시계열 피처 엔지니어링")
            st.markdown("- **다양한 시간 관련 피처 생성**: 시간대, 요일, 월, 분기 등 기본 시간 정보 추출")
            st.markdown("- **Fourier Transform 적용**: 주기적인 패턴을 sin/cos 함수로 부드럽게 표현")
            st.markdown("- **작업유형 인코딩**: 범주형 변수를 숫자로 변환하여 모델 학습에 활용")
            
            # 피처 그룹별 정리
            feature_groups = {
                "기본 시간 피처": [
                    "slot_15m: 15분 단위 시간 슬롯 (0-95)",
                    "hour: 시간대 (0-23)",
                    "weekday: 요일 (0=월요일, 6=일요일)",
                    "is_weekend: 주말 여부",
                    "weekofmonth: 월 중 주차",
                    "month: 월 (1-12)",
                    "quarter: 분기 (1-4)",
                    "hour_of_week: 주 단위 시간 인덱스"
                ],
                "Daily Fourier 피처 (주기=96)": [
                    "sin_day_1, cos_day_1: 하루 주기 (기본파)",
                    "sin_day_2, cos_day_2: 하루 주기 (2차 고조파)",
                    "sin_day_3, cos_day_3: 하루 주기 (3차 고조파)",
                    "→ 하루 내 전력 사용 패턴의 부드러운 변화 포착"
                ],
                "Weekly Fourier 피처 (주기=672)": [
                    "sin_week_1, cos_week_1: 주간 주기 (기본파)",
                    "sin_week_2, cos_week_2: 주간 주기 (2차 고조파)",
                    "→ 주중/주말 패턴 변화 포착"
                ],
                "작업유형 인코딩": [
                    "작업유형_enc: Light/Medium/Maximum Load를 숫자로 인코딩"
                ]
            }
            
            for group_name, features in feature_groups.items():
                st.markdown(f"**{group_name}**")
                for feature in features:
                    st.markdown(f"  - {feature}")
                st.markdown("")
        
        st.markdown("---")
        
        # 3. 휴일 피처 (2018년→2024년 매핑 전략) 상자
        with st.container(border=True):
            st.markdown("##### 3. 휴가 피처")
            st.markdown("- **핵심 아이디어**: 2018년 공장 가동 데이터의 휴가 패턴을 2024년에 적용")
            st.markdown("- **2018년 휴가 리스트 기반**: 실제 LS 그룹의 휴가를 반영한 24개 휴가")
            st.markdown("- **다양한 휴가 관련 피처 생성**: 휴가 전후 효과, 연휴 길이, 특수 기간 등을 반영")
            
            # 휴일 피처 상세 설명
            holiday_features = {
                "피처명": [
                    "is_weekend_or_holiday",
                    "holiday_block_len",
                    "holiday_block_len_log1p",
                    "pre_holiday_d1/d2/d3",
                    "post_holiday_d1/d2/d3",
                    "is_friday_before_holiday",
                    "is_monday_after_holiday",
                    "is_year_end",
                    "is_year_start"
                ],
                "설명": [
                    "주말 또는 휴가 여부 (복합 지표)",
                    "연속된 휴가 블록의 길이 (일 단위)",
                    "휴가 블록 길이의 log1p 변환",
                    "휴가 1/2/3일 전",
                    "휴가 1/2/3일 후",
                    "휴가 전 금요일 (브리지 연휴 효과)",
                    "휴가 다음 월요일 (복귀 효과)",
                    "연말 (12/24-12/31)",
                    "연초 (1/1-1/3)"
                ],
                "목적": [
                    "비가동일 통합 지표",
                    "연휴 길이에 따른 패턴 차이",
                    "긴 연휴의 비선형 효과 포착",
                    "휴가 전 수요 변화 패턴",
                    "휴가 후 복귀 패턴",
                    "장기 연휴 시작 효과",
                    "장기 연휴 종료 효과",
                    "특수 기간 (연말정산 등)",
                    "특수 기간 (신년 가동)"
                ]
            }
            
            st.dataframe(pd.DataFrame(holiday_features), use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # 4. 데이터 분할 전략 상자
        with st.container(border=True):
            st.markdown("##### 4. 데이터 분할")
            st.markdown("- **시계열 특성 고려**: 시간 순서를 유지하는 Hold-out 검증 방식 채택")
            st.markdown("- **학습/검증 분할**: 1-9월 데이터로 학습, 10-11월 데이터로 검증")
            st.markdown("- **최종 학습**: 검증 성능 확인 후 1-11월 전체 데이터로 재학습")
            st.markdown("- **결측치 처리**: 타깃 변수가 결측인 행은 학습에서 제외")
            
            split_strategy = {
                "구분": ["학습 데이터", "검증 데이터", "최종 학습"],
                "기간": ["1월 - 9월", "10월 - 11월", "1월 - 11월 전체"],
                "목적": [
                    "모델 학습 (9개월 데이터)",
                    "성능 검증 및 조기 중단",
                    "최종 제출용 모델 재학습"
                ],
                "특징": [
                    "타깃 결측치 제거 후 학습",
                    "시계열 순서 유지한 Hold-out",
                    "검증 성능 확인 후 전체 데이터 활용"
                ]
            }
            
            st.dataframe(pd.DataFrame(split_strategy), use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # 5. 모델의 강점 상자
        with st.container(border=True):
            st.markdown("##### 5. 모델 강점")
            st.markdown("**이 베이스라인 모델이 효과적인 이유**")
            
            strengths = [
                "**Gradient Boosting 기반**: 비선형 패턴과 변수 간 상호작용을 자동으로 학습",
                "**L1 Loss (MAE 최적화)**: 대회 평가 지표(MAE)와 직접 일치하는 손실 함수 사용",
                "**Fourier Transform**: 주기적 패턴(일간/주간)을 연속적으로 부드럽게 표현",
                "**도메인 특화 휴가 피처**: 실제 LS 그룹 공장 가동 캘린더를 반영한 정확한 휴가 정보",
                "**Robust한 전처리**: 결측치 안전 처리 및 데이터 타입 자동 변환",
                "**시계열 고려 분할**: 미래 데이터 누수 방지를 위한 시간 순서 기반 검증"
            ]
            
            for strength in strengths:
                st.markdown(f"- {strength}")
            
            st.markdown("")