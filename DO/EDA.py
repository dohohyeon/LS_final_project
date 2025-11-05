import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import platform
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import pmdarima as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

train_df = pd.read_csv("data/raw/train.csv")
train_df.columns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# train_df가 이미 로드되어 있다고 가정합니다.
# 예: train_df = pd.read_csv('your_file_path.csv')

## ----------------------------------------------------
## グラフの韓国語フォント設定 (環境に合わせて修正)
## ----------------------------------------------------
# Colab/Linux環境の場合
# !sudo apt-get install -y fonts-nanum
# !sudo fc-cache -fv
# !rm ~/.cache/matplotlib -rf
# plt.rc('font', family='NanumBarunGothic')

# Windows環境の場合
# font_path = 'c:/Windows/Fonts/malgun.ttf'
# font_name = fm.FontProperties(fname=font_path).get_name()
# plt.rc('font', family=font_name)

# Mac環境の場合
# plt.rc('font', family='AppleGothic')

# マイナス記号が壊れるのを防ぐ
plt.rcParams['axes.unicode_minus'] = False


## ----------------------------------------------------
## 1. データの前処理
## ----------------------------------------------------

# '측정일시' 컬럼을 datetime 형식으로 변환 (오류 방지)
train_df['측정일시'] = pd.to_datetime(train_df['측정일시'])

# 시계열 분석을 위해 '측정일시'를 인덱스로 설정
train_df.set_index('측정일시', inplace=True)

# 월(month)과 일(day) 컬럼 추가
train_df['month'] = train_df.index.month
train_df['day'] = train_df.index.day

## ----------------------------------------------------
## 2. グラフ作成用のデータ構造に変更 (ピボット)
## ----------------------------------------------------

# 일별 평균 전력사용량을 계산하여 피벗 테이블 생성
# 각 월이 컬럼이 되고, 각 일이 인덱스가 됩니다.
# 시간 단위 데이터일 경우, 일별 평균으로 집계됩니다.
monthly_pivot = train_df.pivot_table(
    values='전력사용량(kWh)',
    index='day',
    columns='month',
    aggfunc='mean' # 일별 평균값 사용
)

## ----------------------------------------------------
## 3. 시계열 그래프 그리기
## ----------------------------------------------------

# 그래프 크기 설정
plt.figure(figsize=(18, 8))

# 12개월치 데이터를 반복하여 하나의 그래프에 겹쳐서 그리기
for month in range(1, 13):
    # 해당 월의 데이터가 있는 경우에만 그립니다.
    if month in monthly_pivot.columns:
        plt.plot(monthly_pivot.index, monthly_pivot[month], label=f'{month}월')

# 그래프 제목 및 라벨 설정
plt.title('월별 전력사용량(kWh) 추세 비교', fontsize=20)
plt.xlabel('일(Day)', fontsize=12)
plt.ylabel('평균 전력사용량(kWh)', fontsize=12)
plt.xticks(range(1, 32)) # X축 눈금을 1일부터 31일까지 표시
plt.grid(True) # 그리드 표시
plt.legend(title='월(Month)') # 범례 표시

# 그래프를 화면에 보여주기
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 이미 train_df가 있다면 다음 2줄은 생략하세요
train_df = pd.read_csv("data/raw/train.csv")

# '24:00' 형태 안전 파싱
def _fix_24h(s):
    s = str(s)
    if " 24:" in s:
        parts = s.split()
        if len(parts) >= 2:
            d, t = parts[0], parts[1]
            try:
                d0 = pd.to_datetime(d)
                return f"{(d0 + pd.Timedelta(days=1)).strftime('%Y-%m-%d')} 00:{t.split(':',1)[1]}"
            except Exception:
                return s
    return s

# 날짜/숫자형 전처리
train_df = train_df.copy()
train_df["측정일시"] = pd.to_datetime(train_df["측정일시"].astype(str).map(_fix_24h), errors="coerce")
train_df = train_df.dropna(subset=["측정일시"])
train_df["전력사용량(kWh)"] = pd.to_numeric(train_df["전력사용량(kWh)"], errors="coerce")

# 요일 컬럼(0=월 ~ 6=일)
train_df["dow"] = train_df["측정일시"].dt.dayofweek
weekday_labels = {0:"월", 1:"화", 2:"수", 3:"목", 4:"금", 5:"토", 6:"일"}

# 요일별 전력사용량 합계 (평균으로 보고 싶으면 .mean()으로 바꾸세요)
dow_sum = (train_df
           .groupby("dow", dropna=True)["전력사용량(kWh)"]
           .sum()
           .reindex(range(7))  # 월~일 순서 보장
           .rename(index=weekday_labels))

# 막대그래프
plt.figure(figsize=(8,4))
plt.bar(dow_sum.index, dow_sum.values)
plt.title("요일별 전력사용량 합계")
plt.xlabel("요일")
plt.ylabel("전력사용량 (kWh)")
plt.tight_layout()
plt.show()



import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- (안전) Datetime 파서 ---
def _fix_24h_token(s: str) -> str:
    s = str(s).strip().replace("\u3000", " ").replace("\t", " ")
    if " 24:" in s:
        d, t = s.split()
        try:
            d0 = pd.to_datetime(d)
            return f"{(d0 + pd.Timedelta(days=1)).date()} 00:{t.split(':',1)[1]}"
        except Exception:
            return s
    return s

def safe_parse_datetime(series: pd.Series) -> pd.DatetimeIndex:
    s = series.astype(str).map(_fix_24h_token)
    dt = pd.to_datetime(s, errors="coerce")
    if dt.isna().all():
        return pd.date_range("2024-01-01", periods=len(dt), freq="15min")
    return dt.ffill().bfill()

# --- 월별 오버레이 함수 (x축: 월 내 경과일) ---
def plot_monthly_overlay(trian_df: pd.DataFrame,
                         time_col: str = "측정일시",
                         y_col: str = "전기요금(원)",
                         year: int = None,      # 특정 연도만 보고 싶으면 예: 2024
                         freq: str = "H",       # 리샘플 간격: None / "30min" / "H" / "3H" / "D" ...
                         agg: str = "sum",      # 리샘플 집계: "sum" 또는 "mean"
                         normalize: str = None, # None / "minmax" / "zscore"
                         title: str = "월별 오버레이(전기요금)"):

    g = trian_df[[time_col, y_col]].copy()
    g[y_col] = pd.to_numeric(g[y_col].astype(str).str.replace(",", ""), errors="coerce")
    g[time_col] = safe_parse_datetime(g[time_col])
    g = g.dropna(subset=[time_col, y_col]).sort_values(time_col)

    # 연도 필터(선택)
    if year is not None:
        g = g[g[time_col].dt.year == year]

    # 리샘플(선택)
    if freq:
        g = (g.set_index(time_col)[y_col]
               .resample(freq)
               .agg(agg)
               .reset_index())

    # 월 키, 월 내 경과일(x축)
    g["month_key"] = g[time_col].dt.to_period("M").astype(str)  # 예: "2024-01"
    g["_x_in_month"] = (g[time_col].dt.day
                        + g[time_col].dt.hour/24
                        + g[time_col].dt.minute/1440
                        + g[time_col].dt.second/86400)

    # (선택) 월별 정규화
    def _normalize(s: pd.Series, how: str):
        if how is None:
            return s
        if how == "minmax":
            mn, mx = s.min(), s.max()
            return (s - mn) / (mx - mn) if mx > mn else s*0
        if how == "zscore":
            mu, sd = s.mean(), s.std(ddof=1)
            return (s - mu) / sd if sd > 0 else s*0
        return s

    fig = go.Figure()
    for mk, sub in g.groupby("month_key", sort=True):
        yy = _normalize(sub[y_col], normalize)
        # f-string 쓰지 말고, hovertemplate는 평범한 문자열!
        hover_t = ("day: %{x:.2f}<br>" + (f"{y_col}: %{y:,.0f}" if normalize is None else "norm: %{y:.3f}") + "<extra></extra>")
        fig.add_trace(go.Scatter(
            x=sub["_x_in_month"], y=yy,
            mode="lines",
            name=mk,
            hovertemplate=hover_t
        ))

    fig.update_layout(
        title=title + ("" if normalize is None else f" (norm={normalize})"),
        xaxis_title="Day within month (d + h/24)",
        yaxis_title=(y_col if normalize is None else f"{y_col} (normalized)"),
        hovermode="x unified",
        xaxis=dict(range=[1, 31], tick0=1, dtick=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0)
    )
    fig.show()

# ===================== 사용 예시 =====================
# 1) 표본 많으면 시간별로 가볍게(합계) 비교
plot_monthly_overlay(train_df, year=2024, freq="H", agg="sum",
                     title="월별 오버레이 (시간합계)")

# 2) 형태만 비교하고 싶으면 정규화 켜기
plot_monthly_overlay(train_df, year=2024, freq="H", agg="sum", normalize="minmax",
                     title="월별 오버레이 (시간합계, min-max 정규화)")

# ====== (옵션) 안전한 Datetime 파서 ======
def _fix_24h_token(s: str) -> str:
    s = str(s).strip().replace("\u3000", " ").replace("\t", " ")
    if " 24:" in s:
        d, t = s.split()
        try:
            d0 = pd.to_datetime(d)
            return f"{(d0 + pd.Timedelta(days=1)).date()} 00:{t.split(':',1)[1]}"
        except Exception:
            return s
    return s

def safe_parse_datetime(series: pd.Series) -> pd.DatetimeIndex:
    s = series.astype(str).map(_fix_24h_token)
    dt = pd.to_datetime(s, errors="coerce")
    return dt.ffill().bfill() if not dt.isna().all() else pd.date_range("2024-01-01", periods=len(dt), freq="15min")

# ====== 시계열 플롯 함수 ======
def plot_cost_timeseries(df: pd.DataFrame,
                         time_col="측정일시",
                         cost_col="전기요금(원)",
                         freq: str = None,    # 예: None(원자료), "30min","H","3H","D"
                         agg: str = "sum",    # 리샘플 집계: "sum" 또는 "mean"
                         title: str = "전기요금 시계열"):
    g = df[[time_col, cost_col]].copy()
    # 숫자/시간 변환
    g[cost_col] = pd.to_numeric(g[cost_col].astype(str).str.replace(",", ""), errors="coerce")
    g[time_col] = safe_parse_datetime(g[time_col])
    g = g.dropna(subset=[time_col, cost_col]).sort_values(time_col)

    # 선택적으로 리샘플링(표본 많을 때)
    if freq:
        ts = (g.set_index(time_col)[cost_col]
              .resample(freq).agg(agg).reset_index())
    else:
        ts = g.rename(columns={time_col:"dt", cost_col:"cost"})
        ts = ts[["dt","cost"]]

    if freq:
        ts = ts.rename(columns={time_col:"dt", cost_col:"cost"})

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts["dt"], y=ts["cost"], mode="lines",
                             name=f"{cost_col} ({'raw' if not freq else f'resample={freq}/{agg}'})"))

    fig.update_layout(
        title=title,
        xaxis_title="측정일시",
        yaxis_title="전기요금(원)",
        hovermode="x unified",
        xaxis=dict(
            rangeslider=dict(visible=True),
            rangeselector=dict(buttons=[
                dict(count=1, step="day",   stepmode="backward", label="1d"),
                dict(count=3, step="day",   stepmode="backward", label="3d"),
                dict(count=7, step="day",   stepmode="backward", label="1w"),
                dict(count=1, step="month", stepmode="backward", label="1m"),
                dict(step="all", label="All")
            ])
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0)
    )
    fig.show()

# ====== 사용 예시 ======
# 1) 원자료 그대로(슬라이더로 구간 확대/축소)
plot_cost_timeseries(train_df, freq=None, title="전기요금 시계열 (원자료)")

# 2) 1시간 단위 합계로 가볍게 보기
plot_cost_timeseries(train_df, freq="H", agg="sum", title="전기요금 시계열 (1시간 합계)")

# 3) 1일 합계로 장기 추세만 보기
plot_cost_timeseries(train_df, freq="D", agg="sum", title="전기요금 시계열 (일 합계)")





import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = "browser"   # VS Code면 "vscode"도 OK

# ====== (옵션) 안전한 Datetime 파서 ======
def _fix_24h_token(s: str) -> str:
    s = str(s).strip().replace("\u3000", " ").replace("\t", " ")
    if " 24:" in s:
        d, t = s.split()
        try:
            d0 = pd.to_datetime(d)
            return f"{(d0 + pd.Timedelta(days=1)).date()} 00:{t.split(':',1)[1]}"
        except Exception:
            return s
    return s

def safe_parse_datetime(series: pd.Series) -> pd.DatetimeIndex:
    s = series.astype(str).map(_fix_24h_token)
    dt = pd.to_datetime(s, errors="coerce")
    return dt.ffill().bfill() if not dt.isna().all() else pd.date_range("2024-01-01", periods=len(dt), freq="15min")

# ====== 시계열 플롯 함수 ======
def plot_cost_timeseries(df: pd.DataFrame,
                         time_col="측정일시",
                         cost_col="전기요금(원)",
                         freq: str = None,    # 예: None(원자료), "30min","H","3H","D"
                         agg: str = "sum",    # 리샘플 집계: "sum" 또는 "mean"
                         title: str = "전기요금 시계열"):
    g = df[[time_col, cost_col]].copy()
    # 숫자/시간 변환
    g[cost_col] = pd.to_numeric(g[cost_col].astype(str).str.replace(",", ""), errors="coerce")
    g[time_col] = safe_parse_datetime(g[time_col])
    g = g.dropna(subset=[time_col, cost_col]).sort_values(time_col)

    # 선택적으로 리샘플링(표본 많을 때)
    if freq:
        ts = (g.set_index(time_col)[cost_col]
              .resample(freq).agg(agg).reset_index())
    else:
        ts = g.rename(columns={time_col:"dt", cost_col:"cost"})
        ts = ts[["dt","cost"]]

    if freq:
        ts = ts.rename(columns={time_col:"dt", cost_col:"cost"})

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts["dt"], y=ts["cost"], mode="lines",
                             name=f"{cost_col} ({'raw' if not freq else f'resample={freq}/{agg}'})"))

    fig.update_layout(
        title=title,
        xaxis_title="측정일시",
        yaxis_title="전기요금(원)",
        hovermode="x unified",
        xaxis=dict(
            rangeslider=dict(visible=True),
            rangeselector=dict(buttons=[
                dict(count=1, step="day",   stepmode="backward", label="1d"),
                dict(count=3, step="day",   stepmode="backward", label="3d"),
                dict(count=7, step="day",   stepmode="backward", label="1w"),
                dict(count=1, step="month", stepmode="backward", label="1m"),
                dict(step="all", label="All")
            ])
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0)
    )
    fig.show()

# ====== 사용 예시 ======
# 1) 원자료 그대로(슬라이더로 구간 확대/축소)
plot_cost_timeseries(train_df, freq=None, title="전기요금 시계열 (원자료)")

# 2) 1시간 단위 합계로 가볍게 보기
plot_cost_timeseries(train_df, freq="H", agg="sum", title="전기요금 시계열 (1시간 합계)")

# 3) 1일 합계로 장기 추세만 보기
plot_cost_timeseries(train_df, freq="D", agg="sum", title="전기요금 시계열 (일 합계)")






















test_df = pd.read_csv("data/raw/test.csv")
train_df = pd.read_csv("data/raw/train.csv")
sub_df = pd.read_csv("data/raw/sample_submission.csv")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===== 설정 =====
DT_COL   = "측정일시"
KWH_COL  = "전력사용량(kWh)"

# ===== 1) 시간대(0~23시) 기준 월별 오버레이 =====
df = train_df.copy()
df[DT_COL] = pd.to_datetime(df[DT_COL], errors="coerce")
df = df.dropna(subset=[DT_COL]).copy()
df[KWH_COL] = pd.to_numeric(df[KWH_COL], errors="coerce")

df["month"] = df[DT_COL].dt.month
df["hour"]  = df[DT_COL].dt.hour

# 필요 시: 1~11월만 보려면 주석 해제
# df = df[df["month"].between(1, 11)]

# 시간대×월 평균
g = (df.groupby(["hour","month"], as_index=False)[KWH_COL]
       .mean()
       .sort_values(["hour","month"]))

pivot = g.pivot(index="hour", columns="month", values=KWH_COL).sort_index()

plt.figure(figsize=(14,5))
for m in pivot.columns:
    plt.plot(pivot.index, pivot[m], marker="o", linewidth=2, label=f"{m}월")

plt.xticks(range(0,24), [f"{h}시" for h in range(24)])
plt.xlabel("시간대")
plt.ylabel("전력사용량 (kWh)")
plt.title("시간대별 전력사용량 평균 — 월별 오버레이")
plt.grid(True, axis="y", linestyle="--", alpha=0.4)
plt.legend(title="월", ncol=6, bbox_to_anchor=(1.02,1.05), loc="upper left", frameon=True)
plt.tight_layout()
plt.show()



# 1) datetime 보장
train_df['측정일시'] = pd.to_datetime(train_df['측정일시'])

# 2) 자정(00:00:00) 마스크
mid_mask = (
    (train_df['측정일시'].dt.hour == 0) &
    (train_df['측정일시'].dt.minute == 0) &
    (train_df['측정일시'].dt.second == 0)
)

# 3) 원본 컬럼을 직접 갱신(+1일)
train_df.loc[mid_mask, '측정일시'] = train_df.loc[mid_mask, '측정일시'] + pd.Timedelta(days=1)





# 1. 수치형 데이터만 선택
numerical_df = train_df.select_dtypes(include=np.number)

# 2. 상관계수 행렬 계산
correlation_matrix = numerical_df.corr()
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')

# 3. 히트맵 생성
plt.figure(figsize=(10, 8))
plt.rcParams['axes.unicode_minus'] = False
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Correlation Heatmap of Numerical Columns')
plt.show()

train_df.info()
train_df['작업유형'].value_counts()
train_df.isna().sum()
train_df.describe()

# 1) 시간 파생
train_df["ts"] = pd.to_datetime(train_df["측정일시"], errors="coerce")
train_df["hour"] = train_df["ts"].dt.hour
train_df["dow"] = train_df["ts"].dt.dayofweek      # 0=월 … 6=일
train_df["month"] = train_df["ts"].dt.month
train_df["is_weekend"] = (train_df["dow"] >= 5).astype(int)

# 2) 단가 계산 (그냥 바로 연산)
if "id" in train_df.columns:
    train_df.sort_values("id", inplace=True)  # id가 시간순 인덱스라면 정렬
train_df["단가"] = train_df["전기요금(원)"] / train_df["전력사용량(kWh)"]

# 3) 전력사용량이 0인 행의 단가를 앞/뒤 이웃 평균으로 채우기
mask_zero = train_df["전력사용량(kWh)"].eq(0)
ff = train_df["단가"].ffill()   # 바로 위(앞) 값
bf = train_df["단가"].bfill()   # 바로 아래(뒤) 값

train_df.loc[mask_zero, "단가"] = np.where(
    ff[mask_zero].notna() & bf[mask_zero].notna(),
    (ff[mask_zero] + bf[mask_zero]) / 2.0,           # 양쪽 있으면 평균
    ff[mask_zero].fillna(bf[mask_zero])              # 한쪽만 있으면 그 값
)



# 숫자형 보장(문자 섞였어도 그냥 처리)
for col in ['전력사용량(kWh)', '전기요금(원)','단가']:
    train_df[col] = pd.to_numeric(train_df[col], errors='coerce')

# 대상 컬럼
target_cols = ['단가', '전력사용량(kWh)', '전기요금(원)']

for col in target_cols:
    # inf → NaN 치환 후 유효행 마스크
    train_df[col].replace([np.inf, -np.inf], np.nan, inplace=True)
    mask = train_df[col].notna()

    # 1) 시간대별
    plt.figure(figsize=(10,4))
    train_df.loc[mask].boxplot(column=col, by='hour', grid=False, showfliers=False)
    plt.suptitle('')
    plt.title(f'{col} by 시간대 (hour)')
    plt.xlabel('시간대 (0–23시)')
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()

    # 2) 주말여부별
    plt.figure(figsize=(6,4))
    train_df.loc[mask].boxplot(column=col, by='is_weekend', grid=False, showfliers=False)
    plt.suptitle('')
    plt.title(f'{col} by 주말여부 (0=평일, 1=주말)')
    plt.xlabel('주말여부')
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()

    # 3) 작업유형별 (있을 때만)
    if '작업유형' in train_df.columns:
        order = train_df.loc[mask, '작업유형'].value_counts().index
        # 카테고리 순서 고정(많이 나온 순)
        train_df['작업유형'] = pd.Categorical(train_df['작업유형'], categories=order, ordered=True)

        plt.figure(figsize=(max(8, len(order)*0.9), 4.8))
        train_df.loc[mask].boxplot(column=col, by='작업유형', grid=False, rot=45, showfliers=False)
        plt.suptitle('')
        plt.title(f'{col} by 작업유형')
        plt.xlabel('작업유형')
        plt.ylabel(col)
        plt.tight_layout()
        plt.show()

train_df.columns
test_df.columns

# ============================================================================================================
# ================== 설정 ==================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "./data/raw/train.csv"
DATETIME_COL_CANDS = ["측정일시", "date", "datetime", "timestamp"]
ID_LIKE = {"id", "index"}  # 식별자 컬럼명(있으면 제외)

# 어떤 쌍을 그릴지 선택
PAIRS_MODE = "all"   # "target" 또는 "all"
TARGET_COL = "전력사용량(kWh)"  # PAIRS_MODE="target"일 때만 사용

# 그리기 옵션
SAMPLE_N = None          # 행이 많으면 샘플링(None이면 전체)
ALPHA = 0.35             # 점 투명도
FIGSIZE = (9, 7)         # 그림 크기(인치)
ADD_TREND = None         # 1차 추세선 표시
SAVE_DIR = None          # e.g., "./figs_scatter" 로 지정하면 PNG로 저장, None이면 저장 안 함
TOP_K = None             # PAIRS_MODE="all"일 때 상관계수(|r|) 상위 K쌍만 그림. None이면 전부.

# 폰트(윈도우 한글)
plt.rc("font", family="Malgun Gothic")
plt.rcParams["axes.unicode_minus"] = False

# ================== 유틸 함수 ==================
def read_csv_smart(path):
    for enc in ("cp949", "utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    # 마지막 시도 실패 시 에러 재발생
    return pd.read_csv(path)  # 기본

def _to_numeric_pair(df, x, y):
    tmp = df[[x, y]].copy()
    tmp[x] = pd.to_numeric(tmp[x], errors="coerce")
    tmp[y] = pd.to_numeric(tmp[y], errors="coerce")
    tmp = tmp.dropna()
    return tmp

def _trendline(ax, xs, ys):
    try:
        b1, b0 = np.polyfit(xs, ys, deg=1)
        xx = np.linspace(xs.min(), xs.max(), 200)
        yy = b1 * xx + b0
        ax.plot(xx, yy, linewidth=2)
    except Exception:
        pass

def plot_one_big(df, x, y, i=None):
    tmp = _to_numeric_pair(df, x, y)
    if len(tmp) == 0:
        print(f"[SKIP] {x} vs {y}: 유효 데이터 없음")
        return
    # 샘플링
    if SAMPLE_N is not None and len(tmp) > SAMPLE_N:
        tmp = tmp.sample(SAMPLE_N, random_state=42).sort_index()

    r = tmp[x].corr(tmp[y])

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.scatter(tmp[x], tmp[y], alpha=ALPHA, s=18)
    ax.set_xlabel(x); ax.set_ylabel(y)
    title_idx = f"[{i}]" if i is not None else ""
    ax.set_title(f"{title_idx} {x} vs {y} (n={len(tmp)}, r={r:.3f})")
    ax.grid(True, alpha=0.25)

    if ADD_TREND and len(tmp) >= 2:
        _trendline(ax, tmp[x].to_numpy(), tmp[y].to_numpy())

    plt.tight_layout()
    if SAVE_DIR:
        os.makedirs(SAVE_DIR, exist_ok=True)
        fname = f"{str(i).zfill(3) if i is not None else ''}_{x}__vs__{y}.png"
        # 파일명 안전화
        for bad in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '\n', '\t']:
            fname = fname.replace(bad, "_")
        path = os.path.join(SAVE_DIR, fname)
        plt.savefig(path, dpi=160)
        print(f"[SAVE] {path}")
    plt.show()

# ================== 실행 파트 ==================
# 1) 로드
train_df = read_csv_smart(CSV_PATH)

# 2) 날짜 컬럼 있으면 파싱(EDA용)
for c in DATETIME_COL_CANDS:
    if c in train_df.columns:
        train_df[c] = pd.to_datetime(train_df[c], errors="coerce")
        break

# 3) 수치형 컬럼 추출(+ id류 제외)
num_cols = train_df.select_dtypes(include=np.number).columns.tolist()
num_cols = [c for c in num_cols if c.lower() not in ID_LIKE]

if PAIRS_MODE == "target":
    if TARGET_COL not in train_df.columns:
        raise KeyError(f"TARGET_COL '{TARGET_COL}'이(가) 없습니다. 실제 컬럼명 확인하세요.")
    # 타깃도 숫자형으로 변환 가능해야 함
    num_cols_t = [c for c in num_cols if c != TARGET_COL]
    pairs = [(TARGET_COL, c) for c in num_cols_t]
else:  # "all"
    pairs_all = []
    for i, a in enumerate(num_cols):
        for b in num_cols[i+1:]:
            pairs_all.append((a, b))
    # TOP_K 옵션(상관계수 기준 상위만)
    if TOP_K is not None:
        # 빠르게 추정: 각 쌍 r 계산(샘플 일부로 가볍게)
        preview = train_df[num_cols].copy()
        if len(preview) > 12000:
            preview = preview.sample(12000, random_state=42)
        # r 계산
        scores = []
        for a, b in pairs_all:
            tmp = _to_numeric_pair(preview, a, b)
            r = tmp[a].corr(tmp[b]) if len(tmp) else np.nan
            scores.append((abs(r) if pd.notna(r) else -1, a, b))
        scores.sort(reverse=True)  # |r| 큰 순
        pairs = [(a, b) for _, a, b in scores[:TOP_K]]
        print(f"[INFO] TOP_K={TOP_K} 적용: 총 {len(pairs)}쌍")
    else:
        pairs = pairs_all

print(f"[INFO] 그릴 쌍 개수: {len(pairs)}")

# 4) 루프 돌며 큰 산점도 하나씩
for i, (x, y) in enumerate(pairs, start=1):
    plot_one_big(train_df, x, y, i=i)

"""
사용 예시
- 타깃 기준으로만 그리고 싶으면:
    PAIRS_MODE = "target"; TARGET_COL = "전기요금(원)"

- 모든 수치형 쌍 중 |r| 상위 20쌍만:
    PAIRS_MODE = "all"; TOP_K = 20

- 이미지로 저장하고 싶으면:
    SAVE_DIR = "./figs_scatter"
"""