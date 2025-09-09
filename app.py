# app.py
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="네이버 제품 상세페이지 데이터: 전환율(X) · 조회수(Y)", layout="wide")
st.title("네이버 제품 상세페이지 데이터: 전환율(X) · 조회수(Y)")
st.caption("좌표에 마우스를 올리면 상품명이 보입니다. (Plotly 인터랙션)")

# ✅ 사분면 해석 추가
st.markdown("""
### 📊 사분면 해석
- **1사분면 (오른쪽 위)** → 효자상품 (조회수↑ 전환율↑)
- **2사분면 (왼쪽 위)** → 전환개선 필요 (조회수↑ 전환율↓)
- **3사분면 (왼쪽 아래)** → 비인기상품 (조회수↓ 전환율↓)
- **4사분면 (오른쪽 아래)** → 노출확대 필요 (조회수↓ 전환율↑)
""")

# =========================
# 사이드바
# =========================
with st.sidebar:
    st.header("데이터 설정")
    st.write("두 개 엑셀에서 1~8월 시트를 읽어 통합합니다.")
    path_conv = st.text_input("엑셀① (전환율 기준)", "data/제품 상세페이지 순위(전환율 기준).xlsx")
    path_view = st.text_input("엑셀② (조회수 기준)", "data/제품 상세페이지 순위(조회수 기준).xlsx")

    months = [f"{i}월" for i in range(1, 9)]
    use_months = st.multiselect("대상 월", months, default=months)

    remove_misc = st.checkbox("상품명에 '기타' 포함 제외", value=True)
    min_views = st.number_input("최소 조회수 필터", value=0, step=100)

    st.markdown("---")
    st.subheader("버블 크기")
    size_mode = st.radio("방식 선택", ["고정", "결제수 기반"], index=0)

    # ✅ 픽셀 단위로 직접 지정 (float 슬라이더)
    fixed_size = st.slider("고정 크기(px)", min_value=3.0, max_value=30.0, value=7.0, step=0.5)
    size_min   = st.slider("최소 버블(px)",  0.5, 30.0, 2.0, 0.5)
    size_max   = st.slider("최대 버블(px)",  1.0, 40.0, 15.0, 0.5)

    st.markdown("---")
    with st.expander("Y축(조회수) 범위 조절"):
        y_pct = st.slider("상한 퍼센타일(%)", 90, 100, 99, 1)
        y_pad = st.slider("하단 여유(%)", 0, 20, 10, 1)  # 축 하한을 -여유만큼 내림

# =========================
# 데이터 읽기/정리
# =========================
def read_all(path_list, months):
    df_list = []
    for p in path_list:
        p = Path(p)
        if not p.exists():
            st.warning(f"❗ 파일을 찾을 수 없습니다: {p}")
            continue

        xls = pd.ExcelFile(p, engine="openpyxl")
        for m in months:
            if m not in xls.sheet_names:
                continue
            df = pd.read_excel(p, sheet_name=m, engine="openpyxl")

            # 1) 컬럼명 공백 제거
            df.rename(columns=lambda c: str(c).strip(), inplace=True)

            # 2) 순위/상품명 → 숫자 제거 후 상품명
            if "순위/상품명" in df.columns:
                df["순위/상품명"] = df["순위/상품명"].astype(str).str.replace(r"^\d+\s*", "", regex=True)
                df.rename(columns={"순위/상품명": "상품명"}, inplace=True)

            # 3) 컬럼명 통일
            col_map = {}
            for c in df.columns:
                key = str(c).replace(" ", "")
                if key in ("상품상세조회수", "조회수"):
                    col_map[c] = "조회수"
                elif key in ("결제수", "구매수", "결제건수"):
                    col_map[c] = "결제수"
                elif key in ("결제금액", "결제금액(원)", "매출"):
                    col_map[c] = "결제금액"
            if col_map:
                df.rename(columns=col_map, inplace=True)

            # 4) 숫자형 변환
            for col in ["조회수", "결제수", "결제금액"]:
                if col in df.columns:
                    df[col] = (
                        df[col].astype(str)
                        .str.replace(",", "")
                        .str.replace("%", "")
                        .str.replace("\u200b", "")
                    )
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # 메타
            df["월"] = m
            df["파일"] = p.name
            df_list.append(df)

    if not df_list:
        return pd.DataFrame()

    out = pd.concat(df_list, ignore_index=True)

    # 상품명+월 기준 중복 제거 (합집합에서 중복 제거)
    if "상품명" in out.columns:
        out = out.drop_duplicates(subset=["상품명", "월"])

    # 필수 체크
    required = {"조회수", "결제수"}
    missing = required - set(out.columns)
    if missing:
        st.error(f"필수 컬럼 누락: {', '.join(sorted(missing))}")
        st.stop()

    return out

df_all = read_all([path_conv, path_view], use_months)
if df_all.empty:
    st.stop()

# 기타 제외
if remove_misc and "상품명" in df_all.columns:
    df_all = df_all[~df_all["상품명"].str.contains("기타", na=False)]

# 전환율 계산
df_all["전환율"] = np.where(df_all["조회수"] > 0, df_all["결제수"] / df_all["조회수"], np.nan)
df_all.replace([np.inf, -np.inf], np.nan, inplace=True)
df_all.dropna(subset=["전환율", "조회수"], inplace=True)

# 최소 조회수 필터
df_all = df_all[df_all["조회수"] >= min_views]

# =========================
# 그림 함수 (버블 픽셀 직접 지정)
# =========================
def quadrant_fig(df, month_label, size_mode="고정",
                 fixed_size=3.0, size_min=2.0, size_max=10.0,
                 y_pct=99, y_pad=3):
    d = df[df["월"] == month_label].copy()
    if d.empty:
        return px.scatter(title=f"{month_label} 사분면")

    # 버블 크기 계산
    if size_mode == "결제수 기반" and "결제수" in d.columns:
        s_raw = d["결제수"].astype(float).values
        if len(s_raw) > 1:
            lo, hi = np.nanpercentile(s_raw, [5, 95])
            if hi - lo == 0:
                sizes = np.full_like(s_raw, fixed_size, dtype=float)
            else:
                scaled = (np.clip(s_raw, lo, hi) - lo) / (hi - lo)
                sizes = size_min + scaled * (size_max - size_min)
        else:
            sizes = np.full_like(s_raw, fixed_size, dtype=float)
    else:
        sizes = np.full(len(d), fixed_size, dtype=float)

    # 최소 0.5px 보장
    sizes = np.clip(sizes, 1.5, None)

    # 중앙값 기준선
    x = d["전환율"].values
    y = d["조회수"].values
    xv = float(np.nanmedian(x))
    yv = float(np.nanmedian(y))

    # px.scatter에서 size 자동 스케일링 사용 안 함
    fig = px.scatter(
        d,
        x="전환율",
        y="조회수",
        hover_name="상품명",
        hover_data={
            "월": True,
            "조회수": ":,",
            "결제수": ":,",
            "결제금액": ":,",
            "전환율": ":.2%",
            "파일": True,
        },
        labels={"전환율": "전환율", "조회수": "상품상세 조회수"},
        title=f"{month_label} 사분면",
    )

    # 픽셀 크기 직접 지정
    fig.update_traces(marker=dict(size=sizes, sizemode="diameter", line=dict(width=0)))

    # 기준선
    fig.add_hline(y=yv, line_dash="dash", line_width=1)
    fig.add_vline(x=xv, line_dash="dash", line_width=1)

    # 축 포맷 + 범위
    fig.update_xaxes(tickformat=".1%")
    if np.isfinite(y).any():
        y_cap = float(np.nanpercentile(y, y_pct))
        bottom = -y_cap * (y_pad / 100)
        top = y_cap * 1.30
        if top <= 0:  # 방어적 처리
            top = float(np.nanmax(y)) * 1.10
        fig.update_yaxes(range=[bottom, top], tickformat=",", separatethousands=True)
    else:
        fig.update_yaxes(tickformat=",", separatethousands=True)

    fig.update_layout(margin=dict(l=40, r=20, t=60, b=90), height=520, hovermode="closest")
    return fig

# =========================
# 레이아웃
# =========================
tab_keys = use_months if use_months else ["1월", "8월"]
tabs = st.tabs(tab_keys)
for i, m in enumerate(tab_keys):
    with tabs[i]:
        fig = quadrant_fig(
            df_all, m,
            size_mode=size_mode,
            fixed_size=fixed_size,
            size_min=size_min,
            size_max=size_max,
            y_pct=y_pct,
            y_pad=y_pad,
        )
        st.plotly_chart(fig, use_container_width=True)

# ✅ 요약 지표 (전체 + 월별) — 기존 방식 유지, 월별은 단순 필터+합계
with st.expander("요약 지표 보기", expanded=True):
    # ----- 전체 합계 (지금 되던 방식 그대로) -----
    total_views  = df_all["조회수"].sum()
    total_orders = df_all["결제수"].sum()
    total_sales  = df_all["결제금액"].sum() if "결제금액" in df_all.columns else 0
    avg_cvr      = (total_orders / total_views) if total_views > 0 else 0

    st.markdown(f"""
    **[1-8월 전환율, 조회수 TOP 50 data 전체 합계]**
    - **총 조회수**: {total_views:,.0f}
    - **총 결제수**: {total_orders:,.0f}
    - **총 결제금액**: {total_sales:,.0f} 원
    - **평균 전환율**: {avg_cvr:.2%}
    """)
