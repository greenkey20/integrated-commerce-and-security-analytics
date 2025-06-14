import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Ellipse
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings

warnings.filterwarnings("ignore")


@st.cache_resource
def setup_korean_font_for_streamlit():
    """Streamlit용 한글 폰트 설정 (캐싱 적용)"""

    # 진단에서 확인된 신뢰할 수 있는 폰트들
    reliable_fonts = [
        {
            "name": "AppleGothic",
            "path": "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        },
        {"name": "Arial Unicode MS", "path": "/Library/Fonts/Arial Unicode.ttf"},
        {"name": "Helvetica", "path": "/System/Library/Fonts/Helvetica.ttc"},
    ]

    for font_info in reliable_fonts:
        font_path = font_info["path"]
        font_name = font_info["name"]

        if os.path.exists(font_path):
            try:
                # 폰트를 matplotlib에 등록
                fm.fontManager.addfont(font_path)

                # FontProperties 객체 생성
                font_prop = fm.FontProperties(fname=font_path)
                actual_name = font_prop.get_name()

                # matplotlib 전역 설정 적용
                plt.rcParams["font.family"] = [actual_name]
                plt.rcParams["font.sans-serif"] = [actual_name] + plt.rcParams[
                    "font.sans-serif"
                ]
                plt.rcParams["axes.unicode_minus"] = False

                return font_prop, actual_name

            except Exception:
                continue

    # 폰트 설정 실패 시 기본값 반환
    return None, None


def analyze_cluster_characteristics(data_with_clusters, n_clusters):
    """클러스터별 특성을 분석하여 동적 라벨과 색상을 생성"""

    cluster_profiles = []

    for cluster_id in range(n_clusters):
        cluster_data = data_with_clusters[data_with_clusters["Cluster"] == cluster_id]

        if len(cluster_data) == 0:
            continue

        profile = {
            "cluster_id": cluster_id,
            "size": len(cluster_data),
            "avg_income": cluster_data["Annual Income (k$)"].mean(),
            "avg_spending": cluster_data["Spending Score (1-100)"].mean(),
            "avg_age": cluster_data["Age"].mean(),
            "std_income": cluster_data["Annual Income (k$)"].std(),
            "std_spending": cluster_data["Spending Score (1-100)"].std(),
        }
        cluster_profiles.append(profile)

    # 전체 클러스터 대비 상대적 위치 계산
    all_incomes = [p["avg_income"] for p in cluster_profiles]
    all_spendings = [p["avg_spending"] for p in cluster_profiles]
    all_ages = [p["avg_age"] for p in cluster_profiles]

    income_quartiles = np.percentile(all_incomes, [25, 50, 75])
    spending_quartiles = np.percentile(all_spendings, [25, 50, 75])
    age_quartiles = np.percentile(all_ages, [25, 50, 75])

    # 각 클러스터에 대한 동적 라벨 생성
    for profile in cluster_profiles:
        # 소득 수준 분류 (더 세분화)
        if profile["avg_income"] >= income_quartiles[2]:
            if profile["avg_income"] >= np.percentile(all_incomes, 90):
                income_level = "최고소득"
            else:
                income_level = "고소득"
        elif profile["avg_income"] >= income_quartiles[1]:
            income_level = "중상소득"
        elif profile["avg_income"] >= income_quartiles[0]:
            income_level = "중하소득"
        else:
            income_level = "저소득"

        # 지출 수준 분류 (더 세분화)
        if profile["avg_spending"] >= spending_quartiles[2]:
            if profile["avg_spending"] >= np.percentile(all_spendings, 90):
                spending_level = "최고지출"
            else:
                spending_level = "고지출"
        elif profile["avg_spending"] >= spending_quartiles[1]:
            spending_level = "중상지출"
        elif profile["avg_spending"] >= spending_quartiles[0]:
            spending_level = "중하지출"
        else:
            spending_level = "저지출"

        # 연령대 분류
        if profile["avg_age"] <= age_quartiles[0]:
            age_group = "청년층"
        elif profile["avg_age"] <= age_quartiles[1]:
            age_group = "청장년층"
        elif profile["avg_age"] <= age_quartiles[2]:
            age_group = "중년층"
        else:
            age_group = "장년층"

        # 고객 유형 결정 (소득과 지출 조합)
        if income_level in ["최고소득", "고소득"] and spending_level in [
            "최고지출",
            "고지출",
        ]:
            customer_type = "프리미엄"
        elif income_level in ["최고소득", "고소득"] and spending_level in [
            "저지출",
            "중하지출",
        ]:
            customer_type = "보수적"
        elif income_level in ["저소득", "중하소득"] and spending_level in [
            "고지출",
            "최고지출",
        ]:
            customer_type = "적극소비"
        elif income_level in ["저소득", "중하소득"] and spending_level in [
            "저지출",
            "중하지출",
        ]:
            customer_type = "절약형"
        else:
            customer_type = "일반"

        # 최종 라벨 생성
        profile["label"] = f"{customer_type} {age_group}"
        profile["income_level"] = income_level
        profile["spending_level"] = spending_level
        profile["age_group"] = age_group
        profile["customer_type"] = customer_type

    return cluster_profiles


def generate_dynamic_colors(cluster_profiles):
    """클러스터 특성에 따른 일관된 색상 매핑 생성"""

    # 기본 색상 팔레트 (더 많은 색상)
    base_colors = [
        "#e41a1c",  # 빨강 - 프리미엄/고소득
        "#377eb8",  # 파랑 - 보수적/안정적
        "#4daf4a",  # 초록 - 일반/균형적
        "#984ea3",  # 보라 - 적극소비/젊은층
        "#ff7f00",  # 주황 - 절약형/실용적
        "#ffff33",  # 노랑 - 특별 카테고리
        "#a65628",  # 갈색 - 중년층/전통적
        "#f781bf",  # 분홍 - 여성적/감성적
        "#999999",  # 회색 - 중립적
        "#66c2a5",  # 청록
    ]

    colors = []
    for i, profile in enumerate(cluster_profiles):
        # 고객 유형에 따른 색상 선택
        if profile["customer_type"] == "프리미엄":
            colors.append("#e41a1c")  # 빨강
        elif profile["customer_type"] == "보수적":
            colors.append("#377eb8")  # 파랑
        elif profile["customer_type"] == "적극소비":
            colors.append("#984ea3")  # 보라
        elif profile["customer_type"] == "절약형":
            colors.append("#ff7f00")  # 주황
        else:  # 일반
            colors.append(base_colors[i % len(base_colors)])

    return colors


def generate_dynamic_interpretation_guide(cluster_profiles):
    """동적 클러스터 해석 가이드 생성"""

    if len(cluster_profiles) == 0:
        return "클러스터 분석 결과를 확인할 수 없습니다."

    # 소득과 지출 범위 계산
    min_income = min(p["avg_income"] for p in cluster_profiles)
    max_income = max(p["avg_income"] for p in cluster_profiles)
    min_spending = min(p["avg_spending"] for p in cluster_profiles)
    max_spending = max(p["avg_spending"] for p in cluster_profiles)

    guide_text = f"""
    **현재 {len(cluster_profiles)}개 클러스터 분석 결과 해석:**
    
    **소득 범위:** ${min_income:.1f}k ~ ${max_income:.1f}k
    **지출점수 범위:** {min_spending:.1f} ~ {max_spending:.1f}
    
    **각 클러스터의 특성:**
    """

    # 소득 순으로 정렬하여 설명
    sorted_profiles = sorted(
        cluster_profiles, key=lambda x: x["avg_income"], reverse=True
    )

    for profile in sorted_profiles:
        guide_text += f"""
    - **클러스터 {profile['cluster_id']} ({profile['label']})**: 
      평균 소득 ${profile['avg_income']:.1f}k, 지출점수 {profile['avg_spending']:.1f}, 
      고객 수 {profile['size']}명 ({profile['customer_type']} 고객층)
        """

    guide_text += f"""
    
    **클러스터링 품질 지표:**
    - 클러스터 간 소득 격차: ${max_income - min_income:.1f}k
    - 클러스터 간 지출성향 차이: {max_spending - min_spending:.1f}점
    - 가장 큰 클러스터: {max(cluster_profiles, key=lambda x: x['size'])['size']}명
    - 가장 작은 클러스터: {min(cluster_profiles, key=lambda x: x['size'])['size']}명
    """

    return guide_text


# 한글 폰트 설정 실행
korean_font_prop, korean_font_name = setup_korean_font_for_streamlit()

# 페이지 설정
st.set_page_config(
    page_title="고객 세분화 분석 서비스",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 제목 및 소개
st.title("🛍️ Mall Customer Segmentation Analysis")
st.markdown(
    """
이 애플리케이션은 K-means 클러스터링을 활용하여 쇼핑몰 고객을 세분화하고 
각 그룹별 특성을 분석하여 맞춤형 마케팅 전략을 제공합니다.
"""
)

# 사이드바 메뉴
st.sidebar.title("📋 Navigation")
menu = st.sidebar.selectbox(
    "메뉴를 선택하세요:",
    [
        "데이터 개요",
        "탐색적 데이터 분석",
        "클러스터링 분석",
        "고객 예측",
        "마케팅 전략",
    ],
)


@st.cache_data
def load_data():
    """데이터 로드 및 전처리"""
    try:
        # GitHub에서 직접 데이터 로드
        url = "https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Datasets/Mall_Customers.csv"
        data = pd.read_csv(url)
        return data
    except:
        # 샘플 데이터 생성 (실제 환경에서는 실제 데이터 사용)
        np.random.seed(42)
        sample_data = {
            "CustomerID": range(1, 201),
            "Gender": np.random.choice(["Male", "Female"], 200),
            "Age": np.random.normal(40, 15, 200).astype(int),
            "Annual Income (k$)": np.random.normal(60, 20, 200).astype(int),
            "Spending Score (1-100)": np.random.normal(50, 25, 200).astype(int),
        }
        data = pd.DataFrame(sample_data)
        data["Age"] = np.clip(data["Age"], 18, 80)
        data["Annual Income (k$)"] = np.clip(data["Annual Income (k$)"], 15, 150)
        data["Spending Score (1-100)"] = np.clip(data["Spending Score (1-100)"], 1, 100)
        return data


@st.cache_data
def perform_clustering(data, n_clusters=5):
    """K-means 클러스터링 수행"""
    # 클러스터링을 위한 특성 선택
    features = data[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]

    # 데이터 정규화
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # K-means 클러스터링
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_features)

    # 실루엣 점수 계산
    silhouette_avg = silhouette_score(scaled_features, clusters)

    return clusters, kmeans, scaler, silhouette_avg


def find_optimal_clusters(data, max_k=10):
    """엘보우 방법으로 최적 클러스터 수 찾기"""
    features = data[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    inertias = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled_features)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(scaled_features, clusters))

    return k_range, inertias, silhouette_scores


def get_dynamic_marketing_strategy(cluster_id, profile, all_profiles):
    """각 클러스터의 상대적 특성을 고려한 동적 마케팅 전략 생성"""

    # 전체 클러스터 대비 상대적 위치 계산
    all_incomes = [p["avg_income"] for p in all_profiles.values()]
    all_spendings = [p["avg_spending"] for p in all_profiles.values()]
    all_ages = [p["avg_age"] for p in all_profiles.values()]

    income_percentile = (
        sum(1 for x in all_incomes if x < profile["avg_income"]) / len(all_incomes)
    ) * 100
    spending_percentile = (
        sum(1 for x in all_spendings if x < profile["avg_spending"])
        / len(all_spendings)
    ) * 100
    age_percentile = (
        sum(1 for x in all_ages if x < profile["avg_age"]) / len(all_ages)
    ) * 100

    # 소득 수준 분류
    if income_percentile >= 75:
        income_level = "고소득"
    elif income_percentile >= 40:
        income_level = "중간소득"
    else:
        income_level = "저소득"

    # 지출 수준 분류
    if spending_percentile >= 75:
        spending_level = "고지출"
    elif spending_percentile >= 40:
        spending_level = "중간지출"
    else:
        spending_level = "저지출"

    # 연령대 분류
    if age_percentile <= 25:
        age_group = "젊은층"
    elif age_percentile >= 75:
        age_group = "중장년층"
    else:
        age_group = "중간연령층"

    # 세그먼트 명 생성
    segment_name = f"{income_level} {spending_level} {age_group}"

    # 전략 생성
    strategies = []
    priorities = []

    # 소득 기반 전략
    if income_level == "고소득":
        if spending_level == "고지출":
            strategies.append("프리미엄 제품 라인 집중, VIP 서비스")
            priorities.append("최우선")
        elif spending_level == "저지출":
            strategies.append("가치 제안 마케팅, 투자 상품 소개")
            priorities.append("높음")
        else:
            strategies.append("품질 중심 마케팅, 브랜드 가치 강조")
            priorities.append("높음")
    elif income_level == "중간소득":
        if spending_level == "고지출":
            strategies.append("할부 서비스, 캐시백 혜택")
            priorities.append("중간")
        else:
            strategies.append("합리적 가격대 제품, 프로모션 활용")
            priorities.append("중간")
    else:  # 저소득
        strategies.append("저가 제품 라인, 대량 할인, 멤버십 혜택")
        priorities.append("낮음")

    # 연령 기반 추가 전략
    if age_group == "젊은층":
        strategies.append("소셜미디어 마케팅, 온라인 채널 강화")
    elif age_group == "중장년층":
        strategies.append("오프라인 매장 서비스, 전화 상담 강화")
    else:
        strategies.append("옴니채널 접근, 다양한 커뮤니케이션")

    # 특별한 조합에 대한 맞춤 전략
    if income_level == "저소득" and spending_level == "고지출":
        strategies.append("신용 관리 서비스, 예산 관리 도구 제공")
    elif income_level == "고소득" and spending_level == "저지출":
        strategies.append("절약 보상 프로그램, 장기 고객 혜택")

    return {
        "segment": segment_name,
        "strategy": "; ".join(strategies),
        "priority": priorities[0] if priorities else "보통",
        "income_level": income_level,
        "spending_level": spending_level,
        "age_group": age_group,
        "percentiles": {
            "income": f"{income_percentile:.0f}%",
            "spending": f"{spending_percentile:.0f}%",
            "age": f"{age_percentile:.0f}%",
        },
    }


# 데이터 로드
data = load_data()

if menu == "데이터 개요":
    st.header("📊 데이터 개요")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("데이터셋 정보")
        st.write(f"전체 고객 수: {len(data):,}명")
        st.write(f"특성 수: {len(data.columns)}개")
        st.write("데이터 타입:")
        # DataFrame으로 변환하여 안전하게 출력
        dtypes_df = pd.DataFrame(
            {
                "컬럼명": data.columns,
                "데이터 타입": [str(dtype) for dtype in data.dtypes],
            }
        )
        st.dataframe(dtypes_df, use_container_width=True)

    with col2:
        st.subheader("기본 통계")
        st.write(data.describe())

    st.subheader("데이터 미리보기")
    st.dataframe(data.head(10))

    # 결측값 확인
    st.subheader("데이터 품질 검사")
    missing_values = data.isnull().sum()
    if missing_values.sum() == 0:
        st.success("✅ 결측값이 없습니다.")
    else:
        st.warning("⚠️ 결측값이 발견되었습니다:")
        st.write(missing_values[missing_values > 0])

elif menu == "탐색적 데이터 분석":
    st.header("🔍 탐색적 데이터 분석")

    # 성별 분포
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("성별 분포")
        gender_counts = data["Gender"].value_counts()
        fig = px.pie(
            values=gender_counts.values,
            names=gender_counts.index,
            title="고객 성별 분포",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("연령 분포")
        fig = px.histogram(data, x="Age", nbins=20, title="연령 분포")
        fig.update_layout(xaxis_title="연령", yaxis_title="고객 수")
        st.plotly_chart(fig, use_container_width=True)

    # 소득 vs 지출 점수 산점도
    st.subheader("소득 대비 지출 점수 분석")
    fig = px.scatter(
        data,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        color="Gender",
        title="연간 소득 vs 지출 점수",
        hover_data=["Age"],
    )
    fig.update_layout(
        xaxis_title="연간 소득 (천 달러)", yaxis_title="지출 점수 (1-100)"
    )
    st.plotly_chart(fig, use_container_width=True)

    # 상관관계 히트맵
    st.subheader("특성 간 상관관계")
    numeric_cols = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    correlation_matrix = data[numeric_cols].corr()

    fig = px.imshow(
        correlation_matrix,
        labels=dict(color="상관계수"),
        x=numeric_cols,
        y=numeric_cols,
        title="특성 간 상관관계 히트맵",
    )
    st.plotly_chart(fig, use_container_width=True)

elif menu == "클러스터링 분석":
    st.header("🎯 클러스터링 분석")

    # 클러스터링 이론 설명 섹션 추가
    with st.expander("📚 클러스터링 분석 이론 가이드", expanded=True):
        st.markdown(
            """
        ### 🤔 왜 클러스터 개수를 미리 결정해야 할까요?
        
        K-means 알고리즘의 가장 큰 특징 중 하나는 **사전에 클러스터 개수(K)를 지정해야 한다는 것**입니다. 
        이는 마치 케이크를 자를 때 "몇 조각으로 나눌까?"를 미리 정해야 하는 것과 같습니다. 
        하지만 실제 데이터에서는 최적의 클러스터 개수를 모르기 때문에, 과학적인 방법으로 이를 결정해야 합니다.
        
        ### 📈 엘보우 방법 (Elbow Method)
        
        **핵심 아이디어**: 클러스터 개수에 따른 "성능 대비 효율성"을 측정하는 방법입니다.
        
        - **Inertia(관성)**: 각 데이터 포인트와 해당 클러스터 중심점 간의 거리 제곱의 총합
        - **해석 방법**: 그래프에서 급격히 꺾이는 지점(팔꿈치 모양)을 찾습니다
        - **비유**: 마치 가격 대비 성능을 따질 때 "가성비"가 급격히 나빠지는 지점을 찾는 것과 같습니다
        
        **📊 그래프 읽는 법**: 
        - 클러스터가 적으면 → Inertia 높음 (분류가 거침)
        - 클러스터가 많으면 → Inertia 낮음 (하지만 과도한 세분화)
        - **최적점**: Inertia가 급격히 감소하다가 완만해지는 지점
        
        ### 🎯 실루엣 점수 (Silhouette Score)
        
        **핵심 아이디어**: 각 데이터가 자신의 클러스터에 얼마나 "잘 맞는지"를 측정합니다.
        
        - **점수 범위**: -1 ~ 1 (높을수록 좋음)
        - **의미**: 
          - 0.7~1.0: 매우 좋은 클러스터링
          - 0.5~0.7: 적절한 클러스터링  
          - 0.25~0.5: 약한 클러스터링
          - 0 이하: 잘못된 클러스터링
        
        **📊 그래프 읽는 법**:
        - 실루엣 점수가 가장 높은 지점이 최적의 클러스터 개수
        - 점수가 지속적으로 감소한다면 더 적은 클러스터가 적합
        
        ### 🎲 두 방법을 함께 사용하는 이유
        
        엘보우 방법과 실루엣 점수는 서로 다른 관점에서 클러스터 품질을 평가합니다:
        - **엘보우**: "효율성" 관점 (비용 대비 효과)
        - **실루엣**: "품질" 관점 (분류의 명확성)
        
        **최종 결정**: 두 방법에서 공통으로 좋은 결과를 보이는 클러스터 개수를 선택하는 것이 가장 안전합니다.
        """
        )

    # 최적 클러스터 수 찾기
    st.subheader("🔍 최적 클러스터 수 결정")
    st.write(
        "다양한 클러스터 개수에 대해 엘보우 방법과 실루엣 분석을 수행하여 최적의 K값을 찾아보겠습니다."
    )

    with st.spinner("최적 클러스터 수를 분석중입니다..."):
        k_range, inertias, silhouette_scores = find_optimal_clusters(data)

    col1, col2 = st.columns(2)

    with col1:
        # 엘보우 방법
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(k_range),
                y=inertias,
                mode="lines+markers",
                name="Inertia",
                line=dict(color="blue", width=3),
                marker=dict(size=8),
            )
        )
        fig.update_layout(
            title="엘보우 방법: Inertia 변화",
            xaxis_title="클러스터 수",
            yaxis_title="Inertia (관성)",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        # 엘보우 방법 해석
        st.info(
            """
        **📊 이 그래프 해석하기:**
        - 클러스터 수가 증가할수록 Inertia는 감소합니다
        - 급격히 꺾이는 지점(엘보우)을 찾으세요
        - 보통 2-3번 클러스터 지점에서 기울기가 완만해집니다
        """
        )

    with col2:
        # 실루엣 점수
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(k_range),
                y=silhouette_scores,
                mode="lines+markers",
                name="Silhouette Score",
                line=dict(color="red", width=3),
                marker=dict(size=8),
            )
        )
        fig.update_layout(
            title="실루엣 점수 변화",
            xaxis_title="클러스터 수",
            yaxis_title="실루엣 점수",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        # 실루엣 점수 해석
        best_k_silhouette = k_range[np.argmax(silhouette_scores)]
        best_silhouette_score = max(silhouette_scores)

        st.info(
            f"""
        **📊 이 그래프 해석하기:**
        - 가장 높은 점수: {best_silhouette_score:.3f} (K={best_k_silhouette})
        - 점수가 0.5 이상이면 적절한 클러스터링
        - 가장 높은 지점이 최적의 클러스터 개수입니다
        """
        )

    # 분석 결과 종합 및 권장사항 제시
    st.subheader("🎯 분석 결과 종합 및 권장사항")

    # 엘보우 방법으로 최적 K 추정 (간단한 휴리스틱)
    inertia_diffs = np.diff(inertias)
    inertia_diffs2 = np.diff(inertia_diffs)
    elbow_k = (
        k_range[np.argmax(inertia_diffs2) + 2]
        if len(inertia_diffs2) > 0
        else k_range[0]
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="엘보우 방법 추천",
            value=f"{elbow_k}개 클러스터",
            help="Inertia 감소율이 가장 크게 변하는 지점",
        )

    with col2:
        st.metric(
            label="실루엣 점수 추천",
            value=f"{best_k_silhouette}개 클러스터",
            delta=f"점수: {best_silhouette_score:.3f}",
            help="실루엣 점수가 가장 높은 지점",
        )

    with col3:
        # 최종 권장값 (두 방법의 절충안)
        if elbow_k == best_k_silhouette:
            recommended_k = elbow_k
            agreement = "✅ 완전 일치"
        else:
            recommended_k = int((elbow_k + best_k_silhouette) / 2)
            agreement = f"📊 절충안"

        st.metric(
            label="최종 권장",
            value=f"{recommended_k}개 클러스터",
            delta=agreement,
            help="두 방법을 종합한 최종 권장사항",
        )

    # 권장사항 설명
    if elbow_k == best_k_silhouette:
        st.success(
            f"🎉 **두 방법이 모두 {elbow_k}개 클러스터를 추천합니다!** 이는 매우 신뢰할 수 있는 결과입니다."
        )
    else:
        st.warning(
            f"""
        📊 **두 방법의 결과가 다릅니다:**
        - 엘보우 방법: {elbow_k}개 (효율성 관점)
        - 실루엣 점수: {best_k_silhouette}개 (품질 관점)
        
        이런 경우 도메인 지식과 비즈니스 목적을 고려하여 최종 결정하시기 바랍니다.
        """
        )

    # 클러스터 수 선택 슬라이더 (Session State 활용)
    st.subheader("⚙️ 클러스터 수 선택")
    st.write(
        "위 분석 결과를 참고하여 최종 클러스터 개수를 선택하세요. 이 설정은 다음 페이지들에서도 일관되게 적용됩니다."
    )

    # Session State 초기화
    if "selected_clusters" not in st.session_state:
        st.session_state.selected_clusters = recommended_k

    selected_k = st.slider(
        "클러스터 수 선택:",
        min_value=2,
        max_value=10,
        value=st.session_state.selected_clusters,
        help=f"분석 결과 권장: {recommended_k}개",
    )

    # Session State 업데이트
    st.session_state.selected_clusters = selected_k

    # 선택된 클러스터 수에 대한 실시간 피드백
    if selected_k == recommended_k:
        st.success(f"✅ 분석 권장값과 일치합니다. ({selected_k}개)")
    elif selected_k in [elbow_k, best_k_silhouette]:
        st.info(f"📊 분석 방법 중 하나가 추천하는 값입니다. ({selected_k}개)")
    else:
        st.warning(f"⚠️ 분석 권장값과 다릅니다. 특별한 이유가 있는지 확인해보세요.")

    # 선택된 K로 클러스터링 수행
    optimal_k = selected_k

    # 클러스터링 수행
    clusters, kmeans, scaler, silhouette_avg = perform_clustering(data, optimal_k)
    data_with_clusters = data.copy()
    data_with_clusters["Cluster"] = clusters

    st.success(f"✅ 클러스터링 완료! 실루엣 점수: {silhouette_avg:.3f}")

    # 동적 클러스터 분석 수행
    cluster_profiles = analyze_cluster_characteristics(data_with_clusters, optimal_k)
    dynamic_colors = generate_dynamic_colors(cluster_profiles)
    interpretation_guide = generate_dynamic_interpretation_guide(cluster_profiles)

    # 클러스터별 시각화
    st.subheader("클러스터 시각화")

    # 3D 산점도
    fig = px.scatter_3d(
        data_with_clusters,
        x="Age",
        y="Annual Income (k$)",
        z="Spending Score (1-100)",
        color="Cluster",
        title="3D 클러스터 시각화",
        hover_data=["Gender"],
    )
    st.plotly_chart(fig, use_container_width=True)

    # 2D 산점도 (소득 vs 지출점수)
    fig = px.scatter(
        data_with_clusters,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        color="Cluster",
        title="클러스터별 소득 vs 지출점수",
        hover_data=["Age", "Gender"],
    )
    st.plotly_chart(fig, use_container_width=True)

    # 클러스터별 특성 분석
    st.subheader("클러스터별 특성 분석")

    cluster_summary = (
        data_with_clusters.groupby("Cluster")
        .agg(
            {
                "Age": ["mean", "std"],
                "Annual Income (k$)": ["mean", "std"],
                "Spending Score (1-100)": ["mean", "std"],
            }
        )
        .round(2)
    )

    cluster_summary.columns = [
        "평균_연령",
        "표준편차_연령",
        "평균_소득",
        "표준편차_소득",
        "평균_지출점수",
        "표준편차_지출점수",
    ]

    st.dataframe(cluster_summary)

    # 클러스터별 고객 수
    cluster_counts = data_with_clusters["Cluster"].value_counts().sort_index()
    fig = px.bar(
        x=cluster_counts.index, y=cluster_counts.values, title="클러스터별 고객 수"
    )
    fig.update_layout(xaxis_title="클러스터", yaxis_title="고객 수")
    st.plotly_chart(fig, use_container_width=True)

    # 새로 추가: 클러스터 결과가 표시된 산점도 (matplotlib 사용)
    st.subheader("🎯 클러스터 분석 결과 상세 시각화")
    st.write("각 고객이 어떤 클러스터에 속하는지 색상과 영역으로 구분하여 보여줍니다.")

    # matplotlib를 사용한 상세 클러스터 시각화
    fig_detailed, ax = plt.subplots(figsize=(12, 8))

    # 클러스터 중심점을 원본 스케일로 역변환
    cluster_centers_scaled = kmeans.cluster_centers_
    cluster_centers_original = scaler.inverse_transform(cluster_centers_scaled)

    # 2D 시각화를 위해 Annual Income(index 1)과 Spending Score(index 2) 좌표만 추출
    cluster_centers_2d = cluster_centers_original[:, [1, 2]]

    # 각 클러스터별로 점들 그리기 (동적 색상과 라벨 사용)
    for i, profile in enumerate(cluster_profiles):
        cluster_id = profile["cluster_id"]
        mask = data_with_clusters["Cluster"] == cluster_id
        cluster_data = data_with_clusters[mask]

        ax.scatter(
            cluster_data["Annual Income (k$)"],
            cluster_data["Spending Score (1-100)"],
            c=dynamic_colors[i],
            alpha=0.7,
            s=60,
            label=f'클러스터 {cluster_id}: {profile["label"]} ({profile["size"]}명)',
            edgecolors="white",
            linewidth=0.5,
        )

    # 클러스터 중심점 표시
    for i, center in enumerate(cluster_centers_2d):
        ax.scatter(
            center[0],
            center[1],
            c="black",
            marker="x",
            s=300,
            linewidths=4,
            label="클러스터 중심점" if i == 0 else "",
        )

    # 클러스터 영역을 타원으로 표시
    for i, profile in enumerate(cluster_profiles):
        cluster_id = profile["cluster_id"]
        cluster_data = data_with_clusters[data_with_clusters["Cluster"] == cluster_id]

        if len(cluster_data) > 1:
            # 각 클러스터의 평균과 표준편차 계산
            mean_income = cluster_data["Annual Income (k$)"].mean()
            mean_spending = cluster_data["Spending Score (1-100)"].mean()
            std_income = cluster_data["Annual Income (k$)"].std()
            std_spending = cluster_data["Spending Score (1-100)"].std()

            # 타원 생성 (2 표준편차 범위)
            ellipse = Ellipse(
                (mean_income, mean_spending),
                width=4 * std_income,
                height=4 * std_spending,
                fill=False,
                color=dynamic_colors[i],
                linewidth=2,
                linestyle="--",
                alpha=0.8,
            )
            ax.add_patch(ellipse)

    # 한글 폰트 적용된 레이블 설정
    if korean_font_prop:
        ax.set_xlabel(
            "연간 소득 (천 달러)", fontproperties=korean_font_prop, fontsize=14
        )
        ax.set_ylabel("지출 점수 (1-100)", fontproperties=korean_font_prop, fontsize=14)
        ax.set_title(
            f"클러스터링 결과: {optimal_k}개 고객 세분화 완성!",
            fontproperties=korean_font_prop,
            fontsize=16,
            fontweight="bold",
        )

        # 범례에도 한글 폰트 적용
        legend = ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
        for text in legend.get_texts():
            if korean_font_prop:
                text.set_fontproperties(korean_font_prop)
    else:
        ax.set_xlabel("Annual Income (k$)", fontsize=14)
        ax.set_ylabel("Spending Score (1-100)", fontsize=14)
        ax.set_title(
            f"Clustering Results: {optimal_k} Customer Segments Complete!",
            fontsize=16,
            fontweight="bold",
        )
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)

    ax.grid(True, alpha=0.3)
    ax.set_xlim(
        data["Annual Income (k$)"].min() - 5, data["Annual Income (k$)"].max() + 5
    )
    ax.set_ylim(
        data["Spending Score (1-100)"].min() - 5,
        data["Spending Score (1-100)"].max() + 5,
    )

    plt.tight_layout()
    st.pyplot(fig_detailed)

    # 동적 클러스터 해석 및 인사이트 제공
    with st.expander("🔍 동적 클러스터 해석 가이드"):
        st.markdown(interpretation_guide)

    st.success(
        f"✅ 총 {len(data)}명의 고객이 {optimal_k}개 그룹으로 성공적으로 분류되었습니다!"
    )

elif menu == "고객 예측":
    st.header("🔮 새로운 고객 클러스터 예측")

    # Session State에서 클러스터 개수 가져오기
    if "selected_clusters" not in st.session_state:
        st.session_state.selected_clusters = 5  # 기본값

    selected_k = st.session_state.selected_clusters

    # 현재 설정 표시
    st.info(
        f"🎯 현재 선택된 클러스터 개수: **{selected_k}개** (클러스터링 분석 페이지에서 설정됨)"
    )

    # 선택된 클러스터 개수로 클러스터링 수행
    clusters, kmeans, scaler, silhouette_avg = perform_clustering(data, selected_k)

    # 동적 클러스터 분석
    data_with_clusters = data.copy()
    data_with_clusters["Cluster"] = clusters
    cluster_profiles = analyze_cluster_characteristics(data_with_clusters, selected_k)

    st.subheader("고객 정보 입력")

    col1, col2, col3 = st.columns(3)

    with col1:
        input_age = st.number_input("연령", min_value=18, max_value=80, value=30)

    with col2:
        input_income = st.number_input(
            "연간 소득 (천 달러)", min_value=15, max_value=150, value=50
        )

    with col3:
        input_spending = st.number_input(
            "지출 점수 (1-100)", min_value=1, max_value=100, value=50
        )

    if st.button("클러스터 예측하기", type="primary"):
        # 입력 데이터 전처리
        input_data = np.array([[input_age, input_income, input_spending]])
        input_scaled = scaler.transform(input_data)

        # 예측
        predicted_cluster = kmeans.predict(input_scaled)[0]

        # 클러스터 중심점까지의 거리
        distances = kmeans.transform(input_scaled)[0]
        confidence = 1 / (1 + distances[predicted_cluster])

        # 해당 클러스터의 동적 라벨 찾기
        predicted_profile = next(
            (p for p in cluster_profiles if p["cluster_id"] == predicted_cluster), None
        )
        cluster_label = (
            predicted_profile["label"]
            if predicted_profile
            else f"클러스터 {predicted_cluster}"
        )

        # 결과 표시
        st.success(f"🎯 예측된 클러스터: **{predicted_cluster}번 ({cluster_label})**")
        st.info(f"📊 예측 신뢰도: **{confidence:.2%}**")

        # 해당 클러스터의 특성 표시
        cluster_info = data_with_clusters[
            data_with_clusters["Cluster"] == predicted_cluster
        ]

        st.subheader(
            f"클러스터 {predicted_cluster}의 특성 ({selected_k}개 클러스터 기준)"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            avg_age = cluster_info["Age"].mean()
            st.metric("평균 연령", f"{avg_age:.1f}세")

        with col2:
            avg_income = cluster_info["Annual Income (k$)"].mean()
            st.metric("평균 소득", f"${avg_income:.1f}k")

        with col3:
            avg_spending = cluster_info["Spending Score (1-100)"].mean()
            st.metric("평균 지출점수", f"{avg_spending:.1f}")

        # 예측된 클러스터의 상세 특성
        if predicted_profile:
            st.subheader("예측된 고객 세그먼트 특성")
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**고객 유형**: {predicted_profile['customer_type']}")
                st.write(f"**소득 수준**: {predicted_profile['income_level']}")
                st.write(f"**지출 성향**: {predicted_profile['spending_level']}")
                st.write(f"**연령 그룹**: {predicted_profile['age_group']}")

            with col2:
                st.write(f"**클러스터 크기**: {predicted_profile['size']}명")
                st.write(f"**소득 표준편차**: ${predicted_profile['std_income']:.1f}k")
                st.write(f"**지출 표준편차**: {predicted_profile['std_spending']:.1f}")

        # 유사한 고객들 표시
        st.subheader("유사한 고객 프로필")
        similar_customers = cluster_info.sample(min(5, len(cluster_info)))
        st.dataframe(
            similar_customers[
                ["Age", "Annual Income (k$)", "Spending Score (1-100)", "Gender"]
            ]
        )

elif menu == "마케팅 전략":
    st.header("📈 클러스터별 마케팅 전략")

    # Session State에서 클러스터 개수 가져오기
    if "selected_clusters" not in st.session_state:
        st.session_state.selected_clusters = 5  # 기본값

    selected_k = st.session_state.selected_clusters

    # 현재 설정 표시
    st.info(
        f"🎯 현재 선택된 클러스터 개수: **{selected_k}개** (클러스터링 분석 페이지에서 설정됨)"
    )

    # 선택된 클러스터 개수로 클러스터링 수행
    clusters, kmeans, scaler, silhouette_avg = perform_clustering(data, selected_k)
    data_with_clusters = data.copy()
    data_with_clusters["Cluster"] = clusters

    # 동적 클러스터 분석
    cluster_profiles_list = analyze_cluster_characteristics(
        data_with_clusters, selected_k
    )

    # 클러스터별 특성 분석 (기존 형식으로 변환)
    cluster_profiles = {}
    for profile in cluster_profiles_list:
        cluster_id = profile["cluster_id"]
        cluster_data = data_with_clusters[data_with_clusters["Cluster"] == cluster_id]
        cluster_profiles[cluster_id] = {
            "size": profile["size"],
            "avg_age": profile["avg_age"],
            "avg_income": profile["avg_income"],
            "avg_spending": profile["avg_spending"],
            "gender_ratio": cluster_data["Gender"]
            .value_counts(normalize=True)
            .to_dict(),
        }

    st.subheader("클러스터별 마케팅 전략 개요")

    for profile in cluster_profiles_list:
        cluster_id = profile["cluster_id"]
        strategy = get_dynamic_marketing_strategy(
            cluster_id, cluster_profiles[cluster_id], cluster_profiles
        )

        with st.expander(
            f"🎯 클러스터 {cluster_id}: {profile['label']} ({profile['size']}명)"
        ):
            col1, col2 = st.columns(2)

            with col1:
                st.write("**고객 프로필 분석:**")
                st.write(
                    f"- 평균 연령: {profile['avg_age']:.1f}세 ({profile['age_group']})"
                )
                st.write(
                    f"- 평균 소득: ${profile['avg_income']:.1f}k ({profile['income_level']})"
                )
                st.write(
                    f"- 평균 지출점수: {profile['avg_spending']:.1f} ({profile['spending_level']})"
                )
                st.write(f"- 고객 수: {profile['size']}명")
                st.write(f"- 고객 유형: {profile['customer_type']}")

                st.write("**상대적 위치:**")
                st.write(
                    f"- 소득 순위: 상위 {100-float(strategy['percentiles']['income'][:-1]):.0f}%"
                )
                st.write(
                    f"- 지출 순위: 상위 {100-float(strategy['percentiles']['spending'][:-1]):.0f}%"
                )

            with col2:
                st.write("**맞춤 마케팅 전략:**")
                st.write(f"- 세그먼트: {strategy['segment']}")
                st.write(f"- 우선순위: {strategy['priority']}")
                st.write("**전략 세부사항:**")

                # 전략을 줄바꿈으로 구분하여 표시
                strategy_items = strategy["strategy"].split("; ")
                for i, item in enumerate(strategy_items, 1):
                    st.write(f"  {i}. {item}")

                # 특별 권장사항
                if profile["customer_type"] == "프리미엄":
                    st.success(
                        "💎 **최우선 관리 대상**: 매출 기여도가 가장 높은 핵심 고객층"
                    )
                elif profile["customer_type"] == "적극소비":
                    st.warning("⚠️ **주의 필요**: 과소비 경향, 신용 관리 지원 필요")
                elif profile["customer_type"] == "보수적":
                    st.info("🎯 **잠재력 높음**: 추가 소비 유도 가능한 보수적 고소득층")

    # 전체 요약 대시보드
    st.subheader("📊 마케팅 대시보드")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_customers = len(data)
        st.metric("총 고객 수", f"{total_customers:,}명")

    with col2:
        avg_income = data["Annual Income (k$)"].mean()
        st.metric("평균 소득", f"${avg_income:.1f}k")

    with col3:
        avg_spending = data["Spending Score (1-100)"].mean()
        st.metric("평균 지출점수", f"{avg_spending:.1f}")

    with col4:
        high_value_customers = len(
            data_with_clusters[
                (data_with_clusters["Annual Income (k$)"] > 70)
                & (data_with_clusters["Spending Score (1-100)"] > 70)
            ]
        )
        st.metric("프리미엄 고객", f"{high_value_customers}명")

# 푸터
st.markdown("---")
st.markdown(
    """
**개발 정보:** 이 애플리케이션은 K-means 클러스터링을 활용한 고객 세분화 분석 도구입니다.  
**데이터:** Mall Customer Segmentation Dataset  
**기술 스택:** Python, Streamlit, Scikit-learn, Plotly
"""
)
