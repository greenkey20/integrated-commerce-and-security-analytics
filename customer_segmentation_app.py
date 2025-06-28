import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Ellipse
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA  # PCA 추가
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings

warnings.filterwarnings("ignore")

# TensorFlow 및 관련 라이브러리 동적 로딩
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.model_selection import train_test_split

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


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
    min_age = min(p["avg_age"] for p in cluster_profiles)
    max_age = max(p["avg_age"] for p in cluster_profiles)

    # 분류 기준 계산 (사분위수)
    all_incomes = [p["avg_income"] for p in cluster_profiles]
    all_spendings = [p["avg_spending"] for p in cluster_profiles]
    all_ages = [p["avg_age"] for p in cluster_profiles]

    income_quartiles = np.percentile(all_incomes, [25, 50, 75, 90])
    spending_quartiles = np.percentile(all_spendings, [25, 50, 75, 90])
    age_quartiles = np.percentile(all_ages, [25, 50, 75])

    guide_text = f"""
    **현재 {len(cluster_profiles)}개 클러스터 분석 결과 해석:**
    
    **전체 데이터 범위:**
    - 소득 범위: ${min_income:.1f}k ~ ${max_income:.1f}k
    - 지출점수 범위: {min_spending:.1f} ~ {max_spending:.1f}
    - 연령 범위: {min_age:.1f}세 ~ {max_age:.1f}세
    
    **동적 라벨링 분류 기준:**
    
    **소득 수준 분류 기준:**
    - 최고소득: ${income_quartiles[3]:.1f}k 이상 (상위 10%)
    - 고소득: ${income_quartiles[2]:.1f}k ~ ${income_quartiles[3]:.1f}k (상위 25%)
    - 중상소득: ${income_quartiles[1]:.1f}k ~ ${income_quartiles[2]:.1f}k (상위 50%)
    - 중하소득: ${income_quartiles[0]:.1f}k ~ ${income_quartiles[1]:.1f}k (하위 50%)
    - 저소득: ${income_quartiles[0]:.1f}k 미만 (하위 25%)
    
    **지출 성향 분류 기준:**
    - 최고지출: {spending_quartiles[3]:.1f}점 이상 (상위 10%)
    - 고지출: {spending_quartiles[2]:.1f}점 ~ {spending_quartiles[3]:.1f}점 (상위 25%)
    - 중상지출: {spending_quartiles[1]:.1f}점 ~ {spending_quartiles[2]:.1f}점 (상위 50%)
    - 중하지출: {spending_quartiles[0]:.1f}점 ~ {spending_quartiles[1]:.1f}점 (하위 50%)
    - 저지출: {spending_quartiles[0]:.1f}점 미만 (하위 25%)
    
    **연령대 분류 기준:**
    - 청년층: {age_quartiles[0]:.1f}세 미만
    - 청장년층: {age_quartiles[0]:.1f}세 ~ {age_quartiles[1]:.1f}세
    - 중년층: {age_quartiles[1]:.1f}세 ~ {age_quartiles[2]:.1f}세
    - 장년층: {age_quartiles[2]:.1f}세 이상
    
    **고객 유형 정의:**
    - **프리미엄**: 고소득 + 고지출 조합 → 최우선 관리 대상
    - **보수적**: 고소득 + 저지출 조합 → 추가 소비 유도 가능
    - **적극소비**: 저소득 + 고지출 조합 → 신용 관리 지원 필요
    - **절약형**: 저소득 + 저지출 조합 → 가성비 중심 접근
    - **일반**: 위 조합에 해당하지 않는 중간 성향
    
    **각 클러스터의 상세 특성:**
    """

    # 소득 순으로 정렬하여 설명
    sorted_profiles = sorted(
        cluster_profiles, key=lambda x: x["avg_income"], reverse=True
    )

    for profile in sorted_profiles:
        guide_text += f"""
    - **클러스터 {profile['cluster_id']} ({profile['label']})**: 
      평균 소득 ${profile['avg_income']:.1f}k, 지출점수 {profile['avg_spending']:.1f}, 평균 연령 {profile['avg_age']:.1f}세
      고객 수 {profile['size']}명, 고객 유형: {profile['customer_type']}
      ({profile['income_level']} × {profile['spending_level']} × {profile['age_group']} 조합)
        """

    guide_text += f"""
    
    **클러스터링 품질 지표:**
    - 클러스터 간 소득 격차: ${max_income - min_income:.1f}k
    - 클러스터 간 지출성향 차이: {max_spending - min_spending:.1f}점
    - 클러스터 간 연령 차이: {max_age - min_age:.1f}세
    - 가장 큰 클러스터: {max(cluster_profiles, key=lambda x: x['size'])['size']}명
    - 가장 작은 클러스터: {min(cluster_profiles, key=lambda x: x['size'])['size']}명
    - 클러스터 크기 편차: {np.std([p['size'] for p in cluster_profiles]):.1f}명
    """

    return guide_text


# 한글 폰트 설정 실행
korean_font_prop, korean_font_name = setup_korean_font_for_streamlit()

# 2025.6.28(토) 22h45 customer_segmentation_app.py 파일에 추가할 코드
# =============================================================================
# 1단계: 파일 상단의 import 섹션 바로 아래에 추가할 함수들
# =============================================================================
def create_safe_classification_model(input_dim, n_clusters, hidden_units, dropout_rate, learning_rate):
    """
    안전한 분류 모델 생성 함수
    
    Args:
        input_dim (int): 입력 특성의 차원 수 (보통 3: 나이, 소득, 지출점수)
        n_clusters (int): 클러스터 개수
        hidden_units (int): 은닉층 뉴런 수
        dropout_rate (float): 드롭아웃 비율
        learning_rate (float): 학습률
    
    Returns:
        tuple: (모델, 에러메시지) 형태로 반환
               성공 시: (model, None)
               실패 시: (None, error_message)
    """
    try:
        # 핵심 수정사항: 모델 생성 전 항상 세션 초기화
        keras.backend.clear_session()
        
        # 재현 가능한 결과를 위한 시드 설정
        tf.keras.utils.set_random_seed(42)
        
        # 고유한 타임스탬프 생성으로 레이어 이름 충돌 방지
        import time
        timestamp = str(int(time.time() * 1000000))[-8:]  # 마이크로초 단위 8자리
        
        # Sequential 모델 생성 - 각 레이어에 고유 이름 부여
        model = Sequential([
            Dense(
                hidden_units,
                activation="relu",
                input_shape=(input_dim,),
                name=f"input_dense_{timestamp}"
            ),
            Dropout(dropout_rate, name=f"dropout_1_{timestamp}"),
            Dense(
                hidden_units // 2,
                activation="relu",
                name=f"hidden_dense_{timestamp}"
            ),
            Dropout(dropout_rate / 2, name=f"dropout_2_{timestamp}"),
            Dense(
                n_clusters,
                activation="softmax",
                name=f"output_dense_{timestamp}"
            )
        ])

        # 모델 컴파일
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        return model, None
        
    except Exception as e:
        return None, f"모델 생성 실패: {str(e)}"


def train_model_with_progress(model, X_train, y_train, X_test, y_test, epochs, 
                            progress_bar, status_text):
    """
    진행상황을 표시하면서 모델을 훈련하는 함수
    
    Args:
        model: 훈련할 Keras 모델
        X_train, y_train: 훈련 데이터
        X_test, y_test: 검증 데이터
        epochs: 훈련 에포크 수
        progress_bar: Streamlit 진행 표시줄
        status_text: Streamlit 상태 텍스트
    
    Returns:
        tuple: (history, error_message)
    """
    try:
        # 조기 종료 콜백 설정
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True
        )
        
        # Streamlit 전용 콜백 클래스
        class StreamlitProgressCallback(keras.callbacks.Callback):
            def __init__(self, total_epochs, progress_bar, status_text):
                super().__init__()
                self.total_epochs = total_epochs
                self.progress_bar = progress_bar
                self.status_text = status_text

            def on_epoch_end(self, epoch, logs=None):
                # 진행률 업데이트
                progress = (epoch + 1) / self.total_epochs
                self.progress_bar.progress(progress)

                # 상태 텍스트 업데이트
                if logs:
                    self.status_text.text(
                        f"에포크 {epoch + 1}/{self.total_epochs} - "
                        f"손실: {logs.get('loss', 0):.4f}, "
                        f"정확도: {logs.get('accuracy', 0):.4f}, "
                        f"검증 정확도: {logs.get('val_accuracy', 0):.4f}"
                    )

        # 콜백 인스턴스 생성
        progress_callback = StreamlitProgressCallback(epochs, progress_bar, status_text)
        
        # 모델 훈련 실행
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stopping, progress_callback],
            verbose=0  # 콘솔 출력 비활성화
        )
        
        return history, None
        
    except Exception as e:
        return None, f"모델 훈련 실패: {str(e)}"


def display_model_architecture_info(hidden_units, dropout_rate, n_clusters):
    """
    모델 아키텍처 정보를 사용자 친화적으로 표시하는 함수
    """
    st.write("**🏗️ 구성된 신경망 구조:**")
    
    architecture_info = [
        f"입력층: 3개 특성 (나이, 소득, 지출점수)",
        f"은닉층 1: {hidden_units}개 뉴런 + ReLU 활성화 함수",
        f"드롭아웃 1: {dropout_rate*100:.0f}% 뉴런 무작위 비활성화 (과적합 방지)",
        f"은닉층 2: {hidden_units//2}개 뉴런 + ReLU 활성화 함수",
        f"드롭아웃 2: {dropout_rate/2*100:.0f}% 뉴런 무작위 비활성화",
        f"출력층: {n_clusters}개 뉴런 + Softmax (각 클러스터 확률 계산)"
    ]
    
    for i, layer_info in enumerate(architecture_info, 1):
        st.write(f"{i}. {layer_info}")


def evaluate_and_display_results(model, X_test, y_test, history, n_clusters):
    """
    모델 성능을 평가하고 결과를 시각화하는 함수
    
    Returns:
        dict: 평가 결과를 담은 딕셔너리
    """
    try:
        # 모델 성능 평가
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        # 예측 수행
        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)
        
        # 예측 신뢰도 계산
        confidence_scores = np.max(y_pred_probs, axis=1)
        avg_confidence = np.mean(confidence_scores)
        
        # 결과 표시
        st.subheader("📈 모델 성능 분석")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("테스트 정확도", f"{test_accuracy:.3f}")
            st.metric("훈련된 에포크 수", len(history.history["loss"]))
            st.metric("평균 예측 신뢰도", f"{avg_confidence:.3f}")
        
        with col2:
            # 훈련 과정 시각화
            fig = go.Figure()
            
            epochs_range = range(1, len(history.history["accuracy"]) + 1)
            
            fig.add_trace(go.Scatter(
                x=list(epochs_range),
                y=history.history["accuracy"],
                mode="lines",
                name="훈련 정확도",
                line=dict(color="blue")
            ))
            
            fig.add_trace(go.Scatter(
                x=list(epochs_range),
                y=history.history["val_accuracy"],
                mode="lines",
                name="검증 정확도",
                line=dict(color="red")
            ))
            
            fig.update_layout(
                title="모델 훈련 과정",
                xaxis_title="에포크",
                yaxis_title="정확도",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        return {
            "test_accuracy": test_accuracy,
            "predictions": y_pred_classes,
            "probabilities": y_pred_probs,
            "confidence": avg_confidence
        }
        
    except Exception as e:
        st.error(f"모델 평가 중 오류 발생: {str(e)}")
        return None

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

# 사이드바 메뉴 - 모든 메뉴가 한 눈에 보이도록 개선
st.sidebar.title("📋 Navigation")
st.sidebar.markdown("---")

# 메뉴 구조를 시각적으로 더 명확하게 구성
menu_options = [
    "📊 데이터 개요",
    "🔍 탐색적 데이터 분석",
    "🎯 클러스터링 분석",
    "🔬 주성분 분석",
    "🧠 딥러닝 분석",
    "🔮 고객 예측",
    "📈 마케팅 전략",
]

# 라디오 버튼으로 변경하여 모든 메뉴가 한 눈에 보이도록 함
selected_menu = st.sidebar.radio(
    "분석 단계를 선택하세요:", menu_options, index=0  # 기본값을 첫 번째 메뉴로 설정
)

# 선택된 메뉴에서 이모지 제거하여 기존 조건문과 호환
menu = selected_menu.split(" ", 1)[1]  # 이모지와 공백 제거

# 현재 선택된 메뉴 강조 표시
st.sidebar.markdown("---")
st.sidebar.info(f"현재 페이지: **{selected_menu}**")

# 간단한 도움말 추가
with st.sidebar.expander("💡 사용 가이드"):
    st.markdown(
        """
    **분석 순서 권장:**
    1. 📊 데이터 개요 - 기본 정보 파악
    2. 🔍 탐색적 분석 - 패턴 발견
    3. 🎯 클러스터링 - 고객 세분화
    4. 🔬 주성분 분석 - 차원 축소
    5. 🧠 딥러닝 - 고급 모델링
    6. 🔮 고객 예측 - 실제 적용
    7. 📈 마케팅 전략 - 비즈니스 활용
    """
    )

# 프로젝트 정보
st.sidebar.markdown("---")
st.sidebar.markdown("**🛠️ 프로젝트 정보**")
st.sidebar.markdown("고객 세분화 분석 도구")
st.sidebar.markdown("v2.0 - 딥러닝 지원")


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

elif menu == "주성분 분석":
    st.header("🔬 주성분 분석 (Principal Component Analysis)")

    # PCA 이론 설명 섹션
    with st.expander("📚 주성분 분석 이론 가이드", expanded=True):
        st.markdown(
            """
        ### 🤔 왜 주성분 분석이 필요할까요?
        
        고객 데이터를 분석할 때 연령, 소득, 지출점수 등 **여러 변수가 동시에 존재**합니다.
        이런 다차원 데이터에서는 변수들 간의 복잡한 관계를 파악하기 어렵고, 시각화도 제한적입니다.
        
        **차원의 저주**: 변수가 많아질수록 데이터 간 거리가 비슷해져서 패턴 찾기가 어려워집니다.
        마치 3차원 공간에서는 쉽게 구분되던 물체들이 10차원 공간에서는 모두 비슷한 거리에 있는 것처럼 보이는 현상입니다.
        
        ### 🎯 주성분 분석의 핵심 아이디어
        
        PCA는 **"정보 손실을 최소화하면서 차원을 줄이는"** 방법입니다.
        
        **핵심 원리**: 
        - 데이터의 **분산(퍼짐 정도)을 가장 잘 설명하는 새로운 축**을 찾습니다
        - 이 새로운 축들을 **주성분(Principal Component)**이라고 부릅니다
        - 첫 번째 주성분은 데이터 분산을 가장 많이 설명하고, 두 번째는 그 다음으로 많이 설명합니다
        
        **비유로 이해하기**: 
        그림자 놀이를 생각해보세요. 3차원 물체를 벽에 비춘 그림자는 2차원이지만, 
        조명 각도에 따라 물체의 특징을 잘 보여주거나 못 보여줄 수 있습니다.
        PCA는 가장 많은 정보를 담은 "최적의 그림자 각도"를 찾는 것과 같습니다.
        
        ### 📊 주요 개념 설명
        
        **주성분 (Principal Component)**:
        - 원래 변수들의 **선형 결합**으로 만들어진 새로운 변수
        - 서로 **직교(수직)**하며 **독립적**인 관계
        - PC1 > PC2 > PC3... 순으로 설명력이 높습니다
        
        **설명 가능한 분산 비율 (Explained Variance Ratio)**:
        - 각 주성분이 전체 데이터 변동의 몇 %를 설명하는지 나타냄
        - 모든 주성분의 설명 비율을 합하면 100%
        - 처음 몇 개 주성분으로 80-90% 이상 설명되면 차원 축소 효과적
        
        **누적 기여율 (Cumulative Explained Variance)**:
        - 첫 번째부터 n번째 주성분까지의 누적 설명 비율
        - 보통 85-95% 수준에서 적절한 차원 수를 결정
        
        ### 🏢 비즈니스 활용 사례
        
        **고객 분석에서의 PCA 활용**:
        - **고객 특성의 핵심 요인 발견**: 수십 개 변수를 2-3개 핵심 요인으로 압축
        - **시각화 개선**: 3차원 이상 데이터를 2D 평면에서 직관적으로 표현
        - **노이즈 제거**: 중요하지 않은 변동을 걸러내어 핵심 패턴에 집중
        - **저장 공간 절약**: 데이터 압축을 통한 효율적 저장 및 처리
        """
        )

    # 고객 데이터에 PCA 적용
    st.subheader("🔬 고객 데이터 주성분 분석")
    st.write(
        "고객의 연령, 소득, 지출점수 데이터에 PCA를 적용하여 숨겨진 패턴을 발견해보겠습니다."
    )

    # 데이터 준비 및 전처리
    features = data[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
    feature_names = ["연령", "연간소득(k$)", "지출점수"]

    # 데이터 정규화 (PCA는 변수의 스케일에 민감하므로 필수)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    st.write("**1단계: 데이터 전처리 완료**")
    st.info(
        "PCA는 변수의 스케일에 매우 민감하므로, 모든 변수를 평균 0, 표준편차 1로 정규화했습니다."
    )

    # PCA 적용
    # 모든 주성분 계산 (최대 3개 - 원래 변수 개수와 같음)
    pca_full = PCA()
    pca_components = pca_full.fit_transform(scaled_features)

    # 주성분별 설명 가능한 분산 비율
    explained_variance_ratio = pca_full.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    st.write("**2단계: 주성분 분석 결과**")

    # 결과 테이블 생성
    pca_results = pd.DataFrame(
        {
            "주성분": [f"PC{i+1}" for i in range(len(explained_variance_ratio))],
            "설명 분산 비율": explained_variance_ratio,
            "누적 설명 비율": cumulative_variance,
            "설명 분산 비율(%)": explained_variance_ratio * 100,
            "누적 설명 비율(%)": cumulative_variance * 100,
        }
    )

    col1, col2 = st.columns(2)

    with col1:
        st.write("**주성분별 기여도:**")
        # 소수점 3자리까지 표시하되 백분율은 1자리까지
        display_results = pca_results[
            ["주성분", "설명 분산 비율(%)", "누적 설명 비율(%)"]
        ].copy()
        display_results["설명 분산 비율(%)"] = display_results[
            "설명 분산 비율(%)"
        ].round(1)
        display_results["누적 설명 비율(%)"] = display_results[
            "누적 설명 비율(%)"
        ].round(1)
        st.dataframe(display_results, use_container_width=True)

        # 주요 발견사항 요약
        pc1_ratio = explained_variance_ratio[0] * 100
        pc2_ratio = explained_variance_ratio[1] * 100
        pc12_cumulative = cumulative_variance[1] * 100

        st.success(
            f"""
        **📈 주요 발견사항:**
        - PC1이 전체 변동의 {pc1_ratio:.1f}%를 설명
        - PC2가 추가로 {pc2_ratio:.1f}%를 설명
        - PC1+PC2로 {pc12_cumulative:.1f}%의 정보 보존
        """
        )

    with col2:
        # 설명 분산 비율 시각화
        fig = go.Figure()

        # 개별 기여도 막대 그래프
        fig.add_trace(
            go.Bar(
                x=[f"PC{i+1}" for i in range(len(explained_variance_ratio))],
                y=explained_variance_ratio * 100,
                name="개별 기여도",
                marker_color="lightblue",
            )
        )

        # 누적 기여도 선 그래프
        fig.add_trace(
            go.Scatter(
                x=[f"PC{i+1}" for i in range(len(cumulative_variance))],
                y=cumulative_variance * 100,
                mode="lines+markers",
                name="누적 기여도",
                line=dict(color="red", width=3),
                marker=dict(size=8),
                yaxis="y2",
            )
        )

        # 85% 기준선 추가
        fig.add_hline(
            y=85,
            line_dash="dash",
            line_color="gray",
            annotation_text="85% 기준선",
            yref="y2",
        )

        fig.update_layout(
            title="주성분별 설명력 분석",
            xaxis_title="주성분",
            yaxis=dict(title="개별 기여도 (%)", side="left"),
            yaxis2=dict(title="누적 기여도 (%)", side="right", overlaying="y"),
            legend=dict(x=0.7, y=0.95),
        )

        st.plotly_chart(fig, use_container_width=True)

    # 주성분 해석
    st.subheader("🔍 주성분 구성 요소 분석")
    st.write(
        "각 주성분이 원래 변수들(연령, 소득, 지출점수)과 어떤 관계인지 분석해보겠습니다."
    )

    # 주성분 계수 (로딩) 분석
    components_df = pd.DataFrame(
        pca_full.components_.T,
        columns=[f"PC{i+1}" for i in range(pca_full.n_components_)],
        index=feature_names,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.write("**주성분 구성 계수 (Component Loadings):**")
        # 계수 테이블 표시 (색상으로 강도 표현)
        styled_components = components_df.style.background_gradient(
            cmap="RdBu_r", axis=None
        ).format("{:.3f}")
        st.dataframe(styled_components, use_container_width=True)

        # 해석 가이드
        st.info(
            """
        **해석 방법:**
        - 양수(+): 해당 변수가 증가하면 주성분 값도 증가
        - 음수(-): 해당 변수가 증가하면 주성분 값은 감소
        - 절댓값이 클수록: 해당 변수의 영향력이 큼
        """
        )

    with col2:
        # 주성분 구성 히트맵
        fig = px.imshow(
            components_df.T,
            labels=dict(x="원래 변수", y="주성분", color="계수"),
            x=feature_names,
            y=[f"PC{i+1}" for i in range(pca_full.n_components_)],
            color_continuous_scale="RdBu_r",
            aspect="auto",
            title="주성분 구성 히트맵",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # 주성분 해석 생성 (동적)
    st.write("**🎯 주성분 의미 해석:**")

    # PC1 해석
    pc1_coeffs = components_df["PC1"]
    max_pc1_var = pc1_coeffs.abs().idxmax()
    pc1_direction = "높은" if pc1_coeffs[max_pc1_var] > 0 else "낮은"

    # PC2 해석
    pc2_coeffs = components_df["PC2"]
    max_pc2_var = pc2_coeffs.abs().idxmax()
    pc2_direction = "높은" if pc2_coeffs[max_pc2_var] > 0 else "낮은"

    st.write(
        f"""
    - **PC1 ({explained_variance_ratio[0]*100:.1f}% 설명)**: {max_pc1_var} 중심의 축으로, {pc1_direction} {max_pc1_var}를 가진 고객들을 구분합니다.
    - **PC2 ({explained_variance_ratio[1]*100:.1f}% 설명)**: {max_pc2_var} 중심의 축으로, {pc2_direction} {max_pc2_var}를 가진 고객들을 구분합니다.
    """
    )

    # 2D PCA 시각화
    st.subheader("📊 주성분 공간에서의 고객 분포")

    # 2D PCA 결과를 DataFrame에 추가
    pca_2d = PCA(n_components=2)
    pca_2d_result = pca_2d.fit_transform(scaled_features)

    data_pca = data.copy()
    data_pca["PC1"] = pca_2d_result[:, 0]
    data_pca["PC2"] = pca_2d_result[:, 1]

    # 성별로 구분한 PCA 시각화
    fig = px.scatter(
        data_pca,
        x="PC1",
        y="PC2",
        color="Gender",
        title="주성분 공간에서의 고객 분포",
        hover_data=["Age", "Annual Income (k$)", "Spending Score (1-100)"],
        labels={
            "PC1": f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% 설명)",
            "PC2": f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% 설명)",
        },
    )
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # 원래 변수들의 벡터 표시 (선택적)
    show_vectors = st.checkbox("원래 변수들의 방향 벡터 표시", value=False)

    if show_vectors:
        # Biplot 생성
        fig_biplot = go.Figure()

        # 데이터 포인트
        fig_biplot.add_trace(
            go.Scatter(
                x=data_pca["PC1"],
                y=data_pca["PC2"],
                mode="markers",
                marker=dict(size=6, opacity=0.6),
                name="고객 데이터",
                hovertemplate="PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>",
            )
        )

        # 변수 벡터 추가
        scale_factor = 3  # 벡터 크기 조정
        for i, feature in enumerate(feature_names):
            fig_biplot.add_trace(
                go.Scatter(
                    x=[0, pca_2d.components_[0, i] * scale_factor],
                    y=[0, pca_2d.components_[1, i] * scale_factor],
                    mode="lines+markers",
                    line=dict(color="red", width=2),
                    marker=dict(size=8),
                    name=f"{feature} 벡터",
                    showlegend=True,
                )
            )

            # 변수명 라벨 추가
            fig_biplot.add_annotation(
                x=pca_2d.components_[0, i] * scale_factor * 1.1,
                y=pca_2d.components_[1, i] * scale_factor * 1.1,
                text=feature,
                showarrow=False,
                font=dict(size=12, color="red"),
            )

        fig_biplot.update_layout(
            title="PCA Biplot: 고객 분포와 변수 방향",
            xaxis_title=f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% 설명)",
            yaxis_title=f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% 설명)",
            height=600,
        )

        st.plotly_chart(fig_biplot, use_container_width=True)

        st.info(
            """
        **Biplot 해석 가이드:**
        - 빨간 화살표는 원래 변수들이 주성분 공간에서 향하는 방향을 나타냅니다
        - 화살표가 길수록 해당 변수가 주성분에 더 많이 기여합니다
        - 화살표들 사이의 각도가 작을수록 변수들이 비슷한 패턴을 가집니다
        - 데이터 포인트가 화살표 방향에 있을수록 해당 변수 값이 높습니다
        """
        )

    # 클러스터링과 PCA 비교 (선택적)
    st.subheader("🔄 PCA와 클러스터링 결과 비교")
    st.write("PCA 공간에서 클러스터링을 수행하면 어떤 결과가 나올까요?")

    if st.button("PCA 공간에서 클러스터링 수행"):
        # Session State에서 클러스터 개수 가져오기
        n_clusters = st.session_state.get("selected_clusters", 5)

        # PCA 공간에서 클러스터링
        kmeans_pca = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters_pca = kmeans_pca.fit_predict(pca_2d_result)

        # 원래 공간에서 클러스터링
        kmeans_original = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters_original = kmeans_original.fit_predict(scaled_features)

        # 결과 비교
        data_comparison = data_pca.copy()
        data_comparison["PCA_Cluster"] = clusters_pca
        data_comparison["Original_Cluster"] = clusters_original

        col1, col2 = st.columns(2)

        with col1:
            # PCA 공간 클러스터링 결과
            fig1 = px.scatter(
                data_comparison,
                x="PC1",
                y="PC2",
                color="PCA_Cluster",
                title="PCA 공간에서의 클러스터링",
                color_discrete_sequence=px.colors.qualitative.Set1,
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            # 원래 공간 클러스터링을 PCA 공간에 투영
            fig2 = px.scatter(
                data_comparison,
                x="PC1",
                y="PC2",
                color="Original_Cluster",
                title="원래 공간 클러스터링의 PCA 투영",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            st.plotly_chart(fig2, use_container_width=True)

        # 클러스터링 결과 비교 분석
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

        ari_score = adjusted_rand_score(clusters_original, clusters_pca)
        nmi_score = normalized_mutual_info_score(clusters_original, clusters_pca)

        st.write("**클러스터링 결과 유사도 분석:**")
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Adjusted Rand Index (ARI)",
                value=f"{ari_score:.3f}",
                help="1에 가까울수록 두 클러스터링 결과가 유사함",
            )

        with col2:
            st.metric(
                label="Normalized Mutual Information (NMI)",
                value=f"{nmi_score:.3f}",
                help="1에 가까울수록 두 클러스터링이 같은 정보를 공유함",
            )

        if ari_score > 0.7:
            st.success("🎉 PCA 공간과 원래 공간의 클러스터링 결과가 매우 유사합니다!")
        elif ari_score > 0.5:
            st.info("📊 PCA 공간과 원래 공간의 클러스터링 결과가 어느 정도 유사합니다.")
        else:
            st.warning(
                "⚠️ PCA 공간과 원래 공간의 클러스터링 결과가 다릅니다. 차원 축소로 인한 정보 손실이 있을 수 있습니다."
            )

    # 마무리 인사이트
    with st.expander("💡 주성분 분석 인사이트 및 활용 방안"):
        st.markdown(
            f"""
        ### 🎯 이번 분석에서 얻은 주요 인사이트:
        
        **차원 축소 효과:**
        - 3차원 고객 데이터를 2차원으로 축소하면서 {cumulative_variance[1]*100:.1f}%의 정보를 보존했습니다
        - 첫 번째 주성분이 {explained_variance_ratio[0]*100:.1f}%의 고객 특성 변동을 설명합니다
        
        **고객 데이터의 숨겨진 패턴:**
        - 고객들의 주요 구분 축은 '{max_pc1_var}'와 '{max_pc2_var}'입니다
        - 이는 마케팅 전략 수립 시 핵심 고려사항이 될 수 있습니다
        
        ### 🏢 비즈니스 활용 방안:
        
        **마케팅 세분화:**
        - PCA 결과를 바탕으로 고객을 2차원 매트릭스로 구분 가능
        - 각 사분면별로 차별화된 마케팅 전략 수립
        
        **데이터 압축 및 효율성:**
        - 고객 프로필을 2-3개 주성분으로 요약하여 저장 공간 절약
        - 실시간 분석 시 처리 속도 향상
        
        **신규 고객 분류:**
        - 새로운 고객의 주성분 점수를 계산하여 즉시 세그먼트 분류 가능
        - 자동화된 고객 온보딩 프로세스 구축
        
        ### 📈 추가 분석 제안:
        
        **시계열 분석:**
        - 시간에 따른 고객의 주성분 변화 추적
        - 고객 생애주기 모델링
        
        **예측 모델링:**
        - 주성분을 특성으로 활용한 구매 예측 모델
        - 이탈 고객 조기 감지 시스템
        """
        )

    st.success("✅ 주성분 분석을 통해 고객 데이터의 핵심 구조를 파악했습니다!")

elif menu == "딥러닝 분석":
    st.header("🧠 딥러닝을 활용한 고객 분석")
        
    # [수정] 딥러닝 메뉴에 진입할 때마다 Keras 세션을 초기화합니다.
    # 이것이 모델 간 이름 충돌을 막는 가장 확실한 방법입니다.
    if TENSORFLOW_AVAILABLE:
        keras.backend.clear_session()

    # TensorFlow 설치 확인
    if not TENSORFLOW_AVAILABLE:
        st.error(
            """
        🚨 **TensorFlow가 설치되지 않았습니다!**
        
        딥러닝 기능을 사용하려면 TensorFlow를 설치해야 합니다.
        터미널에서 다음 명령어를 실행해주세요:
        
        ```bash
        pip install tensorflow
        ```
        
        설치 후 애플리케이션을 다시 시작해주세요.
        """
        )
        st.stop()

    # 딥러닝 이론 설명 섹션
    with st.expander("🤔 왜 고객 분석에 딥러닝을 사용할까요?", expanded=True):
        st.markdown(
            """
        ### 🎯 비지도 학습에서 지도 학습으로의 전환
        
        지금까지 우리는 **비지도 학습**인 클러스터링을 사용해서 고객을 그룹으로 나누었습니다.
        이제 이 클러스터 결과를 **"정답 라벨"**로 활용하여 **지도 학습** 모델을 만들 수 있습니다.
        
        **왜 이런 전환이 필요할까요?**
        - 클러스터링: 기존 고객을 분석하여 패턴 발견
        - 딥러닝 분류: 새로운 고객이 어떤 그룹에 속할지 **즉시 예측**
        
        **실무적 가치:**
        마치 숙련된 영업사원이 고객을 보자마자 어떤 유형인지 판단하는 것처럼,
        딥러닝 모델은 새로운 고객의 특성을 입력받아 즉시 세그먼트를 예측할 수 있습니다.
        
        ### 🧠 딥러닝이 전통적 방법보다 나은 점
        
        **비선형 패턴 학습:**
        - 전통적 방법: 변수들 간의 **선형적 관계**만 포착
        - 딥러닝: 복잡하고 **비선형적인 관계**까지 학습 가능
        
        **예시로 이해하기:**
        젊은 고소득층이지만 절약형인 고객 vs 중년 중소득층이지만 소비적인 고객
        → 이런 복잡한 조합의 패턴을 딥러닝이 더 잘 포착할 수 있습니다.
        
        **자동 특성 추출:**
        - 전통적 방법: 사람이 직접 중요한 특성을 선택
        - 딥러닝: 데이터에서 **숨겨진 패턴을 자동으로 발견**
        
        ### 🔬 딥러닝 접근법 종류
        
        **1. 분류 모델 (Classification)**
        - 클러스터 결과를 정답으로 사용
        - 새로운 고객 → 어떤 세그먼트에 속할지 예측
        - 오늘 구현할 주요 방법
        
        **2. 오토인코더 (Autoencoder)**
        - 차원 축소의 비선형 버전 (PCA의 업그레이드)
        - 입력 → 압축 → 복원 과정을 통해 핵심 특성 학습
        - 더 복잡한 데이터 구조 포착 가능
        
        **3. 딥 클러스터링 (Deep Clustering)**
        - 클러스터링과 신경망을 동시에 학습
        - 고급 기법으로 추후 확장 가능
        """
        )

    # 데이터 준비 및 클러스터링 수행
    st.subheader("📊 1단계: 기본 데이터 준비")

    # Session State에서 클러스터 개수 가져오기
    n_clusters = st.session_state.get("selected_clusters", 5)
    st.info(f"현재 설정된 클러스터 개수: {n_clusters}개")

    # 특성 준비 및 정규화
    features = data[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # 클러스터링 수행하여 라벨 생성
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_features)

    st.success(f"✅ {len(data)}명의 고객을 {n_clusters}개 클러스터로 분류 완료!")

    # 클러스터 분포 확인
    col1, col2 = st.columns(2)
    with col1:
        cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
        st.write("**클러스터별 고객 수:**")
        for i, count in cluster_counts.items():
            st.write(f"- 클러스터 {i}: {count}명")

    with col2:
        fig = px.pie(
            values=cluster_counts.values,
            names=cluster_counts.index,
            title="클러스터 분포",
        )
        st.plotly_chart(fig, use_container_width=True)

    # 딥러닝 모델 옵션 선택
    st.subheader("🧠 2단계: 딥러닝 모델 선택")

    model_type = st.selectbox(
        "어떤 딥러닝 접근법을 사용하시겠습니까?",
        ["분류 모델 (Classification)", "오토인코더 (Autoencoder)", "두 모델 비교"],
    )

    if model_type in ["분류 모델 (Classification)", "두 모델 비교"]:
        st.subheader("🎯 분류 모델 구축")

        with st.expander("분류 모델이 하는 일", expanded=False):
            st.markdown(
                """
            **분류 모델의 동작 원리:**
            
            1. **입력**: 새로운 고객의 (나이, 소득, 지출점수)
            2. **처리**: 여러 층의 신경망을 통해 패턴 분석
            3. **출력**: 각 클러스터에 속할 확률
            
            **예시:**
            - 입력: (35세, 70k$, 80점)
            - 출력: [클러스터0: 5%, 클러스터1: 85%, 클러스터2: 10%, ...]
            - 결론: 클러스터1에 속할 가능성이 가장 높음
            """
            )

        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_features,
            cluster_labels,
            test_size=0.2,
            random_state=42,
            stratify=cluster_labels,
        )

        st.write(
            f"**데이터 분할 완료:** 훈련용 {len(X_train)}명, 테스트용 {len(X_test)}명"
        )

        # 모델 아키텍처 설정
        st.write("**🏗️ 신경망 아키텍처 설계:**")

        col1, col2 = st.columns(2)
        with col1:
            hidden_units = st.slider(
                "은닉층 뉴런 수", min_value=8, max_value=128, value=64, step=8
            )
            dropout_rate = st.slider(
                "드롭아웃 비율", min_value=0.0, max_value=0.5, value=0.2, step=0.1
            )

        with col2:
            learning_rate = st.selectbox("학습률", [0.001, 0.01, 0.1], index=0)
            epochs = st.slider(
                "학습 에포크", min_value=20, max_value=200, value=100, step=20
            )
            
        # =============================================================================
        # Session State를 활용한 개선된 분류 모델 코드
        # 기존 코드의 if st.button("🚀 분류 모델 훈련 시작", type="primary"): 블록을 교체
        # =============================================================================

        # 세션 상태 초기화 함수
        def initialize_model_session_state():
            """모델 훈련과 관련된 세션 상태를 초기화합니다."""
            if 'model_trained' not in st.session_state:
                st.session_state.model_trained = False
            if 'dl_model' not in st.session_state:
                st.session_state.dl_model = None
            if 'dl_scaler' not in st.session_state:
                st.session_state.dl_scaler = None
            if 'dl_history' not in st.session_state:
                st.session_state.dl_history = None
            if 'dl_evaluation_results' not in st.session_state:
                st.session_state.dl_evaluation_results = None
            if 'dl_X_test' not in st.session_state:
                st.session_state.dl_X_test = None
            if 'dl_y_test' not in st.session_state:
                st.session_state.dl_y_test = None

        # 세션 상태 초기화 실행
        initialize_model_session_state()

        # 모델 훈련 버튼과 상태 표시
        if not st.session_state.model_trained:
            # 아직 모델이 훈련되지 않은 경우
            train_button_clicked = st.button("🚀 분류 모델 훈련 시작", type="primary")
        else:
            # 이미 모델이 훈련된 경우
            st.success("✅ 모델이 이미 훈련되었습니다!")
            if st.button("🔄 모델 다시 훈련하기"):
                # 세션 상태 초기화
                st.session_state.model_trained = False
                st.session_state.dl_model = None
                st.session_state.dl_scaler = None
                st.session_state.dl_history = None
                st.session_state.dl_evaluation_results = None
                st.session_state.dl_X_test = None
                st.session_state.dl_y_test = None
                st.rerun()  # 페이지 새로고침
            
            train_button_clicked = False

        # 모델 훈련 실행
        if train_button_clicked:
            
            # 1단계: 모델 생성
            st.write("**1️⃣ 신경망 모델 생성 중...**")
            
            with st.spinner("모델 아키텍처 구성 중..."):
                model, create_error = create_safe_classification_model(
                    input_dim=3,
                    n_clusters=n_clusters,
                    hidden_units=hidden_units,
                    dropout_rate=dropout_rate,
                    learning_rate=learning_rate
                )
            
            # 모델 생성 실패 시 즉시 중단
            if create_error:
                st.error(f"❌ {create_error}")
                st.info("""
                **💡 문제 해결 방법:**
                1. 페이지를 새로고침(F5)해보세요
                2. 다른 하이퍼파라미터 조합을 시도해보세요
                3. 브라우저 캐시를 지우고 다시 시도해보세요
                """)
                st.stop()
            
            st.success("✅ 신경망 모델 생성 완료!")
            
            # 2단계: 모델 아키텍처 정보 표시
            st.write("**2️⃣ 신경망 구조 확인**")
            display_model_architecture_info(hidden_units, dropout_rate, n_clusters)
            
            # 3단계: 모델 훈련
            st.write("**3️⃣ 신경망 훈련 시작**")
            
            # 진행 상황 표시 요소 준비
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("🏃‍♂️ 신경망 훈련 준비 중...")
            
            # 훈련 실행
            history, train_error = train_model_with_progress(
                model, X_train, y_train, X_test, y_test, epochs,
                progress_bar, status_text
            )
            
            # 훈련 실패 시 처리
            if train_error:
                st.error(f"❌ {train_error}")
                st.info("다른 하이퍼파라미터 설정을 시도해보세요.")
                st.stop()
            
            # 훈련 완료 표시
            status_text.text("✅ 신경망 훈련 완료!")
            progress_bar.progress(1.0)
            st.success("🎉 모델 훈련이 성공적으로 완료되었습니다!")
            
            # 4단계: 모델 평가 및 결과 표시
            st.write("**4️⃣ 모델 성능 평가 및 결과 분석**")
            
            evaluation_results = evaluate_and_display_results(
                model, X_test, y_test, history, n_clusters
            )
            
            if evaluation_results is None:
                st.warning("모델 평가 과정에서 문제가 발생했습니다.")
                st.stop()
            
            # ===== 중요: 훈련 완료 후 세션 상태에 저장 =====
            st.session_state.model_trained = True
            st.session_state.dl_model = model
            st.session_state.dl_scaler = scaler  # 전역 scaler 사용
            st.session_state.dl_history = history
            st.session_state.dl_evaluation_results = evaluation_results
            st.session_state.dl_X_test = X_test
            st.session_state.dl_y_test = y_test
            
            st.info("🔄 모델과 결과가 세션에 저장되었습니다. 이제 다른 기능을 사용할 수 있습니다!")

        # ===== 모델이 훈련된 경우에만 결과 표시 =====
        if st.session_state.model_trained and st.session_state.dl_model is not None:
            
            # 세션에서 데이터 복원
            model = st.session_state.dl_model
            scaler = st.session_state.dl_scaler
            history = st.session_state.dl_history
            evaluation_results = st.session_state.dl_evaluation_results
            X_test = st.session_state.dl_X_test
            y_test = st.session_state.dl_y_test
            
            # 예측 결과 준비
            y_pred_classes = evaluation_results["predictions"]
            
            # 기존의 결과 표시 섹션들을 여기에 다시 표시
            st.write("**📊 훈련된 모델 결과 요약**")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("테스트 정확도", f"{evaluation_results['test_accuracy']:.3f}")
            with col2:
                st.metric("평균 예측 신뢰도", f"{evaluation_results['confidence']:.3f}")
            with col3:
                st.metric("훈련 에포크 수", len(history.history["loss"]))
            
            # 🔍 혼동 행렬 시각화
            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(y_test, y_pred_classes)

            fig = px.imshow(
                cm,
                labels=dict(x="예측 클러스터", y="실제 클러스터", color="고객 수"),
                x=[f"클러스터 {i}" for i in range(n_clusters)],
                y=[f"클러스터 {i}" for i in range(n_clusters)],
                title="혼동 행렬 (Confusion Matrix)",
            )
            st.plotly_chart(fig, use_container_width=True)

            # 🎯 클러스터별 성능 분석
            st.write("**🎯 클러스터별 예측 성능:**")

            from sklearn.metrics import classification_report

            report = classification_report(y_test, y_pred_classes, output_dict=True)

            performance_data = []
            for cluster_id in range(n_clusters):
                if str(cluster_id) in report:
                    cluster_info = report[str(cluster_id)]
                    performance_data.append(
                        {
                            "클러스터": f"클러스터 {cluster_id}",
                            "정밀도": f"{cluster_info['precision']:.3f}",
                            "재현율": f"{cluster_info['recall']:.3f}",
                            "F1-점수": f"{cluster_info['f1-score']:.3f}",
                            "지원 수": cluster_info["support"],
                        }
                    )

            performance_df = pd.DataFrame(performance_data)
            st.dataframe(performance_df, use_container_width=True)

            # 🔮 새로운 고객 예측 기능 (항상 표시)
            st.subheader("🔮 새로운 고객 예측 테스트")

            col1, col2, col3 = st.columns(3)
            with col1:
                test_age = st.number_input(
                    "테스트 고객 연령", min_value=18, max_value=80, value=35
                )
            with col2:
                test_income = st.number_input(
                    "연간 소득 (k$)", min_value=15, max_value=150, value=60
                )
            with col3:
                test_spending = st.number_input(
                    "지출 점수", min_value=1, max_value=100, value=70
                )

            # 중요: 이 버튼은 세션 상태에 관계없이 항상 작동함
            if st.button("🎯 딥러닝으로 클러스터 예측"):
                try:
                    # 새로운 고객 데이터 전처리
                    new_customer = np.array([[test_age, test_income, test_spending]])
                    new_customer_scaled = scaler.transform(new_customer)

                    # 예측 수행
                    prediction_probs = model.predict(new_customer_scaled, verbose=0)[0]
                    predicted_cluster = np.argmax(prediction_probs)

                    st.success(f"🎯 예측된 클러스터: **클러스터 {predicted_cluster}**")

                    # 각 클러스터별 확률 표시
                    st.write("**각 클러스터별 소속 확률:**")
                    prob_data = pd.DataFrame(
                        {
                            "클러스터": [f"클러스터 {i}" for i in range(n_clusters)],
                            "확률": [f"{prob:.1%}" for prob in prediction_probs],
                        }
                    )
                    st.dataframe(prob_data, use_container_width=True)

                    # 확률 시각화
                    fig = px.bar(
                        x=[f"클러스터 {i}" for i in range(n_clusters)],
                        y=prediction_probs,
                        title="클러스터별 소속 확률",
                    )
                    fig.update_layout(xaxis_title="클러스터", yaxis_title="확률")
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"예측 중 오류가 발생했습니다: {str(e)}")
                    st.info("모델을 다시 훈련해보세요.")

        else:
            # 모델이 훈련되지 않은 경우 안내 메시지
            if not st.session_state.model_trained:
                st.info("🔄 먼저 '분류 모델 훈련 시작' 버튼을 클릭하여 모델을 훈련해주세요.")

        # ═══ 여기까지가 완전한 교체 블록입니다 ═══

    if model_type in ["오토인코더 (Autoencoder)", "두 모델 비교"]:
        st.subheader("🔄 오토인코더를 활용한 차원 축소")

        with st.expander("오토인코더가 하는 일", expanded=False):
            st.markdown(
                """
            **오토인코더의 동작 원리:**
            
            1. **인코더**: 입력 데이터를 더 작은 차원으로 압축
            2. **잠재 공간**: 압축된 핵심 정보만 보존
            3. **디코더**: 압축된 정보로부터 원본 재구성
            
            **PCA vs 오토인코더:**
            - PCA: 선형 변환만 가능
            - 오토인코더: 비선형 변환으로 더 복잡한 패턴 포착
            
            **활용 목적:**
            - 데이터 압축
            - 노이즈 제거  
            - 이상치 탐지
            - 더 나은 시각화
            """
            )
        
        # ===== autoencoder 관련 교체 시작 ========================================================================
        # 오토인코더 세션 상태 초기화 함수
        def initialize_autoencoder_session_state():
                """오토인코더와 관련된 세션 상태를 초기화합니다."""
                session_keys = [
                    'autoencoder_trained', 'autoencoder_model', 'encoder_model',
                    'encoded_data', 'reconstruction_error', 'pca_result_ae', 
                    'pca_variance_ratio_ae', 'encoding_dim_value'
                ]
                
                default_values = [False, None, None, None, None, None, None, 2]
                
                for key, default in zip(session_keys, default_values):
                    if key not in st.session_state:
                        st.session_state[key] = default

        # 세션 상태 초기화 실행
        initialize_autoencoder_session_state()

        # 오토인코더 설정 부분
        encoding_dim = st.slider("압축 차원 수", min_value=2, max_value=10, value=2)
        st.session_state.encoding_dim = encoding_dim  # 차원 수도 세션에 저장

        # 오토인코더 훈련 버튼과 상태 관리
        if not st.session_state.autoencoder_trained:
            autoencoder_button_clicked = st.button("🔄 오토인코더 훈련 시작")
        else:
            st.success("✅ 오토인코더가 이미 훈련되었습니다!")
            if st.button("🔄 오토인코더 다시 훈련하기"):
                # 세션 상태 초기화
                st.session_state.autoencoder_trained = False
                st.session_state.autoencoder_model = None
                st.session_state.encoder_model = None
                st.session_state.encoded_data = None
                st.session_state.reconstruction_error = None
                st.session_state.pca_result = None
                st.session_state.pca_variance_ratio = None                
                st.rerun()  # 페이지 새로고침
            
            autoencoder_button_clicked = False

        # 오토인코더 훈련 실행
        if autoencoder_button_clicked:
            # 현재 차원 설정 확인
            current_encoding_dim = st.session_state.encoding_dim_value
            
            st.write(f"**🔄 {current_encoding_dim}차원 오토인코더 훈련 시작**")
            
            try:
                # 모델 구성을 위한 고유 이름 생성
                import time
                timestamp = str(int(time.time() * 1000))[-6:]
                
                # 오토인코더 모델 구성
                input_layer = layers.Input(shape=(3,), name=f"ae_input_{timestamp}")

                # 인코더
                encoded = layers.Dense(8, activation="relu", name=f"ae_encode1_{timestamp}")(input_layer)
                encoded = layers.Dense(
                    current_encoding_dim, 
                    activation="relu", 
                    name=f"ae_encoded_{timestamp}"
                )(encoded)

                # 디코더
                decoded = layers.Dense(8, activation="relu", name=f"ae_decode1_{timestamp}")(encoded)
                decoded = layers.Dense(3, activation="linear", name=f"ae_output_{timestamp}")(decoded)

                # 모델 생성
                autoencoder = keras.Model(input_layer, decoded, name=f"autoencoder_{timestamp}")
                encoder = keras.Model(input_layer, encoded, name=f"encoder_{timestamp}")

                # 컴파일
                autoencoder.compile(optimizer="adam", loss="mse")

                # 진행 상황 표시
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text("🔄 오토인코더 훈련 중...")

                # 훈련 실행
                training_epochs = 100
                history = autoencoder.fit(
                    scaled_features,
                    scaled_features,
                    epochs=training_epochs,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=0,
                    callbacks=[
                        keras.callbacks.LambdaCallback(
                            on_epoch_end=lambda epoch, logs: (
                                progress_bar.progress((epoch + 1) / training_epochs),
                                status_text.text(f"에포크 {epoch + 1}/{training_epochs} - 손실: {logs.get('loss', 0):.4f}")
                            )
                        )
                    ]
                )

                # 결과 계산
                encoded_data = encoder.predict(scaled_features, verbose=0)
                reconstructed = autoencoder.predict(scaled_features, verbose=0)
                reconstruction_error = np.mean(np.square(scaled_features - reconstructed))

                # PCA 비교를 위한 계산
                pca = PCA(n_components=current_encoding_dim)
                pca_result = pca.fit_transform(scaled_features)
                pca_variance_ratio = np.sum(pca.explained_variance_ratio_)
            
                # 훈련 완료 후 세션 상태에 저장
                st.session_state.autoencoder_trained = True
                st.session_state.autoencoder_model = autoencoder
                st.session_state.encoder_model = encoder
                st.session_state.encoded_data = encoded_data
                st.session_state.reconstruction_error = reconstruction_error
                st.session_state.pca_result = pca_result
                st.session_state.pca_variance_ratio = pca_variance_ratio
                
                # 완료 표시
                status_text.text("✅ 오토인코더 훈련 완료!")
                progress_bar.progress(1.0)
                st.success("🎉 오토인코더 훈련이 성공적으로 완료되었습니다!")

            except Exception as e:
                st.error(f"❌ 오토인코더 훈련 중 오류 발생: {str(e)}")
                st.info("다른 설정으로 다시 시도해보세요.")

        # 오토인코더가 훈련된 경우에만 결과 표시
        if st.session_state.autoencoder_trained:
            # 세션에서 데이터 복원
            autoencoder = st.session_state.autoencoder_model
            encoder = st.session_state.encoder_model
            encoded_data = st.session_state.encoded_data
            reconstruction_error = st.session_state.reconstruction_error
            pca_result = st.session_state.pca_result
            pca_variance_ratio = st.session_state.pca_variance_ratio
            encoding_dim = st.session_state.encoding_dim
            
            # 기존의 결과 표시 코드들
            st.metric("재구성 오차 (MSE)", f"{reconstruction_error:.4f}")
            
            # 오토인코더 vs PCA 비교
            st.subheader("🔍 오토인코더 vs PCA 비교")

            # PCA 수행
            pca = PCA(n_components=encoding_dim)
            pca_result = pca.fit_transform(scaled_features)

            col1, col2 = st.columns(2)

            with col1:
                # 오토인코더 결과
                fig1 = px.scatter(
                    x=encoded_data[:, 0],
                    y=encoded_data[:, 1],
                    color=data["Gender"],
                    title=f"오토인코더 결과 ({encoding_dim}D)",
                )
                fig1.update_layout(
                    xaxis_title="인코딩 차원 1", yaxis_title="인코딩 차원 2"
                )
                st.plotly_chart(fig1, use_container_width=True)

            with col2:
                # PCA 결과
                fig2 = px.scatter(
                    x=pca_result[:, 0],
                    y=pca_result[:, 1],
                    color=data["Gender"],
                    title=f"PCA 결과 ({encoding_dim}D)",
                )
                fig2.update_layout(xaxis_title="PC1", yaxis_title="PC2")
                st.plotly_chart(fig2, use_container_width=True)

            # 비교 지표
            pca_variance_ratio = np.sum(pca.explained_variance_ratio_)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("오토인코더 재구성 오차", f"{reconstruction_error:.4f}")
            with col2:
                st.metric("PCA 설명 분산 비율", f"{pca_variance_ratio:.3f}")

            # 클러스터링 성능 비교
            if st.button("🔄 차원 축소 결과로 클러스터링 비교"):

                # 각 차원 축소 결과로 클러스터링
                kmeans_ae = KMeans(n_clusters=n_clusters, random_state=42)
                kmeans_pca = KMeans(n_clusters=n_clusters, random_state=42)

                clusters_ae = kmeans_ae.fit_predict(encoded_data)
                clusters_pca = kmeans_pca.fit_predict(pca_result)

                # 원본 클러스터링과 비교
                from sklearn.metrics import adjusted_rand_score

                ari_ae = adjusted_rand_score(cluster_labels, clusters_ae)
                ari_pca = adjusted_rand_score(cluster_labels, clusters_pca)

                st.write("**원본 클러스터링과의 유사도:**")
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("오토인코더 ARI", f"{ari_ae:.3f}")

                with col2:
                    st.metric("PCA ARI", f"{ari_pca:.3f}")

                if ari_ae > ari_pca:
                    st.success("🎉 오토인코더가 더 나은 클러스터 보존 성능을 보입니다!")
                elif ari_pca > ari_ae:
                    st.info("📊 PCA가 더 나은 클러스터 보존 성능을 보입니다.")
                else:
                    st.write("🤝 두 방법의 성능이 비슷합니다.") 
        
        # ===== autoencoder 관련 교체 끝 ====================================================
        
    # =============================================================================
    # "두 모델 비교" 기능 완성을 위한 추가 코드
    # 딥러닝 분석 섹션의 마지막 부분 (딥러닝 활용 가이드 expander 위)에 추가하세요
    # =============================================================================

    # "두 모델 비교" 전용 섹션
    if model_type == "두 모델 비교":
        st.markdown("---")  # 구분선
        st.subheader("🔀 분류 모델 vs 오토인코더 종합 비교")
        
        # 두 모델의 훈련 상태 확인
        classification_trained = st.session_state.get('model_trained', False)
        autoencoder_trained = st.session_state.get('autoencoder_trained', False)
        
        if not classification_trained and not autoencoder_trained:
            st.warning("🔄 **두 모델 모두 훈련이 필요합니다.**")
            st.info("위의 '분류 모델 훈련 시작'과 '오토인코더 훈련 시작' 버튼을 각각 클릭하여 두 모델을 모두 훈련해주세요.")
            
        elif not classification_trained:
            st.warning("🔄 **분류 모델 훈련이 필요합니다.**")
            st.info("위의 '분류 모델 훈련 시작' 버튼을 클릭하여 훈련을 완료해주세요.")
            
        elif not autoencoder_trained:
            st.warning("🔄 **오토인코더 훈련이 필요합니다.**")
            st.info("위의 '오토인코더 훈련 시작' 버튼을 클릭하여 훈련을 완료해주세요.")
            
        else:
            # 두 모델이 모두 훈련된 경우 → 비교 분석 실행
            st.success("✅ 두 모델이 모두 훈련 완료되었습니다! 종합 비교를 시작합니다.")
            
            # 세션에서 모델 결과 불러오기
            classification_results = st.session_state.get('dl_evaluation_results', {})
            autoencoder_results = {
                'reconstruction_error': st.session_state.get('reconstruction_error', 0),
                'encoded_data': st.session_state.get('encoded_data', None),
                'autoencoder_model': st.session_state.get('autoencoder_model', None)
            }
            
            # =============================================================================
            # 1. 성능 지표 비교 대시보드
            # =============================================================================
            st.subheader("📊 1. 성능 지표 종합 비교")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="분류 모델 정확도",
                    value=f"{classification_results.get('test_accuracy', 0):.3f}",
                    help="새로운 고객을 올바른 클러스터로 분류하는 정확도"
                )
                
            with col2:
                st.metric(
                    label="분류 모델 신뢰도",
                    value=f"{classification_results.get('confidence', 0):.3f}",
                    help="예측에 대한 평균 신뢰도 (확률)"
                )
                
            with col3:
                st.metric(
                    label="오토인코더 재구성 오차",
                    value=f"{autoencoder_results.get('reconstruction_error', 0):.4f}",
                    help="원본 데이터를 얼마나 정확히 재구성하는지 (낮을수록 좋음)"
                )
                
            with col4:
                # 종합 성능 점수 계산 (0-100 스케일)
                classification_score = classification_results.get('test_accuracy', 0) * 100
                # 재구성 오차를 0-100 스케일로 변환 (낮을수록 좋으므로 역변환)
                recon_error = autoencoder_results.get('reconstruction_error', 1)
                autoencoder_score = max(0, (1 - min(recon_error, 1)) * 100)
                
                overall_score = (classification_score + autoencoder_score) / 2
                st.metric(
                    label="종합 성능 점수",
                    value=f"{overall_score:.1f}/100",
                    help="두 모델의 성능을 종합한 점수"
                )
            
            # =============================================================================
            # 2. 시각적 성능 비교
            # =============================================================================
            st.subheader("📈 2. 시각적 성능 비교")
            
            # 성능 비교 레이더 차트
            fig_radar = go.Figure()
            
            categories = ['예측 정확도', '신뢰도', '데이터 재구성', '해석 가능성', '실시간 처리']
            
            # 분류 모델 점수 (0-100 스케일)
            classification_scores = [
                classification_results.get('test_accuracy', 0) * 100,  # 예측 정확도
                classification_results.get('confidence', 0) * 100,     # 신뢰도
                60,  # 데이터 재구성 (분류모델은 재구성하지 않으므로 중간값)
                80,  # 해석 가능성 (분류 결과는 직관적)
                90   # 실시간 처리 (빠른 예측)
            ]
            
            # 오토인코더 점수 (0-100 스케일)  
            autoencoder_scores = [
                70,  # 예측 정확도 (직접적 분류는 아니지만 패턴 학습)
                65,  # 신뢰도 (재구성 품질 기반)
                max(0, (1 - min(autoencoder_results.get('reconstruction_error', 1), 1)) * 100),  # 데이터 재구성
                60,  # 해석 가능성 (잠재 공간은 해석이 어려움)
                75   # 실시간 처리 (인코딩 과정 필요)
            ]
            
            fig_radar.add_trace(go.Scatterpolar(
                r=classification_scores,
                theta=categories,
                fill='toself',
                name='분류 모델',
                line_color='blue'
            ))
            
            fig_radar.add_trace(go.Scatterpolar(
                r=autoencoder_scores,
                theta=categories,
                fill='toself',
                name='오토인코더',
                line_color='red'
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=True,
                title="모델 성능 비교 (레이더 차트)"
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # =============================================================================
            # 3. 상세 비교 분석표
            # =============================================================================
            st.subheader("📋 3. 상세 비교 분석")
            
            comparison_data = {
                "비교 항목": [
                    "주요 목적", "학습 방식", "출력 결과", "새 고객 예측", 
                    "데이터 압축", "이상치 감지", "해석 가능성", "훈련 시간", 
                    "메모리 사용량", "실용성"
                ],
                "분류 모델": [
                    "고객 세그먼트 예측", "지도 학습", "클러스터 확률", "즉시 가능",
                    "불가능", "제한적", "높음", "보통", "적음", "매우 높음"
                ],
                "오토인코더": [
                    "데이터 압축 및 재구성", "비지도 학습", "압축된 특성", "간접적",
                    "가능", "우수", "낮음", "길음", "많음", "높음"
                ],
                "우수한 모델": [
                    "분류 모델", "각각 장점", "분류 모델", "분류 모델",
                    "오토인코더", "오토인코더", "분류 모델", "분류 모델",
                    "분류 모델", "분류 모델"
                ]
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # 색상 적용을 위한 스타일링 함수
            def highlight_winner(val):
                if val == "분류 모델":
                    return 'background-color: lightblue'
                elif val == "오토인코더":
                    return 'background-color: lightcoral'
                elif val == "각각 장점":
                    return 'background-color: lightgreen'
                else:
                    return ''
            
            # 우수한 모델 컬럼에만 색상 적용
            styled_df = comparison_df.style.applymap(
                highlight_winner, subset=['우수한 모델']
            )
            
            st.dataframe(styled_df, use_container_width=True)
            
            # =============================================================================
            # 4. 사용 사례별 추천
            # =============================================================================
            st.subheader("🎯 4. 비즈니스 상황별 모델 추천")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🔵 분류 모델 추천 상황")
                classification_cases = [
                    "**실시간 고객 분류**가 필요한 경우",
                    "**새로운 고객 예측**이 주 목적인 경우", 
                    "**해석 가능한 결과**가 중요한 경우",
                    "**빠른 응답 시간**이 필요한 경우",
                    "**제한된 컴퓨팅 자원** 환경",
                    "**명확한 비즈니스 액션**이 필요한 경우"
                ]
                for case in classification_cases:
                    st.write(f"✅ {case}")
                    
            with col2:
                st.markdown("### 🔴 오토인코더 추천 상황")
                autoencoder_cases = [
                    "**데이터 압축**이 주 목적인 경우",
                    "**이상치 탐지**가 중요한 경우",
                    "**데이터 시각화** 개선이 필요한 경우",
                    "**노이즈 제거**가 필요한 경우",
                    "**숨겨진 패턴 발견**이 목적인 경우",
                    "**차원 축소**를 통한 저장 공간 절약"
                ]
                for case in autoencoder_cases:
                    st.write(f"✅ {case}")
            
            # =============================================================================
            # 5. 하이브리드 접근법 제안
            # =============================================================================
            st.subheader("🔄 5. 하이브리드 접근법 제안")
            
            st.info("""
            **💡 두 모델을 함께 사용하는 최적 전략:**
            
            1. **1단계**: 오토인코더로 고객 데이터의 핵심 특성 추출 및 이상치 제거
            2. **2단계**: 정제된 데이터로 분류 모델 훈련하여 예측 정확도 향상
            3. **3단계**: 분류 모델로 실시간 고객 분류, 오토인코더로 주기적 데이터 분석
            4. **4단계**: 두 모델의 결과를 종합하여 더 신뢰성 있는 최종 의사결정
            """)
            
            # =============================================================================
            # 6. 실행 가능한 액션 플랜
            # =============================================================================
            st.subheader("📝 6. 실행 가능한 액션 플랜")
            
            # 현재 성능 기반 추천
            if classification_results.get('test_accuracy', 0) > 0.8:
                primary_recommendation = "분류 모델"
                reason = "높은 예측 정확도"
            elif autoencoder_results.get('reconstruction_error', 1) < 0.1:
                primary_recommendation = "오토인코더" 
                reason = "우수한 데이터 재구성 성능"
            else:
                primary_recommendation = "하이브리드 접근"
                reason = "두 모델 모두 개선 여지"
                
            st.success(f"""
            **🎯 현재 데이터에 대한 최종 추천: {primary_recommendation}**
            
            **추천 이유**: {reason}
            
            **단계별 실행 계획**:
            1. **즉시 실행**: 성능이 우수한 모델을 프로덕션 환경에 우선 적용
            2. **1주일 내**: A/B 테스트를 통한 실제 성능 검증
            3. **1개월 내**: 사용자 피드백 수집 및 모델 성능 모니터링
            4. **3개월 내**: 추가 데이터로 모델 재훈련 및 하이브리드 접근법 도입
            """)
            
            # =============================================================================
            # 7. 상세 기술 분석 (선택적)
            # =============================================================================
            with st.expander("🔬 상세 기술 분석 (고급 사용자용)"):
                st.markdown("""
                ### 모델 아키텍처 비교
                
                **분류 모델 아키텍처:**
                - 입력층: 3개 특성 (연령, 소득, 지출점수)
                - 은닉층: 다층 퍼셉트론 (MLP) 구조
                - 출력층: Softmax로 각 클러스터 확률 계산
                - 손실함수: Sparse Categorical Crossentropy
                - 최적화: Adam Optimizer
                
                **오토인코더 아키텍처:**
                - 인코더: 3차원 → 압축차원으로 비선형 변환
                - 잠재공간: 압축된 고차원 특성 표현
                - 디코더: 압축차원 → 3차원으로 재구성
                - 손실함수: Mean Squared Error (MSE)
                - 최적화: Adam Optimizer
                
                ### 계산 복잡도 분석
                
                **분류 모델:**
                - 훈련 시간 복잡도: O(n × epochs × layers)
                - 예측 시간 복잡도: O(layers) - 매우 빠름
                - 메모리 복잡도: O(parameters) - 상대적으로 적음
                
                **오토인코더:**
                - 훈련 시간 복잡도: O(n × epochs × (encoder + decoder))
                - 예측 시간 복잡도: O(encoder + decoder) - 보통
                - 메모리 복잡도: O(2 × parameters) - 상대적으로 많음
                
                ### 수치적 안정성
                
                **분류 모델:**
                - Softmax 함수의 수치적 안정성 우수
                - Dropout을 통한 과적합 방지
                - 그래디언트 소실 문제 적음
                
                **오토인코더:**
                - 깊은 네트워크로 인한 그래디언트 소실 가능성
                - 재구성 손실의 스케일링 필요
                - 잠재공간의 규제 필요할 수 있음
                """)

    # 이 코드는 기존 딥러닝 분석 섹션의 마지막 부분에 추가하세요
    # 구체적으로는 '딥러닝 활용 가이드' expander 바로 위에 삽입하면 됩니다

    # 딥러닝 활용 가이드
    with st.expander("💡 딥러닝 결과 활용 가이드"):
        st.markdown(
            """
        ### 🎯 언제 딥러닝을 사용해야 할까요?
        
        **딥러닝이 유리한 경우:**
        - 대량의 고객 데이터 (수천 명 이상)
        - 복잡한 고객 행동 패턴
        - 실시간 고객 분류가 필요한 경우
        - 높은 예측 정확도가 중요한 비즈니스
        
        **전통적 방법이 나은 경우:**
        - 소규모 데이터 (수백 명 이하)
        - 해석 가능성이 중요한 경우
        - 빠른 프로토타이핑이 필요한 경우
        - 컴퓨팅 자원이 제한적인 환경
        
        ### 🏢 비즈니스 활용 방안
        
        **실시간 고객 분류 시스템:**
        - 온라인 쇼핑몰에서 즉시 고객 세그먼트 파악
        - 맞춤형 상품 추천 시스템 구축
        - 개인화된 마케팅 메시지 자동 생성
        
        **고객 여정 예측:**
        - 고객의 다음 행동 패턴 예측
        - 이탈 위험 고객 조기 감지
        - 생애 가치 예측 모델링
        
        ### 📈 모델 개선 방향
        
        **더 많은 특성 추가:**
        - 구매 이력, 웹사이트 행동, 소셜미디어 활동
        - 시계열 특성 (구매 주기, 계절성)
        - 외부 데이터 (경제 지표, 트렌드)
        
        **앙상블 방법:**
        - 여러 모델의 예측을 결합
        - 더 안정적이고 정확한 예측
        - 리스크 분산 효과
        """
        )

    st.success("🧠 딥러닝을 통한 고객 분석이 완료되었습니다!")

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
