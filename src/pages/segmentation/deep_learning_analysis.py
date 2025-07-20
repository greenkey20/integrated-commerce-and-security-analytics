"""
딥러닝 분석 페이지

기존 customer_segmentation_app.py의 "딥러닝 분석" 메뉴 내용을 모듈화
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from core.segmentation.data_processing import CustomerDataProcessor
from core.segmentation.clustering import ClusterAnalyzer
from core.segmentation.models import DeepLearningModels, TENSORFLOW_AVAILABLE

# TensorFlow 관련 import (이미 deep_learning_models에서 처리됨)
if TENSORFLOW_AVAILABLE:
    from tensorflow import keras


def show_deep_learning_analysis_page():
    """딥러닝 분석 페이지를 표시하는 함수"""
    st.header("🧠 딥러닝을 활용한 고객 분석")
    
    # 딥러닝 메뉴에 진입할 때마다 Keras 세션을 초기화
    if TENSORFLOW_AVAILABLE:
        keras.backend.clear_session()

    # TensorFlow 설치 확인
    if not TENSORFLOW_AVAILABLE:
        st.error("""
        🚨 **TensorFlow가 설치되지 않았습니다!**
        
        딥러닝 기능을 사용하려면 TensorFlow를 설치해야 합니다.
        터미널에서 다음 명령어를 실행해주세요:
        
        ```bash
        pip install tensorflow
        ```
        
        설치 후 애플리케이션을 다시 시작해주세요.
        """)
        st.stop()

    # 딥러닝 이론 설명 섹션
    with st.expander("🤔 왜 고객 분석에 딥러닝을 사용할까요?", expanded=True):
        st.markdown("""
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
        """)

    # 데이터 준비 및 클러스터링 수행
    st.subheader("📊 1단계: 기본 데이터 준비")

    # 데이터 로드
    data_processor = CustomerDataProcessor()
    data = data_processor.load_data()
    
    # Session State에서 클러스터 개수 가져오기
    n_clusters = st.session_state.get("selected_clusters", 5)
    st.info(f"현재 설정된 클러스터 개수: {n_clusters}개")

    # 특성 준비 및 정규화
    features = data[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # 클러스터링 수행하여 라벨 생성
    cluster_analyzer = ClusterAnalyzer()
    clusters, kmeans, _, silhouette_avg = cluster_analyzer.perform_clustering(data, n_clusters)

    st.success(f"✅ {len(data)}명의 고객을 {n_clusters}개 클러스터로 분류 완료!")

    # 클러스터 분포 확인
    col1, col2 = st.columns(2)
    with col1:
        cluster_counts = pd.Series(clusters).value_counts().sort_index()
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

    # 딥러닝 모델 인스턴스 생성
    dl_models = DeepLearningModels()

    if model_type in ["분류 모델 (Classification)", "두 모델 비교"]:
        st.subheader("🎯 분류 모델 구축")

        with st.expander("분류 모델이 하는 일", expanded=False):
            st.markdown("""
            **분류 모델의 동작 원리:**
            
            1. **입력**: 새로운 고객의 (나이, 소득, 지출점수)
            2. **처리**: 여러 층의 신경망을 통해 패턴 분석
            3. **출력**: 각 클러스터에 속할 확률
            
            **예시:**
            - 입력: (35세, 70k$, 80점)
            - 출력: [클러스터0: 5%, 클러스터1: 85%, 클러스터2: 10%, ...]
            - 결론: 클러스터1에 속할 가능성이 가장 높음
            """)

        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_features,
            clusters,
            test_size=0.2,
            random_state=42,
            stratify=clusters,
        )

        st.write(f"**데이터 분할 완료:** 훈련용 {len(X_train)}명, 테스트용 {len(X_test)}명")

        # 모델 아키텍처 설정
        st.write("**🏗️ 신경망 아키텍처 설계:**")

        col1, col2 = st.columns(2)
        with col1:
            hidden_units = st.slider("은닉층 뉴런 수", min_value=8, max_value=128, value=64, step=8)
            dropout_rate = st.slider("드롭아웃 비율", min_value=0.0, max_value=0.5, value=0.2, step=0.1)

        with col2:
            learning_rate = st.selectbox("학습률", [0.001, 0.01, 0.1], index=0)
            epochs = st.slider("학습 에포크", min_value=20, max_value=200, value=100, step=20)

        # 세션 상태 초기화
        _initialize_model_session_state()

        # 모델 훈련 버튼과 상태 표시
        if not st.session_state.model_trained:
            train_button_clicked = st.button("🚀 분류 모델 훈련 시작", type="primary")
        else:
            st.success("✅ 모델이 이미 훈련되었습니다!")
            if st.button("🔄 모델 다시 훈련하기"):
                _reset_model_session_state()
                st.rerun()
            train_button_clicked = False

        # 모델 훈련 실행
        if train_button_clicked:
            _train_classification_model(
                dl_models, X_train, y_train, X_test, y_test, 
                n_clusters, hidden_units, dropout_rate, learning_rate, epochs
            )

        # 모델이 훈련된 경우에만 결과 표시
        if st.session_state.model_trained and st.session_state.dl_model is not None:
            _display_classification_results(n_clusters)

    if model_type in ["오토인코더 (Autoencoder)", "두 모델 비교"]:
        st.subheader("🔄 오토인코더를 활용한 차원 축소")

        with st.expander("오토인코더가 하는 일", expanded=False):
            st.markdown("""
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
            """)

        # 오토인코더 세션 상태 초기화
        _initialize_autoencoder_session_state()

        # 오토인코더 설정
        encoding_dim = st.slider("압축 차원 수", min_value=2, max_value=10, value=2)
        st.session_state.encoding_dim = encoding_dim

        # 오토인코더 훈련 버튼과 상태 관리
        if not st.session_state.autoencoder_trained:
            autoencoder_button_clicked = st.button("🔄 오토인코더 훈련 시작")
        else:
            st.success("✅ 오토인코더가 이미 훈련되었습니다!")
            if st.button("🔄 오토인코더 다시 훈련하기"):
                _reset_autoencoder_session_state()
                st.rerun()
            autoencoder_button_clicked = False

        # 오토인코더 훈련 실행
        if autoencoder_button_clicked:
            _train_autoencoder_model(dl_models, scaled_features, encoding_dim)

        # 오토인코더가 훈련된 경우에만 결과 표시
        if st.session_state.autoencoder_trained:
            _display_autoencoder_results(encoding_dim)

    # "두 모델 비교" 전용 섹션
    if model_type == "두 모델 비교":
        _show_model_comparison()

    # 딥러닝 활용 가이드
    with st.expander("💡 딥러닝 결과 활용 가이드"):
        st.markdown("""
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
        """)

    st.success("🧠 딥러닝을 통한 고객 분석이 완료되었습니다!")


def _initialize_model_session_state():
    """분류 모델 훈련과 관련된 세션 상태를 초기화"""
    session_keys = [
        'model_trained', 'dl_model', 'dl_scaler', 'dl_history',
        'dl_evaluation_results', 'dl_X_test', 'dl_y_test'
    ]
    default_values = [False, None, None, None, None, None, None]
    
    for key, default in zip(session_keys, default_values):
        if key not in st.session_state:
            st.session_state[key] = default


def _reset_model_session_state():
    """분류 모델 세션 상태 초기화"""
    keys_to_reset = [
        'model_trained', 'dl_model', 'dl_scaler', 'dl_history',
        'dl_evaluation_results', 'dl_X_test', 'dl_y_test'
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]


def _initialize_autoencoder_session_state():
    """오토인코더와 관련된 세션 상태를 초기화"""
    session_keys = [
        'autoencoder_trained', 'autoencoder_model', 'encoder_model',
        'encoded_data', 'reconstruction_error', 'pca_result_ae', 
        'pca_variance_ratio_ae', 'encoding_dim_value'
    ]
    default_values = [False, None, None, None, None, None, None, 2]
    
    for key, default in zip(session_keys, default_values):
        if key not in st.session_state:
            st.session_state[key] = default


def _reset_autoencoder_session_state():
    """오토인코더 세션 상태 초기화"""
    keys_to_reset = [
        'autoencoder_trained', 'autoencoder_model', 'encoder_model',
        'encoded_data', 'reconstruction_error', 'pca_result_ae', 
        'pca_variance_ratio_ae'
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]


def _train_classification_model(dl_models, X_train, y_train, X_test, y_test, 
                               n_clusters, hidden_units, dropout_rate, learning_rate, epochs):
    """분류 모델 훈련 실행"""
    # 1단계: 모델 생성
    st.write("**1️⃣ 신경망 모델 생성 중...**")
    
    with st.spinner("모델 아키텍처 구성 중..."):
        model, create_error = dl_models.create_safe_classification_model(
            input_dim=3,
            n_clusters=n_clusters,
            hidden_units=hidden_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate
        )
    
    if create_error:
        st.error(f"❌ {create_error}")
        st.info("페이지를 새로고침하거나 다른 하이퍼파라미터를 시도해보세요.")
        st.stop()
    
    st.success("✅ 신경망 모델 생성 완료!")
    
    # 2단계: 모델 아키텍처 정보 표시
    st.write("**2️⃣ 신경망 구조 확인**")
    dl_models.display_model_architecture_info(hidden_units, dropout_rate, n_clusters)
    
    # 3단계: 모델 훈련
    st.write("**3️⃣ 신경망 훈련 시작**")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("🏃‍♂️ 신경망 훈련 준비 중...")
    
    history, train_error = dl_models.train_model_with_progress(
        model, X_train, y_train, X_test, y_test, epochs, progress_bar, status_text
    )
    
    if train_error:
        st.error(f"❌ {train_error}")
        st.info("다른 하이퍼파라미터 설정을 시도해보세요.")
        st.stop()
    
    status_text.text("✅ 신경망 훈련 완료!")
    progress_bar.progress(1.0)
    st.success("🎉 모델 훈련이 성공적으로 완료되었습니다!")
    
    # 4단계: 모델 평가
    st.write("**4️⃣ 모델 성능 평가 및 결과 분석**")
    
    evaluation_results = dl_models.evaluate_and_display_results(
        model, X_test, y_test, history, n_clusters
    )
    
    if evaluation_results is None:
        st.warning("모델 평가 과정에서 문제가 발생했습니다.")
        st.stop()
    
    # 세션 상태에 저장
    st.session_state.model_trained = True
    st.session_state.dl_model = model
    st.session_state.dl_scaler = StandardScaler().fit(X_train)  # 새로운 scaler 생성
    st.session_state.dl_history = history
    st.session_state.dl_evaluation_results = evaluation_results
    st.session_state.dl_X_test = X_test
    st.session_state.dl_y_test = y_test
    
    st.info("🔄 모델과 결과가 세션에 저장되었습니다.")


def _display_classification_results(n_clusters):
    """분류 모델 결과 표시"""
    # 세션에서 데이터 복원
    model = st.session_state.dl_model
    scaler = st.session_state.dl_scaler
    history = st.session_state.dl_history
    evaluation_results = st.session_state.dl_evaluation_results
    X_test = st.session_state.dl_X_test
    y_test = st.session_state.dl_y_test
    
    y_pred_classes = evaluation_results["predictions"]
    
    st.write("**📊 훈련된 모델 결과 요약**")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("테스트 정확도", f"{evaluation_results['test_accuracy']:.3f}")
    with col2:
        st.metric("평균 예측 신뢰도", f"{evaluation_results['confidence']:.3f}")
    with col3:
        st.metric("훈련 에포크 수", len(history.history["loss"]))
    
    # 혼동 행렬 시각화
    cm = confusion_matrix(y_test, y_pred_classes)
    fig = px.imshow(
        cm,
        labels=dict(x="예측 클러스터", y="실제 클러스터", color="고객 수"),
        x=[f"클러스터 {i}" for i in range(n_clusters)],
        y=[f"클러스터 {i}" for i in range(n_clusters)],
        title="혼동 행렬 (Confusion Matrix)",
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 클러스터별 성능 분석
    st.write("**🎯 클러스터별 예측 성능:**")
    
    report = classification_report(y_test, y_pred_classes, output_dict=True)
    performance_data = []
    for cluster_id in range(n_clusters):
        if str(cluster_id) in report:
            cluster_info = report[str(cluster_id)]
            performance_data.append({
                "클러스터": f"클러스터 {cluster_id}",
                "정밀도": f"{cluster_info['precision']:.3f}",
                "재현율": f"{cluster_info['recall']:.3f}",
                "F1-점수": f"{cluster_info['f1-score']:.3f}",
                "지원 수": cluster_info["support"],
            })
    
    performance_df = pd.DataFrame(performance_data)
    st.dataframe(performance_df, use_container_width=True)
    
    # 새로운 고객 예측 기능
    st.subheader("🔮 새로운 고객 예측 테스트")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        test_age = st.number_input("테스트 고객 연령", min_value=18, max_value=80, value=35)
    with col2:
        test_income = st.number_input("연간 소득 (k$)", min_value=15, max_value=150, value=60)
    with col3:
        test_spending = st.number_input("지출 점수", min_value=1, max_value=100, value=70)
    
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
            prob_data = pd.DataFrame({
                "클러스터": [f"클러스터 {i}" for i in range(n_clusters)],
                "확률": [f"{prob:.1%}" for prob in prediction_probs],
            })
            st.dataframe(prob_data, use_container_width=True)
            
            # 확률 시각화
            fig = px.bar(
                x=[f"클러스터 {i}" for i in range(n_clusters)],
                y=prediction_probs,
                title="클러스터별 소속 확률",
                labels={"x": "클러스터", "y": "확률"}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"예측 중 오류가 발생했습니다: {str(e)}")
            st.info("모델을 다시 훈련해보세요.")


def _train_autoencoder_model(dl_models, scaled_features, encoding_dim):
    """오토인코더 모델 훈련"""
    st.write(f"**🔄 {encoding_dim}차원 오토인코더 훈련 시작**")
    
    try:
        # 오토인코더 생성
        autoencoder, encoder, create_error = dl_models.create_autoencoder(
            input_dim=3, encoding_dim=encoding_dim
        )
        
        if create_error:
            st.error(f"❌ {create_error}")
            return
        
        # 진행 상황 표시
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("🔄 오토인코더 훈련 중...")
        
        # 훈련 실행
        history, train_error = dl_models.train_autoencoder(
            autoencoder, scaled_features, progress_bar=progress_bar, status_text=status_text
        )
        
        if train_error:
            st.error(f"❌ {train_error}")
            return
        
        # 결과 계산
        encoded_data = encoder.predict(scaled_features, verbose=0)
        reconstructed = autoencoder.predict(scaled_features, verbose=0)
        reconstruction_error = np.mean(np.square(scaled_features - reconstructed))
        
        # PCA 비교를 위한 계산
        pca = PCA(n_components=encoding_dim)
        pca_result = pca.fit_transform(scaled_features)
        pca_variance_ratio = np.sum(pca.explained_variance_ratio_)
        
        # 세션 상태에 저장
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


def _display_autoencoder_results(encoding_dim):
    """오토인코더 결과 표시"""
    # 세션에서 데이터 복원
    autoencoder = st.session_state.autoencoder_model
    encoder = st.session_state.encoder_model
    encoded_data = st.session_state.encoded_data
    reconstruction_error = st.session_state.reconstruction_error
    pca_result = st.session_state.pca_result
    pca_variance_ratio = st.session_state.pca_variance_ratio
    
    st.metric("재구성 오차 (MSE)", f"{reconstruction_error:.4f}")
    
    # 오토인코더 vs PCA 비교
    st.subheader("🔍 오토인코더 vs PCA 비교")
    
    # 데이터 로드 (시각화용)
    data_processor = CustomerDataProcessor()
    data = data_processor.load_data()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 오토인코더 결과
        fig1 = px.scatter(
            x=encoded_data[:, 0],
            y=encoded_data[:, 1],
            color=data["Gender"],
            title=f"오토인코더 결과 ({encoding_dim}D)",
            labels={"x": "인코딩 차원 1", "y": "인코딩 차원 2"}
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # PCA 결과
        fig2 = px.scatter(
            x=pca_result[:, 0],
            y=pca_result[:, 1],
            color=data["Gender"],
            title=f"PCA 결과 ({encoding_dim}D)",
            labels={"x": "PC1", "y": "PC2"}
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # 비교 지표
    col1, col2 = st.columns(2)
    with col1:
        st.metric("오토인코더 재구성 오차", f"{reconstruction_error:.4f}")
    with col2:
        st.metric("PCA 설명 분산 비율", f"{pca_variance_ratio:.3f}")


def _show_model_comparison():
    """두 모델 비교 섹션"""
    st.markdown("---")
    st.subheader("🔀 분류 모델 vs 오토인코더 종합 비교")
    
    # 두 모델의 훈련 상태 확인
    classification_trained = st.session_state.get('model_trained', False)
    autoencoder_trained = st.session_state.get('autoencoder_trained', False)
    
    if not classification_trained and not autoencoder_trained:
        st.warning("🔄 **두 모델 모두 훈련이 필요합니다.**")
        st.info("위의 '분류 모델 훈련 시작'과 '오토인코더 훈련 시작' 버튼을 각각 클릭하여 두 모델을 모두 훈련해주세요.")
    elif not classification_trained:
        st.warning("🔄 **분류 모델 훈련이 필요합니다.**")
    elif not autoencoder_trained:
        st.warning("🔄 **오토인코더 훈련이 필요합니다.**")
    else:
        st.success("✅ 두 모델이 모두 훈련 완료되었습니다! 종합 비교를 시작합니다.")
        
        # 비교 분석 실행
        classification_results = st.session_state.get('dl_evaluation_results', {})
        autoencoder_results = {
            'reconstruction_error': st.session_state.get('reconstruction_error', 0),
            'encoded_data': st.session_state.get('encoded_data', None),
            'autoencoder_model': st.session_state.get('autoencoder_model', None)
        }
        
        # 성능 지표 비교 대시보드
        st.subheader("📊 성능 지표 종합 비교")
        
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
            # 종합 성능 점수 계산
            classification_score = classification_results.get('test_accuracy', 0) * 100
            recon_error = autoencoder_results.get('reconstruction_error', 1)
            autoencoder_score = max(0, (1 - min(recon_error, 1)) * 100)
            overall_score = (classification_score + autoencoder_score) / 2
            
            st.metric(
                label="종합 성능 점수",
                value=f"{overall_score:.1f}/100",
                help="두 모델의 성능을 종합한 점수"
            )
        
        # 상세 비교 분석표
        st.subheader("📋 상세 비교 분석")
        
        comparison_data = {
            "비교 항목": [
                "주요 목적", "학습 방식", "출력 결과", "새 고객 예측", 
                "데이터 압축", "이상치 감지", "해석 가능성", "실용성"
            ],
            "분류 모델": [
                "고객 세그먼트 예측", "지도 학습", "클러스터 확률", "즉시 가능",
                "불가능", "제한적", "높음", "매우 높음"
            ],
            "오토인코더": [
                "데이터 압축 및 재구성", "비지도 학습", "압축된 특성", "간접적",
                "가능", "우수", "낮음", "높음"
            ],
            "우수한 모델": [
                "분류 모델", "각각 장점", "분류 모델", "분류 모델",
                "오토인코더", "오토인코더", "분류 모델", "분류 모델"
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # 최종 추천
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
        
        **실행 계획**:
        1. **즉시 실행**: 성능이 우수한 모델을 프로덕션 환경에 우선 적용
        2. **1주일 내**: A/B 테스트를 통한 실제 성능 검증
        3. **1개월 내**: 사용자 피드백 수집 및 모델 성능 모니터링
        4. **3개월 내**: 추가 데이터로 모델 재훈련 및 하이브리드 접근법 도입
        """)
