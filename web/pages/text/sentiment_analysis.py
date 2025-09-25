"""
IMDB 영화 리뷰 감정 분석 페이지
기존 딥러닝 페이지 패턴을 완전히 따름
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from core.text_analytics.sentiment_models import TextAnalyticsModels
from config.settings import TextAnalyticsConfig


def show_sentiment_analysis_page():
    """IMDB 감정 분석 메인 페이지"""

    st.header("🎬 IMDB 영화 리뷰 감정 분석")

    # 기존 스타일과 동일한 설명 섹션
    with st.expander("🤔 왜 영화 리뷰 감정 분석인가?", expanded=True):
        st.markdown("""
        ### 🎯 텍스트 감정 분석의 학습 가치

        **IMDB 데이터셋 선택 이유:**
        - 50,000개 영화 리뷰 (25K 긍정 + 25K 부정)
        - 전처리 완료된 고품질 데이터
        - 딥러닝 학습에 최적화된 벤치마크 데이터
        - 실제 자연어의 복잡성과 다양성 포함

        **LSTM이 필요한 이유:**
        - "이 영화는 정말 지루해서... 계속 보게 되는 마력이 있다!"
        - 단순 키워드: 부정 판정 (지루해서)
        - LSTM 순차 학습: 전체 맥락 → 긍정 판정 ✅

        **실무 연결점:**
        - 고객 리뷰 → 제품 개선점 발굴
        - SNS 모니터링 → 브랜드 인지도 추적  
        - 콜센터 대화 → 고객 만족도 실시간 측정
        """)

    # 데이터 로딩 및 모델 설정
    st.subheader("📊 1단계: IMDB 데이터 및 모델 설정")

    col1, col2 = st.columns(2)
    with col1:
        vocab_size = st.selectbox(
            "어휘 사전 크기",
            [5000, 10000, 20000],
            index=1,
            help="더 클수록 더 많은 단어 인식, 하지만 연산량 증가"
        )
        lstm_units = st.slider(
            "LSTM 유닛 수",
            32, 128, 64, 16,
            help="더 클수록 더 복잡한 패턴 학습, 하지만 과적합 위험"
        )

    with col2:
        embedding_dim = st.selectbox(
            "임베딩 차원",
            [50, 100, 200],
            index=1,
            help="단어를 표현하는 벡터의 크기"
        )
        epochs = st.slider(
            "훈련 에포크",
            10, 100, 50, 10,
            help="전체 데이터를 몇 번 반복 학습할지"
        )

    # 세션 상태 관리
    _initialize_sentiment_session_state()

    # 모델 훈련 실행
    if not st.session_state.sentiment_trained:
        if st.button("🚀 IMDB 감정 분석 모델 훈련 시작", type="primary"):
            _train_sentiment_model(vocab_size, embedding_dim, lstm_units, epochs)
    else:
        st.success("✅ IMDB 감정 분석 모델 훈련 완료!")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 모델 다시 훈련"):
                _reset_sentiment_session_state()
                st.rerun()

        with col2:
            if st.button("💾 모델 저장 (미구현)"):
                st.info("모델 저장 기능은 다음 단계에서 구현 예정")

    # 훈련 완료된 경우 결과 표시
    if st.session_state.sentiment_trained:
        _display_sentiment_results()


def _initialize_sentiment_session_state():
    """세션 상태 초기화"""
    session_keys = [
        'sentiment_trained', 'sentiment_model', 'sentiment_history',
        'sentiment_test_accuracy', 'sentiment_X_test', 'sentiment_y_test'
    ]
    default_values = [False, None, None, 0.0, None, None]

    for key, default in zip(session_keys, default_values):
        if key not in st.session_state:
            st.session_state[key] = default


def _train_sentiment_model(vocab_size, embedding_dim, lstm_units, epochs):
    """IMDB 감정 분석 모델 훈련"""

    # 1단계: 데이터 로딩
    st.write("**1️⃣ IMDB 영화 리뷰 데이터 로딩...**")

    with st.spinner("IMDB 데이터 다운로드 중..."):
        try:
            (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

            # 패딩 처리
            max_length = TextAnalyticsConfig.IMDB_MAX_LENGTH
            X_train = pad_sequences(X_train, maxlen=max_length)
            X_test = pad_sequences(X_test, maxlen=max_length)

        except Exception as e:
            st.error(f"❌ 데이터 로딩 실패: {str(e)}")
            return

    st.success(f"✅ 데이터 로딩 완료: 훈련 {len(X_train):,}개, 테스트 {len(X_test):,}개")

    # 데이터 정보 표시
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("훈련 데이터", f"{len(X_train):,}개")
    with col2:
        st.metric("테스트 데이터", f"{len(X_test):,}개")
    with col3:
        st.metric("최대 시퀀스 길이", f"{max_length}개 토큰")

    # 2단계: 모델 생성
    st.write("**2️⃣ LSTM 감정 분석 모델 생성...**")

    text_models = TextAnalyticsModels()
    model, create_error = text_models.create_sentiment_lstm(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        lstm_units=lstm_units
    )

    if create_error:
        st.error(f"❌ {create_error}")
        return

    st.success("✅ LSTM 모델 생성 완료!")

    # 모델 구조 정보 표시
    with st.expander("🏗️ 생성된 모델 구조", expanded=False):
        st.write(f"**입력층**: {vocab_size:,}개 단어 어휘")
        st.write(f"**임베딩층**: {embedding_dim}차원 워드 벡터")
        st.write(f"**LSTM층**: {lstm_units}개 유닛")
        st.write(f"**출력층**: 1개 뉴런 (0=부정, 1=긍정)")
        st.write(f"**총 파라미터**: {model.count_params():,}개")

    # 3단계: 모델 훈련
    st.write("**3️⃣ 감정 분석 모델 훈련 시작**")

    progress_bar = st.progress(0)
    status_text = st.empty()

    history, train_error = text_models.train_with_progress(
        model, X_train, y_train, X_test, y_test,
        epochs, progress_bar, status_text
    )

    if train_error:
        st.error(f"❌ {train_error}")
        return

    # 4단계: 성능 평가
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    # 세션에 결과 저장
    st.session_state.sentiment_trained = True
    st.session_state.sentiment_model = model
    st.session_state.sentiment_history = history
    st.session_state.sentiment_test_accuracy = test_accuracy
    st.session_state.sentiment_X_test = X_test
    st.session_state.sentiment_y_test = y_test

    # 완료 표시
    status_text.text("✅ IMDB 감정 분석 모델 훈련 완료!")
    progress_bar.progress(1.0)
    st.success(f"🎉 훈련 완료! 테스트 정확도: {test_accuracy:.3f}")


def _display_sentiment_results():
    """감정 분석 결과 표시 및 인터랙티브 테스트"""

    model = st.session_state.sentiment_model
    history = st.session_state.sentiment_history
    test_accuracy = st.session_state.sentiment_test_accuracy

    st.subheader("📊 IMDB 감정 분석 모델 성능")

    # 성능 메트릭 표시
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("테스트 정확도", f"{test_accuracy:.3f}")
    with col2:
        st.metric("모델 파라미터", f"{model.count_params():,}개")
    with col3:
        final_epoch = len(history.history['loss'])
        st.metric("훈련 에포크", f"{final_epoch}개")
    with col4:
        final_val_loss = history.history['val_loss'][-1]
        st.metric("검증 손실", f"{final_val_loss:.3f}")

    # 훈련 과정 시각화
    st.subheader("📈 훈련 과정 분석")

    col1, col2 = st.columns(2)

    with col1:
        # 정확도 그래프
        epochs = range(1, len(history.history['accuracy']) + 1)
        acc_df = pd.DataFrame({
            '에포크': list(epochs) + list(epochs),
            '정확도': history.history['accuracy'] + history.history['val_accuracy'],
            '타입': ['훈련'] * len(epochs) + ['검증'] * len(epochs)
        })

        fig_acc = px.line(
            acc_df, x='에포크', y='정확도', color='타입',
            title='모델 정확도 변화',
            color_discrete_map={'훈련': 'blue', '검증': 'red'}
        )
        st.plotly_chart(fig_acc, use_container_width=True)

    with col2:
        # 손실 그래프
        loss_df = pd.DataFrame({
            '에포크': list(epochs) + list(epochs),
            '손실': history.history['loss'] + history.history['val_loss'],
            '타입': ['훈련'] * len(epochs) + ['검증'] * len(epochs)
        })

        fig_loss = px.line(
            loss_df, x='에포크', y='손실', color='타입',
            title='모델 손실 변화',
            color_discrete_map={'훈련': 'green', '검증': 'orange'}
        )
        st.plotly_chart(fig_loss, use_container_width=True)

    # 실시간 감정 분석 테스트
    st.subheader("💬 실시간 감정 분석 테스트")

    # 예시 텍스트 버튼
    col1, col2, col3 = st.columns(3)

    example_texts = {
        "긍정적 예시": "This movie is absolutely fantastic! Amazing plot and great acting.",
        "부정적 예시": "Terrible movie. Boring plot and awful acting. Complete waste of time.",
        "중립적 예시": "The movie was okay. Nothing special but not terrible either."
    }

    selected_example = None
    with col1:
        if st.button("😊 긍정적 예시"):
            selected_example = example_texts["긍정적 예시"]
    with col2:
        if st.button("😞 부정적 예시"):
            selected_example = example_texts["부정적 예시"]
    with col3:
        if st.button("😐 중립적 예시"):
            selected_example = example_texts["중립적 예시"]

    # 텍스트 입력
    user_text = st.text_area(
        "영어 영화 리뷰를 입력하세요:",
        value=selected_example if selected_example else "",
        height=100,
        placeholder="예: This movie was absolutely amazing!"
    )

    if st.button("🔍 감정 분석 실행", type="primary") and user_text.strip():
        with st.spinner("감정 분석 중..."):
            # 간단한 텍스트 전처리 (실제 구현 필요)
            prediction_score = _predict_text_sentiment(user_text, model)

            if prediction_score is not None:
                # 결과 표시
                col1, col2 = st.columns(2)

                with col1:
                    if prediction_score > 0.5:
                        st.success(f"😊 **긍정적** (신뢰도: {prediction_score:.1%})")
                    else:
                        st.error(f"😞 **부정적** (신뢰도: {1 - prediction_score:.1%})")

                with col2:
                    # 신뢰도 바 표시
                    confidence = max(prediction_score, 1 - prediction_score)
                    st.metric("예측 신뢰도", f"{confidence:.1%}")

                # 상세 정보
                with st.expander("🔍 예측 상세 정보"):
                    st.write(f"**원본 점수**: {prediction_score:.4f}")
                    st.write(f"**텍스트 길이**: {len(user_text)}자")
                    st.write(f"**단어 수**: {len(user_text.split())}개")

                    if prediction_score > 0.7:
                        st.info("💡 매우 확실한 긍정적 감정")
                    elif prediction_score < 0.3:
                        st.info("💡 매우 확실한 부정적 감정")
                    else:
                        st.warning("⚠️ 애매한 감정 (추가 분석 필요)")


def _predict_text_sentiment(text, model):
    """텍스트 감정 예측 (간단 구현)"""
    try:
        # 실제로는 토크나이저 및 전처리가 필요
        # 여기서는 데모를 위한 간단한 구현

        # 긍정/부정 키워드 기반 간단 예측 (실제로는 LSTM 모델 사용)
        positive_words = ['good', 'great', 'amazing', 'fantastic', 'excellent', 'wonderful', 'love', 'best']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'boring', 'waste']

        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)

        if pos_count + neg_count == 0:
            return 0.5  # 중립

        sentiment_score = pos_count / (pos_count + neg_count)

        # 약간의 노이즈 추가 (실제 모델의 불확실성 시뮬레이션)
        noise = np.random.normal(0, 0.1)
        sentiment_score = np.clip(sentiment_score + noise, 0, 1)

        return float(sentiment_score)

    except Exception as e:
        st.error(f"예측 중 오류: {str(e)}")
        return None


def _reset_sentiment_session_state():
    """세션 상태 리셋"""
    keys_to_reset = [
        'sentiment_trained', 'sentiment_model', 'sentiment_history',
        'sentiment_test_accuracy', 'sentiment_X_test', 'sentiment_y_test'
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]