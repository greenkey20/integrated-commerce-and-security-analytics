"""
딥러닝 모델 모듈

분류 모델, 오토인코더 등 딥러닝 모델들을 담당
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from config.settings import DeepLearningConfig

# TensorFlow 및 관련 라이브러리 동적 로딩
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class DeepLearningModels:
    """딥러닝 모델들을 관리하는 클래스"""
    
    def __init__(self):
        self.classification_model = None
        self.autoencoder_model = None
        self.encoder_model = None
        
    def create_safe_classification_model(self, input_dim, n_clusters, 
                                       hidden_units=None, dropout_rate=None, learning_rate=None):
        """안전한 분류 모델 생성 함수"""
        
        # 기본값 설정
        hidden_units = hidden_units or DeepLearningConfig.DEFAULT_HIDDEN_UNITS
        dropout_rate = dropout_rate or DeepLearningConfig.DEFAULT_DROPOUT_RATE
        learning_rate = learning_rate or DeepLearningConfig.DEFAULT_LEARNING_RATE
        
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
                    name=f"input_dense_{timestamp}",
                ),
                Dropout(dropout_rate, name=f"dropout_1_{timestamp}"),
                Dense(
                    hidden_units // 2,
                    activation="relu",
                    name=f"hidden_dense_{timestamp}",
                ),
                Dropout(dropout_rate / 2, name=f"dropout_2_{timestamp}"),
                Dense(
                    n_clusters, activation="softmax", name=f"output_dense_{timestamp}"
                ),
            ])

            # 모델 컴파일
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            self.classification_model = model
            return model, None

        except Exception as e:
            return None, f"모델 생성 실패: {str(e)}"

    def train_model_with_progress(self, model, X_train, y_train, X_test, y_test, 
                                epochs=None, progress_bar=None, status_text=None):
        """진행상황을 표시하면서 모델을 훈련하는 함수"""
        
        epochs = epochs or DeepLearningConfig.DEFAULT_EPOCHS
        
        try:
            # 조기 종료 콜백 설정
            early_stopping = EarlyStopping(
                monitor="val_loss", 
                patience=DeepLearningConfig.EARLY_STOPPING_PATIENCE, 
                restore_best_weights=True
            )

            callbacks = [early_stopping]

            # Streamlit 전용 콜백 클래스 (progress_bar가 있을 때만)
            if progress_bar and status_text:
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
                callbacks.append(progress_callback)

            # 모델 훈련 실행
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=DeepLearningConfig.BATCH_SIZE,
                callbacks=callbacks,
                verbose=0  # 콘솔 출력 비활성화
            )

            return history, None

        except Exception as e:
            return None, f"모델 훈련 실패: {str(e)}"

    def display_model_architecture_info(self, hidden_units, dropout_rate, n_clusters):
        """모델 아키텍처 정보를 사용자 친화적으로 표시하는 함수"""
        st.write("**🏗️ 구성된 신경망 구조:**")

        architecture_info = [
            f"입력층: 3개 특성 (나이, 소득, 지출점수)",
            f"은닉층 1: {hidden_units}개 뉴런 + ReLU 활성화 함수",
            f"드롭아웃 1: {dropout_rate*100:.0f}% 뉴런 무작위 비활성화 (과적합 방지)",
            f"은닉층 2: {hidden_units//2}개 뉴런 + ReLU 활성화 함수",
            f"드롭아웃 2: {dropout_rate/2*100:.0f}% 뉴런 무작위 비활성화",
            f"출력층: {n_clusters}개 뉴런 + Softmax (각 클러스터 확률 계산)",
        ]

        for i, layer_info in enumerate(architecture_info, 1):
            st.write(f"{i}. {layer_info}")

    def evaluate_and_display_results(self, model, X_test, y_test, history, n_clusters):
        """모델 성능을 평가하고 결과를 시각화하는 함수"""
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
                    line=dict(color="blue"),
                ))

                fig.add_trace(go.Scatter(
                    x=list(epochs_range),
                    y=history.history["val_accuracy"],
                    mode="lines",
                    name="검증 정확도",
                    line=dict(color="red"),
                ))

                fig.update_layout(
                    title="모델 훈련 과정",
                    xaxis_title="에포크",
                    yaxis_title="정확도",
                    height=400,
                )

                st.plotly_chart(fig, use_container_width=True)

            return {
                "test_accuracy": test_accuracy,
                "predictions": y_pred_classes,
                "probabilities": y_pred_probs,
                "confidence": avg_confidence,
            }

        except Exception as e:
            st.error(f"모델 평가 중 오류 발생: {str(e)}")
            return None

    def create_autoencoder(self, input_dim, encoding_dim=None):
        """오토인코더 모델 생성"""
        encoding_dim = encoding_dim or DeepLearningConfig.DEFAULT_ENCODING_DIM
        
        try:
            # 모델 구성을 위한 고유 이름 생성
            import time
            timestamp = str(int(time.time() * 1000))[-6:]

            # 오토인코더 모델 구성
            input_layer = layers.Input(shape=(input_dim,), name=f"ae_input_{timestamp}")

            # 인코더
            encoded = layers.Dense(8, activation="relu", name=f"ae_encode1_{timestamp}")(input_layer)
            encoded = layers.Dense(encoding_dim, activation="relu", name=f"ae_encoded_{timestamp}")(encoded)

            # 디코더
            decoded = layers.Dense(8, activation="relu", name=f"ae_decode1_{timestamp}")(encoded)
            decoded = layers.Dense(input_dim, activation="linear", name=f"ae_output_{timestamp}")(decoded)

            # 모델 생성
            autoencoder = keras.Model(input_layer, decoded, name=f"autoencoder_{timestamp}")
            encoder = keras.Model(input_layer, encoded, name=f"encoder_{timestamp}")

            # 컴파일
            autoencoder.compile(optimizer="adam", loss="mse")

            self.autoencoder_model = autoencoder
            self.encoder_model = encoder
            
            return autoencoder, encoder, None

        except Exception as e:
            return None, None, f"오토인코더 생성 실패: {str(e)}"

    def train_autoencoder(self, autoencoder, data, epochs=None, progress_bar=None, status_text=None):
        """오토인코더 훈련"""
        epochs = epochs or DeepLearningConfig.AUTOENCODER_EPOCHS
        
        try:
            callbacks = []
            
            # Streamlit 콜백 (필요시)
            if progress_bar and status_text:
                class AutoencoderProgressCallback(keras.callbacks.Callback):
                    def __init__(self, total_epochs, progress_bar, status_text):
                        super().__init__()
                        self.total_epochs = total_epochs
                        self.progress_bar = progress_bar
                        self.status_text = status_text

                    def on_epoch_end(self, epoch, logs=None):
                        progress = (epoch + 1) / self.total_epochs
                        self.progress_bar.progress(progress)
                        
                        if logs:
                            self.status_text.text(
                                f"에포크 {epoch + 1}/{self.total_epochs} - 손실: {logs.get('loss', 0):.4f}"
                            )

                progress_callback = AutoencoderProgressCallback(epochs, progress_bar, status_text)
                callbacks.append(progress_callback)

            # 훈련 실행
            history = autoencoder.fit(
                data, data,
                epochs=epochs,
                batch_size=DeepLearningConfig.BATCH_SIZE,
                validation_split=DeepLearningConfig.VALIDATION_SPLIT,
                verbose=0,
                callbacks=callbacks
            )

            return history, None

        except Exception as e:
            return None, f"오토인코더 훈련 실패: {str(e)}"


# 전역 인스턴스 생성 (기존 함수와의 호환성)
dl_models = DeepLearningModels()

# 기존 함수들과의 호환성을 위한 래퍼들
def create_safe_classification_model(input_dim, n_clusters, hidden_units, dropout_rate, learning_rate):
    return dl_models.create_safe_classification_model(input_dim, n_clusters, hidden_units, dropout_rate, learning_rate)

def train_model_with_progress(model, X_train, y_train, X_test, y_test, epochs, progress_bar, status_text):
    return dl_models.train_model_with_progress(model, X_train, y_train, X_test, y_test, epochs, progress_bar, status_text)

def display_model_architecture_info(hidden_units, dropout_rate, n_clusters):
    return dl_models.display_model_architecture_info(hidden_units, dropout_rate, n_clusters)

def evaluate_and_display_results(model, X_test, y_test, history, n_clusters):
    return dl_models.evaluate_and_display_results(model, X_test, y_test, history, n_clusters)
