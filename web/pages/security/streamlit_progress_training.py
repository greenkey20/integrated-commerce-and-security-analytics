"""
Streamlit에서 TensorFlow 모델 훈련 진행상황 표시

security_analysis_page.py의 train_hybrid_model 함수를 개선하여
실시간 progress bar와 epoch별 진행상황을 표시
"""

import streamlit as st
import tensorflow as tf
import numpy as np
import time
from typing import Dict, Any
import threading
import queue

class StreamlitProgressCallback(tf.keras.callbacks.Callback):
    """Streamlit용 실시간 진행상황 콜백"""
    
    def __init__(self, total_epochs: int):
        super().__init__()
        self.total_epochs = total_epochs
        self.current_epoch = 0
        
        # Streamlit 컴포넌트 초기화
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        self.metrics_container = st.empty()
        
        # 메트릭 기록용
        self.epoch_metrics = []
    
    def on_train_begin(self, logs=None):
        """훈련 시작시 호출"""
        self.status_text.text("🚀 모델 훈련을 시작합니다...")
        self.progress_bar.progress(0)
    
    def on_epoch_begin(self, epoch, logs=None):
        """에포크 시작시 호출"""
        self.current_epoch = epoch + 1
        self.status_text.text(f"📈 Epoch {self.current_epoch}/{self.total_epochs} 훈련 중...")
    
    def on_batch_end(self, batch, logs=None):
        """배치 종료시 호출 (선택적)"""
        if hasattr(self.model, 'total_batches'):
            batch_progress = (batch + 1) / self.model.total_batches
            epoch_progress = (self.current_epoch - 1 + batch_progress) / self.total_epochs
            self.progress_bar.progress(min(epoch_progress, 1.0))
    
    def on_epoch_end(self, epoch, logs=None):
        """에포크 종료시 호출"""
        logs = logs or {}
        
        # 진행률 업데이트
        progress = (epoch + 1) / self.total_epochs
        self.progress_bar.progress(progress)
        
        # 메트릭 기록
        epoch_data = {
            'epoch': epoch + 1,
            'loss': logs.get('loss', 0.0),
            'accuracy': logs.get('accuracy', 0.0),
            'val_loss': logs.get('val_loss', 0.0),
            'val_accuracy': logs.get('val_accuracy', 0.0)
        }
        self.epoch_metrics.append(epoch_data)
        
        # 실시간 메트릭 표시
        self._update_metrics_display(epoch_data)
        
        # 상태 텍스트 업데이트
        self.status_text.text(
            f"✅ Epoch {epoch + 1}/{self.total_epochs} 완료 - "
            f"Loss: {logs.get('loss', 0.0):.4f}, "
            f"Accuracy: {logs.get('accuracy', 0.0):.4f}"
        )
    
    def on_train_end(self, logs=None):
        """훈련 완료시 호출"""
        self.progress_bar.progress(1.0)
        self.status_text.text("🎉 모델 훈련이 완료되었습니다!")
    
    def _update_metrics_display(self, epoch_data: Dict[str, float]):
        """실시간 메트릭 표시 업데이트"""
        with self.metrics_container.container():
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Loss",
                    value=f"{epoch_data['loss']:.4f}",
                    delta=self._calculate_delta('loss', epoch_data['loss'])
                )
            
            with col2:
                st.metric(
                    label="Accuracy", 
                    value=f"{epoch_data['accuracy']:.4f}",
                    delta=self._calculate_delta('accuracy', epoch_data['accuracy'])
                )
            
            with col3:
                if epoch_data['val_loss'] > 0:
                    st.metric(
                        label="Val Loss",
                        value=f"{epoch_data['val_loss']:.4f}",
                        delta=self._calculate_delta('val_loss', epoch_data['val_loss'])
                    )
            
            with col4:
                if epoch_data['val_accuracy'] > 0:
                    st.metric(
                        label="Val Accuracy",
                        value=f"{epoch_data['val_accuracy']:.4f}",
                        delta=self._calculate_delta('val_accuracy', epoch_data['val_accuracy'])
                    )
    
    def _calculate_delta(self, metric_name: str, current_value: float) -> str:
        """이전 에포크와의 변화량 계산"""
        if len(self.epoch_metrics) < 2:
            return None
        
        previous_value = self.epoch_metrics[-2].get(metric_name, current_value)
        delta = current_value - previous_value
        
        if metric_name in ['loss', 'val_loss']:
            # Loss는 감소가 좋음
            return f"{delta:+.4f}" if delta != 0 else None
        else:
            # Accuracy는 증가가 좋음
            return f"{delta:+.4f}" if delta != 0 else None


class EnhancedProgressTracker:
    """고급 진행상황 추적기"""
    
    def __init__(self):
        self.start_time = None
        self.epoch_times = []
    
    def create_detailed_callback(self, total_epochs: int, validation_data=None):
        """상세한 콜백 생성"""
        callbacks = [
            StreamlitProgressCallback(total_epochs),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        return callbacks


def train_model_with_progress(model_builder, X_train, X_test, y_train, y_test, 
                            model_type="hybrid", epochs=50):
    """진행상황을 표시하면서 모델 훈련"""
    
    st.subheader("🚀 실시간 모델 훈련")
    
    # 진행상황 추적기 초기화
    tracker = EnhancedProgressTracker()
    
    # 훈련 시작 안내
    with st.expander("훈련 설정 확인", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("훈련 샘플", len(X_train))
        with col2:
            st.metric("테스트 샘플", len(X_test))
        with col3:
            st.metric("에포크 수", epochs)
    
    # 모델 구축
    if model_type == "hybrid":
        model = model_builder.build_hybrid_model(X_train.shape[1])
        st.info("🔥 하이브리드 모델 (MLP + CNN) 구조로 훈련합니다")
    elif model_type == "mlp":
        model = model_builder.build_mlp_model(X_train.shape[1])
        st.info("⚡ MLP 분류 모델로 훈련합니다")
    else:
        raise ValueError(f"지원하지 않는 모델 타입: {model_type}")
    
    # 콜백 설정
    callbacks = tracker.create_detailed_callback(epochs, (X_test, y_test))
    
    # 훈련 데이터 준비 (모델 타입에 따라)
    if model_type == "hybrid":
        # 하이브리드 모델용 시퀀스 데이터 생성
        sequence_length = 10
        X_train_seq = model_builder.create_sequences(X_train, sequence_length)
        X_train_ind = X_train[sequence_length-1:]
        y_train_seq = y_train[sequence_length-1:]
        
        X_test_seq = model_builder.create_sequences(X_test, sequence_length)
        X_test_ind = X_test[sequence_length-1:]
        y_test_seq = y_test[sequence_length-1:]
        
        train_data = ([X_train_ind, X_train_seq], y_train_seq)
        validation_data = ([X_test_ind, X_test_seq], y_test_seq)
    else:
        train_data = (X_train, y_train)
        validation_data = (X_test, y_test)
    
    # 실제 훈련 시작
    st.write("### 📊 실시간 훈련 진행상황")
    
    try:
        # 훈련 실행
        history = model.fit(
            train_data[0], train_data[1],
            validation_data=validation_data,
            epochs=epochs,
            batch_size=64,
            callbacks=callbacks,
            verbose=0  # Streamlit 콜백에서 처리하므로 0으로 설정
        )
        
        # 훈련 완료 후 결과 표시
        st.success("🎉 모델 훈련이 성공적으로 완료되었습니다!")
        
        # 최종 성능 평가
        show_final_evaluation(model_builder, X_test, y_test, model_type)
        
        return history
        
    except Exception as e:
        st.error(f"❌ 훈련 중 오류 발생: {str(e)}")
        return None


def show_final_evaluation(model_builder, X_test, y_test, model_type):
    """최종 성능 평가 표시"""
    st.write("### 📈 최종 성능 평가")
    
    # 성능 메트릭 계산
    metrics = model_builder.evaluate_binary_model(X_test, y_test)
    
    # 메트릭 표시
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        value = metrics['accuracy']
        delta_color = "normal" if value < 0.95 else "off"  # 0.95 이상이면 의심
        st.metric("정확도", f"{value:.3f}", delta_color=delta_color)
        if value >= 0.98:
            st.warning("⚠️ 정확도가 너무 높습니다. Overfitting 가능성이 있습니다.")
    
    with col2:
        value = metrics['precision']
        st.metric("정밀도", f"{value:.3f}")
    
    with col3:
        value = metrics['recall']
        st.metric("재현율", f"{value:.3f}")
    
    with col4:
        value = metrics['f1_score']
        st.metric("F1 점수", f"{value:.3f}")
    
    # 성능 해석
    interpret_performance_results(metrics)


def interpret_performance_results(metrics):
    """성능 결과 해석 및 조언"""
    st.write("### 🤔 성능 해석 및 조언")
    
    accuracy = metrics['accuracy']
    precision = metrics['precision']
    recall = metrics['recall']
    f1_score = metrics['f1_score']
    
    if accuracy > 0.98:
        st.error("""
        **🚨 Overfitting 의심됨**
        - 정확도가 98% 이상으로 너무 높습니다
        - 실제 환경에서는 이런 성능이 나오기 어렵습니다
        
        **개선 방안:**
        1. 더 현실적인 데이터 사용
        2. 정규화 강화 (Dropout 증가)
        3. 모델 복잡도 감소
        4. 더 많은 훈련 데이터 확보
        """)
    elif accuracy > 0.95:
        st.warning("""
        **⚠️ 성능이 매우 높음**
        - 실제 배포 전 교차검증 필요
        - 다양한 데이터셋으로 검증 권장
        """)
    elif accuracy > 0.90:
        st.success("""
        **✅ 우수한 성능**
        - 실용적인 수준의 성능입니다
        - 실제 배포 가능한 품질입니다
        """)
    else:
        st.info("""
        **📊 개선 여지 있음**
        - 모델 아키텍처 개선 고려
        - 하이퍼파라미터 튜닝 필요
        - 더 많은 특성 엔지니어링 고려
        """)
    
    # 균형 지표 분석
    if abs(precision - recall) > 0.1:
        st.info(f"""
        **⚖️ 정밀도-재현율 불균형 감지**
        - 정밀도: {precision:.3f}, 재현율: {recall:.3f}
        - 비즈니스 요구사항에 따라 임계값 조정 고려
        """)


# ============================================================================
# security_analysis_page.py에서 사용할 개선된 함수
# ============================================================================

def improved_train_hybrid_model(model_builder, X_train, X_test, y_train, y_test, feature_names):
    """개선된 하이브리드 모델 훈련 (progress bar 포함)"""
    st.write("**2️⃣ 하이브리드 모델 구축 (MLP + CNN)**")
    
    with st.expander("하이브리드 모델 구조 설명"):
        st.markdown("""
        **MLP 브랜치**: 개별 패킷의 특성 분석
        **CNN 브랜치**: 시계열 패턴 분석  
        **융합 레이어**: 두 관점을 통합하여 최종 판단
        """)
    
    if st.button("🚀 하이브리드 모델 훈련 시작"):
        # 개선된 훈련 함수 호출
        history = train_model_with_progress(
            model_builder, 
            X_train, X_test, y_train, y_test,
            model_type="hybrid",
            epochs=50
        )
        
        if history:
            # 세션에 모델 저장
            st.session_state.security_model = model_builder.model
            st.session_state.security_scaler = model_builder.scaler
            
            st.balloons()


# 사용 예시
if __name__ == "__main__":
    st.set_page_config(page_title="진행상황 표시 테스트")
    st.title("TensorFlow 훈련 진행상황 표시 테스트")
    
    # 테스트용 간단한 모델 훈련
    if st.button("테스트 시작"):
        import numpy as np
        from tensorflow import keras
        
        # 더미 데이터 생성
        X_train = np.random.random((1000, 10))
        y_train = np.random.randint(0, 2, 1000)
        X_test = np.random.random((200, 10))
        y_test = np.random.randint(0, 2, 200)
        
        # 간단한 모델 생성
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(10,)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # 진행상황 콜백 생성
        progress_callback = StreamlitProgressCallback(total_epochs=20)
        
        # 훈련 실행
        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=20,
            batch_size=32,
            callbacks=[progress_callback],
            verbose=0
        )
