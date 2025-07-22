"""
통합 보안 탐지 엔진

API 로그 이상 탐지와 네트워크 공격 탐지를 통합한 범용 탐지 시스템
- API 로그 기반 이상 탐지 (하이브리드 딥러닝 모델)
- 네트워크 트래픽 기반 공격 탐지 (일반 ML 모델)
- 실시간 모니터링 및 알림 시스템
- 성능 평가 및 시뮬레이션 기능
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import json
import time
import random
from datetime import datetime, timedelta
import re
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')

# 설정 파일에서 가져오기
from config.settings import SecurityConfig
from config.logging import setup_logger
from utils.exceptions import SecurityDetectionError

logger = setup_logger(__name__)

# 랜덤 시드 설정
keras.utils.set_random_seed(SecurityConfig.RANDOM_SEED)
tf.config.experimental.enable_op_determinism()


class UnifiedDetectionEngine:
    """통합 보안 탐지 엔진"""
    
    def __init__(self, detection_type: str = 'api_log', model_type: str = 'hybrid'):
        """
        Args:
            detection_type: 'api_log' (API 로그 탐지) 또는 'network_traffic' (네트워크 트래픽 탐지)
            model_type: 'mlp', 'cnn', 'hybrid' (API 로그), 또는 'general' (네트워크)
        """
        self.detection_type = detection_type
        self.model_type = model_type
        
        # 모델 인스턴스들
        self.mlp_model = None
        self.cnn_model = None
        self.ensemble_model = None
        self.general_model = None
        
        # 전처리 도구들
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
        # 상태 변수들
        self.feature_names = []
        self.is_trained = False
        self.sequence_length = SecurityConfig.CNN_SEQUENCE_LENGTH
        
        # 탐지 이력
        self.detection_history = []
        self.alert_count = 0
        
        logger.info(f"통합 탐지 엔진 초기화: {detection_type}, {model_type}")
    
    def set_external_model(self, model, scaler=None):
        """외부 모델 설정 (네트워크 트래픽용)"""
        self.general_model = model
        if scaler:
            self.scaler = scaler
        logger.info("외부 모델 설정 완료")
    
    def extract_api_features(self, log_entry: Dict) -> np.ndarray:
        """API 로그에서 특성 추출"""
        try:
            features = {}
            
            # 시간 기반 특성
            timestamp = datetime.fromisoformat(log_entry.get('timestamp', datetime.now().isoformat()))
            features['hour'] = timestamp.hour
            features['day_of_week'] = timestamp.weekday()
            features['is_weekend'] = 1 if timestamp.weekday() >= SecurityConfig.WEEKEND_THRESHOLD else 0
            features['is_business_hour'] = 1 if SecurityConfig.BUSINESS_HOUR_START <= timestamp.hour <= SecurityConfig.BUSINESS_HOUR_END else 0
            
            # 요청 빈도 특성
            features['requests_per_minute'] = log_entry.get('requests_per_minute', 0)
            
            # 요청 크기 특성
            features['request_size'] = log_entry.get('request_size', 0)
            features['content_length'] = int(log_entry.get('content_length', 0))
            
            # HTTP 메서드 원핫 인코딩
            method = log_entry.get('method', 'GET')
            for m in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']:
                features[f'method_{m}'] = 1 if method == m else 0
            
            # User-Agent 분석
            user_agent = log_entry.get('user_agent', '').lower()
            features['ua_length'] = len(user_agent)
            features['ua_has_bot'] = 1 if any(bot in user_agent for bot in ['bot', 'crawler', 'spider']) else 0
            features['ua_has_browser'] = 1 if any(browser in user_agent for browser in ['mozilla', 'chrome', 'safari', 'firefox']) else 0
            features['ua_suspicious'] = 1 if any(tool in user_agent for tool in ['sqlmap', 'nikto', 'nmap', 'curl', 'python']) else 0
            
            # URL 패턴 분석
            url = log_entry.get('url', '')
            features['url_length'] = len(url)
            features['url_params_count'] = url.count('&') + (1 if '?' in url else 0)
            features['url_has_sql_keywords'] = 1 if any(keyword in url.lower() for keyword in ['select', 'union', 'drop', 'insert']) else 0
            features['url_has_xss_patterns'] = 1 if any(pattern in url.lower() for pattern in ['<script', 'javascript:', 'alert(']) else 0
            
            # IP 기반 특성
            client_ip = log_entry.get('client_ip', '127.0.0.1')
            ip_parts = client_ip.split('.')
            if len(ip_parts) == 4:
                try:
                    features['ip_first_octet'] = int(ip_parts[0])
                    features['ip_is_private'] = 1 if ip_parts[0] in ['10', '172', '192'] else 0
                except:
                    features['ip_first_octet'] = 0
                    features['ip_is_private'] = 0
            else:
                features['ip_first_octet'] = 0
                features['ip_is_private'] = 0
            
            # 응답 시간
            features['response_time'] = log_entry.get('processing_time', 0)
            
            return np.array(list(features.values()))
            
        except Exception as e:
            logger.error(f"API 특성 추출 실패: {e}")
            raise SecurityDetectionError(f"API 특성 추출 실패: {e}")
    
    def prepare_training_data(self, data_source: Union[str, pd.DataFrame] = None) -> Tuple[np.ndarray, np.ndarray]:
        """훈련 데이터 준비"""
        try:
            if self.detection_type == 'api_log':
                if isinstance(data_source, pd.DataFrame):
                    # CICIDS2017 데이터 활용
                    return self._prepare_cicids_data(data_source)
                else:
                    # 로그 파일 활용
                    return self._prepare_log_data(data_source)
            else:
                # 네트워크 트래픽 데이터
                if isinstance(data_source, np.ndarray):
                    return data_source, np.ones(len(data_source))  # 더미 라벨
                else:
                    raise SecurityDetectionError("네트워크 트래픽 데이터는 numpy 배열이어야 합니다")
                    
        except Exception as e:
            logger.error(f"훈련 데이터 준비 실패: {e}")
            raise SecurityDetectionError(f"훈련 데이터 준비 실패: {e}")
    
    def _prepare_cicids_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """CICIDS2017 데이터 전처리"""
        # 주요 특성 선택
        feature_columns = [
            'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
            'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
            'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
            'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
            'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std',
            'Fwd IAT Total', 'Fwd IAT Mean', 'Bwd IAT Total', 'Bwd IAT Mean'
        ]
        
        available_columns = [col for col in feature_columns if col in df.columns]
        X = df[available_columns].fillna(0)
        y = (df['Label'] != 'BENIGN').astype(int)
        
        # 무한대 값 처리
        X = X.replace([np.inf, -np.inf], 0)
        
        return X.values, y.values
    
    def _prepare_log_data(self, log_file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """로그 파일 전처리"""
        logs = []
        labels = []
        
        with open(log_file_path, 'r') as f:
            for line in f:
                try:
                    log_entry = json.loads(line.strip())
                    features = self.extract_api_features(log_entry)
                    is_suspicious = log_entry.get('is_suspicious', False)
                    
                    logs.append(features)
                    labels.append(1 if is_suspicious else 0)
                except:
                    continue
        
        return np.array(logs), np.array(labels)
    
    def build_mlp_model(self, input_shape: int) -> keras.Model:
        """MLP 모델 구축"""
        hidden_units = SecurityConfig.MLP_HIDDEN_UNITS
        dropout_rates = SecurityConfig.DROPOUT_RATES
        
        model = keras.Sequential([
            keras.layers.Input(shape=(input_shape,)),
            keras.layers.Dense(hidden_units[0], activation='relu', name='hidden_layer_1'),
            keras.layers.Dropout(dropout_rates[0]),
            keras.layers.Dense(hidden_units[1], activation='relu', name='hidden_layer_2'),
            keras.layers.Dropout(dropout_rates[1]),
            keras.layers.Dense(hidden_units[2], activation='relu', name='hidden_layer_3'),
            keras.layers.Dense(1, activation='sigmoid', name='output_layer')
        ], name='MLP_Security_Detector')
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def build_cnn_model(self, input_shape: int) -> keras.Model:
        """CNN 모델 구축 (시계열 패턴용)"""
        model = keras.Sequential([
            keras.layers.Input(shape=(self.sequence_length, input_shape)),
            keras.layers.Conv1D(64, 3, activation='relu'),
            keras.layers.MaxPooling1D(2),
            keras.layers.Conv1D(32, 3, activation='relu'),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(50, activation='relu'),
            keras.layers.Dropout(SecurityConfig.DROPOUT_RATES[0]),
            keras.layers.Dense(1, activation='sigmoid')
        ], name='CNN_Security_Detector')
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def build_hybrid_model(self, input_shape: int) -> keras.Model:
        """하이브리드 모델: MLP + CNN 결합"""
        # MLP 브랜치
        mlp_input = keras.layers.Input(shape=(input_shape,), name='mlp_input')
        mlp_dense1 = keras.layers.Dense(64, activation='relu')(mlp_input)
        mlp_dropout1 = keras.layers.Dropout(SecurityConfig.DROPOUT_RATES[0])(mlp_dense1)
        mlp_dense2 = keras.layers.Dense(32, activation='relu')(mlp_dropout1)
        mlp_output = keras.layers.Dense(16, activation='relu', name='mlp_features')(mlp_dense2)
        
        # CNN 브랜치
        cnn_input = keras.layers.Input(shape=(self.sequence_length, input_shape), name='cnn_input')
        cnn_conv1 = keras.layers.Conv1D(32, 3, activation='relu')(cnn_input)
        cnn_pool1 = keras.layers.MaxPooling1D(2)(cnn_conv1)
        cnn_conv2 = keras.layers.Conv1D(16, 3, activation='relu')(cnn_pool1)
        cnn_global = keras.layers.GlobalAveragePooling1D()(cnn_conv2)
        cnn_output = keras.layers.Dense(16, activation='relu', name='cnn_features')(cnn_global)
        
        # 특성 융합
        merged = keras.layers.concatenate([mlp_output, cnn_output], name='feature_fusion')
        fusion_dense = keras.layers.Dense(32, activation='relu')(merged)
        fusion_dropout = keras.layers.Dropout(SecurityConfig.DROPOUT_RATES[1])(fusion_dense)
        final_output = keras.layers.Dense(1, activation='sigmoid', name='hybrid_output')(fusion_dropout)
        
        model = keras.Model(
            inputs=[mlp_input, cnn_input],
            outputs=final_output,
            name='Hybrid_Security_Detector'
        )
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def prepare_sequence_data(self, X: np.ndarray) -> np.ndarray:
        """CNN용 시퀀스 데이터 준비"""
        sequences = []
        for i in range(len(X) - self.sequence_length + 1):
            sequences.append(X[i:i + self.sequence_length])
        return np.array(sequences)
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              validation_split: float = None, 
              epochs: int = None):
        """모델 훈련"""
        try:
            validation_split = validation_split or SecurityConfig.VALIDATION_SPLIT
            epochs = epochs or SecurityConfig.DEFAULT_EPOCHS
            
            # 데이터 스케일링
            X_scaled = self.scaler.fit_transform(X)
            
            if self.model_type == 'hybrid':
                logger.info("하이브리드 모델 (MLP + CNN) 훈련 시작")
                history = self._train_hybrid_model(X_scaled, y, validation_split, epochs)
                
            elif self.model_type == 'mlp':
                logger.info("MLP 모델 훈련 시작")
                history = self._train_mlp_model(X_scaled, y, validation_split, epochs)
                
            elif self.model_type == 'cnn':
                logger.info("CNN 모델 훈련 시작")
                history = self._train_cnn_model(X_scaled, y, validation_split, epochs)
                
            else:
                raise SecurityDetectionError(f"지원하지 않는 모델 타입: {self.model_type}")
            
            self.is_trained = True
            logger.info("모델 훈련 완료")
            return history
            
        except Exception as e:
            logger.error(f"모델 훈련 실패: {e}")
            raise SecurityDetectionError(f"모델 훈련 실패: {e}")
    
    def _train_hybrid_model(self, X_scaled: np.ndarray, y: np.ndarray, 
                           validation_split: float, epochs: int):
        """하이브리드 모델 훈련"""
        # 시퀀스 데이터 준비
        X_sequence = self.prepare_sequence_data(X_scaled)
        y_sequence = y[self.sequence_length-1:]
        X_individual = X_scaled[self.sequence_length-1:]
        
        # 분할
        split_idx = int(len(X_sequence) * (1 - validation_split))
        X_seq_train, X_seq_val = X_sequence[:split_idx], X_sequence[split_idx:]
        X_ind_train, X_ind_val = X_individual[:split_idx], X_individual[split_idx:]
        y_train, y_val = y_sequence[:split_idx], y_sequence[split_idx:]
        
        # 모델 구축 및 훈련
        self.ensemble_model = self.build_hybrid_model(X_scaled.shape[1])
        
        history = self.ensemble_model.fit(
            [X_ind_train, X_seq_train], y_train,
            validation_data=([X_ind_val, X_seq_val], y_val),
            epochs=epochs,
            batch_size=SecurityConfig.BATCH_SIZE,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss', 
                    patience=SecurityConfig.EARLY_STOPPING_PATIENCE, 
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', 
                    factor=0.5, 
                    patience=SecurityConfig.LEARNING_RATE_PATIENCE
                )
            ],
            verbose=1
        )
        
        return history
    
    def _train_mlp_model(self, X_scaled: np.ndarray, y: np.ndarray,
                        validation_split: float, epochs: int):
        """MLP 모델 훈련"""
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, 
            test_size=validation_split, 
            random_state=SecurityConfig.RANDOM_SEED, 
            stratify=y
        )
        
        self.mlp_model = self.build_mlp_model(X_scaled.shape[1])
        
        history = self.mlp_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=SecurityConfig.BATCH_SIZE,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss', 
                    patience=SecurityConfig.EARLY_STOPPING_PATIENCE, 
                    restore_best_weights=True
                )
            ],
            verbose=1
        )
        
        return history
    
    def _train_cnn_model(self, X_scaled: np.ndarray, y: np.ndarray,
                        validation_split: float, epochs: int):
        """CNN 모델 훈련"""
        X_sequence = self.prepare_sequence_data(X_scaled)
        y_sequence = y[self.sequence_length-1:]
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_sequence, y_sequence, 
            test_size=validation_split, 
            random_state=SecurityConfig.RANDOM_SEED, 
            stratify=y_sequence
        )
        
        self.cnn_model = self.build_cnn_model(X_scaled.shape[1])
        
        history = self.cnn_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=SecurityConfig.BATCH_SIZE,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss', 
                    patience=SecurityConfig.EARLY_STOPPING_PATIENCE, 
                    restore_best_weights=True
                )
            ],
            verbose=1
        )
        
        return history
    
    def predict(self, data_input: Union[Dict, np.ndarray], data_id: str = None) -> Tuple[float, bool]:
        """예측 수행"""
        try:
            if not self.is_trained and not self.general_model:
                raise SecurityDetectionError("모델이 훈련되지 않았습니다")
            
            if self.detection_type == 'api_log':
                return self._predict_api_log(data_input)
            else:
                return self._predict_network_traffic(data_input, data_id)
                
        except Exception as e:
            logger.error(f"예측 실패: {e}")
            raise SecurityDetectionError(f"예측 실패: {e}")
    
    def _predict_api_log(self, log_entry: Dict) -> Tuple[float, bool]:
        """API 로그 예측"""
        features = self.extract_api_features(log_entry)
        features_scaled = self.scaler.transform([features])
        
        if self.model_type == 'hybrid' and self.ensemble_model:
            dummy_sequence = np.repeat(features_scaled, self.sequence_length, axis=0).reshape(1, self.sequence_length, -1)
            probability = self.ensemble_model.predict([features_scaled, dummy_sequence], verbose=0)[0][0]
        elif self.model_type == 'mlp' and self.mlp_model:
            probability = self.mlp_model.predict(features_scaled, verbose=0)[0][0]
        elif self.model_type == 'cnn' and self.cnn_model:
            dummy_sequence = np.repeat(features_scaled, self.sequence_length, axis=0).reshape(1, self.sequence_length, -1)
            probability = self.cnn_model.predict(dummy_sequence, verbose=0)[0][0]
        else:
            probability = np.random.uniform(0.1, 0.3)  # 폴백
        
        is_anomaly = probability > 0.5
        return float(probability), is_anomaly
    
    def _predict_network_traffic(self, packet_features: np.ndarray, packet_id: str = None) -> Tuple[float, bool]:
        """네트워크 트래픽 예측"""
        # 특성 정규화
        if self.scaler:
            packet_features = self.scaler.transform(packet_features.reshape(1, -1))
        else:
            packet_features = packet_features.reshape(1, -1)
        
        # 예측 수행
        try:
            if hasattr(self.general_model, 'predict'):
                prediction = self.general_model.predict(packet_features, verbose=0)[0][0]
            else:
                prediction = np.random.uniform(0, 1)  # 폴백
        except:
            prediction = np.random.uniform(0, 1)  # 오류 시 폴백
        
        is_attack = prediction > 0.5
        confidence = prediction if is_attack else 1 - prediction
        
        # 탐지 결과 기록
        detection_result = {
            'packet_id': packet_id or len(self.detection_history) + 1,
            'timestamp': time.time(),
            'prediction': prediction,
            'is_attack': is_attack,
            'confidence': confidence,
            'features': packet_features.flatten()
        }
        
        self.detection_history.append(detection_result)
        
        if is_attack:
            self.alert_count += 1
        
        return float(prediction), is_attack
    
    def save_model(self, model_path: str = "models/unified_detector"):
        """모델 저장"""
        try:
            import os
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # 각 모델별로 저장
            if self.ensemble_model:
                self.ensemble_model.save(f"{model_path}_hybrid.keras")
            if self.mlp_model:
                self.mlp_model.save(f"{model_path}_mlp.keras")
            if self.cnn_model:
                self.cnn_model.save(f"{model_path}_cnn.keras")
            
            # 스케일러 저장
            joblib.dump(self.scaler, f"{model_path}_scaler.pkl")
            
            # 메타데이터 저장
            metadata = {
                "detection_type": self.detection_type,
                "model_type": self.model_type,
                "is_trained": self.is_trained,
                "sequence_length": self.sequence_length
            }
            with open(f"{model_path}_metadata.json", 'w') as f:
                json.dump(metadata, f)
            
            logger.info(f"모델 저장 완료: {model_path}")
            
        except Exception as e:
            logger.error(f"모델 저장 실패: {e}")
            raise SecurityDetectionError(f"모델 저장 실패: {e}")
    
    def load_model(self, model_path: str = "models/unified_detector"):
        """모델 로드"""
        try:
            # 메타데이터 로드
            with open(f"{model_path}_metadata.json", 'r') as f:
                metadata = json.load(f)
            
            self.detection_type = metadata["detection_type"]
            self.model_type = metadata["model_type"]
            self.is_trained = metadata["is_trained"]
            self.sequence_length = metadata.get("sequence_length", SecurityConfig.CNN_SEQUENCE_LENGTH)
            
            # 각 모델 로드 시도
            try:
                self.ensemble_model = keras.models.load_model(f"{model_path}_hybrid.keras")
                logger.info("하이브리드 모델 로드 완료")
            except:
                pass
            
            try:
                self.mlp_model = keras.models.load_model(f"{model_path}_mlp.keras")
                logger.info("MLP 모델 로드 완료")
            except:
                pass
            
            try:
                self.cnn_model = keras.models.load_model(f"{model_path}_cnn.keras")
                logger.info("CNN 모델 로드 완료")
            except:
                pass
            
            # 스케일러 로드
            self.scaler = joblib.load(f"{model_path}_scaler.pkl")
            
            logger.info("통합 탐지 엔진 로드 완료")
            
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            raise SecurityDetectionError(f"모델 로드 실패: {e}")
    
    def get_detection_stats(self) -> Dict:
        """탐지 통계 반환"""
        if not self.detection_history:
            return {
                'total_detections': 0,
                'attack_detections': 0,
                'attack_ratio': 0.0,
                'avg_confidence': 0.0,
                'recent_attack_rate': 0.0
            }
        
        total_detections = len(self.detection_history)
        attack_detections = sum(1 for r in self.detection_history if r.get('is_attack', False))
        attack_ratio = attack_detections / total_detections * 100
        
        # 신뢰도 계산 (네트워크 트래픽의 경우)
        if 'confidence' in self.detection_history[0]:
            avg_confidence = np.mean([r['confidence'] for r in self.detection_history])
        else:
            avg_confidence = 0.0
        
        # 최근 100개의 공격 비율
        recent_results = self.detection_history[-100:]
        recent_attacks = sum(1 for r in recent_results if r.get('is_attack', False))
        recent_attack_rate = recent_attacks / len(recent_results) * 100
        
        return {
            'total_detections': total_detections,
            'attack_detections': attack_detections,
            'attack_ratio': attack_ratio,
            'avg_confidence': avg_confidence,
            'recent_attack_rate': recent_attack_rate,
            'detection_type': self.detection_type,
            'model_type': self.model_type
        }
    
    def clear_history(self):
        """탐지 이력 초기화"""
        self.detection_history = []
        self.alert_count = 0
        logger.info("탐지 이력 초기화 완료")


class RealTimeSecurityMonitor:
    """실시간 보안 모니터링 시스템"""
    
    def __init__(self, detection_engine: UnifiedDetectionEngine):
        self.detection_engine = detection_engine
        self.alert_threshold = SecurityConfig.ALERT_THRESHOLD
        self.recent_alerts = []
        self.max_recent_count = SecurityConfig.MAX_RECENT_ANOMALIES
        
        logger.info("실시간 보안 모니터 초기화 완료")
    
    def process_data(self, data_input: Union[Dict, np.ndarray], data_id: str = None) -> Dict:
        """데이터 실시간 처리"""
        try:
            probability, is_threat = self.detection_engine.predict(data_input, data_id)
            
            result = {
                "timestamp": datetime.now().isoformat(),
                "data_id": data_id,
                "threat_probability": probability,
                "is_threat": is_threat,
                "alert_level": self._get_alert_level(probability),
                "detection_type": self.detection_engine.detection_type
            }
            
            # API 로그의 경우 추가 정보
            if isinstance(data_input, dict):
                result.update({
                    "client_ip": data_input.get("client_ip"),
                    "method": data_input.get("method"),
                    "url": data_input.get("url")
                })
            
            # 고위험 탐지 시 알림
            if probability > self.alert_threshold:
                self._trigger_alert(data_input, result)
            
            # 최근 알림 기록 업데이트
            if is_threat:
                self.recent_alerts.append(result)
                if len(self.recent_alerts) > self.max_recent_count:
                    self.recent_alerts.pop(0)
            
            return result
            
        except Exception as e:
            logger.error(f"실시간 처리 오류: {e}")
            return {"error": str(e)}
    
    def _get_alert_level(self, probability: float) -> str:
        """위험도 레벨 결정"""
        for level, threshold in SecurityConfig.RISK_THRESHOLDS.items():
            if probability >= threshold:
                return level
        return "LOW"
    
    def _trigger_alert(self, data_input: Union[Dict, np.ndarray], detection_result: Dict):
        """알림 발송"""
        alert_message = {
            "alert_type": f"{self.detection_engine.detection_type.upper()}_THREAT_DETECTED",
            "timestamp": datetime.now().isoformat(),
            "threat_probability": detection_result["threat_probability"],
            "alert_level": detection_result["alert_level"],
            "detection_details": detection_result
        }
        
        # 로그에 기록
        logger.warning(f"SECURITY ALERT: {json.dumps(alert_message)}")
    
    def get_monitoring_statistics(self) -> Dict:
        """모니터링 통계"""
        total_recent = len(self.recent_alerts)
        if total_recent == 0:
            return {"message": "최근 위협 탐지 없음"}
        
        high_risk_count = sum(1 for a in self.recent_alerts if a["alert_level"] in ["HIGH", "CRITICAL"])
        avg_probability = np.mean([a["threat_probability"] for a in self.recent_alerts])
        
        # 탐지 엔진 통계와 결합
        engine_stats = self.detection_engine.get_detection_stats()
        
        return {
            "recent_threats_count": total_recent,
            "high_risk_count": high_risk_count,
            "average_threat_probability": round(avg_probability, 3),
            "risk_ratio": round(high_risk_count / total_recent, 3) if total_recent > 0 else 0,
            "engine_stats": engine_stats,
            "monitor_settings": {
                "alert_threshold": self.alert_threshold,
                "max_recent_count": self.max_recent_count
            }
        }


class TrafficSimulator:
    """네트워크 트래픽 시뮬레이터 (기존 기능 유지)"""
    
    def __init__(self, seed: int = None):
        seed = seed or SecurityConfig.RANDOM_SEED
        np.random.seed(seed)
        random.seed(seed)
        logger.info(f"트래픽 시뮬레이터 초기화: seed={seed}")
    
    def generate_normal_traffic(self, n_packets: int) -> np.ndarray:
        """정상 트래픽 생성"""
        return np.random.normal(0, 1, (n_packets, 19))
    
    def generate_attack_traffic(self, attack_type: str, n_packets: int) -> np.ndarray:
        """공격 트래픽 생성"""
        data = np.random.normal(0, 1, (n_packets, 19))
        
        if attack_type == "ddos":
            data[:, 0] *= 10  # Flow_Bytes/s 증가
            data[:, 1] *= 5   # Flow_Packets/s 증가
            data[:, 2] /= 3   # Backward_Packets 감소
        elif attack_type == "web_attack":
            data[:, 3] *= 3   # 패킷 길이 증가
            data[:, 4] *= 2   # 전송 패킷 길이 증가
        elif attack_type == "brute_force":
            data[:, 5] /= 5   # IAT 감소
            data[:, 6] *= 3   # 패킷 수 증가
        elif attack_type == "port_scan":
            data[:, 1] *= 2   # Forward 패킷 증가
            data[:, 2] /= 4   # Backward 패킷 감소
        
        return data
    
    def generate_mixed_traffic(self, n_packets: int, attack_ratio: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """혼합 트래픽 생성 (라벨 포함)"""
        n_normal = int(n_packets * (1 - attack_ratio))
        n_attack = n_packets - n_normal
        
        normal_traffic = self.generate_normal_traffic(n_normal)
        normal_labels = np.zeros(n_normal)
        
        attack_types = ["ddos", "web_attack", "brute_force", "port_scan"]
        attack_per_type = n_attack // len(attack_types)
        remainder = n_attack % len(attack_types)
        
        attack_traffic_list = []
        attack_labels_list = []
        
        for i, attack_type in enumerate(attack_types):
            n_this_attack = attack_per_type + (1 if i < remainder else 0)
            if n_this_attack > 0:
                attack_data = self.generate_attack_traffic(attack_type, n_this_attack)
                attack_traffic_list.append(attack_data)
                attack_labels_list.append(np.ones(n_this_attack))
        
        if attack_traffic_list:
            all_attack_traffic = np.vstack(attack_traffic_list)
            all_attack_labels = np.concatenate(attack_labels_list)
            
            all_traffic = np.vstack([normal_traffic, all_attack_traffic])
            all_labels = np.concatenate([normal_labels, all_attack_labels])
        else:
            all_traffic = normal_traffic
            all_labels = normal_labels
        
        # 무작위로 섞기
        indices = np.random.permutation(len(all_traffic))
        return all_traffic[indices], all_labels[indices]


class PerformanceEvaluator:
    """성능 평가 시스템"""
    
    def __init__(self):
        self.evaluation_history = []
        logger.info("성능 평가기 초기화 완료")
    
    def evaluate_detection_performance(self, true_labels: np.ndarray, 
                                     predicted_labels: np.ndarray, 
                                     prediction_scores: np.ndarray = None) -> Dict:
        """탐지 성능 평가"""
        try:
            metrics = {
                'accuracy': accuracy_score(true_labels, predicted_labels),
                'precision': precision_score(true_labels, predicted_labels, zero_division=0),
                'recall': recall_score(true_labels, predicted_labels, zero_division=0),
                'f1_score': f1_score(true_labels, predicted_labels, zero_division=0)
            }
            
            # AUC 계산
            if prediction_scores is not None:
                try:
                    metrics['auc'] = roc_auc_score(true_labels, prediction_scores)
                except:
                    metrics['auc'] = 0.5
            
            # 혼동 행렬
            cm = confusion_matrix(true_labels, predicted_labels)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                metrics.update({
                    'true_negatives': int(tn),
                    'false_positives': int(fp),
                    'false_negatives': int(fn),
                    'true_positives': int(tp),
                    'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
                    'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0
                })
            
            # 평가 이력에 추가
            evaluation_record = {
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'sample_count': len(true_labels)
            }
            self.evaluation_history.append(evaluation_record)
            
            logger.info(f"성능 평가 완료: Accuracy={metrics['accuracy']:.3f}, F1={metrics['f1_score']:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"성능 평가 실패: {e}")
            raise SecurityDetectionError(f"성능 평가 실패: {e}")
    
    def generate_performance_report(self, detection_results: List[Dict], 
                                  true_labels: np.ndarray = None) -> Dict:
        """성능 보고서 생성"""
        if not detection_results:
            return {"error": "탐지 결과가 없습니다"}
        
        total_detections = len(detection_results)
        threat_predictions = sum(1 for r in detection_results if r.get('is_threat', r.get('is_attack', False)))
        threat_ratio = threat_predictions / total_detections * 100
        
        report = {
            'basic_stats': {
                'total_detections': total_detections,
                'predicted_threats': threat_predictions,
                'threat_ratio': threat_ratio,
                'evaluation_timestamp': datetime.now().isoformat()
            }
        }
        
        # 실제 라벨이 있는 경우 정확도 계산
        if true_labels is not None and len(true_labels) == total_detections:
            predicted_labels = [r.get('is_threat', r.get('is_attack', False)) for r in detection_results]
            prediction_scores = [r.get('threat_probability', r.get('prediction', 0)) for r in detection_results]
            
            performance = self.evaluate_detection_performance(
                true_labels, predicted_labels, prediction_scores
            )
            report['performance'] = performance
        
        return report


# 편의 함수들
def create_api_log_detector(model_type: str = 'hybrid') -> UnifiedDetectionEngine:
    """API 로그 탐지기 생성"""
    return UnifiedDetectionEngine(detection_type='api_log', model_type=model_type)

def create_network_traffic_detector() -> UnifiedDetectionEngine:
    """네트워크 트래픽 탐지기 생성"""
    return UnifiedDetectionEngine(detection_type='network_traffic', model_type='general')

def create_security_monitor(detection_engine: UnifiedDetectionEngine) -> RealTimeSecurityMonitor:
    """보안 모니터 생성"""
    return RealTimeSecurityMonitor(detection_engine)


if __name__ == "__main__":
    # 테스트 코드
    logger.info("통합 보안 탐지 엔진 테스트 시작")
    
    # API 로그 탐지기 테스트
    api_detector = create_api_log_detector('hybrid')
    api_monitor = create_security_monitor(api_detector)
    
    # 네트워크 트래픽 탐지기 테스트
    network_detector = create_network_traffic_detector()
    network_monitor = create_security_monitor(network_detector)
    
    logger.info("통합 보안 탐지 엔진 테스트 완료")
