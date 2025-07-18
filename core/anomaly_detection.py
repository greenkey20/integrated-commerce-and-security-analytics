# core/anomaly_detection.py
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import json
from datetime import datetime, timedelta
import re
from typing import Dict, List, Tuple
import logging

# 07-2.ipynb 스타일로 랜덤 시드 설정
keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

class APILogAnomalyDetector:
    """API 로그 이상 탐지 모델 (MLP 기반)"""
    
    def __init__(self, model_type: str = 'mlp'):
        self.model_type = model_type
        self.mlp_model = None
        self.cnn_model = None
        self.ensemble_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.is_trained = False
        self.sequence_length = 10  # CNN용 시퀀스 길이
        
    def extract_features_from_log(self, log_entry: Dict) -> np.ndarray:
        """로그 엔트리에서 특성 추출"""
        features = {}
        
        # 시간 기반 특성
        timestamp = datetime.fromisoformat(log_entry.get('timestamp', datetime.now().isoformat()))
        features['hour'] = timestamp.hour
        features['day_of_week'] = timestamp.weekday()
        features['is_weekend'] = 1 if timestamp.weekday() >= 5 else 0
        features['is_business_hour'] = 1 if 9 <= timestamp.hour <= 17 else 0
        
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
        
        # IP 기반 특성 (간단화)
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
        
        # 응답 시간 (있는 경우)
        features['response_time'] = log_entry.get('processing_time', 0)
        
        return np.array(list(features.values()))
    
    def prepare_training_data(self, log_file_path: str = None, cicids_data: pd.DataFrame = None) -> Tuple[np.ndarray, np.ndarray]:
        """훈련 데이터 준비"""
        
        if cicids_data is not None:
            # CICIDS2017 데이터 활용
            X, y = self.prepare_cicids_data(cicids_data)
        else:
            # 자체 로그 파일 활용
            X, y = self.prepare_log_data(log_file_path)
        
        return X, y
    
    def prepare_cicids_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """CICIDS2017 데이터 전처리"""
        # CICIDS2017의 주요 특성 선택
        feature_columns = [
            'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
            'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
            'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
            'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
            'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std',
            'Fwd IAT Total', 'Fwd IAT Mean', 'Bwd IAT Total', 'Bwd IAT Mean'
        ]
        
        # 사용 가능한 컬럼만 선택
        available_columns = [col for col in feature_columns if col in df.columns]
        X = df[available_columns].fillna(0)
        
        # 라벨 처리 (BENIGN = 0, 공격 = 1)
        y = (df['Label'] != 'BENIGN').astype(int)
        
        # 무한대 값 처리
        X = X.replace([np.inf, -np.inf], 0)
        
        return X.values, y.values
    
    def prepare_log_data(self, log_file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """자체 로그 데이터 전처리"""
        logs = []
        labels = []
        
        with open(log_file_path, 'r') as f:
            for line in f:
                try:
                    log_entry = json.loads(line.strip())
                    features = self.extract_features_from_log(log_entry)
                    
                    # 라벨 결정 (휴리스틱 기반)
                    is_suspicious = log_entry.get('is_suspicious', False)
                    
                    logs.append(features)
                    labels.append(1 if is_suspicious else 0)
                except:
                    continue
        
        return np.array(logs), np.array(labels)
    
    def build_mlp_model(self, input_shape: int) -> keras.Model:
        """07-2.ipynb 스타일의 MLP 모델 구축"""
        
        model = keras.Sequential([
            keras.layers.Input(shape=(input_shape,)),
            keras.layers.Dense(128, activation='relu', name='hidden_layer_1'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu', name='hidden_layer_2'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu', name='hidden_layer_3'),
            keras.layers.Dense(1, activation='sigmoid', name='output_layer')
        ], name='MLP_Anomaly_Detector')
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def build_cnn_model(self, input_shape: int) -> keras.Model:
        """08-2.ipynb 스타일의 CNN 모델 구축 (시계열 패턴 용)"""
        
        model = keras.Sequential([
            keras.layers.Input(shape=(self.sequence_length, input_shape)),
            keras.layers.Conv1D(64, 3, activation='relu'),
            keras.layers.MaxPooling1D(2),
            keras.layers.Conv1D(32, 3, activation='relu'),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(50, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(1, activation='sigmoid')
        ], name='CNN_Anomaly_Detector')
        
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
        mlp_dropout1 = keras.layers.Dropout(0.3)(mlp_dense1)
        mlp_dense2 = keras.layers.Dense(32, activation='relu')(mlp_dropout1)
        mlp_output = keras.layers.Dense(16, activation='relu', name='mlp_features')(mlp_dense2)
        
        # CNN 브랜치 (시계열)
        cnn_input = keras.layers.Input(shape=(self.sequence_length, input_shape), name='cnn_input')
        cnn_conv1 = keras.layers.Conv1D(32, 3, activation='relu')(cnn_input)
        cnn_pool1 = keras.layers.MaxPooling1D(2)(cnn_conv1)
        cnn_conv2 = keras.layers.Conv1D(16, 3, activation='relu')(cnn_pool1)
        cnn_global = keras.layers.GlobalAveragePooling1D()(cnn_conv2)
        cnn_output = keras.layers.Dense(16, activation='relu', name='cnn_features')(cnn_global)
        
        # 특성 융합
        merged = keras.layers.concatenate([mlp_output, cnn_output], name='feature_fusion')
        fusion_dense = keras.layers.Dense(32, activation='relu')(merged)
        fusion_dropout = keras.layers.Dropout(0.2)(fusion_dense)
        final_output = keras.layers.Dense(1, activation='sigmoid', name='hybrid_output')(fusion_dropout)
        
        model = keras.Model(
            inputs=[mlp_input, cnn_input],
            outputs=final_output,
            name='Hybrid_Anomaly_Detector'
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
    
    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2, epochs: int = 10):
        """하이브리드 모델 훈련"""
        
        # 데이터 스케일링
        X_scaled = self.scaler.fit_transform(X)
        
        if self.model_type == 'hybrid':
            # 하이브리드 모델 훈련
            print("🔥 하이브리드 모델 (MLP + CNN) 훈련 시작")
            
            # CNN용 시퀀스 데이터 준비
            X_sequence = self.prepare_sequence_data(X_scaled)
            y_sequence = y[self.sequence_length-1:]  # 시퀀스에 맞게 라벨 조정
            
            # MLP용 개별 데이터 (시퀀스 마지막 값들)
            X_individual = X_scaled[self.sequence_length-1:]
            
            # 훈련/검증 분할
            split_idx = int(len(X_sequence) * (1 - validation_split))
            
            X_seq_train, X_seq_val = X_sequence[:split_idx], X_sequence[split_idx:]
            X_ind_train, X_ind_val = X_individual[:split_idx], X_individual[split_idx:]
            y_train, y_val = y_sequence[:split_idx], y_sequence[split_idx:]
            
            # 하이브리드 모델 구축
            self.ensemble_model = self.build_hybrid_model(X.shape[1])
            
            # 훈련 실행
            history = self.ensemble_model.fit(
                [X_ind_train, X_seq_train], y_train,
                validation_data=([X_ind_val, X_seq_val], y_val),
                epochs=epochs,
                batch_size=32,
                callbacks=[
                    keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
                    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
                ],
                verbose=1
            )
            
        elif self.model_type == 'mlp':
            # MLP 모델만 훈련
            print("✅ MLP 모델 훈련 시작")
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y, test_size=validation_split, random_state=42, stratify=y
            )
            
            self.mlp_model = self.build_mlp_model(X.shape[1])
            
            history = self.mlp_model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=32,
                callbacks=[
                    keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
                ],
                verbose=1
            )
        
        elif self.model_type == 'cnn':
            # CNN 모델만 훈련
            print("📊 CNN 모델 훈련 시작")
            X_sequence = self.prepare_sequence_data(X_scaled)
            y_sequence = y[self.sequence_length-1:]
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_sequence, y_sequence, test_size=validation_split, random_state=42, stratify=y_sequence
            )
            
            self.cnn_model = self.build_cnn_model(X.shape[1])
            
            history = self.cnn_model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=32,
                callbacks=[
                    keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
                ],
                verbose=1
            )
        
        self.is_trained = True
        return history
    
    def predict(self, log_entry: Dict) -> Tuple[float, bool]:
        """단일 로그 엔트리에 대한 예측"""
        if not self.is_trained:
            raise ValueError("모델이 훈련되지 않았습니다.")
        
        features = self.extract_features_from_log(log_entry)
        features_scaled = self.scaler.transform([features])
        
        if self.model_type == 'hybrid' and self.ensemble_model:
            # 하이브리드 예측 (더미 시퀀스 사용)
            dummy_sequence = np.repeat(features_scaled, self.sequence_length, axis=0).reshape(1, self.sequence_length, -1)
            probability = self.ensemble_model.predict([features_scaled, dummy_sequence], verbose=0)[0][0]
        elif self.model_type == 'mlp' and self.mlp_model:
            probability = self.mlp_model.predict(features_scaled, verbose=0)[0][0]
        elif self.model_type == 'cnn' and self.cnn_model:
            dummy_sequence = np.repeat(features_scaled, self.sequence_length, axis=0).reshape(1, self.sequence_length, -1)
            probability = self.cnn_model.predict(dummy_sequence, verbose=0)[0][0]
        else:
            # 폴백: 랜덤 예측
            probability = np.random.uniform(0.1, 0.3)
        
        is_anomaly = probability > 0.5
        return float(probability), is_anomaly
    
    def save_model(self, model_path: str = "models/hybrid_detector"):
        """모델 저장"""
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
            "model_type": self.model_type,
            "is_trained": self.is_trained,
            "sequence_length": self.sequence_length
        }
        with open(f"{model_path}_metadata.json", 'w') as f:
            json.dump(metadata, f)
    
    def load_model(self, model_path: str = "models/hybrid_detector"):
        """모델 로드"""
        try:
            # 메타데이터 로드
            with open(f"{model_path}_metadata.json", 'r') as f:
                metadata = json.load(f)
            
            self.model_type = metadata["model_type"]
            self.is_trained = metadata["is_trained"]
            self.sequence_length = metadata.get("sequence_length", 10)
            
            # 각 모델 로드 시도
            try:
                self.ensemble_model = keras.models.load_model(f"{model_path}_hybrid.keras")
                print("✅ 하이브리드 모델 로드 완료")
            except:
                pass
            
            try:
                self.mlp_model = keras.models.load_model(f"{model_path}_mlp.keras")
                print("✅ MLP 모델 로드 완료")
            except:
                pass
            
            try:
                self.cnn_model = keras.models.load_model(f"{model_path}_cnn.keras")
                print("✅ CNN 모델 로드 완료")
            except:
                pass
            
            # 스케일러 로드
            self.scaler = joblib.load(f"{model_path}_scaler.pkl")
            
            print("🔥 모델 로드 성공")
        except Exception as e:
            print(f"모델 로드 실패: {e}")

# 실시간 모니터링 클래스
class RealTimeAnomalyMonitor:
    """실시간 이상 탐지 모니터"""
    
    def __init__(self, detector: APILogAnomalyDetector):
        self.detector = detector
        self.alert_threshold = 0.7
        self.recent_anomalies = []
        self.max_recent_count = 100
        
    def process_log_entry(self, log_entry: Dict) -> Dict:
        """로그 엔트리 실시간 처리"""
        try:
            probability, is_anomaly = self.detector.predict(log_entry)
            
            result = {
                "timestamp": log_entry.get("timestamp"),
                "client_ip": log_entry.get("client_ip"),
                "anomaly_probability": probability,
                "is_anomaly": is_anomaly,
                "alert_level": self.get_alert_level(probability)
            }
            
            # 고위험 이상 탐지 시 알림
            if probability > self.alert_threshold:
                self.trigger_alert(log_entry, result)
            
            # 최근 이상 기록 업데이트
            if is_anomaly:
                self.recent_anomalies.append(result)
                if len(self.recent_anomalies) > self.max_recent_count:
                    self.recent_anomalies.pop(0)
            
            return result
            
        except Exception as e:
            logging.error(f"이상 탐지 처리 오류: {e}")
            return {"error": str(e)}
    
    def get_alert_level(self, probability: float) -> str:
        """위험도 레벨 결정"""
        if probability >= 0.9:
            return "CRITICAL"
        elif probability >= 0.7:
            return "HIGH"
        elif probability >= 0.5:
            return "MEDIUM"
        else:
            return "LOW"
    
    def trigger_alert(self, log_entry: Dict, detection_result: Dict):
        """알림 발송"""
        alert_message = {
            "alert_type": "API_ANOMALY_DETECTED",
            "timestamp": datetime.now().isoformat(),
            "client_ip": log_entry.get("client_ip"),
            "anomaly_probability": detection_result["anomaly_probability"],
            "alert_level": detection_result["alert_level"],
            "details": {
                "user_agent": log_entry.get("user_agent"),
                "url": log_entry.get("url"),
                "method": log_entry.get("method"),
                "requests_per_minute": log_entry.get("requests_per_minute")
            }
        }
        
        # 로그에 기록
        logging.warning(f"ANOMALY ALERT: {json.dumps(alert_message)}")
    
    def get_statistics(self) -> Dict:
        """모니터링 통계"""
        total_recent = len(self.recent_anomalies)
        if total_recent == 0:
            return {"message": "최근 이상 탐지 없음"}
        
        high_risk_count = sum(1 for a in self.recent_anomalies if a["alert_level"] in ["HIGH", "CRITICAL"])
        avg_probability = np.mean([a["anomaly_probability"] for a in self.recent_anomalies])
        
        return {
            "recent_anomalies_count": total_recent,
            "high_risk_count": high_risk_count,
            "average_anomaly_probability": round(avg_probability, 3),
            "risk_ratio": round(high_risk_count / total_recent, 3) if total_recent > 0 else 0
        }
    
    def get_advanced_statistics(self) -> Dict:
        """고급 통계 (하이브리드 시스템용)"""
        stats = self.get_statistics()
        
        # 고위험 IP 추출
        high_risk_ips = []
        for anomaly in self.recent_anomalies:
            if anomaly.get("alert_level") in ["HIGH", "CRITICAL"]:
                ip = anomaly.get("client_ip")
                if ip and ip not in high_risk_ips:
                    high_risk_ips.append(ip)
        
        stats.update({
            "total_anomalies": len(self.recent_anomalies),
            "high_risk_ips": high_risk_ips[:10],  # 최대 10개만
            "detection_model": self.detector.model_type,
            "model_trained": self.detector.is_trained
        })
        
        return stats

if __name__ == "__main__":
    # 테스트 코드
    detector = APILogAnomalyDetector(model_type='hybrid')
    monitor = RealTimeAnomalyMonitor(detector)
    
    # 테스트 로그
    test_log = {
        "timestamp": datetime.now().isoformat(),
        "client_ip": "192.168.1.100",
        "method": "POST",
        "url": "/api/v1/customer/segment",
        "user_agent": "Mozilla/5.0",
        "requests_per_minute": 5,
        "request_size": 256,
        "content_length": 128,
        "processing_time": 0.15
    }
    
    print("하이브리드 이상 탐지 시스템 테스트 완료!")
