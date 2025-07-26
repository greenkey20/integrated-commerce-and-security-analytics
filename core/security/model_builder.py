"""
CICIDS2017 보안 탐지를 위한 딥러닝 모델 구축 모듈

하이브리드, MLP, CNN, 오토인코더 등 다양한 모델 아키텍처 제공
화면 코드와 분리된 순수 모델링 로직
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import warnings
warnings.filterwarnings('ignore')

# TensorFlow 관련 import (조건부)
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
    TF_VERSION = tf.__version__
except ImportError:
    TENSORFLOW_AVAILABLE = False
    TF_VERSION = None


class SecurityModelBuilder:
    """보안 탐지를 위한 딥러닝 모델 구축 클래스"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.model_type = None
        
    def diagnose_data_quality(self, X, feature_names=None):
        """데이터 품질 진단"""
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        diagnosis = {
            'total_samples': X.shape[0],
            'total_features': X.shape[1],
            'inf_count': np.sum(np.isinf(X)),
            'nan_count': np.sum(np.isnan(X)),
            'problematic_features': []
        }
        
        # 특성별 문제 확인
        for i in range(X.shape[1]):
            feature_name = feature_names[i] if feature_names and i < len(feature_names) else f'Feature_{i}'
            
            inf_count = np.sum(np.isinf(X[:, i]))
            nan_count = np.sum(np.isnan(X[:, i]))
            max_val = np.max(X[np.isfinite(X[:, i]), i]) if np.any(np.isfinite(X[:, i])) else 0
            min_val = np.min(X[np.isfinite(X[:, i]), i]) if np.any(np.isfinite(X[:, i])) else 0
            
            if inf_count > 0 or nan_count > 0 or abs(max_val) > 1e10 or abs(min_val) > 1e10:
                diagnosis['problematic_features'].append({
                    'name': feature_name,
                    'inf_count': inf_count,
                    'nan_count': nan_count,
                    'max_val': max_val,
                    'min_val': min_val
                })
        
        return diagnosis
    
    def clean_data(self, X):
        """데이터 정제: 무한대, NaN, 극대값 처리"""
        # NumPy 배열로 변환
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        # 1. 무한대 값 처리
        X = np.where(np.isinf(X), 0, X)
        
        # 2. NaN 값 처리
        X = np.where(np.isnan(X), 0, X)
        
        # 3. 극대값 클리핑 (float64 범위 내로)
        max_val = np.finfo(np.float64).max / 1000  # 안전 마진
        min_val = np.finfo(np.float64).min / 1000
        
        X = np.clip(X, min_val, max_val)
        
        # 4. 데이터 타입 확인
        X = X.astype(np.float64)
        
        return X
    
    def prepare_data(self, X, y, test_size=0.2, binary_classification=True):
        """데이터 전처리 및 분할"""
        # 데이터 정제 (무한대, NaN, 극대값 처리)
        X_cleaned = self.clean_data(X)
        
        # 데이터 품질 검증
        invalid_count = np.sum(np.isinf(X_cleaned)) + np.sum(np.isnan(X_cleaned))
        if invalid_count > 0:
            print(f"⚠️ 여전히 {invalid_count}개의 비정상 값이 있습니다. 0으로 대체합니다.")
            X_cleaned = np.where(np.isinf(X_cleaned) | np.isnan(X_cleaned), 0, X_cleaned)
        
        # 특성 정규화
        X_scaled = self.scaler.fit_transform(X_cleaned)
        
        if binary_classification:
            # 이진 분류: BENIGN=0, 공격=1
            y_binary = (y != 'BENIGN').astype(int) if isinstance(y[0], str) else y
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_binary, test_size=test_size, random_state=42, stratify=y_binary
            )
            return X_train, X_test, y_train, y_test
        else:
            # 다중 분류
            y_encoded = self.label_encoder.fit_transform(y)
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
            )
            return X_train, X_test, y_train, y_test
    
    def build_hybrid_model(self, input_dim, sequence_length=10):
        """하이브리드 모델 (MLP + CNN) 구축"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow가 필요합니다. pip install tensorflow로 설치하세요.")
        
        # MLP 브랜치
        mlp_input = keras.layers.Input(shape=(input_dim,), name='mlp_input')
        mlp_dense1 = keras.layers.Dense(128, activation='relu')(mlp_input)
        mlp_dropout1 = keras.layers.Dropout(0.3)(mlp_dense1)
        mlp_dense2 = keras.layers.Dense(64, activation='relu')(mlp_dropout1)
        mlp_features = keras.layers.Dense(32, activation='relu', name='mlp_features')(mlp_dense2)
        
        # CNN 브랜치
        cnn_input = keras.layers.Input(shape=(sequence_length, input_dim), name='cnn_input')
        cnn_conv1 = keras.layers.Conv1D(64, 3, activation='relu')(cnn_input)
        cnn_pool1 = keras.layers.MaxPooling1D(2)(cnn_conv1)
        cnn_conv2 = keras.layers.Conv1D(32, 3, activation='relu')(cnn_pool1)
        cnn_global = keras.layers.GlobalAveragePooling1D()(cnn_conv2)
        cnn_features = keras.layers.Dense(32, activation='relu', name='cnn_features')(cnn_global)
        
        # 특성 융합
        merged = keras.layers.concatenate([mlp_features, cnn_features])
        fusion_dense = keras.layers.Dense(64, activation='relu')(merged)
        fusion_dropout = keras.layers.Dropout(0.2)(fusion_dense)
        output = keras.layers.Dense(1, activation='sigmoid')(fusion_dropout)
        
        model = keras.Model(inputs=[mlp_input, cnn_input], outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        self.model = model
        self.model_type = 'hybrid'
        return model
    
    def build_mlp_model(self, input_dim, n_classes=1, classification_type='binary'):
        """MLP 분류 모델 구축"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow가 필요합니다. pip install tensorflow로 설치하세요.")
        
        if classification_type == 'binary':
            loss = 'binary_crossentropy'
            activation = 'sigmoid'
            n_classes = 1
        else:
            loss = 'categorical_crossentropy'
            activation = 'softmax'
        
        model = keras.Sequential([
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(n_classes, activation=activation)
        ])
        
        model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
        
        self.model = model
        self.model_type = 'mlp'
        return model
    
    def build_cnn_model(self, input_dim, sequence_length=10):
        """CNN 시계열 모델 구축"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow가 필요합니다. pip install tensorflow로 설치하세요.")
        
        model = keras.Sequential([
            keras.layers.Input(shape=(sequence_length, input_dim)),
            keras.layers.Conv1D(64, 3, activation='relu'),
            keras.layers.MaxPooling1D(2),
            keras.layers.Conv1D(32, 3, activation='relu'),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(50, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        self.model = model
        self.model_type = 'cnn'
        return model
    
    def build_autoencoder_model(self, input_dim, encoding_dim=20):
        """오토인코더 이상 탐지 모델 구축"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow가 필요합니다. pip install tensorflow로 설치하세요.")
        
        # 인코더
        input_layer = keras.layers.Input(shape=(input_dim,))
        encoded = keras.layers.Dense(128, activation='relu')(input_layer)
        encoded = keras.layers.Dense(64, activation='relu')(encoded)
        encoded = keras.layers.Dense(encoding_dim, activation='relu')(encoded)
        
        # 디코더
        decoded = keras.layers.Dense(64, activation='relu')(encoded)
        decoded = keras.layers.Dense(128, activation='relu')(decoded)
        decoded = keras.layers.Dense(input_dim, activation='linear')(decoded)
        
        autoencoder = keras.Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        self.model = autoencoder
        self.model_type = 'autoencoder'
        return autoencoder
    
    def create_sequences(self, data, sequence_length):
        """시계열 시퀀스 데이터 생성"""
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i:i + sequence_length])
        return np.array(sequences)
    
    def train_model(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=64, verbose=1, custom_callbacks=None):
        """모델 훈련"""
        if self.model is None:
            raise ValueError("먼저 모델을 구축해야 합니다.")
        
        # 기본 콜백
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        ]
        
        # 커스텀 콜백 추가
        if custom_callbacks:
            callbacks.extend(custom_callbacks)
        
        if self.model_type == 'hybrid':
            # 하이브리드 모델의 경우 시퀀스 데이터 준비
            sequence_length = 10
            X_train_seq = self.create_sequences(X_train, sequence_length)
            X_train_ind = X_train[sequence_length-1:]
            y_train_seq = y_train[sequence_length-1:]
            
            if X_val is not None:
                X_val_seq = self.create_sequences(X_val, sequence_length)
                X_val_ind = X_val[sequence_length-1:]
                y_val_seq = y_val[sequence_length-1:]
                validation_data = ([X_val_ind, X_val_seq], y_val_seq)
            else:
                validation_data = None
            
            history = self.model.fit(
                [X_train_ind, X_train_seq], y_train_seq,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=verbose
            )
            
        elif self.model_type == 'cnn':
            # CNN 모델의 경우 시퀀스 데이터 준비
            sequence_length = 10
            X_train_seq = self.create_sequences(X_train, sequence_length)
            y_train_seq = y_train[sequence_length-1:]
            
            if X_val is not None:
                X_val_seq = self.create_sequences(X_val, sequence_length)
                y_val_seq = y_val[sequence_length-1:]
                validation_data = (X_val_seq, y_val_seq)
            else:
                validation_data = None
            
            history = self.model.fit(
                X_train_seq, y_train_seq,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=verbose
            )
            
        elif self.model_type == 'autoencoder':
            # 오토인코더는 정상 데이터만 사용
            X_train_normal = X_train[y_train == 0]
            
            history = self.model.fit(
                X_train_normal, X_train_normal,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=verbose
            )
            
        else:
            # MLP 등 일반 모델
            validation_data = (X_val, y_val) if X_val is not None else None
            
            history = self.model.fit(
                X_train, y_train,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=verbose
            )
        
        return history
    
    def predict(self, X_test):
        """예측 수행"""
        if self.model is None:
            raise ValueError("모델이 훈련되지 않았습니다.")
        
        if self.model_type == 'hybrid':
            sequence_length = 10
            X_test_seq = self.create_sequences(X_test, sequence_length)
            X_test_ind = X_test[sequence_length-1:]
            return self.model.predict([X_test_ind, X_test_seq])
            
        elif self.model_type == 'cnn':
            sequence_length = 10
            X_test_seq = self.create_sequences(X_test, sequence_length)
            return self.model.predict(X_test_seq)
            
        else:
            return self.model.predict(X_test)
    
    def evaluate_binary_model(self, X_test, y_test):
        """이진 분류 모델 평가"""
        y_pred = self.predict(X_test)
        
        if self.model_type in ['hybrid', 'cnn']:
            # 시퀀스 모델의 경우 라벨 조정
            sequence_length = 10
            y_test = y_test[sequence_length-1:]
        
        if self.model_type == 'autoencoder':
            # 오토인코더의 경우 재구성 오차 기반 평가
            train_pred = self.model.predict(X_test)
            mse = np.mean(np.square(X_test - train_pred), axis=1)
            # 임계값 설정 (정상 데이터의 95 퍼센타일)
            threshold = np.percentile(mse, 95)
            y_pred_binary = (mse > threshold).astype(int)
        else:
            y_pred_binary = (y_pred > 0.5).astype(int).flatten()
        
        # 성능 메트릭 계산
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred_binary),
            'precision': precision_score(y_test, y_pred_binary, zero_division=0),
            'recall': recall_score(y_test, y_pred_binary, zero_division=0),
            'f1_score': f1_score(y_test, y_pred_binary, zero_division=0)
        }
        
        # ROC AUC (오토인코더 제외)
        if self.model_type != 'autoencoder':
            fpr, tpr, _ = roc_curve(y_test, y_pred.flatten())
            metrics['auc'] = auc(fpr, tpr)
            metrics['roc_data'] = {'fpr': fpr, 'tpr': tpr}
        
        return metrics
    
    def evaluate_multiclass_model(self, X_test, y_test):
        """다중 분류 모델 평가"""
        y_pred = self.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1) if len(y_test.shape) > 1 else y_test
        
        # 혼동 행렬
        cm = confusion_matrix(y_test_classes, y_pred_classes)
        
        # 성능 메트릭
        metrics = {
            'accuracy': accuracy_score(y_test_classes, y_pred_classes),
            'confusion_matrix': cm,
            'classification_report': classification_report(
                y_test_classes, y_pred_classes, 
                target_names=self.label_encoder.classes_,
                output_dict=True
            )
        }
        
        return metrics
    
    def cross_validate(self, X, y, cv_folds=5, model_params=None):
        """교차검증 수행"""
        if model_params is None:
            model_params = {}
        
        kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            # 폴드별 데이터 분할
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]
            
            # 데이터 정규화
            scaler = StandardScaler()
            X_fold_train_scaled = scaler.fit_transform(X_fold_train)
            X_fold_val_scaled = scaler.transform(X_fold_val)
            
            # 모델 생성 및 훈련
            if self.model_type == 'mlp':
                fold_model = self.build_mlp_model(X.shape[1], **model_params)
            elif self.model_type == 'hybrid':
                fold_model = self.build_hybrid_model(X.shape[1], **model_params)
            elif self.model_type == 'cnn':
                fold_model = self.build_cnn_model(X.shape[1], **model_params)
            else:
                raise ValueError(f"교차검증을 지원하지 않는 모델 타입: {self.model_type}")
            
            # 훈련
            fold_model.fit(
                X_fold_train_scaled, y_fold_train,
                validation_data=(X_fold_val_scaled, y_fold_val),
                epochs=50,
                batch_size=128,
                callbacks=[keras.callbacks.EarlyStopping(patience=5)],
                verbose=0
            )
            
            # 평가
            val_loss, val_acc = fold_model.evaluate(X_fold_val_scaled, y_fold_val, verbose=0)
            cv_scores.append(val_acc)
        
        return {
            'scores': cv_scores,
            'mean': np.mean(cv_scores),
            'std': np.std(cv_scores),
            'confidence_interval': 1.96 * np.std(cv_scores) / np.sqrt(len(cv_scores))
        }
    
    def save_model(self, filepath):
        """모델 저장"""
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다.")
        
        self.model.save(filepath)
    
    def load_model(self, filepath):
        """모델 로드"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow가 필요합니다.")
        
        self.model = keras.models.load_model(filepath)
        return self.model


class AttackPatternAnalyzer:
    """공격 패턴 분석 클래스"""
    
    def __init__(self):
        pass
    
    def analyze_feature_importance(self, data, feature_names):
        """특성 중요도 분석"""
        normal_data = data[data['Label'] == 'BENIGN']
        attack_data = data[data['Label'] != 'BENIGN']
        
        feature_comparison = []
        for feature in feature_names:
            if feature != 'Label':
                normal_mean = normal_data[feature].mean()
                attack_mean = attack_data[feature].mean()
                
                if normal_mean != 0:
                    ratio = attack_mean / normal_mean
                    difference = abs(attack_mean - normal_mean)
                    
                    feature_comparison.append({
                        'feature': feature,
                        'normal_mean': normal_mean,
                        'attack_mean': attack_mean,
                        'ratio': ratio,
                        'absolute_difference': difference
                    })
        
        # 비율 기준으로 정렬
        feature_comparison.sort(key=lambda x: x['ratio'], reverse=True)
        return feature_comparison
    
    def analyze_attack_type_patterns(self, data, attack_type):
        """특정 공격 유형의 패턴 분석"""
        normal_data = data[data['Label'] == 'BENIGN']
        attack_data = data[data['Label'] == attack_type]
        
        if len(attack_data) == 0:
            return None
        
        numeric_features = data.select_dtypes(include=[np.number]).columns
        patterns = {}
        
        for feature in numeric_features:
            patterns[feature] = {
                'normal_stats': {
                    'mean': normal_data[feature].mean(),
                    'std': normal_data[feature].std(),
                    'median': normal_data[feature].median()
                },
                'attack_stats': {
                    'mean': attack_data[feature].mean(),
                    'std': attack_data[feature].std(),
                    'median': attack_data[feature].median()
                }
            }
        
        return patterns


# 편의 함수들
def check_tensorflow_availability():
    """TensorFlow 사용 가능 여부 확인"""
    return TENSORFLOW_AVAILABLE, TF_VERSION

def install_tensorflow():
    """TensorFlow 자동 설치 시도"""
    try:
        import subprocess
        import sys
        
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', 'tensorflow'
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            # 설치 후 재import 시도
            global TENSORFLOW_AVAILABLE, TF_VERSION
            try:
                import tensorflow as tf
                from tensorflow import keras
                TENSORFLOW_AVAILABLE = True
                TF_VERSION = tf.__version__
                return True, f"TensorFlow {TF_VERSION} 설치 성공"
            except ImportError:
                return False, "설치는 성공했지만 import 실패"
        else:
            return False, f"설치 실패: {result.stderr}"
            
    except Exception as e:
        return False, f"설치 중 오류: {str(e)}"
