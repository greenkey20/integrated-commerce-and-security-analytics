#!/usr/bin/env python3
"""
🧠 Customer Segmentation - 통합 모델 훈련 스크립트
모든 머신러닝 모델을 훈련하고 저장하는 통합 스크립트
"""

import sys
import os
from pathlib import Path
import argparse
import json
import logging
from datetime import datetime

# 프로젝트 루트 설정
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_directories():
    """필요한 디렉토리 생성"""
    directories = [
        PROJECT_ROOT / "models" / "saved_models",
        PROJECT_ROOT / "models" / "checkpoints",
        PROJECT_ROOT / "data" / "processed",
        PROJECT_ROOT / "logs"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"디렉토리 생성/확인: {directory}")

def check_dependencies():
    """필수 의존성 확인"""
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'tensorflow', 
        'plotly', 'streamlit', 'fastapi', 'uvicorn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"누락된 패키지: {', '.join(missing_packages)}")
        print(f"❌ 다음 패키지들을 설치해주세요: {' '.join(missing_packages)}")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    logger.info("✅ 모든 의존성 확인 완료")
    return True

def load_data():
    """데이터 로드"""
    try:
        import pandas as pd
        
        # 데이터 파일 경로
        data_path = PROJECT_ROOT / "data" / "Mall_Customers.csv"
        
        if not data_path.exists():
            # GitHub에서 데이터 다운로드
            logger.info("데이터 파일이 없습니다. GitHub에서 다운로드 중...")
            url = "https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Datasets/Mall_Customers.csv"
            data = pd.read_csv(url)
            
            # 데이터 저장
            data.to_csv(data_path, index=False)
            logger.info(f"데이터 다운로드 완료: {data_path}")
        else:
            data = pd.read_csv(data_path)
            logger.info(f"데이터 로드 완료: {len(data)}개 샘플")
        
        return data
    
    except Exception as e:
        logger.error(f"데이터 로드 실패: {e}")
        return None

def train_clustering_model(data, n_clusters=5):
    """클러스터링 모델 훈련"""
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import silhouette_score
        import joblib
        
        logger.info("🎯 클러스터링 모델 훈련 시작...")
        
        # 특성 선택 및 정규화
        features = data[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # K-means 클러스터링
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_features)
        
        # 성능 평가
        silhouette_avg = silhouette_score(scaled_features, cluster_labels)
        
        # 모델 저장
        model_dir = PROJECT_ROOT / "models" / "saved_models"
        joblib.dump(kmeans, model_dir / "kmeans_model.pkl")
        joblib.dump(scaler, model_dir / "scaler.pkl")
        
        # 메타데이터 저장
        metadata = {
            "model_type": "kmeans",
            "n_clusters": n_clusters,
            "silhouette_score": float(silhouette_avg),
            "training_date": datetime.now().isoformat(),
            "n_samples": len(data),
            "feature_names": features.columns.tolist()
        }
        
        with open(model_dir / "clustering_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ 클러스터링 모델 훈련 완료 (실루엣 점수: {silhouette_avg:.3f})")
        return True
        
    except Exception as e:
        logger.error(f"클러스터링 모델 훈련 실패: {e}")
        return False

def train_deep_learning_model(data, n_clusters=5):
    """딥러닝 모델 훈련"""
    try:
        import tensorflow as tf
        from tensorflow import keras
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.cluster import KMeans
        import joblib
        
        logger.info("🧠 딥러닝 모델 훈련 시작...")
        
        # 데이터 준비
        features = data[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # 클러스터링으로 라벨 생성
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_features)
        
        # 훈련/테스트 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_features, cluster_labels, test_size=0.2, 
            random_state=42, stratify=cluster_labels
        )
        
        # 모델 구성
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(3,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(n_clusters, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 모델 훈련
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=10, restore_best_weights=True
                )
            ]
        )
        
        # 모델 평가
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        # 모델 저장
        model_dir = PROJECT_ROOT / "models" / "saved_models"
        model.save(model_dir / "deep_learning_model.h5")
        
        # 메타데이터 저장
        metadata = {
            "model_type": "deep_learning",
            "n_clusters": n_clusters,
            "test_accuracy": float(test_accuracy),
            "test_loss": float(test_loss),
            "training_date": datetime.now().isoformat(),
            "n_samples": len(data),
            "epochs_trained": len(history.history['loss']),
            "architecture": {
                "layers": ["Dense(64, relu)", "Dropout(0.2)", "Dense(32, relu)", "Dropout(0.1)", f"Dense({n_clusters}, softmax)"],
                "optimizer": "adam",
                "loss": "sparse_categorical_crossentropy"
            }
        }
        
        with open(model_dir / "deep_learning_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ 딥러닝 모델 훈련 완료 (정확도: {test_accuracy:.3f})")
        return True
        
    except Exception as e:
        logger.error(f"딥러닝 모델 훈련 실패: {e}")
        return False

def create_dummy_models():
    """더미 모델 생성 (API 테스트용)"""
    try:
        import json
        
        logger.info("🔧 더미 모델 파일 생성 중...")
        
        model_dir = PROJECT_ROOT / "models" / "saved_models"
        
        # 더미 메타데이터들
        dummy_files = {
            "hybrid_detector_metadata.json": {
                "model_type": "hybrid",
                "version": "1.0.0",
                "training_date": datetime.now().isoformat(),
                "sequence_length": 10,
                "feature_columns": ["requests_per_minute", "request_size", "content_length", "processing_time"],
                "is_trained": True
            },
            "hybrid_detector_scaler.json": {
                "mean_": [10.5, 1024.0, 512.0, 0.15],
                "scale_": [5.2, 2048.0, 1024.0, 0.08],
                "n_features_in_": 4
            },
            "hybrid_detector_weights.json": {
                "mlp_weights": "dummy_weights_placeholder",
                "cnn_weights": "dummy_weights_placeholder", 
                "ensemble_weights": "dummy_weights_placeholder",
                "note": "이것은 테스트용 더미 데이터입니다."
            }
        }
        
        for filename, content in dummy_files.items():
            with open(model_dir / filename, "w") as f:
                json.dump(content, f, indent=2)
        
        logger.info("✅ 더미 모델 파일 생성 완료")
        return True
        
    except Exception as e:
        logger.error(f"더미 모델 생성 실패: {e}")
        return False

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Customer Segmentation 모델 훈련 스크립트")
    parser.add_argument("--mode", choices=["all", "clustering", "deep_learning", "dummy"], 
                       default="all", help="훈련할 모델 타입")
    parser.add_argument("--clusters", type=int, default=5, help="클러스터 개수")
    parser.add_argument("--skip-deps", action="store_true", help="의존성 체크 건너뛰기")
    
    args = parser.parse_args()
    
    print("🧠 Customer Segmentation 모델 훈련 시작")
    print(f"📊 훈련 모드: {args.mode}")
    print(f"🎯 클러스터 개수: {args.clusters}")
    print("-" * 50)
    
    # 의존성 확인
    if not args.skip_deps and not check_dependencies():
        sys.exit(1)
    
    # 디렉토리 설정
    setup_directories()
    
    # 데이터 로드
    data = load_data()
    if data is None:
        logger.error("데이터 로드 실패")
        sys.exit(1)
    
    # 모델 훈련 실행
    success = True
    
    if args.mode in ["all", "clustering"]:
        success &= train_clustering_model(data, args.clusters)
    
    if args.mode in ["all", "deep_learning"]:
        success &= train_deep_learning_model(data, args.clusters)
    
    if args.mode in ["all", "dummy"]:
        success &= create_dummy_models()
    
    # 결과 출력
    if success:
        print("\n" + "="*50)
        print("🎉 모델 훈련 성공적으로 완료!")
        print(f"📁 모델 저장 위치: {PROJECT_ROOT / 'models' / 'saved_models'}")
        print("\n💡 다음 단계:")
        print("1. 웹 앱 실행: streamlit run main_app.py")
        print("2. API 서버 실행: python api_server.py")
        print("="*50)
    else:
        print("\n❌ 모델 훈련 중 오류가 발생했습니다.")
        print("로그를 확인하여 문제를 해결하세요.")
        sys.exit(1)

if __name__ == "__main__":
    main()
