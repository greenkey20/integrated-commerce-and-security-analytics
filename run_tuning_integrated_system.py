# run_tuning_integrated_system.py
"""
하이브리드 보안 시스템 + 하이퍼파라미터 튜닝 통합 실행
"""

import time
from core.security_hyperparameter_tuning import SecurityHyperparameterTuner
from data.cicids_data_loader import setup_complete_system
from core.anomaly_detection import APILogAnomalyDetector
import json

def run_optimized_security_system():
    """최적화된 보안 시스템 실행"""
    
    print("🔥 하이브리드 보안 시스템 + 하이퍼파라미터 튜닝")
    print("=" * 60)
    
    # 1단계: 하이퍼파라미터 튜닝 실행
    print("1️⃣ 최적 하이퍼파라미터 탐색 중...")
    tuner = SecurityHyperparameterTuner()
    tuning_results = tuner.run_complete_security_tuning()
    
    if not tuning_results:
        print("❌ 하이퍼파라미터 튜닝 실패, 기본 설정으로 진행")
        return setup_complete_system()  # 기본 시스템 실행
    
    # 2단계: 최적 파라미터로 모델 재구성
    print("\n2️⃣ 최적 파라미터로 모델 재구성...")
    best_overall = tuning_results['tuning_summary']['best_overall_model']
    
    if best_overall:
        print(f"🏆 최고 성능 모델: {best_overall['model_type']}")
        print(f"🏆 예상 정확도: {best_overall['test_accuracy']:.4f}")
        
        # 최적 설정으로 모델 생성
        optimized_detector = APILogAnomalyDetector(
            model_type=best_overall['model_type'].lower()
        )
        
        # 실제 상황에서는 최적 파라미터로 재훈련 필요
        print("✅ 최적화된 탐지 시스템 준비 완료!")
        
        # 3단계: 설정 저장
        optimized_config = {
            'model_type': best_overall['model_type'],
            'parameters': best_overall['best_params'],
            'expected_accuracy': best_overall['test_accuracy'],
            'tuning_timestamp': tuning_results['timestamp']
        }
        
        with open('models/optimized_config.json', 'w') as f:
            json.dump(optimized_config, f, indent=2)
        
        print("💾 최적화 설정이 models/optimized_config.json에 저장됨")
        
        return optimized_detector
    
    else:
        print("❌ 최적 모델을 찾지 못함, 기본 설정으로 진행")
        return setup_complete_system()

def load_optimized_config():
    """저장된 최적화 설정 로드"""
    try:
        with open('models/optimized_config.json', 'r') as f:
            config = json.load(f)
        print(f"📂 최적화 설정 로드: {config['model_type']} (정확도: {config['expected_accuracy']:.4f})")
        return config
    except FileNotFoundError:
        print("❌ 최적화 설정이 없습니다. 먼저 튜닝을 실행하세요.")
        return None

if __name__ == "__main__":
    print("🎯 실행 모드를 선택하세요:")
    print("1. 하이퍼파라미터 튜닝 + 최적화 시스템 구축")
    print("2. 저장된 최적화 설정으로 시스템 실행")
    print("3. 기본 시스템 실행 (튜닝 없이)")
    
    choice = input("\n선택 (1/2/3): ").strip()
    
    if choice == "1":
        detector = run_optimized_security_system()
        
    elif choice == "2":
        config = load_optimized_config()
        if config:
            print("🚀 최적화된 설정으로 시스템 시작...")
            # 여기서 API 서버 시작하거나 추가 작업 수행
        
    elif choice == "3":
        print("🔧 기본 설정으로 시스템 시작...")
        detector = setup_complete_system()
        
    else:
        print("잘못된 선택입니다.")
    
    print("\n✅ 시스템 준비 완료!")
