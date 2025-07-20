"""
보안 모듈 - CICIDS2017 네트워크 이상 탐지

이 패키지는 네트워크 보안 이상 탐지를 위한 모듈들을 포함합니다:
- data_loader: 데이터 로딩 및 전처리
- model_builder: 딥러닝 모델 구축
- attack_detector: 실시간 공격 탐지
"""

from .data_loader import (
    CICIDSDataLoader,
    check_cicids_data_availability,
    generate_cicids_sample_data,
    generate_enhanced_sample_data
)

from .model_builder import (
    SecurityModelBuilder,
    AttackPatternAnalyzer,
    check_tensorflow_availability,
    install_tensorflow
)

from .attack_detector import (
    RealTimeAttackDetector,
    TrafficSimulator,
    PerformanceEvaluator,
    AlertManager,
    DetectionOrchestrator,
    create_detection_system,
    run_quick_simulation,
    evaluate_attack_detection
)

__all__ = [
    # 데이터 로딩
    'CICIDSDataLoader',
    'check_cicids_data_availability',
    'generate_cicids_sample_data',
    'generate_enhanced_sample_data',
    
    # 모델 구축
    'SecurityModelBuilder',
    'AttackPatternAnalyzer',
    'check_tensorflow_availability',
    'install_tensorflow',
    
    # 공격 탐지
    'RealTimeAttackDetector',
    'TrafficSimulator',
    'PerformanceEvaluator',
    'AlertManager',
    'DetectionOrchestrator',
    'create_detection_system',
    'run_quick_simulation',
    'evaluate_attack_detection'
]

__version__ = "1.0.0"
__author__ = "Customer Segmentation Project"
__description__ = "CICIDS2017 네트워크 이상 탐지 모듈"
