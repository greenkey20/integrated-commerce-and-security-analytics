"""
보안 모듈 - CICIDS2017 네트워크 이상 탐지

이 패키지는 네트워크 보안 이상 탐지를 위한 모듈들을 포함합니다:
- data_loader: CICIDS2017 데이터 로딩 및 전처리 (백업에서 복원 완료)
- model_builder: 딥러닝 모델 구축
- attack_detector: 실시간 공격 탐지 (백업에서 복원 완료)

참고: detection_engine은 복잡성으로 인해 임시로 비활성화됨
"""

# ✅ 데이터 로더 - 백업에서 복원 완료
from .data_loader import (
    CICIDSDataLoader,
    check_cicids_data_availability,
    generate_cicids_sample_data,
    generate_enhanced_sample_data
)

# ✅ 모델 빌더 - 기존 유지 (우수한 상태)
from .model_builder import (
    SecurityModelBuilder,
    AttackPatternAnalyzer,
    check_tensorflow_availability,
    install_tensorflow
)

# ✅ 공격 탐지기 - 백업에서 복원 완료
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

# ❌ detection_engine - 복잡성으로 인해 임시 비활성화
# 복잡한 detection_engine.py는 detection_engine_backup.py로 백업됨
DETECTION_ENGINE_AVAILABLE = False

__all__ = [
    # 데이터 로딩 (✅ 복원 완료)
    'CICIDSDataLoader',
    'check_cicids_data_availability',
    'generate_cicids_sample_data',
    'generate_enhanced_sample_data',

    # 모델 구축 (✅ 기존 유지)
    'SecurityModelBuilder',
    'AttackPatternAnalyzer',
    'check_tensorflow_availability',
    'install_tensorflow',

    # 실시간 공격 탐지 (✅ 복원 완료)
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
__description__ = "CICIDS2017 네트워크 이상 탐지 모듈 (보안 기능 복원 완료)"