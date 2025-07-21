"""
보안 모듈 - CICIDS2017 네트워크 이상 탐지

이 패키지는 네트워크 보안 이상 탐지를 위한 모듈들을 포함합니다:
- model_builder: 딥러닝 모델 구축
- attack_detector: 실시간 공격 탐지
- detection_engine: 통합 탐지 엔진

참고: data_loader는 data.loaders.security_loader로 마이그레이션됨
"""

# 임시로 data_loader import 비활성화 (Phase 1-2 완료 후 수정)
# from .data_loader import (
#     CICIDSDataLoader,
#     check_cicids_data_availability,
#     generate_cicids_sample_data,
#     generate_enhanced_sample_data
# )

from .model_builder import (
    SecurityModelBuilder,
    AttackPatternAnalyzer,
    check_tensorflow_availability,
    install_tensorflow
)

# 임시로 detection_engine import 완전 비활성화 (Phase 1-2 완료 후 수정)
# try:
#     from .detection_engine import (
#         UnifiedDetectionEngine,
#         RealTimeSecurityMonitor,
#         TrafficSimulator,
#         PerformanceEvaluator
#     )
#     DETECTION_ENGINE_AVAILABLE = True
# except ImportError:
#     DETECTION_ENGINE_AVAILABLE = False

DETECTION_ENGINE_AVAILABLE = False

# 임시로 attack_detector import 비활성화
# from .attack_detector import (
#     RealTimeAttackDetector,
#     TrafficSimulator,
#     PerformanceEvaluator,
#     AlertManager,
#     DetectionOrchestrator,
#     create_detection_system,
#     run_quick_simulation,
#     evaluate_attack_detection
# )

__all__ = [
    # 데이터 로딩 (임시 비활성화)
    # 'CICIDSDataLoader',
    # 'check_cicids_data_availability',
    # 'generate_cicids_sample_data', 
    # 'generate_enhanced_sample_data',
    
    # 모델 구축
    'SecurityModelBuilder',
    'AttackPatternAnalyzer',
    'check_tensorflow_availability',
    'install_tensorflow',
]

# detection_engine이 사용 가능한 경우 추가
if DETECTION_ENGINE_AVAILABLE:
    __all__.extend([
        'UnifiedDetectionEngine',
        'RealTimeSecurityMonitor', 
        'TrafficSimulator',
        'PerformanceEvaluator'
    ])

__version__ = "1.0.0"
__author__ = "Customer Segmentation Project"
__description__ = "CICIDS2017 네트워크 이상 탐지 모듈 (Phase 1-2 리팩토링 진행 중)"
