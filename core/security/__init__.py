"""
보안 모듈 - CICIDS2017 네트워크 이상 탐지

이 패키지는 네트워크 보안 이상 탐지를 위한 모듈들을 포함합니다:
- data_loader: CICIDS2017 데이터 로딩 및 전처리 (백업에서 복원 완료)
- model_builder: 딥러닝 모델 구축
- attack_detector: 실시간 공격 탐지 (백업에서 복원 완료)

참고: detection_engine은 복잡성으로 인해 임시로 비활성화됨
"""

# ✅ 데이터 로더 - 백업에서 복원 완료 (Safe Import)
try:
    from data.loaders.unified_security_loader import (
        CICIDSDataLoader,
        check_cicids_data_availability,
        generate_cicids_sample_data,
        generate_enhanced_sample_data
    )
except ImportError as e:
    print(f"Warning: Security data loader import failed: {e}")
    # Fallback: 빈 클래스들로 대체
    class CICIDSDataLoader:
        def __init__(self, *args, **kwargs):
            pass
    def check_cicids_data_availability():
        return {"available": False, "error": "Module not available"}
    def generate_cicids_sample_data(*args, **kwargs):
        return None
    def generate_enhanced_sample_data(*args, **kwargs):
        return None

# ✅ 모델 빌더 - 기존 유지 (우수한 상태) (Safe Import)
try:
    from .model_builder import (
        SecurityModelBuilder,
        AttackPatternAnalyzer,
        check_tensorflow_availability,
        install_tensorflow
    )
except ImportError as e:
    print(f"Warning: Model builder import failed: {e}")
    class SecurityModelBuilder:
        def __init__(self, *args, **kwargs):
            pass
    class AttackPatternAnalyzer:
        def __init__(self, *args, **kwargs):
            pass
    def check_tensorflow_availability():
        return False
    def install_tensorflow():
        return False

# ✅ 공격 탐지기 - 백업에서 복원 완료 (Safe Import)
try:
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
except ImportError as e:
    print(f"Warning: Attack detector import failed: {e}")
    class RealTimeAttackDetector:
        def __init__(self, *args, **kwargs):
            pass
    class TrafficSimulator:
        def __init__(self, *args, **kwargs):
            pass
    class PerformanceEvaluator:
        def __init__(self, *args, **kwargs):
            pass
    class AlertManager:
        def __init__(self, *args, **kwargs):
            pass
    class DetectionOrchestrator:
        def __init__(self, *args, **kwargs):
            pass
    def create_detection_system(*args, **kwargs):
        return None
    def run_quick_simulation(*args, **kwargs):
        return None
    def evaluate_attack_detection(*args, **kwargs):
        return None

# ✅ detection_engine - 고도화된 통합 탐지 엔진 복원 완료 (Safe Import)
try:
    from .detection_engine import (
        UnifiedDetectionEngine,
        RealTimeSecurityMonitor,
        TrafficSimulator as EnhancedTrafficSimulator,
        PerformanceEvaluator as EnhancedPerformanceEvaluator,
        create_api_log_detector,
        create_network_traffic_detector,
        create_security_monitor
    )
    DETECTION_ENGINE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Detection engine import failed: {e}")
    class UnifiedDetectionEngine:
        def __init__(self, *args, **kwargs):
            pass
    class RealTimeSecurityMonitor:
        def __init__(self, *args, **kwargs):
            pass
    class EnhancedTrafficSimulator:
        def __init__(self, *args, **kwargs):
            pass
    class EnhancedPerformanceEvaluator:
        def __init__(self, *args, **kwargs):
            pass
    def create_api_log_detector(*args, **kwargs):
        return None
    def create_network_traffic_detector(*args, **kwargs):
        return None
    def create_security_monitor(*args, **kwargs):
        return None
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

    # 기본 공격 탐지 (호환성 유지)
    'RealTimeAttackDetector',
    'TrafficSimulator',          # 기본 버전
    'PerformanceEvaluator',      # 기본 버전
    'AlertManager',
    'DetectionOrchestrator',
    'create_detection_system',
    'run_quick_simulation',
    'evaluate_attack_detection',

    # 🆕 통합 탐지 엔진 (고도화 버전 - 권장)
    'UnifiedDetectionEngine',
    'RealTimeSecurityMonitor',
    'EnhancedTrafficSimulator',   # 고도화 버전
    'EnhancedPerformanceEvaluator', # 고도화 버전
    'create_api_log_detector',
    'create_network_traffic_detector',
    'create_security_monitor'
]

__version__ = "1.0.0"
__author__ = "Customer Segmentation Project"
__description__ = "CICIDS2017 네트워크 이상 탐지 모듈 (통합 탐지 엔진 포함)"
