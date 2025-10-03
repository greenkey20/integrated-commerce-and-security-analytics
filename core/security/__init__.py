"""
보안 모듈 - CICIDS2017 네트워크 이상 탐지

이 패키지는 네트워크 보안 이상 탐지를 위한 모듈들을 포함합니다.
상위 패키지 임포트 시 무거운 서브모듈(TensorFlow 등)이 자동으로 로드되지 않도록
심볼 접근 시점에 지연 로딩(lazy import)을 사용합니다.
"""

import importlib
import warnings

# 심볼 -> 실제 모듈 경로 매핑
_ATTR_MODULE_MAP = {
    # data loader
    'CICIDSDataLoader': 'data.loaders.unified_security_loader',
    'check_cicids_data_availability': 'data.loaders.unified_security_loader',
    'generate_cicids_sample_data': 'data.loaders.unified_security_loader',
    'generate_enhanced_sample_data': 'data.loaders.unified_security_loader',

    # model builder (무거울 수 있음 - 지연 로딩)
    'SecurityModelBuilder': 'core.security.model_builder',
    'AttackPatternAnalyzer': 'core.security.model_builder',
    'check_tensorflow_availability': 'core.security.model_builder',
    'install_tensorflow': 'core.security.model_builder',

    # attack detector
    'RealTimeAttackDetector': 'core.security.attack_detector',
    'TrafficSimulator': 'core.security.attack_detector',
    'PerformanceEvaluator': 'core.security.attack_detector',
    'AlertManager': 'core.security.attack_detector',
    'DetectionOrchestrator': 'core.security.attack_detector',
    'create_detection_system': 'core.security.attack_detector',
    'run_quick_simulation': 'core.security.attack_detector',
    'evaluate_attack_detection': 'core.security.attack_detector',

    # detection engine (고도화)
    'UnifiedDetectionEngine': 'core.security.detection_engine',
    'RealTimeSecurityMonitor': 'core.security.detection_engine',
    'EnhancedTrafficSimulator': 'core.security.detection_engine',
    'EnhancedPerformanceEvaluator': 'core.security.detection_engine',
    'create_api_log_detector': 'core.security.detection_engine',
    'create_network_traffic_detector': 'core.security.detection_engine',
    'create_security_monitor': 'core.security.detection_engine',
}


__all__ = list(_ATTR_MODULE_MAP.keys())

__version__ = "1.0.0"
__author__ = "Customer Segmentation Project"
__description__ = "CICIDS2017 네트워크 이상 탐지 모듈 (지연 로딩 지원)"


def _make_stub(name, exc):
    """심볼 접근 시 import 실패하면 호출되는 스텁 함수/클래스 반환."""
    def _stub(*args, **kwargs):
        raise ImportError(f"심볼 '{name}'을(를) 로드할 수 없습니다: {exc}")
    _stub.__name__ = name
    return _stub


def __getattr__(name: str):
    """요청된 심볼을 실제 모듈에서 동적으로 로드해서 반환합니다.
    실패 시 경고 후, 호출 시 ImportError를 발생시키는 스텁을 반환합니다.
    """
    if name not in _ATTR_MODULE_MAP:
        raise AttributeError(f"module {__name__} has no attribute {name}")

    module_path = _ATTR_MODULE_MAP[name]
    try:
        module = importlib.import_module(module_path)
        attr = getattr(module, name)
        # 캐싱: 다음 접근부터는 재import하지 않도록 globals에 저장
        globals()[name] = attr
        return attr
    except Exception as e:
        warnings.warn(f"Lazy import failed for {name} from {module_path}: {e}", ImportWarning)
        stub = _make_stub(name, e)
        globals()[name] = stub
        return stub


def __dir__():
    return sorted(list(globals().keys()) + __all__)
