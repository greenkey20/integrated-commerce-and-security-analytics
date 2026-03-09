"""
Text 패키지 초기화 - 지연 로딩 지원

이 파일은 `core.text`에서 제공하는 핵심 심볼만 노출합니다.
심볼 접근 시점에 관련 모듈을 import 하도록 하여, 상위 패키지 import 시
불필요한 무거운 의존성(예: TensorFlow)이 자동으로 로드되지 않도록 합니다.
"""

import importlib
import warnings

_ATTR_MODULE_MAP = {
    'TextAnalyticsModels': 'core.text.sentiment_models',
}

__all__ = list(_ATTR_MODULE_MAP.keys())

__version__ = "1.0.0"
__author__ = "Customer Segmentation Project"
__description__ = "텍스트 분석 관련 모듈 (지연 로딩 지원)"


def __getattr__(name: str):
    if name not in _ATTR_MODULE_MAP:
        raise AttributeError(f"module {__name__} has no attribute {name}")
    module_path = _ATTR_MODULE_MAP[name]
    try:
        module = importlib.import_module(module_path)
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    except Exception as e:
        warnings.warn(f"Lazy import failed for {name} from {module_path}: {e}", ImportWarning)
        def _stub(*args, **kwargs):
            raise ImportError(f"심볼 '{name}'을(를) 로드할 수 없습니다: {e}")
        globals()[name] = _stub
        return _stub


def __dir__():
    return sorted(list(globals().keys()) + __all__)
