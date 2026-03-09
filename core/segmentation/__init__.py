"""
Segmentation 패키지 초기화 - 지연 로딩 지원
"""

import importlib
import warnings

_ATTR_MODULE_MAP = {
    # clustering
    'ClusterAnalyzer': 'core.segmentation.clustering',
    'cluster_analyzer': 'core.segmentation.clustering',
    'perform_clustering': 'core.segmentation.clustering',
    'find_optimal_clusters': 'core.segmentation.clustering',
    'analyze_cluster_characteristics': 'core.segmentation.clustering',
    'generate_dynamic_colors': 'core.segmentation.clustering',
    'generate_dynamic_interpretation_guide': 'core.segmentation.clustering',
    'get_dynamic_marketing_strategy': 'core.segmentation.clustering',

    # models
    'DeepLearningModels': 'core.segmentation.models',
    'TENSORFLOW_AVAILABLE': 'core.segmentation.models',

    # hyperparameter tuning
    'HyperparameterTuner': 'core.segmentation.hyperparameter_tuning',
}

__all__ = list(_ATTR_MODULE_MAP.keys())

__version__ = "1.0.0"
__author__ = "Customer Segmentation Project"
__description__ = "고객 세분화(세그멘테이션) 관련 모듈 (지연 로딩 지원)"


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
