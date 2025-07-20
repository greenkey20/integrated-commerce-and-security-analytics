"""
공통 예외 처리 클래스

프로젝트 전반에서 사용할 커스텀 예외들
"""


class CustomerSegmentationError(Exception):
    """고객 세분화 프로젝트 기본 예외"""
    pass


class DataLoadingError(CustomerSegmentationError):
    """데이터 로딩 관련 예외"""
    pass


class DataProcessingError(CustomerSegmentationError):
    """데이터 처리 관련 예외"""
    pass


class ModelTrainingError(CustomerSegmentationError):
    """모델 훈련 관련 예외"""
    pass


class SecurityDetectionError(CustomerSegmentationError):
    """보안 탐지 관련 예외"""
    pass


class ConfigurationError(CustomerSegmentationError):
    """설정 관련 예외"""
    pass


class VisualizationError(CustomerSegmentationError):
    """시각화 관련 예외"""
    pass
