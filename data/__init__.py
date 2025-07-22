"""
데이터 모듈 - 최소 구조

Phase 1-2 완료 후 필요시 기능 추가
"""

# 기본 클래스들만 expose (직접 import 없음)
__all__ = [
    'DataValidator',
    'DataCleaner', 
    'FeatureEngineer',
    'DataSplitter',
    'DataProcessor'
]

# 사용 시점에 import하도록 유도
def get_data_validator():
    from data.base.validator import DataValidator
    return DataValidator

def get_data_cleaner():
    from data.base.cleaner import DataCleaner
    return DataCleaner

def get_feature_engineer():
    from data.base.engineer import FeatureEngineer
    return FeatureEngineer

def get_data_splitter():
    from data.base.splitter import DataSplitter
    return DataSplitter

def get_data_processor():
    from data.processors.segmentation_data_processor import DataProcessor
    return DataProcessor

# 편의 함수들
DataValidator = get_data_validator
DataCleaner = get_data_cleaner
FeatureEngineer = get_feature_engineer
DataSplitter = get_data_splitter
DataProcessor = get_data_processor
