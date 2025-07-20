"""
임시 RetailDataProcessor 클래스

실제 구현은 data/base/ 모듈들을 사용해야 함
"""

import pandas as pd
import warnings

class RetailDataProcessor:
    """임시 데이터 프로세서 클래스"""
    
    def __init__(self, column_mapping=None):
        self.column_mapping = column_mapping or {}
        warnings.warn("이것은 임시 클래스입니다. 실제로는 data/base/ 모듈을 사용해야 합니다.", UserWarning)
    
    def clean_data(self, data):
        """기본 데이터 정제"""
        return data.copy()
    
    def validate_data_quality(self, data):
        """기본 데이터 품질 검증"""
        return {
            'records_count': len(data),
            'status': 'basic_validation_completed'
        }
