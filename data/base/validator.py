"""
데이터 검증 클래스

모든 도메인에서 공통으로 사용하는 데이터 검증 로직
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from utils.exceptions import DataProcessingError
from config.logging import setup_logger

logger = setup_logger(__name__)


class DataValidator:
    """데이터 검증 클래스"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: List[str] = None) -> bool:
        """기본 데이터프레임 검증"""
        if df is None or df.empty:
            raise DataProcessingError("데이터프레임이 비어있습니다.")
        
        if required_columns:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise DataProcessingError(f"필수 컬럼이 없습니다: {missing_cols}")
        
        return True
    
    @staticmethod
    def check_missing_values(df: pd.DataFrame, threshold: float = 0.5) -> Dict:
        """결측값 검사"""
        missing_info = {}
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_ratio = missing_count / len(df)
            
            missing_info[col] = {
                'count': missing_count,
                'ratio': missing_ratio,
                'critical': missing_ratio > threshold
            }
        
        return missing_info
    
    @staticmethod
    def detect_outliers(df: pd.DataFrame, method: str = 'iqr') -> Dict:
        """이상치 탐지"""
        outliers_info = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                
                outliers_info[col] = {
                    'count': len(outliers),
                    'ratio': len(outliers) / len(df),
                    'bounds': {'lower': lower_bound, 'upper': upper_bound}
                }
            
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = df[z_scores > 3][col]
                
                outliers_info[col] = {
                    'count': len(outliers),
                    'ratio': len(outliers) / len(df),
                    'threshold': 3
                }
        
        return outliers_info
