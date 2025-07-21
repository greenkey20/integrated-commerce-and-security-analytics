"""
데이터 정제 클래스

모든 도메인에서 공통으로 사용하는 데이터 정제 로직
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils.exceptions import DataProcessingError
from config.logging import setup_logger

logger = setup_logger(__name__)


class DataCleaner:
    """데이터 정제 클래스"""
    
    @staticmethod
    def handle_missing_values(df: pd.DataFrame, 
                            strategy: Dict[str, str] = None) -> pd.DataFrame:
        """결측값 처리"""
        df_clean = df.copy()
        
        if strategy is None:
            # 기본 전략: 수치형은 중앙값, 범주형은 최빈값
            strategy = {}
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    strategy[col] = 'median'
                else:
                    strategy[col] = 'mode'
        
        for col, method in strategy.items():
            if col not in df.columns:
                continue
                
            if method == 'mean':
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
            elif method == 'median':
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            elif method == 'mode':
                mode_value = df_clean[col].mode()
                if len(mode_value) > 0:
                    df_clean[col].fillna(mode_value.iloc[0], inplace=True)
            elif method == 'drop':
                df_clean.dropna(subset=[col], inplace=True)
            elif isinstance(method, (str, int, float)):
                df_clean[col].fillna(method, inplace=True)
        
        logger.info(f"결측값 처리 완료: {len(df)} → {len(df_clean)} 레코드")
        return df_clean
    
    @staticmethod
    def remove_outliers(df: pd.DataFrame, 
                       columns: List[str] = None,
                       method: str = 'iqr',
                       threshold: float = 1.5) -> pd.DataFrame:
        """이상치 제거"""
        df_clean = df.copy()
        
        if columns is None:
            columns = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if col not in df_clean.columns:
                continue
                
            if method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                before_count = len(df_clean)
                df_clean = df_clean[
                    (df_clean[col] >= lower_bound) & 
                    (df_clean[col] <= upper_bound)
                ]
                after_count = len(df_clean)
                
                logger.info(f"{col} 이상치 제거: {before_count - after_count}개 제거")
        
        return df_clean
    
    @staticmethod
    def normalize_data(df: pd.DataFrame,
                      columns: List[str] = None,
                      method: str = 'standard') -> Tuple[pd.DataFrame, object]:
        """데이터 정규화"""
        df_norm = df.copy()
        
        if columns is None:
            columns = df_norm.select_dtypes(include=[np.number]).columns
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise DataProcessingError(f"지원하지 않는 정규화 방법: {method}")
        
        df_norm[columns] = scaler.fit_transform(df_norm[columns])
        
        logger.info(f"데이터 정규화 완료: {method} 방법, {len(columns)}개 컬럼")
        return df_norm, scaler
