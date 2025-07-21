"""
특성 공학 클래스

모든 도메인에서 공통으로 사용하는 특성 공학 로직
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Union
from utils.exceptions import DataProcessingError
from config.logging import setup_logger

logger = setup_logger(__name__)


class FeatureEngineer:
    """특성 공학 클래스"""
    
    @staticmethod
    def create_datetime_features(df: pd.DataFrame, 
                               datetime_col: str) -> pd.DataFrame:
        """날짜/시간 특성 생성"""
        df_feat = df.copy()
        
        if datetime_col not in df.columns:
            raise DataProcessingError(f"날짜 컬럼이 없습니다: {datetime_col}")
        
        # 날짜/시간 변환
        df_feat[datetime_col] = pd.to_datetime(df_feat[datetime_col])
        
        # 특성 생성
        df_feat[f'{datetime_col}_year'] = df_feat[datetime_col].dt.year
        df_feat[f'{datetime_col}_month'] = df_feat[datetime_col].dt.month
        df_feat[f'{datetime_col}_day'] = df_feat[datetime_col].dt.day
        df_feat[f'{datetime_col}_dayofweek'] = df_feat[datetime_col].dt.dayofweek
        df_feat[f'{datetime_col}_hour'] = df_feat[datetime_col].dt.hour
        df_feat[f'{datetime_col}_quarter'] = df_feat[datetime_col].dt.quarter
        
        # 비즈니스 로직 특성
        df_feat[f'{datetime_col}_is_weekend'] = df_feat[f'{datetime_col}_dayofweek'] >= 5
        df_feat[f'{datetime_col}_is_business_hour'] = (
            (df_feat[f'{datetime_col}_hour'] >= 9) & 
            (df_feat[f'{datetime_col}_hour'] <= 17)
        )
        
        logger.info(f"날짜/시간 특성 생성 완료: {datetime_col}")
        return df_feat
    
    @staticmethod
    def create_interaction_features(df: pd.DataFrame,
                                  feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """상호작용 특성 생성"""
        df_feat = df.copy()
        
        for col1, col2 in feature_pairs:
            if col1 in df.columns and col2 in df.columns:
                # 곱셈 상호작용
                df_feat[f'{col1}_x_{col2}'] = df_feat[col1] * df_feat[col2]
                
                # 비율 특성 (0 나누기 방지)
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = df_feat[col1] / df_feat[col2]
                    ratio = np.where(np.isfinite(ratio), ratio, 0)
                    df_feat[f'{col1}_div_{col2}'] = ratio
        
        logger.info(f"상호작용 특성 생성 완료: {len(feature_pairs)}개 쌍")
        return df_feat
    
    @staticmethod
    def create_binning_features(df: pd.DataFrame,
                              column: str,
                              bins: Union[int, List] = 5,
                              labels: List[str] = None) -> pd.DataFrame:
        """구간화 특성 생성"""
        df_feat = df.copy()
        
        if column not in df.columns:
            raise DataProcessingError(f"구간화할 컬럼이 없습니다: {column}")
        
        try:
            df_feat[f'{column}_binned'] = pd.cut(
                df_feat[column], 
                bins=bins, 
                labels=labels,
                include_lowest=True
            )
            
            # 원핫 인코딩
            binned_dummies = pd.get_dummies(
                df_feat[f'{column}_binned'], 
                prefix=f'{column}_bin'
            )
            df_feat = pd.concat([df_feat, binned_dummies], axis=1)
            
            logger.info(f"구간화 특성 생성 완료: {column}")
            
        except Exception as e:
            logger.error(f"구간화 실패 ({column}): {e}")
            raise DataProcessingError(f"구간화 실패: {e}")
        
        return df_feat
