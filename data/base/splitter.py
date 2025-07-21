"""
데이터 분할 클래스

모든 도메인에서 공통으로 사용하는 데이터 분할 로직
"""

import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from config.logging import setup_logger

logger = setup_logger(__name__)


class DataSplitter:
    """데이터 분할 클래스"""
    
    @staticmethod
    def train_test_split(X: pd.DataFrame, 
                        y: pd.Series,
                        test_size: float = 0.2,
                        random_state: int = 42,
                        stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """훈련/테스트 데이터 분할"""
        stratify_data = y if stratify and len(y.unique()) > 1 else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_data
        )
        
        logger.info(f"데이터 분할 완료: 훈련 {len(X_train)}, 테스트 {len(X_test)}")
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def time_series_split(df: pd.DataFrame,
                         date_column: str,
                         split_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """시계열 데이터 분할"""
        df_sorted = df.sort_values(date_column)
        split_timestamp = pd.to_datetime(split_date)
        
        train_data = df_sorted[df_sorted[date_column] < split_timestamp]
        test_data = df_sorted[df_sorted[date_column] >= split_timestamp]
        
        logger.info(f"시계열 분할 완료: 훈련 {len(train_data)}, 테스트 {len(test_data)}")
        return train_data, test_data
