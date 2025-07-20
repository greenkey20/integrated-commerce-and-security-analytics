"""
공통 데이터 처리 유틸리티

모든 도메인에서 공통으로 사용하는 데이터 처리 로직들
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional, Union
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

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


class DataSplitter:
    """데이터 분할 클래스"""
    
    @staticmethod
    def train_test_split(X: pd.DataFrame, 
                        y: pd.Series,
                        test_size: float = 0.2,
                        random_state: int = 42,
                        stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """훈련/테스트 데이터 분할"""
        from sklearn.model_selection import train_test_split
        
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


# 편의를 위한 통합 클래스
class DataProcessor:
    """데이터 처리 통합 클래스"""
    
    def __init__(self):
        self.validator = DataValidator()
        self.cleaner = DataCleaner()
        self.engineer = FeatureEngineer()
        self.splitter = DataSplitter()
        self.processing_history = []
    
    def process_pipeline(self, df: pd.DataFrame, 
                        pipeline_config: Dict) -> pd.DataFrame:
        """데이터 처리 파이프라인 실행"""
        processed_df = df.copy()
        
        # 검증
        if 'validation' in pipeline_config:
            self.validator.validate_dataframe(
                processed_df, 
                pipeline_config['validation'].get('required_columns')
            )
        
        # 결측값 처리
        if 'missing_values' in pipeline_config:
            processed_df = self.cleaner.handle_missing_values(
                processed_df,
                pipeline_config['missing_values']
            )
        
        # 이상치 제거
        if 'outliers' in pipeline_config:
            processed_df = self.cleaner.remove_outliers(
                processed_df,
                **pipeline_config['outliers']
            )
        
        # 특성 공학
        if 'feature_engineering' in pipeline_config:
            fe_config = pipeline_config['feature_engineering']
            
            if 'datetime_features' in fe_config:
                for datetime_col in fe_config['datetime_features']:
                    processed_df = self.engineer.create_datetime_features(
                        processed_df, datetime_col
                    )
            
            if 'interaction_features' in fe_config:
                processed_df = self.engineer.create_interaction_features(
                    processed_df, fe_config['interaction_features']
                )
        
        # 정규화
        if 'normalization' in pipeline_config:
            processed_df, scaler = self.cleaner.normalize_data(
                processed_df,
                **pipeline_config['normalization']
            )
        
        self.processing_history.append({
            'timestamp': pd.Timestamp.now(),
            'input_shape': df.shape,
            'output_shape': processed_df.shape,
            'config': pipeline_config
        })
        
        logger.info(f"데이터 처리 파이프라인 완료: {df.shape} → {processed_df.shape}")
        return processed_df
    
    def get_processing_summary(self) -> Dict:
        """처리 이력 요약"""
        if not self.processing_history:
            return {"message": "처리 이력이 없습니다."}
        
        latest = self.processing_history[-1]
        return {
            "total_processing_runs": len(self.processing_history),
            "latest_run": latest['timestamp'],
            "latest_input_shape": latest['input_shape'],
            "latest_output_shape": latest['output_shape'],
            "config_used": latest['config']
        }
