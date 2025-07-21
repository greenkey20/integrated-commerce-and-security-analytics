"""
통합 데이터 처리기 구현

pandas와 기타 heavy 라이브러리들은 여기서만 import되어
필요할 때만 로딩됩니다.
"""

import pandas as pd
import logging
from typing import Dict
from data.base import DataValidator, DataCleaner, FeatureEngineer, DataSplitter

logger = logging.getLogger(__name__)


class DataProcessor:
    """데이터 처리 통합 클래스"""
    
    def __init__(self):
        self.validator = DataValidator()
        self.cleaner = DataCleaner()
        self.engineer = FeatureEngineer()
        self.splitter = DataSplitter()
        self.processing_history = []
        self._cached_data = None
    
    def load_data(self) -> pd.DataFrame:
        """Mall Customer 데이터 로딩"""
        if self._cached_data is not None:
            return self._cached_data.copy()
            
        try:
            # 프로젝트 내 데이터 파일 시도
            data = pd.read_csv('data/Mall_Customers.csv')
            logger.info(f"로컬 데이터 파일 로딩 성공: {data.shape}")
        except FileNotFoundError:
            try:
                # GitHub에서 다운로드 시도
                url = "https://raw.githubusercontent.com/SteffiPeTaffy/machineLearningAZ/master/Machine%20Learning%20A-Z%20Template%20Folder/Part%204%20-%20Clustering/Section%2025%20-%20Hierarchical%20Clustering/Mall_Customers.csv"
                data = pd.read_csv(url)
                logger.info(f"온라인 데이터 로딩 성공: {data.shape}")
            except Exception as e:
                logger.warning(f"온라인 데이터 로딩 실패: {e}")
                # 샘플 데이터 생성
                data = self._generate_sample_data()
                logger.info(f"샘플 데이터 생성: {data.shape}")
        
        # 데이터 캐싱
        self._cached_data = data.copy()
        return data
    
    def _generate_sample_data(self) -> pd.DataFrame:
        """Mall Customer 형태의 샘플 데이터 생성"""
        import numpy as np
        np.random.seed(42)
        
        n_customers = 200
        data = pd.DataFrame({
            'CustomerID': range(1, n_customers + 1),
            'Gender': np.random.choice(['Male', 'Female'], n_customers),
            'Age': np.random.randint(18, 70, n_customers),
            'Annual Income (k$)': np.random.randint(15, 140, n_customers),
            'Spending Score (1-100)': np.random.randint(1, 100, n_customers)
        })
        
        return data
    
    def get_feature_names(self) -> list:
        """특성 컬럼명 반환"""
        return ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    
    def get_numeric_columns(self) -> list:
        """숫자형 컬럼명 반환"""
        return ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    
    def validate_data(self, data: pd.DataFrame) -> dict:
        """데이터 검증 결과 반환"""
        return {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'data_types': data.dtypes,
            'has_missing': data.isnull().any().any(),
            'missing_values': data.isnull().sum()
        }
    
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
    
    def save_processed_data(self, df: pd.DataFrame, 
                           domain: str, 
                           filename: str) -> str:
        """처리된 데이터를 파일로 저장"""
        save_path = f"data/processed/{domain}/{filename}"
        
        # 확장자에 따라 저장 방식 결정
        if filename.endswith('.csv'):
            df.to_csv(save_path, index=False)
        elif filename.endswith('.parquet'):
            df.to_parquet(save_path, index=False)
        elif filename.endswith('.pkl'):
            df.to_pickle(save_path)
        else:
            # 기본값은 CSV
            save_path += '.csv'
            df.to_csv(save_path, index=False)
        
        logger.info(f"처리된 데이터 저장 완료: {save_path}")
        return save_path
    
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
