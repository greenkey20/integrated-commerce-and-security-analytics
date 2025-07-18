"""
데이터 처리 모듈

데이터 로딩, 전처리, 검증 등을 담당
"""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from config.settings import AppConfig


class DataProcessor:
    """데이터 로딩과 전처리를 담당하는 클래스"""
    
    def __init__(self):
        self.data = None
        self.processed_data = None
        self.scaler = StandardScaler()
        
    @st.cache_data
    def load_data(_self):
        """데이터 로드 및 전처리"""
        try:
            # GitHub에서 직접 데이터 로드
            url = AppConfig.DATA_URL
            data = pd.read_csv(url)
            return data
        except:
            # 샘플 데이터 생성 (실제 환경에서는 실제 데이터 사용)
            np.random.seed(42)
            sample_data = {
                "CustomerID": range(1, 201),
                "Gender": np.random.choice(["Male", "Female"], 200),
                "Age": np.random.normal(40, 15, 200).astype(int),
                "Annual Income (k$)": np.random.normal(60, 20, 200).astype(int),
                "Spending Score (1-100)": np.random.normal(50, 25, 200).astype(int),
            }
            data = pd.DataFrame(sample_data)
            data["Age"] = np.clip(data["Age"], 18, 80)
            data["Annual Income (k$)"] = np.clip(data["Annual Income (k$)"], 15, 150)
            data["Spending Score (1-100)"] = np.clip(data["Spending Score (1-100)"], 1, 100)
            return data
    
    def preprocess_features(self, data):
        """클러스터링을 위한 특성 전처리"""
        features = data[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
        scaled_features = self.scaler.fit_transform(features)
        return scaled_features, self.scaler
    
    def validate_data(self, data):
        """데이터 품질 검사"""
        validation_results = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'missing_values': data.isnull().sum(),
            'data_types': data.dtypes,
            'has_missing': data.isnull().sum().sum() > 0
        }
        return validation_results
    
    def get_feature_names(self):
        """분석에 사용되는 특성명 반환"""
        return ["연령", "연간소득(k$)", "지출점수"]
    
    def get_numeric_columns(self):
        """수치형 컬럼 이름 반환"""
        return ["Age", "Annual Income (k$)", "Spending Score (1-100)"]


# 전역 인스턴스 생성 (기존 함수와의 호환성)
data_processor = DataProcessor()

# 기존 함수와의 호환성을 위한 래퍼
def load_data():
    """기존 load_data 함수와의 호환성"""
    return data_processor.load_data()
