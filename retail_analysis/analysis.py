"""
Online Retail 분석 통합 모듈

리팩토링된 모듈들의 통합 진입점을 제공합니다.
이 파일은 기존 코드와의 호환성을 위해 유지됩니다.
"""

# 리팩토링된 모듈들 import
from retail_analysis.data_loader import RetailDataLoader
from retail_analysis.data_processor import RetailDataProcessor
from retail_analysis.feature_engineer import RetailFeatureEngineer
from retail_analysis.model_trainer import RetailModelTrainer
from retail_analysis.visualizer import RetailVisualizer

import pandas as pd
import numpy as np
import warnings
from typing import Tuple, Dict, List, Optional

warnings.filterwarnings("ignore")


class RetailAnalysisManager:
    """
    Online Retail 분석 전체 과정을 관리하는 통합 클래스
    
    리팩토링된 개별 모듈들을 조율하여 전체 분석 워크플로우를 제공합니다.
    """
    
    def __init__(self):
        """초기화 메서드"""
        self.data_loader = None
        self.data_processor = None
        self.feature_engineer = None
        self.model_trainer = None
        
        # 상태 관리
        self.raw_data = None
        self.cleaned_data = None
        self.customer_features = None
        self.target_data = None
        self.training_results = None
        self.evaluation_results = None
        
        # 메타데이터
        self.column_mapping = {}
        self.analysis_metadata = {}
    
    def initialize_components(self):
        """분석 컴포넌트들 초기화"""
        print("🔧 분석 컴포넌트 초기화 중...")
        
        # 데이터 로더 초기화
        self.data_loader = RetailDataLoader()
        
        print("✅ 컴포넌트 초기화 완료")
    
    def run_full_analysis(self, target_months: int = 3, test_size: float = 0.2, 
                         scale_features: bool = True, random_state: int = 42) -> Dict:
        """
        전체 분석 과정을 한 번에 실행
        
        Args:
            target_months: 예측 기간 (개월)
            test_size: 테스트 데이터 비율
            scale_features: 특성 정규화 여부
            random_state: 랜덤 시드
            
        Returns:
            Dict: 전체 분석 결과
        """
        print("🚀 Online Retail 전체 분석 시작...")
        
        try:
            # 1단계: 데이터 로딩
            self.load_and_analyze_data()
            
            # 2단계: 데이터 정제
            self.clean_data()
            
            # 3단계: 특성 공학
            self.create_features()
            
            # 4단계: 타겟 생성
            self.create_target(target_months)
            
            # 5단계: 모델 훈련
            self.train_model(test_size, scale_features, random_state)
            
            # 6단계: 모델 평가
            self.evaluate_model()
            
            # 결과 요약
            analysis_summary = self.get_analysis_summary()
            
            print("✅ 전체 분석 완료!")
            return analysis_summary
            
        except Exception as e:
            print(f"❌ 분석 중 오류 발생: {str(e)}")
            raise e
    
    def load_and_analyze_data(self):
        """1단계: 데이터 로딩 및 품질 분석"""
        print("1️⃣ 데이터 로딩 및 품질 분석...")
        
        if self.data_loader is None:
            self.initialize_components()
        
        # 데이터 로딩
        self.raw_data = self.data_loader.load_data()
        self.column_mapping = self.data_loader.get_column_mapping()
        
        # 품질 분석
        quality_report = self.data_loader.analyze_data_quality(self.raw_data)
        self.analysis_metadata['data_quality'] = quality_report
        
        print(f"   ✅ 데이터 로딩 완료: {len(self.raw_data):,}개 레코드")
    
    def clean_data(self):
        """2단계: 데이터 정제"""
        print("2️⃣ 데이터 정제...")
        
        if self.raw_data is None:
            raise ValueError("먼저 데이터를 로딩해주세요.")
        
        # 데이터 프로세서 초기화
        self.data_processor = RetailDataProcessor(self.column_mapping)
        
        # 데이터 정제
        self.cleaned_data = self.data_processor.clean_data(self.raw_data)
        
        # 검증
        validation_report = self.data_processor.validate_data_quality(self.cleaned_data)
        self.analysis_metadata['data_validation'] = validation_report
        
        print(f"   ✅ 데이터 정제 완료: {len(self.cleaned_data):,}개 레코드")
    
    def create_features(self):
        """3단계: 특성 공학"""
        print("3️⃣ 특성 공학...")
        
        if self.cleaned_data is None:
            raise ValueError("먼저 데이터를 정제해주세요.")
        
        # 특성 엔지니어 초기화
        self.feature_engineer = RetailFeatureEngineer(self.column_mapping)
        
        # 고객 특성 생성
        self.customer_features = self.feature_engineer.create_customer_features(self.cleaned_data)
        
        # 특성 중요도 분석
        importance_analysis = self.feature_engineer.get_feature_importance_analysis(self.customer_features)
        self.analysis_metadata['feature_importance'] = importance_analysis
        
        print(f"   ✅ 특성 공학 완료: {len(self.customer_features):,}명 고객, {len(self.customer_features.columns)}개 특성")
    
    def create_target(self, target_months: int = 3):
        """4단계: 타겟 변수 생성"""
        print(f"4️⃣ 타겟 변수 생성 (예측 기간: {target_months}개월)...")
        
        if self.customer_features is None:
            raise ValueError("먼저 특성 공학을 완료해주세요.")
        
        # 타겟 변수 생성
        self.target_data = self.feature_engineer.create_target_variable(
            self.customer_features, target_months=target_months
        )
        
        self.analysis_metadata['target_months'] = target_months
        
        print(f"   ✅ 타겟 생성 완료: 평균 예측 금액 £{self.target_data['predicted_next_amount'].mean():.2f}")
    
    def train_model(self, test_size: float = 0.2, scale_features: bool = True, random_state: int = 42):
        """5단계: 모델 훈련"""
        print("5️⃣ 모델 훈련...")
        
        if self.target_data is None:
            raise ValueError("먼저 타겟 변수를 생성해주세요.")
        
        # 모델 트레이너 초기화
        self.model_trainer = RetailModelTrainer()
        
        # 데이터 준비
        X, y = self.model_trainer.prepare_modeling_data(self.target_data)
        
        # 모델 훈련
        self.training_results = self.model_trainer.train_model(
            X, y, test_size=test_size, scale_features=scale_features, random_state=random_state
        )
        
        print(f"   ✅ 모델 훈련 완료: R² = {self.training_results['model'].score(self.training_results['X_test'], self.training_results['y_test']):.3f}")
    
    def evaluate_model(self):
        """6단계: 모델 평가"""
        print("6️⃣ 모델 평가...")
        
        if self.training_results is None:
            raise ValueError("먼저 모델을 훈련해주세요.")
        
        # 모델 평가
        self.evaluation_results = self.model_trainer.evaluate_model()
        
        print(f"   ✅ 모델 평가 완료: 테스트 R² = {self.evaluation_results['r2_test']:.3f}")
    
    def get_analysis_summary(self) -> Dict:
        """분석 결과 요약"""
        if self.evaluation_results is None:
            raise ValueError("전체 분석이 완료되지 않았습니다.")
        
        # 모델 요약
        model_summary = self.model_trainer.get_model_summary()
        
        summary = {
            'data_overview': {
                'raw_records': len(self.raw_data) if self.raw_data is not None else 0,
                'cleaned_records': len(self.cleaned_data) if self.cleaned_data is not None else 0,
                'customers_analyzed': len(self.customer_features) if self.customer_features is not None else 0,
                'data_retention_rate': (len(self.cleaned_data) / len(self.raw_data) * 100) if self.raw_data is not None and self.cleaned_data is not None else 0
            },
            'feature_engineering': {
                'total_features': len(self.customer_features.columns) if self.customer_features is not None else 0,
                'target_months': self.analysis_metadata.get('target_months', 0),
                'avg_predicted_amount': self.target_data['predicted_next_amount'].mean() if self.target_data is not None else 0
            },
            'model_performance': {
                'r2_score': self.evaluation_results['r2_test'],
                'mae': self.evaluation_results['mae_test'],
                'rmse': self.evaluation_results['rmse_test'],
                'relative_error': self.evaluation_results['relative_error']
            },
            'model_info': model_summary,
            'metadata': self.analysis_metadata
        }
        
        return summary
    
    def get_predictions(self, customer_data: pd.DataFrame = None) -> pd.DataFrame:
        """새로운 고객 데이터에 대한 예측"""
        if self.model_trainer is None or self.training_results is None:
            raise ValueError("모델이 훈련되지 않았습니다.")
        
        if customer_data is None:
            customer_data = self.target_data
        
        return self.model_trainer.predict_customer_value(customer_data)


# 기존 코드와의 호환성을 위한 클래스들 (Deprecated)
class RetailDataProcessor_Legacy:
    """기존 코드 호환성을 위한 레거시 클래스"""
    
    def __init__(self):
        self.manager = RetailAnalysisManager()
        warnings.warn("이 클래스는 더 이상 사용되지 않습니다. RetailAnalysisManager를 사용하세요.", DeprecationWarning)
    
    def load_data(self):
        """레거시 메서드"""
        if self.manager.data_loader is None:
            self.manager.initialize_components()
        return self.manager.data_loader.load_data()
    
    def analyze_data_quality(self, data):
        """레거시 메서드"""
        if self.manager.data_loader is None:
            self.manager.initialize_components()
        return self.manager.data_loader.analyze_data_quality(data)


# 기존 코드 호환성을 위한 별칭
RetailDataProcessor = RetailAnalysisManager
