"""
Online Retail 모델 훈련 및 평가 모듈

이 모듈은 Online Retail 데이터를 활용한 선형회귀 모델의 
훈련, 검증, 평가를 담당합니다.
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats

warnings.filterwarnings("ignore")


class RetailModelTrainer:
    """
    Online Retail 모델 훈련 및 평가를 담당하는 클래스
    
    선형회귀 모델을 기반으로 한 예측 모델링을 수행합니다.
    """
    
    def __init__(self):
        """초기화 메서드"""
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.training_results = None
        self.evaluation_results = None
        
    def prepare_modeling_data(self, customer_features: pd.DataFrame, target_column: str = 'predicted_next_amount') -> Tuple[pd.DataFrame, pd.Series]:
        """머신러닝 모델링을 위한 데이터 준비"""
        print("⚙️ 모델링 데이터 준비 중...")
        
        df = customer_features.copy()
        
        # 타겟 변수 확인
        if target_column not in df.columns:
            raise ValueError(f"타겟 변수 '{target_column}'이 데이터에 없습니다.")
        
        # 타겟 변수 분리
        y = df[target_column].copy()
        
        # 모델링에 사용할 특성 선택
        exclude_cols = [
            target_column, 'customer_value_category', 'monthly_avg_amount',
            'first_purchase', 'last_purchase', 'customer_segment'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols].copy()
        
        # 범주형 변수 인코딩
        categorical_cols = X.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print(f"   범주형 변수 인코딩: {list(categorical_cols)}")
            X = pd.get_dummies(X, columns=categorical_cols, prefix=categorical_cols)
        
        # 무한값 및 결측값 처리
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # 특성명 저장
        self.feature_names = X.columns.tolist()
        
        print(f"✅ 모델링 데이터 준비 완료")
        print(f"   특성 수: {X.shape[1]}개")
        print(f"   샘플 수: {X.shape[0]}개")
        print(f"   타겟 변수: {target_column}")
        
        return X, y
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.2, 
                   scale_features: bool = True,
                   random_state: int = 42) -> Dict:
        """선형회귀 모델 훈련"""
        print("🚀 선형회귀 모델 훈련 시작...")
        
        # 훈련/테스트 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # 특성 정규화
        if scale_features:
            print("   특성 정규화 수행 중...")
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # DataFrame 형태로 변환
            X_train_final = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
            X_test_final = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        else:
            X_train_final = X_train.copy()
            X_test_final = X_test.copy()
        
        # 모델 훈련
        print("   선형회귀 모델 훈련 중...")
        self.model = LinearRegression()
        self.model.fit(X_train_final, y_train)
        
        # 예측
        y_train_pred = self.model.predict(X_train_final)
        y_test_pred = self.model.predict(X_test_final)
        
        # 훈련 결과 저장
        self.training_results = {
            'model': self.model,
            'scaler': self.scaler,
            'X_train': X_train_final,
            'X_test': X_test_final,
            'y_train': y_train,
            'y_test': y_test,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred,
            'feature_names': self.feature_names,
            'test_size': test_size,
            'scale_features': scale_features,
            'random_state': random_state
        }
        
        print("✅ 모델 훈련 완료")
        
        return self.training_results
    
    def evaluate_model(self) -> Dict:
        """모델 성능 평가"""
        if self.training_results is None:
            raise ValueError("모델이 아직 훈련되지 않았습니다.")
        
        print("📊 모델 성능 평가 중...")
        
        # 훈련 결과에서 데이터 추출
        y_train = self.training_results['y_train']
        y_test = self.training_results['y_test']
        y_train_pred = self.training_results['y_train_pred']
        y_test_pred = self.training_results['y_test_pred']
        
        # 성능 메트릭 계산
        evaluation_metrics = {
            # 훈련 성능
            'r2_train': r2_score(y_train, y_train_pred),
            'mae_train': mean_absolute_error(y_train, y_train_pred),
            'rmse_train': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            
            # 테스트 성능
            'r2_test': r2_score(y_test, y_test_pred),
            'mae_test': mean_absolute_error(y_test, y_test_pred),
            'rmse_test': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            
            # 과적합 분석
            'performance_gap': abs(r2_score(y_test, y_test_pred) - r2_score(y_train, y_train_pred)),
            'relative_error': (mean_absolute_error(y_test, y_test_pred) / y_test.mean()) * 100,
            
            # 잔차 분석
            'residuals': y_test - y_test_pred,
            'residuals_std': np.std(y_test - y_test_pred),
            'residuals_mean': np.mean(y_test - y_test_pred)
        }
        
        # 정규성 검정
        residuals = y_test - y_test_pred
        shapiro_stat, shapiro_p = stats.shapiro(residuals[:min(5000, len(residuals))])  # 샘플 크기 제한
        
        evaluation_metrics['normality_test'] = {
            'shapiro_stat': shapiro_stat,
            'shapiro_p_value': shapiro_p,
            'is_normal': shapiro_p > 0.05
        }
        
        # 이분산성 검정 (Breusch-Pagan test 간소화 버전)
        squared_residuals = residuals ** 2
        X_test = self.training_results['X_test']
        
        # 간단한 상관관계 기반 이분산성 검정
        heteroscedasticity_correlation = np.corrcoef(y_test_pred, squared_residuals)[0, 1]
        
        evaluation_metrics['heteroscedasticity_test'] = {
            'correlation': heteroscedasticity_correlation,
            'is_homoscedastic': abs(heteroscedasticity_correlation) < 0.1
        }
        
        # 특성 중요도 분석
        feature_importance = self._analyze_feature_importance()
        evaluation_metrics['feature_importance'] = feature_importance
        
        # 모델 해석
        model_interpretation = self._interpret_model_performance(evaluation_metrics)
        evaluation_metrics['interpretation'] = model_interpretation
        
        self.evaluation_results = evaluation_metrics
        
        print("✅ 모델 성능 평가 완료")
        
        return evaluation_metrics
    
    def _analyze_feature_importance(self) -> Dict:
        """특성 중요도 분석"""
        if self.model is None or self.feature_names is None:
            return {"error": "모델이 훈련되지 않았습니다."}
        
        # 회귀 계수 기반 중요도
        coefficients = self.model.coef_
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        }).sort_values('abs_coefficient', ascending=False)
        
        # 상위 10개 특성
        top_features = feature_importance.head(10).to_dict('records')
        
        # 양의 영향과 음의 영향 특성 분리
        positive_impact = feature_importance[feature_importance['coefficient'] > 0].head(5)
        negative_impact = feature_importance[feature_importance['coefficient'] < 0].head(5)
        
        return {
            'top_10_features': top_features,
            'positive_impact_features': positive_impact.to_dict('records'),
            'negative_impact_features': negative_impact.to_dict('records'),
            'intercept': self.model.intercept_,
            'total_features': len(self.feature_names)
        }
    
    def _interpret_model_performance(self, metrics: Dict) -> Dict:
        """모델 성능 해석"""
        
        interpretation = {
            'overall_performance': '',
            'overfitting_status': '',
            'prediction_accuracy': '',
            'business_applicability': '',
            'improvement_suggestions': []
        }
        
        # 전체 성능 평가
        test_r2 = metrics['r2_test']
        if test_r2 >= 0.8:
            interpretation['overall_performance'] = "우수한 성능 (R² ≥ 0.8)"
        elif test_r2 >= 0.6:
            interpretation['overall_performance'] = "양호한 성능 (R² ≥ 0.6)"
        elif test_r2 >= 0.4:
            interpretation['overall_performance'] = "보통 성능 (R² ≥ 0.4)"
        else:
            interpretation['overall_performance'] = "개선 필요 (R² < 0.4)"
        
        # 과적합 분석
        performance_gap = metrics['performance_gap']
        if performance_gap <= 0.05:
            interpretation['overfitting_status'] = "과적합 없음"
        elif performance_gap <= 0.1:
            interpretation['overfitting_status'] = "경미한 과적합"
        else:
            interpretation['overfitting_status'] = "과적합 발생"
            interpretation['improvement_suggestions'].append("정규화 기법 적용 고려")
        
        # 예측 정확도 분석
        relative_error = metrics['relative_error']
        if relative_error <= 15:
            interpretation['prediction_accuracy'] = "고정밀도 예측 (상대오차 ≤ 15%)"
            interpretation['business_applicability'] = "개인화 마케팅 전략 수립 가능"
        elif relative_error <= 25:
            interpretation['prediction_accuracy'] = "중간 정밀도 예측 (상대오차 ≤ 25%)"
            interpretation['business_applicability'] = "세그먼트별 전략 수립 적합"
        else:
            interpretation['prediction_accuracy'] = "낮은 정밀도 예측 (상대오차 > 25%)"
            interpretation['business_applicability'] = "전반적 트렌드 파악 수준"
            interpretation['improvement_suggestions'].append("추가 특성 공학 필요")
        
        # 개선 제안
        if test_r2 < 0.6:
            interpretation['improvement_suggestions'].append("비선형 모델 고려")
        
        if not metrics['normality_test']['is_normal']:
            interpretation['improvement_suggestions'].append("잔차 정규성 개선 필요")
        
        if not metrics['heteroscedasticity_test']['is_homoscedastic']:
            interpretation['improvement_suggestions'].append("등분산성 가정 위반, 가중회귀 고려")
        
        return interpretation
    
    def predict_customer_value(self, customer_data: pd.DataFrame) -> pd.DataFrame:
        """새로운 고객 데이터에 대한 예측"""
        if self.model is None:
            raise ValueError("모델이 아직 훈련되지 않았습니다.")
        
        # 특성 전처리
        X = customer_data[self.feature_names].copy()
        
        # 결측값 처리
        X = X.fillna(X.median())
        
        # 정규화 적용
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
            X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # 예측
        predictions = self.model.predict(X)
        
        # 결과 DataFrame 생성
        result = customer_data.copy()
        result['predicted_value'] = predictions
        result['prediction_confidence'] = 'High' if self.evaluation_results['r2_test'] >= 0.6 else 'Medium'
        
        return result
    
    def get_model_summary(self) -> Dict:
        """모델 요약 정보 반환"""
        if self.training_results is None or self.evaluation_results is None:
            return {"error": "모델이 훈련되거나 평가되지 않았습니다."}
        
        summary = {
            'model_type': 'Linear Regression',
            'training_samples': len(self.training_results['y_train']),
            'test_samples': len(self.training_results['y_test']),
            'features_count': len(self.feature_names),
            'performance_metrics': {
                'r2_score': round(self.evaluation_results['r2_test'], 4),
                'mae': round(self.evaluation_results['mae_test'], 2),
                'rmse': round(self.evaluation_results['rmse_test'], 2),
                'relative_error_pct': round(self.evaluation_results['relative_error'], 2)
            },
            'model_quality': {
                'overfitting_risk': 'Low' if self.evaluation_results['performance_gap'] <= 0.05 else 'Medium',
                'residuals_normal': self.evaluation_results['normality_test']['is_normal'],
                'homoscedastic': self.evaluation_results['heteroscedasticity_test']['is_homoscedastic']
            },
            'business_impact': self.evaluation_results['interpretation']['business_applicability'],
            'top_features': [f['feature'] for f in self.evaluation_results['feature_importance']['top_10_features'][:5]]
        }
        
        return summary
