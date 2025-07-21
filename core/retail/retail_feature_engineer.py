"""
Online Retail 특성 공학 및 파생변수 생성 모듈

이 모듈은 Online Retail 데이터에서 고객별 특성을 생성하고
RFM 분석 등의 고급 특성 공학을 수행합니다.
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")


class RetailFeatureEngineer:
    """
    Online Retail 특성 공학 및 파생변수 생성을 담당하는 클래스
    
    고객별 RFM 분석, 행동 패턴 분석, 구매 성향 분석 등을 수행합니다.
    """
    
    def __init__(self, column_mapping: Dict[str, str]):
        """
        초기화 메서드
        
        Args:
            column_mapping: 컬럼 매핑 정보
        """
        self.column_mapping = column_mapping
        self.customer_features = None
        
    def create_customer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """고객별 특성 생성 - 동적 컬럼 매핑 지원"""
        print("🏗️ 고객별 특성 생성을 시작합니다...")
        
        # 실제 데이터 컬럼 구조 확인
        print(f"📊 실제 데이터 컬럼: {list(data.columns)}")
        print(f"📊 데이터 형태: {data.shape}")
        print(f"🔄 사용 중인 컬럼 매핑: {self.column_mapping}")
        
        # 필수 컬럼 확인
        customer_id_col = self.column_mapping.get('customer_id')
        if not customer_id_col or customer_id_col not in data.columns:
            raise ValueError(f"고객 ID 컬럼을 찾을 수 없습니다. 매핑: {self.column_mapping}")
        
        # CustomerID가 있는 데이터만 사용
        customer_data = data[data[customer_id_col].notna()].copy()
        print(f"분석 대상 고객 수: {customer_data[customer_id_col].nunique():,}명")
        print(f"분석 대상 거래 수: {len(customer_data):,}건")
        
        # 기본 집계 통계
        print("📊 기본 통계량 계산 중...")
        
        # 집계에 사용할 컬럼들 준비
        agg_dict = {}
        
        # 수량 관련 통계
        quantity_col = self.column_mapping.get('quantity')
        if quantity_col and quantity_col in customer_data.columns:
            agg_dict[quantity_col] = ['sum', 'mean', 'std', 'min', 'max']
        
        # 단가 관련 통계
        unit_price_col = self.column_mapping.get('unit_price')
        if unit_price_col and unit_price_col in customer_data.columns:
            agg_dict[unit_price_col] = ['mean', 'std', 'min', 'max']
        
        # 날짜 관련 통계
        invoice_date_col = self.column_mapping.get('invoice_date')
        if invoice_date_col and invoice_date_col in customer_data.columns:
            agg_dict[invoice_date_col] = ['min', 'max', 'count']
        
        # 총액 관련 통계
        if 'TotalAmount' in customer_data.columns:
            agg_dict['TotalAmount'] = ['sum', 'mean', 'std', 'min', 'max']
        
        # 인보이스 관련 통계
        invoice_no_col = self.column_mapping.get('invoice_no')
        if invoice_no_col and invoice_no_col in customer_data.columns:
            agg_dict[invoice_no_col] = ['nunique', 'count']
        
        # 상품 관련 통계
        stock_code_col = self.column_mapping.get('stock_code')
        if stock_code_col and stock_code_col in customer_data.columns:
            agg_dict[stock_code_col] = 'nunique'
        
        description_col = self.column_mapping.get('description')
        if description_col and description_col in customer_data.columns:
            agg_dict[description_col] = 'nunique'
        
        # 반품 관련 통계
        if 'IsReturn' in customer_data.columns:
            agg_dict['IsReturn'] = ['sum', 'mean']
        
        print(f"집계 사전: {agg_dict}")
        
        # 고객별 집계 수행
        customer_features = customer_data.groupby(customer_id_col).agg(agg_dict).round(2)
        
        # 컬럼명 정리
        new_column_names = []
        for col in customer_features.columns:
            if isinstance(col, tuple):
                original_col, agg_func = col
                new_name = self._generate_feature_name(original_col, agg_func)
                new_column_names.append(new_name)
            else:
                new_column_names.append(str(col))
        
        customer_features.columns = new_column_names
        
        # 고급 파생 변수 생성
        print("🔧 고급 파생 변수 생성 중...")
        customer_features = self._create_advanced_features(customer_features, customer_data)
        
        # 결측값 처리
        print("🔧 결측값 최종 처리 중...")
        customer_features = self._handle_missing_values(customer_features)
        
        print(f"✅ 고객별 특성 생성 완료: {len(customer_features):,}명 고객, {len(customer_features.columns)}개 특성")
        
        self.customer_features = customer_features.copy()
        return customer_features
    
    def _generate_feature_name(self, original_col, agg_func):
        """집계 함수 기반 특성명 생성"""
        
        # 컬럼 매핑 역방향 탐색
        mapped_name = None
        for standard_name, actual_name in self.column_mapping.items():
            if actual_name == original_col:
                mapped_name = standard_name
                break
        
        # 표준 특성명 생성
        if mapped_name == 'invoice_no':
            if agg_func == 'nunique':
                return 'unique_invoices'
            elif agg_func == 'count':
                return 'total_items'
        elif mapped_name == 'quantity':
            if agg_func == 'sum':
                return 'total_quantity'
            elif agg_func == 'mean':
                return 'avg_quantity'
            elif agg_func == 'std':
                return 'std_quantity'
            elif agg_func == 'min':
                return 'min_quantity'
            elif agg_func == 'max':
                return 'max_quantity'
        elif mapped_name == 'unit_price':
            if agg_func == 'mean':
                return 'avg_unit_price'
            elif agg_func == 'std':
                return 'std_unit_price'
            elif agg_func == 'min':
                return 'min_unit_price'
            elif agg_func == 'max':
                return 'max_unit_price'
        elif mapped_name == 'invoice_date':
            if agg_func == 'min':
                return 'first_purchase'
            elif agg_func == 'max':
                return 'last_purchase'
            elif agg_func == 'count':
                return 'purchase_frequency'
        elif mapped_name == 'stock_code':
            return 'unique_products'
        elif mapped_name == 'description':
            return 'unique_descriptions'
        elif original_col == 'TotalAmount':
            if agg_func == 'sum':
                return 'total_amount'
            elif agg_func == 'mean':
                return 'avg_amount'
            elif agg_func == 'std':
                return 'std_amount'
            elif agg_func == 'min':
                return 'min_amount'
            elif agg_func == 'max':
                return 'max_amount'
        elif original_col == 'IsReturn':
            if agg_func == 'sum':
                return 'return_count'
            elif agg_func == 'mean':
                return 'return_rate'
        
        # 기본값
        return f'{original_col}_{agg_func}'
    
    def _create_advanced_features(self, customer_features: pd.DataFrame, customer_data: pd.DataFrame) -> pd.DataFrame:
        """고급 파생 변수 생성"""
        
        # 기본 RFM 분석 변수
        if 'first_purchase' in customer_features.columns and 'last_purchase' in customer_features.columns:
            customer_features['customer_lifespan_days'] = (
                customer_features['last_purchase'] - customer_features['first_purchase']
            ).dt.days
        
        # Recency, Frequency, Monetary
        if 'last_purchase' in customer_features.columns:
            analysis_date = customer_features['last_purchase'].max()
            customer_features['recency_days'] = (
                analysis_date - customer_features['last_purchase']
            ).dt.days
        
        if 'unique_invoices' in customer_features.columns:
            customer_features['frequency'] = customer_features['unique_invoices']
        
        if 'total_amount' in customer_features.columns:
            customer_features['monetary'] = customer_features['total_amount']
        
        # 기본 비율 계산
        if 'total_items' in customer_features.columns and 'unique_invoices' in customer_features.columns:
            customer_features['avg_items_per_transaction'] = (
                customer_features['total_items'] / customer_features['unique_invoices']
            ).round(2)
        
        # 가격 민감도
        if 'std_unit_price' in customer_features.columns and 'avg_unit_price' in customer_features.columns:
            customer_features['price_sensitivity'] = (
                customer_features['std_unit_price'] / customer_features['avg_unit_price']
            ).round(3)
        
        # 구매 주기 분석
        if 'customer_lifespan_days' in customer_features.columns and 'unique_invoices' in customer_features.columns:
            customer_features['avg_purchase_interval'] = (
                customer_features['customer_lifespan_days'] / customer_features['unique_invoices'].clip(lower=1)
            ).round(1)
        
        # 고객 가치 세그먼트 생성
        customer_features = self._create_customer_segments(customer_features)
        
        return customer_features
    
    def _create_customer_segments(self, customer_features: pd.DataFrame) -> pd.DataFrame:
        """고객 가치 세그먼트 생성"""
        
        if 'recency_days' in customer_features.columns and 'frequency' in customer_features.columns and 'monetary' in customer_features.columns:
            # RFM 점수 계산 (1-5 점수)
            customer_features['recency_score'] = pd.qcut(
                customer_features['recency_days'], 
                q=5, 
                labels=[5, 4, 3, 2, 1]  # 최근성은 역순 (최근일수록 높은 점수)
            ).astype(int)
            
            customer_features['frequency_score'] = pd.qcut(
                customer_features['frequency'].rank(method='first'), 
                q=5, 
                labels=[1, 2, 3, 4, 5]
            ).astype(int)
            
            customer_features['monetary_score'] = pd.qcut(
                customer_features['monetary'].rank(method='first'), 
                q=5, 
                labels=[1, 2, 3, 4, 5]
            ).astype(int)
            
            # 전체 RFM 점수
            customer_features['rfm_score'] = (
                customer_features['recency_score'] * 100 +
                customer_features['frequency_score'] * 10 +
                customer_features['monetary_score']
            )
            
            # 고객 세그먼트 분류
            def classify_customer_segment(row):
                r, f, m = row['recency_score'], row['frequency_score'], row['monetary_score']
                
                if r >= 4 and f >= 4 and m >= 4:
                    return 'Champions'
                elif r >= 4 and f >= 3 and m >= 3:
                    return 'Loyal Customers'
                elif r >= 3 and f >= 3 and m >= 3:
                    return 'Potential Loyalists'
                elif r >= 4 and f < 3 and m >= 3:
                    return 'New Customers'
                elif r >= 3 and f < 3 and m >= 3:
                    return 'Promising'
                elif r < 3 and f >= 3 and m >= 3:
                    return 'Need Attention'
                elif r < 3 and f >= 3 and m < 3:
                    return 'About to Sleep'
                elif r < 3 and f < 3 and m >= 3:
                    return 'At Risk'
                elif r < 3 and f < 3 and m < 3:
                    return 'Cannot Lose Them'
                else:
                    return 'Others'
            
            customer_features['customer_segment'] = customer_features.apply(classify_customer_segment, axis=1)
        
        return customer_features
    
    def _handle_missing_values(self, customer_features: pd.DataFrame) -> pd.DataFrame:
        """결측값 처리"""
        
        # 수치형 변수의 결측값을 중앙값으로 대체
        numeric_columns = customer_features.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if customer_features[col].isnull().any():
                median_value = customer_features[col].median()
                customer_features[col].fillna(median_value, inplace=True)
        
        # 범주형 변수의 결측값을 최빈값으로 대체
        categorical_columns = customer_features.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if customer_features[col].isnull().any():
                mode_value = customer_features[col].mode().iloc[0] if len(customer_features[col].mode()) > 0 else 'Unknown'
                customer_features[col].fillna(mode_value, inplace=True)
        
        return customer_features
    
    def create_target_variable(self, customer_features: pd.DataFrame, target_months: int = 3) -> pd.DataFrame:
        """예측 타겟 변수 생성"""
        print(f"🎯 타겟 변수 생성: 다음 {target_months}개월 구매 예상 금액")
        
        df = customer_features.copy()
        
        # 월평균 구매 금액 계산
        if 'total_amount' in df.columns and 'customer_lifespan_days' in df.columns:
            df['monthly_avg_amount'] = df['total_amount'] / (df['customer_lifespan_days'] / 30.44).clip(lower=1)
        elif 'total_amount' in df.columns:
            df['monthly_avg_amount'] = df['total_amount'] / 3  # 기본값으로 3개월 가정
        else:
            df['monthly_avg_amount'] = 100  # 기본값
        
        # 최근성을 고려한 가중치 적용
        if 'recency_days' in df.columns:
            recency_weight = np.exp(-df['recency_days'] / 30)
        else:
            recency_weight = 1.0
        
        # 구매 빈도를 고려한 가중치
        if 'frequency' in df.columns:
            frequency_weight = np.log1p(df['frequency']) / np.log1p(df['frequency'].max())
        else:
            frequency_weight = 1.0
        
        # 최종 예측값 계산
        df['predicted_next_amount'] = (
            df['monthly_avg_amount'] * target_months * recency_weight * frequency_weight
        ).round(2)
        
        # 현실적 범위로 조정
        if 'total_amount' in df.columns:
            df['predicted_next_amount'] = df['predicted_next_amount'].clip(
                lower=0, 
                upper=df['total_amount'] * 2
            )
        
        # 고객 등급 생성
        amount_quartiles = df['predicted_next_amount'].quantile([0.25, 0.5, 0.75])
        
        def categorize_customer(amount):
            if amount <= amount_quartiles[0.25]:
                return 'Low'
            elif amount <= amount_quartiles[0.5]:
                return 'Medium-Low'
            elif amount <= amount_quartiles[0.75]:
                return 'Medium-High'
            else:
                return 'High'
        
        df['customer_value_category'] = df['predicted_next_amount'].apply(categorize_customer)
        
        print(f"✅ 타겟 변수 생성 완료")
        print(f"   예측 금액 범위: £{df['predicted_next_amount'].min():.2f} ~ £{df['predicted_next_amount'].max():.2f}")
        print(f"   평균 예측 금액: £{df['predicted_next_amount'].mean():.2f}")
        
        return df
    
    def get_feature_importance_analysis(self, customer_features: pd.DataFrame) -> Dict:
        """특성 중요도 분석"""
        
        if self.customer_features is None:
            return {"error": "특성이 아직 생성되지 않았습니다."}
        
        analysis = {
            'total_features': len(customer_features.columns),
            'feature_categories': {
                'rfm_features': [],
                'behavioral_features': [],
                'statistical_features': [],
                'derived_features': []
            },
            'correlation_analysis': {},
            'feature_distributions': {}
        }
        
        # 특성 카테고리 분류
        for col in customer_features.columns:
            if any(x in col.lower() for x in ['recency', 'frequency', 'monetary', 'rfm']):
                analysis['feature_categories']['rfm_features'].append(col)
            elif any(x in col.lower() for x in ['return', 'segment', 'interval', 'sensitivity']):
                analysis['feature_categories']['behavioral_features'].append(col)
            elif any(x in col.lower() for x in ['avg', 'std', 'min', 'max', 'sum', 'count']):
                analysis['feature_categories']['statistical_features'].append(col)
            else:
                analysis['feature_categories']['derived_features'].append(col)
        
        # 수치형 특성 간 상관관계 분석
        numeric_features = customer_features.select_dtypes(include=[np.number]).columns
        if len(numeric_features) > 1:
            correlation_matrix = customer_features[numeric_features].corr()
            
            # 높은 상관관계 특성 쌍 찾기
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:  # 상관계수 절댓값 0.7 이상
                        high_corr_pairs.append({
                            'feature1': correlation_matrix.columns[i],
                            'feature2': correlation_matrix.columns[j],
                            'correlation': round(corr_value, 3)
                        })
            
            analysis['correlation_analysis']['high_correlation_pairs'] = high_corr_pairs
        
        return analysis
