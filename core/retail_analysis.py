"""
Online Retail 데이터 분석 모듈

이 모듈은 UCI Machine Learning Repository의 Online Retail 데이터셋을 
분석하기 위한 핵심 로직을 담고 있습니다.

ADP 실기 시험 준비를 위한 데이터 전처리와 특성 공학에 중점을 두고 구현되었습니다.
- 결측값 처리
- 이상치 탐지 및 처리  
- 집계 함수 활용
- 파생 변수 생성
- 시간 기반 특성 생성
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
from typing import Tuple, Dict, List, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")


class RetailDataProcessor:
    """
    Online Retail 데이터 전처리 및 특성 공학을 담당하는 클래스
    
    이 클래스는 ADP 실기에서 자주 다루는 데이터 처리 패턴들을 
    실무 수준에서 경험할 수 있도록 설계되었습니다.
    """
    
    def __init__(self):
        """초기화 메서드"""
        self.raw_data = None
        self.cleaned_data = None
        self.customer_features = None
        self.data_quality_report = {}
        
    def load_data(self) -> pd.DataFrame:
        """
        UCI ML Repository에서 Online Retail 데이터 로딩
        
        Returns:
            pd.DataFrame: 원본 데이터
            
        Note:
            실무에서는 보통 데이터베이스나 API에서 데이터를 가져오지만,
            여기서는 공개 데이터셋을 활용합니다.
        """
        try:
            # UCI ML Repository에서 데이터 로딩 시도
            from ucimlrepo import fetch_ucirepo
            print("📥 UCI ML Repository에서 데이터를 다운로드하는 중...")
            
            # Online Retail 데이터셋 (ID: 352)
            online_retail = fetch_ucirepo(id=352)
            
            # 데이터와 메타데이터 추출
            if hasattr(online_retail.data, 'features'):
                # 특성과 타겟이 분리된 경우
                data = online_retail.data.features.copy()
                if hasattr(online_retail.data, 'targets') and online_retail.data.targets is not None:
                    # 타겟이 있다면 병합
                    data = pd.concat([data, online_retail.data.targets], axis=1)
            else:
                # 전체 데이터가 하나로 되어 있는 경우
                data = online_retail.data.copy()
            
            print(f"✅ 데이터 로딩 완료: {data.shape[0]:,}개 레코드, {data.shape[1]}개 컬럼")
            
        except ImportError:
            print("⚠️  ucimlrepo 패키지가 없습니다. 대체 방법으로 데이터를 로딩합니다...")
            # 대체 방법: 직접 URL에서 다운로드
            data = self._load_data_fallback()
            
        except Exception as e:
            print(f"⚠️  UCI 데이터 로딩 실패: {e}")
            print("대체 방법으로 데이터를 로딩합니다...")
            data = self._load_data_fallback()
            
        self.raw_data = data.copy()
        return data
    
    def _load_data_fallback(self) -> pd.DataFrame:
        """
        대체 데이터 로딩 방법
        
        Returns:
            pd.DataFrame: 샘플 데이터 또는 다른 소스에서 로딩한 데이터
        """
        try:
            # Kaggle의 공개 데이터셋 URL 시도
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
            print(f"📥 직접 URL에서 데이터를 다운로드하는 중: {url}")
            
            data = pd.read_excel(url, engine='openpyxl')
            print(f"✅ 대체 방법으로 데이터 로딩 완료: {data.shape[0]:,}개 레코드")
            return data
            
        except Exception as e:
            print(f"⚠️  대체 방법도 실패: {e}")
            print("🔄 샘플 데이터를 생성합니다...")
            return self._generate_sample_data()
    
    def _generate_sample_data(self) -> pd.DataFrame:
        """
        네트워크 문제 시 사용할 샘플 데이터 생성
        
        Returns:
            pd.DataFrame: 실제 데이터와 유사한 구조의 샘플 데이터
        """
        print("🔧 샘플 데이터를 생성하는 중...")
        
        # 실제 Online Retail 데이터와 같은 구조로 샘플 생성
        np.random.seed(42)
        n_records = 10000
        
        # 날짜 범위: 2010년 12월 ~ 2011년 12월
        start_date = datetime(2010, 12, 1)
        end_date = datetime(2011, 12, 9)
        date_range = pd.date_range(start_date, end_date, freq='H')
        
        data = {
            'InvoiceNo': [f'C{np.random.randint(536365, 581587)}' if np.random.random() < 0.02 
                         else str(np.random.randint(536365, 581587)) for _ in range(n_records)],
            'StockCode': [f'{np.random.choice(["POST", "BANK", "M", "S", "AMAZONFEE"])}'  if np.random.random() < 0.05
                         else f'{np.random.randint(10000, 99999)}{np.random.choice(["", "A", "B", "C"])}' 
                         for _ in range(n_records)],
            'Description': [np.random.choice([
                'WHITE HANGING HEART T-LIGHT HOLDER',
                'WHITE METAL LANTERN', 
                'CREAM CUPID HEARTS COAT HANGER',
                'KNITTED UNION FLAG HOT WATER BOTTLE',
                'RED WOOLLY HOTTIE WHITE HEART',
                'SET 7 BABUSHKA NESTING BOXES',
                'GLASS STAR FROSTED T-LIGHT HOLDER',
                np.nan  # 일부 결측값 포함
            ]) for _ in range(n_records)],
            'Quantity': np.random.randint(-80, 80, n_records),  # 음수는 반품
            'InvoiceDate': np.random.choice(date_range, n_records),
            'UnitPrice': np.round(np.random.lognormal(1.5, 1) * np.random.choice([0.1, 0.5, 1, 2, 5, 10]), 2),
            'CustomerID': [np.random.randint(12346, 18287) if np.random.random() < 0.85 
                          else np.nan for _ in range(n_records)],  # 일부 결측값
            'Country': np.random.choice([
                'United Kingdom', 'France', 'Australia', 'Netherlands', 
                'Germany', 'Norway', 'EIRE', 'Spain', 'Belgium', 'Sweden'
            ], n_records, p=[0.7, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03, 0.02, 0.02, 0.02])
        }
        
        df = pd.DataFrame(data)
        print(f"✅ 샘플 데이터 생성 완료: {df.shape[0]:,}개 레코드")
        print("ℹ️  실제 프로젝트에서는 인터넷 연결 후 실제 데이터를 사용하세요.")
        
        return df
    
    def analyze_data_quality(self, data: pd.DataFrame) -> Dict:
        """
        데이터 품질 분석 - ADP 실기의 핵심 영역
        
        Args:
            data: 분석할 데이터프레임
            
        Returns:
            Dict: 데이터 품질 리포트
            
        Note:
            실무에서는 데이터 수집 후 가장 먼저 수행하는 작업입니다.
            ADP 실기에서도 자주 출제되는 영역이죠.
        """
        print("🔍 데이터 품질 분석을 시작합니다...")
        
        quality_report = {
            'total_records': len(data),
            'total_columns': len(data.columns),
            'memory_usage_mb': round(data.memory_usage(deep=True).sum() / 1024**2, 2),
            'duplicates': {
                'total_duplicate_rows': data.duplicated().sum(),
                'duplicate_invoices': data['InvoiceNo'].duplicated().sum() if 'InvoiceNo' in data.columns else 0
            },
            'missing_values': {},
            'data_types': {},
            'outliers': {},
            'data_range': {}
        }
        
        # 결측값 분석 - ADP 실기 필수 영역
        print("📊 결측값 분석 중...")
        for col in data.columns:
            missing_count = data[col].isnull().sum()
            missing_pct = (missing_count / len(data)) * 100
            quality_report['missing_values'][col] = {
                'count': missing_count,
                'percentage': round(missing_pct, 2)
            }
        
        # 데이터 타입 분석
        print("🔤 데이터 타입 분석 중...")
        for col in data.columns:
            quality_report['data_types'][col] = {
                'current_type': str(data[col].dtype),
                'non_null_count': data[col].count(),
                'unique_values': data[col].nunique()
            }
        
        # 수치형 컬럼의 이상치 분석 - ADP 실기 출제 빈도 높음
        print("📈 이상치 분석 중...")
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if data[col].count() > 0:  # 데이터가 있는 경우만
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col]
                
                quality_report['outliers'][col] = {
                    'Q1': round(Q1, 2),
                    'Q3': round(Q3, 2),
                    'IQR': round(IQR, 2),
                    'lower_bound': round(lower_bound, 2),
                    'upper_bound': round(upper_bound, 2),
                    'outlier_count': len(outliers),
                    'outlier_percentage': round((len(outliers) / data[col].count()) * 100, 2)
                }
        
        # 데이터 범위 분석
        print("📏 데이터 범위 분석 중...")
        for col in data.columns:
            if data[col].dtype in ['int64', 'float64']:
                quality_report['data_range'][col] = {
                    'min': data[col].min(),
                    'max': data[col].max(),
                    'mean': round(data[col].mean(), 2),
                    'std': round(data[col].std(), 2)
                }
            elif data[col].dtype == 'object':
                quality_report['data_range'][col] = {
                    'unique_count': data[col].nunique(),
                    'most_common': data[col].mode().iloc[0] if len(data[col].mode()) > 0 else None,
                    'sample_values': data[col].dropna().head(3).tolist()
                }
        
        self.data_quality_report = quality_report
        print("✅ 데이터 품질 분석 완료")
        
        return quality_report
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        데이터 정제 - ADP 실기 핵심 프로세스
        
        Args:
            data: 정제할 원본 데이터
            
        Returns:
            pd.DataFrame: 정제된 데이터
            
        Note:
            실무에서 가장 시간이 많이 걸리는 작업입니다.
            각 단계의 처리 이유와 방법을 이해하는 것이 중요해요.
        """
        print("🧹 데이터 정제를 시작합니다...")
        df = data.copy()
        
        print(f"정제 전: {len(df):,}개 레코드")
        
        # 1단계: 기본적인 데이터 타입 확인 및 변환
        print("1️⃣ 데이터 타입 검증 및 변환 중...")
        
        # InvoiceDate가 문자열인 경우 datetime으로 변환
        if df['InvoiceDate'].dtype == 'object':
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
        
        # CustomerID를 문자열로 변환 (결측값 유지)
        df['CustomerID'] = df['CustomerID'].astype('Int64')  # nullable integer
        
        # 2단계: 명백한 오류 데이터 제거
        print("2️⃣ 명백한 오류 데이터 제거 중...")
        
        initial_count = len(df)
        
        # 수량이 0인 레코드 제거 (의미없는 거래)
        df = df[df['Quantity'] != 0]
        print(f"   수량 0 제거: {initial_count - len(df):,}개 레코드 제거")
        
        # 단가가 0 이하인 레코드 제거 (잘못된 가격 정보)
        current_count = len(df)
        df = df[df['UnitPrice'] > 0]
        print(f"   단가 0 이하 제거: {current_count - len(df):,}개 레코드 제거")
        
        # InvoiceDate가 결측값인 레코드 제거
        current_count = len(df)
        df = df[df['InvoiceDate'].notna()]
        print(f"   날짜 결측값 제거: {current_count - len(df):,}개 레코드 제거")
        
        # 3단계: Description 결측값 처리
        print("3️⃣ 상품 설명 결측값 처리 중...")
        
        # StockCode를 기반으로 Description 결측값 보완
        description_mapping = df.groupby('StockCode')['Description'].apply(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown Product'
        ).to_dict()
        
        df['Description'] = df['Description'].fillna(df['StockCode'].map(description_mapping))
        df['Description'] = df['Description'].fillna('Unknown Product')
        
        # 4단계: 이상치 처리 - 비즈니스 로직 기반
        print("4️⃣ 이상치 처리 중...")
        
        # 극단적인 수량 (99% 범위를 벗어나는 값) 제거
        quantity_99 = df['Quantity'].quantile(0.99)
        quantity_1 = df['Quantity'].quantile(0.01)
        
        current_count = len(df)
        df = df[(df['Quantity'] >= quantity_1) & (df['Quantity'] <= quantity_99)]
        print(f"   극단적 수량 제거: {current_count - len(df):,}개 레코드 제거")
        
        # 극단적인 단가 (99.5% 범위를 벗어나는 값) 제거
        price_995 = df['UnitPrice'].quantile(0.995)
        
        current_count = len(df)
        df = df[df['UnitPrice'] <= price_995]
        print(f"   극단적 단가 제거: {current_count - len(df):,}개 레코드 제거")
        
        # 5단계: 총 거래 금액 계산
        print("5️⃣ 파생 변수 생성 중...")
        df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
        
        # 반품 여부 플래그 생성
        df['IsReturn'] = df['Quantity'] < 0
        
        # 월, 요일, 시간 정보 추출
        df['Year'] = df['InvoiceDate'].dt.year
        df['Month'] = df['InvoiceDate'].dt.month
        df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek  # 0=월요일
        df['Hour'] = df['InvoiceDate'].dt.hour
        
        print(f"✅ 데이터 정제 완료: {len(df):,}개 레코드 (원본 대비 {(len(df)/len(data)*100):.1f}% 유지)")
        
        self.cleaned_data = df.copy()
        return df
    
    def create_customer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        고객별 특성 생성 - ADP 실기의 하이라이트
        
        Args:
            data: 정제된 거래 데이터
            
        Returns:
            pd.DataFrame: 고객별 집계된 특성 데이터
            
        Note:
            이 함수에서 ADP 실기의 핵심인 groupby, agg, transform 등을
            실무 수준에서 활용합니다. 각 특성의 비즈니스적 의미도 중요해요.
        """
        print("🏗️ 고객별 특성 생성을 시작합니다...")
        
        # CustomerID가 있는 데이터만 사용 (B2C 거래)
        customer_data = data[data['CustomerID'].notna()].copy()
        print(f"분석 대상 고객 수: {customer_data['CustomerID'].nunique():,}명")
        print(f"분석 대상 거래 수: {len(customer_data):,}건")
        
        # 기본 집계 통계 - ADP 실기 필수 함수들
        print("📊 기본 통계량 계산 중...")
        
        customer_features = customer_data.groupby('CustomerID').agg({
            # 거래 관련 통계
            'InvoiceNo': ['nunique', 'count'],  # 거래 횟수, 총 구매 아이템 수
            'Quantity': ['sum', 'mean', 'std', 'min', 'max'],  # 수량 통계
            'UnitPrice': ['mean', 'std', 'min', 'max'],  # 단가 통계
            'TotalAmount': ['sum', 'mean', 'std', 'min', 'max'],  # 총액 통계
            
            # 시간 관련 통계
            'InvoiceDate': ['min', 'max', 'count'],  # 첫/마지막 구매일, 거래 빈도
            
            # 상품 관련 통계
            'StockCode': 'nunique',  # 구매한 고유 상품 수
            'Description': 'nunique',  # 구매한 고유 상품명 수
            
            # 반품 관련 통계
            'IsReturn': ['sum', 'mean'],  # 반품 횟수, 반품 비율
        }).round(2)
        
        # 컬럼명 정리 - 실무에서 중요한 가독성
        customer_features.columns = [
            'unique_invoices', 'total_items', 'total_quantity', 'avg_quantity', 
            'std_quantity', 'min_quantity', 'max_quantity', 'avg_unit_price', 
            'std_unit_price', 'min_unit_price', 'max_unit_price', 'total_amount',
            'avg_amount', 'std_amount', 'min_amount', 'max_amount', 'first_purchase',
            'last_purchase', 'purchase_frequency', 'unique_products', 'unique_descriptions',
            'return_count', 'return_rate'
        ]
        
        # 고급 파생 변수 생성 - 실무에서 자주 사용되는 패턴
        print("🔧 고급 파생 변수 생성 중...")
        
        # 1. 고객 생애주기 관련 변수
        customer_features['customer_lifespan_days'] = (
            customer_features['last_purchase'] - customer_features['first_purchase']
        ).dt.days
        
        customer_features['avg_days_between_purchases'] = (
            customer_features['customer_lifespan_days'] / 
            customer_features['unique_invoices'].where(customer_features['unique_invoices'] > 1, np.nan)
        ).round(2)
        
        # 2. 구매 행동 패턴 변수
        customer_features['avg_items_per_transaction'] = (
            customer_features['total_items'] / customer_features['unique_invoices']
        ).round(2)
        
        customer_features['price_sensitivity'] = (
            customer_features['std_unit_price'] / customer_features['avg_unit_price']
        ).round(3)
        
        # 3. 고객 가치 관련 변수 (RFM 분석의 기초)
        # Recency: 마지막 구매일부터 현재까지의 일수
        analysis_date = customer_features['last_purchase'].max()
        customer_features['recency_days'] = (
            analysis_date - customer_features['last_purchase']
        ).dt.days
        
        # Frequency: 구매 빈도 (이미 unique_invoices로 계산됨)
        customer_features['frequency'] = customer_features['unique_invoices']
        
        # Monetary: 총 구매 금액 (이미 total_amount로 계산됨)
        customer_features['monetary'] = customer_features['total_amount']
        
        # 4. 계절성 및 시간 패턴 분석
        print("📅 시간 패턴 분석 중...")
        
        # 월별 구매 분포
        monthly_purchases = customer_data.groupby(['CustomerID', 'Month']).size().unstack(fill_value=0)
        customer_features['most_active_month'] = monthly_purchases.idxmax(axis=1)
        customer_features['purchase_month_variety'] = (monthly_purchases > 0).sum(axis=1)
        
        # 요일별 구매 분포  
        daily_purchases = customer_data.groupby(['CustomerID', 'DayOfWeek']).size().unstack(fill_value=0)
        customer_features['most_active_day'] = daily_purchases.idxmax(axis=1)
        customer_features['weekend_purchase_ratio'] = (
            (daily_purchases[5] + daily_purchases[6]) / daily_purchases.sum(axis=1)
        ).round(3)
        
        # 5. 상품 관련 고급 변수
        print("🛍️ 상품 선호도 분석 중...")
        
        # 상품 다양성 지수 (심슨 다양성 지수 응용)
        product_diversity = customer_data.groupby('CustomerID')['StockCode'].apply(
            lambda x: 1 - sum((x.value_counts() / len(x)) ** 2)
        ).round(3)
        customer_features['product_diversity_index'] = product_diversity
        
        # 평균 바스켓 크기 (한 번의 구매에서 구매하는 상품 종류 수)
        basket_sizes = customer_data.groupby(['CustomerID', 'InvoiceNo'])['StockCode'].nunique()
        avg_basket_size = basket_sizes.groupby('CustomerID').mean().round(2)
        customer_features['avg_basket_size'] = avg_basket_size
        
        # 6. 이상 행동 탐지 변수
        print("🚨 이상 행동 패턴 탐지 중...")
        
        # 대량 구매 비율 (상위 10% 수량을 차지하는 거래 비율)
        large_quantity_threshold = customer_data['Quantity'].quantile(0.9)
        large_purchases = customer_data[customer_data['Quantity'] >= large_quantity_threshold]
        large_purchase_ratio = large_purchases.groupby('CustomerID').size() / customer_features['unique_invoices']
        customer_features['large_purchase_ratio'] = large_purchase_ratio.fillna(0).round(3)
        
        # 고가 구매 비율 (상위 10% 금액을 차지하는 거래 비율)
        high_value_threshold = customer_data['TotalAmount'].quantile(0.9)
        high_value_purchases = customer_data[customer_data['TotalAmount'] >= high_value_threshold]
        high_value_ratio = high_value_purchases.groupby('CustomerID').size() / customer_features['unique_invoices']
        customer_features['high_value_ratio'] = high_value_ratio.fillna(0).round(3)
        
        # 7. 정규화 및 스케일링을 위한 로그 변환 변수
        print("📐 로그 변환 변수 생성 중...")
        
        # 금액 관련 변수들의 로그 변환 (양의 값만)
        for col in ['total_amount', 'avg_amount', 'monetary']:
            if col in customer_features.columns:
                customer_features[f'log_{col}'] = np.log1p(customer_features[col].clip(lower=0))
        
        # 8. 결측값 처리
        print("🔧 결측값 최종 처리 중...")
        
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
        
        print(f"✅ 고객별 특성 생성 완료: {len(customer_features):,}명 고객, {len(customer_features.columns)}개 특성")
        
        # 특성 요약 정보 출력
        print("\n📋 생성된 특성 요약:")
        feature_categories = {
            '기본 거래 통계': ['unique_invoices', 'total_items', 'total_quantity', 'total_amount'],
            '구매 행동 패턴': ['avg_items_per_transaction', 'price_sensitivity', 'product_diversity_index'],
            'RFM 분석': ['recency_days', 'frequency', 'monetary'],
            '시간 패턴': ['customer_lifespan_days', 'most_active_month', 'weekend_purchase_ratio'],
            '상품 관련': ['unique_products', 'avg_basket_size'],
            '이상 행동': ['return_rate', 'large_purchase_ratio', 'high_value_ratio']
        }
        
        for category, features in feature_categories.items():
            available_features = [f for f in features if f in customer_features.columns]
            print(f"  {category}: {len(available_features)}개 - {', '.join(available_features[:3])}{'...' if len(available_features) > 3 else ''}")
        
        self.customer_features = customer_features.copy()
        return customer_features
    
    def create_target_variable(self, customer_features: pd.DataFrame, target_months: int = 3) -> pd.DataFrame:
        """
        예측 타겟 변수 생성: 다음 N개월 구매 예상 금액
        
        Args:
            customer_features: 고객별 특성 데이터
            target_months: 예측 기간 (개월)
            
        Returns:
            pd.DataFrame: 타겟 변수가 포함된 특성 데이터
            
        Note:
            실무에서는 비즈니스 요구사항에 따라 타겟을 정의합니다.
            여기서는 학습 목적으로 간단한 휴리스틱을 사용합니다.
        """
        print(f"🎯 타겟 변수 생성: 다음 {target_months}개월 구매 예상 금액")
        
        df = customer_features.copy()
        
        # 방법 1: 과거 평균 기반 예측 (베이스라인)
        # 월평균 구매 금액 계산
        df['monthly_avg_amount'] = df['total_amount'] / (df['customer_lifespan_days'] / 30.44).clip(lower=1)
        
        # 최근성을 고려한 가중치 적용
        # Recency가 낮을수록 (최근 구매) 더 높은 구매 확률
        recency_weight = np.exp(-df['recency_days'] / 30)  # 30일마다 가중치 감소
        
        # 구매 빈도를 고려한 가중치
        frequency_weight = np.log1p(df['frequency']) / np.log1p(df['frequency'].max())
        
        # 계절성 고려 (현재 월과 고객의 최활성 월 비교)
        current_month = df['last_purchase'].dt.month.mode().iloc[0]  # 분석 기준 월
        seasonal_weight = df['most_active_month'].apply(
            lambda x: 1.2 if x == current_month else 0.8
        )
        
        # 최종 예측값 계산
        df['predicted_next_amount'] = (
            df['monthly_avg_amount'] * target_months * 
            recency_weight * frequency_weight * seasonal_weight
        ).round(2)
        
        # 현실적 범위로 조정 (과거 최대 구매액의 2배를 상한으로)
        df['predicted_next_amount'] = df['predicted_next_amount'].clip(
            lower=0, 
            upper=df['total_amount'] * 2
        )
        
        # 방법 2: 단순화된 카테고리 타겟 (분류 문제용)
        # 예측 금액 기준 고객 등급
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
        print(f"   고객 등급 분포:")
        print(df['customer_value_category'].value_counts().to_string())
        
        return df
    
    def prepare_modeling_data(self, customer_features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        머신러닝 모델링을 위한 데이터 준비
        
        Args:
            customer_features: 타겟이 포함된 고객 특성 데이터
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: 특성 행렬 X, 타겟 벡터 y
            
        Note:
            실무에서는 모델링 전 데이터 준비가 성공의 핵심입니다.
            특성 선택, 스케일링, 인코딩 등을 체계적으로 수행해야 해요.
        """
        print("⚙️ 모델링 데이터 준비 중...")
        
        df = customer_features.copy()
        
        # 타겟 변수 분리
        target_col = 'predicted_next_amount'
        y = df[target_col].copy()
        
        # 모델링에 사용할 특성 선택
        # 타겟 생성에 직접 사용된 변수들은 제외 (데이터 누수 방지)
        exclude_cols = [
            target_col, 'customer_value_category', 'monthly_avg_amount',
            'first_purchase', 'last_purchase', 'predicted_next_amount'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X = df[feature_cols].copy()
        
        # 범주형 변수 인코딩
        categorical_cols = X.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print(f"   범주형 변수 인코딩: {list(categorical_cols)}")
            X = pd.get_dummies(X, columns=categorical_cols, prefix=categorical_cols)
        
        print(f"✅ 모델링 데이터 준비 완료")
        print(f"   특성 수: {X.shape[1]}개")
        print(f"   샘플 수: {X.shape[0]}개")
        print(f"   타겟 변수: {target_col}")
        
        return X, y


class RetailVisualizer:
    """
    Online Retail 데이터 시각화 전담 클래스
    
    EDA와 모델 결과 시각화를 담당합니다.
    """
    
    @staticmethod
    def create_data_quality_dashboard(quality_report: Dict) -> go.Figure:
        """데이터 품질 리포트 시각화"""
        
        # 서브플롯 생성
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('결측값 분포', '데이터 타입 분포', '이상치 현황', '메모리 사용량'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "indicator"}]]
        )
        
        # 1. 결측값 분포
        missing_data = quality_report['missing_values']
        cols = list(missing_data.keys())
        missing_pcts = [missing_data[col]['percentage'] for col in cols]
        
        fig.add_trace(
            go.Bar(x=cols, y=missing_pcts, name="결측값 %", marker_color='lightcoral'),
            row=1, col=1
        )
        
        # 2. 데이터 타입 분포
        type_counts = {}
        for col, info in quality_report['data_types'].items():
            dtype = info['current_type']
            type_counts[dtype] = type_counts.get(dtype, 0) + 1
        
        fig.add_trace(
            go.Pie(labels=list(type_counts.keys()), values=list(type_counts.values()), name="데이터 타입"),
            row=1, col=2
        )
        
        # 3. 이상치 현황
        outlier_data = quality_report['outliers']
        if outlier_data:
            outlier_cols = list(outlier_data.keys())
            outlier_pcts = [outlier_data[col]['outlier_percentage'] for col in outlier_cols]
            
            fig.add_trace(
                go.Bar(x=outlier_cols, y=outlier_pcts, name="이상치 %", marker_color='orange'),
                row=2, col=1
            )
        
        # 4. 메모리 사용량
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=quality_report['memory_usage_mb'],
                title={'text': "메모리 (MB)"},
                gauge={'axis': {'range': [None, quality_report['memory_usage_mb'] * 1.5]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, quality_report['memory_usage_mb'] * 0.5], 'color': "lightgray"},
                                {'range': [quality_report['memory_usage_mb'] * 0.5, quality_report['memory_usage_mb']], 'color': "gray"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75, 'value': quality_report['memory_usage_mb'] * 1.2}}
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="📊 데이터 품질 종합 대시보드",
            showlegend=False,
            height=600
        )
        
        return fig
    
    @staticmethod  
    def create_customer_distribution_plots(customer_features: pd.DataFrame) -> go.Figure:
        """고객 특성 분포 시각화"""
        
        # 주요 지표들 선택
        key_metrics = ['total_amount', 'frequency', 'recency_days', 'unique_products']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f'{metric} 분포' for metric in key_metrics]
        )
        
        for i, metric in enumerate(key_metrics):
            row = i // 2 + 1
            col = i % 2 + 1
            
            if metric in customer_features.columns:
                fig.add_trace(
                    go.Histogram(x=customer_features[metric], name=metric, nbinsx=30),
                    row=row, col=col
                )
        
        fig.update_layout(
            title_text="👥 고객 특성 분포 분석",
            showlegend=False,
            height=600
        )
        
        return fig
