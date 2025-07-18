"""
Online Retail 데이터 분석 모듈 - 동적 컬럼 매핑 버전

이 모듈은 UCI Machine Learning Repository의 Online Retail 데이터셋을 
분석하기 위한 핵심 로직을 담고 있습니다.

실제 데이터 구조에 맞춰 동적으로 컬럼을 매핑하여 처리합니다.
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
    
    실제 데이터 구조에 맞춰 동적으로 컬럼을 매핑하여 처리합니다.
    """
    
    def __init__(self):
        """초기화 메서드"""
        self.raw_data = None
        self.cleaned_data = None
        self.customer_features = None
        self.data_quality_report = {}
        self.column_mapping = {}
        
    def load_data(self) -> pd.DataFrame:
        """
        UCI ML Repository에서 Online Retail 데이터 로딩
        
        Returns:
            pd.DataFrame: 원본 데이터
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
            print(f"📊 실제 컬럼명: {list(data.columns)}")
            print(f"📊 데이터 타입:\n{data.dtypes}")
            print(f"📊 데이터 샘플 (5행):\n{data.head()}")
            
            # 컬럼 매핑 생성
            self.column_mapping = self._create_column_mapping(data.columns)
            print(f"🔄 컬럼 매핑: {self.column_mapping}")
            
        except ImportError:
            print("⚠️  ucimlrepo 패키지가 없습니다. 대체 방법으로 데이터를 로딩합니다...")
            data = self._load_data_fallback()
            
        except Exception as e:
            print(f"⚠️  UCI 데이터 로딩 실패: {e}")
            print("대체 방법으로 데이터를 로딩합니다...")
            data = self._load_data_fallback()
            
        self.raw_data = data.copy()
        return data
    
    def _create_column_mapping(self, columns: list) -> dict:
        """실제 데이터 컬럼을 표준 컬럼명에 매핑"""
        
        column_mapping = {
            'invoice_no': None,
            'stock_code': None,
            'description': None,
            'quantity': None,
            'invoice_date': None,
            'unit_price': None,
            'customer_id': None,
            'country': None
        }
        
        # 컬럼명 매핑 룰 (case-insensitive)
        for col in columns:
            col_lower = col.lower().replace(' ', '_').replace('-', '_')
            
            if any(x in col_lower for x in ['invoice']) and any(x in col_lower for x in ['no', 'number', 'id']):
                column_mapping['invoice_no'] = col
            elif any(x in col_lower for x in ['invoice']) and any(x in col_lower for x in ['date', 'time']):
                column_mapping['invoice_date'] = col
            elif any(x in col_lower for x in ['stock', 'item', 'product']) and any(x in col_lower for x in ['code', 'id', 'no']):
                column_mapping['stock_code'] = col
            elif any(x in col_lower for x in ['description', 'desc', 'name']) and 'customer' not in col_lower:
                column_mapping['description'] = col
            elif any(x in col_lower for x in ['quantity', 'qty']) and 'unit' not in col_lower:
                column_mapping['quantity'] = col
            elif any(x in col_lower for x in ['price', 'cost']) and any(x in col_lower for x in ['unit', 'per']):
                column_mapping['unit_price'] = col
            elif any(x in col_lower for x in ['customer', 'client']) and any(x in col_lower for x in ['id', 'no']):
                column_mapping['customer_id'] = col
            elif any(x in col_lower for x in ['country', 'nation']):
                column_mapping['country'] = col
        
        return column_mapping
    
    def _load_data_fallback(self) -> pd.DataFrame:
        """대체 데이터 로딩 방법"""
        try:
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
            print(f"📥 직접 URL에서 데이터를 다운로드하는 중: {url}")
            
            data = pd.read_excel(url, engine='openpyxl')
            print(f"✅ 대체 방법으로 데이터 로딩 완료: {data.shape[0]:,}개 레코드")
            
            # 컬럼 매핑 생성
            self.column_mapping = self._create_column_mapping(data.columns)
            print(f"🔄 컬럼 매핑: {self.column_mapping}")
            
            return data
            
        except Exception as e:
            print(f"⚠️  대체 방법도 실패: {e}")
            print("🔄 샘플 데이터를 생성합니다...")
            return self._generate_sample_data()
    
    def _generate_sample_data(self) -> pd.DataFrame:
        """샘플 데이터 생성"""
        print("🔧 샘플 데이터를 생성하는 중...")
        
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
                np.nan
            ]) for _ in range(n_records)],
            'Quantity': np.random.randint(-80, 80, n_records),
            'InvoiceDate': np.random.choice(date_range, n_records),
            'UnitPrice': np.round(np.random.lognormal(1.5, 1) * np.random.choice([0.1, 0.5, 1, 2, 5, 10]), 2),
            'CustomerID': [np.random.randint(12346, 18287) if np.random.random() < 0.85 
                          else np.nan for _ in range(n_records)],
            'Country': np.random.choice([
                'United Kingdom', 'France', 'Australia', 'Netherlands', 
                'Germany', 'Norway', 'EIRE', 'Spain', 'Belgium', 'Sweden'
            ], n_records, p=[0.7, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03, 0.02, 0.02, 0.02])
        }
        
        df = pd.DataFrame(data)
        
        # 컬럼 매핑 생성
        self.column_mapping = self._create_column_mapping(df.columns)
        print(f"🔄 샘플 데이터 컬럼 매핑: {self.column_mapping}")
        
        print(f"✅ 샘플 데이터 생성 완료: {df.shape[0]:,}개 레코드")
        return df
    
    def analyze_data_quality(self, data: pd.DataFrame) -> Dict:
        """데이터 품질 분석"""
        print("🔍 데이터 품질 분석을 시작합니다...")
        
        quality_report = {
            'total_records': len(data),
            'total_columns': len(data.columns),
            'memory_usage_mb': round(data.memory_usage(deep=True).sum() / 1024**2, 2),
            'duplicates': {
                'total_duplicate_rows': data.duplicated().sum(),
            },
            'missing_values': {},
            'data_types': {},
            'outliers': {},
            'data_range': {}
        }
        
        # 결측값 분석
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
        
        # 수치형 컬럼의 이상치 분석
        print("📈 이상치 분석 중...")
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if data[col].count() > 0:
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
        """데이터 정제 - 동적 컬럼 매핑 지원"""
        print("🧹 데이터 정제를 시작합니다...")
        df = data.copy()
        
        print(f"정제 전: {len(df):,}개 레코드")
        print(f"사용 가능한 컬럼: {list(df.columns)}")
        print(f"컬럼 매핑: {self.column_mapping}")
        
        # 1단계: 기본적인 데이터 타입 확인 및 변환
        print("1️⃣ 데이터 타입 검증 및 변환 중...")
        
        # InvoiceDate 처리
        invoice_date_col = self.column_mapping.get('invoice_date')
        if invoice_date_col and invoice_date_col in df.columns:
            if df[invoice_date_col].dtype == 'object':
                df[invoice_date_col] = pd.to_datetime(df[invoice_date_col], errors='coerce')
        
        # CustomerID 처리
        customer_id_col = self.column_mapping.get('customer_id')
        if customer_id_col and customer_id_col in df.columns:
            df[customer_id_col] = df[customer_id_col].astype('Int64')
        
        # 2단계: 명백한 오류 데이터 제거
        print("2️⃣ 명백한 오류 데이터 제거 중...")
        
        initial_count = len(df)
        
        # 수량이 0인 레코드 제거
        quantity_col = self.column_mapping.get('quantity')
        if quantity_col and quantity_col in df.columns:
            df = df[df[quantity_col] != 0]
            print(f"   수량 0 제거: {initial_count - len(df):,}개 레코드 제거")
        
        # 단가가 0 이하인 레코드 제거
        unit_price_col = self.column_mapping.get('unit_price')
        if unit_price_col and unit_price_col in df.columns:
            current_count = len(df)
            df = df[df[unit_price_col] > 0]
            print(f"   단가 0 이하 제거: {current_count - len(df):,}개 레코드 제거")
        
        # InvoiceDate가 결측값인 레코드 제거
        if invoice_date_col and invoice_date_col in df.columns:
            current_count = len(df)
            df = df[df[invoice_date_col].notna()]
            print(f"   날짜 결측값 제거: {current_count - len(df):,}개 레코드 제거")
        
        # 3단계: Description 결측값 처리
        print("3️⃣ 상품 설명 결측값 처리 중...")
        
        description_col = self.column_mapping.get('description')
        stock_code_col = self.column_mapping.get('stock_code')
        
        if description_col and description_col in df.columns:
            try:
                if stock_code_col and stock_code_col in df.columns:
                    # StockCode가 유효한 데이터만 사용
                    valid_stock_data = df[df[stock_code_col].notna() & (df[stock_code_col] != '')].copy()
                    
                    if len(valid_stock_data) > 0:
                        # StockCode별로 Description의 최빈값 계산
                        description_mapping = {}
                        for stock_code, group in valid_stock_data.groupby(stock_code_col)[description_col]:
                            valid_descriptions = group.dropna()
                            if len(valid_descriptions) > 0:
                                mode_values = valid_descriptions.mode()
                                if len(mode_values) > 0:
                                    description_mapping[stock_code] = mode_values.iloc[0]
                                else:
                                    description_mapping[stock_code] = 'Unknown Product'
                            else:
                                description_mapping[stock_code] = 'Unknown Product'
                        
                        # Description 결측값 보완
                        df[description_col] = df[description_col].fillna(
                            df[stock_code_col].map(description_mapping)
                        )
                        
                        print(f"   StockCode 기반 Description 보완: {len(description_mapping)}개 상품코드 매핑")
                    else:
                        print("   유효한 StockCode 데이터가 없어 기본값으로 대체")
                else:
                    print("   StockCode 컬럼이 없어 기본값으로 대체")
                
                # 남은 결측값을 기본값으로 대체
                df[description_col] = df[description_col].fillna('Unknown Product')
                
                missing_count = df[description_col].isnull().sum()
                print(f"   최종 Description 결측값: {missing_count}개")
                
            except Exception as e:
                print(f"   ⚠️ Description 결측값 처리 중 오류 발생: {str(e)}")
                print("   모든 결측값을 'Unknown Product'로 대체합니다.")
                df[description_col] = df[description_col].fillna('Unknown Product')
        
        # 4단계: 이상치 처리
        print("4️⃣ 이상치 처리 중...")
        
        # 극단적인 수량 제거
        if quantity_col and quantity_col in df.columns:
            quantity_99 = df[quantity_col].quantile(0.99)
            quantity_1 = df[quantity_col].quantile(0.01)
            
            current_count = len(df)
            df = df[(df[quantity_col] >= quantity_1) & (df[quantity_col] <= quantity_99)]
            print(f"   극단적 수량 제거: {current_count - len(df):,}개 레코드 제거")
        
        # 극단적인 단가 제거
        if unit_price_col and unit_price_col in df.columns:
            price_995 = df[unit_price_col].quantile(0.995)
            
            current_count = len(df)
            df = df[df[unit_price_col] <= price_995]
            print(f"   극단적 단가 제거: {current_count - len(df):,}개 레코드 제거")
        
        # 5단계: 파생 변수 생성
        print("5️⃣ 파생 변수 생성 중...")
        
        # 총 거래 금액 계산
        if quantity_col and unit_price_col and quantity_col in df.columns and unit_price_col in df.columns:
            df['TotalAmount'] = df[quantity_col] * df[unit_price_col]
        
        # 반품 여부 플래그 생성
        if quantity_col and quantity_col in df.columns:
            df['IsReturn'] = df[quantity_col] < 0
        
        # 월, 요일, 시간 정보 추출
        if invoice_date_col and invoice_date_col in df.columns:
            df['Year'] = df[invoice_date_col].dt.year
            df['Month'] = df[invoice_date_col].dt.month
            df['DayOfWeek'] = df[invoice_date_col].dt.dayofweek
            df['Hour'] = df[invoice_date_col].dt.hour
        
        print(f"✅ 데이터 정제 완료: {len(df):,}개 레코드 (원본 대비 {(len(df)/len(data)*100):.1f}% 유지)")
        
        self.cleaned_data = df.copy()
        return df
    
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
    
    def prepare_modeling_data(self, customer_features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """머신러닝 모델링을 위한 데이터 준비"""
        print("⚙️ 모델링 데이터 준비 중...")
        
        df = customer_features.copy()
        
        # 타겟 변수 분리
        target_col = 'predicted_next_amount'
        y = df[target_col].copy()
        
        # 모델링에 사용할 특성 선택
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
    """Online Retail 데이터 시각화 전담 클래스"""
    
    @staticmethod
    def create_data_quality_dashboard(quality_report: Dict) -> go.Figure:
        """데이터 품질 리포트 시각화"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('결측값 분포', '데이터 타입 분포', '이상치 현황', '메모리 사용량'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "indicator"}]]
        )
        
        # 결측값 분포
        missing_data = quality_report['missing_values']
        cols = list(missing_data.keys())
        missing_pcts = [missing_data[col]['percentage'] for col in cols]
        
        fig.add_trace(
            go.Bar(x=cols, y=missing_pcts, name="결측값 %", marker_color='lightcoral'),
            row=1, col=1
        )
        
        # 데이터 타입 분포
        type_counts = {}
        for col, info in quality_report['data_types'].items():
            dtype = info['current_type']
            type_counts[dtype] = type_counts.get(dtype, 0) + 1
        
        fig.add_trace(
            go.Pie(labels=list(type_counts.keys()), values=list(type_counts.values()), name="데이터 타입"),
            row=1, col=2
        )
        
        # 이상치 현황
        outlier_data = quality_report['outliers']
        if outlier_data:
            outlier_cols = list(outlier_data.keys())
            outlier_pcts = [outlier_data[col]['outlier_percentage'] for col in outlier_cols]
            
            fig.add_trace(
                go.Bar(x=outlier_cols, y=outlier_pcts, name="이상치 %", marker_color='orange'),
                row=2, col=1
            )
        
        # 메모리 사용량
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
