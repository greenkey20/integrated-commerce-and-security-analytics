"""
Online Retail 데이터 로딩 및 품질 분석 모듈

이 모듈은 UCI Machine Learning Repository의 Online Retail 데이터셋을 
로딩하고 데이터 품질을 분석하는 기능을 제공합니다.
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional

warnings.filterwarnings("ignore")


class RetailDataLoader:
    """
    Online Retail 데이터 로딩 및 품질 분석을 담당하는 클래스
    
    실제 데이터 구조에 맞춰 동적으로 컬럼을 매핑하여 처리합니다.
    """
    
    def __init__(self):
        """초기화 메서드"""
        self.raw_data = None
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
    
    def get_column_mapping(self) -> dict:
        """컬럼 매핑 정보 반환"""
        return self.column_mapping.copy()
