"""
Online Retail 데이터 정제 및 전처리 모듈

이 모듈은 Online Retail 데이터의 정제, 전처리, 
그리고 기본적인 파생 변수 생성을 담당합니다.
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Optional

warnings.filterwarnings("ignore")


class RetailDataProcessor:
    """
    Online Retail 데이터 정제 및 전처리를 담당하는 클래스
    
    실제 데이터 구조에 맞춰 동적으로 컬럼을 매핑하여 처리합니다.
    """
    
    def __init__(self, column_mapping: Dict[str, str]):
        """
        초기화 메서드
        
        Args:
            column_mapping: 컬럼 매핑 정보
        """
        self.column_mapping = column_mapping
        self.cleaned_data = None
        
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
    
    def validate_data_quality(self, data: pd.DataFrame) -> Dict:
        """정제된 데이터의 품질 검증"""
        print("🔍 정제된 데이터 품질 검증 중...")
        
        validation_report = {
            'total_records': len(data),
            'data_quality_score': 0,
            'issues_found': [],
            'quality_metrics': {}
        }
        
        score = 100
        
        # 결측값 검사
        missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        if missing_pct > 5:
            validation_report['issues_found'].append(f"높은 결측값 비율: {missing_pct:.1f}%")
            score -= 20
        elif missing_pct > 1:
            score -= 10
        
        validation_report['quality_metrics']['missing_percentage'] = round(missing_pct, 2)
        
        # 중복 레코드 검사
        duplicate_pct = (data.duplicated().sum() / len(data)) * 100
        if duplicate_pct > 1:
            validation_report['issues_found'].append(f"중복 레코드 발견: {duplicate_pct:.1f}%")
            score -= 15
        
        validation_report['quality_metrics']['duplicate_percentage'] = round(duplicate_pct, 2)
        
        # 데이터 일관성 검사
        quantity_col = self.column_mapping.get('quantity')
        unit_price_col = self.column_mapping.get('unit_price')
        
        if quantity_col and unit_price_col and quantity_col in data.columns and unit_price_col in data.columns:
            # 수량과 단가가 모두 양수인지 확인
            invalid_quantity = (data[quantity_col] == 0).sum()
            invalid_price = (data[unit_price_col] <= 0).sum()
            
            if invalid_quantity > 0:
                validation_report['issues_found'].append(f"수량 0인 레코드: {invalid_quantity}개")
                score -= 10
            
            if invalid_price > 0:
                validation_report['issues_found'].append(f"단가 0 이하인 레코드: {invalid_price}개")
                score -= 10
        
        # 날짜 유효성 검사
        invoice_date_col = self.column_mapping.get('invoice_date')
        if invoice_date_col and invoice_date_col in data.columns:
            invalid_dates = data[invoice_date_col].isnull().sum()
            if invalid_dates > 0:
                validation_report['issues_found'].append(f"유효하지 않은 날짜: {invalid_dates}개")
                score -= 10
        
        validation_report['data_quality_score'] = max(0, score)
        
        # 품질 등급 부여
        if score >= 90:
            validation_report['quality_grade'] = 'A (우수)'
        elif score >= 80:
            validation_report['quality_grade'] = 'B (양호)'
        elif score >= 70:
            validation_report['quality_grade'] = 'C (보통)'
        else:
            validation_report['quality_grade'] = 'D (개선 필요)'
        
        print(f"✅ 데이터 품질 검증 완료: {validation_report['quality_grade']}")
        
        return validation_report
    
    def get_preprocessing_summary(self) -> Dict:
        """전처리 요약 정보 반환"""
        if self.cleaned_data is None:
            return {"error": "데이터가 아직 정제되지 않았습니다."}
        
        summary = {
            'total_records': len(self.cleaned_data),
            'total_columns': len(self.cleaned_data.columns),
            'derived_columns': [],
            'column_types': {},
            'memory_usage_mb': round(self.cleaned_data.memory_usage(deep=True).sum() / 1024**2, 2)
        }
        
        # 파생 변수 식별
        base_columns = list(self.column_mapping.values())
        for col in self.cleaned_data.columns:
            if col not in base_columns:
                summary['derived_columns'].append(col)
        
        # 컬럼 타입 정보
        for col in self.cleaned_data.columns:
            summary['column_types'][col] = str(self.cleaned_data[col].dtype)
        
        return summary
