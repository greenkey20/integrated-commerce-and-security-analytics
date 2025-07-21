"""
데이터 개요 페이지

기존 customer_segmentation_app.py의 "데이터 개요" 메뉴 내용을 모듈화
"""

import streamlit as st
import pandas as pd

# 새로운 데이터 계층 사용
try:
    from data import DataProcessor
except ImportError:
    # 폴백: 기본 pandas 사용
    DataProcessor = None


def show_data_overview_page():
    """데이터 개요 페이지를 표시하는 함수"""
    st.header("📊 데이터 개요")
    
    # 데이터 로드
    if DataProcessor is not None:
        try:
            data_processor = DataProcessor()
            data = data_processor.load_data()
            validation_results = data_processor.validate_data(data)
        except Exception as e:
            st.warning(f"DataProcessor 사용 중 오류: {e}")
            DataProcessor = None
    
    if DataProcessor is None:
        # 기본 데이터 로드 (Mall Customer 데이터셋)
        try:
            # 기본 URL에서 데이터 로드
            data_url = "https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Datasets/Mall_Customers.csv"
            data = pd.read_csv(data_url)
            
            # 기본 검증
            validation_results = {
                'total_rows': len(data),
                'total_columns': len(data.columns),
                'data_types': data.dtypes,
                'has_missing': data.isnull().any().any(),
                'missing_values': data.isnull().sum()
            }
            
        except Exception as e:
            st.error(f"데이터 로딩 실패: {e}")
            st.info("샘플 데이터를 사용합니다.")
            
            # 샘플 데이터 생성
            import numpy as np
            np.random.seed(42)
            
            data = pd.DataFrame({
                'CustomerID': range(1, 201),
                'Gender': np.random.choice(['Male', 'Female'], 200),
                'Age': np.random.randint(18, 70, 200),
                'Annual Income (k$)': np.random.randint(15, 140, 200),
                'Spending Score (1-100)': np.random.randint(1, 100, 200)
            })
            
            validation_results = {
                'total_rows': len(data),
                'total_columns': len(data.columns), 
                'data_types': data.dtypes,
                'has_missing': False,
                'missing_values': data.isnull().sum()
            }

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("데이터셋 정보")
        st.write(f"전체 고객 수: {validation_results['total_rows']:,}명")
        st.write(f"특성 수: {validation_results['total_columns']}개")
        st.write("데이터 타입:")
        
        # DataFrame으로 변환하여 안전하게 출력
        dtypes_df = pd.DataFrame(
            {
                "컬럼명": data.columns,
                "데이터 타입": [str(dtype) for dtype in validation_results['data_types']],
            }
        )
        st.dataframe(dtypes_df, use_container_width=True)

    with col2:
        st.subheader("기본 통계")
        st.write(data.describe())

    st.subheader("데이터 미리보기")
    st.dataframe(data.head(10))

    # 결측값 확인
    st.subheader("데이터 품질 검사")
    missing_values = validation_results['missing_values']
    
    if not validation_results['has_missing']:
        st.success("✅ 결측값이 없습니다.")
    else:
        st.warning("⚠️ 결측값이 발견되었습니다:")
        st.write(missing_values[missing_values > 0])
        
    return data  # 다른 페이지에서 사용할 수 있도록 반환
