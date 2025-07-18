"""
데이터 개요 페이지

기존 customer_segmentation_app.py의 "데이터 개요" 메뉴 내용을 모듈화
"""

import streamlit as st
import pandas as pd
from core.data_processing import DataProcessor


def show_data_overview_page():
    """데이터 개요 페이지를 표시하는 함수"""
    st.header("📊 데이터 개요")
    
    # 데이터 로드
    data_processor = DataProcessor()
    data = data_processor.load_data()
    
    # 데이터 검증
    validation_results = data_processor.validate_data(data)

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
