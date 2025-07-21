"""
고객 데이터 개요 페이지

고객 세분화를 위한 데이터 개요 및 품질 검사 UI
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Core 모듈에서 비즈니스 로직 import
from core.segmentation.data_processing import CustomerDataProcessor, CustomerSegmentationFeatureEngineer


def show_data_overview_page():
    """고객 데이터 개요 페이지를 표시하는 함수"""
    st.header("📊 고객 데이터 개요")
    
    # 데이터 프로세서 초기화
    data_processor = CustomerDataProcessor()
    
    # 데이터 로드
    with st.spinner("데이터 로딩 중..."):
        data = data_processor.load_data()
    
    if data is None:
        st.error("❌ 데이터를 로드할 수 없습니다.")
        return None
    
    st.success(f"✅ 데이터 로드 완료: {len(data):,}명의 고객 데이터")
    
    # 데이터 검증
    validation_results = data_processor.validate_data(data)
    
    # 기본 정보 표시
    show_basic_info(data, validation_results)
    
    # 데이터 품질 검사
    show_data_quality(data, data_processor, validation_results)
    
    # 데이터 분포 시각화
    show_data_distribution(data, data_processor)
    
    # 상관관계 분석
    show_correlation_analysis(data, data_processor)
    
    # 데이터 미리보기
    show_data_preview(data)
    
    return data


def show_basic_info(data, validation_results):
    """기본 데이터 정보 표시"""
    st.subheader("📋 데이터셋 기본 정보")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("전체 고객 수", f"{validation_results['total_rows']:,}명")
    with col2:
        st.metric("특성 수", f"{validation_results['total_columns']}개")
    with col3:
        st.metric("수치형 특성", f"{validation_results['numeric_columns']}개")
    with col4:
        st.metric("범주형 특성", f"{validation_results['categorical_columns']}개")
    
    # 데이터 타입 정보
    with st.expander("📊 컬럼별 데이터 타입"):
        dtypes_df = pd.DataFrame({
            "컬럼명": data.columns,
            "데이터 타입": [str(dtype) for dtype in validation_results['data_types']],
            "결측값 수": validation_results['missing_values'].values
        })
        st.dataframe(dtypes_df, use_container_width=True)


def show_data_quality(data, data_processor, validation_results):
    """데이터 품질 검사 결과 표시"""
    st.subheader("🔍 데이터 품질 검사")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 결측값 검사
        st.write("**📉 결측값 현황**")
        if not validation_results['has_missing']:
            st.success("✅ 결측값이 없습니다.")
        else:
            missing_data = validation_results['missing_values'][validation_results['missing_values'] > 0]
            st.warning(f"⚠️ {len(missing_data)}개 컬럼에 결측값 발견")
            st.dataframe(missing_data.to_frame('결측값 수'), use_container_width=True)
        
        # 중복값 검사
        st.write("**🔄 중복값 현황**")
        duplicate_count = validation_results['duplicate_rows']
        if duplicate_count == 0:
            st.success("✅ 중복 행이 없습니다.")
        else:
            st.warning(f"⚠️ {duplicate_count}개의 중복 행 발견")
    
    with col2:
        # 이상치 탐지
        st.write("**📊 이상치 탐지 (IQR 방법)**")
        outliers = data_processor.detect_outliers(data)
        
        for col, outlier_info in outliers.items():
            if outlier_info['count'] > 0:
                st.warning(f"⚠️ {col}: {outlier_info['count']}개 ({outlier_info['percentage']:.1f}%)")
            else:
                st.success(f"✅ {col}: 이상치 없음")


def show_data_distribution(data, data_processor):
    """데이터 분포 시각화"""
    st.subheader("📈 고객 특성 분포")
    
    # 수치형 특성 분포
    numeric_cols = data_processor.get_numeric_columns()
    
    # 히스토그램
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=numeric_cols,
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, col in enumerate(numeric_cols):
        fig.add_histogram(
            x=data[col],
            name=col,
            row=1, col=i+1,
            marker_color=colors[i],
            opacity=0.7,
            nbinsx=20
        )
    
    fig.update_layout(
        height=400,
        title_text="고객 특성별 분포",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 성별 분포 (있는 경우)
    if 'Gender' in data.columns:
        show_gender_distribution(data, data_processor)


def show_gender_distribution(data, data_processor):
    """성별 분포 표시"""
    st.write("**👥 성별 분포**")
    
    gender_dist = data_processor.get_gender_distribution(data)
    
    if gender_dist:
        col1, col2 = st.columns(2)
        
        with col1:
            # 파이 차트
            fig = px.pie(
                values=list(gender_dist['counts'].values()),
                names=list(gender_dist['counts'].keys()),
                title="성별 분포",
                color_discrete_sequence=['#FF9999', '#66B2FF']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 성별별 통계
            st.write("**성별별 기본 통계:**")
            for gender, count in gender_dist['counts'].items():
                percentage = gender_dist['percentages'][gender]
                st.write(f"- {gender}: {count:,}명 ({percentage:.1f}%)")


def show_correlation_analysis(data, data_processor):
    """상관관계 분석"""
    st.subheader("🔗 특성 간 상관관계")
    
    numeric_cols = data_processor.get_numeric_columns()
    correlation_matrix = data[numeric_cols].corr()
    
    # 히트맵
    fig = px.imshow(
        correlation_matrix,
        labels=dict(color="상관계수"),
        title="특성 간 상관관계 히트맵",
        color_continuous_scale='RdBu',
        aspect="auto"
    )
    
    # 상관계수 값 표시
    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.columns)):
            fig.add_annotation(
                x=j, y=i,
                text=f"{correlation_matrix.iloc[i, j]:.2f}",
                showarrow=False,
                font=dict(color="white" if abs(correlation_matrix.iloc[i, j]) > 0.5 else "black")
            )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # 상관관계 해석
    with st.expander("📊 상관관계 해석"):
        st.markdown("""
        **상관계수 해석:**
        - **0.7 이상**: 강한 양의 상관관계
        - **-0.7 이하**: 강한 음의 상관관계  
        - **0.3 ~ 0.7**: 중간 정도의 양의 상관관계
        - **-0.3 ~ -0.7**: 중간 정도의 음의 상관관계
        - **-0.3 ~ 0.3**: 약한 상관관계
        """)


def show_data_preview(data):
    """데이터 미리보기"""
    st.subheader("👀 데이터 미리보기")
    
    # 샘플 크기 선택
    sample_size = st.selectbox("표시할 행 수:", [5, 10, 20, 50], index=1)
    
    # 데이터 표시
    st.dataframe(data.head(sample_size), use_container_width=True)
    
    # 기본 통계
    with st.expander("📊 기본 통계 정보"):
        st.dataframe(data.describe(), use_container_width=True)


# 특성 엔지니어링 페이지 (추가 기능)
def show_feature_engineering_section(data):
    """특성 엔지니어링 섹션"""
    st.subheader("🔧 특성 엔지니어링")
    
    data_processor = CustomerDataProcessor()
    feature_engineer = CustomerSegmentationFeatureEngineer(data_processor)
    
    if st.button("🚀 추가 특성 생성"):
        with st.spinner("특성 엔지니어링 중..."):
            enhanced_data = feature_engineer.create_all_features(data)
        
        st.success("✅ 추가 특성 생성 완료!")
        
        # 새로 생성된 특성들 표시
        new_columns = ['Age_Group', 'Income_Group', 'Spending_Group', 'Spending_Propensity']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🎯 연령 그룹 분포:**")
            age_group_dist = enhanced_data['Age_Group'].value_counts()
            st.dataframe(age_group_dist.to_frame('고객 수'), use_container_width=True)
        
        with col2:
            st.write("**💰 소득 그룹 분포:**")
            income_group_dist = enhanced_data['Income_Group'].value_counts()
            st.dataframe(income_group_dist.to_frame('고객 수'), use_container_width=True)
        
        # 지출 성향 분석
        st.write("**💳 지출 성향 분석:**")
        fig = px.histogram(
            enhanced_data, 
            x='Spending_Propensity',
            title="고객별 지출 성향 분포 (소득 대비 지출)",
            labels={'Spending_Propensity': '지출 성향 점수', 'count': '고객 수'},
            nbins=30
        )
        st.plotly_chart(fig, use_container_width=True)
        
        return enhanced_data
    
    return data
