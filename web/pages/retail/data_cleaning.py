"""
Online Retail 데이터 정제 페이지

데이터 정제 및 전처리를 담당하는 Streamlit 페이지
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from data.processors.retail_data_processor import RetailDataProcessor
import warnings

warnings.filterwarnings("ignore")


def show_data_cleaning_page():
    """데이터 정제 및 전처리 페이지"""
    
    st.header("2️⃣ 데이터 정제 & 전처리")
    
    # 이전 단계 완료 확인
    if not st.session_state.get('retail_data_loaded', False):
        st.warning("⚠️ 먼저 1단계에서 데이터를 로딩해주세요.")
        return
    
    st.markdown("""
    ### 📖 학습 목표  
    - 실무에서 가장 시간이 많이 걸리는 데이터 정제 과정 체험
    - 비즈니스 로직에 기반한 합리적 정제 기준 수립
    - ADP 실기의 핵심인 데이터 변환 및 파생변수 생성
    """)
    
    # 세션 상태 초기화
    if 'retail_data_processor' not in st.session_state:
        column_mapping = st.session_state.get('retail_column_mapping', {})
        st.session_state.retail_data_processor = RetailDataProcessor(column_mapping)
    
    # 데이터 정제 실행
    if not st.session_state.get('retail_data_cleaned', False):
        if st.button("🧹 데이터 정제 시작", type="primary"):
            with st.spinner("데이터를 정제하는 중입니다..."):
                try:
                    # 세션 상태에서 데이터 가져오기
                    raw_data = st.session_state.retail_raw_data
                    processor = st.session_state.retail_data_processor
                    
                    original_shape = raw_data.shape
                    cleaned_data = processor.clean_data(raw_data)
                    
                    # 정제된 데이터 저장
                    st.session_state.retail_cleaned_data = cleaned_data.copy()
                    st.session_state.retail_data_cleaned = True
                    
                    st.success("✅ 데이터 정제 완료!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ 데이터 정제 실패: {str(e)}")
    
    # 정제 결과 표시
    if st.session_state.get('retail_data_cleaned', False):
        cleaned_data = st.session_state.retail_cleaned_data
        raw_data = st.session_state.retail_raw_data
        
        st.success("✅ 데이터 정제가 완료되었습니다!")
        
        # 정제 통계
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("정제 전 레코드", f"{len(raw_data):,}")
        with col2:
            st.metric("정제 후 레코드", f"{len(cleaned_data):,}")
        with col3:
            retention_rate = (len(cleaned_data) / len(raw_data)) * 100
            st.metric("데이터 보존율", f"{retention_rate:.1f}%")
        
        # 정제된 데이터 샘플
        st.subheader("🔍 정제된 데이터 샘플")
        st.dataframe(cleaned_data.head(10), use_container_width=True)
        
        # 새로 생성된 변수들
        st.subheader("🆕 생성된 파생 변수들")
        new_columns = ['TotalAmount', 'IsReturn', 'Year', 'Month', 'DayOfWeek', 'Hour']
        
        derived_cols = []
        for col in new_columns:
            if col in cleaned_data.columns:
                derived_cols.append(col)
        
        if derived_cols:
            for col in derived_cols:
                st.info(f"**{col}**: {get_column_description(col)}")
        
        # 정제 전후 비교 시각화
        st.subheader("📊 정제 전후 비교")
        
        # 데이터 크기 비교
        comparison_data = pd.DataFrame({
            '단계': ['정제 전', '정제 후'],
            '레코드 수': [len(raw_data), len(cleaned_data)],
            '컬럼 수': [len(raw_data.columns), len(cleaned_data.columns)]
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_records = px.bar(
                comparison_data, 
                x='단계', 
                y='레코드 수',
                title="레코드 수 변화",
                color='단계',
                color_discrete_map={'정제 전': 'lightcoral', '정제 후': 'lightgreen'}
            )
            st.plotly_chart(fig_records, use_container_width=True)
        
        with col2:
            fig_columns = px.bar(
                comparison_data, 
                x='단계', 
                y='컬럼 수',
                title="컬럼 수 변화",
                color='단계',
                color_discrete_map={'정제 전': 'lightblue', '정제 후': 'lightcyan'}
            )
            st.plotly_chart(fig_columns, use_container_width=True)
        
        # 파생 변수 분포 분석
        if derived_cols:
            st.subheader("📈 파생 변수 분포 분석")
            
            # TotalAmount 분포
            if 'TotalAmount' in cleaned_data.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_amount = px.histogram(
                        cleaned_data, 
                        x='TotalAmount', 
                        title="총 거래 금액 분포",
                        nbins=50
                    )
                    st.plotly_chart(fig_amount, use_container_width=True)
                
                with col2:
                    # 반품 비율
                    if 'IsReturn' in cleaned_data.columns:
                        return_stats = cleaned_data['IsReturn'].value_counts()
                        fig_return = px.pie(
                            values=return_stats.values,
                            names=['정상 거래', '반품 거래'],
                            title="반품 거래 비율"
                        )
                        st.plotly_chart(fig_return, use_container_width=True)
            
            # 시간대별 분석
            time_cols = ['Year', 'Month', 'DayOfWeek', 'Hour']
            available_time_cols = [col for col in time_cols if col in cleaned_data.columns]
            
            if available_time_cols:
                st.subheader("🕐 시간대별 거래 분석")
                
                selected_time_col = st.selectbox(
                    "분석할 시간 단위를 선택하세요:",
                    available_time_cols,
                    format_func=lambda x: {
                        'Year': '연도별',
                        'Month': '월별',
                        'DayOfWeek': '요일별',
                        'Hour': '시간대별'
                    }.get(x, x)
                )
                
                if selected_time_col:
                    time_distribution = cleaned_data[selected_time_col].value_counts().sort_index()
                    
                    fig_time = px.bar(
                        x=time_distribution.index,
                        y=time_distribution.values,
                        title=f"{selected_time_col} 별 거래 건수",
                        labels={'x': selected_time_col, 'y': '거래 건수'}
                    )
                    st.plotly_chart(fig_time, use_container_width=True)
        
        # 데이터 품질 검증
        st.subheader("🔍 정제 후 데이터 품질 검증")
        
        if not st.session_state.get('retail_quality_validated', False):
            if st.button("🔍 품질 검증 실행", type="secondary"):
                with st.spinner("정제된 데이터의 품질을 검증하는 중입니다..."):
                    try:
                        processor = st.session_state.retail_data_processor
                        validation_report = processor.validate_data_quality(cleaned_data)
                        
                        # 검증 결과 저장
                        st.session_state.retail_validation_report = validation_report
                        st.session_state.retail_quality_validated = True
                        
                        st.success("✅ 품질 검증 완료!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ 품질 검증 실패: {str(e)}")
        
        # 검증 결과 표시
        if st.session_state.get('retail_quality_validated', False):
            validation_report = st.session_state.retail_validation_report
            
            st.success("✅ 품질 검증이 완료되었습니다!")
            
            # 품질 점수 표시
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("품질 점수", f"{validation_report['data_quality_score']}/100")
            with col2:
                st.metric("품질 등급", validation_report['quality_grade'])
            with col3:
                issues_count = len(validation_report['issues_found'])
                st.metric("발견된 이슈", f"{issues_count}개")
            
            # 발견된 이슈들
            if validation_report['issues_found']:
                st.markdown("#### 🚨 발견된 품질 이슈")
                for issue in validation_report['issues_found']:
                    st.warning(f"• {issue}")
            else:
                st.success("🎉 품질 이슈가 발견되지 않았습니다!")
            
            # 품질 메트릭 상세
            with st.expander("📊 품질 메트릭 상세"):
                metrics_df = pd.DataFrame([
                    {'메트릭': k, '값': v} 
                    for k, v in validation_report['quality_metrics'].items()
                ])
                st.dataframe(metrics_df, use_container_width=True)
        
        # 전처리 요약 정보
        st.subheader("📋 전처리 요약")
        
        processor = st.session_state.retail_data_processor
        summary = processor.get_preprocessing_summary()
        
        if 'error' not in summary:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 데이터 정보:**")
                st.write(f"• 총 레코드: {summary['total_records']:,}개")
                st.write(f"• 총 컬럼: {summary['total_columns']}개")
                st.write(f"• 메모리 사용량: {summary['memory_usage_mb']:.1f} MB")
            
            with col2:
                st.markdown("**🆕 파생 변수:**")
                if summary['derived_columns']:
                    for col in summary['derived_columns']:
                        st.write(f"• {col}")
                else:
                    st.write("• 생성된 파생 변수 없음")
        
        # 다음 단계 안내
        st.markdown("---")
        st.info("💡 데이터 정제가 완료되었습니다. 다음 단계인 '특성 공학 & 파생변수'로 진행해주세요.")
    
    else:
        st.info("💡 '데이터 정제 시작' 버튼을 클릭하여 시작해주세요.")


def get_column_description(col_name):
    """컬럼별 설명 반환"""
    descriptions = {
        'TotalAmount': '수량 × 단가로 계산된 거래 총액',
        'IsReturn': '수량이 음수인 경우 True (반품 거래)',
        'Year': '거래 발생 연도',
        'Month': '거래 발생 월 (1-12)',
        'DayOfWeek': '거래 발생 요일 (0=월요일, 6=일요일)',
        'Hour': '거래 발생 시간 (0-23)'
    }
    return descriptions.get(col_name, '파생 변수')


def get_data_cleaning_status():
    """데이터 정제 상태 반환"""
    return {
        'data_cleaned': st.session_state.get('retail_data_cleaned', False),
        'quality_validated': st.session_state.get('retail_quality_validated', False),
        'records_count': len(st.session_state.retail_cleaned_data) if st.session_state.get('retail_data_cleaned', False) else 0,
        'quality_score': st.session_state.retail_validation_report.get('data_quality_score', 0) if st.session_state.get('retail_quality_validated', False) else 0
    }
