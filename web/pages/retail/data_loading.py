"""
Online Retail 데이터 로딩 페이지

데이터 로딩 및 품질 분석을 담당하는 Streamlit 페이지
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data.loaders.retail_loader import RetailDataLoader
from core.retail.visualizer import RetailVisualizer
import warnings

warnings.filterwarnings("ignore")


def show_data_loading_page():
    """데이터 로딩 및 품질 분석 페이지"""
    
    st.header("1️⃣ 데이터 로딩 & 품질 분석")
    
    st.markdown("""
    ### 📖 학습 목표
    - 실무급 대용량 데이터 로딩 경험
    - 체계적인 데이터 품질 분석 방법론 학습
    - ADP 실기의 핵심인 결측값, 이상치 탐지 기법 익히기
    """)
    
    # 세션 상태 초기화
    if 'retail_data_loader' not in st.session_state:
        st.session_state.retail_data_loader = RetailDataLoader()
    
    # 데이터 로딩 섹션
    if not st.session_state.get('retail_data_loaded', False):
        if st.button("📥 Online Retail 데이터 로딩 시작", type="primary"):
            with st.spinner("데이터를 로딩하는 중입니다..."):
                try:
                    loader = st.session_state.retail_data_loader
                    data = loader.load_data()
                    
                    # 세션 상태에 데이터 저장
                    st.session_state.retail_raw_data = data.copy()
                    st.session_state.retail_column_mapping = loader.get_column_mapping()
                    st.session_state.retail_data_loaded = True
                    
                    st.success(f"✅ 데이터 로딩 완료: {len(data):,}개 레코드")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ 데이터 로딩 실패: {str(e)}")
                    st.info("💡 인터넷 연결을 확인하고 다시 시도해주세요.")
    
    # 로딩된 데이터 정보 표시
    if st.session_state.get('retail_data_loaded', False):
        data = st.session_state.retail_raw_data
        
        st.success("✅ 데이터가 성공적으로 로딩되었습니다!")
        
        # 기본 정보 표시
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("총 레코드 수", f"{len(data):,}")
        with col2:
            st.metric("컬럼 수", data.shape[1])
        with col3:
            st.metric("메모리 사용량", f"{data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        with col4:
            st.metric("기간", f"{data.shape[0] // 1000}K+ 거래")
        
        # 데이터 샘플 보기
        st.subheader("🔍 데이터 샘플")
        st.dataframe(data.head(10), use_container_width=True)
        
        # 컬럼 매핑 정보 표시
        if st.session_state.get('retail_column_mapping'):
            with st.expander("🔄 컬럼 매핑 정보"):
                mapping_df = pd.DataFrame([
                    {'표준명': k, '실제 컬럼명': v} 
                    for k, v in st.session_state.retail_column_mapping.items()
                    if v is not None
                ])
                st.dataframe(mapping_df, use_container_width=True)
        
        # 품질 분석 섹션
        st.markdown("---")
        st.subheader("📊 데이터 품질 분석")
        
        if not st.session_state.get('retail_quality_analyzed', False):
            if st.button("🔍 품질 분석 실행", type="secondary"):
                with st.spinner("데이터 품질을 분석하는 중입니다..."):
                    try:
                        loader = st.session_state.retail_data_loader
                        quality_report = loader.analyze_data_quality(data)
                        
                        # 품질 분석 결과 저장
                        st.session_state.retail_quality_report = quality_report
                        st.session_state.retail_quality_analyzed = True
                        
                        st.success("✅ 품질 분석 완료!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ 품질 분석 실패: {str(e)}")
        
        # 품질 분석 결과 표시
        if st.session_state.get('retail_quality_analyzed', False):
            quality_report = st.session_state.retail_quality_report
            
            st.success("✅ 품질 분석이 완료되었습니다!")
            
            # 주요 발견사항
            st.markdown("### 📋 주요 발견사항")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🚨 주의 필요:**")
                high_missing = [(col, info['percentage']) for col, info in quality_report['missing_values'].items() 
                               if info['percentage'] > 10]
                if high_missing:
                    for col, pct in high_missing:
                        st.warning(f"• {col}: {pct}% 결측값")
                else:
                    st.success("• 심각한 결측값 문제 없음")
            
            with col2:
                st.markdown("**✅ 긍정적 요소:**")
                st.success(f"• 총 {quality_report['total_records']:,}개의 풍부한 데이터")
                st.success(f"• {quality_report['total_columns']}개의 다양한 특성")
            
            # 시각화 대시보드
            st.markdown("### 📊 품질 분석 대시보드")
            
            try:
                quality_fig = RetailVisualizer.create_data_quality_dashboard(quality_report)
                st.plotly_chart(quality_fig, use_container_width=True)
            except Exception as e:
                st.warning(f"시각화 생성 중 오류: {str(e)}")
            
            # 상세 분석 결과
            st.markdown("### 📊 상세 분석 결과")
            
            # 결측값 상세 분석
            with st.expander("🔍 결측값 상세 분석"):
                missing_df = pd.DataFrame([
                    {
                        '컬럼명': col,
                        '결측값 개수': info['count'],
                        '결측률(%)': info['percentage'],
                        '심각도': '높음' if info['percentage'] > 20 else '보통' if info['percentage'] > 5 else '낮음'
                    }
                    for col, info in quality_report['missing_values'].items()
                ]).sort_values('결측률(%)', ascending=False)
                
                st.dataframe(missing_df, use_container_width=True)
            
            # 이상치 분석
            if quality_report['outliers']:
                with st.expander("🚨 이상치 분석"):
                    outlier_df = pd.DataFrame([
                        {
                            '컬럼명': col,
                            '이상치 개수': info['outlier_count'],
                            '이상치 비율(%)': info['outlier_percentage'],
                            '하한값': info['lower_bound'],
                            '상한값': info['upper_bound']
                        }
                        for col, info in quality_report['outliers'].items()
                    ]).sort_values('이상치 비율(%)', ascending=False)
                    
                    st.dataframe(outlier_df, use_container_width=True)
            
            # 다음 단계 안내
            st.markdown("---")
            st.info("💡 품질 분석이 완료되었습니다. 다음 단계인 '데이터 정제 & 전처리'로 진행해주세요.")
    
    else:
        st.info("💡 '데이터 로딩 시작' 버튼을 클릭하여 시작해주세요.")


def get_data_loading_status():
    """데이터 로딩 상태 반환"""
    return {
        'data_loaded': st.session_state.get('retail_data_loaded', False),
        'quality_analyzed': st.session_state.get('retail_quality_analyzed', False),
        'records_count': len(st.session_state.retail_raw_data) if st.session_state.get('retail_data_loaded', False) else 0
    }
