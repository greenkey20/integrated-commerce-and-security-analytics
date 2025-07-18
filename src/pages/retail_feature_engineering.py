"""
Online Retail 특성 공학 페이지

특성 공학 및 파생변수 생성을 담당하는 Streamlit 페이지
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.retail_feature_engineer import RetailFeatureEngineer
from core.retail_visualizer import RetailVisualizer
import warnings

warnings.filterwarnings("ignore")


def show_feature_engineering_page():
    """특성 공학 및 파생변수 생성 페이지"""
    
    st.header("3️⃣ 특성 공학 & 파생변수 생성")
    
    # 이전 단계 완료 확인
    if not st.session_state.get('retail_data_cleaned', False):
        st.warning("⚠️ 먼저 2단계에서 데이터 정제를 완료해주세요.")
        return
    
    st.markdown("""
    ### 📖 학습 목표
    - 실무에서 가장 중요한 특성 공학(Feature Engineering) 전 과정 체험
    - ADP 실기의 핵심인 groupby, agg 함수 마스터
    - RFM 분석 등 마케팅 분석 기법 적용
    """)
    
    # 세션 상태 초기화
    if 'retail_feature_engineer' not in st.session_state:
        column_mapping = st.session_state.get('retail_column_mapping', {})
        st.session_state.retail_feature_engineer = RetailFeatureEngineer(column_mapping)
    
    # 특성 공학 실행
    if not st.session_state.get('retail_features_created', False):
        if st.button("🏗️ 특성 공학 시작", type="primary"):
            with st.spinner("고객별 특성을 생성하는 중입니다..."):
                try:
                    # 세션 상태에서 정제된 데이터 가져오기
                    cleaned_data = st.session_state.retail_cleaned_data
                    engineer = st.session_state.retail_feature_engineer
                    
                    customer_features = engineer.create_customer_features(cleaned_data)
                    
                    # 고객 특성 저장
                    st.session_state.retail_customer_features = customer_features.copy()
                    st.session_state.retail_features_created = True
                    
                    st.success("✅ 특성 공학 완료!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ 특성 공학 실패: {str(e)}")
    
    # 특성 공학 결과 표시
    if st.session_state.get('retail_features_created', False):
        customer_features = st.session_state.retail_customer_features
        
        st.success("✅ 특성 공학이 완료되었습니다!")
        
        # 기본 통계
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("분석 대상 고객 수", f"{len(customer_features):,}명")
        with col2:
            st.metric("생성된 특성 수", f"{len(customer_features.columns)}개")
        with col3:
            st.metric("데이터 품질", "우수" if customer_features.isnull().sum().sum() == 0 else "양호")
        
        # 생성된 특성 미리보기
        st.subheader("🔍 생성된 고객 특성 미리보기")
        st.dataframe(customer_features.head(10), use_container_width=True)
        
        # 특성 카테고리 분석
        st.subheader("📊 특성 카테고리 분석")
        
        # 특성을 카테고리별로 분류
        feature_categories = {
            'RFM 특성': [col for col in customer_features.columns if any(x in col.lower() for x in ['recency', 'frequency', 'monetary', 'rfm'])],
            '행동 특성': [col for col in customer_features.columns if any(x in col.lower() for x in ['return', 'segment', 'interval', 'sensitivity'])],
            '통계 특성': [col for col in customer_features.columns if any(x in col.lower() for x in ['avg', 'std', 'min', 'max', 'sum', 'count'])],
            '기타 특성': [col for col in customer_features.columns if not any(any(x in col.lower() for x in category_keywords) for category_keywords in [['recency', 'frequency', 'monetary', 'rfm'], ['return', 'segment', 'interval', 'sensitivity'], ['avg', 'std', 'min', 'max', 'sum', 'count']])]
        }
        
        # 카테고리별 특성 수 시각화
        category_counts = {k: len(v) for k, v in feature_categories.items() if v}
        
        if category_counts:
            fig_categories = px.bar(
                x=list(category_counts.keys()),
                y=list(category_counts.values()),
                title="특성 카테고리별 분포",
                labels={'x': '카테고리', 'y': '특성 수'}
            )
            st.plotly_chart(fig_categories, use_container_width=True)
        
        # 카테고리별 특성 상세 정보
        with st.expander("📋 카테고리별 특성 상세 정보"):
            for category, features in feature_categories.items():
                if features:
                    st.markdown(f"**{category} ({len(features)}개):**")
                    for feature in features:
                        st.write(f"• {feature}")
                    st.write("")
        
        # RFM 분석 시각화
        st.subheader("📊 RFM 분석 결과")
        
        if all(col in customer_features.columns for col in ['recency_days', 'frequency', 'monetary']):
            # RFM 히스토그램
            try:
                rfm_fig = RetailVisualizer.create_rfm_analysis_plots(customer_features)
                st.plotly_chart(rfm_fig, use_container_width=True)
            except Exception as e:
                st.warning(f"RFM 시각화 생성 중 오류: {str(e)}")
                
                # 대안 시각화
                fig_rfm = make_subplots(
                    rows=1, cols=3,
                    subplot_titles=['Recency (최근성)', 'Frequency (빈도)', 'Monetary (금액)']
                )
                
                fig_rfm.add_trace(
                    go.Histogram(x=customer_features['recency_days'], name="Recency", marker_color='lightcoral'),
                    row=1, col=1
                )
                fig_rfm.add_trace(
                    go.Histogram(x=customer_features['frequency'], name="Frequency", marker_color='lightblue'),
                    row=1, col=2
                )
                fig_rfm.add_trace(
                    go.Histogram(x=customer_features['monetary'], name="Monetary", marker_color='lightgreen'),
                    row=1, col=3
                )
                
                fig_rfm.update_layout(title="고객 RFM 분석 분포", showlegend=False, height=400)
                st.plotly_chart(fig_rfm, use_container_width=True)
        
        # 고객 세그먼트 분석
        if 'customer_segment' in customer_features.columns:
            st.subheader("🎯 고객 세그먼트 분석")
            
            try:
                segment_fig = RetailVisualizer.create_customer_segment_analysis(customer_features)
                st.plotly_chart(segment_fig, use_container_width=True)
            except Exception as e:
                st.warning(f"세그먼트 시각화 생성 중 오류: {str(e)}")
                
                # 기본 세그먼트 분포
                segment_counts = customer_features['customer_segment'].value_counts()
                fig_segment = px.pie(
                    values=segment_counts.values,
                    names=segment_counts.index,
                    title="고객 세그먼트 분포"
                )
                st.plotly_chart(fig_segment, use_container_width=True)
            
            # 세그먼트 설명
            with st.expander("📖 고객 세그먼트 설명"):
                segment_descriptions = {
                    'Champions': '🏆 가장 가치 있는 고객 - 최근 구매, 높은 빈도, 높은 구매액',
                    'Loyal Customers': '💎 충성 고객 - 정기적 구매, 높은 구매액',
                    'Potential Loyalists': '🌟 잠재적 충성 고객 - 최근 구매, 향후 충성도 증가 가능',
                    'New Customers': '🆕 신규 고객 - 최근 구매했으나 빈도 낮음',
                    'Promising': '🎯 유망 고객 - 최근 구매, 평균적 특성',
                    'Need Attention': '⚠️ 관심 필요 고객 - 구매 빈도 감소 추세',
                    'About to Sleep': '😴 이탈 위험 고객 - 구매 빈도 크게 감소',
                    'At Risk': '🚨 위험 고객 - 이탈 가능성 높음',
                    'Cannot Lose Them': '🔥 절대 잃어서는 안 될 고객 - 과거 고가치 고객',
                    'Others': '🔄 기타 고객 - 분류되지 않은 고객'
                }
                
                for segment in customer_features['customer_segment'].unique():
                    if segment in segment_descriptions:
                        st.write(f"**{segment}**: {segment_descriptions[segment]}")
        
        # 특성 요약 통계
        st.subheader("📋 주요 특성 요약 통계")
        
        # 주요 특성들 선택
        key_features = []
        for col in ['total_amount', 'frequency', 'recency_days', 'unique_products', 'return_rate']:
            if col in customer_features.columns:
                key_features.append(col)
        
        if key_features:
            feature_summary = customer_features[key_features].describe().round(2)
            st.dataframe(feature_summary, use_container_width=True)
        
        # 특성 상관관계 분석
        if len(key_features) > 1:
            st.subheader("🔗 주요 특성 간 상관관계")
            
            corr_matrix = customer_features[key_features].corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="특성 간 상관관계 히트맵",
                color_continuous_scale='RdBu_r'
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # 높은 상관관계 특성 쌍 표시
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.5:
                        high_corr_pairs.append({
                            '특성1': corr_matrix.columns[i],
                            '특성2': corr_matrix.columns[j],
                            '상관계수': round(corr_value, 3)
                        })
            
            if high_corr_pairs:
                with st.expander("📊 높은 상관관계 특성 쌍 (|r| > 0.5)"):
                    corr_df = pd.DataFrame(high_corr_pairs)
                    st.dataframe(corr_df, use_container_width=True)
        
        # 특성 중요도 분석
        st.subheader("🎯 특성 중요도 분석")
        
        if not st.session_state.get('retail_feature_importance_analyzed', False):
            if st.button("🔍 특성 중요도 분석 실행", type="secondary"):
                with st.spinner("특성 중요도를 분석하는 중입니다..."):
                    try:
                        engineer = st.session_state.retail_feature_engineer
                        importance_analysis = engineer.get_feature_importance_analysis(customer_features)
                        
                        # 분석 결과 저장
                        st.session_state.retail_feature_importance = importance_analysis
                        st.session_state.retail_feature_importance_analyzed = True
                        
                        st.success("✅ 특성 중요도 분석 완료!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ 특성 중요도 분석 실패: {str(e)}")
        
        # 분석 결과 표시
        if st.session_state.get('retail_feature_importance_analyzed', False):
            importance_analysis = st.session_state.retail_feature_importance
            
            if 'error' not in importance_analysis:
                st.success("✅ 특성 중요도 분석이 완료되었습니다!")
                
                # 카테고리별 특성 수 표시
                col1, col2, col3, col4 = st.columns(4)
                
                categories = importance_analysis['feature_categories']
                with col1:
                    st.metric("RFM 특성", f"{len(categories.get('rfm_features', []))}개")
                with col2:
                    st.metric("행동 특성", f"{len(categories.get('behavioral_features', []))}개")
                with col3:
                    st.metric("통계 특성", f"{len(categories.get('statistical_features', []))}개")
                with col4:
                    st.metric("파생 특성", f"{len(categories.get('derived_features', []))}개")
                
                # 높은 상관관계 특성 쌍 표시
                if 'high_correlation_pairs' in importance_analysis.get('correlation_analysis', {}):
                    high_corr_pairs = importance_analysis['correlation_analysis']['high_correlation_pairs']
                    
                    if high_corr_pairs:
                        st.markdown("#### 🔗 높은 상관관계 특성 쌍 (|r| > 0.7)")
                        for pair in high_corr_pairs[:5]:  # 상위 5개만 표시
                            st.warning(f"• {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']}")
            else:
                st.warning(f"⚠️ {importance_analysis['error']}")
        
        # 고객 분포 시각화
        st.subheader("👥 고객 분포 분석")
        
        try:
            distribution_fig = RetailVisualizer.create_customer_distribution_plots(customer_features)
            st.plotly_chart(distribution_fig, use_container_width=True)
        except Exception as e:
            st.warning(f"고객 분포 시각화 생성 중 오류: {str(e)}")
        
        # 다음 단계 안내
        st.markdown("---")
        st.info("💡 특성 공학이 완료되었습니다. 다음 단계인 '타겟 변수 생성'으로 진행해주세요.")
    
    else:
        st.info("💡 '특성 공학 시작' 버튼을 클릭하여 시작해주세요.")


def get_feature_engineering_status():
    """특성 공학 상태 반환"""
    return {
        'features_created': st.session_state.get('retail_features_created', False),
        'importance_analyzed': st.session_state.get('retail_feature_importance_analyzed', False),
        'customer_count': len(st.session_state.retail_customer_features) if st.session_state.get('retail_features_created', False) else 0,
        'feature_count': len(st.session_state.retail_customer_features.columns) if st.session_state.get('retail_features_created', False) else 0
    }
