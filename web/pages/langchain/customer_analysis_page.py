"""
LangChain Customer Analysis Page

LangChain 기반 고객 분석 기능을 제공하는 Streamlit 페이지
"""

# numpy 호환성 문제 해결 (numpy 1.24+ 대응) - 다른 import보다 먼저 실행
import numpy as np
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'complex'):
    np.complex = complex
if not hasattr(np, 'object'):
    np.object = object
if not hasattr(np, 'str'):
    np.str = str

import streamlit as st
import pandas as pd
import json
from typing import Dict, Any
import warnings

# numpy 호환성 경고 억제
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*numpy.*")

try:
    # 프로젝트 모듈 import - 기존 프로젝트의 data processors 사용
    from core.langchain_analysis.customer_analysis_chain import CustomerAnalysisChain, CustomerInsightGenerator
    from data.processors.segmentation_data_processor import DataProcessor
    from core.segmentation.clustering import ClusterAnalyzer
    MODULES_LOADED = True
except ImportError as e:
    st.error(f"필요한 모듈을 불러올 수 없습니다: {e}")
    MODULES_LOADED = False


def show_customer_analysis_page():
    """LangChain 기반 고객 분석 페이지를 표시"""
    
    st.header("🧠 LangChain 고객 분석")
    st.write("LangChain을 활용하여 고객 데이터를 심층 분석하고 인사이트를 생성합니다.")
    
    if not MODULES_LOADED:
        st.error("⚠️ 필요한 모듈이 로드되지 않아 이 기능을 사용할 수 없습니다.")
        st.info("해결 방법: 필요한 의존성 패키지를 설치하고 페이지를 새로고침하세요.")
        return
    
    # 사이드바에서 분석 옵션 설정
    st.sidebar.header("분석 설정")
    analysis_type = st.sidebar.selectbox(
        "분석 유형 선택",
        ["세그먼트 분석", "개별 고객 분석", "트렌드 분석", "종합 리포트"]
    )
    
    # 데이터 로드
    try:
        data_processor = DataProcessor()
        customer_data = data_processor.load_data()
        
        if customer_data is None or customer_data.empty:
            st.error("고객 데이터를 로드할 수 없습니다.")
            return
        
        st.success(f"✅ 고객 데이터 로드 완료: {len(customer_data)}명의 고객 데이터")
        
        # 클러스터 분석 수행
        cluster_analyzer = ClusterAnalyzer(customer_data)
        
        # Session State에서 클러스터 개수 가져오기 (기본값: 5)
        if "selected_clusters" not in st.session_state:
            st.session_state.selected_clusters = 5
        
        n_clusters = st.session_state.selected_clusters
        
        # 클러스터링 수행
        with st.spinner("클러스터링 분석 중..."):
            results = cluster_analyzer.perform_clustering(n_clusters=n_clusters, method='kmeans')
            cluster_labels = results['labels']
        
        # LangChain 분석 체인 초기화
        analysis_chain = CustomerAnalysisChain()
        insight_generator = CustomerInsightGenerator(analysis_chain)
        
        # 분석 유형에 따른 처리
        if analysis_type == "세그먼트 분석":
            show_segment_analysis(analysis_chain, customer_data, cluster_labels)
        
        elif analysis_type == "개별 고객 분석":
            show_individual_analysis(analysis_chain, customer_data, cluster_labels)
        
        elif analysis_type == "트렌드 분석":
            show_trend_analysis(analysis_chain, customer_data)
        
        elif analysis_type == "종합 리포트":
            show_comprehensive_report(insight_generator, customer_data, cluster_labels)
    
    except Exception as e:
        st.error(f"분석 중 오류가 발생했습니다: {str(e)}")
        st.info("페이지를 새로고침하거나 다른 분석 유형을 시도해보세요.")


def show_segment_analysis(analysis_chain: CustomerAnalysisChain, customer_data: pd.DataFrame, cluster_labels: list):
    """고객 세그먼트 분석 결과를 표시"""
    
    st.subheader("🎯 고객 세그먼트 분석")
    
    with st.spinner("세그먼트 분석 중..."):
        segment_results = analysis_chain.analyze_customer_segments(customer_data, cluster_labels)
    
    if "error" in segment_results:
        st.error(segment_results["error"])
        return
    
    # 세그먼트 정보 표시
    if "segments" in segment_results:
        st.write("### 📊 식별된 고객 세그먼트")
        
        for segment in segment_results["segments"]:
            segment_name = segment.get('segment_name', f'세그먼트 {segment.get("cluster_id", 0)}')
            with st.expander(f"🏷️ {segment_name}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**클러스터 ID:** {segment.get('cluster_id', 'N/A')}")
                    st.write(f"**고객 수:** {segment.get('size', 'N/A')}명")
                    st.write(f"**평균 연령:** {segment.get('avg_age', 'N/A')}세")
                
                with col2:
                    st.write(f"**평균 소득:** ${segment.get('avg_income', 'N/A')}k")
                    st.write(f"**평균 지출점수:** {segment.get('avg_spending', 'N/A')}")
                
                st.write("**주요 특징:**")
                characteristics = segment.get('characteristics', [])
                for char in characteristics:
                    st.write(f"• {char}")
    
    # 전체 인사이트 표시
    if "overall_insights" in segment_results:
        st.write("### 💡 주요 인사이트")
        for insight in segment_results["overall_insights"]:
            st.info(f"🔍 {insight}")
    
    # 비즈니스 추천사항 표시
    if "business_recommendations" in segment_results:
        st.write("### 🚀 비즈니스 추천사항")
        for recommendation in segment_results["business_recommendations"]:
            st.success(f"💼 {recommendation}")


def show_individual_analysis(analysis_chain: CustomerAnalysisChain, customer_data: pd.DataFrame, cluster_labels: list):
    """개별 고객 분석 결과를 표시"""
    
    st.subheader("👤 개별 고객 분석")
    
    # 고객 선택
    customer_ids = customer_data.index.tolist()
    selected_customer_idx = st.selectbox(
        "분석할 고객 선택",
        range(len(customer_ids)),
        format_func=lambda x: f"고객 #{customer_ids[x]} ({customer_data.iloc[x]['Gender']}, {customer_data.iloc[x]['Age']}세)"
    )
    
    if st.button("개별 고객 분석 실행"):
        # 선택된 고객 정보
        customer_row = customer_data.iloc[selected_customer_idx]
        customer_profile = {
            "CustomerID": customer_ids[selected_customer_idx],
            "Gender": customer_row["Gender"],
            "Age": customer_row["Age"],
            "Annual Income (k$)": customer_row["Annual Income (k$)"],
            "Spending Score (1-100)": customer_row["Spending Score (1-100)"],
            "Cluster": cluster_labels[selected_customer_idx]
        }
        
        # 세그먼트 정보 (간단한 버전)
        segment_info = {
            "cluster_id": cluster_labels[selected_customer_idx],
            "segment_description": "고객이 속한 세그먼트의 일반적 특성"
        }
        
        with st.spinner("개별 고객 분석 중..."):
            individual_results = analysis_chain.analyze_individual_customer(customer_profile, segment_info)
        
        if "error" in individual_results:
            st.error(individual_results["error"])
            return
        
        # 고객 프로필 표시
        st.write("### 👤 고객 프로필")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("연령", f"{customer_profile['Age']}세")
            st.write(f"**성별:** {customer_profile['Gender']}")
        
        with col2:
            st.metric("연소득", f"${customer_profile['Annual Income (k$)']}k")
        
        with col3:
            st.metric("지출점수", f"{customer_profile['Spending Score (1-100)']}점")
        
        # 분석 결과 표시
        st.write("### 🔍 분석 결과")
        
        if "customer_type" in individual_results:
            st.write(f"**고객 유형:** {individual_results['customer_type']}")
        
        if "behavioral_analysis" in individual_results:
            st.write(f"**행동 분석:** {individual_results['behavioral_analysis']}")
        
        if "retention_risk" in individual_results:
            risk_color = {
                "낮음": "green",
                "보통": "orange", 
                "높음": "red"
            }.get(individual_results["retention_risk"], "gray")
            
            st.markdown(f"**이탈 위험도:** :{risk_color}[{individual_results['retention_risk']}]")
        
        # 추천사항 표시
        if "personalized_offers" in individual_results:
            st.write("### 🎁 맞춤형 제안")
            for offer in individual_results["personalized_offers"]:
                st.success(f"💡 {offer}")


def show_trend_analysis(analysis_chain: CustomerAnalysisChain, customer_data: pd.DataFrame):
    """트렌드 분석 결과를 표시"""
    
    st.subheader("📈 트렌드 분석")
    
    # 데이터 요약 통계 생성
    data_summary = {
        "total_customers": len(customer_data),
        "avg_age": customer_data['Age'].mean(),
        "avg_income": customer_data['Annual Income (k$)'].mean(), 
        "avg_spending": customer_data['Spending Score (1-100)'].mean(),
        "gender_distribution": customer_data['Gender'].value_counts().to_dict(),
        "age_distribution": {
            "under_30": len(customer_data[customer_data['Age'] < 30]),
            "30_to_50": len(customer_data[(customer_data['Age'] >= 30) & (customer_data['Age'] < 50)]),
            "over_50": len(customer_data[customer_data['Age'] >= 50])
        }
    }
    
    with st.spinner("트렌드 분석 중..."):
        trend_results = analysis_chain.analyze_trends(data_summary)
    
    if "error" in trend_results:
        st.error(trend_results["error"])
        return
    
    # 주요 트렌드 표시
    if "key_trends" in trend_results:
        st.write("### 📊 주요 트렌드")
        for trend in trend_results["key_trends"]:
            st.info(f"📈 {trend}")
    
    # 인구통계학적 인사이트
    if "demographic_insights" in trend_results:
        st.write("### 👥 인구통계학적 인사이트")
        insights = trend_results["demographic_insights"]
        
        col1, col2 = st.columns(2)
        with col1:
            if "age_patterns" in insights:
                st.write(f"**연령 패턴:** {insights['age_patterns']}")
            if "gender_patterns" in insights:
                st.write(f"**성별 패턴:** {insights['gender_patterns']}")
        
        with col2:
            if "income_patterns" in insights:
                st.write(f"**소득 패턴:** {insights['income_patterns']}")
    
    # 시장 기회 및 위험 요소
    col1, col2 = st.columns(2)
    
    with col1:
        if "market_opportunities" in trend_results:
            st.write("### 🚀 시장 기회")
            for opportunity in trend_results["market_opportunities"]:
                st.success(f"💡 {opportunity}")
    
    with col2:
        if "risk_factors" in trend_results:
            st.write("### ⚠️ 위험 요소")
            for risk in trend_results["risk_factors"]:
                st.warning(f"🚨 {risk}")


def show_comprehensive_report(insight_generator: CustomerInsightGenerator, customer_data: pd.DataFrame, cluster_labels: list):
    """종합 분석 리포트를 표시"""
    
    st.subheader("📋 종합 분석 리포트")
    st.write("모든 분석 결과를 종합한 완전한 고객 분석 리포트입니다.")
    
    with st.spinner("종합 리포트 생성 중..."):
        comprehensive_report = insight_generator.generate_comprehensive_report(customer_data, cluster_labels)
    
    if "error" in comprehensive_report:
        st.error(comprehensive_report["error"])
        return
    
    # 경영진 요약
    if "executive_summary" in comprehensive_report:
        st.write("### 📊 경영진 요약")
        summary = comprehensive_report["executive_summary"]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("총 고객 수", f"{summary.get('total_customers', 0):,}명")
        
        with col2:
            st.metric("세그먼트 수", f"{summary.get('segments_identified', 0)}개")
        
        with col3:
            avg_age = comprehensive_report.get("data_summary", {}).get("avg_age", 0)
            st.metric("평균 연령", f"{avg_age:.1f}세")
        
        with col4:
            avg_income = comprehensive_report.get("data_summary", {}).get("avg_income", 0)
            st.metric("평균 소득", f"${avg_income:.1f}k")
    
    # 주요 인사이트 및 트렌드
    if "executive_summary" in comprehensive_report:
        st.write("### 💡 주요 인사이트")
        for insight in comprehensive_report["executive_summary"].get("key_insights", []):
            st.info(f"🔍 {insight}")
        
        st.write("### 📈 주요 트렌드")
        for trend in comprehensive_report["executive_summary"].get("main_trends", []):
            st.info(f"📊 {trend}")
    
    # 추천사항
    if "recommendations" in comprehensive_report:
        st.write("### 🎯 추천사항")
        recommendations = comprehensive_report["recommendations"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            if "immediate_actions" in recommendations:
                st.write("**즉시 실행 가능한 액션:**")
                for action in recommendations["immediate_actions"]:
                    st.success(f"⚡ {action}")
        
        with col2:
            if "long_term_strategy" in recommendations:
                st.write("**장기 전략:**")
                for strategy in recommendations["long_term_strategy"]:
                    st.info(f"🎯 {strategy}")
        
        if "risk_mitigation" in recommendations:
            st.write("**위험 완화 방안:**")
            for risk in recommendations["risk_mitigation"]:
                st.warning(f"🛡️ {risk}")
    
    # 리포트 다운로드 기능
    st.write("### 💾 리포트 다운로드")
    
    if st.button("JSON 리포트 다운로드"):
        report_json = json.dumps(comprehensive_report, indent=2, ensure_ascii=False)
        st.download_button(
            label="📄 JSON 파일 다운로드",
            data=report_json,
            file_name="customer_analysis_report.json",
            mime="application/json"
        )