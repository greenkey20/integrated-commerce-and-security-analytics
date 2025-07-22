"""
마케팅 전략 페이지

기존 customer_segmentation_app.py의 "마케팅 전략" 메뉴 내용을 모듈화
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from data.processors.segmentation_data_processor import DataProcessor
from core.segmentation.clustering import ClusterAnalyzer


def show_marketing_strategy_page():
    """마케팅 전략 페이지를 표시하는 함수"""
    st.header("📈 클러스터별 마케팅 전략")
    
    # 데이터 로드
    data_processor = DataProcessor()
    data = data_processor.load_data()
    
    # Session State에서 클러스터 개수 가져오기
    if "selected_clusters" not in st.session_state:
        st.session_state.selected_clusters = 5  # 기본값

    selected_k = st.session_state.selected_clusters

    # 현재 설정 표시
    st.info(f"🎯 현재 선택된 클러스터 개수: **{selected_k}개** (클러스터링 분석 페이지에서 설정됨)")

    # 클러스터링 수행
    cluster_analyzer = ClusterAnalyzer()
    clusters, kmeans, scaler, silhouette_avg = cluster_analyzer.perform_clustering(data, selected_k)
    data_with_clusters = data.copy()
    data_with_clusters["Cluster"] = clusters

    # 동적 클러스터 분석
    cluster_profiles_list = cluster_analyzer.analyze_cluster_characteristics(data_with_clusters, selected_k)

    # 클러스터별 특성 분석 (기존 형식으로 변환)
    cluster_profiles = {}
    for profile in cluster_profiles_list:
        cluster_id = profile["cluster_id"]
        cluster_data = data_with_clusters[data_with_clusters["Cluster"] == cluster_id]
        cluster_profiles[cluster_id] = {
            "size": profile["size"],
            "avg_age": profile["avg_age"],
            "avg_income": profile["avg_income"],
            "avg_spending": profile["avg_spending"],
            "gender_ratio": cluster_data["Gender"].value_counts(normalize=True).to_dict(),
        }

    st.subheader("📊 마케팅 대시보드")

    # 전체 요약 메트릭
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_customers = len(data)
        st.metric("총 고객 수", f"{total_customers:,}명")

    with col2:
        avg_income = data["Annual Income (k$)"].mean()
        st.metric("평균 소득", f"${avg_income:.1f}k")

    with col3:
        avg_spending = data["Spending Score (1-100)"].mean()
        st.metric("평균 지출점수", f"{avg_spending:.1f}")

    with col4:
        high_value_customers = len(
            data_with_clusters[
                (data_with_clusters["Annual Income (k$)"] > 70)
                & (data_with_clusters["Spending Score (1-100)"] > 70)
            ]
        )
        st.metric("프리미엄 고객", f"{high_value_customers}명")

    # 클러스터 분포 시각화
    st.subheader("📊 클러스터 분포 및 가치 분석")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 클러스터별 고객 수
        cluster_counts = data_with_clusters["Cluster"].value_counts().sort_index()
        fig = px.pie(
            values=cluster_counts.values, 
            names=[f"클러스터 {i}" for i in cluster_counts.index],
            title="클러스터별 고객 분포"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # 클러스터별 평균 소득 vs 지출점수
        cluster_metrics = []
        for profile in cluster_profiles_list:
            cluster_metrics.append({
                "클러스터": f"클러스터 {profile['cluster_id']}",
                "평균_소득": profile["avg_income"],
                "평균_지출점수": profile["avg_spending"],
                "고객_수": profile["size"],
                "라벨": profile["label"]
            })
        
        metrics_df = pd.DataFrame(cluster_metrics)
        
        fig = px.scatter(
            metrics_df,
            x="평균_소득",
            y="평균_지출점수", 
            size="고객_수",
            color="클러스터",
            hover_data=["라벨"],
            title="클러스터별 소득 vs 지출 특성",
            labels={
                "평균_소득": "평균 소득 (k$)",
                "평균_지출점수": "평균 지출점수"
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    # 클러스터별 상세 마케팅 전략
    st.subheader("🎯 클러스터별 마케팅 전략 상세")

    for profile in cluster_profiles_list:
        cluster_id = profile["cluster_id"]
        strategy = cluster_analyzer.get_dynamic_marketing_strategy(
            cluster_id, cluster_profiles[cluster_id], cluster_profiles
        )

        with st.expander(f"🎯 클러스터 {cluster_id}: {profile['label']} ({profile['size']}명)", expanded=False):
            
            # 클러스터 핵심 정보 요약
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("평균 연령", f"{profile['avg_age']:.1f}세")
                st.metric("연령 그룹", profile['age_group'])
            
            with col2:
                st.metric("평균 소득", f"${profile['avg_income']:.1f}k")
                st.metric("소득 수준", profile['income_level'])
            
            with col3:
                st.metric("평균 지출점수", f"{profile['avg_spending']:.1f}")
                st.metric("지출 성향", profile['spending_level'])

            # 마케팅 전략 상세
            col1, col2 = st.columns(2)

            with col1:
                st.write("**🎯 고객 프로필 분석:**")
                st.write(f"- **고객 유형**: {profile['customer_type']}")
                st.write(f"- **클러스터 크기**: {profile['size']}명 (전체의 {profile['size']/total_customers*100:.1f}%)")
                st.write(f"- **소득 변동성**: 표준편차 ${profile['std_income']:.1f}k")
                st.write(f"- **지출 변동성**: 표준편차 {profile['std_spending']:.1f}점")
                
                # 성별 분포 (있는 경우)
                gender_ratio = cluster_profiles[cluster_id]["gender_ratio"]
                if gender_ratio:
                    st.write("**성별 분포:**")
                    for gender, ratio in gender_ratio.items():
                        st.write(f"  - {gender}: {ratio:.1%}")

            with col2:
                st.write("**📈 맞춤 마케팅 전략:**")
                st.write(f"**세그먼트**: {strategy['segment']}")
                st.write(f"**우선순위**: {strategy['priority']}")
                
                st.write("**전략 세부사항:**")
                strategy_items = strategy["strategy"].split("; ")
                for i, item in enumerate(strategy_items, 1):
                    st.write(f"  {i}. {item}")

                st.write("**상대적 위치:**")
                st.write(f"- 소득 순위: 상위 {100-float(strategy['percentiles']['income'][:-1]):.0f}%")
                st.write(f"- 지출 순위: 상위 {100-float(strategy['percentiles']['spending'][:-1]):.0f}%")
                st.write(f"- 연령 순위: 상위 {100-float(strategy['percentiles']['age'][:-1]):.0f}%")

            # 특별 권장사항
            if profile["customer_type"] == "프리미엄":
                st.success("💎 **최우선 관리 대상**: 매출 기여도가 가장 높은 핵심 고객층")
            elif profile["customer_type"] == "적극소비":
                st.warning("⚠️ **주의 필요**: 과소비 경향, 신용 관리 지원 필요")
            elif profile["customer_type"] == "보수적":
                st.info("🎯 **잠재력 높음**: 추가 소비 유도 가능한 보수적 고소득층")
            elif profile["customer_type"] == "절약형":
                st.info("💰 **가성비 중심**: 합리적 가격과 효율성을 중시하는 실용 고객층")

            # ROI 예상 및 KPI 제안
            st.write("**💰 예상 ROI 및 추천 KPI:**")
            
            # 고객 유형별 ROI 예상치 계산
            if profile["customer_type"] == "프리미엄":
                expected_roi = "150-200%"
                key_kpis = ["고객생애가치(LTV)", "재구매율", "프리미엄 상품 구매율"]
            elif profile["customer_type"] == "보수적":
                expected_roi = "120-150%"
                key_kpis = ["전환율", "평균 구매금액", "신상품 구매율"]
            elif profile["customer_type"] == "적극소비":
                expected_roi = "100-130%"
                key_kpis = ["구매빈도", "결제완료율", "할부이용률"]
            elif profile["customer_type"] == "절약형":
                expected_roi = "80-110%"
                key_kpis = ["할인상품 구매율", "쿠폰 사용률", "비교구매 횟수"]
            else:
                expected_roi = "100-120%"
                key_kpis = ["구매전환율", "세션당 매출", "재방문율"]
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**예상 ROI**: {expected_roi}")
            with col2:
                st.write("**핵심 KPI**:")
                for kpi in key_kpis:
                    st.write(f"• {kpi}")

    # 마케팅 캠페인 시뮬레이션
    st.subheader("🎮 마케팅 캠페인 시뮬레이션")
    
    st.write("특정 클러스터를 타겟으로 한 캠페인의 예상 효과를 시뮬레이션해보세요.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        target_cluster = st.selectbox(
            "타겟 클러스터 선택",
            options=list(range(selected_k)),
            format_func=lambda x: f"클러스터 {x}: {cluster_profiles_list[x]['label']}"
        )
    
    with col2:
        campaign_budget = st.number_input(
            "캠페인 예산 ($)",
            min_value=1000,
            max_value=100000,
            value=10000,
            step=1000
        )
    
    with col3:
        expected_response_rate = st.slider(
            "예상 반응률 (%)",
            min_value=1.0,
            max_value=20.0,
            value=5.0,
            step=0.5
        )

    if st.button("🚀 캠페인 효과 시뮬레이션 실행"):
        target_profile = cluster_profiles_list[target_cluster]
        target_size = target_profile["size"]
        
        # 시뮬레이션 계산
        expected_responses = int(target_size * expected_response_rate / 100)
        cost_per_customer = campaign_budget / target_size
        cost_per_response = campaign_budget / expected_responses if expected_responses > 0 else 0
        
        # 고객 유형별 예상 수익 계산
        if target_profile["customer_type"] == "프리미엄":
            avg_purchase_value = target_profile["avg_income"] * 2.5
        elif target_profile["customer_type"] == "보수적":
            avg_purchase_value = target_profile["avg_income"] * 1.8
        elif target_profile["customer_type"] == "적극소비":
            avg_purchase_value = target_profile["avg_income"] * 2.0
        else:
            avg_purchase_value = target_profile["avg_income"] * 1.5
            
        expected_revenue = expected_responses * avg_purchase_value
        expected_profit = expected_revenue - campaign_budget
        roi = (expected_profit / campaign_budget * 100) if campaign_budget > 0 else 0

        # 결과 표시
        st.subheader("📊 시뮬레이션 결과")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("예상 반응 고객 수", f"{expected_responses:,}명")
        
        with col2:
            st.metric("고객당 마케팅 비용", f"${cost_per_customer:.2f}")
        
        with col3:
            st.metric("반응당 획득 비용", f"${cost_per_response:.2f}")
        
        with col4:
            st.metric("예상 ROI", f"{roi:+.1f}%")

        # 상세 분석
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**💰 수익성 분석:**")
            st.write(f"- 총 투자 비용: ${campaign_budget:,}")
            st.write(f"- 예상 매출: ${expected_revenue:,.0f}")
            st.write(f"- 예상 순이익: ${expected_profit:,.0f}")
            st.write(f"- 손익분기점 반응률: {(campaign_budget/avg_purchase_value/target_size*100):.1f}%")
        
        with col2:
            st.write("**🎯 캠페인 최적화 제안:**")
            if roi > 50:
                st.success("✅ 매우 수익성 높은 캠페인 - 예산 증액 검토")
            elif roi > 20:
                st.info("📈 수익성 양호 - 현재 전략 유지")
            elif roi > 0:
                st.warning("⚠️ 낮은 수익성 - 타겟팅 또는 메시지 개선 필요")
            else:
                st.error("❌ 손실 예상 - 전략 전면 재검토 필요")
            
            # 개선 제안
            if roi < 20:
                st.write("**개선 방안:**")
                st.write("• 더 높은 반응률을 위한 개인화 메시지")
                st.write("• 특별 할인이나 인센티브 추가")
                st.write("• 다른 클러스터와의 조합 캠페인 고려")

    # 통합 마케팅 전략 요약
    st.subheader("🎯 통합 마케팅 전략 요약")
    
    # 우선순위별 클러스터 정렬
    priority_order = {"최우선": 1, "높음": 2, "중간": 3, "낮음": 4, "보통": 5}
    
    strategy_summary = []
    for profile in cluster_profiles_list:
        cluster_id = profile["cluster_id"]
        strategy = cluster_analyzer.get_dynamic_marketing_strategy(
            cluster_id, cluster_profiles[cluster_id], cluster_profiles
        )
        
        strategy_summary.append({
            "우선순위": strategy["priority"],
            "우선순위_값": priority_order.get(strategy["priority"], 5),
            "클러스터": f"클러스터 {cluster_id}",
            "라벨": profile["label"],
            "고객 수": f"{profile['size']}명",
            "비중": f"{profile['size']/total_customers*100:.1f}%",
            "고객 유형": profile["customer_type"],
            "핵심 전략": strategy["strategy"].split(";")[0],  # 첫 번째 전략만
            "예상 ROI": f"{expected_roi}" if 'expected_roi' in locals() else "계산 필요"
        })
    
    strategy_df = pd.DataFrame(strategy_summary)
    strategy_df = strategy_df.sort_values("우선순위_값").drop("우선순위_값", axis=1)
    
    st.dataframe(strategy_df, use_container_width=True)

    # 실행 로드맵
    with st.expander("🗓️ 마케팅 전략 실행 로드맵"):
        st.markdown("""
        ### 📅 단계별 실행 계획
        
        **1단계: 즉시 실행 (1-2주)**
        - 최우선 클러스터 대상 파일럿 캠페인 실시
        - 고객 세분화 결과를 마케팅 팀에 공유
        - 각 클러스터별 메시지 템플릿 개발
        
        **2단계: 확장 실행 (1-2개월)**
        - 전체 클러스터 대상 차별화된 캠페인 론칭
        - 개인화 추천 시스템 도입
        - 고객별 맞춤 콘텐츠 제작
        
        **3단계: 최적화 (3-6개월)**
        - 캠페인 성과 분석 및 ROI 측정
        - 클러스터링 모델 재조정
        - 예측 모델 기반 선제적 마케팅 도입
        
        **4단계: 고도화 (6-12개월)**
        - AI 기반 실시간 개인화 시스템 구축
        - 고객 생애주기별 자동화 마케팅 실현
        - 크로스셀/업셀 전략 고도화
        
        ### 📊 성과 측정 지표
        
        **단기 지표 (1-3개월)**
        - 클러스터별 캠페인 반응률
        - 고객별 평균 구매금액 변화
        - 개인화 콘텐츠 클릭률
        
        **중기 지표 (3-12개월)**
        - 고객 생애 가치(LTV) 증가율
        - 클러스터 간 이동 패턴 분석
        - 세그먼트별 수익성 개선도
        
        **장기 지표 (1년 이상)**
        - 전체 마케팅 ROI 개선도
        - 고객 만족도 및 브랜드 충성도
        - 시장 점유율 증가
        """)

    return data_with_clusters, cluster_profiles_list
