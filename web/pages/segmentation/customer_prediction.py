"""
고객 예측 페이지

기존 customer_segmentation_app.py의 "고객 예측" 메뉴 내용을 모듈화
"""

import streamlit as st
import pandas as pd
import numpy as np
from data._processor import DataProcessor
from core.segmentation.clustering import ClusterAnalyzer


def show_customer_prediction_page():
    """고객 예측 페이지를 표시하는 함수"""
    st.header("🔮 새로운 고객 클러스터 예측")
    
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

    # 동적 클러스터 분석
    data_with_clusters = data.copy()
    data_with_clusters["Cluster"] = clusters
    cluster_profiles = cluster_analyzer.analyze_cluster_characteristics(data_with_clusters, selected_k)

    st.subheader("고객 정보 입력")

    col1, col2, col3 = st.columns(3)

    with col1:
        input_age = st.number_input("연령", min_value=18, max_value=80, value=30)

    with col2:
        input_income = st.number_input("연간 소득 (천 달러)", min_value=15, max_value=150, value=50)

    with col3:
        input_spending = st.number_input("지출 점수 (1-100)", min_value=1, max_value=100, value=50)

    if st.button("클러스터 예측하기", type="primary"):
        # 입력 데이터 전처리
        input_data = np.array([[input_age, input_income, input_spending]])
        input_scaled = scaler.transform(input_data)

        # 예측
        predicted_cluster = kmeans.predict(input_scaled)[0]

        # 클러스터 중심점까지의 거리
        distances = kmeans.transform(input_scaled)[0]
        confidence = 1 / (1 + distances[predicted_cluster])

        # 해당 클러스터의 동적 라벨 찾기
        predicted_profile = next(
            (p for p in cluster_profiles if p["cluster_id"] == predicted_cluster), None
        )
        cluster_label = (
            predicted_profile["label"]
            if predicted_profile
            else f"클러스터 {predicted_cluster}"
        )

        # 결과 표시
        st.success(f"🎯 예측된 클러스터: **{predicted_cluster}번 ({cluster_label})**")
        st.info(f"📊 예측 신뢰도: **{confidence:.2%}**")

        # 해당 클러스터의 특성 표시
        cluster_info = data_with_clusters[data_with_clusters["Cluster"] == predicted_cluster]

        st.subheader(f"클러스터 {predicted_cluster}의 특성 ({selected_k}개 클러스터 기준)")

        col1, col2, col3 = st.columns(3)

        with col1:
            avg_age = cluster_info["Age"].mean()
            st.metric("평균 연령", f"{avg_age:.1f}세")

        with col2:
            avg_income = cluster_info["Annual Income (k$)"].mean()
            st.metric("평균 소득", f"${avg_income:.1f}k")

        with col3:
            avg_spending = cluster_info["Spending Score (1-100)"].mean()
            st.metric("평균 지출점수", f"{avg_spending:.1f}")

        # 예측된 클러스터의 상세 특성
        if predicted_profile:
            st.subheader("예측된 고객 세그먼트 특성")
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**고객 유형**: {predicted_profile['customer_type']}")
                st.write(f"**소득 수준**: {predicted_profile['income_level']}")
                st.write(f"**지출 성향**: {predicted_profile['spending_level']}")
                st.write(f"**연령 그룹**: {predicted_profile['age_group']}")

            with col2:
                st.write(f"**클러스터 크기**: {predicted_profile['size']}명")
                st.write(f"**소득 표준편차**: ${predicted_profile['std_income']:.1f}k")
                st.write(f"**지출 표준편차**: {predicted_profile['std_spending']:.1f}")

            # 마케팅 전략 제안
            st.subheader("🎯 추천 마케팅 전략")
            
            # 클러스터별 마케팅 전략 생성
            cluster_profiles_dict = {}
            for profile in cluster_profiles:
                cluster_id = profile["cluster_id"]
                cluster_data = data_with_clusters[data_with_clusters["Cluster"] == cluster_id]
                cluster_profiles_dict[cluster_id] = {
                    "size": profile["size"],
                    "avg_age": profile["avg_age"],
                    "avg_income": profile["avg_income"],
                    "avg_spending": profile["avg_spending"],
                    "gender_ratio": cluster_data["Gender"].value_counts(normalize=True).to_dict(),
                }

            strategy = cluster_analyzer.get_dynamic_marketing_strategy(
                predicted_cluster, cluster_profiles_dict[predicted_cluster], cluster_profiles_dict
            )

            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**세그먼트 분류:**")
                st.write(f"- {strategy['segment']}")
                st.write(f"- 우선순위: **{strategy['priority']}**")
                
                st.write("**상대적 위치:**")
                st.write(f"- 소득 순위: 상위 {100-float(strategy['percentiles']['income'][:-1]):.0f}%")
                st.write(f"- 지출 순위: 상위 {100-float(strategy['percentiles']['spending'][:-1]):.0f}%")
                st.write(f"- 연령 순위: 상위 {100-float(strategy['percentiles']['age'][:-1]):.0f}%")

            with col2:
                st.write("**구체적 전략:**")
                strategy_items = strategy["strategy"].split("; ")
                for i, item in enumerate(strategy_items, 1):
                    st.write(f"{i}. {item}")

            # 특별 권장사항
            if predicted_profile["customer_type"] == "프리미엄":
                st.success("💎 **최우선 관리 대상**: 매출 기여도가 가장 높은 핵심 고객층입니다.")
            elif predicted_profile["customer_type"] == "적극소비":
                st.warning("⚠️ **주의 필요**: 과소비 경향이 있어 신용 관리 지원이 필요할 수 있습니다.")
            elif predicted_profile["customer_type"] == "보수적":
                st.info("🎯 **잠재력 높음**: 추가 소비를 유도할 수 있는 보수적 고소득층입니다.")
            elif predicted_profile["customer_type"] == "절약형":
                st.info("💰 **가성비 중심**: 합리적인 가격과 가치를 중시하는 고객층입니다.")

        # 유사한 고객들 표시
        st.subheader("👥 유사한 고객 프로필")
        st.write("같은 클러스터에 속한 다른 고객들의 특성을 참고하세요.")
        
        # 클러스터 내에서 입력 고객과 가장 유사한 고객들 찾기
        similar_customers = cluster_info.copy()
        
        # 유사도 계산 (유클리드 거리 기반)
        distances_to_input = []
        for idx, customer in similar_customers.iterrows():
            distance = np.sqrt(
                (customer["Age"] - input_age)**2 + 
                (customer["Annual Income (k$)"] - input_income)**2 + 
                (customer["Spending Score (1-100)"] - input_spending)**2
            )
            distances_to_input.append(distance)
        
        similar_customers["유사도_거리"] = distances_to_input
        similar_customers = similar_customers.sort_values("유사도_거리").head(5)
        
        # 표시할 컬럼 선택
        display_columns = ["Age", "Annual Income (k$)", "Spending Score (1-100)", "Gender"]
        st.dataframe(
            similar_customers[display_columns],
            use_container_width=True
        )
        
        # 입력 고객과의 비교 정보
        st.subheader("📊 클러스터 평균과의 비교")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age_diff = input_age - avg_age
            age_direction = "높음" if age_diff > 0 else "낮음" if age_diff < 0 else "동일"
            st.metric(
                "연령 비교", 
                f"{abs(age_diff):.1f}세 {age_direction}",
                delta=f"{age_diff:+.1f}세"
            )
        
        with col2:
            income_diff = input_income - avg_income
            income_direction = "높음" if income_diff > 0 else "낮음" if income_diff < 0 else "동일"
            st.metric(
                "소득 비교", 
                f"${abs(income_diff):.1f}k {income_direction}",
                delta=f"${income_diff:+.1f}k"
            )
        
        with col3:
            spending_diff = input_spending - avg_spending
            spending_direction = "높음" if spending_diff > 0 else "낮음" if spending_diff < 0 else "동일"
            st.metric(
                "지출점수 비교", 
                f"{abs(spending_diff):.1f}점 {spending_direction}",
                delta=f"{spending_diff:+.1f}점"
            )

    # 클러스터별 개요 정보 표시
    st.subheader("📋 전체 클러스터 개요")
    st.write("현재 설정된 클러스터들의 특성을 한눈에 확인해보세요.")
    
    # 클러스터 요약 테이블 생성
    cluster_summary_data = []
    for profile in cluster_profiles:
        cluster_summary_data.append({
            "클러스터": f"클러스터 {profile['cluster_id']}",
            "라벨": profile['label'],
            "고객 수": f"{profile['size']}명",
            "평균 연령": f"{profile['avg_age']:.1f}세",
            "평균 소득": f"${profile['avg_income']:.1f}k",
            "평균 지출점수": f"{profile['avg_spending']:.1f}",
            "고객 유형": profile['customer_type']
        })
    
    cluster_summary_df = pd.DataFrame(cluster_summary_data)
    st.dataframe(cluster_summary_df, use_container_width=True)
    
    # 예측 가이드 정보
    with st.expander("💡 고객 예측 활용 가이드"):
        st.markdown("""
        ### 🎯 예측 결과 활용 방법
        
        **신뢰도 해석:**
        - **80% 이상**: 매우 높은 신뢰도 - 즉시 해당 전략 적용 가능
        - **60-80%**: 높은 신뢰도 - 기본 전략 적용 후 모니터링
        - **40-60%**: 보통 신뢰도 - 추가 정보 수집 후 재평가 권장
        - **40% 미만**: 낮은 신뢰도 - 다른 분석 방법 고려 필요
        
        **실무 활용 시나리오:**
        
        **온라인 쇼핑몰:**
        - 회원가입 시 기본 정보로 즉시 세그먼트 분류
        - 맞춤형 상품 추천 및 할인 쿠폰 제공
        - 개인화된 이메일 마케팅 캠페인 설계
        
        **오프라인 매장:**
        - 고객 상담 시 세그먼트 정보 활용
        - 매장별 타겟 고객층에 맞는 진열 및 프로모션
        - 판매 직원 교육 시 고객 유형별 접근법 안내
        
        **금융 서비스:**
        - 신규 고객 대상 적절한 금융 상품 추천
        - 리스크 평가 및 신용한도 설정 참고 자료
        - 고객별 맞춤 투자 상품 제안
        
        ### 🔄 지속적 개선 방안
        
        **피드백 수집:**
        - 예측 결과와 실제 고객 행동 비교 분석
        - 마케팅 캠페인 효과 측정 및 모델 정확도 검증
        - 고객 만족도 조사를 통한 세분화 품질 평가
        
        **모델 업데이트:**
        - 주기적인 클러스터링 재수행 (분기별 권장)
        - 새로운 고객 데이터 축적 시 모델 재훈련
        - 시장 변화나 트렌드 반영을 위한 특성 변수 조정
        """)

    return data_with_clusters, predicted_cluster if 'predicted_cluster' in locals() else None
