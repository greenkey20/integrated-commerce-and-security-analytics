"""
클러스터링 분석 모듈

K-means 클러스터링, 최적 클러스터 수 찾기, 클러스터 특성 분석 등을 담당
"""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from config.settings import ClusteringConfig, VisualizationConfig


class ClusterAnalyzer:
    """클러스터링 분석을 담당하는 클래스"""
    
    def __init__(self, n_clusters=None):
        self.n_clusters = n_clusters or ClusteringConfig.DEFAULT_CLUSTERS
        self.kmeans = None
        self.labels = None
        self.scaler = StandardScaler()
        
    @st.cache_data
    def perform_clustering(_self, data, n_clusters=5):
        """K-means 클러스터링 수행"""
        # 클러스터링을 위한 특성 선택
        features = data[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]

        # 데이터 정규화
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        # K-means 클러스터링
        kmeans = KMeans(
            n_clusters=n_clusters, 
            random_state=ClusteringConfig.RANDOM_STATE, 
            n_init=ClusteringConfig.N_INIT
        )
        clusters = kmeans.fit_predict(scaled_features)

        # 실루엣 점수 계산
        silhouette_avg = silhouette_score(scaled_features, clusters)

        return clusters, kmeans, scaler, silhouette_avg

    def find_optimal_clusters(self, data, max_k=None):
        """엘보우 방법으로 최적 클러스터 수 찾기"""
        max_k = max_k or ClusteringConfig.MAX_CLUSTERS
        features = data[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
        scaled_features = self.scaler.fit_transform(features)

        inertias = []
        silhouette_scores = []
        k_range = range(ClusteringConfig.MIN_CLUSTERS, max_k + 1)

        for k in k_range:
            kmeans = KMeans(
                n_clusters=k, 
                random_state=ClusteringConfig.RANDOM_STATE, 
                n_init=ClusteringConfig.N_INIT
            )
            clusters = kmeans.fit_predict(scaled_features)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(scaled_features, clusters))

        return k_range, inertias, silhouette_scores

    def analyze_cluster_characteristics(self, data_with_clusters, n_clusters):
        """클러스터별 특성을 분석하여 동적 라벨과 색상을 생성"""
        cluster_profiles = []

        for cluster_id in range(n_clusters):
            cluster_data = data_with_clusters[data_with_clusters["Cluster"] == cluster_id]

            if len(cluster_data) == 0:
                continue

            profile = {
                "cluster_id": cluster_id,
                "size": len(cluster_data),
                "avg_income": cluster_data["Annual Income (k$)"].mean(),
                "avg_spending": cluster_data["Spending Score (1-100)"].mean(),
                "avg_age": cluster_data["Age"].mean(),
                "std_income": cluster_data["Annual Income (k$)"].std(),
                "std_spending": cluster_data["Spending Score (1-100)"].std(),
            }
            cluster_profiles.append(profile)

        # 전체 클러스터 대비 상대적 위치 계산
        all_incomes = [p["avg_income"] for p in cluster_profiles]
        all_spendings = [p["avg_spending"] for p in cluster_profiles]
        all_ages = [p["avg_age"] for p in cluster_profiles]

        income_quartiles = np.percentile(all_incomes, [25, 50, 75])
        spending_quartiles = np.percentile(all_spendings, [25, 50, 75])
        age_quartiles = np.percentile(all_ages, [25, 50, 75])

        # 각 클러스터에 대한 동적 라벨 생성
        for profile in cluster_profiles:
            profile.update(self._generate_cluster_labels(
                profile, income_quartiles, spending_quartiles, age_quartiles, all_incomes, all_spendings
            ))

        return cluster_profiles

    def _generate_cluster_labels(self, profile, income_quartiles, spending_quartiles, age_quartiles, all_incomes, all_spendings):
        """클러스터 라벨 생성 헬퍼 메서드"""
        # 소득 수준 분류 (더 세분화)
        if profile["avg_income"] >= income_quartiles[2]:
            if profile["avg_income"] >= np.percentile(all_incomes, 90):
                income_level = "최고소득"
            else:
                income_level = "고소득"
        elif profile["avg_income"] >= income_quartiles[1]:
            income_level = "중상소득"
        elif profile["avg_income"] >= income_quartiles[0]:
            income_level = "중하소득"
        else:
            income_level = "저소득"

        # 지출 수준 분류 (더 세분화)
        if profile["avg_spending"] >= spending_quartiles[2]:
            if profile["avg_spending"] >= np.percentile(all_spendings, 90):
                spending_level = "최고지출"
            else:
                spending_level = "고지출"
        elif profile["avg_spending"] >= spending_quartiles[1]:
            spending_level = "중상지출"
        elif profile["avg_spending"] >= spending_quartiles[0]:
            spending_level = "중하지출"
        else:
            spending_level = "저지출"

        # 연령대 분류
        if profile["avg_age"] <= age_quartiles[0]:
            age_group = "청년층"
        elif profile["avg_age"] <= age_quartiles[1]:
            age_group = "청장년층"
        elif profile["avg_age"] <= age_quartiles[2]:
            age_group = "중년층"
        else:
            age_group = "장년층"

        # 고객 유형 결정 (소득과 지출 조합)
        if income_level in ["최고소득", "고소득"] and spending_level in ["최고지출", "고지출"]:
            customer_type = "프리미엄"
        elif income_level in ["최고소득", "고소득"] and spending_level in ["저지출", "중하지출"]:
            customer_type = "보수적"
        elif income_level in ["저소득", "중하소득"] and spending_level in ["고지출", "최고지출"]:
            customer_type = "적극소비"
        elif income_level in ["저소득", "중하소득"] and spending_level in ["저지출", "중하지출"]:
            customer_type = "절약형"
        else:
            customer_type = "일반"

        return {
            "label": f"{customer_type} {age_group}",
            "income_level": income_level,
            "spending_level": spending_level,
            "age_group": age_group,
            "customer_type": customer_type,
        }

    def generate_dynamic_colors(self, cluster_profiles):
        """클러스터 특성에 따른 일관된 색상 매핑 생성"""
        colors = []
        for i, profile in enumerate(cluster_profiles):
            # 고객 유형에 따른 색상 선택
            if profile["customer_type"] == "프리미엄":
                colors.append("#e41a1c")  # 빨강
            elif profile["customer_type"] == "보수적":
                colors.append("#377eb8")  # 파랑
            elif profile["customer_type"] == "적극소비":
                colors.append("#984ea3")  # 보라
            elif profile["customer_type"] == "절약형":
                colors.append("#ff7f00")  # 주황
            else:  # 일반
                colors.append(VisualizationConfig.COLOR_PALETTE[i % len(VisualizationConfig.COLOR_PALETTE)])

        return colors

    def generate_dynamic_interpretation_guide(self, cluster_profiles):
        """동적 클러스터 해석 가이드 생성"""
        if len(cluster_profiles) == 0:
            return "클러스터 분석 결과를 확인할 수 없습니다."

        # 소득과 지출 범위 계산
        min_income = min(p["avg_income"] for p in cluster_profiles)
        max_income = max(p["avg_income"] for p in cluster_profiles)
        min_spending = min(p["avg_spending"] for p in cluster_profiles)
        max_spending = max(p["avg_spending"] for p in cluster_profiles)
        min_age = min(p["avg_age"] for p in cluster_profiles)
        max_age = max(p["avg_age"] for p in cluster_profiles)

        # 분류 기준 계산 (사분위수)
        all_incomes = [p["avg_income"] for p in cluster_profiles]
        all_spendings = [p["avg_spending"] for p in cluster_profiles]
        all_ages = [p["avg_age"] for p in cluster_profiles]

        income_quartiles = np.percentile(all_incomes, [25, 50, 75, 90])
        spending_quartiles = np.percentile(all_spendings, [25, 50, 75, 90])
        age_quartiles = np.percentile(all_ages, [25, 50, 75])

        guide_text = f"""
    **현재 {len(cluster_profiles)}개 클러스터 분석 결과 해석:**
    
    **전체 데이터 범위:**
    - 소득 범위: ${min_income:.1f}k ~ ${max_income:.1f}k
    - 지출점수 범위: {min_spending:.1f} ~ {max_spending:.1f}
    - 연령 범위: {min_age:.1f}세 ~ {max_age:.1f}세
    
    **동적 라벨링 분류 기준:**
    
    **소득 수준 분류 기준:**
    - 최고소득: ${income_quartiles[3]:.1f}k 이상 (상위 10%)
    - 고소득: ${income_quartiles[2]:.1f}k ~ ${income_quartiles[3]:.1f}k (상위 25%)
    - 중상소득: ${income_quartiles[1]:.1f}k ~ ${income_quartiles[2]:.1f}k (상위 50%)
    - 중하소득: ${income_quartiles[0]:.1f}k ~ ${income_quartiles[1]:.1f}k (하위 50%)
    - 저소득: ${income_quartiles[0]:.1f}k 미만 (하위 25%)
    
    **지출 성향 분류 기준:**
    - 최고지출: {spending_quartiles[3]:.1f}점 이상 (상위 10%)
    - 고지출: {spending_quartiles[2]:.1f}점 ~ {spending_quartiles[3]:.1f}점 (상위 25%)
    - 중상지출: {spending_quartiles[1]:.1f}점 ~ {spending_quartiles[2]:.1f}점 (상위 50%)
    - 중하지출: {spending_quartiles[0]:.1f}점 ~ {spending_quartiles[1]:.1f}점 (하위 50%)
    - 저지출: {spending_quartiles[0]:.1f}점 미만 (하위 25%)
    
    **연령대 분류 기준:**
    - 청년층: {age_quartiles[0]:.1f}세 미만
    - 청장년층: {age_quartiles[0]:.1f}세 ~ {age_quartiles[1]:.1f}세
    - 중년층: {age_quartiles[1]:.1f}세 ~ {age_quartiles[2]:.1f}세
    - 장년층: {age_quartiles[2]:.1f}세 이상
    
    **고객 유형 정의:**
    - **프리미엄**: 고소득 + 고지출 조합 → 최우선 관리 대상
    - **보수적**: 고소득 + 저지출 조합 → 추가 소비 유도 가능
    - **적극소비**: 저소득 + 고지출 조합 → 신용 관리 지원 필요
    - **절약형**: 저소득 + 저지출 조합 → 가성비 중심 접근
    - **일반**: 위 조합에 해당하지 않는 중간 성향
    
    **각 클러스터의 상세 특성:**
    """

        # 소득 순으로 정렬하여 설명
        sorted_profiles = sorted(cluster_profiles, key=lambda x: x["avg_income"], reverse=True)

        for profile in sorted_profiles:
            guide_text += f"""
    - **클러스터 {profile['cluster_id']} ({profile['label']})**: 
      평균 소득 ${profile['avg_income']:.1f}k, 지출점수 {profile['avg_spending']:.1f}, 평균 연령 {profile['avg_age']:.1f}세
      고객 수 {profile['size']}명, 고객 유형: {profile['customer_type']}
      ({profile['income_level']} × {profile['spending_level']} × {profile['age_group']} 조합)
        """

        guide_text += f"""
    
    **클러스터링 품질 지표:**
    - 클러스터 간 소득 격차: ${max_income - min_income:.1f}k
    - 클러스터 간 지출성향 차이: {max_spending - min_spending:.1f}점
    - 클러스터 간 연령 차이: {max_age - min_age:.1f}세
    - 가장 큰 클러스터: {max(cluster_profiles, key=lambda x: x['size'])['size']}명
    - 가장 작은 클러스터: {min(cluster_profiles, key=lambda x: x['size'])['size']}명
    - 클러스터 크기 편차: {np.std([p['size'] for p in cluster_profiles]):.1f}명
    """

        return guide_text

    def get_dynamic_marketing_strategy(self, cluster_id, profile, all_profiles):
        """각 클러스터의 상대적 특성을 고려한 동적 마케팅 전략 생성"""
        # 전체 클러스터 대비 상대적 위치 계산
        all_incomes = [p["avg_income"] for p in all_profiles.values()]
        all_spendings = [p["avg_spending"] for p in all_profiles.values()]
        all_ages = [p["avg_age"] for p in all_profiles.values()]

        income_percentile = (
            sum(1 for x in all_incomes if x < profile["avg_income"]) / len(all_incomes)
        ) * 100
        spending_percentile = (
            sum(1 for x in all_spendings if x < profile["avg_spending"]) / len(all_spendings)
        ) * 100
        age_percentile = (
            sum(1 for x in all_ages if x < profile["avg_age"]) / len(all_ages)
        ) * 100

        # 소득 수준 분류
        if income_percentile >= 75:
            income_level = "고소득"
        elif income_percentile >= 40:
            income_level = "중간소득"
        else:
            income_level = "저소득"

        # 지출 수준 분류
        if spending_percentile >= 75:
            spending_level = "고지출"
        elif spending_percentile >= 40:
            spending_level = "중간지출"
        else:
            spending_level = "저지출"

        # 연령대 분류
        if age_percentile <= 25:
            age_group = "젊은층"
        elif age_percentile >= 75:
            age_group = "중장년층"
        else:
            age_group = "중간연령층"

        # 세그먼트 명 생성
        segment_name = f"{income_level} {spending_level} {age_group}"

        # 전략 생성
        strategies = []
        priorities = []

        # 소득 기반 전략
        if income_level == "고소득":
            if spending_level == "고지출":
                strategies.append("프리미엄 제품 라인 집중, VIP 서비스")
                priorities.append("최우선")
            elif spending_level == "저지출":
                strategies.append("가치 제안 마케팅, 투자 상품 소개")
                priorities.append("높음")
            else:
                strategies.append("품질 중심 마케팅, 브랜드 가치 강조")
                priorities.append("높음")
        elif income_level == "중간소득":
            if spending_level == "고지출":
                strategies.append("할부 서비스, 캐시백 혜택")
                priorities.append("중간")
            else:
                strategies.append("합리적 가격대 제품, 프로모션 활용")
                priorities.append("중간")
        else:  # 저소득
            strategies.append("저가 제품 라인, 대량 할인, 멤버십 혜택")
            priorities.append("낮음")

        # 연령 기반 추가 전략
        if age_group == "젊은층":
            strategies.append("소셜미디어 마케팅, 온라인 채널 강화")
        elif age_group == "중장년층":
            strategies.append("오프라인 매장 서비스, 전화 상담 강화")
        else:
            strategies.append("옴니채널 접근, 다양한 커뮤니케이션")

        # 특별한 조합에 대한 맞춤 전략
        if income_level == "저소득" and spending_level == "고지출":
            strategies.append("신용 관리 서비스, 예산 관리 도구 제공")
        elif income_level == "고소득" and spending_level == "저지출":
            strategies.append("절약 보상 프로그램, 장기 고객 혜택")

        return {
            "segment": segment_name,
            "strategy": "; ".join(strategies),
            "priority": priorities[0] if priorities else "보통",
            "income_level": income_level,
            "spending_level": spending_level,
            "age_group": age_group,
            "percentiles": {
                "income": f"{income_percentile:.0f}%",
                "spending": f"{spending_percentile:.0f}%",
                "age": f"{age_percentile:.0f}%",
            },
        }


# 전역 인스턴스 생성 (기존 함수와의 호환성)
cluster_analyzer = ClusterAnalyzer()

# 기존 함수들과의 호환성을 위한 래퍼들
def perform_clustering(data, n_clusters=5):
    return cluster_analyzer.perform_clustering(data, n_clusters)

def find_optimal_clusters(data, max_k=10):
    return cluster_analyzer.find_optimal_clusters(data, max_k)

def analyze_cluster_characteristics(data_with_clusters, n_clusters):
    return cluster_analyzer.analyze_cluster_characteristics(data_with_clusters, n_clusters)

def generate_dynamic_colors(cluster_profiles):
    return cluster_analyzer.generate_dynamic_colors(cluster_profiles)

def generate_dynamic_interpretation_guide(cluster_profiles):
    return cluster_analyzer.generate_dynamic_interpretation_guide(cluster_profiles)

def get_dynamic_marketing_strategy(cluster_id, profile, all_profiles):
    return cluster_analyzer.get_dynamic_marketing_strategy(cluster_id, profile, all_profiles)
