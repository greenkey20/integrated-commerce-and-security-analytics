"""
클러스터링 분석 페이지

기존 customer_segmentation_app.py의 "클러스터링 분석" 메뉴 내용을 모듈화
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from core.data_processing import DataProcessor
from core.clustering import ClusterAnalyzer
from utils.font_manager import FontManager


def show_clustering_analysis_page():
    """클러스터링 분석 페이지를 표시하는 함수"""
    st.header("🎯 클러스터링 분석")
    
    # 데이터 로드
    data_processor = DataProcessor()
    data = data_processor.load_data()
    
    # 클러스터 분석기 초기화
    cluster_analyzer = ClusterAnalyzer()
    
    # 클러스터링 이론 설명 섹션
    with st.expander("📚 클러스터링 분석 이론 가이드", expanded=True):
        st.markdown("""
        ### 🤔 왜 클러스터 개수를 미리 결정해야 할까요?
        
        K-means 알고리즘의 가장 큰 특징 중 하나는 **사전에 클러스터 개수(K)를 지정해야 한다는 것**입니다. 
        이는 마치 케이크를 자를 때 "몇 조각으로 나눌까?"를 미리 정해야 하는 것과 같습니다. 
        하지만 실제 데이터에서는 최적의 클러스터 개수를 모르기 때문에, 과학적인 방법으로 이를 결정해야 합니다.
        
        ### 📈 엘보우 방법 (Elbow Method)
        
        **핵심 아이디어**: 클러스터 개수에 따른 "성능 대비 효율성"을 측정하는 방법입니다.
        
        - **Inertia(관성)**: 각 데이터 포인트와 해당 클러스터 중심점 간의 거리 제곱의 총합
        - **해석 방법**: 그래프에서 급격히 꺾이는 지점(팔꿈치 모양)을 찾습니다
        
        ### 🎯 실루엣 점수 (Silhouette Score)
        
        **핵심 아이디어**: 각 데이터가 자신의 클러스터에 얼마나 "잘 맞는지"를 측정합니다.
        
        - **점수 범위**: -1 ~ 1 (높을수록 좋음)
        - **의미**: 
          - 0.7~1.0: 매우 좋은 클러스터링
          - 0.5~0.7: 적절한 클러스터링  
          - 0.25~0.5: 약한 클러스터링
          - 0 이하: 잘못된 클러스터링
        """)

    # 최적 클러스터 수 찾기
    st.subheader("🔍 최적 클러스터 수 결정")
    st.write("다양한 클러스터 개수에 대해 엘보우 방법과 실루엣 분석을 수행하여 최적의 K값을 찾아보겠습니다.")

    with st.spinner("최적 클러스터 수를 분석중입니다..."):
        k_range, inertias, silhouette_scores = cluster_analyzer.find_optimal_clusters(data)

    col1, col2 = st.columns(2)

    with col1:
        # 엘보우 방법
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(k_range),
            y=inertias,
            mode='lines+markers',
            name='Inertia',
            line=dict(color='blue', width=3),
            marker=dict(size=8)
        ))
        fig.update_layout(
            title="엘보우 방법: Inertia 변화",
            xaxis_title="클러스터 수",
            yaxis_title="Inertia (관성)",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

        st.info("""
        **📊 이 그래프 해석하기:**
        - 클러스터 수가 증가할수록 Inertia는 감소합니다
        - 급격히 꺾이는 지점(엘보우)을 찾으세요
        - 보통 2-3번 클러스터 지점에서 기울기가 완만해집니다
        """)

    with col2:
        # 실루엣 점수
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(k_range),
            y=silhouette_scores,
            mode='lines+markers',
            name='Silhouette Score',
            line=dict(color='red', width=3),
            marker=dict(size=8)
        ))
        fig.update_layout(
            title="실루엣 점수 변화",
            xaxis_title="클러스터 수",
            yaxis_title="실루엣 점수",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

        # 실루엣 점수 해석
        best_k_silhouette = k_range[np.argmax(silhouette_scores)]
        best_silhouette_score = max(silhouette_scores)

        st.info(f"""
        **📊 이 그래프 해석하기:**
        - 가장 높은 점수: {best_silhouette_score:.3f} (K={best_k_silhouette})
        - 점수가 0.5 이상이면 적절한 클러스터링
        - 가장 높은 지점이 최적의 클러스터 개수입니다
        """)

    # 분석 결과 종합 및 권장사항 제시
    st.subheader("🎯 분석 결과 종합 및 권장사항")

    # 엘보우 방법으로 최적 K 추정
    inertia_diffs = np.diff(inertias)
    inertia_diffs2 = np.diff(inertia_diffs)
    elbow_k = k_range[np.argmax(inertia_diffs2) + 2] if len(inertia_diffs2) > 0 else k_range[0]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="엘보우 방법 추천",
            value=f"{elbow_k}개 클러스터",
            help="Inertia 감소율이 가장 크게 변하는 지점",
        )

    with col2:
        st.metric(
            label="실루엣 점수 추천",
            value=f"{best_k_silhouette}개 클러스터",
            delta=f"점수: {best_silhouette_score:.3f}",
            help="실루엣 점수가 가장 높은 지점",
        )

    with col3:
        # 최종 권장값
        if elbow_k == best_k_silhouette:
            recommended_k = elbow_k
            agreement = "✅ 완전 일치"
        else:
            recommended_k = int((elbow_k + best_k_silhouette) / 2)
            agreement = f"📊 절충안"

        st.metric(
            label="최종 권장",
            value=f"{recommended_k}개 클러스터",
            delta=agreement,
            help="두 방법을 종합한 최종 권장사항",
        )

    # 클러스터 수 선택 슬라이더
    st.subheader("⚙️ 클러스터 수 선택")
    
    if "selected_clusters" not in st.session_state:
        st.session_state.selected_clusters = recommended_k

    selected_k = st.slider(
        "클러스터 수 선택:",
        min_value=2,
        max_value=10,
        value=st.session_state.selected_clusters,
        help=f"분석 결과 권장: {recommended_k}개",
    )

    st.session_state.selected_clusters = selected_k

    # 선택된 K로 클러스터링 수행
    clusters, kmeans, scaler, silhouette_avg = cluster_analyzer.perform_clustering(data, selected_k)
    data_with_clusters = data.copy()
    data_with_clusters["Cluster"] = clusters

    st.success(f"✅ 클러스터링 완료! 실루엣 점수: {silhouette_avg:.3f}")

    # 동적 클러스터 분석 수행
    cluster_profiles = cluster_analyzer.analyze_cluster_characteristics(data_with_clusters, selected_k)
    dynamic_colors = cluster_analyzer.generate_dynamic_colors(cluster_profiles)
    interpretation_guide = cluster_analyzer.generate_dynamic_interpretation_guide(cluster_profiles)

    # 클러스터별 시각화
    st.subheader("클러스터 시각화")

    # 3D 산점도
    fig = px.scatter_3d(
        data_with_clusters,
        x="Age",
        y="Annual Income (k$)",
        z="Spending Score (1-100)",
        color="Cluster",
        title="3D 클러스터 시각화",
        hover_data=["Gender"],
    )
    st.plotly_chart(fig, use_container_width=True)

    # 2D 산점도 (소득 vs 지출점수)
    fig = px.scatter(
        data_with_clusters,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        color="Cluster",
        title="클러스터별 소득 vs 지출점수",
        hover_data=["Age", "Gender"],
    )
    st.plotly_chart(fig, use_container_width=True)

    # 클러스터별 특성 분석
    st.subheader("클러스터별 특성 분석")

    cluster_summary = (
        data_with_clusters.groupby("Cluster")
        .agg({
            "Age": ["mean", "std"],
            "Annual Income (k$)": ["mean", "std"],
            "Spending Score (1-100)": ["mean", "std"],
        })
        .round(2)
    )

    cluster_summary.columns = [
        "평균_연령", "표준편차_연령", "평균_소득", "표준편차_소득", 
        "평균_지출점수", "표준편차_지출점수"
    ]

    st.dataframe(cluster_summary)

    # 클러스터별 고객 수
    cluster_counts = data_with_clusters["Cluster"].value_counts().sort_index()
    fig = px.bar(
        x=cluster_counts.index, 
        y=cluster_counts.values, 
        title="클러스터별 고객 수",
        labels={"x": "클러스터", "y": "고객 수"}
    )
    st.plotly_chart(fig, use_container_width=True)

    # 상세 시각화 (matplotlib 사용)
    st.subheader("🎯 클러스터 분석 결과 상세 시각화")
    
    font_manager = FontManager()
    korean_font_prop = font_manager.get_font_property()
    
    fig_detailed, ax = plt.subplots(figsize=(12, 8))

    # 클러스터 중심점을 원본 스케일로 역변환
    cluster_centers_scaled = kmeans.cluster_centers_
    cluster_centers_original = scaler.inverse_transform(cluster_centers_scaled)
    cluster_centers_2d = cluster_centers_original[:, [1, 2]]

    # 각 클러스터별로 점들 그리기
    for i, profile in enumerate(cluster_profiles):
        cluster_id = profile["cluster_id"]
        mask = data_with_clusters["Cluster"] == cluster_id
        cluster_data = data_with_clusters[mask]

        ax.scatter(
            cluster_data["Annual Income (k$)"],
            cluster_data["Spending Score (1-100)"],
            c=dynamic_colors[i],
            alpha=0.7,
            s=60,
            label=f'클러스터 {cluster_id}: {profile["label"]} ({profile["size"]}명)',
            edgecolors="white",
            linewidth=0.5,
        )

    # 클러스터 중심점 표시
    for i, center in enumerate(cluster_centers_2d):
        ax.scatter(
            center[0], center[1],
            c="black", marker="x", s=300, linewidths=4,
            label="클러스터 중심점" if i == 0 else "",
        )

    # 클러스터 영역을 타원으로 표시
    for i, profile in enumerate(cluster_profiles):
        cluster_id = profile["cluster_id"]
        cluster_data = data_with_clusters[data_with_clusters["Cluster"] == cluster_id]

        if len(cluster_data) > 1:
            mean_income = cluster_data["Annual Income (k$)"].mean()
            mean_spending = cluster_data["Spending Score (1-100)"].mean()
            std_income = cluster_data["Annual Income (k$)"].std()
            std_spending = cluster_data["Spending Score (1-100)"].std()

            ellipse = Ellipse(
                (mean_income, mean_spending),
                width=4 * std_income,
                height=4 * std_spending,
                fill=False,
                color=dynamic_colors[i],
                linewidth=2,
                linestyle="--",
                alpha=0.8,
            )
            ax.add_patch(ellipse)

    # 한글 폰트 적용된 레이블 설정
    if korean_font_prop:
        ax.set_xlabel("연간 소득 (천 달러)", fontproperties=korean_font_prop, fontsize=14)
        ax.set_ylabel("지출 점수 (1-100)", fontproperties=korean_font_prop, fontsize=14)
        ax.set_title(f"클러스터링 결과: {selected_k}개 고객 세분화 완성!", 
                    fontproperties=korean_font_prop, fontsize=16, fontweight="bold")
        
        legend = ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
        for text in legend.get_texts():
            if korean_font_prop:
                text.set_fontproperties(korean_font_prop)
    else:
        ax.set_xlabel("Annual Income (k$)", fontsize=14)
        ax.set_ylabel("Spending Score (1-100)", fontsize=14)
        ax.set_title(f"Clustering Results: {selected_k} Customer Segments Complete!", 
                    fontsize=16, fontweight="bold")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)

    ax.grid(True, alpha=0.3)
    ax.set_xlim(data["Annual Income (k$)"].min() - 5, data["Annual Income (k$)"].max() + 5)
    ax.set_ylim(data["Spending Score (1-100)"].min() - 5, data["Spending Score (1-100)"].max() + 5)

    plt.tight_layout()
    st.pyplot(fig_detailed)

    # 동적 클러스터 해석
    with st.expander("🔍 동적 클러스터 해석 가이드"):
        st.markdown(interpretation_guide)

    st.success(f"✅ 총 {len(data)}명의 고객이 {selected_k}개 그룹으로 성공적으로 분류되었습니다!")
    
    return data_with_clusters, cluster_profiles
