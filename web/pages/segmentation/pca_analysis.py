"""
주성분 분석(PCA) 페이지

기존 customer_segmentation_app.py의 "주성분 분석" 메뉴 내용을 모듈화
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from data.processors.segmentation_data_processor import DataProcessor


def show_pca_analysis_page():
    """주성분 분석 페이지를 표시하는 함수"""
    st.header("🔬 주성분 분석 (Principal Component Analysis)")
    
    # 데이터 로드
    data_processor = DataProcessor()
    data = data_processor.load_data()

    # PCA 이론 설명 섹션
    with st.expander("📚 주성분 분석 이론 가이드", expanded=True):
        st.markdown("""
        ### 🤔 왜 주성분 분석이 필요할까요?
        
        고객 데이터를 분석할 때 연령, 소득, 지출점수 등 **여러 변수가 동시에 존재**합니다.
        이런 다차원 데이터에서는 변수들 간의 복잡한 관계를 파악하기 어렵고, 시각화도 제한적입니다.
        
        **차원의 저주**: 변수가 많아질수록 데이터 간 거리가 비슷해져서 패턴 찾기가 어려워집니다.
        
        ### 🎯 주성분 분석의 핵심 아이디어
        
        PCA는 **"정보 손실을 최소화하면서 차원을 줄이는"** 방법입니다.
        
        **핵심 원리**: 
        - 데이터의 **분산(퍼짐 정도)을 가장 잘 설명하는 새로운 축**을 찾습니다
        - 이 새로운 축들을 **주성분(Principal Component)**이라고 부릅니다
        - 첫 번째 주성분은 데이터 분산을 가장 많이 설명하고, 두 번째는 그 다음으로 많이 설명합니다
        
        ### 🏢 비즈니스 활용 사례
        
        **고객 분석에서의 PCA 활용**:
        - **고객 특성의 핵심 요인 발견**: 수십 개 변수를 2-3개 핵심 요인으로 압축
        - **시각화 개선**: 3차원 이상 데이터를 2D 평면에서 직관적으로 표현
        - **노이즈 제거**: 중요하지 않은 변동을 걸러내어 핵심 패턴에 집중
        - **저장 공간 절약**: 데이터 압축을 통한 효율적 저장 및 처리
        """)

    # 고객 데이터에 PCA 적용
    st.subheader("🔬 고객 데이터 주성분 분석")
    st.write("고객의 연령, 소득, 지출점수 데이터에 PCA를 적용하여 숨겨진 패턴을 발견해보겠습니다.")

    # 데이터 준비 및 전처리
    features = data[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
    feature_names = data_processor.get_feature_names()

    # 데이터 정규화
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    st.write("**1단계: 데이터 전처리 완료**")
    st.info("PCA는 변수의 스케일에 매우 민감하므로, 모든 변수를 평균 0, 표준편차 1로 정규화했습니다.")

    # PCA 적용
    pca_full = PCA()
    pca_components = pca_full.fit_transform(scaled_features)

    # 주성분별 설명 가능한 분산 비율
    explained_variance_ratio = pca_full.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    st.write("**2단계: 주성분 분석 결과**")

    # 결과 테이블 생성
    pca_results = pd.DataFrame({
        "주성분": [f"PC{i+1}" for i in range(len(explained_variance_ratio))],
        "설명 분산 비율": explained_variance_ratio,
        "누적 설명 비율": cumulative_variance,
        "설명 분산 비율(%)": explained_variance_ratio * 100,
        "누적 설명 비율(%)": cumulative_variance * 100,
    })

    col1, col2 = st.columns(2)

    with col1:
        st.write("**주성분별 기여도:**")
        display_results = pca_results[["주성분", "설명 분산 비율(%)", "누적 설명 비율(%)"]].copy()
        display_results["설명 분산 비율(%)"] = display_results["설명 분산 비율(%)"].round(1)
        display_results["누적 설명 비율(%)"] = display_results["누적 설명 비율(%)"].round(1)
        st.dataframe(display_results, use_container_width=True)

        # 주요 발견사항 요약
        pc1_ratio = explained_variance_ratio[0] * 100
        pc2_ratio = explained_variance_ratio[1] * 100
        pc12_cumulative = cumulative_variance[1] * 100

        st.success(f"""
        **📈 주요 발견사항:**
        - PC1이 전체 변동의 {pc1_ratio:.1f}%를 설명
        - PC2가 추가로 {pc2_ratio:.1f}%를 설명
        - PC1+PC2로 {pc12_cumulative:.1f}%의 정보 보존
        """)

    with col2:
        # 설명 분산 비율 시각화
        fig = go.Figure()

        # 개별 기여도 막대 그래프
        fig.add_trace(go.Bar(
            x=[f"PC{i+1}" for i in range(len(explained_variance_ratio))],
            y=explained_variance_ratio * 100,
            name="개별 기여도",
            marker_color="lightblue",
        ))

        # 누적 기여도 선 그래프
        fig.add_trace(go.Scatter(
            x=[f"PC{i+1}" for i in range(len(cumulative_variance))],
            y=cumulative_variance * 100,
            mode="lines+markers",
            name="누적 기여도",
            line=dict(color="red", width=3),
            marker=dict(size=8),
            yaxis="y2",
        ))

        # 85% 기준선 추가
        fig.add_hline(y=85, line_dash="dash", line_color="gray", 
                     annotation_text="85% 기준선", yref="y2")

        fig.update_layout(
            title="주성분별 설명력 분석",
            xaxis_title="주성분",
            yaxis=dict(title="개별 기여도 (%)", side="left"),
            yaxis2=dict(title="누적 기여도 (%)", side="right", overlaying="y"),
            legend=dict(x=0.7, y=0.95),
        )

        st.plotly_chart(fig, use_container_width=True)

    # 주성분 해석
    st.subheader("🔍 주성분 구성 요소 분석")
    st.write("각 주성분이 원래 변수들(연령, 소득, 지출점수)과 어떤 관계인지 분석해보겠습니다.")

    # 주성분 계수 분석
    components_df = pd.DataFrame(
        pca_full.components_.T,
        columns=[f"PC{i+1}" for i in range(pca_full.n_components_)],
        index=feature_names,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.write("**주성분 구성 계수 (Component Loadings):**")
        styled_components = components_df.style.background_gradient(
            cmap="RdBu_r", axis=None
        ).format("{:.3f}")
        st.dataframe(styled_components, use_container_width=True)

        st.info("""
        **해석 방법:**
        - 양수(+): 해당 변수가 증가하면 주성분 값도 증가
        - 음수(-): 해당 변수가 증가하면 주성분 값은 감소
        - 절댓값이 클수록: 해당 변수의 영향력이 큼
        """)

    with col2:
        # 주성분 구성 히트맵
        fig = px.imshow(
            components_df.T,
            labels=dict(x="원래 변수", y="주성분", color="계수"),
            x=feature_names,
            y=[f"PC{i+1}" for i in range(pca_full.n_components_)],
            color_continuous_scale="RdBu_r",
            aspect="auto",
            title="주성분 구성 히트맵",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # 주성분 해석 생성
    st.write("**🎯 주성분 의미 해석:**")

    # PC1 해석
    pc1_coeffs = components_df["PC1"]
    max_pc1_var = pc1_coeffs.abs().idxmax()
    pc1_direction = "높은" if pc1_coeffs[max_pc1_var] > 0 else "낮은"

    # PC2 해석
    pc2_coeffs = components_df["PC2"]
    max_pc2_var = pc2_coeffs.abs().idxmax()
    pc2_direction = "높은" if pc2_coeffs[max_pc2_var] > 0 else "낮은"

    st.write(f"""
    - **PC1 ({explained_variance_ratio[0]*100:.1f}% 설명)**: {max_pc1_var} 중심의 축으로, {pc1_direction} {max_pc1_var}를 가진 고객들을 구분합니다.
    - **PC2 ({explained_variance_ratio[1]*100:.1f}% 설명)**: {max_pc2_var} 중심의 축으로, {pc2_direction} {max_pc2_var}를 가진 고객들을 구분합니다.
    """)

    # 2D PCA 시각화
    st.subheader("📊 주성분 공간에서의 고객 분포")

    # 2D PCA 결과를 DataFrame에 추가
    pca_2d = PCA(n_components=2)
    pca_2d_result = pca_2d.fit_transform(scaled_features)

    data_pca = data.copy()
    data_pca["PC1"] = pca_2d_result[:, 0]
    data_pca["PC2"] = pca_2d_result[:, 1]

    # 성별로 구분한 PCA 시각화
    fig = px.scatter(
        data_pca,
        x="PC1",
        y="PC2",
        color="Gender",
        title="주성분 공간에서의 고객 분포",
        hover_data=["Age", "Annual Income (k$)", "Spending Score (1-100)"],
        labels={
            "PC1": f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% 설명)",
            "PC2": f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% 설명)",
        },
    )
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # 원래 변수들의 벡터 표시
    show_vectors = st.checkbox("원래 변수들의 방향 벡터 표시", value=False)

    if show_vectors:
        # Biplot 생성
        fig_biplot = go.Figure()

        # 데이터 포인트
        fig_biplot.add_trace(go.Scatter(
            x=data_pca["PC1"],
            y=data_pca["PC2"],
            mode="markers",
            marker=dict(size=6, opacity=0.6),
            name="고객 데이터",
            hovertemplate="PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>",
        ))

        # 변수 벡터 추가
        scale_factor = 3
        for i, feature in enumerate(feature_names):
            fig_biplot.add_trace(go.Scatter(
                x=[0, pca_2d.components_[0, i] * scale_factor],
                y=[0, pca_2d.components_[1, i] * scale_factor],
                mode="lines+markers",
                line=dict(color="red", width=2),
                marker=dict(size=8),
                name=f"{feature} 벡터",
                showlegend=True,
            ))

            # 변수명 라벨 추가
            fig_biplot.add_annotation(
                x=pca_2d.components_[0, i] * scale_factor * 1.1,
                y=pca_2d.components_[1, i] * scale_factor * 1.1,
                text=feature,
                showarrow=False,
                font=dict(size=12, color="red"),
            )

        fig_biplot.update_layout(
            title="PCA Biplot: 고객 분포와 변수 방향",
            xaxis_title=f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% 설명)",
            yaxis_title=f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% 설명)",
            height=600,
        )

        st.plotly_chart(fig_biplot, use_container_width=True)

        st.info("""
        **Biplot 해석 가이드:**
        - 빨간 화살표는 원래 변수들이 주성분 공간에서 향하는 방향을 나타냅니다
        - 화살표가 길수록 해당 변수가 주성분에 더 많이 기여합니다
        - 화살표들 사이의 각도가 작을수록 변수들이 비슷한 패턴을 가집니다
        - 데이터 포인트가 화살표 방향에 있을수록 해당 변수 값이 높습니다
        """)

    # 클러스터링과 PCA 비교
    st.subheader("🔄 PCA와 클러스터링 결과 비교")
    st.write("PCA 공간에서 클러스터링을 수행하면 어떤 결과가 나올까요?")

    if st.button("PCA 공간에서 클러스터링 수행"):
        # Session State에서 클러스터 개수 가져오기
        n_clusters = st.session_state.get("selected_clusters", 5)

        # PCA 공간에서 클러스터링
        kmeans_pca = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters_pca = kmeans_pca.fit_predict(pca_2d_result)

        # 원래 공간에서 클러스터링
        kmeans_original = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters_original = kmeans_original.fit_predict(scaled_features)

        # 결과 비교
        data_comparison = data_pca.copy()
        data_comparison["PCA_Cluster"] = clusters_pca
        data_comparison["Original_Cluster"] = clusters_original

        col1, col2 = st.columns(2)

        with col1:
            # PCA 공간 클러스터링 결과
            fig1 = px.scatter(
                data_comparison,
                x="PC1",
                y="PC2",
                color="PCA_Cluster",
                title="PCA 공간에서의 클러스터링",
                color_discrete_sequence=px.colors.qualitative.Set1,
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            # 원래 공간 클러스터링을 PCA 공간에 투영
            fig2 = px.scatter(
                data_comparison,
                x="PC1",
                y="PC2",
                color="Original_Cluster",
                title="원래 공간 클러스터링의 PCA 투영",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            st.plotly_chart(fig2, use_container_width=True)

        # 클러스터링 결과 비교 분석
        ari_score = adjusted_rand_score(clusters_original, clusters_pca)
        nmi_score = normalized_mutual_info_score(clusters_original, clusters_pca)

        st.write("**클러스터링 결과 유사도 분석:**")
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Adjusted Rand Index (ARI)",
                value=f"{ari_score:.3f}",
                help="1에 가까울수록 두 클러스터링 결과가 유사함",
            )

        with col2:
            st.metric(
                label="Normalized Mutual Information (NMI)",
                value=f"{nmi_score:.3f}",
                help="1에 가까울수록 두 클러스터링이 같은 정보를 공유함",
            )

        if ari_score > 0.7:
            st.success("🎉 PCA 공간과 원래 공간의 클러스터링 결과가 매우 유사합니다!")
        elif ari_score > 0.5:
            st.info("📊 PCA 공간과 원래 공간의 클러스터링 결과가 어느 정도 유사합니다.")
        else:
            st.warning("⚠️ PCA 공간과 원래 공간의 클러스터링 결과가 다릅니다. 차원 축소로 인한 정보 손실이 있을 수 있습니다.")

    # 마무리 인사이트
    with st.expander("💡 주성분 분석 인사이트 및 활용 방안"):
        st.markdown(f"""
        ### 🎯 이번 분석에서 얻은 주요 인사이트:
        
        **차원 축소 효과:**
        - 3차원 고객 데이터를 2차원으로 축소하면서 {cumulative_variance[1]*100:.1f}%의 정보를 보존했습니다
        - 첫 번째 주성분이 {explained_variance_ratio[0]*100:.1f}%의 고객 특성 변동을 설명합니다
        
        **고객 데이터의 숨겨진 패턴:**
        - 고객들의 주요 구분 축은 '{max_pc1_var}'와 '{max_pc2_var}'입니다
        - 이는 마케팅 전략 수립 시 핵심 고려사항이 될 수 있습니다
        
        ### 🏢 비즈니스 활용 방안:
        
        **마케팅 세분화:**
        - PCA 결과를 바탕으로 고객을 2차원 매트릭스로 구분 가능
        - 각 사분면별로 차별화된 마케팅 전략 수립
        
        **데이터 압축 및 효율성:**
        - 고객 프로필을 2-3개 주성분으로 요약하여 저장 공간 절약
        - 실시간 분석 시 처리 속도 향상
        
        **신규 고객 분류:**
        - 새로운 고객의 주성분 점수를 계산하여 즉시 세그먼트 분류 가능
        - 자동화된 고객 온보딩 프로세스 구축
        """)

    st.success("✅ 주성분 분석을 통해 고객 데이터의 핵심 구조를 파악했습니다!")
    
    return data_pca, pca_2d_result
