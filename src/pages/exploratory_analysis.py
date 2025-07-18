"""
탐색적 데이터 분석 페이지

기존 customer_segmentation_app.py의 "탐색적 데이터 분석" 메뉴 내용을 모듈화
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from core.data_processing import DataProcessor


def show_exploratory_analysis_page():
    """탐색적 데이터 분석 페이지를 표시하는 함수"""
    st.header("🔍 탐색적 데이터 분석")
    
    # 데이터 로드
    data_processor = DataProcessor()
    data = data_processor.load_data()

    # 성별 분포
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("성별 분포")
        gender_counts = data["Gender"].value_counts()
        fig = px.pie(
            values=gender_counts.values,
            names=gender_counts.index,
            title="고객 성별 분포",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("연령 분포")
        fig = px.histogram(data, x="Age", nbins=20, title="연령 분포")
        fig.update_layout(xaxis_title="연령", yaxis_title="고객 수")
        st.plotly_chart(fig, use_container_width=True)

    # 소득 vs 지출 점수 산점도
    st.subheader("소득 대비 지출 점수 분석")
    fig = px.scatter(
        data,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        color="Gender",
        title="연간 소득 vs 지출 점수",
        hover_data=["Age"],
    )
    fig.update_layout(
        xaxis_title="연간 소득 (천 달러)", yaxis_title="지출 점수 (1-100)"
    )
    st.plotly_chart(fig, use_container_width=True)

    # 상관관계 히트맵
    st.subheader("특성 간 상관관계")
    numeric_cols = data_processor.get_numeric_columns()
    correlation_matrix = data[numeric_cols].corr()

    fig = px.imshow(
        correlation_matrix,
        labels=dict(color="상관계수"),
        x=numeric_cols,
        y=numeric_cols,
        title="특성 간 상관관계 히트맵",
    )
    st.plotly_chart(fig, use_container_width=True)
    
    return data
