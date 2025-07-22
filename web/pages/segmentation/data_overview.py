"""
고객 데이터 개요 페이지 - 완전 작동 버전

새로운 데이터 계층(data/processors/segmentation_data_processor.py)을 활용하여
고품질 데이터 개요 및 분석 기능을 제공
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def get_data_processor():
    """데이터 프로세서 초기화 - 안전한 폴백 메커니즘"""
    try:
        # 새로운 데이터 계층 사용
        from data.processors.segmentation_data_processor import DataProcessor
        processor = DataProcessor()
        return processor, 'new_data_layer'
    except ImportError as e:
        st.error(f"❌ 새로운 데이터 계층 로딩 실패: {e}")
        return None, 'failed'


def create_sample_data():
    """표준 Mall Customer 샘플 데이터 생성 - Arror 호환성 개선"""
    np.random.seed(42)
    n_customers = 200

    data = pd.DataFrame({
        'CustomerID': range(1, n_customers + 1),
        'Gender': np.random.choice(['Male', 'Female'], n_customers, p=[0.44, 0.56]),
        'Age': np.random.normal(38.85, 13.97, n_customers).astype(int).clip(18, 70),
        'Annual Income (k$)': np.random.normal(60.56, 26.26, n_customers).astype(int).clip(15, 137),
        'Spending Score (1-100)': np.random.normal(50.2, 25.82, n_customers).astype(int).clip(1, 99)
    })

    # Arrow 호환성을 위해 명시적 타입 지정
    data['Gender'] = data['Gender'].astype(str)
    data['CustomerID'] = data['CustomerID'].astype('int64')
    data['Age'] = data['Age'].astype('int64')
    data['Annual Income (k$)'] = data['Annual Income (k$)'].astype('int64')
    data['Spending Score (1-100)'] = data['Spending Score (1-100)'].astype('int64')

    return data


def show_data_overview_page():
    """고객 데이터 개요 페이지 메인 함수"""
    st.header("📊 고객 데이터 개요 및 분석")

    # 데이터 프로세서 초기화
    data_processor, status = get_data_processor()

    if data_processor is None:
        st.warning("⚠️ 데이터 프로세서를 로드할 수 없습니다. 샘플 데이터를 사용합니다.")
        data = create_sample_data()
        st.info("🎯 표준 Mall Customer 샘플 데이터 (200명)")
    else:
        # 새로운 데이터 계층 사용
        with st.spinner("데이터 로딩 중..."):
            try:
                data = data_processor.load_data()
                st.success(f"✅ 데이터 로드 완료: {len(data):,}명의 고객 데이터")
            except Exception as e:
                st.error(f"데이터 로딩 실패: {e}")
                data = create_sample_data()
                st.info("🔄 샘플 데이터로 대체")

    if data is None or data.empty:
        st.error("❌ 데이터를 로드할 수 없습니다.")
        return None

    # 데이터 검증 및 기본 정보
    validation_results = get_validation_results(data, data_processor)

    # 페이지 구성
    show_basic_info(data, validation_results)
    show_data_quality_check(data, validation_results)
    show_statistical_summary(data)
    show_data_distribution(data)
    show_correlation_analysis(data)
    show_advanced_analysis(data)
    show_data_preview(data)

    return data


def get_validation_results(data, data_processor):
    """데이터 검증 결과 생성"""
    try:
        if data_processor is not None:
            return data_processor.validate_data(data)
        else:
            # 기본 검증
            return {
                'total_rows': len(data),
                'total_columns': len(data.columns),
                'data_types': data.dtypes,
                'has_missing': data.isnull().any().any(),
                'missing_values': data.isnull().sum(),
                'numeric_columns': len(data.select_dtypes(include=[np.number]).columns),
                'categorical_columns': len(data.select_dtypes(exclude=[np.number]).columns),
                'duplicate_rows': data.duplicated().sum()
            }
    except Exception as e:
        st.warning(f"데이터 검증 중 오류: {e}")
        return {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'data_types': data.dtypes,
            'has_missing': False,
            'missing_values': pd.Series([0] * len(data.columns), index=data.columns),
            'numeric_columns': 3,
            'categorical_columns': 2,
            'duplicate_rows': 0
        }


def show_basic_info(data, validation_results):
    """기본 데이터셋 정보 표시"""
    st.subheader("📋 데이터셋 기본 정보")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("전체 고객 수", f"{validation_results['total_rows']:,}명")
    with col2:
        st.metric("특성 수", f"{validation_results['total_columns']}개")
    with col3:
        numeric_count = validation_results.get('numeric_columns',
                                               len(data.select_dtypes(include=[np.number]).columns))
        st.metric("수치형 특성", f"{numeric_count}개")
    with col4:
        categorical_count = validation_results.get('categorical_columns',
                                                   len(data.select_dtypes(exclude=[np.number]).columns))
        st.metric("범주형 특성", f"{categorical_count}개")

    # 데이터 타입 정보 (확장 가능한 섹션)
    with st.expander("📊 컬럼별 상세 정보"):
        dtypes_df = pd.DataFrame({
            "컬럼명": data.columns,
            "데이터 타입": [str(dtype) for dtype in data.dtypes],
            "결측값 수": validation_results['missing_values'].values,
            "유니크 값 수": [data[col].nunique() for col in data.columns]
        })
        st.dataframe(dtypes_df, use_container_width=True)


def show_data_quality_check(data, validation_results):
    """데이터 품질 검사 결과"""
    st.subheader("🔍 데이터 품질 검사")

    col1, col2 = st.columns(2)

    with col1:
        # 결측값 현황
        st.write("**📉 결측값 현황**")
        if not validation_results['has_missing']:
            st.success("✅ 결측값이 없습니다.")
        else:
            missing_data = validation_results['missing_values'][validation_results['missing_values'] > 0]
            st.warning(f"⚠️ {len(missing_data)}개 컬럼에 결측값 발견")
            for col, count in missing_data.items():
                percentage = (count / len(data)) * 100
                st.write(f"- {col}: {count}개 ({percentage:.1f}%)")

        # 중복값 현황
        st.write("**🔄 중복값 현황**")
        duplicate_count = validation_results.get('duplicate_rows', 0)
        if duplicate_count == 0:
            st.success("✅ 중복 행이 없습니다.")
        else:
            st.warning(f"⚠️ {duplicate_count}개의 중복 행 발견")

    with col2:
        # 이상치 탐지 (IQR 방법)
        st.write("**📊 이상치 탐지 (IQR 방법)**")
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col != 'CustomerID':  # ID는 제외
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
                outlier_count = len(outliers)
                outlier_percentage = (outlier_count / len(data)) * 100

                if outlier_count > 0:
                    st.warning(f"⚠️ {col}: {outlier_count}개 ({outlier_percentage:.1f}%)")
                else:
                    st.success(f"✅ {col}: 이상치 없음")


def show_statistical_summary(data):
    """통계적 요약"""
    st.subheader("📊 통계적 요약")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**수치형 특성 기본 통계**")
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            st.dataframe(numeric_data.describe(), use_container_width=True)
        else:
            st.info("수치형 데이터가 없습니다.")

    with col2:
        st.write("**범주형 특성 분포**")
        categorical_data = data.select_dtypes(exclude=[np.number])
        if not categorical_data.empty:
            for col in categorical_data.columns:
                if col != 'CustomerID':
                    st.write(f"**{col}:**")
                    value_counts = data[col].value_counts()
                    for value, count in value_counts.items():
                        percentage = (count / len(data)) * 100
                        st.write(f"  - {value}: {count}명 ({percentage:.1f}%)")
        else:
            st.info("범주형 데이터가 없습니다.")


def show_data_distribution(data):
    """데이터 분포 시각화"""
    st.subheader("📈 고객 특성 분포")

    # 수치형 컬럼 선택 (CustomerID 제외)
    numeric_cols = [col for col in data.select_dtypes(include=[np.number]).columns
                    if col != 'CustomerID']

    if len(numeric_cols) == 0:
        st.warning("시각화할 수치형 데이터가 없습니다.")
        return

    # 히스토그램 생성
    if len(numeric_cols) > 0:
        # 동적으로 subplot 구성
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=numeric_cols
        )

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']

        for i, col in enumerate(numeric_cols):
            row = i // n_cols + 1
            col_pos = i % n_cols + 1

            fig.add_histogram(
                x=data[col],
                name=col,
                row=row, col=col_pos,
                marker_color=colors[i % len(colors)],
                opacity=0.7,
                nbinsx=25,
                showlegend=False
            )

        fig.update_layout(
            height=400 * n_rows,
            title_text="고객 특성별 분포"
        )
        st.plotly_chart(fig, use_container_width=True)

    # 범주형 데이터 분포
    show_categorical_visualizations(data)


def show_categorical_visualizations(data):
    """범주형 데이터 시각화"""
    categorical_cols = [col for col in data.select_dtypes(exclude=[np.number]).columns
                        if col != 'CustomerID']

    if not categorical_cols:
        return

    st.write("**👥 범주형 특성 분포**")

    for col in categorical_cols:
        if data[col].nunique() <= 10:  # 카테고리가 10개 이하인 경우만 시각화
            col1, col2 = st.columns(2)

            with col1:
                # 파이 차트
                value_counts = data[col].value_counts()
                fig = px.pie(
                    values=value_counts.values,
                    names=value_counts.index,
                    title=f"{col} 분포",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # 막대 차트
                fig = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f"{col} 개수",
                    labels={'x': col, 'y': '개수'}
                )
                st.plotly_chart(fig, use_container_width=True)


def show_correlation_analysis(data):
    """상관관계 분석"""
    st.subheader("🔗 특성 간 상관관계 분석")

    # 수치형 데이터만 선택
    numeric_data = data.select_dtypes(include=[np.number])
    if 'CustomerID' in numeric_data.columns:
        numeric_data = numeric_data.drop('CustomerID', axis=1)

    if numeric_data.shape[1] < 2:
        st.info("상관관계 분석을 위해서는 최소 2개의 수치형 특성이 필요합니다.")
        return

    correlation_matrix = numeric_data.corr()

    col1, col2 = st.columns(2)

    with col1:
        # 히트맵
        fig = px.imshow(
            correlation_matrix,
            labels=dict(color="상관계수"),
            title="상관관계 히트맵",
            color_continuous_scale='RdBu_r',
            aspect="auto",
            text_auto=True
        )

        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # 상관관계 해석
        st.write("**📊 상관관계 해석:**")
        st.markdown("""
        - **0.7 이상**: 강한 양의 상관관계 ✅
        - **-0.7 이하**: 강한 음의 상관관계 ❌
        - **0.3 ~ 0.7**: 중간 정도의 양의 상관관계 📈
        - **-0.3 ~ -0.7**: 중간 정도의 음의 상관관계 📉
        - **-0.3 ~ 0.3**: 약한 상관관계 ➡️
        """)

        # 강한 상관관계 찾기
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.5:
                    col1_name = correlation_matrix.columns[i]
                    col2_name = correlation_matrix.columns[j]
                    strong_correlations.append((col1_name, col2_name, corr_value))

        if strong_correlations:
            st.write("**🎯 주목할 만한 상관관계:**")
            for col1, col2, corr in strong_correlations:
                direction = "양의" if corr > 0 else "음의"
                strength = "강한" if abs(corr) > 0.7 else "중간"
                st.write(f"- {col1} ↔ {col2}: {corr:.3f} ({strength} {direction} 상관)")


def show_advanced_analysis(data):
    """고급 분석 기능"""
    st.subheader("🧠 고급 분석")

    # 특성 엔지니어링 미리보기
    with st.expander("🔧 특성 엔지니어링 미리보기"):
        if st.button("🚀 파생 특성 생성"):
            enhanced_data = create_derived_features(data)

            st.write("**새로 생성된 특성:**")
            new_columns = [col for col in enhanced_data.columns if col not in data.columns]

            if new_columns:
                col1, col2 = st.columns(2)

                # 컬럼을 두 그룹으로 나누기
                half = len(new_columns) // 2 if len(new_columns) > 1 else 1

                with col1:
                    for col in new_columns[:half]:
                        st.write(f"**{col}:**")
                        try:
                            value_counts = enhanced_data[col].value_counts()
                            # Arrow 호환성을 위해 use_container_width=False로 변경
                            st.dataframe(value_counts.to_frame('고객 수'), use_container_width=False)
                        except Exception as e:
                            st.write(f"표시 오류: {e}")
                            st.write(enhanced_data[col].value_counts().to_dict())

                with col2:
                    for col in new_columns[half:]:
                        st.write(f"**{col}:**")
                        try:
                            value_counts = enhanced_data[col].value_counts()
                            st.dataframe(value_counts.to_frame('고객 수'), use_container_width=False)
                        except Exception as e:
                            st.write(f"표시 오류: {e}")
                            st.write(enhanced_data[col].value_counts().to_dict())
            else:
                st.warning("새로운 특성이 생성되지 않았습니다.")
                st.write("디버깅 정보:")
                st.write(f"원본 컬럼: {list(data.columns)}")
                st.write(f"향상된 데이터 컬럼: {list(enhanced_data.columns)}")

    # 세그멘테이션 미리보기
    with st.expander("🎯 고객 세그멘테이션 미리보기"):
        if st.button("📊 간단 세그멘테이션 수행"):
            show_basic_segmentation(data)


def create_derived_features(data):
    """파생 특성 생성"""
    enhanced_data = data.copy()

    # 나이 그룹
    if 'Age' in data.columns:
        enhanced_data['Age_Group'] = pd.cut(
            data['Age'],
            bins=[0, 25, 35, 50, 100],
            labels=['청년층(~25)', '성인층(26-35)', '중년층(36-50)', '장년층(51+)']
        ).astype(str)  # ← 이 부분 추가!

    # 소득 그룹  
    if 'Annual Income (k$)' in data.columns:
        enhanced_data['Income_Group'] = pd.cut(
            data['Annual Income (k$)'],
            bins=[0, 40, 70, 100, 200],
            labels=['저소득(~40k)', '중소득(41-70k)', '고소득(71-100k)', '최고소득(101k+)']
        ).astype(str)  # ← 이 부분 추가!

    # 지출 그룹
    if 'Spending Score (1-100)' in data.columns:
        enhanced_data['Spending_Group'] = pd.cut(
            data['Spending Score (1-100)'],
            bins=[0, 30, 60, 100],
            labels=['저지출(~30)', '중지출(31-60)', '고지출(61+)']
        ).astype(str)  # ← 이 부분 추가!

    # 지출 성향 (소득 대비 지출)
    if all(col in data.columns for col in ['Annual Income (k$)', 'Spending Score (1-100)']):
        enhanced_data['Spending_Propensity'] = (
                data['Spending Score (1-100)'] / data['Annual Income (k$)'] * 100
        ).round(2)

    return enhanced_data


def show_basic_segmentation(data):
    """기본 세그멘테이션 수행"""
    if not all(col in data.columns for col in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']):
        st.warning("세그멘테이션을 위한 필수 컬럼이 부족합니다.")
        return

    # 간단한 규칙 기반 세그멘테이션
    segmented_data = data.copy()

    # 조건 정의
    conditions = [
        (data['Age'] < 35) & (data['Spending Score (1-100)'] > 60),
        (data['Age'] >= 35) & (data['Annual Income (k$)'] > 70) & (data['Spending Score (1-100)'] > 60),
        (data['Annual Income (k$)'] <= 50) & (data['Spending Score (1-100)'] <= 40),
        (data['Annual Income (k$)'] > 70) & (data['Spending Score (1-100)'] <= 40),
    ]

    choices = [
        '젊은 고소비자',
        '성숙한 프리미엄 고객',
        '저소득 절약형',
        '고소득 신중형'
    ]

    segmented_data['Segment'] = np.select(conditions, choices, default='일반 고객')

    # 세그먼트 분포 시각화
    segment_counts = segmented_data['Segment'].value_counts()

    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title="고객 세그먼트 분포"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.write("**세그먼트별 고객 수:**")
        for segment, count in segment_counts.items():
            percentage = (count / len(data)) * 100
            st.write(f"- {segment}: {count}명 ({percentage:.1f}%)")


def show_data_preview(data):
    """데이터 미리보기"""
    st.subheader("👀 데이터 미리보기")

    col1, col2 = st.columns(2)

    with col1:
        sample_size = st.selectbox("표시할 행 수:", [5, 10, 20, 50, 100], index=1)

    with col2:
        show_full_stats = st.checkbox("전체 통계 정보 표시", False)

    # 선택된 행 수만큼 데이터 표시
    st.dataframe(data.head(sample_size), use_container_width=True)

    # 전체 통계 정보 (선택사항)
    if show_full_stats:
        st.write("**📊 전체 통계 정보:**")
        st.dataframe(data.describe(include='all'), use_container_width=True)


# 메인 실행 함수
if __name__ == "__main__":
    show_data_overview_page()
