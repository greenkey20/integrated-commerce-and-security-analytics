"""
Online Retail 분석 페이지 - 완전 수정 버전

"혼자 공부하는 머신러닝, 딥러닝" 교재와 연계하여
실무급 데이터 전처리와 선형회귀 모델링을 경험할 수 있는 페이지입니다.

ADP 실기 시험 준비에 필요한 핵심 기법들을 단계별로 학습할 수 있습니다.

전체 워크플로우가 session_state 기반으로 완전히 재구성되어
모든 단계가 일관되게 동작합니다.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.retail_analysis import RetailDataProcessor, RetailVisualizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")


def show_retail_analysis_page():
    """Online Retail 분석 메인 페이지"""
    
    st.title("🛒 Online Retail 고객 분석")
    st.markdown("""
    실제 영국 온라인 소매업체의 거래 데이터를 활용하여 고객별 구매 예측 모델을 구축합니다.
    
    **📚 "혼공머신" 연계 학습 포인트:**
    - 3장: 회귀 알고리즘과 모델 규제 (선형회귀 적용)
    - 실무급 데이터 전처리와 특성 공학 경험
    
    **🎯 ADP 실기 연계 학습 요소:**
    - 대용량 데이터 품질 분석
    - groupby, agg 함수 활용한 집계 분석  
    - 파생 변수 생성 및 특성 공학
    """)
    
    # 🔧 세션 상태 초기화 - 완전 재구성
    initialize_session_state()
    
    # 사이드바에서 분석 단계 선택
    setup_sidebar()
    
    # 선택된 단계에 따른 페이지 렌더링
    analysis_step = st.session_state.get('analysis_step', "1️⃣ 데이터 로딩 & 품질 분석")
    
    if analysis_step == "1️⃣ 데이터 로딩 & 품질 분석":
        show_data_loading_section()
    elif analysis_step == "2️⃣ 데이터 정제 & 전처리":
        show_data_cleaning_section()
    elif analysis_step == "3️⃣ 특성 공학 & 파생변수":
        show_feature_engineering_section()
    elif analysis_step == "4️⃣ 타겟 변수 생성":
        show_target_creation_section()
    elif analysis_step == "5️⃣ 선형회귀 모델링":
        show_modeling_section()
    elif analysis_step == "6️⃣ 모델 평가 & 해석":
        show_evaluation_section()


def initialize_session_state():
    """세션 상태 초기화 - 모든 단계 상태 관리"""
    
    # 기본 상태 초기화
    default_states = {
        'retail_processor': RetailDataProcessor(),
        'retail_data_loaded': False,
        'retail_data_cleaned': False,
        'retail_features_created': False,
        'retail_target_created': False,
        'retail_model_trained': False,
        'retail_model_evaluated': False,
        'analysis_step': "1️⃣ 데이터 로딩 & 품질 분석"
    }
    
    for key, value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = value


def setup_sidebar():
    """사이드바 설정 - 진행 상태 표시 및 메뉴 선택"""
    
    st.sidebar.title("📋 분석 단계")
    
    # 단계 선택 라디오 버튼
    analysis_step = st.sidebar.radio(
        "학습하고 싶은 단계를 선택하세요:",
        [
            "1️⃣ 데이터 로딩 & 품질 분석",
            "2️⃣ 데이터 정제 & 전처리", 
            "3️⃣ 특성 공학 & 파생변수",
            "4️⃣ 타겟 변수 생성",
            "5️⃣ 선형회귀 모델링",
            "6️⃣ 모델 평가 & 해석"
        ]
    )
    
    # 선택된 단계 저장
    st.session_state.analysis_step = analysis_step
    
    # 진행 상태 표시
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🎯 학습 진도:**")
    
    # 🔧 정확한 진행 상태 표시
    progress_steps = [
        ("1️⃣ 데이터 로딩 & 품질 분석", st.session_state.retail_data_loaded),
        ("2️⃣ 데이터 정제 & 전처리", st.session_state.retail_data_cleaned),
        ("3️⃣ 특성 공학 & 파생변수", st.session_state.retail_features_created),
        ("4️⃣ 타겟 변수 생성", st.session_state.retail_target_created),
        ("5️⃣ 선형회귀 모델링", st.session_state.retail_model_trained),
        ("6️⃣ 모델 평가 & 해석", st.session_state.retail_model_evaluated)
    ]
    
    for step_name, completed in progress_steps:
        icon = "✅" if completed else "⏳"
        step_text = step_name.split(' ', 1)[1]
        st.sidebar.markdown(f"{icon} {step_text}")
    
    # 현재 선택된 메뉴 강조
    st.sidebar.markdown("---")
    st.sidebar.info(f"현재 페이지: **{analysis_step}**")
    
    # 사용 가이드
    with st.sidebar.expander("💡 사용 가이드"):
        st.markdown("""
        **🚀 Online Retail 분석 단계:**
        
        1. **데이터 로딩**: UCI 데이터셋 로딩 및 품질 분석
        2. **데이터 정제**: 결측값, 이상치 처리 및 파생변수 생성
        3. **특성 공학**: 고객별 RFM 분석 및 행동 패턴 분석
        4. **타겟 생성**: 미래 구매 예측을 위한 타겟 변수 설계
        5. **모델 훈련**: 선형회귀 모델 훈련 및 성능 평가
        6. **모델 해석**: 비즈니스 관점에서의 모델 해석
        
        **💡 팁:**
        - 순차적으로 진행하는 것을 권장합니다
        - 각 단계의 결과는 자동으로 저장됩니다
        - 언제든지 이전 단계로 돌아갈 수 있습니다
        """)


def show_data_loading_section():
    """1단계: 데이터 로딩 및 품질 분석 - 완전 수정 버전"""
    
    st.header("1️⃣ 데이터 로딩 & 품질 분석")
    
    st.markdown("""
    ### 📖 학습 목표
    - 실무급 대용량 데이터 로딩 경험
    - 체계적인 데이터 품질 분석 방법론 학습
    - ADP 실기의 핵심인 결측값, 이상치 탐지 기법 익히기
    """)
    
    # 데이터 로딩 섹션
    if not st.session_state.retail_data_loaded:
        if st.button("📥 Online Retail 데이터 로딩 시작", type="primary"):
            with st.spinner("데이터를 로딩하는 중입니다..."):
                try:
                    processor = st.session_state.retail_processor
                    data = processor.load_data()
                    
                    # 🔧 session_state에 데이터 저장
                    st.session_state.retail_raw_data = data.copy()
                    st.session_state.retail_data_loaded = True
                    
                    st.success(f"✅ 데이터 로딩 완료: {len(data):,}개 레코드")
                    st.rerun()  # 페이지 새로고침하여 다음 단계 표시
                    
                except Exception as e:
                    st.error(f"❌ 데이터 로딩 실패: {str(e)}")
                    st.info("💡 인터넷 연결을 확인하고 다시 시도해주세요.")
    
    # 로딩된 데이터 정보 표시
    if st.session_state.retail_data_loaded:
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
        
        # 품질 분석 섹션
        st.markdown("---")
        st.subheader("📊 데이터 품질 분석")
        
        if 'retail_quality_report' not in st.session_state:
            if st.button("🔍 품질 분석 실행", type="secondary"):
                with st.spinner("데이터 품질을 분석하는 중입니다..."):
                    try:
                        processor = st.session_state.retail_processor
                        quality_report = processor.analyze_data_quality(data)
                        
                        # 🔧 품질 분석 결과 저장
                        st.session_state.retail_quality_report = quality_report
                        
                        st.success("✅ 품질 분석 완료!")
                        st.rerun()  # 결과 표시를 위한 새로고침
                        
                    except Exception as e:
                        st.error(f"❌ 품질 분석 실패: {str(e)}")
        
        # 품질 분석 결과 표시
        if 'retail_quality_report' in st.session_state:
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
            st.info("💡 품질 분석이 완료되었습니다. 좌측 메뉴에서 '2️⃣ 데이터 정제 & 전처리' 단계로 진행해주세요.")
    
    else:
        st.info("💡 '데이터 로딩 시작' 버튼을 클릭하여 시작해주세요.")


def show_data_cleaning_section():
    """2단계: 데이터 정제 및 전처리 - 완전 수정 버전"""
    
    st.header("2️⃣ 데이터 정제 & 전처리")
    
    # 이전 단계 완료 확인
    if not st.session_state.retail_data_loaded:
        st.warning("⚠️ 먼저 1단계에서 데이터를 로딩해주세요.")
        return
    
    st.markdown("""
    ### 📖 학습 목표  
    - 실무에서 가장 시간이 많이 걸리는 데이터 정제 과정 체험
    - 비즈니스 로직에 기반한 합리적 정제 기준 수립
    - ADP 실기의 핵심인 데이터 변환 및 파생변수 생성
    """)
    
    # 데이터 정제 실행
    if not st.session_state.retail_data_cleaned:
        if st.button("🧹 데이터 정제 시작", type="primary"):
            with st.spinner("데이터를 정제하는 중입니다..."):
                try:
                    # session_state에서 데이터 가져오기
                    raw_data = st.session_state.retail_raw_data
                    processor = st.session_state.retail_processor
                    
                    original_shape = raw_data.shape
                    cleaned_data = processor.clean_data(raw_data)
                    
                    # 🔧 정제된 데이터 저장
                    st.session_state.retail_cleaned_data = cleaned_data.copy()
                    st.session_state.retail_data_cleaned = True
                    
                    st.success("✅ 데이터 정제 완료!")
                    st.rerun()  # 결과 표시를 위한 새로고침
                    
                except Exception as e:
                    st.error(f"❌ 데이터 정제 실패: {str(e)}")
    
    # 정제 결과 표시
    if st.session_state.retail_data_cleaned:
        cleaned_data = st.session_state.retail_cleaned_data
        raw_data = st.session_state.retail_raw_data
        
        st.success("✅ 데이터 정제가 완료되었습니다!")
        
        # 정제 통계
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("정제 전 레코드", f"{len(raw_data):,}")
        with col2:
            st.metric("정제 후 레코드", f"{len(cleaned_data):,}")
        with col3:
            retention_rate = (len(cleaned_data) / len(raw_data)) * 100
            st.metric("데이터 보존율", f"{retention_rate:.1f}%")
        
        # 정제된 데이터 샘플
        st.subheader("🔍 정제된 데이터 샘플")
        st.dataframe(cleaned_data.head(10), use_container_width=True)
        
        # 새로 생성된 변수들
        st.subheader("🆕 생성된 파생 변수들")
        new_columns = ['TotalAmount', 'IsReturn', 'Year', 'Month', 'DayOfWeek', 'Hour']
        for col in new_columns:
            if col in cleaned_data.columns:
                st.info(f"**{col}**: {get_column_description(col)}")
        
        # 다음 단계 안내
        st.markdown("---")
        st.info("💡 데이터 정제가 완료되었습니다. 좌측 메뉴에서 '3️⃣ 특성 공학 & 파생변수' 단계로 진행해주세요.")
    
    else:
        st.info("💡 '데이터 정제 시작' 버튼을 클릭하여 시작해주세요.")


def show_feature_engineering_section():
    """3단계: 특성 공학 및 파생변수 생성 - 완전 수정 버전"""
    
    st.header("3️⃣ 특성 공학 & 파생변수 생성")
    
    # 이전 단계 완료 확인
    if not st.session_state.retail_data_cleaned:
        st.warning("⚠️ 먼저 2단계에서 데이터 정제를 완료해주세요.")
        return
    
    st.markdown("""
    ### 📖 학습 목표
    - 실무에서 가장 중요한 특성 공학(Feature Engineering) 전 과정 체험
    - ADP 실기의 핵심인 groupby, agg 함수 마스터
    - RFM 분석 등 마케팅 분석 기법 적용
    """)
    
    # 특성 공학 실행
    if not st.session_state.retail_features_created:
        if st.button("🏗️ 특성 공학 시작", type="primary"):
            with st.spinner("고객별 특성을 생성하는 중입니다..."):
                try:
                    # session_state에서 정제된 데이터 가져오기
                    cleaned_data = st.session_state.retail_cleaned_data
                    processor = st.session_state.retail_processor
                    
                    customer_features = processor.create_customer_features(cleaned_data)
                    
                    # 🔧 고객 특성 저장
                    st.session_state.retail_customer_features = customer_features.copy()
                    st.session_state.retail_features_created = True
                    
                    st.success("✅ 특성 공학 완료!")
                    st.rerun()  # 결과 표시를 위한 새로고침
                    
                except Exception as e:
                    st.error(f"❌ 특성 공학 실패: {str(e)}")
    
    # 특성 공학 결과 표시
    if st.session_state.retail_features_created:
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
        
        # RFM 분석 시각화
        st.subheader("📊 RFM 분석 결과")
        
        if all(col in customer_features.columns for col in ['recency_days', 'frequency', 'monetary']):
            fig_rfm = make_subplots(
                rows=1, cols=3,
                subplot_titles=['Recency (최근성)', 'Frequency (빈도)', 'Monetary (금액)']
            )
            
            # RFM 히스토그램
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
        
        # 특성 요약 통계
        st.subheader("📋 생성된 특성 요약 통계")
        
        # 주요 특성들만 선택해서 표시
        key_features = ['total_amount', 'frequency', 'recency_days', 'unique_products', 'return_rate']
        available_features = [col for col in key_features if col in customer_features.columns]
        
        if available_features:
            feature_summary = customer_features[available_features].describe().round(2)
            st.dataframe(feature_summary, use_container_width=True)
        
        # 특성 상관관계 히트맵
        if len(available_features) > 1:
            st.subheader("🔗 주요 특성 간 상관관계")
            corr_matrix = customer_features[available_features].corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="특성 간 상관관계 히트맵",
                color_continuous_scale='RdBu_r'
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        
        # 다음 단계 안내
        st.markdown("---")
        st.info("💡 특성 공학이 완료되었습니다. 좌측 메뉴에서 '4️⃣ 타겟 변수 생성' 단계로 진행해주세요.")
    
    else:
        st.info("💡 '특성 공학 시작' 버튼을 클릭하여 시작해주세요.")


def show_target_creation_section():
    """4단계: 타겟 변수 생성 - 완전 수정 버전"""
    
    st.header("4️⃣ 타겟 변수 생성")
    
    # 이전 단계 완료 확인
    if not st.session_state.retail_features_created:
        st.warning("⚠️ 먼저 3단계에서 특성 공학을 완료해주세요.")
        return
    
    st.markdown("""
    ### 📖 학습 목표
    - 비즈니스 문제를 머신러닝 문제로 정의하는 과정 체험
    - 회귀 문제의 타겟 변수 설계 방법론 학습
    - 실무에서 사용하는 타겟 변수 생성 기법 습득
    """)
    
    # 타겟 변수 설정
    if not st.session_state.retail_target_created:
        st.subheader("🎯 예측 목표 설정")
        
        col1, col2 = st.columns(2)
        with col1:
            target_months = st.slider("예측 기간 (개월)", min_value=1, max_value=12, value=3)
        with col2:
            st.write(f"**목표**: 향후 {target_months}개월간 고객별 예상 구매 금액 예측")
        
        if st.button("🎯 타겟 변수 생성", type="primary"):
            with st.spinner("타겟 변수를 생성하는 중입니다..."):
                try:
                    # session_state에서 고객 특성 가져오기
                    customer_features = st.session_state.retail_customer_features
                    processor = st.session_state.retail_processor
                    
                    target_data = processor.create_target_variable(customer_features, target_months=target_months)
                    
                    # 🔧 타겟 데이터 저장
                    st.session_state.retail_target_data = target_data.copy()
                    st.session_state.retail_target_months = target_months
                    st.session_state.retail_target_created = True
                    
                    st.success("✅ 타겟 변수 생성 완료!")
                    st.rerun()  # 결과 표시를 위한 새로고침
                    
                except Exception as e:
                    st.error(f"❌ 타겟 변수 생성 실패: {str(e)}")
    
    # 타겟 변수 분석 결과 표시
    if st.session_state.retail_target_created:
        target_data = st.session_state.retail_target_data
        target_months = st.session_state.retail_target_months
        
        st.success("✅ 타겟 변수가 성공적으로 생성되었습니다!")
        
        target_col = 'predicted_next_amount'
        
        # 타겟 변수 기본 통계
        st.subheader("📊 타겟 변수 기본 통계")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("평균 예측 금액", f"£{target_data[target_col].mean():.2f}")
        with col2:
            st.metric("중앙값", f"£{target_data[target_col].median():.2f}")
        with col3:
            st.metric("표준편차", f"£{target_data[target_col].std():.2f}")
        with col4:
            st.metric("최대값", f"£{target_data[target_col].max():.2f}")
        
        # 타겟 분포 시각화
        st.subheader("📈 타겟 변수 분포 분석")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 히스토그램
            fig_hist = px.histogram(
                target_data, x=target_col, 
                title=f"예측 금액 분포 ({target_months}개월)",
                labels={target_col: '예측 금액 (£)'},
                nbins=30
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # 박스 플롯
            fig_box = px.box(
                target_data, y=target_col,
                title="예측 금액 박스 플롯",
                labels={target_col: '예측 금액 (£)'}
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        # 고객 등급 분포
        if 'customer_value_category' in target_data.columns:
            st.subheader("👥 고객 등급 분포")
            
            category_counts = target_data['customer_value_category'].value_counts()
            fig_pie = px.pie(
                values=category_counts.values, 
                names=category_counts.index,
                title="고객 가치 등급 분포"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # 타겟 변수 상세 분석
        with st.expander("🔍 타겟 변수 상세 분석"):
            st.write("**분위수 분석:**")
            quantiles = target_data[target_col].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(2)
            quantile_df = pd.DataFrame({
                '분위수': ['10%', '25%', '50%', '75%', '90%'],
                '예측 금액 (£)': quantiles.values
            })
            st.dataframe(quantile_df, use_container_width=True)
        
        # 다음 단계 안내
        st.markdown("---")
        st.info("💡 타겟 변수가 생성되었습니다. 좌측 메뉴에서 '5️⃣ 선형회귀 모델링' 단계로 진행해주세요.")
    
    else:
        st.info("💡 예측 기간을 설정하고 '타겟 변수 생성' 버튼을 클릭하여 시작해주세요.")


def show_modeling_section():
    """5단계: 선형회귀 모델링 - 완전 수정 버전"""
    
    st.header("5️⃣ 선형회귀 모델링")
    
    # 이전 단계 완료 확인
    if not st.session_state.retail_target_created:
        st.warning("⚠️ 먼저 4단계에서 타겟 변수를 생성해주세요.")
        return
    
    st.markdown("""
    ### 📖 학습 목표
    - "혼공머신" 3장 선형회귀 알고리즘의 실무 적용
    - 모델 훈련, 검증, 평가의 전체 파이프라인 구축
    - 실무에서 사용하는 모델 성능 평가 방법 학습
    """)
    
    # 모델링 설정
    if not st.session_state.retail_model_trained:
        st.subheader("⚙️ 모델링 설정")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            test_size = st.slider("테스트 데이터 비율", 0.1, 0.4, 0.2, 0.05)
        with col2:
            scale_features = st.checkbox("특성 정규화 수행", value=True)
        with col3:
            random_state = st.number_input("랜덤 시드", 1, 999, 42)
        
        if st.button("🚀 선형회귀 모델 훈련", type="primary"):
            with st.spinner("모델을 훈련하는 중입니다..."):
                try:
                    # session_state에서 타겟 데이터 가져오기
                    target_data = st.session_state.retail_target_data
                    processor = st.session_state.retail_processor
                    
                    # 데이터 준비
                    X, y = processor.prepare_modeling_data(target_data)
                    
                    # 훈련/테스트 분할
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state
                    )
                    
                    # 특성 정규화
                    scaler = None
                    if scale_features:
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        X_train_final = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
                        X_test_final = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
                    else:
                        X_train_final = X_train.copy()
                        X_test_final = X_test.copy()
                    
                    # 모델 훈련
                    model = LinearRegression()
                    model.fit(X_train_final, y_train)
                    
                    # 예측
                    y_train_pred = model.predict(X_train_final)
                    y_test_pred = model.predict(X_test_final)
                    
                    # 🔧 모델 결과 저장
                    st.session_state.retail_model_results = {
                        'model': model,
                        'scaler': scaler,
                        'X_train': X_train_final,
                        'X_test': X_test_final,
                        'y_train': y_train,
                        'y_test': y_test,
                        'y_train_pred': y_train_pred,
                        'y_test_pred': y_test_pred,
                        'feature_names': X.columns.tolist(),
                        'test_size': test_size,
                        'scale_features': scale_features,
                        'random_state': random_state
                    }
                    st.session_state.retail_model_trained = True
                    
                    st.success("✅ 모델 훈련 완료!")
                    st.rerun()  # 결과 표시를 위한 새로고침
                    
                except Exception as e:
                    st.error(f"❌ 모델 훈련 실패: {str(e)}")
    
    # 모델 훈련 결과 표시
    if st.session_state.retail_model_trained:
        model_results = st.session_state.retail_model_results
        
        st.success("✅ 모델 훈련이 완료되었습니다!")
        
        # 모델 성능 지표
        st.subheader("📊 모델 성능 지표")
        
        model = model_results['model']
        y_train = model_results['y_train']
        y_test = model_results['y_test']
        y_train_pred = model_results['y_train_pred']
        y_test_pred = model_results['y_test_pred']
        
        # 성능 계산
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        # 성능 지표 표시
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R² Score (테스트)", f"{test_r2:.3f}")
        with col2:
            st.metric("MAE (테스트)", f"£{test_mae:.2f}")
        with col3:
            st.metric("RMSE (테스트)", f"£{test_rmse:.2f}")
        with col4:
            overfitting = abs(test_r2 - train_r2) > 0.05
            st.metric("과적합 여부", "있음" if overfitting else "없음")
        
        # 예측 vs 실제값 시각화
        st.subheader("📈 예측 성능 시각화")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 예측 vs 실제값 산점도
            fig_pred = px.scatter(
                x=y_test, y=y_test_pred,
                title="예측값 vs 실제값",
                labels={'x': '실제값 (£)', 'y': '예측값 (£)'}
            )
            # 완벽한 예측선 추가
            min_val = min(y_test.min(), y_test_pred.min())
            max_val = max(y_test.max(), y_test_pred.max())
            fig_pred.add_trace(go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines', name='완벽한 예측', line=dict(color='red', dash='dash')
            ))
            st.plotly_chart(fig_pred, use_container_width=True)
        
        with col2:
            # 잔차 히스토그램
            residuals = y_test - y_test_pred
            fig_residuals = px.histogram(
                x=residuals,
                title="잔차 분포",
                labels={'x': '잔차 (£)'}
            )
            st.plotly_chart(fig_residuals, use_container_width=True)
        
        # 특성 중요도
        st.subheader("📈 특성 중요도 분석")
        
        feature_importance = pd.DataFrame({
            '특성명': model_results['feature_names'],
            '회귀계수': model.coef_,
            '절대계수': np.abs(model.coef_)
        }).sort_values('절대계수', ascending=False)
        
        # 상위 10개 특성만 표시
        top_features = feature_importance.head(10)
        
        fig_importance = px.bar(
            top_features,
            x='절대계수',
            y='특성명',
            title="상위 10개 특성 중요도",
            orientation='h'
        )
        fig_importance.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # 특성 중요도 테이블
        with st.expander("🔍 전체 특성 중요도 테이블"):
            st.dataframe(feature_importance, use_container_width=True)
        
        # 다음 단계 안내
        st.markdown("---")
        st.info("💡 모델 훈련이 완료되었습니다. 좌측 메뉴에서 '6️⃣ 모델 평가 & 해석' 단계로 진행해주세요.")
    
    else:
        st.info("💡 모델링 설정을 완료하고 '선형회귀 모델 훈련' 버튼을 클릭하여 시작해주세요.")


def show_evaluation_section():
    """6단계: 모델 평가 및 해석 - 완전 수정 버전"""
    
    st.header("6️⃣ 모델 평가 & 해석")
    
    # 이전 단계 완료 확인
    if not st.session_state.retail_model_trained:
        st.warning("⚠️ 먼저 5단계에서 모델을 훈련해주세요.")
        return
    
    st.markdown("""
    ### 📖 학습 목표
    - 모델 성능의 종합적 평가 방법 학습
    - 잔차 분석을 통한 모델 진단
    - 비즈니스 관점에서의 모델 해석 및 활용 방안 도출
    """)
    
    model_results = st.session_state.retail_model_results
    
    # 종합 성능 평가 실행
    if not st.session_state.retail_model_evaluated:
        if st.button("📊 종합 모델 평가 실행", type="primary"):
            with st.spinner("모델을 종합적으로 평가하는 중입니다..."):
                try:
                    # 평가 메트릭 계산
                    y_train = model_results['y_train']
                    y_test = model_results['y_test']
                    y_train_pred = model_results['y_train_pred']
                    y_test_pred = model_results['y_test_pred']
                    
                    evaluation_metrics = {
                        'r2_train': r2_score(y_train, y_train_pred),
                        'r2_test': r2_score(y_test, y_test_pred),
                        'mae_train': mean_absolute_error(y_train, y_train_pred),
                        'mae_test': mean_absolute_error(y_test, y_test_pred),
                        'rmse_train': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                        'rmse_test': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                        'performance_gap': abs(r2_score(y_test, y_test_pred) - r2_score(y_train, y_train_pred)),
                        'relative_error': (mean_absolute_error(y_test, y_test_pred) / y_test.mean()) * 100
                    }
                    
                    # 🔧 평가 결과 저장
                    st.session_state.retail_evaluation_metrics = evaluation_metrics
                    st.session_state.retail_model_evaluated = True
                    
                    st.success("✅ 모델 평가 완료!")
                    st.rerun()  # 결과 표시를 위한 새로고침
                    
                except Exception as e:
                    st.error(f"❌ 모델 평가 실패: {str(e)}")
    
    # 평가 결과 표시
    if st.session_state.retail_model_evaluated:
        evaluation_metrics = st.session_state.retail_evaluation_metrics
        
        st.success("✅ 모델 평가가 완료되었습니다!")
        
        # 종합 성능 지표 테이블
        st.subheader("📊 종합 성능 평가")
        
        metrics_df = pd.DataFrame({
            '지표': ['R² Score', 'MAE', 'RMSE'],
            '훈련 성능': [
                f"{evaluation_metrics['r2_train']:.4f}",
                f"{evaluation_metrics['mae_train']:.2f}",
                f"{evaluation_metrics['rmse_train']:.2f}"
            ],
            '테스트 성능': [
                f"{evaluation_metrics['r2_test']:.4f}",
                f"{evaluation_metrics['mae_test']:.2f}",
                f"{evaluation_metrics['rmse_test']:.2f}"
            ],
            '차이': [
                f"{evaluation_metrics['r2_test'] - evaluation_metrics['r2_train']:.4f}",
                f"{evaluation_metrics['mae_test'] - evaluation_metrics['mae_train']:.2f}",
                f"{evaluation_metrics['rmse_test'] - evaluation_metrics['rmse_train']:.2f}"
            ]
        })
        
        st.dataframe(metrics_df, use_container_width=True)
        
        # 성능 해석
        st.subheader("💡 성능 해석")
        
        test_r2 = evaluation_metrics['r2_test']
        performance_gap = evaluation_metrics['performance_gap']
        relative_error = evaluation_metrics['relative_error']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎯 모델 성능 평가:**")
            if test_r2 >= 0.8:
                st.success(f"🎉 **우수한 성능**: R² = {test_r2:.3f}")
            elif test_r2 >= 0.6:
                st.info(f"👍 **양호한 성능**: R² = {test_r2:.3f}")
            else:
                st.warning(f"⚠️ **개선 필요**: R² = {test_r2:.3f}")
        
        with col2:
            st.markdown("**🔍 과적합 분석:**")
            if performance_gap <= 0.05:
                st.success("✅ **과적합 없음**")
            else:
                st.warning("⚠️ **과적합 발생**")
        
        # 잔차 분석
        st.subheader("🔍 잔차 분석")
        
        y_test = model_results['y_test']
        y_test_pred = model_results['y_test_pred']
        residuals = y_test - y_test_pred
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 잔차 vs 예측값
            fig_residuals = px.scatter(
                x=y_test_pred, y=residuals,
                title="잔차 vs 예측값",
                labels={'x': '예측값 (£)', 'y': '잔차 (£)'}
            )
            # 기준선 추가
            fig_residuals.add_trace(go.Scatter(
                x=[y_test_pred.min(), y_test_pred.max()], y=[0, 0],
                mode='lines', name='기준선', line=dict(color='red', dash='dash')
            ))
            st.plotly_chart(fig_residuals, use_container_width=True)
        
        with col2:
            # Q-Q 플롯 (정규성 검정)
            from scipy import stats
            fig_qq = go.Figure()
            qq_data = stats.probplot(residuals, dist="norm")
            fig_qq.add_trace(go.Scatter(
                x=qq_data[0][0], y=qq_data[0][1],
                mode='markers', name='잔차'
            ))
            fig_qq.add_trace(go.Scatter(
                x=qq_data[0][0], y=qq_data[1][0] + qq_data[1][1] * qq_data[0][0],
                mode='lines', name='기준선', line=dict(color='red', dash='dash')
            ))
            fig_qq.update_layout(title="Q-Q 플롯 (정규성 검정)", xaxis_title="이론적 분위수", yaxis_title="표본 분위수")
            st.plotly_chart(fig_qq, use_container_width=True)
        
        # 비즈니스 해석
        st.subheader("💼 비즈니스 관점 해석")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎯 예측 정확도:**")
            if relative_error <= 15:
                st.success("🎯 **고정밀도 예측 가능**")
                st.write("개인화된 마케팅 전략 수립 가능")
            elif relative_error <= 25:
                st.info("👍 **세그먼트별 전략 수립**")
                st.write("고객군별 차별화 전략 권장")
            else:
                st.warning("⚠️ **전반적 트렌드 파악**")
                st.write("추가 데이터 수집 및 모델 개선 필요")
        
        with col2:
            st.markdown("**📈 활용 방안:**")
            st.write("• 고객별 예상 구매 금액 예측")
            st.write("• 마케팅 예산 배분 최적화")
            st.write("• 고객 가치 기반 세분화")
            st.write("• 이탈 위험 고객 식별")
        
        # 학습 완료 축하
        st.markdown("---")
        st.subheader("🎓 학습 여정 완료!")
        
        target_months = st.session_state.retail_target_months
        completion_summary = f"""
        **🎉 축하합니다! Online Retail 분석 프로젝트를 완주하셨습니다!**
        
        **📊 최종 모델 성능:**
        - R² Score: {test_r2:.3f}
        - 예측 오차: {relative_error:.1f}%
        - 과적합 여부: {'없음' if performance_gap <= 0.05 else '있음'}
        - 예측 기간: {target_months}개월
        
        **🚀 다음 단계 제안:**
        1. 🔄 **고급 모델 시도**: 랜덤포레스트, XGBoost 등으로 성능 개선
        2. 📊 **특성 엔지니어링 확장**: 시간 기반 특성, 상품 카테고리 분석 추가
        3. 🎯 **분류 문제 도전**: 고객 이탈 예측, 세그먼트 분류 등
        4. 💼 **비즈니스 적용**: 실제 마케팅 캠페인에 모델 적용
        
        **🎯 학습 성과:**
        - 실무급 데이터 전처리 경험
        - 체계적인 특성 공학 과정 습득
        - 머신러닝 모델링 전체 파이프라인 이해
        - 비즈니스 관점에서의 모델 해석 능력 향상
        """
        
        st.success(completion_summary)
        st.balloons()
        
        # 프로젝트 요약 다운로드
        with st.expander("📋 프로젝트 요약 보고서"):
            project_summary = f"""
# Online Retail 분석 프로젝트 요약

## 📊 데이터 개요
- 원본 데이터: {len(st.session_state.retail_raw_data):,}개 레코드
- 정제 후 데이터: {len(st.session_state.retail_cleaned_data):,}개 레코드
- 분석 대상 고객: {len(st.session_state.retail_customer_features):,}명

## 🎯 모델 성능
- R² Score: {test_r2:.3f}
- MAE: {evaluation_metrics['mae_test']:.2f}£
- RMSE: {evaluation_metrics['rmse_test']:.2f}£
- 상대 오차: {relative_error:.1f}%

## 🔧 모델 설정
- 테스트 비율: {model_results['test_size']:.1%}
- 정규화: {'적용' if model_results['scale_features'] else '미적용'}
- 랜덤 시드: {model_results['random_state']}

## 💡 주요 특성 (상위 5개)
{feature_importance.head()['특성명'].tolist()}

## 📈 비즈니스 해석
- 예측 정확도: {'고정밀도' if relative_error <= 15 else '세그먼트별' if relative_error <= 25 else '트렌드 파악'}
- 과적합 여부: {'없음' if performance_gap <= 0.05 else '있음'}
- 활용 가능성: {'높음' if test_r2 >= 0.6 else '보통'}
"""
            st.text_area("프로젝트 요약", project_summary, height=400)
    
    else:
        st.info("💡 '종합 모델 평가 실행' 버튼을 클릭하여 시작해주세요.")


def get_column_description(col_name):
    """컬럼별 설명 반환"""
    descriptions = {
        'TotalAmount': '수량 × 단가로 계산된 거래 총액',
        'IsReturn': '수량이 음수인 경우 True (반품 거래)',
        'Year': '거래 발생 연도',
        'Month': '거래 발생 월 (1-12)',
        'DayOfWeek': '거래 발생 요일 (0=월요일, 6=일요일)',
        'Hour': '거래 발생 시간 (0-23)'
    }
    return descriptions.get(col_name, '파생 변수')
