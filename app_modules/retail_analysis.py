"""
Online Retail 분석 페이지 - 간결 버전

"혼자 공부하는 머신러닝, 딥러닝" 교재와 연계하여
실무급 데이터 전처리와 선형회귀 모델링을 경험할 수 있는 페이지입니다.

ADP 실기 시험 준비에 필요한 핵심 기법들을 단계별로 학습할 수 있습니다.
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
    
    # 세션 상태 초기화
    if 'retail_processor' not in st.session_state:
        st.session_state.retail_processor = RetailDataProcessor()
    if 'retail_data_loaded' not in st.session_state:
        st.session_state.retail_data_loaded = False
    
    processor = st.session_state.retail_processor
    
    # 사이드바에서 분석 단계 선택
    st.sidebar.title("📋 분석 단계")
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
    
    # 진행 상태 표시
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🎯 학습 진도:**")
    
    progress_steps = {
        "1️⃣ 데이터 로딩 & 품질 분석": st.session_state.retail_data_loaded,
        "2️⃣ 데이터 정제 & 전처리": hasattr(processor, 'cleaned_data') and processor.cleaned_data is not None,
        "3️⃣ 특성 공학 & 파생변수": hasattr(processor, 'customer_features') and processor.customer_features is not None,
        "4️⃣ 타겟 변수 생성": 'retail_target_data' in st.session_state,
        "5️⃣ 선형회귀 모델링": 'retail_model' in st.session_state,
        "6️⃣ 모델 평가 & 해석": 'retail_model_metrics' in st.session_state
    }
    
    for step, completed in progress_steps.items():
        icon = "✅" if completed else "⏳"
        st.sidebar.markdown(f"{icon} {step.split(' ', 1)[1]}")
    
    # 선택된 단계에 따른 페이지 렌더링
    if analysis_step == "1️⃣ 데이터 로딩 & 품질 분석":
        show_data_loading_section(processor)
    elif analysis_step == "2️⃣ 데이터 정제 & 전처리":
        show_data_cleaning_section(processor)
    elif analysis_step == "3️⃣ 특성 공학 & 파생변수":
        show_feature_engineering_section(processor)
    elif analysis_step == "4️⃣ 타겟 변수 생성":
        show_target_creation_section(processor)
    elif analysis_step == "5️⃣ 선형회귀 모델링":
        show_modeling_section(processor)
    elif analysis_step == "6️⃣ 모델 평가 & 해석":
        show_evaluation_section(processor)


def show_data_loading_section(processor):
    """1단계: 데이터 로딩 및 품질 분석"""
    
    st.header("1️⃣ 데이터 로딩 & 품질 분석")
    
    st.markdown("""
    ### 📖 학습 목표
    - 실무급 대용량 데이터 로딩 경험
    - 체계적인 데이터 품질 분석 방법론 학습
    - ADP 실기의 핵심인 결측값, 이상치 탐지 기법 익히기
    """)
    
    if st.button("📥 Online Retail 데이터 로딩 시작", type="primary"):
        with st.spinner("데이터를 로딩하는 중입니다..."):
            try:
                data = processor.load_data()
                st.session_state.retail_data_loaded = True
                
                st.success(f"✅ 데이터 로딩 완료: {len(data):,}개 레코드")
                
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
                
                # 품질 분석
                if st.button("🔍 품질 분석 실행"):
                    quality_report = processor.analyze_data_quality(data)
                    
                    # 주요 발견사항
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
                
            except Exception as e:
                st.error(f"❌ 데이터 로딩 실패: {str(e)}")


def show_data_cleaning_section(processor):
    """2단계: 데이터 정제 및 전처리"""
    
    st.header("2️⃣ 데이터 정제 & 전처리")
    
    if not st.session_state.retail_data_loaded:
        st.warning("⚠️ 먼저 1단계에서 데이터를 로딩해주세요.")
        return
    
    st.markdown("""
    ### 📖 학습 목표  
    - 실무에서 가장 시간이 많이 걸리는 데이터 정제 과정 체험
    - 비즈니스 로직에 기반한 합리적 정제 기준 수립
    - ADP 실기의 핵심인 데이터 변환 및 파생변수 생성
    """)
    
    if st.button("🧹 데이터 정제 시작", type="primary"):
        with st.spinner("데이터를 정제하는 중입니다..."):
            original_shape = processor.raw_data.shape
            cleaned_data = processor.clean_data(processor.raw_data)
            
            st.success("✅ 데이터 정제 완료!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("정제 전 레코드", f"{original_shape[0]:,}")
            with col2:
                st.metric("정제 후 레코드", f"{len(cleaned_data):,}")
            with col3:
                retention_rate = (len(cleaned_data) / original_shape[0]) * 100
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


def show_feature_engineering_section(processor):
    """3단계: 특성 공학 및 파생변수 생성"""
    
    st.header("3️⃣ 특성 공학 & 파생변수 생성")
    
    if not hasattr(processor, 'cleaned_data') or processor.cleaned_data is None:
        st.warning("⚠️ 먼저 2단계에서 데이터 정제를 완료해주세요.")
        return
    
    st.markdown("""
    ### 📖 학습 목표
    - 실무에서 가장 중요한 특성 공학(Feature Engineering) 전 과정 체험
    - ADP 실기의 핵심인 groupby, agg 함수 마스터
    - RFM 분석 등 마케팅 분석 기법 적용
    """)
    
    if st.button("🏗️ 특성 공학 시작", type="primary"):
        with st.spinner("고객별 특성을 생성하는 중입니다..."):
            customer_features = processor.create_customer_features(processor.cleaned_data)
            
            st.success(f"✅ 특성 공학 완료: {len(customer_features):,}명 고객, {len(customer_features.columns)}개 특성")
            
            # RFM 분석 시각화
            st.subheader("📊 RFM 분석")
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
            
            fig_rfm.update_layout(title="RFM 분석", showlegend=False, height=400)
            st.plotly_chart(fig_rfm, use_container_width=True)
            
            # 특성 요약
            st.subheader("📋 생성된 특성 요약")
            feature_summary = customer_features.describe().round(2)
            st.dataframe(feature_summary, use_container_width=True)


def show_target_creation_section(processor):
    """4단계: 타겟 변수 생성"""
    
    st.header("4️⃣ 타겟 변수 생성")
    
    if not hasattr(processor, 'customer_features') or processor.customer_features is None:
        st.warning("⚠️ 먼저 3단계에서 특성 공학을 완료해주세요.")
        return
    
    st.markdown("""
    ### 📖 학습 목표
    - 비즈니스 문제를 머신러닝 문제로 정의하는 과정 체험
    - 회귀 문제의 타겟 변수 설계 방법론 학습
    """)
    
    # 타겟 변수 설정
    col1, col2 = st.columns(2)
    with col1:
        target_months = st.slider("예측 기간 (개월)", min_value=1, max_value=12, value=3)
    with col2:
        st.write(f"**목표**: 향후 {target_months}개월간 고객별 예상 구매 금액 예측")
    
    if st.button("🎯 타겟 변수 생성", type="primary"):
        with st.spinner("타겟 변수를 생성하는 중입니다..."):
            target_data = processor.create_target_variable(processor.customer_features, target_months=target_months)
            st.session_state.retail_target_data = target_data
            st.session_state.retail_target_months = target_months
            
            st.success("✅ 타겟 변수 생성 완료!")
            
            target_col = 'predicted_next_amount'
            
            # 기본 통계
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("평균 예측 금액", f"£{target_data[target_col].mean():.2f}")
            with col2:
                st.metric("중앙값", f"£{target_data[target_col].median():.2f}")
            with col3:
                st.metric("표준편차", f"£{target_data[target_col].std():.2f}")
            
            # 타겟 분포 시각화
            fig_target = px.histogram(
                target_data, x=target_col, 
                title=f"타겟 변수 분포 - {target_months}개월 예측 금액",
                labels={target_col: '예측 금액 (£)'}
            )
            st.plotly_chart(fig_target, use_container_width=True)


def show_modeling_section(processor):
    """5단계: 선형회귀 모델링"""
    
    st.header("5️⃣ 선형회귀 모델링")
    
    if 'retail_target_data' not in st.session_state:
        st.warning("⚠️ 먼저 4단계에서 타겟 변수를 생성해주세요.")
        return
    
    st.markdown("""
    ### 📖 학습 목표
    - "혼공머신" 3장 선형회귀 알고리즘의 실무 적용
    - 모델 훈련, 검증, 평가의 전체 파이프라인 구축
    """)
    
    target_data = st.session_state.retail_target_data
    
    # 모델링 설정
    col1, col2, col3 = st.columns(3)
    with col1:
        test_size = st.slider("테스트 데이터 비율", 0.1, 0.4, 0.2, 0.05)
    with col2:
        scale_features = st.checkbox("특성 정규화 수행", value=True)
    with col3:
        random_state = st.number_input("랜덤 시드", 1, 999, 42)
    
    if st.button("🚀 선형회귀 모델 훈련", type="primary"):
        with st.spinner("모델을 훈련하는 중입니다..."):
            # 데이터 준비
            X, y = processor.prepare_modeling_data(target_data)
            
            # 훈련/테스트 분할
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # 특성 정규화
            if scale_features:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                X_train_final = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
                X_test_final = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
            else:
                X_train_final = X_train.copy()
                X_test_final = X_test.copy()
                scaler = None
            
            # 모델 훈련
            model = LinearRegression()
            model.fit(X_train_final, y_train)
            
            # 예측
            y_train_pred = model.predict(X_train_final)
            y_test_pred = model.predict(X_test_final)
            
            # 결과 저장
            st.session_state.retail_model = {
                'model': model, 'scaler': scaler,
                'X_train': X_train_final, 'X_test': X_test_final,
                'y_train': y_train, 'y_test': y_test,
                'y_train_pred': y_train_pred, 'y_test_pred': y_test_pred,
                'feature_names': X.columns.tolist()
            }
            
            st.success("✅ 모델 훈련 완료!")
            
            # 성능 지표
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R² Score (테스트)", f"{test_r2:.3f}")
            with col2:
                st.metric("MAE (테스트)", f"£{test_mae:.2f}")
            with col3:
                st.metric("과적합 여부", "없음" if abs(test_r2 - train_r2) <= 0.05 else "있음")
            
            # 예측 vs 실제값 시각화
            fig_pred = px.scatter(
                x=y_test, y=y_test_pred,
                title="예측값 vs 실제값",
                labels={'x': '실제값 (£)', 'y': '예측값 (£)'}
            )
            # 완벽한 예측선 추가
            min_val, max_val = min(y_test.min(), y_test_pred.min()), max(y_test.max(), y_test_pred.max())
            fig_pred.add_trace(go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines', name='완벽한 예측', line=dict(color='red', dash='dash')
            ))
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # 특성 중요도
            feature_importance = pd.DataFrame({
                '특성명': X.columns,
                '회귀계수': model.coef_,
                '절대계수': np.abs(model.coef_)
            }).sort_values('절대계수', ascending=False)
            
            st.subheader("📈 특성 중요도 (상위 10개)")
            st.dataframe(feature_importance.head(10), use_container_width=True)


def show_evaluation_section(processor):
    """6단계: 모델 평가 및 해석"""
    
    st.header("6️⃣ 모델 평가 & 해석")
    
    if 'retail_model' not in st.session_state:
        st.warning("⚠️ 먼저 5단계에서 모델을 훈련해주세요.")
        return
    
    st.markdown("""
    ### 📖 학습 목표
    - 모델 성능의 종합적 평가 방법 학습
    - 잔차 분석을 통한 모델 진단
    - 비즈니스 관점에서의 모델 해석 및 활용 방안 도출
    """)
    
    model_data = st.session_state.retail_model
    model = model_data['model']
    y_test = model_data['y_test']
    y_test_pred = model_data['y_test_pred']
    y_train = model_data['y_train']
    y_train_pred = model_data['y_train_pred']
    
    # 종합 성능 평가
    st.subheader("📊 종합 성능 평가")
    
    metrics = {
        'R² Score': {'train': r2_score(y_train, y_train_pred), 'test': r2_score(y_test, y_test_pred)},
        'MAE': {'train': mean_absolute_error(y_train, y_train_pred), 'test': mean_absolute_error(y_test, y_test_pred)},
        'RMSE': {'train': np.sqrt(mean_squared_error(y_train, y_train_pred)), 'test': np.sqrt(mean_squared_error(y_test, y_test_pred))}
    }
    
    metrics_df = pd.DataFrame({
        '지표': list(metrics.keys()),
        '훈련 성능': [f"{metrics[m]['train']:.4f}" for m in metrics.keys()],
        '테스트 성능': [f"{metrics[m]['test']:.4f}" for m in metrics.keys()],
        '차이': [f"{metrics[m]['test'] - metrics[m]['train']:.4f}" for m in metrics.keys()]
    })
    
    st.dataframe(metrics_df, use_container_width=True)
    
    # 성능 해석
    test_r2 = metrics['R² Score']['test']
    test_mae = metrics['MAE']['test']
    performance_gap = abs(metrics['R² Score']['test'] - metrics['R² Score']['train'])
    
    col1, col2 = st.columns(2)
    with col1:
        if test_r2 >= 0.8:
            st.success(f"🎉 **우수한 성능**: R² = {test_r2:.3f}")
        elif test_r2 >= 0.6:
            st.info(f"👍 **양호한 성능**: R² = {test_r2:.3f}")
        else:
            st.warning(f"⚠️ **개선 필요**: R² = {test_r2:.3f}")
    
    with col2:
        if performance_gap <= 0.05:
            st.success("✅ **과적합 없음**")
        else:
            st.warning("⚠️ **과적합 발생**")
    
    # 잔차 분석
    st.subheader("🔍 잔차 분석")
    residuals = y_test - y_test_pred
    
    fig_residuals = make_subplots(rows=1, cols=2, subplot_titles=['잔차 vs 예측값', '잔차 히스토그램'])
    
    # 잔차 vs 예측값
    fig_residuals.add_trace(
        go.Scatter(x=y_test_pred, y=residuals, mode='markers', name='잔차'),
        row=1, col=1
    )
    fig_residuals.add_trace(
        go.Scatter(x=[y_test_pred.min(), y_test_pred.max()], y=[0, 0], 
                  mode='lines', name='기준선', line=dict(color='red', dash='dash')),
        row=1, col=1
    )
    
    # 잔차 히스토그램
    fig_residuals.add_trace(
        go.Histogram(x=residuals, name='잔차 분포'),
        row=1, col=2
    )
    
    fig_residuals.update_layout(title="잔차 분석", showlegend=False, height=400)
    st.plotly_chart(fig_residuals, use_container_width=True)
    
    # 비즈니스 해석
    st.subheader("💼 비즈니스 관점 해석")
    
    avg_target = y_test.mean()
    relative_error = (test_mae / avg_target) * 100
    
    if relative_error <= 15:
        st.success("🎯 **고정밀도 예측 가능** - 개인화된 마케팅 전략 수립 가능")
    elif relative_error <= 25:
        st.info("👍 **세그먼트별 전략 수립** - 고객군별 차별화 전략 권장")
    else:
        st.warning("⚠️ **전반적 트렌드 파악** - 추가 데이터 수집 및 모델 개선 필요")
    
    # 최종 완료 표시
    st.session_state.retail_model_metrics = {
        'test_r2': test_r2, 'test_mae': test_mae,
        'performance_gap': performance_gap, 'relative_error': relative_error
    }
    
    st.markdown("---")
    st.subheader("🎓 학습 여정 완료!")
    
    completion_summary = f"""
    **🎉 축하합니다! Online Retail 분석 프로젝트를 완주하셨습니다!**
    
    **최종 모델 성능:**
    - R² Score: {test_r2:.3f}
    - 예측 오차: {relative_error:.1f}%
    - 과적합 여부: {'없음' if performance_gap <= 0.05 else '있음'}
    
    **다음 단계 제안:**
    1. 🔄 수준2로 확장: 시간 기반 특성과 상품 카테고리 분석 추가
    2. 📊 다른 알고리즘 비교: 랜덤포레스트, XGBoost 등과 성능 비교
    3. 🎯 분류 문제 도전: 고객 이탈 예측, 세그먼트 분류 등
    """
    
    st.success(completion_summary)
    st.balloons()


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
