"""
Online Retail 모델링 페이지

선형회귀 모델 훈련을 담당하는 Streamlit 페이지
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.retail.model_trainer import RetailModelTrainer
from core.retail.visualizer import RetailVisualizer
import warnings

warnings.filterwarnings("ignore")


def show_modeling_page():
    """선형회귀 모델링 페이지"""
    
    st.header("5️⃣ 선형회귀 모델링")
    
    # 이전 단계 완료 확인
    if not st.session_state.get('retail_target_created', False):
        st.warning("⚠️ 먼저 4단계에서 타겟 변수를 생성해주세요.")
        return
    
    st.markdown("""
    ### 📖 학습 목표
    - "혼공머신" 3장 선형회귀 알고리즘의 실무 적용
    - 모델 훈련, 검증, 평가의 전체 파이프라인 구축
    - 실무에서 사용하는 모델 성능 평가 방법 학습
    """)
    
    # 세션 상태 초기화
    if 'retail_model_trainer' not in st.session_state:
        st.session_state.retail_model_trainer = RetailModelTrainer()
    
    # 모델링 설정
    if not st.session_state.get('retail_model_trained', False):
        st.subheader("⚙️ 모델링 설정")
        
        st.markdown("""
        #### 선형회귀 모델 개요
        선형회귀는 독립변수와 종속변수 간의 선형 관계를 모델링하는 통계 기법입니다.
        
        **수식**: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
        - y: 타겟 변수 (예측 구매 금액)
        - x₁, x₂, ..., xₙ: 특성 변수들 (RFM, 구매 패턴 등)
        - β₀, β₁, ..., βₙ: 회귀 계수
        - ε: 오차항
        """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            test_size = st.slider("테스트 데이터 비율", 0.1, 0.4, 0.2, 0.05)
        with col2:
            scale_features = st.checkbox("특성 정규화 수행", value=True)
        with col3:
            random_state = st.number_input("랜덤 시드", 1, 999, 42)
        
        # 설정 설명
        with st.expander("⚙️ 모델링 설정 설명"):
            st.markdown("""
            **테스트 데이터 비율**: 전체 데이터 중 모델 검증용으로 사용할 비율
            - 일반적으로 20-30% 권장
            - 너무 작으면 검증 신뢰도 낮음, 너무 크면 훈련 데이터 부족
            
            **특성 정규화**: 각 특성의 스케일을 0-1 또는 평균 0, 표준편차 1로 조정
            - 선형회귀에서 특성 간 스케일 차이가 클 때 유용
            - 해석 시 원래 스케일로 역변환 필요
            
            **랜덤 시드**: 결과 재현을 위한 난수 생성 시드
            - 동일한 시드 사용 시 동일한 결과 보장
            """)
        
        if st.button("🚀 선형회귀 모델 훈련", type="primary"):
            with st.spinner("모델을 훈련하는 중입니다..."):
                try:
                    # 세션 상태에서 타겟 데이터 가져오기
                    target_data = st.session_state.retail_target_data
                    trainer = st.session_state.retail_model_trainer
                    
                    # 데이터 준비
                    X, y = trainer.prepare_modeling_data(target_data)
                    
                    # 모델 훈련
                    training_results = trainer.train_model(
                        X, y, 
                        test_size=test_size, 
                        scale_features=scale_features,
                        random_state=random_state
                    )
                    
                    # 훈련 결과 저장
                    st.session_state.retail_training_results = training_results
                    st.session_state.retail_model_trained = True
                    
                    st.success("✅ 모델 훈련 완료!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ 모델 훈련 실패: {str(e)}")
                    st.info("데이터나 설정을 확인하고 다시 시도해주세요.")
    
    # 모델 훈련 결과 표시
    if st.session_state.get('retail_model_trained', False):
        training_results = st.session_state.retail_training_results
        
        st.success("✅ 모델 훈련이 완료되었습니다!")
        
        # 모델 기본 정보
        st.subheader("📊 모델 기본 정보")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("훈련 샘플 수", f"{len(training_results['y_train']):,}")
        with col2:
            st.metric("테스트 샘플 수", f"{len(training_results['y_test']):,}")
        with col3:
            st.metric("특성 개수", f"{len(training_results['feature_names'])}")
        with col4:
            st.metric("정규화 여부", "적용" if training_results['scale_features'] else "미적용")
        
        # 모델 성능 지표
        st.subheader("📈 모델 성능 지표")
        
        model = training_results['model']
        y_train = training_results['y_train']
        y_test = training_results['y_test']
        y_train_pred = training_results['y_train_pred']
        y_test_pred = training_results['y_test_pred']
        
        # 성능 계산
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
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
        
        # 성능 지표 설명
        with st.expander("📚 성능 지표 설명"):
            st.markdown("""
            **R² Score (결정계수)**
            - 모델이 데이터의 분산을 얼마나 잘 설명하는지 나타내는 지표
            - 1에 가까울수록 좋음 (0.8 이상: 우수, 0.6 이상: 양호)
            
            **MAE (Mean Absolute Error)**
            - 예측값과 실제값의 절대 오차의 평균
            - 실제 단위로 해석 가능 (£)
            
            **RMSE (Root Mean Square Error)**
            - 예측값과 실제값의 제곱 오차의 평균의 제곱근
            - 큰 오차에 더 민감함
            
            **과적합 (Overfitting)**
            - 훈련 성능과 테스트 성능의 차이가 클 때 발생
            - R² 차이가 0.05 이상이면 과적합 의심
            """)
        
        # 예측 성능 시각화
        st.subheader("📊 예측 성능 시각화")
        
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
        
        # 추가 성능 분석
        st.subheader("🔍 상세 성능 분석")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 훈련 vs 테스트 성능 비교
            performance_comparison = pd.DataFrame({
                '지표': ['R² Score', 'MAE', 'RMSE'],
                '훈련 성능': [
                    f"{train_r2:.4f}",
                    f"{mean_absolute_error(y_train, y_train_pred):.2f}",
                    f"{np.sqrt(mean_squared_error(y_train, y_train_pred)):.2f}"
                ],
                '테스트 성능': [
                    f"{test_r2:.4f}",
                    f"{test_mae:.2f}",
                    f"{test_rmse:.2f}"
                ]
            })
            
            st.markdown("**성능 비교표**")
            st.dataframe(performance_comparison, use_container_width=True)
        
        with col2:
            # 예측 오차 분포
            error_percentages = np.abs(residuals) / y_test * 100
            
            accuracy_ranges = ['< 10%', '10-20%', '20-30%', '30-50%', '> 50%']
            accuracy_counts = [
                np.sum(error_percentages < 10),
                np.sum((error_percentages >= 10) & (error_percentages < 20)),
                np.sum((error_percentages >= 20) & (error_percentages < 30)),
                np.sum((error_percentages >= 30) & (error_percentages < 50)),
                np.sum(error_percentages >= 50)
            ]
            
            fig_accuracy = px.bar(
                x=accuracy_ranges,
                y=accuracy_counts,
                title="예측 정확도 분포",
                labels={'x': '오차율 범위', 'y': '고객 수'}
            )
            st.plotly_chart(fig_accuracy, use_container_width=True)
        
        # 특성 중요도 분석
        st.subheader("📈 특성 중요도 분석")
        
        feature_importance = pd.DataFrame({
            '특성명': training_results['feature_names'],
            '회귀계수': model.coef_,
            '절대계수': np.abs(model.coef_)
        }).sort_values('절대계수', ascending=False)
        
        # 상위 10개 특성만 표시
        top_features = feature_importance.head(10)
        
        # 특성 중요도 시각화
        try:
            importance_fig = RetailVisualizer.create_feature_importance_plot({
                'top_10_features': [
                    {
                        'feature': row['특성명'],
                        'coefficient': row['회귀계수'],
                        'abs_coefficient': row['절대계수']
                    }
                    for _, row in top_features.iterrows()
                ]
            })
            st.plotly_chart(importance_fig, use_container_width=True)
        except Exception as e:
            st.warning(f"특성 중요도 시각화 생성 중 오류: {str(e)}")
            
            # 대안 시각화
            colors = ['blue' if coef > 0 else 'red' for coef in top_features['회귀계수']]
            
            fig_importance = go.Figure()
            fig_importance.add_trace(
                go.Bar(
                    x=top_features['절대계수'],
                    y=top_features['특성명'],
                    orientation='h',
                    marker_color=colors,
                    text=[f'{coef:.3f}' for coef in top_features['회귀계수']],
                    textposition='auto'
                )
            )
            
            fig_importance.update_layout(
                title="상위 10개 특성 중요도",
                xaxis_title="계수 절댓값",
                yaxis_title="특성명",
                yaxis={'categoryorder': 'total ascending'},
                height=600
            )
            st.plotly_chart(fig_importance, use_container_width=True)
        
        # 특성 중요도 해석
        with st.expander("📖 특성 중요도 해석"):
            st.markdown("""
            **회귀계수 해석**
            - **양수**: 해당 특성이 증가하면 예측 구매 금액도 증가
            - **음수**: 해당 특성이 증가하면 예측 구매 금액은 감소
            - **절댓값이 클수록**: 예측에 미치는 영향이 큼
            
            **주의사항**
            - 정규화된 특성의 경우 원래 스케일과 다를 수 있음
            - 특성 간 상관관계가 높으면 계수 해석에 주의 필요
            """)
            
            # 상위 5개 특성의 영향 설명
            st.markdown("**상위 5개 영향력 있는 특성:**")
            for i, (_, row) in enumerate(top_features.head(5).iterrows(), 1):
                impact = "긍정적" if row['회귀계수'] > 0 else "부정적"
                st.write(f"{i}. **{row['특성명']}**: {impact} 영향 (계수: {row['회귀계수']:.3f})")
        
        # 특성 중요도 테이블
        with st.expander("📊 전체 특성 중요도 테이블"):
            st.dataframe(feature_importance, use_container_width=True)
        
        # 모델 방정식
        st.subheader("📐 학습된 모델 방정식")
        
        st.markdown("**선형회귀 방정식:**")
        
        # 상위 5개 특성으로 간소화된 방정식 표시
        top_5_features = feature_importance.head(5)
        equation_parts = [f"{model.intercept_:.3f}"]
        
        for _, row in top_5_features.iterrows():
            coef = row['회귀계수']
            feature = row['특성명']
            sign = "+" if coef > 0 else "-"
            equation_parts.append(f" {sign} {abs(coef):.3f} × {feature}")
        
        equation = "예측 구매 금액 = " + "".join(equation_parts) + " + ..."
        
        st.code(equation)
        
        st.info("💡 실제 방정식에는 모든 특성이 포함되며, 위는 상위 5개 특성만 표시한 간소화 버전입니다.")
        
        # 다음 단계 안내
        st.markdown("---")
        st.info("💡 모델 훈련이 완료되었습니다. 다음 단계인 '모델 평가 & 해석'으로 진행해주세요.")
    
    else:
        st.info("💡 모델링 설정을 완료하고 '선형회귀 모델 훈련' 버튼을 클릭하여 시작해주세요.")


def get_modeling_status():
    """모델링 상태 반환"""
    status = {
        'model_trained': st.session_state.get('retail_model_trained', False),
        'training_samples': 0,
        'test_samples': 0,
        'feature_count': 0,
        'r2_score': 0.0
    }
    
    if status['model_trained']:
        training_results = st.session_state.retail_training_results
        status.update({
            'training_samples': len(training_results['y_train']),
            'test_samples': len(training_results['y_test']),
            'feature_count': len(training_results['feature_names']),
        })
        
        # R² 점수 계산
        try:
            from sklearn.metrics import r2_score
            status['r2_score'] = r2_score(training_results['y_test'], training_results['y_test_pred'])
        except:
            pass
    
    return status
