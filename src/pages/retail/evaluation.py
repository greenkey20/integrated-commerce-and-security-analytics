"""
Online Retail 모델 평가 페이지

모델 평가 및 해석을 담당하는 Streamlit 페이지
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from retail_analysis.model_trainer import RetailModelTrainer
from retail_analysis.visualizer import RetailVisualizer
from scipy import stats
import warnings

warnings.filterwarnings("ignore")


def show_evaluation_page():
    """모델 평가 및 해석 페이지"""
    
    st.header("6️⃣ 모델 평가 & 해석")
    
    # 이전 단계 완료 확인
    if not st.session_state.get('retail_model_trained', False):
        st.warning("⚠️ 먼저 5단계에서 모델을 훈련해주세요.")
        return
    
    st.markdown("""
    ### 📖 학습 목표
    - 모델 성능의 종합적 평가 방법 학습
    - 잔차 분석을 통한 모델 진단
    - 비즈니스 관점에서의 모델 해석 및 활용 방안 도출
    """)
    
    training_results = st.session_state.retail_training_results
    
    # 종합 성능 평가 실행
    if not st.session_state.get('retail_model_evaluated', False):
        if st.button("📊 종합 모델 평가 실행", type="primary"):
            with st.spinner("모델을 종합적으로 평가하는 중입니다..."):
                try:
                    trainer = st.session_state.retail_model_trainer
                    evaluation_results = trainer.evaluate_model()
                    
                    # 평가 결과 저장
                    st.session_state.retail_evaluation_results = evaluation_results
                    st.session_state.retail_model_evaluated = True
                    
                    st.success("✅ 모델 평가 완료!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ 모델 평가 실패: {str(e)}")
    
    # 평가 결과 표시
    if st.session_state.get('retail_model_evaluated', False):
        evaluation_results = st.session_state.retail_evaluation_results
        
        st.success("✅ 모델 평가가 완료되었습니다!")
        
        # 종합 성능 지표 테이블
        st.subheader("📊 종합 성능 평가")
        
        metrics_df = pd.DataFrame({
            '지표': ['R² Score', 'MAE (£)', 'RMSE (£)', '상대오차 (%)'],
            '훈련 성능': [
                f"{evaluation_results['r2_train']:.4f}",
                f"{evaluation_results['mae_train']:.2f}",
                f"{evaluation_results['rmse_train']:.2f}",
                f"{(evaluation_results['mae_train'] / training_results['y_train'].mean()) * 100:.2f}"
            ],
            '테스트 성능': [
                f"{evaluation_results['r2_test']:.4f}",
                f"{evaluation_results['mae_test']:.2f}",
                f"{evaluation_results['rmse_test']:.2f}",
                f"{evaluation_results['relative_error']:.2f}"
            ],
            '차이': [
                f"{evaluation_results['r2_test'] - evaluation_results['r2_train']:.4f}",
                f"{evaluation_results['mae_test'] - evaluation_results['mae_train']:.2f}",
                f"{evaluation_results['rmse_test'] - evaluation_results['rmse_train']:.2f}",
                f"{evaluation_results['relative_error'] - (evaluation_results['mae_train'] / training_results['y_train'].mean()) * 100:.2f}"
            ]
        })
        
        st.dataframe(metrics_df, use_container_width=True)
        
        # 성능 해석
        st.subheader("💡 성능 해석")
        
        test_r2 = evaluation_results['r2_test']
        performance_gap = evaluation_results['performance_gap']
        relative_error = evaluation_results['relative_error']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎯 모델 성능 평가:**")
            if test_r2 >= 0.8:
                st.success(f"🎉 **우수한 성능**: R² = {test_r2:.3f}")
                performance_level = "우수"
            elif test_r2 >= 0.6:
                st.info(f"👍 **양호한 성능**: R² = {test_r2:.3f}")
                performance_level = "양호"
            else:
                st.warning(f"⚠️ **개선 필요**: R² = {test_r2:.3f}")
                performance_level = "개선 필요"
        
        with col2:
            st.markdown("**🔍 과적합 분석:**")
            if performance_gap <= 0.05:
                st.success("✅ **과적합 없음**")
                overfitting_status = "없음"
            else:
                st.warning("⚠️ **과적합 발생**")
                overfitting_status = "있음"
        
        # 종합 모델 성능 시각화
        st.subheader("📈 종합 모델 성능 시각화")
        
        try:
            performance_fig = RetailVisualizer.create_model_performance_plots(evaluation_results)
            st.plotly_chart(performance_fig, use_container_width=True)
        except Exception as e:
            st.warning(f"성능 시각화 생성 중 오류: {str(e)}")
            
            # 대안 시각화
            col1, col2 = st.columns(2)
            
            with col1:
                # 예측 vs 실제값 산점도
                y_test = training_results['y_test']
                y_test_pred = training_results['y_test_pred']
                
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
                residuals = evaluation_results['residuals']
                fig_residuals = px.histogram(
                    x=residuals,
                    title="잔차 분포",
                    labels={'x': '잔차 (£)'}
                )
                st.plotly_chart(fig_residuals, use_container_width=True)
        
        # 잔차 분석
        st.subheader("🔍 잔차 분석")
        
        residuals = evaluation_results['residuals']
        y_test_pred = training_results['y_test_pred']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 잔차 vs 예측값
            fig_residuals_pred = px.scatter(
                x=y_test_pred, y=residuals,
                title="잔차 vs 예측값",
                labels={'x': '예측값 (£)', 'y': '잔차 (£)'}
            )
            # 기준선 추가
            fig_residuals_pred.add_trace(go.Scatter(
                x=[y_test_pred.min(), y_test_pred.max()], y=[0, 0],
                mode='lines', name='기준선', line=dict(color='red', dash='dash')
            ))
            st.plotly_chart(fig_residuals_pred, use_container_width=True)
        
        with col2:
            # Q-Q 플롯 (정규성 검정)
            fig_qq = go.Figure()
            
            # 샘플 크기 제한 (계산 효율성을 위해)
            sample_residuals = residuals[:min(5000, len(residuals))]
            qq_data = stats.probplot(sample_residuals, dist="norm")
            
            fig_qq.add_trace(go.Scatter(
                x=qq_data[0][0], y=qq_data[0][1],
                mode='markers', name='잔차'
            ))
            fig_qq.add_trace(go.Scatter(
                x=qq_data[0][0], y=qq_data[1][0] + qq_data[1][1] * qq_data[0][0],
                mode='lines', name='기준선', line=dict(color='red', dash='dash')
            ))
            fig_qq.update_layout(
                title="Q-Q 플롯 (정규성 검정)", 
                xaxis_title="이론적 분위수", 
                yaxis_title="표본 분위수"
            )
            st.plotly_chart(fig_qq, use_container_width=True)
        
        # 모델 가정 검정 결과
        st.subheader("🧪 모델 가정 검정")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # 정규성 검정
            normality_test = evaluation_results['normality_test']
            st.markdown("**정규성 검정 (Shapiro-Wilk)**")
            st.write(f"통계량: {normality_test['shapiro_stat']:.4f}")
            st.write(f"p-value: {normality_test['shapiro_p_value']:.6f}")
            
            if normality_test['is_normal']:
                st.success("✅ 잔차가 정규분포를 따름")
            else:
                st.warning("⚠️ 잔차가 정규분포를 따르지 않음")
        
        with col2:
            # 등분산성 검정
            hetero_test = evaluation_results['heteroscedasticity_test']
            st.markdown("**등분산성 검정**")
            st.write(f"상관계수: {hetero_test['correlation']:.4f}")
            
            if hetero_test['is_homoscedastic']:
                st.success("✅ 등분산성 가정 만족")
            else:
                st.warning("⚠️ 이분산성 존재")
        
        with col3:
            # 잔차 통계
            st.markdown("**잔차 통계**")
            st.write(f"평균: {evaluation_results['residuals_mean']:.4f}")
            st.write(f"표준편차: {evaluation_results['residuals_std']:.2f}")
            
            if abs(evaluation_results['residuals_mean']) < 0.1:
                st.success("✅ 잔차 평균이 0에 가까움")
            else:
                st.warning("⚠️ 잔차에 편향 존재")
        
        # 모델 가정 검정 해석
        with st.expander("📖 모델 가정 검정 해석"):
            st.markdown("""
            **선형회귀의 주요 가정들:**
            
            1. **선형성**: 독립변수와 종속변수 간의 선형 관계
            2. **정규성**: 잔차가 정규분포를 따름
            3. **등분산성**: 잔차의 분산이 일정함
            4. **독립성**: 잔차들이 서로 독립적임
            
            **가정 위반 시 대처 방안:**
            - **정규성 위반**: 변수 변환, 비모수적 방법 고려
            - **등분산성 위반**: 가중회귀, 로버스트 표준오차 사용
            - **선형성 위반**: 다항 회귀, 비선형 모델 고려
            """)
        
        # 비즈니스 해석
        st.subheader("💼 비즈니스 관점 해석")
        
        # 모델 해석 정보
        interpretation = evaluation_results.get('interpretation', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎯 예측 정확도:**")
            if relative_error <= 15:
                st.success("🎯 **고정밀도 예측 가능**")
                st.write("개인화된 마케팅 전략 수립 가능")
                accuracy_level = "고정밀도"
            elif relative_error <= 25:
                st.info("👍 **세그먼트별 전략 수립**")
                st.write("고객군별 차별화 전략 권장")
                accuracy_level = "중간 정밀도"
            else:
                st.warning("⚠️ **전반적 트렌드 파악**")
                st.write("추가 데이터 수집 및 모델 개선 필요")
                accuracy_level = "낮은 정밀도"
        
        with col2:
            st.markdown("**📈 활용 방안:**")
            st.write("• 고객별 예상 구매 금액 예측")
            st.write("• 마케팅 예산 배분 최적화")
            st.write("• 고객 가치 기반 세분화")
            st.write("• 이탈 위험 고객 식별")
        
        # 특성 중요도 해석
        st.subheader("🎯 특성 중요도 비즈니스 해석")
        
        feature_importance = evaluation_results['feature_importance']
        
        if 'error' not in feature_importance:
            # 상위 영향 특성들
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📈 긍정적 영향 특성 (Top 5)**")
                positive_features = feature_importance['positive_impact_features'][:5]
                for i, feature in enumerate(positive_features, 1):
                    st.write(f"{i}. **{feature['feature']}**: +{feature['coefficient']:.3f}")
                    st.caption(get_feature_business_meaning(feature['feature'], 'positive'))
            
            with col2:
                st.markdown("**📉 부정적 영향 특성 (Top 5)**")
                negative_features = feature_importance['negative_impact_features'][:5]
                for i, feature in enumerate(negative_features, 1):
                    st.write(f"{i}. **{feature['feature']}**: {feature['coefficient']:.3f}")
                    st.caption(get_feature_business_meaning(feature['feature'], 'negative'))
        
        # 모델 개선 제안
        st.subheader("🚀 모델 개선 제안")
        
        improvement_suggestions = interpretation.get('improvement_suggestions', [])
        
        if improvement_suggestions:
            st.markdown("**현재 모델의 개선 포인트:**")
            for suggestion in improvement_suggestions:
                st.warning(f"• {suggestion}")
        
        # 추가 개선 제안
        additional_suggestions = []
        
        if test_r2 < 0.6:
            additional_suggestions.append("비선형 모델 (랜덤포레스트, XGBoost) 고려")
        
        if relative_error > 20:
            additional_suggestions.append("추가 특성 공학 및 외부 데이터 활용")
        
        if overfitting_status == "있음":
            additional_suggestions.append("정규화 기법 (Ridge, Lasso) 적용")
        
        if additional_suggestions:
            st.markdown("**추가 개선 제안:**")
            for suggestion in additional_suggestions:
                st.info(f"• {suggestion}")
        
        # 비즈니스 인사이트 대시보드
        st.subheader("💼 비즈니스 인사이트 대시보드")
        
        try:
            target_data = st.session_state.retail_target_data
            insights_fig = RetailVisualizer.create_business_insights_dashboard(target_data, evaluation_results)
            st.plotly_chart(insights_fig, use_container_width=True)
        except Exception as e:
            st.warning(f"인사이트 대시보드 생성 중 오류: {str(e)}")
        
        # 학습 완료 축하
        st.markdown("---")
        st.subheader("🎓 학습 여정 완료!")
        
        target_months = st.session_state.retail_target_months
        completion_summary = f"""
        **🎉 축하합니다! Online Retail 분석 프로젝트를 완주하셨습니다!**
        
        **📊 최종 모델 성능:**
        - R² Score: {test_r2:.3f} ({performance_level})
        - 예측 오차: {relative_error:.1f}% ({accuracy_level})
        - 과적합 여부: {overfitting_status}
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
            trainer = st.session_state.retail_model_trainer
            model_summary = trainer.get_model_summary()
            
            project_summary = f"""
# Online Retail 분석 프로젝트 요약

## 📊 데이터 개요
- 원본 데이터: {len(st.session_state.retail_raw_data):,}개 레코드
- 정제 후 데이터: {len(st.session_state.retail_cleaned_data):,}개 레코드
- 분석 대상 고객: {len(st.session_state.retail_customer_features):,}명

## 🎯 모델 성능
- R² Score: {test_r2:.3f}
- MAE: {evaluation_results['mae_test']:.2f}£
- RMSE: {evaluation_results['rmse_test']:.2f}£
- 상대 오차: {relative_error:.1f}%

## 🔧 모델 설정
- 테스트 비율: {training_results['test_size']:.1%}
- 정규화: {'적용' if training_results['scale_features'] else '미적용'}
- 랜덤 시드: {training_results['random_state']}

## 💡 주요 특성 (상위 5개)
{chr(10).join([f"- {f['feature']}: {f['coefficient']:.3f}" for f in feature_importance['top_10_features'][:5]])}

## 📈 비즈니스 해석
- 예측 정확도: {accuracy_level}
- 과적합 여부: {overfitting_status}
- 활용 가능성: {'높음' if test_r2 >= 0.6 else '보통'}

## 🚀 개선 제안
{chr(10).join([f"- {s}" for s in improvement_suggestions + additional_suggestions])}
"""
            st.text_area("프로젝트 요약", project_summary, height=500)
    
    else:
        st.info("💡 '종합 모델 평가 실행' 버튼을 클릭하여 시작해주세요.")


def get_feature_business_meaning(feature_name, impact_type):
    """특성의 비즈니스 의미 반환"""
    
    meanings = {
        'total_amount': {
            'positive': '과거 구매 금액이 높을수록 미래 구매 가능성 증가',
            'negative': '과거 구매 금액 대비 미래 구매 감소 예상'
        },
        'frequency': {
            'positive': '구매 빈도가 높을수록 지속적 구매 기대',
            'negative': '높은 구매 빈도가 오히려 감소 요인으로 작용'
        },
        'recency_days': {
            'positive': '오래된 고객일수록 재구매 가능성 증가',
            'negative': '최근 구매 고객일수록 재구매 가능성 높음'
        },
        'monetary': {
            'positive': '높은 구매력을 가진 고객의 지속적 구매',
            'negative': '과도한 과거 구매로 인한 구매력 소진'
        },
        'unique_products': {
            'positive': '다양한 상품 구매 고객의 지속적 관심',
            'negative': '너무 많은 상품 구매로 인한 포화 상태'
        }
    }
    
    return meanings.get(feature_name, {}).get(impact_type, '비즈니스 영향 분석 필요')


def get_evaluation_status():
    """평가 상태 반환"""
    status = {
        'model_evaluated': st.session_state.get('retail_model_evaluated', False),
        'r2_score': 0.0,
        'relative_error': 0.0,
        'overfitting_risk': 'Unknown',
        'business_applicability': 'Unknown'
    }
    
    if status['model_evaluated']:
        evaluation_results = st.session_state.retail_evaluation_results
        status.update({
            'r2_score': evaluation_results['r2_test'],
            'relative_error': evaluation_results['relative_error'],
            'overfitting_risk': 'Low' if evaluation_results['performance_gap'] <= 0.05 else 'High',
            'business_applicability': evaluation_results.get('interpretation', {}).get('business_applicability', 'Unknown')
        })
    
    return status
