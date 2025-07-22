"""
Online Retail 타겟 변수 생성 페이지

타겟 변수 생성을 담당하는 Streamlit 페이지
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import warnings

warnings.filterwarnings("ignore")


def show_target_creation_page():
    """타겟 변수 생성 페이지"""
    
    st.header("4️⃣ 타겟 변수 생성")
    
    # 이전 단계 완료 확인
    if not st.session_state.get('retail_features_created', False):
        st.warning("⚠️ 먼저 3단계에서 특성 공학을 완료해주세요.")
        return
    
    st.markdown("""
    ### 📖 학습 목표
    - 비즈니스 문제를 머신러닝 문제로 정의하는 과정 체험
    - 회귀 문제의 타겟 변수 설계 방법론 학습
    - 실무에서 사용하는 타겟 변수 생성 기법 습득
    """)
    
    # 타겟 변수 설정
    if not st.session_state.get('retail_target_created', False):
        st.subheader("🎯 예측 목표 설정")
        
        st.markdown("""
        #### 비즈니스 문제 정의
        **목표**: 기존 고객의 과거 구매 데이터를 바탕으로 향후 일정 기간 동안의 예상 구매 금액을 예측하여 
        마케팅 전략 수립 및 고객 관리에 활용
        
        **예측 대상**: 고객별 향후 N개월간 예상 구매 금액
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            target_months = st.slider("예측 기간 (개월)", min_value=1, max_value=12, value=3)
        with col2:
            st.write(f"**목표**: 향후 {target_months}개월간 고객별 예상 구매 금액 예측")
        
        # 타겟 변수 생성 방법 설명
        with st.expander("🔧 타겟 변수 생성 방법론"):
            st.markdown("""
            **1. 월평균 구매 금액 계산**
            - 고객별 총 구매 금액을 고객 활동 기간으로 나누어 월평균 구매 금액 산출
            
            **2. 최근성 가중치 적용**
            - 최근 구매일자가 가까울수록 높은 가중치 적용
            - 공식: exp(-recency_days / 30)
            
            **3. 구매 빈도 가중치 적용**
            - 구매 빈도가 높을수록 높은 가중치 적용
            - 공식: log(1 + frequency) / log(1 + max_frequency)
            
            **4. 최종 예측값 계산**
            - 월평균 구매 금액 × 예측 기간 × 최근성 가중치 × 빈도 가중치
            
            **5. 현실적 범위 조정**
            - 과거 최대 구매 금액의 2배를 상한선으로 설정
            """)
        
        if st.button("🎯 타겟 변수 생성", type="primary"):
            with st.spinner("타겟 변수를 생성하는 중입니다..."):
                try:
                    # 세션 상태에서 고객 특성 가져오기
                    customer_features = st.session_state.retail_customer_features
                    engineer = st.session_state.retail_feature_engineer
                    
                    target_data = engineer.create_target_variable(customer_features, target_months=target_months)
                    
                    # 타겟 데이터 저장
                    st.session_state.retail_target_data = target_data.copy()
                    st.session_state.retail_target_months = target_months
                    st.session_state.retail_target_created = True
                    
                    st.success("✅ 타겟 변수 생성 완료!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ 타겟 변수 생성 실패: {str(e)}")
    
    # 타겟 변수 분석 결과 표시
    if st.session_state.get('retail_target_created', False):
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
        
        # 타겟 변수 미리보기
        st.subheader("🔍 타겟 변수 데이터 미리보기")
        
        # 중요 컬럼만 선택해서 표시
        preview_cols = [target_col, 'customer_value_category']
        if 'total_amount' in target_data.columns:
            preview_cols.append('total_amount')
        if 'frequency' in target_data.columns:
            preview_cols.append('frequency')
        if 'recency_days' in target_data.columns:
            preview_cols.append('recency_days')
        if 'monthly_avg_amount' in target_data.columns:
            preview_cols.append('monthly_avg_amount')
        
        available_cols = [col for col in preview_cols if col in target_data.columns]
        st.dataframe(target_data[available_cols].head(10), use_container_width=True)
        
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
            st.subheader("👥 고객 가치 등급 분포")
            
            category_counts = target_data['customer_value_category'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_pie = px.pie(
                    values=category_counts.values, 
                    names=category_counts.index,
                    title="고객 가치 등급 분포"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # 등급별 평균 예측 금액
                avg_by_category = target_data.groupby('customer_value_category')[target_col].mean().sort_values(ascending=False)
                
                fig_avg = px.bar(
                    x=avg_by_category.index,
                    y=avg_by_category.values,
                    title="등급별 평균 예측 금액",
                    labels={'x': '고객 등급', 'y': '평균 예측 금액 (£)'}
                )
                st.plotly_chart(fig_avg, use_container_width=True)
        
        # 예측 금액과 과거 구매 금액 비교
        if 'total_amount' in target_data.columns:
            st.subheader("📊 예측 금액 vs 과거 구매 금액 비교")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 산점도
                fig_scatter = px.scatter(
                    target_data,
                    x='total_amount',
                    y=target_col,
                    title="과거 구매 금액 vs 예측 금액",
                    labels={'total_amount': '과거 총 구매 금액 (£)', target_col: '예측 금액 (£)'},
                    opacity=0.6
                )
                
                # 동일선 추가
                min_val = min(target_data['total_amount'].min(), target_data[target_col].min())
                max_val = max(target_data['total_amount'].max(), target_data[target_col].max())
                fig_scatter.add_trace(
                    go.Scatter(
                        x=[min_val, max_val], 
                        y=[min_val, max_val],
                        mode='lines', 
                        name='동일선', 
                        line=dict(color='red', dash='dash')
                    )
                )
                
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            with col2:
                # 예측 배율 분포
                prediction_ratio = target_data[target_col] / target_data['total_amount']
                
                fig_ratio = px.histogram(
                    x=prediction_ratio,
                    title="예측 배율 분포 (예측금액/과거금액)",
                    labels={'x': '예측 배율'},
                    nbins=30
                )
                st.plotly_chart(fig_ratio, use_container_width=True)
        
        # RFM 특성과 예측 금액의 관계
        if all(col in target_data.columns for col in ['recency_days', 'frequency', 'monetary']):
            st.subheader("🔗 RFM 특성과 예측 금액의 관계")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fig_r = px.scatter(
                    target_data,
                    x='recency_days',
                    y=target_col,
                    title="최근성 vs 예측 금액",
                    labels={'recency_days': '최근성 (일)', target_col: '예측 금액 (£)'},
                    opacity=0.6
                )
                st.plotly_chart(fig_r, use_container_width=True)
            
            with col2:
                fig_f = px.scatter(
                    target_data,
                    x='frequency',
                    y=target_col,
                    title="빈도 vs 예측 금액",
                    labels={'frequency': '구매 빈도', target_col: '예측 금액 (£)'},
                    opacity=0.6
                )
                st.plotly_chart(fig_f, use_container_width=True)
            
            with col3:
                fig_m = px.scatter(
                    target_data,
                    x='monetary',
                    y=target_col,
                    title="구매 금액 vs 예측 금액",
                    labels={'monetary': '과거 구매 금액 (£)', target_col: '예측 금액 (£)'},
                    opacity=0.6
                )
                st.plotly_chart(fig_m, use_container_width=True)
        
        # 타겟 변수 상세 분석
        with st.expander("🔍 타겟 변수 상세 분석"):
            st.write("**분위수 분석:**")
            quantiles = target_data[target_col].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(2)
            quantile_df = pd.DataFrame({
                '분위수': ['10%', '25%', '50%', '75%', '90%'],
                '예측 금액 (£)': quantiles.values
            })
            st.dataframe(quantile_df, use_container_width=True)
            
            # 예측 금액 구간별 고객 수
            st.write("**예측 금액 구간별 고객 분포:**")
            bins = [0, 50, 100, 200, 500, 1000, float('inf')]
            labels = ['£0-50', '£50-100', '£100-200', '£200-500', '£500-1000', '£1000+']
            
            target_data['amount_range'] = pd.cut(target_data[target_col], bins=bins, labels=labels, right=False)
            range_counts = target_data['amount_range'].value_counts().sort_index()
            
            range_df = pd.DataFrame({
                '금액 구간': range_counts.index,
                '고객 수': range_counts.values,
                '비율 (%)': (range_counts.values / len(target_data) * 100).round(1)
            })
            st.dataframe(range_df, use_container_width=True)
        
        # 비즈니스 인사이트
        st.subheader("💼 비즈니스 인사이트")
        
        # 주요 통계 계산
        high_value_threshold = target_data[target_col].quantile(0.8)
        high_value_customers = len(target_data[target_data[target_col] >= high_value_threshold])
        low_value_threshold = target_data[target_col].quantile(0.2)
        low_value_customers = len(target_data[target_data[target_col] <= low_value_threshold])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📈 고가치 고객 (상위 20%)**")
            st.write(f"• 고객 수: {high_value_customers:,}명")
            st.write(f"• 예상 금액 하한: £{high_value_threshold:.2f}")
            st.write(f"• 전체 예상 수익의 비중: {(target_data[target_data[target_col] >= high_value_threshold][target_col].sum() / target_data[target_col].sum() * 100):.1f}%")
        
        with col2:
            st.markdown("**📉 저가치 고객 (하위 20%)**")
            st.write(f"• 고객 수: {low_value_customers:,}명")
            st.write(f"• 예상 금액 상한: £{low_value_threshold:.2f}")
            st.write(f"• 이탈 방지 프로그램 대상")
        
        # 마케팅 전략 제안
        st.markdown("#### 🎯 마케팅 전략 제안")
        
        strategy_suggestions = f"""
        **1. 고가치 고객 (상위 20%, {high_value_customers:,}명)**
        - VIP 프로그램 및 개인화된 서비스 제공
        - 프리미엄 상품 및 한정판 상품 우선 안내
        - 충성도 유지를 위한 특별 혜택
        
        **2. 중간 가치 고객 (중위 60%)**
        - 구매 빈도 증대를 위한 할인 쿠폰 제공
        - 교차 판매 및 상향 판매 기회 탐색
        - 계절별 프로모션 및 번들 상품 제안
        
        **3. 저가치 고객 (하위 20%, {low_value_customers:,}명)**
        - 이탈 방지를 위한 리텐션 캠페인
        - 저가 상품 라인업 및 할인 혜택 제공
        - 재구매 유도를 위한 이메일 마케팅
        
        **4. 예측 기간: {target_months}개월**
        - 분기별 성과 모니터링 및 전략 조정
        - 실제 구매 데이터와 예측의 정확도 검증
        """
        
        st.info(strategy_suggestions)
        
        # 다음 단계 안내
        st.markdown("---")
        st.info("💡 타겟 변수가 생성되었습니다. 다음 단계인 '선형회귀 모델링'으로 진행해주세요.")
    
    else:
        st.info("💡 예측 기간을 설정하고 '타겟 변수 생성' 버튼을 클릭하여 시작해주세요.")


def get_target_creation_status():
    """타겟 생성 상태 반환"""
    return {
        'target_created': st.session_state.get('retail_target_created', False),
        'target_months': st.session_state.get('retail_target_months', 0),
        'customer_count': len(st.session_state.retail_target_data) if st.session_state.get('retail_target_created', False) else 0,
        'avg_prediction': st.session_state.retail_target_data['predicted_next_amount'].mean() if st.session_state.get('retail_target_created', False) else 0
    }
