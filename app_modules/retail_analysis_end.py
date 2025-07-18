        - 앙상블 모델 적용 (여러 모델 결합으로 정확도 향상)
        """
    
    # 최종 학습 완료 표시
    st.session_state.retail_model_metrics = {
        'test_r2': test_r2,
        'test_mae': test_mae,
        'performance_gap': performance_gap,
        'relative_error': relative_error
    }
    
    st.markdown("---")
    st.subheader("🎓 학습 여정 완료!")
    
    # 전체 학습 과정 요약
    completion_summary = f"""
    **🎉 축하합니다! Online Retail 분석 프로젝트를 완주하셨습니다!**
    
    **학습한 주요 기법들:**
    - ✅ 대용량 데이터 로딩 및 품질 분석
    - ✅ 체계적인 데이터 정제 및 전처리
    - ✅ 고급 특성 공학 및 RFM 분석  
    - ✅ 비즈니스 타겟 변수 설계
    - ✅ 선형회귀 모델 구축 및 평가
    - ✅ 잔차 분석을 통한 모델 진단
    
    **최종 모델 성능:**
    - R² Score: {test_r2:.3f}
    - 예측 오차: {relative_error:.1f}%
    - 과적합 여부: {'없음' if performance_gap <= 0.05 else '경미함' if performance_gap <= 0.1 else '있음'}
    
    **다음 단계 제안:**
    1. 🔄 수준2로 확장: 시간 기반 특성과 상품 카테고리 분석 추가
    2. 📊 다른 알고리즘 비교: 랜덤포레스트, XGBoost 등과 성능 비교
    3. 🎯 분류 문제 도전: 고객 이탈 예측, 세그먼트 분류 등
    4. 🧠 딥러닝 확장: 신경망을 활용한 더 복잡한 패턴 학습
    """
    
    st.success(completion_summary)
    
    # 프로젝트 수료증 스타일 표시
    st.balloons()
    
    with st.expander("🏆 프로젝트 수료 인증"):
        st.markdown(f"""
        <div style="
            border: 3px solid #4CAF50;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        ">
            <h2>🎓 수료 인증서</h2>
            <h3>Online Retail Customer Analysis</h3>
            <p><strong>"혼공머신" 연계 프로젝트</strong></p>
            <hr style="border-color: white;">
            <p>📊 데이터 분석: {len(st.session_state.retail_target_data):,}명 고객</p>
            <p>🏗️ 특성 공학: {len(st.session_state.retail_target_data.columns)}개 특성 생성</p> 
            <p>🤖 모델 성능: R² = {test_r2:.3f}</p>
            <p>📈 비즈니스 활용 가능도: {"높음" if relative_error <= 15 else "중간" if relative_error <= 25 else "개선 필요"}</p>
            <hr style="border-color: white;">
            <p><em>ADP 실기 및 실무 역량 강화 완료</em></p>
        </div>
        """, unsafe_allow_html=True)


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
