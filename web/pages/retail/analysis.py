"""
Online Retail 분석 메인 페이지 - 리팩토링 버전

"혼자 공부하는 머신러닝, 딥러닝" 교재와 연계하여
실무급 데이터 전처리와 선형회귀 모델링을 경험할 수 있는 페이지입니다.

전체 워크플로우가 모듈화되어 유지보수성이 크게 향상되었습니다.
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings


def safe_rerun():
    """Streamlit 버전에 관계없이 안전한 rerun 실행"""
    try:
        if hasattr(st, 'rerun'):
            st.rerun()
        elif hasattr(st, 'experimental_rerun'):
            st.experimental_rerun()
        else:
            # fallback: 페이지 새로고침 메시지
            st.info("페이지를 새로고침해주세요.")
    except Exception as e:
        st.warning(f"페이지 새로고침이 필요합니다: {str(e)}")
        st.info("브라우저에서 F5키를 눌러 새로고침해주세요.")

# 리팩토링된 페이지 모듈들 import
from web.pages.retail.data_loading import show_data_loading_page, get_data_loading_status
from web.pages.retail.data_cleaning import show_data_cleaning_page, get_data_cleaning_status
from web.pages.retail.feature_engineering import show_feature_engineering_page, get_feature_engineering_status
from web.pages.retail.target_creation import show_target_creation_page, get_target_creation_status
from web.pages.retail.modeling import show_modeling_page, get_modeling_status
from web.pages.retail.evaluation import show_evaluation_page, get_evaluation_status

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
    
    **🔧 리팩토링 개선사항:**
    - 모듈화된 코드 구조로 유지보수성 향상
    - 각 단계별 독립적인 페이지로 분리
    - 재사용 가능한 컴포넌트 설계
    """)
    
    # 세션 상태 초기화
    initialize_session_state()
    
    # 사이드바에서 분석 단계 선택
    setup_sidebar()
    
    # 선택된 단계에 따른 페이지 렌더링
    analysis_step = st.session_state.get('analysis_step', "1️⃣ 데이터 로딩 & 품질 분석")
    
    if analysis_step == "1️⃣ 데이터 로딩 & 품질 분석":
        show_data_loading_page()
    elif analysis_step == "2️⃣ 데이터 정제 & 전처리":
        show_data_cleaning_page()
    elif analysis_step == "3️⃣ 특성 공학 & 파생변수":
        show_feature_engineering_page()
    elif analysis_step == "4️⃣ 타겟 변수 생성":
        show_target_creation_page()
    elif analysis_step == "5️⃣ 선형회귀 모델링":
        show_modeling_page()
    elif analysis_step == "6️⃣ 모델 평가 & 해석":
        show_evaluation_page()
    elif analysis_step == "📊 전체 분석 요약":
        show_analysis_summary_page()


def initialize_session_state():
    """세션 상태 초기화 - 모든 단계 상태 관리"""
    
    # 기본 상태 초기화
    default_states = {
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
            "6️⃣ 모델 평가 & 해석",
            "📊 전체 분석 요약"
        ]
    )
    
    # 선택된 단계 저장
    st.session_state.analysis_step = analysis_step
    
    # 진행 상태 표시
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🎯 학습 진도:**")
    
    # 각 단계별 상태 확인
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
    
    # 전체 진행률 표시
    completed_steps = sum([
        st.session_state.retail_data_loaded,
        st.session_state.retail_data_cleaned,
        st.session_state.retail_features_created,
        st.session_state.retail_target_created,
        st.session_state.retail_model_trained,
        st.session_state.retail_model_evaluated
    ])
    
    progress_percentage = (completed_steps / 6) * 100
    st.sidebar.progress(progress_percentage / 100)
    st.sidebar.caption(f"전체 진행률: {progress_percentage:.0f}%")
    
    # 각 단계별 상세 상태
    with st.sidebar.expander("📊 단계별 상세 상태"):
        
        # 데이터 로딩 상태
        loading_status = get_data_loading_status()
        st.write(f"**데이터 로딩**: {'✅' if loading_status['data_loaded'] else '⏳'}")
        if loading_status['data_loaded']:
            st.caption(f"레코드 수: {loading_status['records_count']:,}개")
        
        # 데이터 정제 상태
        cleaning_status = get_data_cleaning_status()
        st.write(f"**데이터 정제**: {'✅' if cleaning_status['data_cleaned'] else '⏳'}")
        if cleaning_status['data_cleaned']:
            st.caption(f"품질 점수: {cleaning_status['quality_score']}/100")
        
        # 특성 공학 상태
        feature_status = get_feature_engineering_status()
        st.write(f"**특성 공학**: {'✅' if feature_status['features_created'] else '⏳'}")
        if feature_status['features_created']:
            st.caption(f"고객 수: {feature_status['customer_count']:,}명")
            st.caption(f"특성 수: {feature_status['feature_count']}개")
        
        # 타겟 생성 상태
        target_status = get_target_creation_status()
        st.write(f"**타겟 생성**: {'✅' if target_status['target_created'] else '⏳'}")
        if target_status['target_created']:
            st.caption(f"예측 기간: {target_status['target_months']}개월")
            st.caption(f"평균 예측: £{target_status['avg_prediction']:.2f}")
        
        # 모델링 상태
        modeling_status = get_modeling_status()
        st.write(f"**모델링**: {'✅' if modeling_status['model_trained'] else '⏳'}")
        if modeling_status['model_trained']:
            st.caption(f"R² 점수: {modeling_status['r2_score']:.3f}")
        
        # 평가 상태
        evaluation_status = get_evaluation_status()
        st.write(f"**모델 평가**: {'✅' if evaluation_status['model_evaluated'] else '⏳'}")
        if evaluation_status['model_evaluated']:
            st.caption(f"상대오차: {evaluation_status['relative_error']:.1f}%")
    
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
        
        **🔧 리팩토링 효과:**
        - 각 페이지가 독립적으로 작동
        - 코드 유지보수성 대폭 향상
        - 개별 기능 테스트 및 디버깅 용이
        """)
    
    # 빠른 액션
    st.sidebar.markdown("---")
    st.sidebar.markdown("**⚡ 빠른 액션**")
    
    if st.sidebar.button("🔄 전체 초기화"):
        # 모든 상태 초기화
        keys_to_clear = [key for key in st.session_state.keys() if key.startswith('retail_')]
        for key in keys_to_clear:
            del st.session_state[key]
        st.sidebar.success("초기화 완료!")
        safe_rerun()
    
    if completed_steps == 6:
        if st.sidebar.button("📊 분석 요약 보기"):
            st.session_state.analysis_step = "📊 전체 분석 요약"
            safe_rerun()


def show_analysis_summary_page():
    """전체 분석 요약 페이지"""
    
    st.header("📊 전체 분석 요약")
    
    # 모든 단계 완료 확인
    if not all([
        st.session_state.retail_data_loaded,
        st.session_state.retail_data_cleaned,
        st.session_state.retail_features_created,
        st.session_state.retail_target_created,
        st.session_state.retail_model_trained,
        st.session_state.retail_model_evaluated
    ]):
        st.warning("⚠️ 모든 분석 단계를 완료한 후에 요약을 볼 수 있습니다.")
        return
    
    st.success("🎉 모든 분석 단계가 완료되었습니다!")
    
    # 전체 프로젝트 메트릭
    st.subheader("📈 프로젝트 전체 메트릭")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        raw_records = len(st.session_state.retail_raw_data)
        st.metric("원본 레코드", f"{raw_records:,}")
    
    with col2:
        cleaned_records = len(st.session_state.retail_cleaned_data)
        st.metric("정제 후 레코드", f"{cleaned_records:,}")
    
    with col3:
        customers = len(st.session_state.retail_customer_features)
        st.metric("분석 고객 수", f"{customers:,}")
    
    with col4:
        features = len(st.session_state.retail_customer_features.columns)
        st.metric("생성된 특성", f"{features}")
    
    # 모델 성능 요약
    st.subheader("🎯 최종 모델 성능")
    
    if 'retail_evaluation_results' in st.session_state:
        eval_results = st.session_state.retail_evaluation_results
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("R² Score", f"{eval_results['r2_test']:.3f}")
        
        with col2:
            st.metric("MAE", f"£{eval_results['mae_test']:.2f}")
        
        with col3:
            st.metric("RMSE", f"£{eval_results['rmse_test']:.2f}")
        
        with col4:
            st.metric("상대오차", f"{eval_results['relative_error']:.1f}%")
        
        # 성능 해석
        r2_score = eval_results['r2_test']
        if r2_score >= 0.8:
            st.success("🎉 우수한 모델 성능을 달성했습니다!")
        elif r2_score >= 0.6:
            st.info("👍 양호한 모델 성능입니다.")
        else:
            st.warning("⚠️ 모델 성능 개선이 필요합니다.")
    
    # 단계별 소요 시간 (가상)
    st.subheader("⏱️ 분석 단계별 현황")
    
    stages_df = pd.DataFrame({
        '단계': [
            '데이터 로딩',
            '데이터 정제', 
            '특성 공학',
            '타겟 생성',
            '모델 훈련',
            '모델 평가'
        ],
        '상태': ['완료' for _ in range(6)],
        '주요 산출물': [
            f"{raw_records:,}개 레코드",
            f"{cleaned_records:,}개 레코드 (품질 점수: {st.session_state.retail_validation_report.get('data_quality_score', 0)}/100)",
            f"{customers:,}명 고객, {features}개 특성",
            f"{st.session_state.retail_target_months}개월 예측 타겟",
            f"R² = {eval_results['r2_test']:.3f}" if 'retail_evaluation_results' in st.session_state else "완료",
            f"상대오차 {eval_results['relative_error']:.1f}%" if 'retail_evaluation_results' in st.session_state else "완료"
        ]
    })
    
    st.dataframe(stages_df, use_container_width=True)
    
    # 비즈니스 인사이트
    st.subheader("💼 주요 비즈니스 인사이트")
    
    target_data = st.session_state.retail_target_data
    avg_prediction = target_data['predicted_next_amount'].mean()
    high_value_customers = len(target_data[target_data['predicted_next_amount'] >= target_data['predicted_next_amount'].quantile(0.8)])
    
    insights = [
        f"🎯 평균 고객 예측 구매 금액: £{avg_prediction:.2f}",
        f"👑 고가치 고객 (상위 20%): {high_value_customers:,}명",
        f"📈 데이터 보존율: {(cleaned_records/raw_records*100):.1f}%",
        f"🔧 특성 공학 효과: {features}개 의미있는 특성 생성"
    ]
    
    for insight in insights:
        st.info(insight)
    
    # 학습 성과
    st.subheader("🎓 학습 성과 및 다음 단계")
    
    achievements = """
    **🏆 달성한 학습 목표:**
    - ✅ 실무급 데이터 전처리 경험
    - ✅ 체계적인 특성 공학 과정 습득  
    - ✅ 머신러닝 모델링 전체 파이프라인 이해
    - ✅ 비즈니스 관점에서의 모델 해석 능력 향상
    - ✅ 모듈화된 코드 구조 설계 경험
    
    **🚀 추천 다음 단계:**
    1. **고급 모델 실험**: RandomForest, XGBoost 등으로 성능 비교
    2. **특성 엔지니어링 확장**: 시간 기반 특성, 상품 카테고리 분석
    3. **분류 문제 도전**: 고객 이탈 예측, 세그먼트 분류
    4. **실시간 파이프라인**: 모델 배포 및 실시간 예측 시스템 구축
    5. **A/B 테스트**: 모델 기반 마케팅 전략의 실제 효과 검증
    """
    
    st.success(achievements)
    
    # 프로젝트 파일 구조
    with st.expander("📁 리팩토링된 프로젝트 구조"):
        st.code("""
📦 customer-segmentation/
├── core/
│   ├── retail_data_loader.py      # 데이터 로딩 & 품질 분석
│   ├── retail_data_processor.py   # 데이터 정제 & 전처리
│   ├── retail_feature_engineer.py # 특성 공학 & 파생변수
│   ├── retail_model_trainer.py    # 모델 훈련 & 평가
│   ├── retail_visualizer.py       # 시각화 전담
│   └── retail_analysis.py         # 통합 관리자
├── src/pages/
│   ├── retail_data_loading.py     # 데이터 로딩 페이지
│   ├── retail_data_cleaning.py    # 데이터 정제 페이지
│   ├── retail_feature_engineering.py # 특성 공학 페이지
│   ├── retail_target_creation.py  # 타겟 생성 페이지
│   ├── retail_modeling.py         # 모델링 페이지
│   ├── retail_evaluation.py       # 모델 평가 페이지
│   └── retail_analysis.py         # 메인 라우터 페이지
└── backup/
    ├── retail_analysis_backup.py  # 기존 core 백업
    └── retail_analysis_backup.py  # 기존 pages 백업

🎯 리팩토링 효과:
- 단일 파일 44KB → 6개 모듈로 분산
- 기능별 독립적 개발 및 테스트 가능
- 코드 재사용성 및 유지보수성 향상
- 명확한 책임 분리로 협업 효율성 증대
        """)


# 기존 코드와의 호환성 유지
def show_retail_analysis_page_legacy():
    """기존 함수명과의 호환성을 위한 별칭"""
    return show_retail_analysis_page()
