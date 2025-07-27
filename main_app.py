"""
안정적인 단순 버전 메인 애플리케이션

문제 해결:
1. Tensorflow 경고 억제  
2. 무한루프 방지 (rerun 제거)
3. 단일 selectbox로 단순화
4. 세션 상태 최소화
"""

import os
import sys
import warnings

# Tensorflow 경고 완전 억제
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Python warnings 억제
warnings.filterwarnings("ignore")

# 잠시 stderr 차단 (Tensorflow import 시)
import io
import contextlib

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# Tensorflow import 시 output 억제
with suppress_output():
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')

import streamlit as st

# Python 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, "web"))

# 설정 및 유틸리티 모듈
from config.settings import AppConfig, UIConfig

# 한글 폰트 설정 (간단 버전)
def setup_simple_korean_font():
    """간단한 한글 폰트 설정"""
    try:
        import matplotlib.pyplot as plt
        
        # Windows 환경에서 안정적인 폰트들
        korean_fonts = ['Malgun Gothic', 'Gulim', 'Dotum', 'Arial Unicode MS']
        
        for font in korean_fonts:
            try:
                plt.rcParams["font.family"] = font
                plt.rcParams["axes.unicode_minus"] = False
                break
            except:
                continue
                
    except Exception:
        pass  # 폰트 설정 실패해도 무시

# 페이지 모듈들 (안전한 import)
def safe_import_pages():
    """안전한 페이지 import (실패해도 앱 중단 안됨)"""
    pages = {}
    
    try:
        from web.pages.segmentation.data_overview import show_data_overview_page
        pages['data_overview'] = show_data_overview_page
    except:
        pages['data_overview'] = None
        
    try:
        from web.pages.segmentation.exploratory_analysis import show_exploratory_analysis_page
        pages['exploratory_analysis'] = show_exploratory_analysis_page
    except:
        pages['exploratory_analysis'] = None
        
    try:
        from web.pages.segmentation.clustering_analysis import show_clustering_analysis_page
        pages['clustering_analysis'] = show_clustering_analysis_page
    except:
        pages['clustering_analysis'] = None
        
    try:
        from web.pages.segmentation.pca_analysis import show_pca_analysis_page
        pages['pca_analysis'] = show_pca_analysis_page
    except:
        pages['pca_analysis'] = None
        
    try:
        from web.pages.segmentation.deep_learning_analysis import show_deep_learning_analysis_page
        pages['deep_learning_analysis'] = show_deep_learning_analysis_page
    except:
        pages['deep_learning_analysis'] = None
        
    try:
        from web.pages.segmentation.customer_prediction import show_customer_prediction_page
        pages['customer_prediction'] = show_customer_prediction_page
    except:
        pages['customer_prediction'] = None
        
    try:
        from web.pages.segmentation.marketing_strategy import show_marketing_strategy_page
        pages['marketing_strategy'] = show_marketing_strategy_page
    except:
        pages['marketing_strategy'] = None

    try:
        from web.pages.retail.analysis import show_retail_analysis_page
        pages['retail_analysis'] = show_retail_analysis_page
    except:
        pages['retail_analysis'] = None

    try:
        from web.pages.retail.data_loading import show_data_loading_page
        pages['retail_data_loading'] = show_data_loading_page
    except:
        pages['retail_data_loading'] = None

    try:
        from web.pages.retail.data_cleaning import show_data_cleaning_page
        pages['retail_data_cleaning'] = show_data_cleaning_page
    except:
        pages['retail_data_cleaning'] = None

    try:
        from web.pages.retail.feature_engineering import show_feature_engineering_page
        pages['retail_feature_engineering'] = show_feature_engineering_page
    except:
        pages['retail_feature_engineering'] = None

    try:
        from web.pages.retail.target_creation import show_target_creation_page
        pages['retail_target_creation'] = show_target_creation_page
    except:
        pages['retail_target_creation'] = None

    try:
        from web.pages.retail.modeling import show_modeling_page
        pages['retail_modeling'] = show_modeling_page
    except:
        pages['retail_modeling'] = None

    try:
        from web.pages.retail.evaluation import show_evaluation_page
        pages['retail_evaluation'] = show_evaluation_page
    except:
        pages['retail_evaluation'] = None
        
    # 보안 페이지는 선택적 로딩
    try:
        from web.pages.security.security_analysis_page import show_security_analysis_page
        pages['security_analysis'] = show_security_analysis_page
    except:
        pages['security_analysis'] = None
        
    return pages

def initialize_app():
    """애플리케이션 초기 설정"""
    st.set_page_config(
        page_title="🌿 Integrated Commerce & Security Analytics",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # 한글 폰트 설정
    setup_simple_korean_font()
    
    # Green Theme CSS (Light Mode 전용)
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(180deg, #F8FAFC 0%, #F0FDFA 100%) !important;
        color: #064E3B !important;
    }
    .css-1d391kg {
        background: linear-gradient(180deg, #F0FDF4 0%, #ECFDF5 100%) !important;
    }
    .main .block-container {
        background: rgba(255, 255, 255, 0.95) !important;
        color: #064E3B !important;
        border: 1px solid rgba(34, 197, 94, 0.2) !important;
        border-radius: 16px !important;
        padding: 2rem !important;
        margin-top: 1rem !important;
    }
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #FFFFFF, #F0FDF4) !important;
        border: 1px solid #BBF7D0 !important;
        color: #064E3B !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }
    .stSuccess { 
        background: #F0FDF4 !important; 
        border: 1px solid #BBF7D0 !important; 
        color: #064E3B !important; 
    }
    .stWarning { 
        background: #FFFBEB !important; 
        border: 1px solid #FDE68A !important; 
        color: #92400E !important; 
    }
    .stError { 
        background: #FEF2F2 !important; 
        border: 1px solid #FECACA !important; 
        color: #991B1B !important; 
    }
    .stInfo { 
        background: #F0F9FF !important; 
        border: 1px solid #BAE6FD !important; 
        color: #0C4A6E !important; 
    }
    
    /* 모든 텍스트 요소 색상 강제 지정 */
    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6 {
        color: #064E3B !important;
        font-weight: 600 !important;
    }
    .main p, .main div, .main span, .main label, .main li {
        color: #374151 !important;
    }
    .main a { color: #059669 !important; }
    .main a:hover { color: #047857 !important; }
    
    /* 사이드바 스타일링 */
    .css-1d391kg .stSelectbox > div > div {
        background: #FFFFFF !important;
        border: 1px solid #D1FAE5 !important;
    }
    
    /* Plotly 차트 배경 */
    .js-plotly-plot, .plotly {
        background: rgba(255, 255, 255, 0.95) !important;
        border-radius: 8px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 제목 및 소개
    st.title("🌿 Integrated Commerce & Security Analytics")
    st.markdown("""
    **차세대 이커머스를 위한 통합 인텔리전스 플랫폼**
    
    고객 인사이트부터 보안 모니터링까지, 데이터 기반 비즈니스 성장을 지원합니다.
    
    **버전**: v3.0 - 통합 분석 플랫폼 (Simple Edition)
    """)

def setup_simple_sidebar():
    """계층형 네비게이션 (Business Intelligence + Security Analytics)"""
    st.sidebar.title("📋 Navigation")
    
    # 세션 상태 초기화
    if 'current_focus' not in st.session_state:
        st.session_state.current_focus = 'retail'
    
    # A. Business Intelligence 섹션
    st.sidebar.markdown("### 📊 **A. Business Intelligence**")
    
    # 1. Retail Prediction
    st.sidebar.markdown("#### 💰 **1. Retail Prediction**")
    retail_step = st.sidebar.selectbox(
        "단계 선택:",
        [
            "1️⃣ 데이터 로딩 및 개요",
            "2️⃣ 데이터 정제 & 전처리", 
            "3️⃣ 특성공학 & 파생변수 생성",
            "4️⃣ 타겟변수 생성",
            "5️⃣ 선형회귀 모델링",
            "6️⃣ 모델 평가 & 해석",
            "📊 전체 분석 요약"
        ],
        key="retail_step_select",
        on_change=lambda: setattr(st.session_state, 'current_focus', 'retail')
    )
    
    # 2. Customer Segmentation  
    st.sidebar.markdown("#### 👥 **2. Customer Segmentation**")
    customer_step = st.sidebar.selectbox(
        "단계 선택:",
        [
            "1️⃣ 데이터 로딩 및 개요",
            "2️⃣ 탐색적 데이터 분석",
            "3️⃣ 클러스터링 분석", 
            "4️⃣ 주성분 분석",
            "5️⃣ 딥러닝 분석",
            "6️⃣ customer segmentation 예측",
            "7️⃣ 마케팅 전략"
        ],
        key="customer_step_select",
        on_change=lambda: setattr(st.session_state, 'current_focus', 'customer')
    )
    
    # B. Security Analytics 섹션
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🛡️ **B. Security Analytics**")
    
    # 1. 네트워크 보안 이상 탐지 분석
    st.sidebar.markdown("#### 🔒 **1. 네트워크 보안 이상 탐지 분석**")
    security_step = st.sidebar.selectbox(
        "단계 선택:",
        [
            "1️⃣ 데이터 로딩 및 개요",
            "2️⃣ 탐색적 데이터 분석",
            "3️⃣ 공격 패턴 심화 분석",
            "4️⃣ 딥러닝 모델링",
            "5️⃣ Overfitting 해결 검증",
            "6️⃣ 실시간 예측 테스트",
            "7️⃣ 종합 성능 평가"
        ],
        key="security_step_select",
        on_change=lambda: setattr(st.session_state, 'current_focus', 'security')
    )
    
    # 현재 포커스 표시
    focus_emoji = {'retail': '💰', 'customer': '👥', 'security': '🔒'}
    st.sidebar.markdown(f"**현재 포커스**: {focus_emoji.get(st.session_state.current_focus, '💰')} {st.session_state.current_focus.title()}")
    
    # 빠른 액션
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 새로고침", key="refresh"):
        st.rerun()
    
    return retail_step, customer_step, security_step, st.session_state.current_focus

def route_to_hierarchical_page(retail_step, customer_step, security_step, current_focus, pages):
    """계층형 네비게이션 라우팅 (포커스 기반)"""
    
    try:
        # 현재 포커스된 섹션만 표시
        focus_info = {
            'retail': f"💰 Retail: {retail_step}",
            'customer': f"👥 Customer: {customer_step}", 
            'security': f"🔒 Security: {security_step}"
        }
        st.info(f"📍 **현재 포커스**: {focus_info[current_focus]}")
        
        # 포커스된 섹션에 따라 라우팅
        if current_focus == 'retail':
            # 1. Retail Prediction 라우팅
            if "전체 분석 요약" in retail_step:
                if pages['retail_analysis']:
                    pages['retail_analysis']()
                else:
                    show_fallback_page("💰 Retail 전체 분석", "Online Retail 데이터 분석 페이지")
            elif "1️⃣ 데이터 로딩" in retail_step:
                if pages['retail_data_loading']:
                    pages['retail_data_loading']()
                else:
                    show_fallback_page("📋 Retail 데이터 로딩", "web/pages/retail/data_loading.py")
            elif "2️⃣ 데이터 정제" in retail_step:
                if pages['retail_data_cleaning']:
                    pages['retail_data_cleaning']()
                else:
                    show_fallback_page("🧹 Retail 데이터 정제", "web/pages/retail/data_cleaning.py")
            elif "3️⃣ 특성공학" in retail_step:
                if pages['retail_feature_engineering']:
                    pages['retail_feature_engineering']()
                else:
                    show_fallback_page("⚙️ Retail 특성공학", "web/pages/retail/feature_engineering.py")
            elif "4️⃣ 타겟변수" in retail_step:
                if pages['retail_target_creation']:
                    pages['retail_target_creation']()
                else:
                    show_fallback_page("🎯 Retail 타겟변수", "web/pages/retail/target_creation.py")
            elif "5️⃣ 선형회귀" in retail_step:
                if pages['retail_modeling']:
                    pages['retail_modeling']()
                else:
                    show_fallback_page("🤖 Retail 모델링", "web/pages/retail/modeling.py")
            elif "6️⃣ 모델 평가" in retail_step:
                if pages['retail_evaluation']:
                    pages['retail_evaluation']()
                else:
                    show_fallback_page("📊 Retail 평가", "web/pages/retail/evaluation.py")

        elif current_focus == 'customer':
            # 2. Customer Segmentation 라우팅
            if "1️⃣ 데이터 로딩" in customer_step:
                if pages['data_overview']:
                    pages['data_overview']()
                else:
                    show_fallback_page("📊 Customer 데이터 개요", "web/pages/segmentation/data_overview.py")
            elif "2️⃣ 탐색적" in customer_step:
                if pages['exploratory_analysis']:
                    pages['exploratory_analysis']()
                else:
                    show_fallback_page("🔍 Customer EDA", "web/pages/segmentation/exploratory_analysis.py")
            elif "3️⃣ 클러스터링" in customer_step:
                if pages['clustering_analysis']:
                    pages['clustering_analysis']()
                else:
                    show_fallback_page("🎯 Customer 클러스터링", "web/pages/segmentation/clustering_analysis.py")
            elif "4️⃣ 주성분" in customer_step:
                if pages['pca_analysis']:
                    pages['pca_analysis']()
                else:
                    show_fallback_page("🔬 Customer PCA", "web/pages/segmentation/pca_analysis.py")
            elif "5️⃣ 딥러닝" in customer_step:
                if pages['deep_learning_analysis']:
                    pages['deep_learning_analysis']()
                else:
                    show_fallback_page("🌱 Customer 딥러닝", "web/pages/segmentation/deep_learning_analysis.py")
            elif "6️⃣ customer segmentation" in customer_step:
                if pages['customer_prediction']:
                    pages['customer_prediction']()
                else:
                    show_fallback_page("🔮 Customer 예측", "web/pages/segmentation/customer_prediction.py")
            elif "7️⃣ 마케팅" in customer_step:
                if pages['marketing_strategy']:
                    pages['marketing_strategy']()
                else:
                    show_fallback_page("📈 마케팅 전략", "web/pages/segmentation/marketing_strategy.py")

        elif current_focus == 'security':
            # 3. Security Analytics 라우팅
            if "1️⃣ 데이터 로딩" in security_step:
                if pages['security_analysis']:
                    st.info("📍 Security: 데이터 로딩 섹션")
                    from web.pages.security.security_analysis_page import show_data_download_section
                    show_data_download_section()
                else:
                    show_fallback_page("🔒 Security 데이터", "CICIDS2017 데이터 로딩")
            elif "2️⃣ 탐색적" in security_step:
                if pages['security_analysis']:
                    st.info("📍 Security: 탐색적 분석 섹션")
                    from web.pages.security.security_analysis_page import show_exploratory_analysis_section
                    show_exploratory_analysis_section()
                else:
                    show_fallback_page("🔍 Security EDA", "CICIDS2017 탐색적 분석")
            elif "3️⃣ 공격 패턴" in security_step:
                if pages['security_analysis']:
                    st.info("📍 Security: 공격 패턴 심화 분석")
                    from web.pages.security.security_analysis_page import show_attack_pattern_analysis
                    show_attack_pattern_analysis()
                else:
                    show_fallback_page("⚡ 공격 패턴", "CICIDS2017 공격 패턴 분석")
            elif "4️⃣ 딥러닝" in security_step:
                if pages['security_analysis']:
                    st.info("📍 Security: 딥러닝 모델링")
                    from web.pages.security.security_analysis_page import show_deep_learning_detection
                    show_deep_learning_detection()
                else:
                    show_fallback_page("🌱 Security 딥러닝", "CICIDS2017 딥러닝 모델")
            elif "5️⃣ Overfitting" in security_step:
                if pages['security_analysis']:
                    st.info("📍 Security: Overfitting 해결 검증")
                    from web.pages.security.security_analysis_page import show_overfitting_validation
                    show_overfitting_validation()
                else:
                    show_fallback_page("🎯 Overfitting 검증", "CICIDS2017 Overfitting 해결")
            elif "6️⃣ 실시간" in security_step:
                if pages['security_analysis']:
                    st.info("📍 Security: 실시간 예측 테스트")
                    from web.pages.security.security_analysis_page import show_real_time_prediction
                    show_real_time_prediction()
                else:
                    show_fallback_page("📊 실시간 예측", "CICIDS2017 실시간 탐지")
            elif "7️⃣ 종합" in security_step:
                if pages['security_analysis']:
                    st.info("📍 Security: 종합 성능 평가")
                    from web.pages.security.security_analysis_page import show_comprehensive_evaluation
                    show_comprehensive_evaluation()
                else:
                    show_fallback_page("🏆 종합 평가", "CICIDS2017 성능 평가")

        else:
            # 알 수 없는 포커스 (기본: retail)
            st.session_state.current_focus = 'retail'
            if pages['retail_analysis']:
                pages['retail_analysis']()
            else:
                show_fallback_page("🎆 Welcome", "통합 커머스 & 보안 분석 플랫폼")
            
    except Exception as e:
        st.error(f"라우팅 오류: {str(e)}")
        st.info("기본 페이지로 돌아갑니다.")
        if pages['retail_analysis']:
            pages['retail_analysis']()
        else:
            show_fallback_page("🚑 오류 복구", "기본 페이지")


def route_to_page(selected_page, pages):
    """간단한 페이지 라우팅"""

    try:
        if selected_page == "💰 온라인 리테일 전체 분석 (추천)":
            if pages['retail_analysis']:
                pages['retail_analysis']()
            else:
                show_fallback_page("💰 온라인 리테일 분석", "대용량 리테일 데이터 분석 페이지")

        elif selected_page == "📊 고객 데이터 개요":
            if pages['data_overview']:
                pages['data_overview']()
            else:
                show_fallback_page("📊 데이터 개요", "고객 데이터 개요 페이지")

        elif selected_page == "🔍 탐색적 데이터 분석":
            if pages['exploratory_analysis']:
                pages['exploratory_analysis']()
            else:
                show_fallback_page("🔍 탐색적 분석", "데이터 탐색 및 시각화 페이지")

        elif selected_page == "🎯 클러스터링 분석":
            if pages['clustering_analysis']:
                pages['clustering_analysis']()
            else:
                show_fallback_page("🎯 클러스터링", "K-means 클러스터링 분석 페이지")

        elif selected_page == "🔬 주성분 분석":
            if pages['pca_analysis']:
                pages['pca_analysis']()
            else:
                show_fallback_page("🔬 PCA 분석", "주성분 분석 및 차원 축소 페이지")

        elif selected_page == "🌱 딥러닝 오토인코더":
            if pages['deep_learning_analysis']:
                pages['deep_learning_analysis']()
            else:
                show_fallback_page("🌱 딥러닝", "오토인코더 딥러닝 모델 페이지")

        elif selected_page == "🔮 고객 세그먼트 예측":
            if pages['customer_prediction']:
                pages['customer_prediction']()
            else:
                show_fallback_page("🔮 고객 예측", "신규 고객 세그먼트 예측 페이지")

        elif selected_page == "📈 마케팅 전략 수립":
            if pages['marketing_strategy']:
                pages['marketing_strategy']()
            else:
                show_fallback_page("📈 마케팅 전략", "세그먼트별 마케팅 전략 페이지")

        elif selected_page == "🔒 네트워크 보안 이상 탐지":
            if pages['security_analysis']:
                pages['security_analysis']()
            else:
                show_fallback_page("🔒 보안 분석", "CICIDS2017 이상 탐지 분석 페이지")

        else:
            # 알 수 없는 페이지
            st.error(f"알 수 없는 페이지: {selected_page}")
            show_fallback_page("🚨 오류", "잘못된 페이지 선택")

    except Exception as e:
        st.error(f"페이지 로딩 중 오류: {str(e)}")
        show_fallback_page("🔧 오류 복구", f"페이지 로딩 실패: {selected_page}")

def show_fallback_page(title, description):
    """페이지 로딩 실패시 표시할 대체 페이지"""
    st.markdown(f"### {title}")
    st.info(f"📝 {description}")
    
    st.markdown("""
    **🔧 이 페이지는 현재 다음 이유로 사용할 수 없습니다:**
    - 모듈 import 오류
    - 데이터 파일 누락  
    - 의존성 패키지 문제
    
    **💡 해결 방법:**
    1. 페이지를 새로고침하세요
    2. 다른 페이지를 선택해보세요
    3. 문제가 지속되면 개발자에게 문의하세요
    """)
    
    # 간단한 데모 차트 표시
    try:
        import pandas as pd
        import plotly.express as px
        
        # 샘플 데이터 생성
        sample_data = pd.DataFrame({
            'x': range(10),
            'y': [i*2 + 1 for i in range(10)],
            'category': ['A'] * 5 + ['B'] * 5
        })
        
        fig = px.line(sample_data, x='x', y='y', color='category', 
                     title=f"{title} - 샘플 차트",
                     color_discrete_sequence=['#22C55E', '#14B8A6'])
        
        fig.update_layout(
            plot_bgcolor='rgba(255,255,255,0.9)',
            paper_bgcolor='rgba(255,255,255,0.9)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception:
        st.warning("샘플 차트도 표시할 수 없습니다.")

def show_footer():
    """간단한 푸터"""
    st.markdown("""
    <hr style="margin-top:2em;margin-bottom:1em;">
    <div style="text-align:center; color:#6B7280; font-size:0.9em;">
        <b>🌿 Integrated Commerce & Security Analytics</b><br>
        Simple Edition v3.0 | 데이터 기반 비즈니스 인텔리전스 플랫폼
    </div>
    """, unsafe_allow_html=True)

def main():
    """메인 애플리케이션 (단순 버전)"""
    try:
        # 1. 애플리케이션 초기화
        initialize_app()
        
        # 2. 페이지 모듈들 안전 로딩
        pages = safe_import_pages()
        
        # 로딩된 페이지 개수 표시
        loaded_count = sum(1 for page in pages.values() if page is not None)
        total_count = len(pages)
        
        if loaded_count < total_count:
            st.warning(f"⚠️ 일부 페이지 로딩 실패: {loaded_count}/{total_count}개 페이지 사용 가능")
        else:
            st.success(f"✅ 모든 페이지 로딩 완료: {loaded_count}개 페이지 준비됨")
        
        # 3. 사이드바 설정 및 페이지 선택
        retail_step, customer_step, security_step, current_focus = setup_simple_sidebar()
        
        # 4. 선택된 페이지 표시
        route_to_hierarchical_page(retail_step, customer_step, security_step, current_focus, pages)
        # else:
        #     st.info("📍 좌쪽 메뉴에서 분석할 페이지를 선택하세요.")
        
        # 5. 푸터 표시
        show_footer()
        
    except Exception as e:
        st.error("🚨 애플리케이션 시작 중 치명적인 오류가 발생했습니다.")
        st.error(f"**오류 내용**: {str(e)}")
        
        st.markdown("""
        **🔧 문제 해결 방법:**
        1. 페이지를 새로고침 (F5)
        2. 브라우저 캐시 삭제
        3. 가상환경 및 패키지 재설치 확인
        4. Python 버전 호환성 확인 (3.8-3.11 권장)
        """)
        
        # 에러 상세 정보 (개발자용)
        with st.expander("🔍 개발자용 에러 상세"):
            st.exception(e)

# 애플리케이션 진입점
if __name__ == "__main__":
    main()
