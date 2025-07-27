"""
메인 애플리케이션 파일 (Simple Version - 무한루프 수정)
"""

import streamlit as st
import warnings
import sys
import os

# TensorFlow 경고 제거
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Python 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

web_dir = os.path.join(current_dir, "web")
if web_dir not in sys.path:
    sys.path.insert(0, web_dir)

warnings.filterwarnings("ignore")

# 설정 및 유틸리티 모듈
from config.settings import AppConfig, UIConfig
from utils.font_manager import FontManager

# 페이지 모듈들
from web.pages.segmentation.data_overview import show_data_overview_page
from web.pages.segmentation.exploratory_analysis import show_exploratory_analysis_page
from web.pages.segmentation.clustering_analysis import show_clustering_analysis_page
from web.pages.segmentation.pca_analysis import show_pca_analysis_page
from web.pages.segmentation.deep_learning_analysis import show_deep_learning_analysis_page
from web.pages.segmentation.customer_prediction import show_customer_prediction_page
from web.pages.segmentation.marketing_strategy import show_marketing_strategy_page
from web.pages.retail.analysis import show_retail_analysis_page

# 보안 페이지 (임시 비활성화)
try:
    from web.pages.security.security_analysis_page import show_security_analysis_page
except ImportError:
    show_security_analysis_page = None

def initialize_app():
    """애플리케이션 초기 설정"""
    st.set_page_config(
        page_title=AppConfig.APP_TITLE,
        page_icon=AppConfig.APP_ICON,
        layout=AppConfig.LAYOUT,
        initial_sidebar_state="expanded",
    )
    
    # 한글 폰트 설정
    font_manager = FontManager()
    font_manager.setup_korean_font()
    
    # Green Theme CSS (Light Mode)
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
        background: rgba(255, 255, 255, 0.98) !important;
        color: #064E3B !important;
        border: 1px solid rgba(34, 197, 94, 0.15) !important;
        border-radius: 16px !important;
    }
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #FFFFFF, #F0FDF4) !important;
        border: 1px solid #BBF7D0 !important;
        color: #064E3B !important;
        border-radius: 12px !important;
    }
    .stSuccess { background: #F0FDF4 !important; border: 1px solid #BBF7D0 !important; color: #064E3B !important; }
    .stWarning { background: #FFFBEB !important; border: 1px solid #FDE68A !important; color: #92400E !important; }
    .stError { background: #FEF2F2 !important; border: 1px solid #FECACA !important; color: #991B1B !important; }
    
    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6 {
        color: #064E3B !important;
        font-weight: 600 !important;
    }
    .main p, .main div, .main span, .main label, .main li {
        color: #374151 !important;
    }
    .main a { color: #059669 !important; }
    .main a:hover { color: #047857 !important; }
    
    .js-plotly-plot, .plotly {
        background: rgba(255, 255, 255, 0.95) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 헤더
    st.title(AppConfig.APP_TITLE)
    st.markdown(f"""
    {AppConfig.APP_DESCRIPTION}
    
    **버전**: {AppConfig.VERSION}
    """)

def setup_simple_sidebar():
    """단순한 selectbox 기반 사이드바 (무한루프 없음)"""
    st.sidebar.title("📋 Navigation")
    
    # 세션 상태 초기화 (한 번만)
    if 'selected_section' not in st.session_state:
        st.session_state.selected_section = UIConfig.DEFAULT_SECTION
    if 'selected_module' not in st.session_state:
        st.session_state.selected_module = UIConfig.DEFAULT_MODULE  
    if 'selected_page' not in st.session_state:
        st.session_state.selected_page = UIConfig.DEFAULT_PAGE
    
    # 현재 위치 표시
    st.sidebar.success(f"📍 **현재**: {st.session_state.selected_module}")
    st.sidebar.markdown("---")
    
    # 1단계: 섹션 선택
    section = st.sidebar.selectbox(
        "🏗️ 분석 영역:",
        ["📊 Business Intelligence", "🛡️ Security Analytics"],
        key="section_select"
    )
    
    # 2단계: 모듈 선택  
    if section == "📊 Business Intelligence":
        modules = ["💰 Retail Analytics", "👥 Customer Analytics"]
    else:
        modules = ["🔒 네트워크 보안 분석"]
    
    module = st.sidebar.selectbox("💼 모듈:", modules, key="module_select")
    
    # 3단계: 페이지 선택
    if module == "💰 Retail Analytics":
        pages = ["📊 전체 분석", "1️⃣ 데이터 로딩", "2️⃣ 전처리", "3️⃣ 모델링", "4️⃣ 평가"]
    elif module == "👥 Customer Analytics":
        pages = ["1️⃣ 데이터 개요", "2️⃣ EDA", "3️⃣ 클러스터링", "4️⃣ PCA", "5️⃣ 딥러닝", "6️⃣ 예측", "7️⃣ 마케팅"]
    else:  # Security
        pages = ["1️⃣ 보안 데이터", "2️⃣ 위협 분석", "3️⃣ 딥러닝", "4️⃣ 실시간 탐지"]
    
    page = st.sidebar.selectbox("📄 페이지:", pages, key="page_select")
    
    # 세션 상태 업데이트 (rerun 없이)
    st.session_state.selected_section = section
    st.session_state.selected_module = module
    st.session_state.selected_page = page
    
    st.sidebar.markdown("---")
    
    # 빠른 리셋 (단순화)
    if st.sidebar.button("🔄 초기화"):
        st.session_state.clear()
        st.sidebar.success("✨ 완료!")
    
    return section, module, page

def route_simple(section, module, page):
    """단순한 라우팅 (예외처리 강화)"""
    try:
        # Business Intelligence
        if section == "📊 Business Intelligence":
            if module == "💰 Retail Analytics":
                show_retail_analysis_page()
            elif module == "👥 Customer Analytics":
                if "데이터 개요" in page:
                    show_data_overview_page()
                elif "EDA" in page:
                    show_exploratory_analysis_page()
                elif "클러스터링" in page:
                    show_clustering_analysis_page()
                elif "PCA" in page:
                    show_pca_analysis_page()
                elif "딥러닝" in page:
                    show_deep_learning_analysis_page()
                elif "예측" in page:
                    show_customer_prediction_page()
                elif "마케팅" in page:
                    show_marketing_strategy_page()
                else:
                    show_data_overview_page()
        
        # Security Analytics        
        elif section == "🛡️ Security Analytics":
            if show_security_analysis_page:
                show_security_analysis_page()
            else:
                st.warning("⚠️ 보안 분석 모듈 로딩 중...")
                st.info("Phase 3에서 재활성화 예정")
        
        else:
            st.error("알 수 없는 섹션")
            show_retail_analysis_page()
            
    except Exception as e:
        st.error(f"페이지 로딩 오류: {str(e)}")
        st.info("다른 페이지를 선택해보세요.")

def main():
    """메인 함수 (단순화)"""
    try:
        # 초기화
        initialize_app()
        
        # 사이드바 & 라우팅
        section, module, page = setup_simple_sidebar()
        route_simple(section, module, page)
        
        # 푸터
        st.markdown("---")
        st.markdown(f"**{AppConfig.VERSION}** | 🌿 Green Commerce Intelligence", unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"애플리케이션 오류: {str(e)}")
        st.info("브라우저를 새로고침(F5)해보세요.")

if __name__ == "__main__":
    main()
