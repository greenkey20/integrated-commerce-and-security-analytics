"""
메인 애플리케이션 파일 (Selectbox 기반 네비게이션)

무한루프 방지를 위해 단순한 selectbox 구조로 변경
"""

import streamlit as st
import warnings
import sys
import os

# Python 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# web 디렉토리도 경로에 추가
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

# 보안 페이지
try:
    from web.pages.security.security_analysis_page import show_security_analysis_page
except ImportError:
    show_security_analysis_page = None

def initialize_app():
    """애플리케이션 초기 설정"""
    # 페이지 설정
    st.set_page_config(
        page_title=AppConfig.APP_TITLE,
        page_icon=AppConfig.APP_ICON,
        layout=AppConfig.LAYOUT,
        initial_sidebar_state="expanded",
    )
    
    # 한글 폰트 설정
    font_manager = FontManager()
    font_manager.setup_korean_font()
    
    # Green Theme CSS 추가
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
    
    # 제목 및 소개
    st.title(AppConfig.APP_TITLE)
    st.markdown(f"""
    {AppConfig.APP_DESCRIPTION}
    
    **버전**: {AppConfig.VERSION}
    """)


def setup_sidebar():
    """단순한 selectbox 기반 사이드바 (무한루프 방지)"""
    st.sidebar.title("📋 Navigation")
    
    # 세션 상태 초기화
    if 'current_section' not in st.session_state:
        st.session_state.current_section = UIConfig.DEFAULT_SECTION
        st.session_state.current_module = UIConfig.DEFAULT_MODULE
        st.session_state.current_page = UIConfig.DEFAULT_PAGE
    
    # 현재 위치 표시
    current_path = f"{st.session_state.current_module} > {st.session_state.current_page}"
    st.sidebar.success(f"📍 **현재 위치**\n{current_path}")
    st.sidebar.markdown("---")
    
    # 1단계: 섹션 선택
    section_options = ["📊 Business Intelligence", "🛡️ Security Analytics"]
    selected_section = st.sidebar.selectbox(
        "🏗️ 분석 영역 선택:",
        section_options,
        index=section_options.index(st.session_state.current_section) if st.session_state.current_section in section_options else 0
    )
    
    # 2단계: 모듈 선택
    if selected_section == "📊 Business Intelligence":
        module_options = ["💰 Retail Prediction", "👥 Customer Segmentation"]
    else:  # Security Analytics
        module_options = ["🔒 네트워크 보안 이상 탐지"]
    
    selected_module = st.sidebar.selectbox(
        "💼 모듈 선택:",
        module_options,
        index=module_options.index(st.session_state.current_module) if st.session_state.current_module in module_options else 0
    )
    
    # 3단계: 페이지 선택
    if selected_module == "💰 Retail Prediction":
        page_options = [
            "📊 전체 분석 요약",
            "1️⃣ 데이터 로딩 및 개요",
            "2️⃣ 데이터 정제 & 전처리",
            "3️⃣ 특성공학 & 파생변수 생성",
            "4️⃣ 타겟변수 생성",
            "5️⃣ 선형회귀 모델링",
            "6️⃣ 모델 평가 & 해석"
        ]
    elif selected_module == "👥 Customer Segmentation":
        page_options = [
            "1️⃣ 데이터 로딩 및 개요",
            "2️⃣ 탐색적 데이터 분석",
            "3️⃣ 클러스터링 분석",
            "4️⃣ 주성분 분석",
            "5️⃣ 오토인코더 딥러닝",
            "6️⃣ customer segmentation 예측",
            "7️⃣ 마케팅 전략"
        ]
    else:  # Security
        page_options = [
            "1️⃣ 데이터 로딩 및 개요",
            "2️⃣ 탐색적 데이터 분석",
            "3️⃣ 공격 패턴 심화 분석",
            "4️⃣ 딥러닝 모델링",
            "5️⃣ Overfitting 해결 검증",
            "6️⃣ 실시간 예측 테스트",
            "7️⃣ 종합 성능 평가"
        ]
    
    selected_page = st.sidebar.selectbox(
        "📄 분석 단계 선택:",
        page_options,
        index=page_options.index(st.session_state.current_page) if st.session_state.current_page in page_options else 0
    )
    
    # 상태 업데이트 (변경된 경우에만)
    if (selected_section != st.session_state.current_section or 
        selected_module != st.session_state.current_module or 
        selected_page != st.session_state.current_page):
        
        st.session_state.current_section = selected_section
        st.session_state.current_module = selected_module
        st.session_state.current_page = selected_page
    
    st.sidebar.markdown("---")
    
    # 빠른 액션
    st.sidebar.markdown("### 빠른 액션")
    if st.sidebar.button("🔄 전체 초기화", key="reset_all"):
        # 세션 상태 초기화
        for key in list(st.session_state.keys()):
            if key.startswith(('current_', 'analysis_', 'model_', 'data_')):
                del st.session_state[key]
        
        st.session_state.current_section = UIConfig.DEFAULT_SECTION
        st.session_state.current_module = UIConfig.DEFAULT_MODULE
        st.session_state.current_page = UIConfig.DEFAULT_PAGE
        
        st.sidebar.success("✨ 초기화 완료!")
        st.rerun()
    
    return selected_section, selected_module, selected_page


def route_to_page(section, module, page):
    """기존 페이지들을 selectbox에 연결하는 라우팅 시스템"""
    
    try:
        if section == "📊 Business Intelligence":
            if module == "💰 Retail Prediction":
                # Retail Prediction 분석 페이지들
                if page == "📊 전체 분석 요약":
                    show_retail_analysis_page()  # 기존 Online Retail 분석 페이지
                elif page == "1️⃣ 데이터 로딩 및 개요":
                    st.info("📋 Retail 데이터 로딩 & 개요 페이지 구현 예정")
                    show_retail_analysis_page()  
                elif page == "2️⃣ 데이터 정제 & 전처리":
                    st.info("🧹 데이터 정제 & 전처리 페이지 구현 예정")
                    show_retail_analysis_page()
                elif page == "3️⃣ 특성공학 & 파생변수 생성":
                    st.info("⚙️ 특성공학 & 파생변수 생성 페이지 구현 예정")
                    show_retail_analysis_page()
                elif page == "4️⃣ 타겟변수 생성":
                    st.info("🎯 타겟변수 생성 페이지 구현 예정")
                    show_retail_analysis_page()
                elif page == "5️⃣ 선형회귀 모델링":
                    st.info("🤖 선형회귀 모델링 페이지 구현 예정")
                    show_retail_analysis_page()
                elif page == "6️⃣ 모델 평가 & 해석":
                    st.info("📊 모델 평가 & 해석 페이지 구현 예정")
                    show_retail_analysis_page()
                else:
                    show_retail_analysis_page()
                    
            elif module == "👥 Customer Segmentation": 
                # Customer Segmentation 분석 페이지들 (기존 페이지 연결)
                if page == "1️⃣ 데이터 로딩 및 개요":
                    show_data_overview_page()
                elif page == "2️⃣ 탐색적 데이터 분석":
                    show_exploratory_analysis_page()
                elif page == "3️⃣ 클러스터링 분석":
                    show_clustering_analysis_page()
                elif page == "4️⃣ 주성분 분석":
                    show_pca_analysis_page()
                elif page == "5️⃣ 오토인코더 딥러닝":
                    show_deep_learning_analysis_page()
                elif page == "6️⃣ customer segmentation 예측":
                    show_customer_prediction_page()
                elif page == "7️⃣ 마케팅 전략":
                    show_marketing_strategy_page()
                else:
                    show_data_overview_page()
                    
        elif section == "🛡️ Security Analytics":
            if module == "🔒 네트워크 보안 이상 탐지":
                # Security Analytics 분석 페이지들
                if show_security_analysis_page is not None:
                    st.info(f"🔍 Security Analytics: {page}")
                    st.markdown("""
                    **CICIDS2017 데이터셋을 사용한 네트워크 이상 탐지**
                    - 대용량 네트워크 트래픽 데이터 분석
                    - 하이브리드 딥러닝 모델 (MLP + CNN) 사용
                    - 실시간 보안 위협 탐지 시스템 구현
                    - 금융권 SI 보안 전문가 양성 목적
                    """)
                    show_security_analysis_page()
                else:
                    st.warning("⚠️ 보안 분석 기능은 현재 임시 비활성화되어 있습니다.")
                    st.info("""
                    **개발 진행 상황:**
                    - Phase 1: Green Theme 디자인 완료 ✅
                    - Phase 2: Selectbox 네비게이션 완료 ✅  
                    - Phase 3: Security 모듈 재활성화 예정 🛠️
                    """)
        else:
            st.error(f"알 수 없는 섹션: {section}")
            st.info("기본 페이지(전체 분석 요약)로 이동합니다.")
            show_retail_analysis_page()
            
    except Exception as e:
        st.error(f"페이지 로딩 중 오류: {str(e)}")
        st.info("페이지를 새로고침하거나 다른 메뉴를 선택해보세요.")
        
        with st.expander("🔧 디버그 정보"):
            st.exception(e)


def show_footer():
    """푸터 정보 표시"""
    st.markdown("""
    <hr style="margin-top:2em;margin-bottom:1em;">
    <div style="text-align:center; color:gray; font-size:0.95em;">
        <b>Integrated Commerce & Security Analytics</b> &nbsp;|&nbsp; 
        <a href="https://github.com/greenkey20" target="_blank">GitHub</a> &nbsp;|&nbsp; 
        <a href="mailto:greenkey20@github.com">Contact</a>
        <br>
        <span>
            데이터: Mall Customer, Online Retail, CICIDS2017<br>
            기술: Python, Streamlit, Scikit-learn, TensorFlow, Plotly<br>
            버전: {ver} &nbsp;|&nbsp; © 2025 green umbrella by Eunyoung KANG. All rights reserved.
        </span>
    </div>
    """.format(ver=AppConfig.VERSION), unsafe_allow_html=True)


def main():
    """메인 애플리케이션 함수 (Selectbox 버전)"""
    try:
        # 1. 애플리케이션 초기화
        initialize_app()
        
        # 2. 사이드바 설정 및 메뉴 선택
        selected_section, selected_module, selected_page = setup_sidebar()
        
        # 3. 선택된 페이지로 라우팅
        if selected_section and selected_module and selected_page:
            route_to_page(selected_section, selected_module, selected_page)
        else:
            # 기본 페이지 표시
            st.info("🚀 시작하려면 좌쪽 메뉴에서 Business Intelligence > Retail Prediction을 선택하세요!")
            route_to_page(
                UIConfig.DEFAULT_SECTION, 
                UIConfig.DEFAULT_MODULE, 
                UIConfig.DEFAULT_PAGE
            )
        
        # 4. 푸터 표시
        show_footer()
        
    except Exception as e:
        st.error("애플리케이션 시작 중 치명적인 오류가 발생했습니다.")
        st.error(f"오류 내용: {str(e)}")
        
        st.info("""
        **🔧 문제 해결 방법:**
        1. 페이지를 새로고침 (F5)해 보세요
        2. 브라우저 캐시를 지워보세요
        3. 필요한 패키지가 모두 설치되어 있는지 확인해보세요
        4. 문제가 지속되면 개발자에게 문의하세요
        """)
        
        # 긴급 복구
        try:
            st.markdown("### 🚨 긴급 복구 모드")
            st.markdown("기본 기능만 제공합니다.")
            show_data_overview_page()
        except:
            st.error("긴급 복구도 실패했습니다. 시스템 관리자에게 문의하세요.")


# 애플리케이션 진입점
if __name__ == "__main__":
    main()
