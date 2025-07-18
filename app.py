"""
메인 애플리케이션 파일 (새로운 모듈화 구조)

기존 customer_segmentation_app.py를 완전히 모듈화하여 재구성한 메인 파일.
각 기능별로 분리된 모듈들을 임포트하여 사용.
"""

import streamlit as st
import warnings
import sys
import os

# 현재 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

warnings.filterwarnings("ignore")

# 설정 및 유틸리티 모듈
from config.settings import AppConfig, UIConfig
from utils.font_manager import FontManager

# 페이지 모듈들
from app_modules.data_overview import show_data_overview_page
from app_modules.exploratory_analysis import show_exploratory_analysis_page
from app_modules.clustering_analysis import show_clustering_analysis_page
from app_modules.pca_analysis import show_pca_analysis_page
from app_modules.deep_learning_analysis import show_deep_learning_analysis_page
from app_modules.customer_prediction import show_customer_prediction_page
from app_modules.marketing_strategy import show_marketing_strategy_page
from app_modules.retail_analysis import show_retail_analysis_page


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
    
    # 제목 및 소개
    st.title(AppConfig.APP_TITLE)
    st.markdown(f"""
    이 애플리케이션은 K-means 클러스터링을 활용하여 쇼핑몰 고객을 세분화하고 
    각 그룹별 특성을 분석하여 맞춤형 마케팅 전략을 제공합니다.
    
    **버전**: {AppConfig.VERSION} - 딥러닝 지원
    """)


def setup_sidebar():
    """사이드바 메뉴 설정"""
    st.sidebar.title("📋 Navigation")
    st.sidebar.markdown("---")

    # 라디오 버튼으로 메뉴 선택
    selected_menu = st.sidebar.radio(
        "분석 단계를 선택하세요:", 
        UIConfig.MENU_OPTIONS, 
        index=0
    )

    # 선택된 메뉴에서 이모지 제거하여 반환
    menu = selected_menu.split(" ", 1)[1]  # 이모지와 공백 제거

    # 현재 선택된 메뉴 강조 표시
    st.sidebar.markdown("---")
    st.sidebar.info(f"현재 페이지: **{selected_menu}**")
    
    # 페이지 변경 감지 및 스크롤 초기화 (강화됨)
    if 'current_page' not in st.session_state:
        st.session_state.current_page = menu
        # 첫 로드시에도 스크롤 초기화
        st.session_state.scroll_to_top = True
    elif st.session_state.current_page != menu:
        # 페이지가 변경된 경우 스크롤 초기화 플래그 설정
        st.session_state.current_page = menu
        st.session_state.scroll_to_top = True
        # 페이지 변경 로그 (디버깅용)
        print(f"페이지 변경 감지: {st.session_state.current_page} -> {menu}")

    # 사용 가이드
    with st.sidebar.expander("💡 사용 가이드"):
        st.markdown("""
        **분석 순서 권장:**
        
        **👍 Mall Customer 분석 (기본):**
        1. 📊 데이터 개요 - 기본 정보 파악
        2. 🔍 탐색적 분석 - 패턴 발견
        3. 🎯 클러스터링 - 고객 세분화
        4. 🔬 주성분 분석 - 차원 축소
        5. 🧠 딥러닝 - 고급 모델링
        6. 🔮 고객 예측 - 실제 적용
        7. 📈 마케팅 전략 - 비즈니스 활용
        
        **🚀 Online Retail 분석 (고급):**
        - 대용량 실무 데이터 분석
        - "혼공머신" 연계 선형회귀 학습
        - ADP 실기 대비 특성 공학
        - 단계별 체계적 학습 경험
        """)

    # 프로젝트 정보
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🛠️ 프로젝트 정보**")
    st.sidebar.markdown("고객 세분화 분석 도구")
    st.sidebar.markdown(f"{AppConfig.VERSION} - 딥러닝 지원")
    st.sidebar.markdown("**🏗️ 모듈화 구조**: 각 기능별로 분리된 모듈 구조")

    return menu


def route_to_page(menu):
    """선택된 메뉴에 따라 해당 페이지로 라우팅"""
    
    # 강화된 페이지 로딩 전 스크롤 초기화
    if st.session_state.get('scroll_to_top', False):
        # CSS를 통한 즉시 스크롤 초기화
        st.markdown("""
        <style>
        .main > div {
            scroll-behavior: auto !important;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # 페이지 변경 시에만 스크롤 초기화 수행
    if st.session_state.get('scroll_to_top', False):
        st.components.v1.html("""
            <script>
                // 강화된 스크롤 초기화 함수
                function forceScrollToTop() {
                    // Streamlit 메인 컨테이너들을 찾는 더 정확한 선택자들
                    const selectors = [
                        // Streamlit 특화 선택자들
                        '[data-testid="stAppViewContainer"]',
                        '[data-testid="stMain"]',
                        '.main .block-container',
                        '.stApp > div',
                        '.stApp',
                        'section.main',
                        '.main',
                        '.stMainBlockContainer',
                        
                        // 일반적인 선택자들
                        'main',
                        'body',
                        'html'
                    ];
                    
                    // 각 선택자에 대해 스크롤 초기화 시도
                    selectors.forEach(selector => {
                        try {
                            // 현재 윈도우에서 찾기
                            let elements = document.querySelectorAll(selector);
                            elements.forEach(el => {
                                if (el) {
                                    el.scrollTop = 0;
                                    el.scrollLeft = 0;
                                    if (el.scrollTo) el.scrollTo(0, 0);
                                }
                            });
                            
                            // 부모 윈도우에서 찾기 (iframe인 경우)
                            if (window.parent && window.parent.document) {
                                elements = window.parent.document.querySelectorAll(selector);
                                elements.forEach(el => {
                                    if (el) {
                                        el.scrollTop = 0;
                                        el.scrollLeft = 0;
                                        if (el.scrollTo) el.scrollTo(0, 0);
                                    }
                                });
                            }
                        } catch (e) {
                            // 접근 권한 오류 등은 무시
                            console.log('스크롤 초기화 시도 중 오류:', e);
                        }
                    });
                    
                    // 윈도우 전체 스크롤도 초기화
                    try {
                        window.scrollTo(0, 0);
                        if (window.parent) {
                            window.parent.scrollTo(0, 0);
                        }
                    } catch (e) {
                        console.log('윈도우 스크롤 초기화 중 오류:', e);
                    }
                }
                
                // 즉시 실행 및 다양한 시점에서 재시도
                forceScrollToTop();
                
                // DOM이 준비된 후 실행
                if (document.readyState === 'loading') {
                    document.addEventListener('DOMContentLoaded', forceScrollToTop);
                } else {
                    setTimeout(forceScrollToTop, 50);
                }
                
                // 추가 지연 실행으로 확실하게
                setTimeout(forceScrollToTop, 100);
                setTimeout(forceScrollToTop, 300);
                setTimeout(forceScrollToTop, 500);
                setTimeout(forceScrollToTop, 1000);
                
                // 페이지 로드 완료 후에도 한 번 더
                window.addEventListener('load', () => {
                    setTimeout(forceScrollToTop, 100);
                });
            </script>
        """, height=0)
        
        # 스크롤 초기화 플래그 리셋
        st.session_state.scroll_to_top = False
    
    try:
        if menu == "데이터 개요":
            show_data_overview_page()
            
        elif menu == "탐색적 데이터 분석":
            show_exploratory_analysis_page()
            
        elif menu == "클러스터링 분석":
            show_clustering_analysis_page()
            
        elif menu == "주성분 분석":
            show_pca_analysis_page()
            
        elif menu == "딥러닝 분석":
            show_deep_learning_analysis_page()
            
        elif menu == "고객 예측":
            show_customer_prediction_page()
            
        elif menu == "마케팅 전략":
            show_marketing_strategy_page()
            
        elif menu == "온라인 리테일 분석":
            show_retail_analysis_page()
            
        else:
            st.error(f"알 수 없는 메뉴: {menu}")
            
    except Exception as e:
        st.error(f"페이지 로딩 중 오류가 발생했습니다: {str(e)}")
        st.info("페이지를 새로고침하거나 다른 메뉴를 선택해 보세요.")
        
        # 디버그 정보 (개발 시에만 표시)
        with st.expander("🔧 디버그 정보 (개발자용)"):
            st.exception(e)


def show_footer():
    """푸터 정보 표시"""
    st.markdown("---")
    st.markdown(f"""
    **📋 애플리케이션 정보**
    
    **개발 정보:** 이 애플리케이션은 K-means 클러스터링을 활용한 고객 세분화 분석 도구입니다.  
    **데이터:** Mall Customer Segmentation Dataset & Online Retail Dataset  
    **기술 스택:** Python, Streamlit, Scikit-learn, TensorFlow, Plotly
    **버전:** {AppConfig.VERSION}
    **구조:** 모듈화된 페이지 기반 아키텍처
    
    **🎆 새로운 기능:**
    - **"혼공머신" 연계 학습**: 선형회귀 알고리즘 실무 적용
    - **대용량 데이터 처리**: 50만+ 거래 데이터 분석 경험
    - **ADP 실기 대비**: 특성 공학, groupby/agg 활용
    - **단계별 학습**: 데이터 로딩부터 모델 평가까지
    
    **🏗️ 모듈 구조:**
    - `config/`: 전역 설정 관리
    - `core/`: 핵심 비즈니스 로직 (데이터 처리, 클러스터링, 딥러닝, 리테일 분석)
    - `utils/`: 유틸리티 함수 (폰트 관리 등)
    - `app_modules/`: 각 페이지별 UI 로직
    """)


def main():
    """메인 애플리케이션 함수"""
    try:
        # 1. 애플리케이션 초기화
        initialize_app()
        
        # 2. 사이드바 설정 및 메뉴 선택
        selected_menu = setup_sidebar()
        
        # 3. 선택된 페이지로 라우팅
        route_to_page(selected_menu)
        
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
        
        # 긴급 복구: 기본 데이터 개요 페이지라도 표시
        try:
            st.markdown("### 🚨 긴급 복구 모드")
            st.markdown("기본 기능만 제공합니다.")
            show_data_overview_page()
        except:
            st.error("긴급 복구도 실패했습니다. 시스템 관리자에게 문의하세요.")


# 애플리케이션 진입점
if __name__ == "__main__":
    main()
