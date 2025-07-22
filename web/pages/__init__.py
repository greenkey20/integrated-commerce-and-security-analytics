"""
페이지 모듈 - Streamlit UI 레이어

각 기능별 사용자 인터페이스를 담당하는 모듈들
비즈니스 로직은 core 모듈에서 분리됨
"""

# 보안 분석 페이지 (임시 비활성화 - Phase 1-2 완료 후 재활성화)
try:
    from .security import show_security_analysis_page
except ImportError:
    # 보안 모듈이 없는 경우 None으로 설정
    show_security_analysis_page = None

# show_security_analysis_page = None  # 임시 완전 비활성화

# 리테일 분석 페이지  
try:
    from .retail.analysis import show_retail_analysis_page
except ImportError:
    show_retail_analysis_page = None

# 기존 세분화 페이지들 (향후 정리 예정)
try:
    from .segmentation.data_overview import show_data_overview_page
    from .segmentation.exploratory_analysis import show_exploratory_analysis_page
    from .segmentation.clustering_analysis import show_clustering_analysis_page
    from .segmentation.pca_analysis import show_pca_analysis_page
    from .segmentation.deep_learning_analysis import show_deep_learning_analysis_page
    from .segmentation.customer_prediction import show_customer_prediction_page
    from .segmentation.marketing_strategy import show_marketing_strategy_page
except ImportError as e:
    # 임포트 오류가 있는 경우 None으로 설정
    show_data_overview_page = None
    show_exploratory_analysis_page = None
    show_clustering_analysis_page = None
    show_pca_analysis_page = None
    show_deep_learning_analysis_page = None
    show_customer_prediction_page = None
    show_marketing_strategy_page = None
    print(f"Warning: Could not import segmentation modules: {e}")

__all__ = [
    # 보안 분석
    'show_security_analysis_page',
    
    # 리테일 분석
    'show_retail_analysis_page',
    
    # 고객 세분화 (기존)
    'show_data_overview_page',
    'show_exploratory_analysis_page', 
    'show_clustering_analysis_page',
    'show_pca_analysis_page',
    'show_deep_learning_analysis_page',
    'show_customer_prediction_page',
    'show_marketing_strategy_page'
]

# None인 함수들은 __all__에서 제외
__all__ = [name for name in __all__ if globals().get(name) is not None]

__version__ = "1.0.0"
__author__ = "Customer Segmentation Project"
__description__ = "사용자 인터페이스 페이지 모듈"
