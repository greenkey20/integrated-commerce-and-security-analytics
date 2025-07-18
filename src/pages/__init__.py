# app_modules 모듈
# 각 페이지별 UI 로직을 담당하는 모듈들

# 모든 페이지 함수들을 임포트하여 외부에서 쉽게 사용할 수 있도록 함
from .data_overview import show_data_overview_page
from .exploratory_analysis import show_exploratory_analysis_page
from .clustering_analysis import show_clustering_analysis_page
from .pca_analysis import show_pca_analysis_page
from .deep_learning_analysis import show_deep_learning_analysis_page
from .customer_prediction import show_customer_prediction_page
from .marketing_strategy import show_marketing_strategy_page

__all__ = [
    'show_data_overview_page',
    'show_exploratory_analysis_page', 
    'show_clustering_analysis_page',
    'show_pca_analysis_page',
    'show_deep_learning_analysis_page',
    'show_customer_prediction_page',
    'show_marketing_strategy_page'
]
