"""
고객 세분화 프로젝트 전역 설정

이 파일은 프로젝트 전반에서 사용되는 설정값들을 한 곳에 모아서 관리해.
하드코딩된 값들을 여기로 옮기면 나중에 설정 변경할 때 이 파일만 수정하면 돼.
"""

class AppConfig:
    """애플리케이션 전체 설정"""
    APP_TITLE = "Mall Customer Segmentation Analysis"
    APP_ICON = "🛍️"
    LAYOUT = "wide"
    VERSION = "v2.0"
    
    # 데이터 관련 설정
    DATA_URL = "https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Datasets/Mall_Customers.csv"
    CACHE_TTL = 3600  # 캐시 유지 시간 (초)

class ClusteringConfig:
    """클러스터링 분석 관련 설정"""
    DEFAULT_CLUSTERS = 5
    MIN_CLUSTERS = 2
    MAX_CLUSTERS = 10
    RANDOM_STATE = 42
    N_INIT = 10

class VisualizationConfig:
    """시각화 관련 설정"""
    # 클러스터별 색상 팔레트
    COLOR_PALETTE = [
        "#e41a1c",  # 빨강 - 프리미엄/고소득
        "#377eb8",  # 파랑 - 보수적/안정적
        "#4daf4a",  # 초록 - 일반/균형적
        "#984ea3",  # 보라 - 적극소비/젊은층
        "#ff7f00",  # 주황 - 절약형/실용적
        "#ffff33",  # 노랑 - 특별 카테고리
        "#a65628",  # 갈색 - 중년층/전통적
        "#f781bf",  # 분홍 - 여성적/감성적
        "#999999",  # 회색 - 중립적
        "#66c2a5",  # 청록
    ]
    
    # 그래프 크기 설정
    FIGURE_WIDTH = 12
    FIGURE_HEIGHT = 8
    
    # 폰트 경로들 (우선순위 순)
    FONT_PATHS = [
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/Library/Fonts/Arial Unicode.ttf", 
        "/System/Library/Fonts/Helvetica.ttc"
    ]

class DeepLearningConfig:
    """딥러닝 모델 관련 설정"""
    # 분류 모델 기본값
    DEFAULT_HIDDEN_UNITS = 64
    DEFAULT_DROPOUT_RATE = 0.2
    DEFAULT_LEARNING_RATE = 0.001
    DEFAULT_EPOCHS = 100
    
    # 오토인코더 기본값
    DEFAULT_ENCODING_DIM = 2
    AUTOENCODER_EPOCHS = 100
    
    # 공통 설정
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.2
    EARLY_STOPPING_PATIENCE = 15

class UIConfig:
    """UI 관련 설정"""
    SIDEBAR_WIDTH = 300
    
    # 메뉴 옵션들
    MENU_OPTIONS = [
        "📊 데이터 개요",
        "🔍 탐색적 데이터 분석", 
        "🎯 클러스터링 분석",
        "🔬 주성분 분석",
        "🧠 딥러닝 분석",
        "🔮 고객 예측",
        "📈 마케팅 전략",
        "🛍 온라인 리테일 분석",
        "🔒 보안 이상 탐지 분석",
    ]
