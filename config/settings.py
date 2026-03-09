"""
고객 세분화 프로젝트 전역 설정

이 파일은 프로젝트 전반에서 사용되는 설정값들을 한 곳에 모아서 관리해.
하드코딩된 값들을 여기로 옮기면 나중에 설정 변경할 때 이 파일만 수정하면 돼.
"""

class AppConfig:
    """애플리케이션 전체 설정"""
    APP_TITLE = "🌿 Integrated Commerce & Security Analytics"
    APP_SUBTITLE = "차세대 이커머스를 위한 통합 인텔리전스 플랫폼"
    APP_DESCRIPTION = "고객 인사이트부터 보안 모니터링까지, 데이터 기반 비즈니스 성장을 지원합니다."
    APP_ICON = "🌿"
    LAYOUT = "wide"
    VERSION = "v3.0 - 통합 분석 플랫폼"
    
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

class SecurityConfig:
    """보안 및 이상탐지 관련 설정"""
    # 모델 기본값
    RANDOM_SEED = 42
    DEFAULT_MODEL_TYPE = 'hybrid'
    
    # 신경망 구조
    CNN_SEQUENCE_LENGTH = 10
    MLP_HIDDEN_UNITS = [128, 64, 32]
    DROPOUT_RATES = [0.3, 0.2]
    
    # 훈련 파라미터
    DEFAULT_EPOCHS = 10
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.2
    EARLY_STOPPING_PATIENCE = 3
    LEARNING_RATE_PATIENCE = 2
    
    # 실시간 모니터링
    ALERT_THRESHOLD = 0.7
    MAX_RECENT_ANOMALIES = 100
    
    # 특성 추출 설정
    BUSINESS_HOUR_START = 9
    BUSINESS_HOUR_END = 17
    WEEKEND_THRESHOLD = 5  # 0=월요일, 5=토요일
    
    # 위험도 레벨 임계값
    RISK_THRESHOLDS = {
        'CRITICAL': 0.9,
        'HIGH': 0.7,
        'MEDIUM': 0.5,
        'LOW': 0.0
    }

class LoggingConfig:
    """로깅 관련 설정"""
    LOG_LEVEL = "INFO"
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 로그 파일 설정
    API_ACCESS_LOG = "logs/api_access.log"
    SECURITY_LOG = "logs/security.log"
    ERROR_LOG = "logs/error.log"
    
    # 로그 로테이션
    MAX_LOG_SIZE = "10MB"
    BACKUP_COUNT = 5

class UIConfig:
    """UI 관련 설정"""
    SIDEBAR_WIDTH = 300
    
    # 기존 평면 메뉴 (하위 호환성을 위해 유지)
    MENU_OPTIONS = [
        "📊 데이터 개요",
        "🔍 탐색적 데이터 분석", 
        "🎯 클러스터링 분석",
        "🔬 주성분 분석",
        "🌱 딥러닝 분석",
        "🔮 고객 예측",
        "📈 마케팅 전략",
        "🛍 온라인 리테일 분석",
        "🔒 보안 이상 탐지 분석",
    ]
    
    # 새로운 단순한 메뉴 구조 (라디오버튼 기반)
    SIMPLE_MENU_STRUCTURE = {
        "📊 Business Intelligence": {
            "💰 Retail Prediction": [
                "1️⃣ 데이터 로딩 & 개요",
                "2️⃣ 데이터 정제 & 전처리", 
                "3️⃣ 특성공학 & 파생변수",
                "4️⃣ 타겟변수 생성",
                "5️⃣ 선형회귀 모델링",
                "6️⃣ 모델 평가 & 해석",
                "📊 전체 분석 요약"
            ],
            "👥 Customer Segmentation": [
                "1️⃣ 데이터 로딩 & 개요",
                "2️⃣ 탐색적 데이터 분석",
                "3️⃣ 클러스터링 분석", 
                "4️⃣ 주성분 분석",
                "5️⃣ 오토인코더 딥러닝",
                "6️⃣ 세그먼트 예측",
                "7️⃣ 마케팅 전략"
            ]
        },
        "🛡️ Security Analytics": {
            "🔒 네트워크 이상 탐지": [
                "1️⃣ CICIDS2017 데이터 로딩",
                "2️⃣ 네트워크 트래픽 EDA",
                "3️⃣ 공격 패턴 심화 분석", 
                "4️⃣ 하이브리드 딥러닝 모델링",
                "5️⃣ 오버피팅 해결 & 검증",
                "6️⃣ 실시간 예측 테스트",
                "7️⃣ 종합 성능 평가"
            ]
        }
    }
    
    # 기본 시작 페이지
    DEFAULT_SECTION = "📊 Business Intelligence"
    DEFAULT_MODULE = "💰 Retail Prediction"
    DEFAULT_PAGE = "📊 전체 분석 요약"

# config/settings.py에 추가
class TextAnalyticsConfig:
    """텍스트 분석 도메인 설정"""

    # IMDB 감정 분석
    IMDB_VOCAB_SIZE = 10000
    IMDB_MAX_LENGTH = 500
    IMDB_EMBEDDING_DIM = 100
    IMDB_LSTM_UNITS = 64

    # 뉴스 분류 (확장용)
    NEWS_CATEGORIES = ['business', 'entertainment', 'politics', 'sport', 'tech']

    # 공통 설정
    BATCH_SIZE = 32
    EPOCHS = 50
    VALIDATION_SPLIT = 0.2
    EARLY_STOPPING_PATIENCE = 5