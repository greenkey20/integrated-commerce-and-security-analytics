"""
Core 모듈 - 비즈니스 로직 및 데이터 처리

이 패키지는 애플리케이션의 핵심 비즈니스 로직을 포함합니다:
- security: 네트워크 보안 이상 탐지
- segmentation: 고객 세분화 (향후 추가)
- retail: 리테일 분석 (향후 추가)
"""

# 보안 모듈 import
from . import security

__all__ = [
    'security'
]

__version__ = "1.0.0"
__author__ = "Customer Segmentation Project"
__description__ = "핵심 비즈니스 로직 모듈"
