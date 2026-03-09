"""
Core 모듈 - 비즈니스 로직 및 데이터 처리

이 패키지는 애플리케이션의 핵심 비즈니스 로직을 포함합니다:
- security: 네트워크 보안 이상 탐지
- segmentation: 고객 세분화
- retail: 리테일 분석
- text: 텍스트 분석

상위 패키지 import 시 서브패키지의 무거운 의존성이 자동으로 로드되지 않도록,
__all__에 심볼만 노출하고 실제 import는 필요 시점에 수행합니다.
"""

__all__ = ['security', 'segmentation', 'retail', 'text']

__version__ = "1.0.0"
__author__ = "Customer Segmentation Project"
__description__ = "핵심 비즈니스 로직 모듈"
