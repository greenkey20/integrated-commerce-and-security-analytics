"""
Core retail analysis module

이 모듈은 온라인 리테일 데이터 분석의 핵심 로직을 담당합니다.
UI로부터 분리된 비즈니스 로직 및 데이터 처리 기능을 제공합니다.
"""

from data.loaders.retail_loader import RetailDataLoader
from .retail_data_processor import RetailDataProcessor
from .retail_feature_engineer import RetailFeatureEngineer
from .model_trainer import RetailModelTrainer
from .visualizer import RetailVisualizer
from .analysis_manager import RetailAnalysisManager

__all__ = [
    'RetailDataLoader',
    'RetailDataProcessor', 
    'RetailFeatureEngineer',
    'RetailModelTrainer',
    'RetailVisualizer',
    'RetailAnalysisManager'
]

__version__ = "2.0.0"
__author__ = "Customer Segmentation Project"
__description__ = "리팩토링된 온라인 리테일 분석 핵심 모듈"
