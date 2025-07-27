"""
폰트 관리 유틸리티

기존 setup_korean_font_for_streamlit 함수를 클래스 기반으로 개선
"""

import os
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from config.settings import VisualizationConfig


class FontManager:
    """한글 폰트 설정을 담당하는 클래스"""
    
    def __init__(self):
        self.korean_font_prop = None
        self.korean_font_name = None
    
    @st.cache_resource
    def setup_korean_font(_self):
        """Streamlit용 한글 폰트 설정 (캐싱 적용)"""
        
        # 진단에서 확인된 신뢰할 수 있는 폰트들 (Windows + macOS)
        reliable_fonts = [
            # Windows 폰트
            {"name": "Malgun Gothic", "path": "C:/Windows/Fonts/malgun.ttf"},
            {"name": "Gulim", "path": "C:/Windows/Fonts/gulim.ttc"},
            {"name": "Dotum", "path": "C:/Windows/Fonts/dotum.ttc"},
            {"name": "Batang", "path": "C:/Windows/Fonts/batang.ttc"},
            # macOS 폰트
            {"name": "AppleGothic", "path": "/System/Library/Fonts/Supplemental/AppleGothic.ttf"},
            {"name": "Arial Unicode MS", "path": "/Library/Fonts/Arial Unicode.ttf"},
            {"name": "Helvetica", "path": "/System/Library/Fonts/Helvetica.ttc"},
        ]

        for font_info in reliable_fonts:
            font_path = font_info["path"]
            font_name = font_info["name"]

            if os.path.exists(font_path):
                try:
                    # 폰트를 matplotlib에 등록
                    fm.fontManager.addfont(font_path)

                    # FontProperties 객체 생성
                    font_prop = fm.FontProperties(fname=font_path)
                    actual_name = font_prop.get_name()

                    # matplotlib 전역 설정 적용
                    plt.rcParams["font.family"] = [actual_name]
                    plt.rcParams["font.sans-serif"] = [actual_name] + plt.rcParams[
                        "font.sans-serif"
                    ]
                    plt.rcParams["axes.unicode_minus"] = False

                    return font_prop, actual_name

                except Exception:
                    continue

        # 폰트 설정 실패 시 기본값 반환
        return None, None

    def get_font_property(self):
        """폰트 속성 반환"""
        if self.korean_font_prop is None:
            self.korean_font_prop, self.korean_font_name = self.setup_korean_font()
        return self.korean_font_prop

    def get_font_name(self):
        """폰트 이름 반환"""
        if self.korean_font_name is None:
            self.korean_font_prop, self.korean_font_name = self.setup_korean_font()
        return self.korean_font_name


# 전역 인스턴스 생성
font_manager = FontManager()

# 기존 함수와의 호환성을 위한 래퍼 함수
def setup_korean_font_for_streamlit():
    """기존 함수명과의 호환성을 위한 래퍼"""
    return font_manager.setup_korean_font()
