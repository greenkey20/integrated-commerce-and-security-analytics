#!/usr/bin/env python3
"""
간단한 Streamlit 앱 실행 테스트
"""

import sys
import os

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("=== Streamlit 앱 실행 테스트 ===")

try:
    # main_app.py의 주요 import들 테스트
    from config.settings import AppConfig, UIConfig
    from utils.font_manager import FontManager
    print("✅ config, utils 모듈 import 성공")
    
    # 페이지 모듈들 테스트
    from web.pages.retail.analysis import show_retail_analysis_page
    print("✅ 리테일 분석 페이지 import 성공")
    
    from web.pages.segmentation.data_overview import show_data_overview_page
    print("✅ 세그멘테이션 페이지 import 성공")
    
    print("\n모든 주요 import가 성공했습니다!")
    print("Streamlit 앱 실행 준비 완료")
    print("\n실행 명령어:")
    print("streamlit run main_app.py")
    
except ImportError as e:
    print(f"❌ Import 실패: {e}")
    print("추가 수정이 필요합니다.")

except Exception as e:
    print(f"❌ 기타 오류: {e}")
