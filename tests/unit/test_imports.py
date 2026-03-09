#!/usr/bin/env python3
"""
Phase 1-2 Import 경로 업데이트 테스트
"""

import sys
import os

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("=== Phase 1-2 Import 테스트 시작 ===\n")

# 테스트 1: 새로운 data.loaders 모듈 테스트
print("1. data.loaders 모듈 테스트:")
try:
    from data.loaders.retail_loader import RetailDataLoader
    print("   ✅ RetailDataLoader import 성공")
except ImportError as e:
    print(f"   ❌ RetailDataLoader import 실패: {e}")

# 임시로 SecurityDataLoader 비활성화 상태
print("   ⚠️ SecurityDataLoader 임시 비활성화 (추후 수정 예정)")
# try:
#     from data.loaders import SecurityDataLoader
#     print("   ✅ SecurityDataLoader import 성공")
# except ImportError as e:
#     print(f"   ❌ SecurityDataLoader import 실패: {e}")

# 테스트 2: data.base 모듈 테스트
print("\n2. data.base 모듈 테스트:")
try:
    from data.base import DataValidator, DataCleaner, DataEngineer, DataSplitter
    print("   ✅ data.base 모든 클래스 import 성공")
except ImportError as e:
    print(f"   ❌ data.base import 실패: {e}")

# 테스트 3: 통합 DataProcessor 테스트
print("\n3. 통합 DataProcessor 테스트:")
try:
    from data import DataProcessor
    print("   ✅ DataProcessor import 성공")
except ImportError as e:
    print(f"   ❌ DataProcessor import 실패: {e}")

# 테스트 4: 업데이트된 analysis_manager 테스트
print("\n4. 업데이트된 analysis_manager 테스트:")
try:
    from core.retail.analysis_manager import RetailAnalysisManager
    print("   ✅ RetailAnalysisManager import 성공")
except ImportError as e:
    print(f"   ❌ RetailAnalysisManager import 실패: {e}")
except Exception as e:
    print(f"   ❌ RetailAnalysisManager 기타 오류: {e}")

# 테스트 5: 실제 인스턴스 생성 테스트
print("\n5. 실제 인스턴스 생성 테스트:")
try:
    # RetailDataLoader 인스턴스 생성
    loader = RetailDataLoader()
    print("   ✅ RetailDataLoader 인스턴스 생성 성공")
    
    # DataProcessor 인스턴스 생성
    processor = DataProcessor()
    print("   ✅ DataProcessor 인스턴스 생성 성공")
    
    # RetailAnalysisManager 인스턴스 생성
    manager = RetailAnalysisManager()
    print("   ✅ RetailAnalysisManager 인스턴스 생성 성공")
    
except Exception as e:
    print(f"   ❌ 인스턴스 생성 실패: {e}")

# 테스트 6: 웹 페이지 import 테스트
print("\n6. 웹 페이지 import 테스트:")
try:
    from web.pages.retail.data_loading import show_data_loading_page
    print("   ✅ retail data_loading 페이지 import 성공")
except ImportError as e:
    print(f"   ❌ retail data_loading 페이지 import 실패: {e}")

print("\n=== Phase 1-2 Import 테스트 완료 ===")
print("\n다음 단계:")
print("- 모든 테스트가 성공하면 Phase 1-3으로 진행")
print("- 실패한 테스트가 있다면 해당 import 경로 재확인 필요")
print("- Streamlit 앱 실행: streamlit run main_app.py")
