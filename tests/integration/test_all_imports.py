#!/usr/bin/env python3
"""
전체 프로젝트 import 테스트 스크립트
리팩토링 후 모든 import가 정상 작동하는지 확인

실행 방법:
cd /Users/greenpianorabbit/Documents/Development/customer-segmentation
python test_all_imports.py
"""

import sys
import os
import traceback

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_import(module_name, description):
    """개별 import 테스트"""
    print(f"테스트 중: {description}")
    try:
        exec(f"import {module_name}")
        print(f"✅ {module_name} - 성공")
        return True
    except Exception as e:
        print(f"❌ {module_name} - 실패: {str(e)}")
        if "ModuleNotFoundError" in str(e):
            print(f"   📁 경로 문제 가능성: {str(e)}")
        return False

def test_specific_import(import_statement, description):
    """특정 import 문 테스트"""
    print(f"테스트 중: {description}")
    try:
        exec(import_statement)
        print(f"✅ {description} - 성공")
        return True
    except Exception as e:
        print(f"❌ {description} - 실패: {str(e)}")
        print(f"   🔍 Import 문: {import_statement}")
        return False

def main():
    """메인 테스트 함수"""
    print("=" * 60)
    print("🧪 Customer Segmentation 프로젝트 Import 테스트")
    print("=" * 60)
    
    print(f"📁 프로젝트 루트: {project_root}")
    print(f"🐍 Python 경로: {sys.path[:3]}...")  # 처음 3개만 표시
    print()
    
    # 테스트 결과 추적
    total_tests = 0
    passed_tests = 0
    failed_imports = []
    
    # 1. 기본 패키지 테스트
    print("1️⃣ 기본 패키지 테스트")
    print("-" * 30)
    
    basic_imports = [
        ("config.settings", "설정 모듈"),
        ("utils.font_manager", "폰트 관리자"),
        ("data", "데이터 패키지"),
        ("data.base", "데이터 기본 클래스"),
        ("data.loaders", "데이터 로더들"),
    ]
    
    for module, desc in basic_imports:
        total_tests += 1
        if test_import(module, desc):
            passed_tests += 1
        else:
            failed_imports.append((module, desc))
    
    print()
    
    # 2. 새로운 데이터 로더 테스트
    print("2️⃣ 새로운 데이터 로더 테스트")
    print("-" * 30)
    
    loader_imports = [
        ("from data.loaders.retail_loader import RetailDataLoader", "리테일 데이터 로더"),
        ("from data.base.validator import DataValidator", "데이터 검증기"),
        ("from data.base.cleaner import DataCleaner", "데이터 정제기"),
    ]
    
    for import_stmt, desc in loader_imports:
        total_tests += 1
        if test_specific_import(import_stmt, desc):
            passed_tests += 1
        else:
            failed_imports.append((import_stmt, desc))
    
    print()
    
    # 3. 핵심 모듈 테스트
    print("3️⃣ 핵심 모듈 테스트")
    print("-" * 30)
    
    core_imports = [
        ("core.retail", "리테일 분석 패키지"),
        ("from core.retail import RetailDataLoader", "리테일 데이터 로더 (새 경로)"),
        ("from core.retail.analysis_manager import RetailAnalysisManager", "분석 매니저"),
        ("core.security", "보안 분석 패키지"),
    ]
    
    for import_stmt, desc in core_imports:
        total_tests += 1
        if test_specific_import(import_stmt, desc):
            passed_tests += 1
        else:
            failed_imports.append((import_stmt, desc))
    
    print()
    
    # 4. 웹 페이지 모듈 테스트
    print("4️⃣ 웹 페이지 모듈 테스트")
    print("-" * 30)
    
    web_imports = [
        ("web.pages.retail.analysis", "리테일 분석 페이지"),
        ("web.pages.retail.data_loading", "데이터 로딩 페이지"),
        ("from web.pages.retail.analysis import show_retail_analysis_page", "리테일 분석 함수"),
    ]
    
    for import_stmt, desc in web_imports:
        total_tests += 1
        if test_specific_import(import_stmt, desc):
            passed_tests += 1
        else:
            failed_imports.append((import_stmt, desc))
    
    print()
    
    # 5. 메인 앱 테스트
    print("5️⃣ 메인 애플리케이션 테스트")
    print("-" * 30)
    
    try:
        print("테스트 중: main_app.py 전체 import")
        import main_app
        print("✅ main_app.py - 성공")
        total_tests += 1
        passed_tests += 1
    except Exception as e:
        print(f"❌ main_app.py - 실패: {str(e)}")
        print("   🔍 상세 오류:")
        traceback.print_exc()
        total_tests += 1
        failed_imports.append(("main_app.py", str(e)))
    
    print()
    
    # 결과 요약
    print("=" * 60)
    print("📊 테스트 결과 요약")
    print("=" * 60)
    
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"✅ 성공: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
    print(f"❌ 실패: {len(failed_imports)}/{total_tests}")
    
    if failed_imports:
        print("\n🚨 실패한 Import 목록:")
        for i, (module, desc) in enumerate(failed_imports, 1):
            print(f"   {i}. {desc}")
            print(f"      Module: {module}")
    
    print()
    
    if passed_tests == total_tests:
        print("🎉 모든 Import 테스트 통과! 리팩토링이 성공적으로 완료되었습니다.")
        return True
    else:
        print("⚠️ 일부 Import 테스트 실패. 위의 오류를 확인하고 수정이 필요합니다.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
