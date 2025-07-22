#!/usr/bin/env python3
"""
프로젝트 구조 정리 후 통합 테스트

모든 변경사항이 제대로 적용되었는지 확인하는 종합 테스트
"""

import sys
import os

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("🧪 프로젝트 구조 정리 후 통합 테스트")
print("=" * 60)
print(f"📁 프로젝트 루트: {project_root}")
print()

def test_core_imports():
    """핵심 모듈 import 테스트"""
    print("1️⃣ 핵심 모듈 Import 테스트")
    print("-" * 30)
    
    tests = [
        # 데이터 계층
        ("from data.processor import DataProcessor", "DataProcessor (새 이름)"),
        ("from data.base import DataValidator, DataCleaner", "data.base 모듈들"),
        ("from data.loaders.retail_loader import RetailDataLoader", "RetailDataLoader"),
        
        # 리테일 계층 (새 이름들)
        ("from core.retail.retail_data_processor import RetailDataProcessor", "RetailDataProcessor (새 이름)"),
        ("from core.retail.retail_feature_engineer import RetailFeatureEngineer", "RetailFeatureEngineer (새 이름)"),
        
        # 통합 import
        ("from core.retail import RetailDataProcessor, RetailFeatureEngineer", "core.retail 통합 import"),
    ]
    
    passed = 0
    failed = 0
    
    for import_stmt, description in tests:
        try:
            exec(import_stmt)
            print(f"   ✅ {description}")
            passed += 1
        except Exception as e:
            print(f"   ❌ {description}: {str(e)}")
            failed += 1
    
    print(f"\n   📊 결과: {passed}개 성공, {failed}개 실패")
    return failed == 0

def test_web_pages():
    """웹 페이지 import 테스트"""
    print("\n2️⃣ 웹 페이지 Import 테스트")
    print("-" * 30)
    
    pages = [
        ("web.pages.segmentation.data_overview", "show_data_overview_page"),
        ("web.pages.retail.analysis", "show_retail_analysis_page"),
        ("web.pages.retail.data_cleaning", "show_data_cleaning_page"),
        ("web.pages.retail.feature_engineering", "show_feature_engineering_page"),
        ("web.pages.retail.target_creation", "show_target_creation_page"),
    ]
    
    passed = 0
    failed = 0
    
    for module_name, function_name in pages:
        try:
            module = __import__(module_name, fromlist=[function_name])
            getattr(module, function_name)
            print(f"   ✅ {module_name}.{function_name}")
            passed += 1
        except Exception as e:
            print(f"   ❌ {module_name}.{function_name}: {str(e)}")
            failed += 1
    
    print(f"\n   📊 결과: {passed}개 성공, {failed}개 실패")
    return failed == 0

def test_class_instantiation():
    """클래스 인스턴스 생성 테스트"""
    print("\n3️⃣ 클래스 인스턴스 생성 테스트")
    print("-" * 30)
    
    try:
        # DataProcessor 테스트
        from data.processors.segmentation_data_processor import DataProcessor
        processor = DataProcessor()
        print("   ✅ DataProcessor 인스턴스 생성 성공")
        
        # 기본 기능 테스트
        data = processor.load_data()
        print(f"   ✅ 데이터 로딩 성공: {len(data)}개 레코드")
        
        features = processor.get_feature_names()
        print(f"   ✅ 특성 이름 반환: {len(features)}개")
        
        passed = 3
        failed = 0
        
    except Exception as e:
        print(f"   ❌ DataProcessor 테스트 실패: {str(e)}")
        passed = 0
        failed = 1
    
    try:
        # RetailDataProcessor 테스트
        from data.processors.retail_data_processor import RetailDataProcessor
        
        column_mapping = {
            'invoice_no': 'InvoiceNo',
            'stock_code': 'StockCode',
            'description': 'Description',
            'quantity': 'Quantity',
            'invoice_date': 'InvoiceDate',
            'unit_price': 'UnitPrice',
            'customer_id': 'CustomerID'
        }
        
        retail_processor = RetailDataProcessor(column_mapping)
        print("   ✅ RetailDataProcessor 인스턴스 생성 성공")
        passed += 1
        
    except Exception as e:
        print(f"   ❌ RetailDataProcessor 테스트 실패: {str(e)}")
        failed += 1
    
    try:
        # RetailFeatureEngineer 테스트
        from data.processors.retail_feature_engineer import RetailFeatureEngineer
        
        feature_engineer = RetailFeatureEngineer(column_mapping)
        print("   ✅ RetailFeatureEngineer 인스턴스 생성 성공")
        passed += 1
        
    except Exception as e:
        print(f"   ❌ RetailFeatureEngineer 테스트 실패: {str(e)}")
        failed += 1
    
    print(f"\n   📊 결과: {passed}개 성공, {failed}개 실패")
    return failed == 0

def test_main_app():
    """메인 앱 import 테스트"""
    print("\n4️⃣ 메인 애플리케이션 테스트")
    print("-" * 30)
    
    try:
        import main_app
        print("   ✅ main_app.py import 성공")
        return True
    except Exception as e:
        print(f"   ❌ main_app.py import 실패: {str(e)}")
        return False

def test_file_structure():
    """파일 구조 확인 테스트"""
    print("\n5️⃣ 파일 구조 확인")
    print("-" * 30)
    
    expected_files = [
        "data/segmentation_data_processor.py",
        "core/retail/retail_data_processor.py", 
        "core/retail/retail_feature_engineer.py",
        "test/unit/simple_test.py",
        "test/integration/test_all_imports.py",
        "test/functional/test_streamlit.py",
        "test/debug/debug_imports.py",
        "notebooks/experiments/hyperparameter_tuning/hyperparameter_tuning_experiment.ipynb",
        "docs/backup/data_processor_backup.py",
        "docs/backup/retail_data_processor_backup.py",
        "docs/backup/retail_feature_engineer_backup.py",
    ]
    
    missing_files = [
        "data/_processor.py",
        "core/retail/segmentation_data_processor.py",
        "core/retail/feature_engineer.py",
        "debug_imports.py",
        "simple_test.py",
        "test_all_imports.py",
    ]
    
    # 존재해야 할 파일들 확인
    passed = 0
    failed = 0
    
    for file_path in expected_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
            passed += 1
        else:
            print(f"   ❌ {file_path} (없음)")
            failed += 1
    
    # 존재하지 않아야 할 파일들 확인
    for file_path in missing_files:
        if not os.path.exists(file_path):
            print(f"   ✅ {file_path} (정리됨)")
            passed += 1
        else:
            print(f"   ⚠️ {file_path} (아직 존재)")
            failed += 1
    
    print(f"\n   📊 결과: {passed}개 성공, {failed}개 실패")
    return failed == 0

def main():
    """메인 테스트 함수"""
    results = []
    
    # 모든 테스트 실행
    results.append(test_core_imports())
    results.append(test_web_pages())
    results.append(test_class_instantiation())
    results.append(test_main_app())
    results.append(test_file_structure())
    
    # 전체 결과 요약
    print("\n" + "=" * 60)
    print("📊 전체 테스트 결과 요약")
    print("=" * 60)
    
    passed_tests = sum(results)
    total_tests = len(results)
    
    print(f"✅ 성공한 테스트: {passed_tests}/{total_tests}")
    print(f"❌ 실패한 테스트: {total_tests - passed_tests}/{total_tests}")
    
    if all(results):
        print("\n🎉 모든 테스트 통과! 프로젝트 구조 정리가 성공적으로 완료되었습니다.")
        print("\n🚀 다음 단계:")
        print("1. streamlit run main_app.py 로 앱 실행 테스트")
        print("2. 각 페이지별 기능 정상 작동 확인")
        print("3. 데이터 로딩부터 모델링까지 전체 워크플로우 테스트")
        return True
    else:
        print("\n⚠️ 일부 테스트 실패. 추가 수정이 필요합니다.")
        print("\n🔧 권장 조치:")
        print("1. 실패한 import 경로 재확인")
        print("2. 누락된 파일 생성 또는 이동")
        print("3. 테스트 재실행")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
