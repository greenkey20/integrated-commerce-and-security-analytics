#!/usr/bin/env python3
"""
최종 import 테스트 스크립트

Phase 1-2 완료 후 모든 import가 정상적으로 작동하는지 확인
"""

import sys
import traceback

def test_imports():
    """모든 주요 import 테스트"""
    print("🧪 Final Import Test - Phase 1-2 완료 검증")
    print("=" * 50)
    
    tests = [
        # 1. 데이터 계층 테스트
        ("data.processors.data_processor", "DataProcessor"),
        ("data.loaders.retail_loader", "RetailDataLoader"),
        ("data.loaders.security_loader", "SecurityDataLoader"),
        
        # 2. 세분화 페이지 테스트  
        ("web.pages.segmentation.pca_analysis", "show_pca_analysis_page"),
        ("web.pages.segmentation.customer_prediction", "show_customer_prediction_page"),
        ("web.pages.segmentation.deep_learning_analysis", "show_deep_learning_analysis_page"),
        ("web.pages.segmentation.marketing_strategy", "show_marketing_strategy_page"),
        
        # 3. 리테일 페이지 테스트
        ("web.pages.retail.analysis", "show_retail_analysis_page"),
        ("web.pages.retail.data_loading", "show_data_loading_page"),
        ("web.pages.retail.data_cleaning", "show_data_cleaning_page"),
        
        # 4. 클러스터링 모듈 테스트
        ("core.segmentation.clustering", "ClusterAnalyzer"),
        
        # 5. 메인 앱 페이지 import 테스트
        ("web.pages", "__all__"),
    ]
    
    passed = 0
    failed = 0
    
    for module_name, class_or_func in tests:
        try:
            if class_or_func == "__all__":
                module = __import__(module_name, fromlist=[''])
                print(f"✅ {module_name} - 전체 모듈 import 성공")
            else:
                module = __import__(module_name, fromlist=[class_or_func])
                getattr(module, class_or_func)
                print(f"✅ {module_name}.{class_or_func} - import 성공")
            passed += 1
        except Exception as e:
            print(f"❌ {module_name}.{class_or_func} - import 실패: {str(e)}")
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 테스트 결과: {passed}개 성공, {failed}개 실패")
    
    if failed == 0:
        print("🎉 모든 import 테스트 통과! 앱 실행 준비 완료!")
        return True
    else:
        print("❌ 일부 import 실패. 추가 수정이 필요합니다.")
        return False

def test_data_loading():
    """데이터 로딩 기능 테스트"""
    print("\n🔍 DataProcessor 기능 테스트")
    print("-" * 30)
    
    try:
        from data.processors.data_processor import DataProcessor
        
        processor = DataProcessor()
        data = processor.load_data()
        
        print(f"✅ 데이터 로딩 성공: {len(data)}개 레코드")
        print(f"✅ 특성 컬럼: {processor.get_feature_names()}")
        print(f"✅ 숫자형 컬럼: {processor.get_numeric_columns()}")
        
        return True
    except Exception as e:
        print(f"❌ 데이터 로딩 실패: {str(e)}")
        traceback.print_exc()
        return False

def test_streamlit_imports():
    """Streamlit 관련 import 테스트"""
    print("\n📱 Streamlit 페이지 import 테스트")
    print("-" * 30)
    
    try:
        # main_app.py의 주요 import들 테스트
        from web.pages.segmentation.data_overview import show_data_overview_page
        from web.pages.segmentation.exploratory_analysis import show_exploratory_analysis_page
        from web.pages.segmentation.clustering_analysis import show_clustering_analysis_page
        from web.pages.segmentation.pca_analysis import show_pca_analysis_page
        from web.pages.segmentation.deep_learning_analysis import show_deep_learning_analysis_page
        from web.pages.segmentation.customer_prediction import show_customer_prediction_page
        from web.pages.segmentation.marketing_strategy import show_marketing_strategy_page
        from web.pages.retail.analysis import show_retail_analysis_page
        
        print("✅ 모든 Streamlit 페이지 import 성공")
        return True
    except Exception as e:
        print(f"❌ Streamlit 페이지 import 실패: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Phase 1-2 완료 검증 시작")
    print("Customer Segmentation 프로젝트 리팩토링 검증")
    print("=" * 60)
    
    # 테스트 실행
    import_success = test_imports()
    data_success = test_data_loading()
    streamlit_success = test_streamlit_imports()
    
    print("\n" + "=" * 60)
    print("🎯 최종 결과")
    print(f"Import 테스트: {'✅ 통과' if import_success else '❌ 실패'}")
    print(f"데이터 로딩: {'✅ 통과' if data_success else '❌ 실패'}")
    print(f"Streamlit 페이지: {'✅ 통과' if streamlit_success else '❌ 실패'}")
    
    if import_success and data_success and streamlit_success:
        print("\n🎉 Phase 1-2 완전 성공!")
        print("🚀 이제 'streamlit run main_app.py'로 앱을 실행하세요!")
        sys.exit(0)
    else:
        print("\n❌ 추가 수정이 필요합니다.")
        sys.exit(1)
