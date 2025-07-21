"""
모듈 임포트 테스트 스크립트

새로 생성된 모든 모듈들이 정상적으로 임포트되는지 확인
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/../src"))

def test_imports():
    """모든 모듈 임포트 테스트"""
    
    print("🔧 모듈 임포트 테스트 시작...")
    
    try:
        # 설정 모듈 테스트
        print("📋 Config 모듈 테스트...")
        from config.settings import AppConfig, UIConfig, ClusteringConfig
        print("✅ Config 모듈 임포트 성공")
        
        # 유틸리티 모듈 테스트
        print("🛠️ Utils 모듈 테스트...")
        from utils.font_manager import FontManager
        print("✅ Utils 모듈 임포트 성공")
        
        # 코어 모듈 테스트
        print("🔧 Core 모듈 테스트...")
        from segmentation.data_processing import DataProcessor
        from segmentation.clustering import ClusterAnalyzer
        from segmentation.models import DeepLearningModels
        print("✅ Core 모듈 임포트 성공")
        
        # 페이지 모듈 테스트
        print("📄 Pages 모듈 테스트...")
        from segmentation.data_overview import show_data_overview_page
        from segmentation.exploratory_analysis import show_exploratory_analysis_page
        from segmentation.clustering_analysis import show_clustering_analysis_page
        from segmentation.pca_analysis import show_pca_analysis_page
        from segmentation.deep_learning_analysis import show_deep_learning_analysis_page
        from segmentation.customer_prediction import show_customer_prediction_page
        from segmentation.marketing_strategy import show_marketing_strategy_page
        print("✅ Pages 모듈 임포트 성공")
        
        print("\n🎉 모든 모듈 임포트 테스트 성공!")
        print("✅ 새로운 모듈화 구조가 정상적으로 작동합니다.")
        
        return True
        
    except ImportError as e:
        print(f"❌ 임포트 오류: {e}")
        return False
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        return False


def test_basic_functionality():
    """기본 기능 테스트"""
    
    print("\n🔬 기본 기능 테스트 시작...")
    
    try:
        # 데이터 프로세서 테스트
        from segmentation.data_processing import DataProcessor
        processor = DataProcessor()
        print("✅ DataProcessor 인스턴스 생성 성공")
        
        # 클러스터 분석기 테스트
        from segmentation.clustering import ClusterAnalyzer
        analyzer = ClusterAnalyzer()
        print("✅ ClusterAnalyzer 인스턴스 생성 성공")
        
        # 폰트 매니저 테스트
        from utils.font_manager import FontManager
        font_manager = FontManager()
        print("✅ FontManager 인스턴스 생성 성공")
        
        print("✅ 모든 기본 기능 테스트 성공!")
        return True
        
    except Exception as e:
        print(f"❌ 기본 기능 테스트 실패: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("🧪 Customer Segmentation App - 모듈 테스트")
    print("=" * 50)
    
    # 현재 디렉토리 표시
    import os
    print(f"📁 현재 작업 디렉토리: {os.getcwd()}")
    
    # 테스트 실행
    import_success = test_imports()
    function_success = test_basic_functionality()
    
    # 최종 결과
    print("\n" + "=" * 50)
    if import_success and function_success:
        print("🎉 모든 테스트 통과! 애플리케이션을 실행할 준비가 완료되었습니다.")
        print("🚀 다음 명령어로 앱을 실행하세요:")
        print("   streamlit run app.py")
    else:
        print("❌ 테스트 실패. 오류를 확인하고 수정해주세요.")
    print("=" * 50)
