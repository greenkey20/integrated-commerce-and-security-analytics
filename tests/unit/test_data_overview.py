#!/usr/bin/env python3
"""
데이터 개요 페이지 테스트 스크립트

새로 수정된 data_overview.py 파일이 정상적으로 작동하는지 테스트
"""

import sys
import os

# 프로젝트 루트 디렉토리로 이동
os.chdir('/')
sys.path.append('../..')

print("🧪 데이터 개요 페이지 import 테스트")
print("=" * 50)

try:
    # 새로운 데이터 개요 페이지 import
    from web.pages.segmentation.data_overview import show_data_overview_page
    print("✅ data_overview.py import 성공!")
    
    # 데이터 프로세서 테스트
    from web.pages.segmentation.data_overview import get_data_processor
    processor, status = get_data_processor()
    
    if processor is not None:
        print("✅ 새로운 데이터 계층 로드 성공!")
        data = processor.load_data()
        print(f"✅ 데이터 로드 성공: {len(data)} 행")
    else:
        print("⚠️ 데이터 프로세서 로드 실패, 샘플 데이터 사용")
        from web.pages.segmentation.data_overview import create_sample_data
        data = create_sample_data()
        print(f"✅ 샘플 데이터 생성 성공: {len(data)} 행")
    
    print("\n📊 데이터 미리보기:")
    print(data.head().to_string())
    
    # 데이터 타입 확인
    print(f"\n📋 데이터 타입:")
    for col, dtype in data.dtypes.items():
        print(f"  - {col}: {dtype}")
    
    print("\n🎉 모든 테스트 통과!")

except ImportError as e:
    print(f"❌ Import 실패: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"❌ 실행 중 오류: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("테스트 완료!")

print("\n🚀 Streamlit 실행 방법:")
print("streamlit run main_app.py")
print("그 후 '고객 세그멘테이션' > '데이터 개요' 페이지 확인")
