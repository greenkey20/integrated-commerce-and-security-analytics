#!/usr/bin/env python3
"""
단계별 Import 테스트 - 문제 지점 정확히 찾기
"""

import sys
import os

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("=== 단계별 Import 테스트 ===\n")

# 1단계: config 모듈들 테스트
print("1. config 모듈 테스트:")
try:
    from config.settings import AppConfig
    print("   ✅ config.settings import 성공")
except Exception as e:
    print(f"   ❌ config.settings import 실패: {e}")
    exit(1)

try:
    from config.logging import setup_logger
    print("   ✅ config.logging import 성공")
except Exception as e:
    print(f"   ❌ config.logging import 실패: {e}")
    exit(1)

# 2단계: data.base 모듈들 개별 테스트
print("\n2. data.base 모듈 개별 테스트:")
try:
    from data.base.validator import DataValidator
    print("   ✅ DataValidator import 성공")
except Exception as e:
    print(f"   ❌ DataValidator import 실패: {e}")
    exit(1)

try:
    from data.base.cleaner import DataCleaner
    print("   ✅ DataCleaner import 성공")
except Exception as e:
    print(f"   ❌ DataCleaner import 실패: {e}")
    exit(1)

# 3단계: data.base 통합 import 테스트
print("\n3. data.base 통합 import 테스트:")
try:
    from data.base import DataValidator, DataCleaner
    print("   ✅ data.base 통합 import 성공")
except Exception as e:
    print(f"   ❌ data.base 통합 import 실패: {e}")
    exit(1)

# 4단계: retail_loader 개별 테스트
print("\n4. retail_loader 개별 테스트:")
try:
    from data.loaders.retail_loader import RetailDataLoader
    print("   ✅ RetailDataLoader import 성공")
except Exception as e:
    print(f"   ❌ RetailDataLoader import 실패: {e}")
    exit(1)

# 5단계: security_loader 개별 테스트 (문제가 여기일 가능성 높음)
print("\n5. security_loader 개별 테스트:")
try:
    from data.loaders.security_loader import SecurityDataLoader
    print("   ✅ SecurityDataLoader import 성공")
except Exception as e:
    print(f"   ❌ SecurityDataLoader import 실패: {e}")
    print(f"   오류 상세: {type(e).__name__}: {e}")
    
    # 여기서 멈추지 말고 계속 진행
    print("   ⚠️ SecurityDataLoader 건너뛰고 계속 진행...")

# 6단계: data.loaders 통합 테스트
print("\n6. data.loaders 통합 테스트:")
try:
    from data.loaders import RetailDataLoader
    print("   ✅ data.loaders 통합 import 성공")
except Exception as e:
    print(f"   ❌ data.loaders 통합 import 실패: {e}")
    print(f"   오류 상세: {type(e).__name__}: {e}")

print("\n=== 단계별 테스트 완료 ===")
print("문제가 발생한 모듈이 있다면 해당 모듈을 수정해야 합니다.")
