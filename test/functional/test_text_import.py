# test_text_import.py
# 프로젝트 루트에서 실행하세요.

import sys
import os

# 안전을 위해 현재 작업 디렉토리를 프로젝트 루트로 보장
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__) + '/../../')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print("PYTHONPATH (head):", sys.path[:3])

try:
    # Text Analytics 모듈만 import (지연로딩이 적용된 상태에서는 TF가 바로 로드되지 않아야 함)
    from core.text import TextAnalyticsModels
    print("Imported: core.text ->", TextAnalyticsModels)
except Exception as e:
    print("❌ Import 실패:", repr(e))
    raise

try:
    model_instance = TextAnalyticsModels()
    print("✅ TextAnalyticsModels 인스턴스 생성 성공")
except ImportError as ie:
    print("❌ ImportError 발생:", ie)
    print("설명: 다른 도메인에 의존성이 있어 import 중 오류가 발생했습니다.")
except Exception as ex:
    print("⚠️ 인스턴스 생성 중 일반 예외 발생:", repr(ex))
else:
    print("테스트 완료 — TensorFlow는 모델 생성 함수 호출 전까지 로드되지 않아야 합니다.")

