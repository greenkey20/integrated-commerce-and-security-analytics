#!/usr/bin/env python3
"""
Phase 1-2 간단 Import 테스트 - 빠른 검증용

실제 클래스 로딩 없이 import 경로만 테스트
"""

import sys
import os

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("=== Phase 1-2 간단 Import 테스트 ===\n")

# 1단계: 파일 존재 확인
print("1. 핵심 파일 존재 확인:")
import os
from pathlib import Path

files_to_check = [
    "data/loaders/retail_loader.py",
    "data/loaders/__init__.py", 
    "data/base/__init__.py",
    "core/retail/analysis_manager.py"
]

for file_path in files_to_check:
    if Path(file_path).exists():
        print(f"   ✅ {file_path} 존재")
    else:
        print(f"   ❌ {file_path} 없음")

# 2단계: Python 구문 검증 (실제 import 없이)
print("\n2. Python 구문 검증:")

def check_syntax(file_path):
    """파일의 Python 구문이 올바른지 확인"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        compile(source, file_path, 'exec')
        return True, None
    except SyntaxError as e:
        return False, f"구문 오류: {e}"
    except Exception as e:
        return False, f"기타 오류: {e}"

syntax_files = [
    "data/loaders/retail_loader.py",
    "data/loaders/__init__.py",
    "core/retail/analysis_manager.py"
]

for file_path in syntax_files:
    if Path(file_path).exists():
        is_valid, error = check_syntax(file_path)
        if is_valid:
            print(f"   ✅ {file_path} 구문 정상")
        else:
            print(f"   ❌ {file_path} 구문 오류: {error}")

# 3단계: Import 경로 변경 확인
print("\n3. Import 경로 변경 확인:")

def check_import_path_fixed(file_path, old_import, new_import):
    """파일에서 import 경로가 제대로 변경되었는지 확인"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        has_old = old_import in content
        has_new = new_import in content
        
        if has_old:
            return False, f"아직 기존 import 사용: {old_import}"
        elif has_new:
            return True, f"새로운 import 정상: {new_import}"
        else:
            return True, "해당 import 없음 (정상)"
            
    except Exception as e:
        return False, f"파일 읽기 실패: {e}"

# core/retail/analysis_manager.py 확인
is_fixed, message = check_import_path_fixed(
    "core/retail/analysis_manager.py",
    "from .data_loader import RetailDataLoader",
    "from data.loaders.retail_loader import RetailDataLoader"
)

if is_fixed:
    print(f"   ✅ analysis_manager.py: {message}")
else:
    print(f"   ❌ analysis_manager.py: {message}")

# 4단계: 기본 모듈 구조 확인
print("\n4. 모듈 구조 확인:")

try:
    # 최소한의 import 시도 (heavy 라이브러리 없는 것들)
    import importlib.util
    
    # retail_loader 모듈 확인
    spec = importlib.util.spec_from_file_location(
        "retail_loader", 
        "data/loaders/retail_loader.py"
    )
    if spec and spec.loader:
        print("   ✅ retail_loader 모듈 구조 정상")
    else:
        print("   ❌ retail_loader 모듈 구조 문제")

except Exception as e:
    print(f"   ❌ 모듈 구조 확인 실패: {e}")

print("\n=== 간단 테스트 완료 ===")
print("\n결과 해석:")
print("- 모든 ✅가 나오면 Phase 1-2 완료!")
print("- ❌가 있으면 해당 파일 수정 필요")
print("- 다음 단계: Streamlit 앱 실행 테스트")

print("\n권장 다음 단계:")
print("1. 모든 테스트가 ✅면: streamlit run main_app.py")
print("2. ❌가 있으면: 해당 파일 수정 후 재테스트")
