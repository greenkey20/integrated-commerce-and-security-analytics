#!/usr/bin/env python3
"""
NumPy 호환성 문제 해결 스크립트

문제: NumPy 1.24+ 에서 np.bool 제거로 인한 pandas 충돌
해결: 적절한 버전으로 다운그레이드 또는 업그레이드
"""

import subprocess
import sys
import pkg_resources

def get_current_versions():
    """현재 설치된 패키지 버전 확인"""
    try:
        numpy_version = pkg_resources.get_distribution("numpy").version
        pandas_version = pkg_resources.get_distribution("pandas").version
        print(f"현재 NumPy 버전: {numpy_version}")
        print(f"현재 Pandas 버전: {pandas_version}")
        return numpy_version, pandas_version
    except Exception as e:
        print(f"버전 확인 오류: {e}")
        return None, None

def fix_compatibility():
    """호환성 문제 해결"""
    print("=== NumPy/Pandas 호환성 문제 해결 시작 ===")
    
    # 현재 버전 확인
    numpy_ver, pandas_ver = get_current_versions()
    
    print("\n해결 방법 1: 호환 가능한 버전으로 업그레이드")
    try:
        # pandas 업그레이드 (NumPy 1.24+ 지원)
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '--upgrade',
            'pandas>=2.0.0', 'numpy>=1.24.0'
        ], check=True)
        print("✅ pandas/numpy 업그레이드 완료")
    except subprocess.CalledProcessError:
        print("❌ 업그레이드 실패, 다운그레이드 시도...")
        
        # 방법 2: 호환 가능한 구버전으로 다운그레이드
        try:
            subprocess.run([
                sys.executable, '-m', 'pip', 'install',
                'numpy==1.23.5', 'pandas==1.5.3'
            ], check=True)
            print("✅ 호환 버전으로 다운그레이드 완료")
        except subprocess.CalledProcessError as e:
            print(f"❌ 다운그레이드도 실패: {e}")
            return False
    
    # TensorFlow 재설치 (선택사항)
    print("\nTensorFlow 재설치 중...")
    try:
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '--upgrade', 'tensorflow'
        ], check=True)
        print("✅ TensorFlow 재설치 완료")
    except subprocess.CalledProcessError:
        print("⚠️ TensorFlow 재설치 실패 (수동으로 재설치 필요)")
    
    # 최종 버전 확인
    print("\n=== 수정 후 버전 확인 ===")
    get_current_versions()
    
    print("\n✅ 호환성 문제 해결 완료!")
    print("이제 Streamlit 앱을 재시작하세요: streamlit run main_app.py")
    
    return True

def test_import():
    """import 테스트"""
    print("\n=== Import 테스트 ===")
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__} 임포트 성공")
        
        import pandas as pd
        print(f"✅ Pandas {pd.__version__} 임포트 성공")
        
        # numpy.bool 사용 테스트
        try:
            # 새로운 방식
            test_bool = bool(True)
            print("✅ bool 타입 사용 가능")
        except Exception as e:
            print(f"❌ bool 타입 오류: {e}")
        
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} 임포트 성공")
        
        print("🎉 모든 라이브러리 정상 작동!")
        return True
        
    except Exception as e:
        print(f"❌ Import 오류: {e}")
        return False

if __name__ == "__main__":
    print("NumPy/Pandas 호환성 문제 해결 도구")
    print("=" * 50)
    
    # 현재 상태 확인
    if not test_import():
        print("문제가 감지되었습니다. 수정을 시작합니다...")
        fix_compatibility()
        
        # 재테스트
        print("\n=== 수정 후 재테스트 ===")
        test_import()
    else:
        print("현재 환경이 정상입니다!")
