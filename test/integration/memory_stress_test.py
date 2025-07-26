# 파일명: test/integration/memory_stress_test.py

import sys
import os

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import psutil
import pandas as pd
import numpy as np
import time
import gc
from data.loaders.cicids_working_files_loader import WorkingCICIDSLoader


def get_memory_usage():
    """현재 메모리 사용량 반환 (MB)"""
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # MB 단위


def test_memory_scaling():
    """점진적으로 샘플 수를 늘려가며 메모리 테스트"""

    print("🧪 메모리 사용량 테스트 시작")
    print("=" * 50)

    # 시스템 메모리 정보
    total_memory = psutil.virtual_memory().total / 1024 / 1024 / 1024  # GB
    available_memory = psutil.virtual_memory().available / 1024 / 1024 / 1024

    print(f"💻 시스템 총 메모리: {total_memory:.1f} GB")
    print(f"💻 사용 가능 메모리: {available_memory:.1f} GB")
    print()

    # 테스트 샘플 수들 (점진적 증가)
    test_samples = [50000, 100000, 200000, 300000, 500000, 1000000]

    # 프로젝트 루트 기준으로 경로 설정
    data_dir = os.path.join(project_root, "data", "cicids2017")
    loader = WorkingCICIDSLoader(data_dir)

    results = []

    for target_samples in test_samples:
        print(f"🔍 테스트: {target_samples:,}개 샘플")

        try:
            # 시작 메모리 측정
            start_memory = get_memory_usage()
            start_time = time.time()

            # 데이터 로드
            dataset = loader.load_working_files(target_samples=target_samples)

            # 로드 후 메모리 측정
            load_memory = get_memory_usage()
            load_time = time.time() - start_time

            # 데이터 정보
            data_size_mb = dataset.memory_usage(deep=True).sum() / 1024 / 1024

            # 결과 기록
            result = {
                'samples': target_samples,
                'load_time': load_time,
                'start_memory_mb': start_memory,
                'peak_memory_mb': load_memory,
                'memory_increase_mb': load_memory - start_memory,
                'data_size_mb': data_size_mb,
                'success': True
            }

            results.append(result)

            print(f"   ✅ 성공!")
            print(f"   ⏱️ 로딩 시간: {load_time:.1f}초")
            print(f"   📊 데이터 크기: {data_size_mb:.1f} MB")
            print(f"   💾 메모리 증가: {load_memory - start_memory:.1f} MB")
            print(f"   💾 총 메모리 사용: {load_memory:.1f} MB")

            # 메모리 정리
            del dataset
            gc.collect()
            time.sleep(2)  # 메모리 정리 대기

            # 안전 체크: 메모리 사용률이 80% 넘으면 중단
            current_usage = psutil.virtual_memory().percent
            if current_usage > 80:
                print(f"   ⚠️ 메모리 사용률 {current_usage:.1f}% - 안전을 위해 테스트 중단")
                break

        except MemoryError:
            print(f"   ❌ 메모리 부족!")
            result = {
                'samples': target_samples,
                'success': False,
                'error': 'MemoryError'
            }
            results.append(result)
            break

        except Exception as e:
            print(f"   ❌ 오류: {str(e)[:50]}...")
            result = {
                'samples': target_samples,
                'success': False,
                'error': str(e)[:100]
            }
            results.append(result)

        print()

    # 결과 요약
    print("📋 메모리 테스트 결과 요약")
    print("=" * 50)

    successful_tests = [r for r in results if r.get('success', False)]

    if successful_tests:
        max_successful = max(successful_tests, key=lambda x: x['samples'])
        print(f"✅ 최대 성공 샘플 수: {max_successful['samples']:,}개")
        print(f"⏱️ 최대 로딩 시간: {max_successful['load_time']:.1f}초")
        print(f"💾 최대 메모리 사용: {max_successful['peak_memory_mb']:.1f} MB")

        # 추천 샘플 수 (안전 마진 고려)
        recommended_samples = max_successful['samples'] * 0.8  # 80% 안전 마진
        print(f"💡 권장 샘플 수: {recommended_samples:,.0f}개 (안전 마진 포함)")

    else:
        print("❌ 모든 테스트 실패")

    return results


def test_model_training_memory():
    """모델 훈련 시 메모리 사용량 테스트"""
    print("\n🤖 모델 훈련 메모리 테스트")
    print("=" * 30)

    # 중간 크기 데이터로 훈련 메모리 테스트
    data_dir = os.path.join(project_root, "data", "cicids2017")
    loader = WorkingCICIDSLoader(data_dir)

    # 10만 개 샘플로 테스트
    dataset = loader.load_working_files(target_samples=100000)

    print(f"📊 훈련 데이터: {len(dataset):,}개")

    # 메모리 측정
    before_training = get_memory_usage()

    # 간단한 전처리 (실제 훈련 시뮬레이션)
    print("🔄 전처리 중...")

    # 수치 컬럼만 선택 (Label 제외)
    numeric_columns = dataset.select_dtypes(include=[np.number]).columns
    X = dataset[numeric_columns].fillna(0)  # 결측치 처리

    after_preprocessing = get_memory_usage()

    print(f"💾 전처리 후 메모리: {after_preprocessing - before_training:.1f} MB 증가")

    # 모델 훈련 시뮬레이션 (실제 TensorFlow 없이)
    print("🧠 모델 훈련 시뮬레이션...")

    # 배치 처리 시뮬레이션
    batch_size = 1000
    batches = len(X) // batch_size

    max_memory = before_training

    for i in range(min(10, batches)):  # 처음 10개 배치만 테스트
        batch_data = X.iloc[i * batch_size:(i + 1) * batch_size]
        current_memory = get_memory_usage()
        max_memory = max(max_memory, current_memory)

        if i % 5 == 0:
            print(f"   배치 {i + 1}: {current_memory:.1f} MB")

    print(f"💾 훈련 시 최대 메모리: {max_memory:.1f} MB")
    print(f"💾 총 메모리 증가: {max_memory - before_training:.1f} MB")

    # 정리
    del dataset, X
    gc.collect()


if __name__ == "__main__":
    print("🚀 CICIDS2017 메모리 사용량 종합 테스트")
    print(f"📍 프로젝트 루트: {project_root}")
    print("=" * 60)

    # 1. 데이터 로딩 메모리 테스트
    loading_results = test_memory_scaling()

    # 2. 모델 훈련 메모리 테스트
    test_model_training_memory()

    print("\n🎯 최종 권장사항:")
    successful_tests = [r for r in loading_results if r.get('success', False)]

    if successful_tests:
        max_samples = max(successful_tests, key=lambda x: x['samples'])['samples']

        if max_samples >= 500000:
            print("✅ 50만 개 샘플 처리 가능 - 대용량 훈련 권장")
        elif max_samples >= 200000:
            print("✅ 20만 개 샘플 처리 가능 - 중대용량 훈련 권장")
        elif max_samples >= 100000:
            print("⚡ 10만 개 샘플 처리 가능 - 적정 규모 훈련 권장")
        else:
            print("⚠️ 시뮬레이션 데이터 사용 권장")

    print("\n💡 메모리가 부족하면 배치 처리나 샘플링을 고려하세요!")