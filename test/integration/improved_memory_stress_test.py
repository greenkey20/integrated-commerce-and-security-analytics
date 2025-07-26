# 파일명: test/integration/improved_memory_test.py

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


def quick_capacity_test():
    """빠른 메모리 용량 테스트 (보수적 체크 완화)"""

    print("🚀 개선된 메모리 용량 테스트")
    print("=" * 50)

    # 시스템 메모리 정보
    total_memory = psutil.virtual_memory().total / 1024 / 1024 / 1024  # GB
    available_memory = psutil.virtual_memory().available / 1024 / 1024 / 1024

    print(f"💻 시스템 총 메모리: {total_memory:.1f} GB")
    print(f"💻 사용 가능 메모리: {available_memory:.1f} GB")

    # 테스트 샘플 수들 (더 공격적으로)
    test_samples = [100000, 200000, 500000, 1000000]

    data_dir = os.path.join(project_root, "data", "cicids2017")
    loader = WorkingCICIDSLoader(data_dir)

    max_successful = 0

    for target_samples in test_samples:
        print(f"\n🔍 테스트: {target_samples:,}개 샘플")

        try:
            start_memory = get_memory_usage()
            start_time = time.time()

            # 데이터 로드
            dataset = loader.load_working_files(target_samples=target_samples)

            load_memory = get_memory_usage()
            load_time = time.time() - start_time

            # 성공!
            max_successful = target_samples

            print(f"   ✅ 성공!")
            print(f"   ⏱️ 로딩 시간: {load_time:.1f}초")
            print(f"   💾 메모리 사용: {load_memory:.1f} MB")
            print(f"   💾 메모리 증가: {load_memory - start_memory:.1f} MB")

            # 라벨 분포 확인
            label_dist = dataset['Label'].value_counts()
            print(f"   🏷️ 라벨 분포:")
            for label, count in label_dist.head().items():
                pct = (count / len(dataset)) * 100
                print(f"      - {label}: {count:,}개 ({pct:.1f}%)")

            # 메모리 정리
            del dataset
            gc.collect()
            time.sleep(1)

            # 완화된 안전 체크 (90% 이상에서만 중단)
            current_usage = psutil.virtual_memory().percent
            print(f"   📊 시스템 메모리 사용률: {current_usage:.1f}%")

            if current_usage > 90:
                print(f"   ⚠️ 메모리 사용률 {current_usage:.1f}% - 중단")
                break

        except MemoryError:
            print(f"   ❌ 메모리 부족! 최대 처리량: {max_successful:,}개")
            break

        except Exception as e:
            print(f"   ❌ 오류: {str(e)[:100]}")
            break

    return max_successful


def test_balanced_loading():
    """균형잡힌 데이터 로딩 테스트"""
    print("\n🎯 균형잡힌 데이터 로딩 테스트")
    print("=" * 40)

    data_dir = os.path.join(project_root, "data", "cicids2017")
    loader = WorkingCICIDSLoader(data_dir)

    # 20만 개로 테스트 (4개 파일 균등 분배)
    target_samples = 200000

    try:
        start_time = time.time()
        dataset = loader.load_working_files(target_samples=target_samples)
        load_time = time.time() - start_time

        print(f"✅ {len(dataset):,}개 로드 성공 ({load_time:.1f}초)")

        # 상세 라벨 분포
        label_dist = dataset['Label'].value_counts()
        print(f"\n📊 상세 라벨 분포:")

        attack_count = 0
        for label, count in label_dist.items():
            pct = (count / len(dataset)) * 100
            print(f"   - {label}: {count:,}개 ({pct:.1f}%)")

            if label != 'BENIGN':
                attack_count += count

        attack_ratio = (attack_count / len(dataset)) * 100
        print(f"\n🎯 총 공격 비율: {attack_ratio:.1f}%")

        if attack_ratio > 20:
            print("✅ 균형잡힌 데이터 확보!")
            return dataset
        else:
            print("⚠️ 공격 데이터 부족")
            return None

    except Exception as e:
        print(f"❌ 균형 테스트 실패: {e}")
        return None


def estimate_training_capacity(dataset):
    """실제 모델 훈련 용량 추정"""
    print("\n🤖 실제 훈련 용량 추정")
    print("=" * 30)

    if dataset is None:
        return

    # 전처리
    numeric_columns = dataset.select_dtypes(include=[np.number]).columns
    X = dataset[numeric_columns].fillna(0)

    print(f"📊 특성 수: {len(numeric_columns)}")
    print(f"📊 샘플 수: {len(X)}")

    # 메모리 사용량 측정
    data_memory = X.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"💾 전처리 데이터 크기: {data_memory:.1f} MB")

    # TensorFlow 모델 메모리 추정 (경험적 공식)
    # 대략 데이터 크기의 3-5배가 모델 훈련 시 필요
    estimated_training_memory = data_memory * 4
    print(f"💾 예상 훈련 메모리: {estimated_training_memory:.1f} MB")

    # 시스템 여유 메모리와 비교
    available_mb = psutil.virtual_memory().available / 1024 / 1024
    print(f"💾 사용 가능 메모리: {available_mb:.1f} MB")

    if estimated_training_memory < available_mb * 0.7:  # 70% 마진
        print("✅ 현재 데이터로 안전한 훈련 가능!")

        # 확장 가능성 계산
        max_scale = (available_mb * 0.7) / data_memory
        max_samples = int(len(dataset) * max_scale)
        print(f"💡 이론적 최대 샘플 수: {max_samples:,}개")

    else:
        print("⚠️ 배치 처리 권장")

        # 권장 배치 크기
        safe_samples = int(len(dataset) * (available_mb * 0.7) / estimated_training_memory)
        print(f"💡 권장 샘플 수: {safe_samples:,}개")


if __name__ == "__main__":
    print("🚀 CICIDS2017 개선된 메모리 분석")
    print("=" * 60)

    # 1. 빠른 용량 테스트
    max_capacity = quick_capacity_test()
    print(f"\n🎯 최대 처리 가능: {max_capacity:,}개")

    # 2. 균형잡힌 데이터 테스트
    balanced_data = test_balanced_loading()

    # 3. 훈련 용량 추정
    estimate_training_capacity(balanced_data)

    # 최종 권장사항
    print("\n" + "=" * 60)
    print("🎯 최종 권장사항:")

    if max_capacity >= 500000:
        print("🔥 50만 개+ 처리 가능 - 대용량 실제 데이터 훈련 권장!")
        print("   → CICIDS2017로 overfitting 문제 확실히 해결 가능")
    elif max_capacity >= 200000:
        print("⚡ 20만 개+ 처리 가능 - 중대용량 실제 데이터 훈련 권장!")
        print("   → CICIDS2017 활용으로 성능 대폭 개선 기대")
    else:
        print("📊 RealisticSecurityDataGenerator 확장 권장")
        print("   → 50만 개 시뮬레이션 데이터로 대안 접근")

    if balanced_data is not None:
        print(f"\n✅ 균형잡힌 실제 데이터 {len(balanced_data):,}개 확보!")
        print("   → 즉시 모델 훈련 진행 가능")