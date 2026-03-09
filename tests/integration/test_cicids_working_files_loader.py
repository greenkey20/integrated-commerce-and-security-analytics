#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CICIDS2017 확장된 로더 테스트 스크립트
"""

import sys
import os

# 프로젝트 루트를 Python path에 추가 (2단계 위로)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from data.loaders.cicids_working_files_loader import WorkingCICIDSLoader

def test_expanded_loader():
    """확장된 로더 테스트"""
    
    print("🚀 CICIDS2017 확장된 로더 테스트")
    print("=" * 50)
    
    # 로더 초기화
    data_dir = "C:/keydev/integrated-commerce-and-security-analytics/data/cicids2017"
    loader = WorkingCICIDSLoader(data_dir)
    
    print(f"📁 데이터 디렉토리: {data_dir}")
    print(f"📊 등록된 파일 수: {len(loader.file_info)}")
    print("\n📋 등록된 파일 목록:")
    
    for i, (filename, info) in enumerate(loader.file_info.items(), 1):
        print(f"   {i}. {filename}")
        print(f"      - 공격 시작: {info['attack_start']:,}")
        print(f"      - 예상 라벨: {info['expected_labels']}")
    
    # 파일 존재 여부 확인
    print(f"\n🔍 파일 존재 여부 확인:")
    missing_files = []
    existing_files = []
    
    for filename in loader.file_info.keys():
        file_path = os.path.join(data_dir, filename)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"   ✅ {filename} ({file_size:.1f} MB)")
            existing_files.append(filename)
        else:
            print(f"   ❌ {filename} (파일 없음)")
            missing_files.append(filename)
    
    if missing_files:
        print(f"\n⚠️ 누락된 파일: {len(missing_files)}개")
        return False
    
    print(f"\n✅ 모든 파일 확인 완료: {len(existing_files)}개")
    
    # 작은 샘플로 데이터 로드 테스트
    print(f"\n🧪 작은 샘플 로드 테스트 (파일당 1000개씩)")
    
    try:
        # 작은 샘플만 로드
        dataset = loader.load_working_files(target_samples=6000)  # 파일당 1000개
        
        print(f"\n🎉 로드 성공!")
        print(f"   📊 총 샘플 수: {len(dataset):,}개")
        print(f"   🏷️ 고유 라벨 수: {dataset['Label'].nunique()}개")
        print(f"   📋 라벨 목록: {list(dataset['Label'].unique())}")
        
        # 라벨별 분포
        print(f"\n📈 라벨별 분포:")
        label_counts = dataset['Label'].value_counts()
        for label, count in label_counts.items():
            percentage = (count / len(dataset)) * 100
            print(f"   - {label}: {count:,}개 ({percentage:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ 로드 실패: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_expanded_loader()
    
    if success:
        print(f"\n🎊 테스트 성공! 6개 공격 유형 확보!")
        print(f"   💡 다음 단계: web/pages/security/security_analysis_page.py 업데이트")
    else:
        print(f"\n💥 테스트 실패! 문제를 해결해야 합니다.")
