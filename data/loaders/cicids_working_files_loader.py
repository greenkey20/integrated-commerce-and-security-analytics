# 파일명: cicids_working_files_loader.py
# 위치: data/loaders/cicids_working_files_loader.py

import pandas as pd
import os
from typing import List, Dict


class WorkingCICIDSLoader:
    """작동하는 CICIDS2017 파일만 활용하는 로더"""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

        # 파일별 공격 데이터 위치 정보 (진단 결과 기반)
        self.file_info = {
            # 기존 (유지)
            'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv': {
                'attack_start': 100000,
                'expected_labels': ['PortScan', 'BENIGN']
            },
            
            # 신규 추가 - TODO에서 확인된 파일들
            'Tuesday-WorkingHours.pcap_ISCX.csv': {
                'attack_start': 50000,
                'expected_labels': ['FTP-Patator', 'BENIGN']
            },
            'Wednesday-workingHours.pcap_ISCX.csv': {
                'attack_start': 50000,
                'expected_labels': ['DoS slowloris', 'BENIGN']
            },
            'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv': {
                'attack_start': 50000,
                'expected_labels': ['Web Attack – Brute Force', 'BENIGN']  # 실제 라벨명 사용
            },
            'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv': {
                'attack_start': 50000,
                'expected_labels': ['DDoS', 'BENIGN']
            },
            'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv': {
                'attack_start': 100000,  # Infiltration은 더 뒤에 위치할 수 있음
                'expected_labels': ['Infiltration', 'BENIGN']
            }
        }

    def load_working_files(self, target_samples: int = 300000) -> pd.DataFrame:
        """작동하는 파일들만 로드"""

        all_data = []

        for filename, info in self.file_info.items():
            file_path = os.path.join(self.data_dir, filename)

            if not os.path.exists(file_path):
                print(f"⚠️ 파일 없음: {filename}")
                continue

            print(f"📁 로딩: {filename}")

            # 공격 데이터가 있는 부분부터 로드
            df = pd.read_csv(
                file_path,
                skiprows=range(1, info['attack_start']),  # 앞부분 스킵
                nrows=target_samples // len(self.file_info),
                encoding='utf-8'
            )

            # 컬럼명 정리
            df.columns = [col.strip() for col in df.columns]

            print(f"   📊 로드된 데이터: {len(df)}개")
            print(f"   🏷️ 라벨 분포: {df['Label'].value_counts().to_dict()}")

            all_data.append(df)

        if not all_data:
            raise ValueError("사용 가능한 CICIDS2017 파일이 없습니다")

        # 데이터 결합
        combined_df = pd.concat(all_data, ignore_index=True)

        # 섞기
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"\n✅ 최종 결합 데이터:")
        print(f"   📊 총 샘플 수: {len(combined_df)}")
        print(f"   🏷️ 라벨 분포:")
        for label, count in combined_df['Label'].value_counts().items():
            percentage = (count / len(combined_df)) * 100
            print(f"      - {label}: {count:,}개 ({percentage:.1f}%)")

        return combined_df


def quick_validate_more_files():
    """다른 파일들도 빠르게 검증"""

    data_dir = "C:/keydev/integrated-commerce-and-security-analytics/data/cicids2017"

    candidates = [
        'Monday-WorkingHours.pcap_ISCX.csv',
        'Tuesday-WorkingHours.pcap_ISCX.csv',
        'Wednesday-workingHours.pcap_ISCX.csv',
        'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv'
    ]

    working_files = []

    for filename in candidates:
        file_path = os.path.join(data_dir, filename)

        if not os.path.exists(file_path):
            continue

        print(f"\n🔍 검증: {filename}")

        try:
            # 여러 위치에서 샘플링
            positions = [0, 50000, 100000, 150000, 200000]
            found_attacks = False

            for pos in positions:
                try:
                    if pos == 0:
                        df_sample = pd.read_csv(file_path, nrows=100, encoding='utf-8')
                    else:
                        df_sample = pd.read_csv(
                            file_path,
                            nrows=100,
                            skiprows=range(1, pos),
                            encoding='utf-8'
                        )

                    # 컬럼명 정리
                    df_sample.columns = [col.strip() for col in df_sample.columns]

                    # BENIGN 이외 라벨 찾기
                    labels = df_sample['Label'].unique()
                    non_benign = [l for l in labels if 'BENIGN' not in str(l).upper()]

                    if non_benign:
                        print(f"   ✅ 위치 {pos}: {non_benign} 발견!")
                        found_attacks = True
                        break

                except Exception as e:
                    continue  # 해당 위치에서 읽기 실패시 다음 위치 시도

            if found_attacks:
                working_files.append(filename)
                print(f"   🎯 {filename} → 사용 가능!")
            else:
                print(f"   ❌ {filename} → BENIGN만 있음")

        except Exception as e:
            print(f"   💥 {filename} → 오류: {str(e)[:50]}")

    return working_files


if __name__ == "__main__":
    print("🚀 CICIDS2017 작동 파일 검증")
    print("=" * 50)

    # 1. 추가 파일들 검증
    working_files = quick_validate_more_files()

    print(f"\n📋 사용 가능한 파일들: {working_files}")

    # 2. 작동하는 파일로 데이터 로드 테스트
    try:
        loader = WorkingCICIDSLoader("C:/keydev/integrated-commerce-and-security-analytics/data/cicids2017")
        
        # 이미 file_info에 모든 파일이 설정되어 있으므로 추가 설정 불필요
        
        # 데이터 로드 (더 많은 샘플)
        dataset = loader.load_working_files(target_samples=300000)
        print("\n🎉 CICIDS2017 부분 활용 성공!")
        
    except Exception as e:
        print(f"\n❌ 로드 실패: {e}")
        print("💡 RealisticSecurityDataGenerator 확장을 권장합니다")