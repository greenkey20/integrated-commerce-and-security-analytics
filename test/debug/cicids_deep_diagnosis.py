# 파일명: cicids_fixed_diagnosis.py

import pandas as pd
import os


def safe_diagnosis():
    """안전한 방식으로 CICIDS2017 파일 진단"""

    data_dir = "/data/cicids2017"

    problem_files = [
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
    ]

    for filename in problem_files:
        file_path = os.path.join(data_dir, filename)
        print(f"\n🔍 안전 진단: {filename}")

        try:
            # 1. 먼저 헤더만 확인
            df_header = pd.read_csv(file_path, nrows=0, encoding='utf-8')
            columns = [col.strip() for col in df_header.columns]

            print(f"   📋 전체 컬럼 수: {len(columns)}")
            print(f"   📋 마지막 5개 컬럼: {columns[-5:]}")

            # 2. Label 컬럼 찾기 (더 안전하게)
            label_column = None
            for col in columns:
                if 'label' in col.lower():
                    label_column = col
                    break

            if not label_column:
                label_column = columns[-1]  # 마지막 컬럼을 라벨로 가정

            print(f"   🎯 사용할 라벨 컬럼: '{label_column}'")

            # 3. 안전한 방식으로 샘플 데이터 확인
            df_sample = pd.read_csv(file_path, nrows=1000, encoding='utf-8')
            df_sample.columns = columns  # 정리된 컬럼명 적용

            print(f"   📊 샘플 1000개 라벨 분포:")
            try:
                sample_labels = df_sample[label_column].value_counts()
                for label, count in sample_labels.items():
                    print(f"      - '{label}': {count}개")
            except Exception as e:
                print(f"      ❌ 라벨 분포 확인 실패: {e}")
                print(f"      📝 {label_column} 컬럼 샘플 값들:")
                print(f"         {df_sample[label_column].head().tolist()}")

            # 4. 파일 크기만 확인 (메모리 절약)
            total_lines = sum(1 for line in open(file_path, 'r', encoding='utf-8'))
            print(f"   📊 전체 행 수: {total_lines - 1:,}개 (헤더 제외)")

            # 5. 중간과 끝 부분도 확인
            print(f"   🔍 다른 위치 샘플링:")

            # 중간 부분 (skiprows 사용)
            middle_start = (total_lines - 1) // 2
            df_middle = pd.read_csv(file_path, nrows=100, skiprows=range(1, middle_start), encoding='utf-8')
            df_middle.columns = columns
            middle_labels = df_middle[label_column].value_counts()
            print(f"      📍 중간 부분 ({middle_start}행 근처):")
            for label, count in middle_labels.items():
                print(f"         - '{label}': {count}개")

            # 끝 부분 확인 (tail 효과)
            end_start = max(1, total_lines - 1000)
            df_end = pd.read_csv(file_path, nrows=500, skiprows=range(1, end_start), encoding='utf-8')
            df_end.columns = columns
            end_labels = df_end[label_column].value_counts()
            print(f"      📍 끝 부분 ({end_start}행 이후):")
            for label, count in end_labels.items():
                print(f"         - '{label}': {count}개")

        except Exception as e:
            print(f"   💥 전체 오류: {str(e)}")
            print(f"   🔧 기본 정보만 확인:")

            try:
                # 최소한 파일 읽기 가능한지 확인
                df_tiny = pd.read_csv(file_path, nrows=5, encoding='utf-8')
                print(f"      ✅ 파일 읽기 가능")
                print(f"      📋 컬럼들: {list(df_tiny.columns)}")
                print(f"      📄 첫 5행:")
                print(df_tiny.to_string())
            except Exception as inner_e:
                print(f"      ❌ 파일 읽기도 실패: {inner_e}")


def quick_all_files_check():
    """모든 파일 빠른 체크"""
    print("\n" + "=" * 60)
    print("🚀 전체 CICIDS2017 파일 빠른 체크")
    print("=" * 60)

    data_dir = "/data/cicids2017"
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

    for filename in csv_files:
        file_path = os.path.join(data_dir, filename)
        try:
            df_sample = pd.read_csv(file_path, nrows=100, encoding='utf-8')
            last_col = df_sample.columns[-1].strip()
            unique_values = df_sample[last_col].unique()

            print(f"\n📁 {filename}:")
            print(f"   🏷️ 라벨 컬럼: '{last_col}'")
            print(f"   📊 발견된 값들: {unique_values[:3]}...")

            # BENIGN 이외의 값이 있는지 체크
            non_benign = [v for v in unique_values if 'BENIGN' not in str(v).upper()]
            if non_benign:
                print(f"   ✅ 공격 데이터 발견: {non_benign[:2]}...")
            else:
                print(f"   ❌ BENIGN만 발견")

        except Exception as e:
            print(f"\n📁 {filename}: ❌ 오류 {str(e)[:50]}...")


if __name__ == "__main__":
    safe_diagnosis()
    quick_all_files_check()