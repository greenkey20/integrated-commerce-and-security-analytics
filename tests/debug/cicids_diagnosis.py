# 파일명: cicids_diagnosis.py (프로젝트 루트에 생성)

import pandas as pd
import os
import glob


def diagnose_cicids_files():
    """CICIDS2017 파일들 구조 진단"""

    data_dir = "/data/cicids2017"
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

    print(f"🔍 총 {len(csv_files)}개 파일 발견")

    results = {}

    for i, file_path in enumerate(csv_files[:3]):  # 처음 3개만 확인
        filename = os.path.basename(file_path)
        print(f"\n📁 [{i + 1}] {filename}")

        try:
            # 1. 인코딩 테스트
            encodings = ['utf-8', 'latin-1', 'iso-8859-1']
            working_encoding = None

            for encoding in encodings:
                try:
                    df_sample = pd.read_csv(file_path, nrows=5, encoding=encoding)
                    working_encoding = encoding
                    print(f"   ✅ 인코딩: {encoding}")
                    break
                except:
                    continue

            if not working_encoding:
                print("   ❌ 모든 인코딩 실패")
                continue

            # 2. 컬럼 구조 확인
            df_header = pd.read_csv(file_path, nrows=0, encoding=working_encoding)
            columns = [col.strip() for col in df_header.columns]

            print(f"   📊 총 컬럼 수: {len(columns)}")
            print(f"   🏷️ 마지막 컬럼: '{columns[-1]}'")

            # 3. 라벨 후보 찾기
            label_candidates = []
            for col in columns:
                if 'label' in col.lower() or 'class' in col.lower():
                    label_candidates.append(col)

            print(f"   🎯 라벨 후보: {label_candidates}")

            # 4. 실제 라벨 값들 확인
            df_sample = pd.read_csv(file_path, nrows=100, encoding=working_encoding)
            df_sample.columns = columns

            # 마지막 컬럼을 라벨로 가정
            label_col = label_candidates[0] if label_candidates else columns[-1]
            unique_labels = df_sample[label_col].unique()

            print(f"   📈 라벨 값들: {unique_labels[:5]}")
            print(f"   🔢 라벨 분포:")
            label_counts = df_sample[label_col].value_counts()
            for label, count in label_counts.head().items():
                print(f"      - {label}: {count}개")

            # 결과 저장
            results[filename] = {
                'encoding': working_encoding,
                'columns': len(columns),
                'label_column': label_col,
                'unique_labels': unique_labels.tolist(),
                'label_distribution': label_counts.to_dict()
            }

        except Exception as e:
            print(f"   ❌ 오류: {str(e)}")
            results[filename] = {'error': str(e)}

    return results


if __name__ == "__main__":
    results = diagnose_cicids_files()

    # 결과 요약
    print("\n" + "=" * 50)
    print("📋 진단 결과 요약")
    print("=" * 50)

    for filename, info in results.items():
        if 'error' not in info:
            print(f"\n✅ {filename}:")
            print(f"   - 인코딩: {info['encoding']}")
            print(f"   - 라벨 컬럼: {info['label_column']}")
            print(f"   - 고유 라벨 수: {len(info['unique_labels'])}")