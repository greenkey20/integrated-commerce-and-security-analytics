#!/usr/bin/env python3
"""
CICIDS2017 데이터셋 Label 컬럼 오류 해결 스크립트

실행 방법: python fix_cicids_label_issue.py
"""

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path

def fix_cicids_label_issue():
    """CICIDS2017 Label 오류 진단 및 해결"""
    
    print("🔍 CICIDS2017 Label 오류 진단 시작...")
    
    # 데이터 디렉토리 경로
    data_dir = "/Users/greenpianorabbit/Documents/Development/customer-segmentation/data/cicids2017"
    
    # CSV 파일들 찾기
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not csv_files:
        print("❌ CICIDS2017 CSV 파일을 찾을 수 없습니다!")
        print("💡 해결 방법:")
        print("1. https://www.unb.ca/cic/datasets/ids-2017.html 에서 MachineLearningCSV.zip 다운로드")
        print("2. 압축 해제 후 CSV 파일들을 data/cicids2017/ 폴더에 복사")
        return False
    
    print(f"✅ {len(csv_files)}개 CSV 파일 발견!")
    
    # 각 파일 진단
    for i, file_path in enumerate(csv_files[:3]):  # 처음 3개 파일만 확인
        print(f"\n📁 파일 {i+1}: {os.path.basename(file_path)}")
        
        try:
            # 파일 헤더만 읽기
            df_header = pd.read_csv(file_path, nrows=0)
            columns = df_header.columns.tolist()
            
            print(f"   📊 컬럼 수: {len(columns)}")
            
            # 마지막 컬럼 확인 (일반적으로 Label)
            last_column = columns[-1]
            print(f"   🏷️ 마지막 컬럼: '{last_column}'")
            
            # Label 관련 컬럼 찾기
            label_candidates = []
            for col in columns:
                col_clean = col.strip().lower()
                if 'label' in col_clean or 'class' in col_clean:
                    label_candidates.append(col)
            
            if label_candidates:
                print(f"   ✅ Label 후보: {label_candidates}")
                
                # 실제 데이터 샘플 확인
                df_sample = pd.read_csv(file_path, nrows=5)
                for candidate in label_candidates:
                    unique_values = df_sample[candidate].unique()
                    print(f"      - '{candidate}': {unique_values}")
                    
            else:
                print("   ❌ Label 컬럼을 찾을 수 없습니다!")
                print(f"   📋 모든 컬럼: {columns[-5:]}...")  # 마지막 5개 컬럼만 표시
                
        except Exception as e:
            print(f"   ❌ 파일 읽기 오류: {str(e)}")
            
            # 파일 인코딩 문제 해결 시도
            try:
                print("   🔄 다른 인코딩으로 재시도...")
                df_header = pd.read_csv(file_path, nrows=0, encoding='latin-1')
                print(f"   ✅ latin-1 인코딩으로 성공: {len(df_header.columns)}개 컬럼")
            except:
                print("   ❌ 인코딩 문제로 읽을 수 없습니다")
    
    return True


def create_fixed_security_analysis():
    """수정된 security_analysis.py 생성"""
    
    print("\n🛠️ 수정된 security_analysis.py 생성 중...")
    
    fixed_functions = '''
def load_cicids_with_robust_label_detection(file_path, max_rows=10000):
    """강화된 Label 컬럼 감지로 CICIDS2017 데이터 로드"""
    
    encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings_to_try:
        try:
            print(f"📝 {encoding} 인코딩으로 시도 중...")
            
            # 헤더 먼저 읽기
            df_header = pd.read_csv(file_path, nrows=0, encoding=encoding)
            columns = df_header.columns.tolist()
            
            print(f"✅ 총 {len(columns)}개 컬럼 발견")
            
            # 컬럼명 정리 (공백 제거)
            original_columns = columns.copy()
            cleaned_columns = [col.strip() for col in columns]
            
            # Label 컬럼 찾기 (더 강화된 방법)
            label_column = find_label_column_robust(cleaned_columns)
            
            if label_column is None:
                # 마지막 컬럼을 Label로 가정
                label_column = cleaned_columns[-1]
                print(f"⚠️ Label 컬럼을 찾지 못해 마지막 컬럼 '{label_column}'을 사용합니다")
            
            # 실제 데이터 로드
            df = pd.read_csv(file_path, nrows=max_rows, encoding=encoding)
            df.columns = cleaned_columns  # 정리된 컬럼명 적용
            
            # Label 컬럼을 'Label'로 표준화
            if label_column != 'Label':
                df = df.rename(columns={label_column: 'Label'})
                print(f"🔄 '{label_column}' → 'Label'로 컬럼명 변경")
            
            # 데이터 검증
            if 'Label' in df.columns:
                unique_labels = df['Label'].unique()
                print(f"🏷️ 발견된 라벨들: {unique_labels[:5]}..." if len(unique_labels) > 5 else f"🏷️ 발견된 라벨들: {unique_labels}")
                
                # BENIGN이 있는지 확인
                if any('BENIGN' in str(label).upper() for label in unique_labels):
                    print("✅ 정상적인 CICIDS2017 데이터로 확인됨")
                    return df
                else:
                    print("⚠️ BENIGN 라벨이 없어 일반적이지 않은 데이터일 수 있음")
                    return df
            
            break
            
        except Exception as e:
            print(f"❌ {encoding} 인코딩 실패: {str(e)}")
            continue
    
    raise ValueError("모든 인코딩 방법으로 파일을 읽을 수 없습니다")


def find_label_column_robust(columns):
    """강화된 Label 컬럼 찾기"""
    
    # 1순위: 정확한 일치
    exact_matches = ['Label', 'label', 'LABEL', 'Class', 'class', 'CLASS']
    for match in exact_matches:
        if match in columns:
            return match
    
    # 2순위: 부분 일치 (대소문자 무시)
    for col in columns:
        col_lower = col.lower()
        if 'label' in col_lower or 'class' in col_lower:
            return col
    
    # 3순위: 마지막 컬럼 (일반적으로 target)
    if columns:
        last_col = columns[-1]
        print(f"💡 마지막 컬럼 '{last_col}'을 Label로 사용할 예정")
        return last_col
    
    return None


# 기존 load_and_analyze_cicids_data 함수 수정
def load_and_analyze_cicids_data_fixed(file_paths):
    """수정된 CICIDS2017 데이터 로드 및 분석"""
    try:
        # 첫 번째 파일로 테스트
        first_file = file_paths[0]
        print(f"📁 강화된 방법으로 파일 로드 시도: {os.path.basename(first_file)}")
        
        sample_df = load_cicids_with_robust_label_detection(first_file, max_rows=10000)
        
        print(f"✅ 데이터 로드 성공: {len(sample_df)}개 샘플, {len(sample_df.columns)}개 컬럼")
        
        # 세션에 저장
        st.session_state.cicids_data = sample_df
        
        # 데이터 품질 체크
        check_data_quality(sample_df)
        
        print("🎉 수정된 방법으로 CICIDS2017 데이터 로드 완료!")
        
    except Exception as e:
        print(f"❌ 수정된 방법으로도 실패: {str(e)}")
        print("🔧 샘플 데이터를 대신 생성합니다...")
        
        # 샘플 데이터 생성
        sample_data = generate_cicids_sample_data()
        st.session_state.cicids_data = sample_data
        print("✅ 샘플 데이터 생성 완료!")
'''
    
    # 수정된 함수들을 별도 파일로 저장
    with open('/Users/greenpianorabbit/Documents/Development/customer-segmentation/cicids_label_fix.py', 'w', encoding='utf-8') as f:
        f.write(fixed_functions)
    
    print("✅ 수정된 함수들을 cicids_label_fix.py에 저장했습니다")


if __name__ == "__main__":
    print("🚀 CICIDS2017 Label 오류 해결 스크립트")
    print("=" * 50)
    
    # 1. 문제 진단
    if fix_cicids_label_issue():
        # 2. 수정된 코드 생성
        create_fixed_security_analysis()
        
        print("\n" + "=" * 50)
        print("✅ 진단 및 수정 완료!")
        print("\n💡 다음 단계:")
        print("1. Streamlit 앱을 재시작하세요")
        print("2. '보안 이상 탐지 분석' → '데이터 다운로드 및 로드' 선택")
        print("3. '데이터 로드 및 기본 분석 시작' 버튼 클릭")
        print("\n🔗 문제가 지속되면 샘플 데이터로 시연하기를 선택하세요")
    
    print("\n🎯 추가 도움이 필요하면 언제든 문의하세요!")
