#!/usr/bin/env python3
"""
CICIDS2017 데이터셋 Label 컬럼 오류 즉시 해결 스크립트
실행: python quick_label_fix.py
"""

import pandas as pd
import numpy as np
import os
import glob

def quick_fix_cicids_label():
    """CICIDS2017 Label 오류 즉시 해결"""
    
    print("🔧 CICIDS2017 Label 오류 즉시 해결 시작...")
    
    # 데이터 디렉토리
    data_dir = "/Users/greenpianorabbit/Documents/Development/customer-segmentation/data/cicids2017"
    
    # CSV 파일들 찾기
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not csv_files:
        print("❌ CICIDS2017 CSV 파일을 찾을 수 없습니다!")
        return False
    
    print(f"✅ {len(csv_files)}개 CSV 파일 발견!")
    
    # 첫 번째 파일로 테스트
    test_file = csv_files[0]
    print(f"\n📁 테스트 파일: {os.path.basename(test_file)}")
    
    # 여러 인코딩으로 시도
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            print(f"🔄 {encoding} 인코딩 시도...")
            
            # 헤더만 읽기
            df_header = pd.read_csv(test_file, nrows=0, encoding=encoding)
            columns = df_header.columns.tolist()
            
            print(f"✅ {encoding}으로 성공! 총 {len(columns)}개 컬럼")
            
            # 컬럼명 정리
            original_columns = columns.copy()
            cleaned_columns = [col.strip() for col in columns]
            
            print(f"📋 마지막 5개 컬럼: {cleaned_columns[-5:]}")
            
            # Label 컬럼 찾기
            label_column = find_label_column_advanced(cleaned_columns)
            
            if label_column:
                print(f"🎯 Label 컬럼 발견: '{label_column}'")
                
                # 실제 데이터 샘플 로드
                df_sample = pd.read_csv(test_file, nrows=10, encoding=encoding)
                df_sample.columns = cleaned_columns
                
                # Label 데이터 확인
                if label_column in df_sample.columns:
                    unique_labels = df_sample[label_column].unique()
                    print(f"🏷️ 발견된 라벨: {unique_labels}")
                    
                    # BENIGN 확인
                    if any('BENIGN' in str(label).upper() for label in unique_labels):
                        print("✅ 정상적인 CICIDS2017 데이터 확인!")
                    else:
                        print("⚠️ BENIGN 라벨이 없어 다른 형태의 데이터일 수 있음")
                    
                    return {
                        'success': True,
                        'encoding': encoding,
                        'label_column': label_column,
                        'file_path': test_file,
                        'total_columns': len(cleaned_columns),
                        'sample_labels': unique_labels.tolist()
                    }
                
            else:
                print(f"❌ Label 컬럼을 찾을 수 없음")
                print(f"📝 사용 가능한 컬럼들: {cleaned_columns[-10:]}")
                
        except Exception as e:
            print(f"❌ {encoding} 실패: {str(e)[:100]}...")
            continue
    
    print("❌ 모든 인코딩 방법 실패")
    return False


def find_label_column_advanced(columns):
    """고급 Label 컬럼 감지"""
    
    # 1순위: 정확한 일치
    exact_matches = [
        'Label', 'label', 'LABEL',
        'Class', 'class', 'CLASS',
        'Target', 'target', 'TARGET'
    ]
    
    for match in exact_matches:
        if match in columns:
            return match
    
    # 2순위: 부분 일치 (공백 포함)
    for col in columns:
        col_clean = col.strip().lower()
        if col_clean in ['label', 'class', 'target']:
            return col
    
    # 3순위: 포함 검사
    for col in columns:
        col_lower = col.lower()
        if 'label' in col_lower or 'class' in col_lower:
            return col
    
    # 4순위: 마지막 컬럼 (일반적으로 타겟)
    if columns:
        return columns[-1]
    
    return None


def create_fixed_loader():
    """수정된 데이터 로더 생성"""
    
    print("\n🛠️ 수정된 데이터 로더 생성 중...")
    
    fixed_code = '''
# security_analysis.py에 추가할 수정된 함수

def load_cicids_with_error_handling(file_path, max_rows=10000):
    """오류 처리가 강화된 CICIDS 데이터 로더"""
    
    print(f"📁 파일 로드 시도: {os.path.basename(file_path)}")
    
    # 다양한 인코딩 시도
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            print(f"🔄 {encoding} 인코딩 시도...")
            
            # 헤더 먼저 확인
            df_header = pd.read_csv(file_path, nrows=0, encoding=encoding)
            columns = df_header.columns.tolist()
            
            # 컬럼명 정리 (앞뒤 공백 제거)
            cleaned_columns = [col.strip() for col in columns]
            
            # Label 컬럼 찾기
            label_column = None
            
            # 정확한 일치 우선
            for candidate in ['Label', 'label', 'LABEL', 'Class', 'class']:
                if candidate in cleaned_columns:
                    label_column = candidate
                    break
            
            # 부분 일치 검사
            if not label_column:
                for col in cleaned_columns:
                    if 'label' in col.lower() or 'class' in col.lower():
                        label_column = col
                        break
            
            # 마지막 컬럼 사용
            if not label_column and cleaned_columns:
                label_column = cleaned_columns[-1]
                print(f"⚠️ 마지막 컬럼 '{label_column}'을 Label로 사용")
            
            # 실제 데이터 로드
            df = pd.read_csv(file_path, nrows=max_rows, encoding=encoding)
            df.columns = cleaned_columns
            
            # Label 컬럼을 'Label'로 표준화
            if label_column and label_column != 'Label':
                df = df.rename(columns={label_column: 'Label'})
                print(f"🔄 '{label_column}' → 'Label'로 변경")
            
            # 데이터 정리
            if 'Label' in df.columns:
                # 무한대값 처리
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
                
                # 음수값을 절댓값으로 변경
                df[numeric_cols] = df[numeric_cols].abs()
                
                print(f"✅ 데이터 로드 성공: {len(df)}개 레코드, {len(df.columns)}개 컬럼")
                print(f"🏷️ 라벨 종류: {df['Label'].unique()[:5]}...")
                
                return df
            
        except Exception as e:
            print(f"❌ {encoding} 실패: {str(e)[:100]}...")
            continue
    
    raise ValueError("모든 방법으로 데이터를 로드할 수 없습니다")


# load_and_analyze_cicids_data 함수 수정
def load_and_analyze_cicids_data_fixed(file_paths):
    """수정된 CICIDS 데이터 로드 함수"""
    try:
        # 첫 번째 파일로 시도
        sample_df = load_cicids_with_error_handling(file_paths[0], max_rows=10000)
        
        # 세션에 저장
        st.session_state.cicids_data = sample_df
        
        # 기본 분석
        st.success(f"✅ CICIDS2017 데이터 로드 완료!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("총 레코드", f"{len(sample_df):,}")
        with col2:
            if 'Label' in sample_df.columns:
                st.metric("라벨 종류", f"{sample_df['Label'].nunique()}")
        with col3:
            numeric_cols = sample_df.select_dtypes(include=[np.number]).columns
            st.metric("수치 특성", f"{len(numeric_cols)}")
        
        return True
        
    except Exception as e:
        st.error(f"❌ 데이터 로드 실패: {str(e)}")
        
        # 폴백: 샘플 데이터 생성
        st.info("🔧 샘플 데이터로 대체합니다...")
        sample_data = generate_cicids_sample_data()
        st.session_state.cicids_data = sample_data
        st.success("✅ 샘플 데이터 생성 완료!")
        
        return False
'''
    
    # 파일로 저장
    with open('/Users/greenpianorabbit/Documents/Development/customer-segmentation/cicids_error_fix.py', 'w', encoding='utf-8') as f:
        f.write(fixed_code)
    
    print("✅ 수정된 코드를 cicids_error_fix.py에 저장했습니다")


if __name__ == "__main__":
    print("🚀 CICIDS2017 Label 오류 즉시 해결")
    print("=" * 50)
    
    # 문제 진단 및 해결
    result = quick_fix_cicids_label()
    
    if result:
        print(f"\n✅ 문제 해결 완료!")
        print(f"📊 해결 정보:")
        print(f"  - 인코딩: {result['encoding']}")
        print(f"  - Label 컬럼: {result['label_column']}")
        print(f"  - 총 컬럼 수: {result['total_columns']}")
        print(f"  - 샘플 라벨: {result['sample_labels']}")
        
        # 수정된 로더 생성
        create_fixed_loader()
        
        print(f"\n💡 다음 단계:")
        print(f"1. Streamlit 앱 재시작")
        print(f"2. '보안 이상 탐지 분석' 선택")
        print(f"3. '데이터 다운로드 및 로드' → '데이터 로드 및 기본 분석 시작' 클릭")
        
    else:
        print(f"\n❌ 자동 해결 실패")
        print(f"💡 대안:")
        print(f"1. 샘플 데이터로 시연하기 선택")
        print(f"2. CICIDS2017 데이터셋 재다운로드")
        print(f"3. 수동으로 Label 컬럼 확인")
    
    print(f"\n🎯 추가 도움이 필요하면 문의하세요!")
