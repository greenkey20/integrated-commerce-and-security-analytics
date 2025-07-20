# NumPy/TensorFlow 호환성 문제 해결 가이드

## 🚨 문제 상황
```
AttributeError: module 'numpy' has no attribute 'bool'
```

## 🔍 원인 분석
1. **NumPy 1.24+에서 `np.bool` 제거**
2. **구버전 pandas**가 제거된 API 사용
3. **TensorFlow와 NumPy 버전 충돌**

## ✅ 해결 방법 (우선순위 순)

### 방법 1: 자동 스크립트 실행 (권장)
```bash
cd /Users/greenpianorabbit/Documents/Development/customer-segmentation
python scripts/fix_numpy_compatibility.py
```

### 방법 2: 수동 패키지 업데이트
```bash
# Step 1: pandas/numpy 업그레이드
pip install --upgrade pandas>=2.0.3 numpy>=1.24.0

# Step 2: TensorFlow 재설치
pip install --upgrade tensorflow

# Step 3: 전체 의존성 재설치
pip install -r requirements.txt
```

### 방법 3: 가상환경 새로 생성 (최후 수단)
```bash
# conda 새 환경 생성
conda create -n customer-seg python=3.9
conda activate customer-seg

# 패키지 설치
pip install -r requirements.txt
```

## 🧪 설치 확인
```python
# Python에서 실행
import numpy as np
import pandas as pd
import tensorflow as tf

print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"TensorFlow: {tf.__version__}")

# bool 타입 테스트
test_bool = bool(True)  # 이게 에러 없이 실행되면 OK
print("✅ 호환성 문제 해결 완료!")
```

## 📋 각 단계별 예상 시간
- **방법 1**: 2-3분 (자동화)
- **방법 2**: 5-10분 (수동)
- **방법 3**: 15-20분 (새 환경)

## 🔄 재시작 절차
1. 터미널에서 `Ctrl+C`로 Streamlit 중지
2. 패키지 업데이트 실행
3. `streamlit run main_app.py` 재시작
4. 브라우저에서 http://localhost:8501 접속

## 💡 추가 팁
- **Anaconda 사용자**: `conda update --all` 먼저 실행
- **M1 Mac 사용자**: `pip install tensorflow-macos` 고려
- **문제 지속시**: Python 3.9로 업그레이드 검토
