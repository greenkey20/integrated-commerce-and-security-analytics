#!/bin/bash
# 환경 설정 및 의존성 설치 스크립트

echo "=== Customer Segmentation 프로젝트 환경 설정 ==="

# Python 버전 확인
echo "Python 버전 확인:"
python --version

# 가상환경 활성화 (Anaconda 사용자용)
echo "Anaconda 환경 확인 중..."
conda info --envs

# pip 업그레이드
echo "pip 업그레이드 중..."
python -m pip install --upgrade pip

# 호환성 문제 해결
echo "NumPy/Pandas 호환성 문제 해결 중..."
python scripts/fix_numpy_compatibility.py

# requirements.txt 기반 설치
echo "의존성 패키지 설치 중..."
pip install -r requirements.txt

# 설치 완료 확인
echo "=== 설치 확인 ==="
python -c "
import streamlit as st
import pandas as pd
import numpy as np
try:
    import tensorflow as tf
    print(f'✅ TensorFlow {tf.__version__} 설치 완료')
except ImportError:
    print('⚠️ TensorFlow 설치 필요')

print(f'✅ Streamlit {st.__version__}')
print(f'✅ Pandas {pd.__version__}')
print(f'✅ NumPy {np.__version__}')
print('🎉 모든 패키지 설치 완료!')
"

echo ""
echo "=== 실행 방법 ==="
echo "cd /Users/greenpianorabbit/Documents/Development/customer-segmentation"
echo "streamlit run main_app.py"
echo ""
echo "브라우저에서 http://localhost:8501 접속"
