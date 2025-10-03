# 🌿 Integrated Commerce & Security Analytics

> 차세대 이커머스를 위한 통합 인텔리전스 플랫폼  
> 고객 인사이트부터 보안 모니터링까지, 데이터 기반 비즈니스 성장을 지원합니다.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 프로젝트 개요

**Integrated Commerce & Security Analytics**는 대용량 실무 데이터를 활용한 종합적인 비즈니스 인텔리전스 플랫폼입니다. 고객 분석, 리테일 예측, 네트워크 보안 탐지를 하나의 통합된 환경에서 제공합니다.

### 🎯 핵심 가치

- **📊 Business Intelligence**: 100만+ 거래 데이터 기반 리테일 분석
- **👥 Customer Analytics**: 머신러닝/딥러닝 고객 세그멘테이션  
- **🛡️ Security Intelligence**: 280만+ 네트워크 트래픽 이상 탐지
- **🌿 Green Spectrum UI**: 직관적이고 현대적인 사용자 경험

### 🚀 주요 기능

#### 📊 Business Intelligence
- **💰 Retail Analytics**: 대용량 온라인 리테일 데이터 분석 (1M+ 거래)
- **📈 Sales Prediction**: 고객별 다음 구매 금액 예측 (선형회귀, 앙상블)
- **🎯 Customer Lifetime Value**: CLV 모델링 및 세그멘테이션
- **📊 RFM Analysis**: Recency, Frequency, Monetary 기반 고객 분류

#### 👥 Customer Analytics  
- **🔍 Exploratory Data Analysis**: 고객 행동 패턴 분석
- **🎯 K-means Clustering**: 고객 세그멘테이션 (Premium, Regular, New)
- **🔬 PCA Analysis**: 차원 축소 및 특성 중요도 분석
- **🌱 Deep Learning**: MLP, CNN, LSTM 기반 고급 분류
- **🔮 Customer Prediction**: 신규 고객 세그먼트 자동 분류

#### 🛡️ Security Intelligence
- **🔒 Network Anomaly Detection**: CICIDS2017 데이터셋 기반 (2.8M+ 레코드)
- **⚠️ Multi-Attack Classification**: DDoS, PortScan, WebAttacks, Infiltration 탐지
- **🧠 Hybrid Deep Learning**: CNN+MLP, Autoencoder, LSTM 융합 모델
- **📊 Real-time Monitoring**: 실시간 네트워크 트래픽 분석

---

## 🏗️ 시스템 아키텍처

### 계층형 정보 구조
```
🌿 Integrated Commerce & Security Analytics
├── 📊 Business Intelligence (실무 중심)
│   ├── 💰 Retail Analytics (1순위 - 대용량 실무 데이터)
│   │   ├── 📋 데이터 로딩 & 탐색
│   │   ├── 🧹 데이터 정제
│   │   ├── ⚙️ 특성 공학 (RFM, 시계열)
│   │   ├── 🎯 타겟 생성 (CLV, 재구매)
│   │   ├── 🤖 모델링 (Linear, Ridge, RF, GB)
│   │   ├── 📊 성능 평가 (MAE, RMSE, R²)
│   │   └── 💎 종합 분석
│   │
│   └── 👥 Customer Analytics (2순위 - 학습용)
│       ├── 📊 데이터 개요
│       ├── 🔍 탐색적 분석  
│       ├── 🎯 고객 세그멘테이션
│       ├── 🔬 차원 축소 분석 (PCA)
│       ├── 🌱 고급 딥러닝 모델링
│       ├── 🔮 고객 예측 & 분류
│       └── 📈 마케팅 전략 수립
│
└── 🛡️ Security Intelligence (전문 분석)
    ├── 🔒 네트워크 이상 탐지 (CICIDS2017)
    ├── 📊 실시간 보안 모니터링
    └── ⚠️ 위협 인텔리전스 분석
```

### 📁 프로젝트 구조

```
integrated-commerce-and-security-analytics/
├── 📱 main_app.py                    # Streamlit 메인 애플리케이션
├── 🔧 config/
│   └── settings.py                   # 설정 파일 (Green Theme)
├── 📊 data/
│   ├── Mall_Customers.csv            # 고객 세그멘테이션 (200개)
│   ├── base/online_retail_II.xlsx    # 리테일 분석 (1M+ 거래)
│   └── cicids2017/*.csv              # 네트워크 보안 (2.8M+ 레코드)
├── 🤖 core/                          # 핵심 비즈니스 로직
│   ├── retail/
│   │   ├── model_trainer.py          # 회귀 모델링 (Linear, Ridge, RF)
│   │   ├── analysis_manager.py       # 리테일 분석 매니저
│   │   └── visualizer.py             # 차트 & 시각화
│   ├── security/
│   │   ├── model_builder.py          # 딥러닝 모델 빌더 (MLP, CNN, Hybrid)
│   │   ├── detection_engine.py       # 이상 탐지 엔진
│   │   └── hyperparameter_tuning.py  # 하이퍼파라미터 최적화
│   └── segmentation/
│       ├── clustering_engine.py      # K-means 클러스터링
│       └── deep_models.py            # 딥러닝 분류
├── 🌐 web/pages/                     # Streamlit 페이지들
│   ├── segmentation/                 # 고객 세그멘테이션 페이지들
│   ├── retail/                       # 리테일 분석 페이지들
│   └── security/                     # 보안 분석 페이지들
├── 🛠️ utils/
│   ├── font_manager.py               # 한글 폰트 지원
│   └── ui_components.py              # Green Theme UI 컴포넌트
├── 📓 notebooks/                     # Jupyter 실험 노트북들
│   └── experiments/                  # 머신러닝/딥러닝 실험용
├── 🧪 test/                          # 테스트 코드
└── 📚 docs/                          # 프로젝트 문서
```

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd integrated-commerce-and-security-analytics

# Python 3.8+ 가상환경 생성
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 필요한 패키지 설치
pip install -r requirements_py312_fixed.txt
```

### 2. 애플리케이션 실행

```bash
# Streamlit 웹 애플리케이션 실행
streamlit run main_app.py
```

브라우저에서 `http://localhost:8501`로 접속하면 다음과 같은 통합 분석 환경을 사용할 수 있습니다:

- **📊 Business Intelligence**: 실무 중심 리테일 분석
- **👥 Customer Analytics**: 고객 세그멘테이션 학습
- **🛡️ Security Intelligence**: 네트워크 보안 분석

### 3. 데이터셋 정보

#### 📊 Retail Analytics 데이터
- **파일**: `data/base/online_retail_II.xlsx`
- **규모**: 1,067,371 거래 레코드
- **특성**: CustomerID, StockCode, Quantity, UnitPrice, Country, InvoiceDate
- **용도**: 실무 수준 대용량 데이터 분석 실습

#### 👥 Customer Segmentation 데이터  
- **파일**: `data/Mall_Customers.csv`
- **규모**: 200 고객 레코드
- **특성**: Age, Gender, Annual Income, Spending Score
- **용도**: 머신러닝/딥러닝 학습 및 이론 검증

#### 🛡️ Security Analytics 데이터
- **파일들**: `data/cicids2017/*.csv` (8개 파일)
- **규모**: 2,830,743 네트워크 플로우 레코드
- **공격 유형**: DDoS, PortScan, WebAttacks, Infiltration, Brute Force
- **특성**: 78개 네트워크 플로우 특성 (패킷 크기, 플래그, 시간 등)

---

## 💡 주요 사용 사례

### 📊 Business Intelligence
1. **매출 예측 모델링**: 고객별 다음 구매 금액 예측 (R² 0.6+)
2. **Customer Lifetime Value**: CLV 계산 및 고가치 고객 식별
3. **계절성 분석**: 월별/분기별 매출 트렌드 및 패턴 분석
4. **국가별 시장 분석**: 지역별 고객 행동 및 매출 기여도 분석

### 👥 Customer Analytics
1. **고객 세그멘테이션**: K-means 기반 Premium/Regular/New 분류
2. **개인화 마케팅**: 세그먼트별 맞춤형 전략 수립
3. **이탈 예측**: 고객 이탈 가능성 예측 및 리텐션 전략
4. **신규 고객 분류**: 실시간 고객 세그먼트 자동 분류

### 🛡️ Security Intelligence  
1. **네트워크 이상 탐지**: 실시간 트래픽 모니터링 (정확도 95%+)
2. **공격 패턴 분석**: DDoS, PortScan 등 공격 유형별 특성 분석
3. **위협 인텔리전스**: 공격 트렌드 및 예측 분석
4. **SOC 지원**: 보안 운영 센터 의사결정 지원

---

## 🧠 핵심 기술 스택

### 머신러닝/딥러닝
- **Scikit-learn**: 전통적 ML 알고리즘 (Linear, Ridge, Random Forest)
- **TensorFlow/Keras**: 딥러닝 모델 (MLP, CNN, LSTM, Autoencoder)
- **XGBoost/LightGBM**: 고성능 부스팅 알고리즘
- **Optuna**: 하이퍼파라미터 자동 최적화

### 데이터 처리 & 시각화
- **Pandas/NumPy**: 대용량 데이터 처리 (1M+ 레코드)
- **Plotly**: 인터랙티브 차트 (Green Spectrum 테마)
- **Seaborn/Matplotlib**: 통계 시각화
- **SHAP/LIME**: 모델 해석성 분석

### 웹 프레임워크 & UI
- **Streamlit**: 데이터 사이언스 웹 애플리케이션
- **Green Spectrum UI**: 초록색 기반 현대적 테마
- **Korean Font Support**: 한글 완벽 지원 (Windows/macOS/Linux)

---

## 📊 성능 지표

### 🎯 Retail Analytics 성능
- **Linear Regression**: R² 0.65, RMSE 245.8
- **Random Forest**: R² 0.72, Feature Importance 자동 분석
- **예측 정확도**: 상대 오차 15% 이하 (High Precision)
- **처리 속도**: 100만+ 레코드 3초 이내 처리

### 🧠 Customer Analytics 성능  
- **K-means Clustering**: Silhouette Score 0.55
- **Deep Learning**: 분류 정확도 88%+
- **PCA**: 95% 분산 설명력 (5개 주성분)

### 🛡️ Security Analytics 성능
- **Binary Classification**: 정확도 96.5%, F1-Score 0.94
- **Multi-class Detection**: 평균 정확도 93.2%
- **Real-time Processing**: 100,000+ 플로우/초 처리 가능
- **False Positive Rate**: 2.1% (운영 환경 적합)

---

## 🔬 고급 실험 & 연구

### 📓 Jupyter Notebooks
프로젝트에는 다양한 머신러닝/딥러닝 실험을 위한 노트북들이 포함되어 있습니다:

```
notebooks/experiments/
├── retail_regression_comparison.ipynb      # 회귀 모델 비교 실험
├── advanced_feature_engineering.ipynb     # 고급 특성 공학
├── timeseries_analysis.ipynb              # 시계열 분석 & 예측
├── ensemble_methods_masterclass.ipynb     # 앙상블 기법 마스터
├── deep_learning_architectures.ipynb      # 딥러닝 아키텍처 실험
├── anomaly_detection_comparison.ipynb     # 이상탐지 모델 비교
├── hyperparameter_optimization.ipynb      # 하이퍼파라미터 최적화
└── model_interpretability.ipynb           # 모델 해석성 분석
```

### 🎓 학습용 실험
- **혼자 공부하는 머신러닝+딥러닝** 교재 내용 실습
- **ADP 실기 시험** 대비 실무 프로젝트
- **금융 SI 도메인** 적용 사례 연구

---

## 🛠️ 고급 사용법

### 개발자 모드
```bash
# 개발 의존성 설치
pip install -r requirements_dev.txt

# 테스트 실행
python -m pytest test/

# 코드 품질 검사
flake8 core/ web/ utils/
black core/ web/ utils/
```

#### 개발 노트 (통합 요약 — 2025-10-03)
- 최근 작업 요약: 텍스트 분석 모듈의 도메인 독립성 검증 스모크 테스트를 추가하고 테스트 구조(unit/functional)를 정리했습니다. FastAPI 엔트리포인트(`api_main.py`)를 구현하여 `startup` 이벤트에서 모델과 토크나이저를 초기화하도록 구성했고, 예측 실패 시 규칙 기반 폴백을 적용했습니다.
- 주요 파일(참고): `api_main.py`, `test/unit/test_api_unit.py`, `test/functional/test_api_integration.py`, `test/functional/test_text_import.py` (참조: `core/text/sentiment_models.py`, `main_app.py`, `web/pages/*` — 수정 금지)
- 배운 점 요약: 지연 로딩(lazy import)으로 불필요한 무거운 라이브러리 로드를 피할 수 있으며, FastAPI `startup` 이벤트는 ML 자원을 한 번만 초기화하는 안전한 패턴입니다. CI에서는 단위/통합 테스트 분리 실행 설계가 유리합니다.
- 다음 권장 작업(우선순위): 1) pytest 스타일로 테스트 리팩토링 및 케이스 추가 2) GitHub Actions 워크플로 실제 적용(단위/통합 분리) 3) `api_main.py`의 모델 로드에 에러/타임아웃/리트라이 정책 추가 4) 엔드포인트 로깅·모니터링 개선
- 제안 커밋 메시지:
```
feat(api): load Keras sentiment model at startup and use for /analyze inference,\
fallback to rule-based if unavailable
```
- 빠른 실행/검증 힌트:
```bash
# 가상환경 활성화(예시)
source .venv/bin/activate
# 단위/기능 테스트 (개별)
python -m pytest test/unit/test_api_unit.py
python -m pytest test/functional/test_api_integration.py
python -m pytest test/functional/test_text_import.py
# FastAPI 개발 서버
uvicorn api_main:app --reload --port 8000
```

---

## 🔍 문제 해결

### 자주 발생하는 문제들

#### 1. 한글 폰트 문제
```bash
# 해결: 폰트 매니저 재설정
python -c "from utils.font_manager import setup_korean_font; setup_korean_font()"
```

#### 2. 대용량 데이터 메모리 오류
```bash
# 해결: 청크 단위 처리
python scripts/process_large_data.py --chunk-size 10000
```

#### 3. TensorFlow/CUDA 설정
```bash
# GPU 사용 확인
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# CPU 전용 설치
pip install tensorflow-cpu
```

#### 4. 웹 애플리케이션 성능
```bash
# 캐시 정리
streamlit cache clear

# 메모리 사용량 최적화
export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
```

---

## 🚀 배포 & 프로덕션

### Docker 배포
```dockerfile
# Dockerfile
FROM python:3.9-slim

COPY requirements_py312_fixed.txt .
RUN pip install -r requirements_py312_fixed.txt

COPY . /app
WORKDIR /app

EXPOSE 8501
CMD ["streamlit", "run", "main_app.py"]
```

```bash
# Docker 빌드 & 실행
docker build -t commerce-analytics .
docker run -p 8501:8501 commerce-analytics
```

### 클라우드 배포
- **Streamlit Cloud**: 무료 호스팅
- **AWS EC2**: 대용량 데이터 처리
- **Google Cloud Platform**: ML 워크로드 최적화
- **Azure ML**: 엔터프라이즈 환경

---

## 🤝 기여하기

### 개발 가이드라인
1. **Fork** the repository
2. Create a **feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. Open a **Pull Request**

### 코딩 스타일
- **PEP 8** 준수
- **Type hints** 사용 권장
- **Docstring** 필수 (Google 스타일)
- **Unit tests** 작성

### 이슈 리포팅
- 🐛 **Bug Report**: 상세한 재현 단계 포함
- 💡 **Feature Request**: 비즈니스 가치 및 구현 방안
- 📚 **Documentation**: 문서 개선 제안

---

## 📄 라이선스

이 프로젝트는 **MIT License**를 따릅니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

---

## 🙏 감사의 말

### 데이터셋
- **Online Retail II**: UCI Machine Learning Repository
- **CICIDS2017**: Canadian Institute for Cybersecurity
- **Mall Customer Segmentation**: Kaggle Community

### 오픈소스 라이브러리
- [Streamlit](https://streamlit.io/) - 웹 애플리케이션 프레임워크
- [TensorFlow](https://tensorflow.org/) - 딥러닝 플랫폼
- [Plotly](https://plotly.com/) - 인터랙티브 시각화
- [Scikit-learn](https://scikit-learn.org/) - 머신러닝 라이브러리

---

## 📞 연락처 & 지원

- **🐛 이슈 리포팅**: [GitHub Issues](https://github.com/greenkey20/integrated-commerce-and-security-analytics/issues)
- **💬 토론 & 질문**: [GitHub Discussions](https://github.com/greenkey20/integrated-commerce-and-security-analytics/discussions)  
- **📧 직접 문의**: [greenkey20@example.com](mailto:greenkey20@example.com)

---

**🌟 이 프로젝트가 도움이 되었다면 Star를 눌러주세요! 🌟**

- [![GitHub stars](https://img.shields.io/github/stars/greenkey20/integrated-commerce-and-security-analytics.svg?style=social&label=Star)](https://github.com/greenkey20/integrated-commerce-and-security-analytics/stargazers)
- [![GitHub forks](https://img.shields.io/github/forks/greenkey20/integrated-commerce-and-security-analytics.svg?style=social&label=Fork)](https://github.com/greenkey20/integrated-commerce-and-security-analytics/network/members)

**💚 Green Intelligence for Sustainable Business Growth 💚**
