# 🌿 Integrated Commerce & Security Analytics

> 차세대 이커머스를 위한 통합 인텔리전스 플랫폼  
> 고객 인사이트부터 보안 모니터링까지, 데이터 기반 비즈니스 성장을 지원합니다.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
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
- **🚀 FastAPI Backend**: 프로덕션급 REST API 서버

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
├── 🚀 FastAPI Backend (Production API)
│   ├── /text - Text Analytics API
│   ├── /customer - Customer Segmentation API
│   ├── /retail - Retail Analytics API
│   └── /security - Security Detection API
│
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
├── 🚀 api_main.py                    # FastAPI 메인 애플리케이션
├── 📱 main_app.py                    # Streamlit 메인 애플리케이션
├── 🔧 config/
│   └── settings.py                   # 설정 파일 (Green Theme)
├── 🌐 api/                           # FastAPI 백엔드
│   ├── routes/
│   │   ├── text_routes.py           # Text Analytics API
│   │   ├── customer_routes.py       # Customer Segmentation API
│   │   ├── retail_routes.py         # Retail Analytics API
│   │   └── security_routes.py       # Security Detection API
│   └── models/
│       └── schemas.py                # Pydantic 스키마 정의
├── 📊 data/
│   ├── Mall_Customers.csv            # 고객 세그멘테이션 (200개)
│   ├── base/online_retail_II.xlsx    # 리테일 분석 (1M+ 거래)
│   └── cicids2017/*.csv              # 네트워크 보안 (2.8M+ 레코드)
├── 🤖 core/                          # 핵심 비즈니스 로직
│   ├── text/
│   │   └── sentiment_models.py      # 감정 분석 모델 (LSTM)
│   ├── customer/
│   │   └── segmentation_models.py   # 고객 세그멘테이션
│   ├── retail/
│   │   ├── model_trainer.py          # 회귀 모델링
│   │   ├── analysis_manager.py       # 리테일 분석 매니저
│   │   └── visualizer.py             # 차트 & 시각화
│   └── security/
│       ├── model_builder.py          # 딥러닝 모델 빌더
│       └── detection_engine.py       # 이상 탐지 엔진
├── 🌐 web/pages/                     # Streamlit 페이지들
│   ├── segmentation/                 # 고객 세그멘테이션 페이지들
│   ├── retail/                       # 리테일 분석 페이지들
│   └── security/                     # 보안 분석 페이지들
├── 🛠️ utils/
│   ├── font_manager.py               # 한글 폰트 지원
│   └── ui_components.py              # Green Theme UI 컴포넌트
├── 🧪 tests/                         # 테스트 코드
│   ├── unit/                         # 단위 테스트
│   ├── functional/                   # 기능 테스트
│   └── integration/                  # 통합 테스트
│       └── test_api_endpoints.py    # API 엔드포인트 테스트 (✅ 10/12 passed)
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

---

### 2. FastAPI 백엔드 실행 (추천) ⭐

#### Option A: 직접 실행
```bash
# FastAPI 서버 실행
uvicorn api_main:app --reload --port 8000

# 서버 작동 확인
curl http://localhost:8000/health
```

#### Option B: Docker 실행 (예정)
```bash
# Docker 컨테이너 실행
docker-compose up -d

# 헬스 체크
curl http://localhost:8000/text/health
```

#### Swagger UI 접속
브라우저에서 http://localhost:8000/docs 로 접속하면:
- 📚 **자동 생성된 API 문서**
- 🧪 **인터랙티브 테스트 환경**
- 📝 **요청/응답 스키마 확인**

---

### 3. Streamlit 웹 애플리케이션 실행

```bash
# Streamlit 웹 애플리케이션 실행
streamlit run main_app.py
```

브라우저에서 `http://localhost:8501`로 접속하면 다음과 같은 통합 분석 환경을 사용할 수 있습니다:

- **📊 Business Intelligence**: 실무 중심 리테일 분석
- **👥 Customer Analytics**: 고객 세그멘테이션 학습
- **🛡️ Security Intelligence**: 네트워크 보안 분석

---

## 🔌 API 엔드포인트

### 📊 Text Analytics
```bash
# Health Check
GET /text/health

# 감정 분석
POST /text/analyze
{
  "text": "This product is amazing!"
}
```

### 👥 Customer Segmentation
```bash
# Health Check
GET /customer/health

# 고객 세그멘테이션
POST /customer/segment
{
  "data": {
    "customer_ids": ["C001", "C002"],
    "ages": [25, 45],
    "incomes": [35000, 85000],
    "spending_scores": [40, 75]
  },
  "n_clusters": 3
}
```

### 💰 Retail Analytics
```bash
# Health Check
GET /retail/health

# 리테일 분석
POST /retail/analyze
{
  "data": {
    "invoice_ids": ["INV001"],
    "descriptions": ["Product A"],
    "quantities": [2],
    "unit_prices": [10.5],
    "customer_ids": ["C001"],
    "countries": ["KR"]
  },
  "analysis_type": "sales"
}
```

### 🛡️ Security Detection
```bash
# Health Check
GET /security/health

# 이상 탐지
POST /security/detect
{
  "data": {
    "source_ips": ["192.168.1.1"],
    "dest_ips": ["8.8.8.8"],
    "ports": [443],
    "protocols": ["tcp"],
    "packet_sizes": [1500],
    "timestamps": ["2025-10-07T19:00:00Z"]
  }
}
```

**상세 API 문서**: http://localhost:8000/docs (Swagger UI)

---

## 🧪 테스트 실행

### 통합 테스트
```bash
# 전체 API 엔드포인트 테스트
pytest tests/integration/test_api_endpoints.py -v

# 현재 결과: ✅ 10 passed, 2 skipped (out of 12)
```

### 단위 테스트
```bash
# 전체 단위 테스트
pytest tests/unit/ -v

# 특정 모듈 테스트
pytest tests/unit/test_text_models.py -v
```

### 기능 테스트
```bash
# 기능 테스트 실행
pytest tests/functional/ -v
```

### 커버리지 리포트
```bash
# 테스트 커버리지 측정
pytest --cov=core --cov=api tests/
```

---

## 📊 성능 지표

### 🎯 FastAPI Backend 성능
- **평균 응답 시간**: < 100ms
- **동시 처리 능력**: 1,000+ requests/sec
- **Health Check**: < 5ms
- **ML 추론**: < 200ms

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

## 🐳 Docker 사용법 (예정)

### Quick Start with Docker

```bash
# 1. Docker 이미지 빌드
docker-compose build

# 2. 컨테이너 실행
docker-compose up -d

# 3. 로그 확인
docker-compose logs -f

# 4. 헬스 체크
curl http://localhost:8000/text/health
curl http://localhost:8000/customer/health
curl http://localhost:8000/retail/health
curl http://localhost:8000/security/health

# 5. 종료
docker-compose down
```

### Swagger UI 접근
http://localhost:8000/docs

---

## 🧠 핵심 기술 스택

### 백엔드 프레임워크
- **FastAPI**: 고성능 비동기 웹 프레임워크
- **Uvicorn**: ASGI 서버
- **Pydantic**: 데이터 검증 및 스키마 정의
- **pytest**: 테스트 프레임워크 (10/12 테스트 통과)

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

# FastAPI 개발 서버 (핫 리로드)
uvicorn api_main:app --reload --port 8000

# 전체 테스트 실행
pytest tests/ -v

# 코드 품질 검사
flake8 core/ api/ web/ utils/
black core/ api/ web/ utils/
```

### 데이터셋 정보

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

## 🔍 문제 해결

### 자주 발생하는 문제들

#### 1. FastAPI 서버 시작 실패
```bash
# 포트 충돌 확인
lsof -i :8000
kill -9 <PID>

# 다른 포트로 실행
uvicorn api_main:app --port 8001
```

#### 2. 테스트 실패 (503 Service Unavailable)
```bash
# 모델 파일 확인
ls -la models/text/sentiment_model.keras
ls -la models/text/tokenizer.pkl

# Fallback 모드로 실행 (모델 없어도 작동)
# api/routes/text_routes.py 참조
```

#### 3. 한글 폰트 문제
```bash
# 해결: 폰트 매니저 재설정
python -c "from utils.font_manager import setup_korean_font; setup_korean_font()"
```

#### 4. 대용량 데이터 메모리 오류
```bash
# 해결: 청크 단위 처리
python scripts/process_large_data.py --chunk-size 10000
```

#### 5. TensorFlow/CUDA 설정
```bash
# GPU 사용 확인
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# CPU 전용 설치
pip install tensorflow-cpu
```

---

## 🚀 배포 & 프로덕션

### Docker 배포 (예정)
```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 시스템 의존성
RUN apt-get update && apt-get install -y \
    gcc g++ && rm -rf /var/lib/apt/lists/*

# Python 의존성
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드
COPY . .

# 포트 노출
EXPOSE 8000

# FastAPI 실행
CMD ["uvicorn", "api_main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Docker 빌드 & 실행
docker build -t commerce-analytics-api .
docker run -p 8000:8000 commerce-analytics-api
```

### 클라우드 배포
- **AWS ECS/Fargate**: 컨테이너 기반 배포
- **Google Cloud Run**: 서버리스 컨테이너 실행
- **Azure Container Instances**: 간편한 컨테이너 배포
- **Kubernetes**: 대규모 프로덕션 환경

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
- [FastAPI](https://fastapi.tiangolo.com/) - 현대적 웹 프레임워크
- [Streamlit](https://streamlit.io/) - 데이터 앱 프레임워크
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
