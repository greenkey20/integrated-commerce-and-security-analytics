# 🛍️ Customer Segmentation Analysis System

> 고객 세분화 분석을 위한 통합 시스템 - 머신러닝과 딥러닝을 활용한 고객 분석 도구

## 📋 프로젝트 개요

이 프로젝트는 쇼핑몰 고객 데이터를 분석하여 고객을 세분화하고, 각 세그먼트별 특성을 파악하여 맞춤형 마케팅 전략을 제공하는 시스템입니다.

### 🎯 주요 기능

- **📊 데이터 분석**: 고객 데이터 탐색 및 시각화
- **🎯 클러스터링**: K-means를 활용한 고객 세분화
- **🔬 주성분 분석**: 데이터 차원 축소 및 패턴 발견
- **🧠 딥러닝**: 신경망을 활용한 고객 분류
- **🔮 예측**: 새로운 고객의 세그먼트 예측
- **📈 마케팅 전략**: 세그먼트별 맞춤형 전략 제안
- **🚀 REST API**: 프로덕션 환경을 위한 API 서버

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd customer-segmentation

# 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 필요한 패키지 설치
pip install -r requirements.txt
```

### 2. 데이터 준비 및 모델 훈련

```bash
# 모든 모델 훈련 (최초 실행 시)
python train_models.py

# 특정 모델만 훈련
python train_models.py --mode clustering --clusters 5
python train_models.py --mode deep_learning --clusters 5
python train_models.py --mode dummy  # API 테스트용
```

### 3. 애플리케이션 실행

#### 웹 애플리케이션 (데이터 분석)
```bash
streamlit run main_app.py
```
- 브라우저에서 http://localhost:8501 접속
- 대화형 데이터 분석 및 시각화

#### API 서버 (프로덕션)
```bash
python api_server.py
```
- API 문서: http://localhost:8000/docs
- 상태 확인: http://localhost:8000/api/v1/health

## 📁 프로젝트 구조

```
customer-segmentation/
├── 📱 main_app.py              # 메인 Streamlit 웹 애플리케이션
├── 🚀 api_server.py            # FastAPI REST API 서버
├── 🧠 train_models.py          # 모델 훈련 통합 스크립트
├── 🔧 check_servers.sh         # 서버 상태 확인 스크립트
├── 📋 requirements.txt         # Python 의존성 목록
├── 📄 setup.py                 # 패키지 설치 스크립트
├── 📊 data/
│   ├── Mall_Customers.csv      # 기본 고객 데이터
│   └── processed/              # 전처리된 데이터
├── 🤖 models/
│   └── saved_models/           # 훈련된 모델들
├── 📦 core/                    # 핵심 비즈니스 로직
│   ├── clustering.py
│   ├── deep_learning_models.py
│   ├── anomaly_detection.py
│   └── data_processing.py
├── 📑 app_modules/             # Streamlit 페이지 모듈들
│   ├── data_overview.py
│   ├── clustering_analysis.py
│   ├── deep_learning_analysis.py
│   └── customer_prediction.py
├── 🛠️ utils/                   # 유틸리티 함수들
├── ⚙️ config/                  # 설정 파일들
├── 🧪 test/                    # 테스트 코드들
└── 📚 archive/                 # 백업 파일들
```

## 💡 사용 방법

### 웹 애플리케이션 사용법

1. **데이터 개요**: 기본 데이터 정보 확인
2. **탐색적 분석**: 데이터 패턴 및 분포 분석
3. **클러스터링**: 고객 세분화 수행
4. **주성분 분석**: 데이터 차원 축소
5. **딥러닝 분석**: 신경망 모델 훈련
6. **고객 예측**: 새 고객 세그먼트 예측
7. **마케팅 전략**: 세그먼트별 전략 제안

### API 사용법

```bash
# 고객 세그먼트 예측
curl -X POST "http://localhost:8000/api/v1/customer/segment" \
     -H "Content-Type: application/json" \
     -d '{
       "age": 35,
       "income": 70,
       "spending_score": 80
     }'

# 서버 상태 확인
curl http://localhost:8000/api/v1/health

# API 통계
curl http://localhost:8000/api/v1/stats
```

## 🔧 고급 설정

### 모델 훈련 옵션

```bash
# 클러스터 개수 변경
python train_models.py --clusters 3

# 특정 모델만 훈련
python train_models.py --mode clustering
python train_models.py --mode deep_learning

# 의존성 체크 건너뛰기
python train_models.py --skip-deps
```

### 환경 변수

```bash
# API 서버 설정
export API_HOST=0.0.0.0
export API_PORT=8000

# 모델 경로 설정
export MODEL_PATH=./models/saved_models
```

## 🧪 테스트

```bash
# 모든 테스트 실행
python -m pytest test/

# 특정 테스트 실행
python -m pytest test/test_clustering.py
python -m pytest test/test_api.py
```

## 📊 성능 지표

### 클러스터링 성능
- **실루엣 점수**: 0.4 ~ 0.8 (클러스터 품질)
- **관성**: 클러스터 내 분산의 합

### 딥러닝 성능
- **정확도**: 85% 이상
- **예측 신뢰도**: 평균 90% 이상

## 🔍 문제 해결

### 자주 발생하는 문제들

1. **ModuleNotFoundError**: 
   ```bash
   # 해결: 의존성 재설치
   pip install -r requirements.txt
   ```

2. **모델 파일 없음**:
   ```bash
   # 해결: 모델 재훈련
   python train_models.py --mode all
   ```

3. **포트 이미 사용 중**:
   ```bash
   # 해결: 프로세스 확인 및 종료
   ./check_servers.sh
   lsof -ti:8000 | xargs kill -9
   ```

4. **메모리 부족**:
   ```bash
   # 해결: 배치 크기 줄이기
   python train_models.py --batch-size 16
   ```

## 🤝 기여하기

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 👥 기여자

- **Your Name** - 초기 개발 및 유지보수

## 🙏 감사의 말

- [Mall Customer Segmentation Dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- Streamlit 팀
- TensorFlow 팀
- FastAPI 팀

---

**📞 문의사항**: 이슈 탭에 문의하거나 이메일로 연락주세요.

**🌟 이 프로젝트가 도움이 되었다면 Star를 눌러주세요!**
