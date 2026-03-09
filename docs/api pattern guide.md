# 🎯 FastAPI Domain 구현 패턴 가이드 (Complete & Thorough)

> **목적**: GitHub Copilot이 이 문서를 참조하여 Retail, Security 도메인을 자동 생성  
> **기준**: Text, Customer 도메인의 실제 구현 패턴

---

## 📁 프로젝트 구조

```
integrated-commerce-and-security-analytics/
├── api_main.py                          # 메인 진입점 (90줄)
├── api/
│   ├── __init__.py
│   └── domains/
│       ├── __init__.py
│       ├── text_domain.py              # 180줄 (완료 ✅)
│       ├── customer_domain.py          # 280줄 (완료 ✅)
│       ├── retail_domain.py            # 생성 필요 ⬜
│       └── security_domain.py          # 생성 필요 ⬜
├── core/
│   ├── text/
│   │   └── sentiment_models.py         # TextAnalyticsModels
│   ├── segmentation/
│   │   └── clustering.py               # ClusterAnalyzer
│   ├── retail/
│   │   ├── analysis.py                 # RetailAnalyzer (있음)
│   │   ├── modeling.py                 # RetailModeling (있음)
│   │   └── evaluation.py               # RetailEvaluation (있음)
│   └── security/
│       └── anomaly_detection.py        # SecurityAnalyzer (있음)
└── config/
    └── settings.py
```

---

## 🏗️ 도메인 파일 구조 (표준 템플릿)

### 섹션 순서 (모든 도메인 동일)
```python
"""
{Domain} Analytics Domain API

{Domain} 설명
"""
# 1. Imports
# 2. Pydantic Models
# 3. Router Setup
# 4. Global State
# 5. Initialization Function
# 6. Helper Functions
# 7. API Endpoints
```

---

## 📦 1. Imports (모든 도메인 동일)

```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
import logging

logger = logging.getLogger(__name__)
```

---

## 📋 2. Pydantic Models 패턴

### 기본 원칙
- 입력 데이터용 Model
- 요청 파라미터용 Request Model
- 각 엔드포인트별 전용 Model

### Text Domain 예시
```python
class TextInput(BaseModel):
    text: str
```

### Customer Domain 예시
```python
class CustomerData(BaseModel):
    customer_ids: List[str]
    ages: List[float]
    incomes: List[float]
    spending_scores: List[float]

class SegmentRequest(BaseModel):
    data: CustomerData
    n_clusters: int = Field(default=5, ge=2, le=10)
    include_profiles: bool = Field(default=True, description="프로필 분석 포함 여부")

class AnalyzeRequest(BaseModel):
    data: CustomerData
    clusters: List[int] = Field(description="각 고객의 클러스터 할당")
```

### Retail Domain 가이드
```python
# 필요한 Models:
# 1. RetailData - 기본 거래 데이터
# 2. AnalyzeRequest - 분석 요청
# 3. PredictRequest - 예측 요청
# 4. ReportRequest - 리포트 요청

class RetailData(BaseModel):
    invoice_ids: List[str]
    descriptions: List[str]
    quantities: List[int]
    unit_prices: List[float]
    customer_ids: List[str]
    countries: List[str]

class AnalyzeRequest(BaseModel):
    data: RetailData
    analysis_type: str = Field(default="sales", description="sales|product|customer")

# ... 다른 Models
```

### Security Domain 가이드
```python
# 필요한 Models:
# 1. NetworkTraffic - 네트워크 트래픽 데이터
# 2. DetectRequest - 이상 탐지 요청
# 3. MonitorRequest - 모니터링 요청

class NetworkTraffic(BaseModel):
    source_ips: List[str]
    dest_ips: List[str]
    ports: List[int]
    protocols: List[str]
    packet_sizes: List[int]
    timestamps: List[str]

# ... 다른 Models
```

---

## 🎯 3. Router Setup (모든 도메인 동일 패턴)

```python
# ==================== Router Setup ====================

{domain}_router = APIRouter(prefix="/{domain}", tags=["{domain}"])
```

**예시**:
- Text: `text_router = APIRouter(prefix="/text", tags=["text"])`
- Customer: `customer_router = APIRouter(prefix="/customer", tags=["customer"])`
- Retail: `retail_router = APIRouter(prefix="/retail", tags=["retail"])`
- Security: `security_router = APIRouter(prefix="/security", tags=["security"])`

---

## 🔧 4. Global State 패턴

### 핵심 원칙
- 모델 인스턴스는 전역 변수로 관리
- 로딩 성공/실패 상태 추적
- Import 에러 메시지 저장

### Text Domain 예시
```python
# ==================== Global State ====================

sentiment_model: Optional[Any] = None
keras_model: Optional[Any] = None
text_import_exception: Optional[Exception] = None
```

### Customer Domain 예시
```python
# ==================== Global State ====================

customer_analyzer: Optional[Any] = None
customer_module_available = False
customer_import_error: Optional[str] = None
```

### Retail Domain 가이드
```python
# ==================== Global State ====================

retail_analyzer: Optional[Any] = None
retail_modeler: Optional[Any] = None
retail_module_available = False
retail_import_error: Optional[str] = None
```

### Security Domain 가이드
```python
# ==================== Global State ====================

security_detector: Optional[Any] = None
security_module_available = False
security_import_error: Optional[str] = None
```

---

## 🚀 5. Initialization Function 패턴

### 핵심 원칙
- `async def initialize_{domain}_module()` 함수명 규칙
- global 변수 선언
- try-except로 방어적 import
- 로깅 (✅ 성공, ❌ 실패)

### Text Domain 예시
```python
async def initialize_text_module():
    """Text Analytics 모듈 초기화"""
    global sentiment_model, keras_model, text_import_exception
    
    # Import 시도
    try:
        from core.text.sentiment_models import TextAnalyticsModels
    except Exception as e:
        text_import_exception = e
        logger.error(f"❌ Failed to import TextAnalyticsModels: {e}")
        return
    
    # 모델 생성
    try:
        wrapper = TextAnalyticsModels()
        model_obj, err = wrapper.create_sentiment_lstm()
        
        if model_obj is None:
            keras_model = None
            sentiment_model = wrapper
            logger.warning(f"⚠️ Model creation partial: {err}")
        else:
            keras_model = model_obj
            sentiment_model = wrapper
            sentiment_model.sentiment_model = keras_model
            logger.info("✅ Keras model created successfully.")
    except Exception as e:
        sentiment_model = None
        keras_model = None
        logger.error(f"❌ Failed to create model: {e}")
```

### Customer Domain 예시
```python
async def initialize_customer_module():
    """Customer Analytics 모듈 초기화"""
    global customer_analyzer, customer_module_available, customer_import_error
    
    try:
        from core.segmentation.clustering import ClusterAnalyzer
        customer_analyzer = ClusterAnalyzer()
        customer_module_available = True
        logger.info("✅ Customer analyzer loaded successfully.")
    except Exception as e:
        customer_module_available = False
        customer_import_error = str(e)
        logger.error(f"❌ Failed to load customer analyzer: {e}")
```

### Retail Domain 가이드
```python
async def initialize_retail_module():
    """Retail Analytics 모듈 초기화"""
    global retail_analyzer, retail_modeler, retail_module_available, retail_import_error
    
    try:
        from core.retail.analysis import RetailAnalyzer
        from core.retail.modeling import RetailModeling
        
        retail_analyzer = RetailAnalyzer()
        retail_modeler = RetailModeling()
        retail_module_available = True
        logger.info("✅ Retail modules loaded successfully.")
    except Exception as e:
        retail_module_available = False
        retail_import_error = str(e)
        logger.error(f"❌ Failed to load retail modules: {e}")
```

### Security Domain 가이드
```python
async def initialize_security_module():
    """Security Analytics 모듈 초기화"""
    global security_detector, security_module_available, security_import_error
    
    try:
        from core.security.anomaly_detection import SecurityAnalyzer
        security_detector = SecurityAnalyzer()
        security_module_available = True
        logger.info("✅ Security detector loaded successfully.")
    except Exception as e:
        security_module_available = False
        security_import_error = str(e)
        logger.error(f"❌ Failed to load security detector: {e}")
```

---

## 🛠️ 6. Helper Functions 패턴

### 핵심 원칙
- 폴백 함수: `_simple_{action}()` 형식
- JSON 정제: `_sanitize_for_json()` (numpy/pandas 처리)
- 모두 private 함수 (`_` prefix)

### Text Domain 예시
```python
POSITIVE_WORDS = {"love", "great", "excellent", ...}
NEGATIVE_WORDS = {"hate", "bad", "terrible", ...}

def _simple_rule_sentiment(text: str) -> Dict[str, object]:
    """규칙 기반 감정 분석 (폴백용)"""
    if not text or not text.strip():
        return {"label": "neutral", "score": 0.0}
    
    words = re.findall(r"\w+", text.lower())
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    
    total = pos + neg
    if total == 0:
        return {"label": "neutral", "score": 0.0}
    
    score = (pos - neg) / total
    label = "positive" if score > 0 else ("negative" if score < 0 else "neutral")
    return {"label": label, "score": float(score)}
```

### Customer Domain 예시
```python
def _simple_customer_segment(data: CustomerData, n_clusters: int) -> List[str]:
    """규칙 기반 세그멘테이션 (폴백용)"""
    segments = []
    for i in range(len(data.customer_ids)):
        income = data.incomes[i]
        spending = data.spending_scores[i]
        
        if income > 60 and spending > 60:
            segment = "premium"
        elif income < 40 and spending < 40:
            segment = "economy"
        else:
            segment = "standard"
        segments.append(segment)
    return segments

def _sanitize_for_json(obj):
    """NaN, Infinity를 JSON 호환 값으로 변환"""
    import numpy as np
    import math
    
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj
```

### Retail Domain 가이드
```python
# 필요한 Helper Functions:
# 1. _simple_sales_forecast() - 규칙 기반 매출 예측 (폴백)
# 2. _calculate_basic_metrics() - 기본 메트릭 계산
# 3. _sanitize_for_json() - Customer와 동일

def _simple_sales_forecast(data: RetailData) -> Dict:
    """규칙 기반 매출 예측 (폴백용)"""
    # 간단한 이동평균 또는 규칙 기반 예측
    pass

def _calculate_basic_metrics(data: RetailData) -> Dict:
    """기본 메트릭 계산"""
    # 총매출, 평균 주문액, 고객당 매출 등
    pass
```

### Security Domain 가이드
```python
# 필요한 Helper Functions:
# 1. _simple_anomaly_detection() - 규칙 기반 이상 탐지 (폴백)
# 2. _calculate_traffic_stats() - 트래픽 통계
# 3. _sanitize_for_json() - Customer와 동일

def _simple_anomaly_detection(data: NetworkTraffic) -> List[bool]:
    """규칙 기반 이상 탐지 (폴백용)"""
    # 간단한 임계값 기반 탐지
    anomalies = []
    for i in range(len(data.source_ips)):
        if data.packet_sizes[i] > 10000:  # 예시: 큰 패킷
            anomalies.append(True)
        else:
            anomalies.append(False)
    return anomalies
```

---

## 🎬 7. API Endpoints 패턴

### 7.1 Health Check (모든 도메인 필수)

```python
@{domain}_router.get("/health")
async def {domain}_health():
    """{Domain} domain health check"""
    return {
        "status": "ok",
        "domain": "{domain}",
        "module_available": {domain}_module_available,
        "{main_component}_loaded": {main_component} is not None
    }
```

**예시**:
```python
# Text
@text_router.get("/health")
async def text_health():
    """Text domain health check"""
    return {
        "status": "ok",
        "domain": "text",
        "model_loaded": sentiment_model is not None,
        "keras_model_available": keras_model is not None
    }

# Customer
@customer_router.get("/health")
async def customer_health():
    """Customer domain health check"""
    return {
        "status": "ok",
        "domain": "customer",
        "module_available": customer_module_available,
        "analyzer_loaded": customer_analyzer is not None
    }

# Retail (생성 필요)
@retail_router.get("/health")
async def retail_health():
    """Retail domain health check"""
    return {
        "status": "ok",
        "domain": "retail",
        "module_available": retail_module_available,
        "analyzer_loaded": retail_analyzer is not None
    }

# Security (생성 필요)
@security_router.get("/health")
async def security_health():
    """Security domain health check"""
    return {
        "status": "ok",
        "domain": "security",
        "module_available": security_module_available,
        "detector_loaded": security_detector is not None
    }
```

---

### 7.2 메인 분석 엔드포인트 패턴

#### 구조
```python
@{domain}_router.post("/{action}")
async def {action}_{domain}(request: {Request}):
    """
    {설명}
    
    Args:
        request: {설명}
        
    Returns:
        {설명}
    """
    # 1. 모델 없으면 폴백
    if {main_component} is None:
        fallback_result = _simple_{action}(request.data)
        return {
            "status": "success",
            "mode": "fallback",
            "result": fallback_result,
            "message": "Using rule-based fallback"
        }
    
    # 2. ML 모델 실행
    try:
        import pandas as pd
        import numpy as np
        import math
        
        # DataFrame 생성
        df = pd.DataFrame({...})
        
        # 분석 수행
        result = {main_component}.{method}(df, ...)
        
        # 결과 정제
        clean_result = _sanitize_for_json(result)
        
        return {
            "status": "success",
            "mode": "ml_model",
            "result": clean_result,
            "message": "ML analysis completed"
        }
        
    except Exception as e:
        logger.exception(f"{action} failed: {e}")
        import traceback
        
        # 에러시 폴백
        fallback_result = _simple_{action}(request.data)
        return {
            "status": "partial",
            "mode": "fallback",
            "result": fallback_result,
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc()[-500:],
            "message": "ML failed, using fallback"
        }
```

#### Text Domain 실제 구현
```python
@text_router.post("/analyze")
async def analyze_sentiment(payload: TextInput):
    """감정 분석 수행"""
    if sentiment_model is None:
        if text_import_exception is not None:
            raise HTTPException(status_code=503, detail=f"Model import failed: {text_import_exception}")
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Keras 모델 사용
    if keras_model is not None:
        try:
            text = (payload.text or "").lower()
            tokens = re.findall(r"\w+", text)
            
            # ... 토큰화 및 예측 로직
            
            return {"sentiment": {"label": label, "score": score}, "used_model": "keras_model"}
        except Exception as e:
            logger.exception(f"Keras model prediction failed: {e}")
    
    # 폴백
    fallback_result = _simple_rule_sentiment(payload.text)
    return {"sentiment": fallback_result, "used_model": "fallback"}
```

#### Customer Domain 실제 구현
```python
@customer_router.post("/segment")
async def segment_customers(request: SegmentRequest):
    """고객 세그멘테이션 분석"""
    if customer_analyzer is None:
        segments = _simple_customer_segment(request.data, request.n_clusters)
        return {
            "status": "success",
            "mode": "fallback",
            "clusters": segments,
            "message": "Using rule-based segmentation"
        }
    
    try:
        import pandas as pd
        import numpy as np
        import math
        
        df = pd.DataFrame({...})
        
        # 데이터 검증
        if len(df) <= request.n_clusters:
            return {"status": "warning", "mode": "insufficient_data", ...}
        
        # 클러스터링
        clusters, kmeans, scaler, silhouette = customer_analyzer.perform_clustering(df, request.n_clusters)
        
        # 응답 구성
        response = {
            "status": "success",
            "mode": "ml_model",
            "clusters": clusters.tolist(),
            "silhouette_score": float(silhouette) if not math.isnan(silhouette) else 0.0
        }
        
        # 프로필 포함 옵션
        if request.include_profiles:
            df['Cluster'] = clusters
            profiles = customer_analyzer.analyze_cluster_characteristics(df, request.n_clusters)
            response["profiles"] = _sanitize_for_json(profiles)
        
        return response
        
    except Exception as e:
        logger.exception(f"Segmentation failed: {e}")
        segments = _simple_customer_segment(request.data, request.n_clusters)
        return {"status": "partial", "mode": "fallback", "clusters": segments, "error": str(e)}
```

---

### 7.3 Retail Domain 엔드포인트 가이드

#### POST /retail/analyze - 리테일 분석
```python
@retail_router.post("/analyze")
async def analyze_retail(request: AnalyzeRequest):
    """리테일 데이터 분석
    
    Args:
        request: RetailData + analysis_type
        
    Returns:
        분석 결과 (매출, 상품, 고객 등)
    """
    if retail_analyzer is None:
        # 폴백: 기본 메트릭만 계산
        basic_metrics = _calculate_basic_metrics(request.data)
        return {
            "status": "success",
            "mode": "fallback",
            "metrics": basic_metrics,
            "message": "Basic metrics only (ML model not available)"
        }
    
    try:
        import pandas as pd
        
        # DataFrame 생성
        df = pd.DataFrame({
            'InvoiceNo': request.data.invoice_ids,
            'Description': request.data.descriptions,
            'Quantity': request.data.quantities,
            'UnitPrice': request.data.unit_prices,
            'CustomerID': request.data.customer_ids,
            'Country': request.data.countries
        })
        
        # 분석 수행 (analysis_type에 따라)
        if request.analysis_type == "sales":
            result = retail_analyzer.analyze_sales(df)
        elif request.analysis_type == "product":
            result = retail_analyzer.analyze_products(df)
        else:
            result = retail_analyzer.analyze_customers(df)
        
        return {
            "status": "success",
            "mode": "ml_model",
            "analysis_type": request.analysis_type,
            "result": _sanitize_for_json(result),
            "message": "Retail analysis completed"
        }
        
    except Exception as e:
        logger.exception(f"Retail analysis failed: {e}")
        basic_metrics = _calculate_basic_metrics(request.data)
        return {
            "status": "partial",
            "mode": "fallback",
            "metrics": basic_metrics,
            "error": str(e),
            "message": "ML failed, using basic metrics"
        }
```

#### POST /retail/predict - 매출 예측
```python
@retail_router.post("/predict")
async def predict_retail(request: PredictRequest):
    """매출 예측
    
    Args:
        request: 예측 파라미터
        
    Returns:
        예측 결과
    """
    if retail_modeler is None:
        # 폴백: 단순 예측
        simple_forecast = _simple_sales_forecast(request.data)
        return {
            "status": "success",
            "mode": "fallback",
            "forecast": simple_forecast,
            "message": "Simple forecast (ML model not available)"
        }
    
    try:
        # ML 예측 로직
        # ...
        pass
    except Exception as e:
        # 폴백 처리
        pass
```

#### GET /retail/report - 리포트 생성
```python
@retail_router.get("/report")
async def get_retail_report(period: str = "monthly"):
    """리테일 리포트 생성
    
    Args:
        period: 기간 (daily, weekly, monthly)
        
    Returns:
        종합 리포트
    """
    if retail_analyzer is None:
        raise HTTPException(status_code=503, detail="Retail analyzer not loaded")
    
    try:
        # 리포트 생성 로직
        report = retail_analyzer.generate_report(period)
        return {
            "status": "success",
            "period": period,
            "report": _sanitize_for_json(report),
            "message": "Report generated successfully"
        }
    except Exception as e:
        logger.exception(f"Report generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

---

### 7.4 Security Domain 엔드포인트 가이드

#### POST /security/detect - 이상 탐지
```python
@security_router.post("/detect")
async def detect_anomalies(request: DetectRequest):
    """네트워크 이상 탐지
    
    Args:
        request: NetworkTraffic 데이터
        
    Returns:
        이상 탐지 결과
    """
    if security_detector is None:
        # 폴백: 규칙 기반
        simple_anomalies = _simple_anomaly_detection(request.data)
        return {
            "status": "success",
            "mode": "fallback",
            "anomalies": simple_anomalies,
            "message": "Rule-based detection (ML model not available)"
        }
    
    try:
        import pandas as pd
        
        # DataFrame 생성
        df = pd.DataFrame({
            'src_ip': request.data.source_ips,
            'dst_ip': request.data.dest_ips,
            'port': request.data.ports,
            'protocol': request.data.protocols,
            'packet_size': request.data.packet_sizes,
            'timestamp': request.data.timestamps
        })
        
        # ML 이상 탐지
        anomalies, scores = security_detector.detect_anomalies(df)
        
        return {
            "status": "success",
            "mode": "ml_model",
            "anomalies": anomalies.tolist(),
            "anomaly_scores": _sanitize_for_json(scores),
            "total_anomalies": int(anomalies.sum()),
            "message": "ML anomaly detection completed"
        }
        
    except Exception as e:
        logger.exception(f"Anomaly detection failed: {e}")
        simple_anomalies = _simple_anomaly_detection(request.data)
        return {
            "status": "partial",
            "mode": "fallback",
            "anomalies": simple_anomalies,
            "error": str(e),
            "message": "ML failed, using rule-based detection"
        }
```

#### POST /security/monitor - 실시간 모니터링
```python
@security_router.post("/monitor")
async def monitor_traffic(request: MonitorRequest):
    """실시간 트래픽 모니터링
    
    Args:
        request: 모니터링 파라미터
        
    Returns:
        모니터링 결과
    """
    if security_detector is None:
        raise HTTPException(status_code=503, detail="Security detector not loaded")
    
    try:
        # 실시간 모니터링 로직
        # ...
        pass
    except Exception as e:
        logger.exception(f"Monitoring failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

#### GET /security/alerts - 알림 조회
```python
@security_router.get("/alerts")
async def get_alerts(severity: str = "all"):
    """보안 알림 조회
    
    Args:
        severity: 심각도 (low, medium, high, critical, all)
        
    Returns:
        알림 목록
    """
    if security_detector is None:
        raise HTTPException(status_code=503, detail="Security detector not loaded")
    
    try:
        alerts = security_detector.get_alerts(severity)
        return {
            "status": "success",
            "severity": severity,
            "alerts": _sanitize_for_json(alerts),
            "count": len(alerts),
            "message": "Alerts retrieved successfully"
        }
    except Exception as e:
        logger.exception(f"Alert retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

---

## 📝 8. api_main.py 통합 패턴

### 현재 구조 (Text, Customer 완료)
```python
"""FastAPI Main Entrypoint"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# Domain routers import
from api.domains.text_domain import text_router, initialize_text_module
from api.domains.customer_domain import customer_router, initialize_customer_module
# TODO: from api.domains.retail_domain import retail_router, initialize_retail_module
# TODO: from api.domains.security_domain import security_router, initialize_security_module

app = FastAPI(
    title="Integrated Analytics API",
    description="통합 커머스 & 보안 분석 플랫폼 API",
    version="1.0.0"
)

# CORS
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)

# Include routers
app.include_router(text_router)
app.include_router(customer_router)
# TODO: app.include_router(retail_router)
# TODO: app.include_router(security_router)

# Startup
@app.on_event("startup")
async def startup():
    logger.info("🚀 Starting Integrated Analytics API...")
    await initialize_text_module()
    await initialize_customer_module()
    # TODO: await initialize_retail_module()
    # TODO: await initialize_security_module()
    logger.info("✅ All modules initialized successfully!")

# Root endpoints
@app.get("/")
async def root():
    return {
        "status": "ok",
        "version": "1.0.0",
        "domains": {
            "text": "Text Analytics",
            "customer": "Customer Segmentation",
            "retail": "Retail Analytics",
            "security": "Security Analytics"
        }
    }
```

### Retail/Security 추가 시 (TODO 제거)
```python
# Import 추가
from api.domains.retail_domain import retail_router, initialize_retail_module
from api.domains.security_domain import security_router, initialize_security_module

# Router 등록
app.include_router(retail_router)
app.include_router(security_router)

# Startup 추가
@app.on_event("startup")
async def startup():
    # ...
    await initialize_retail_module()
    await initialize_security_module()
    # ...
```

---

## 🎨 9. 네이밍 컨벤션

### 파일명
- `{domain}_domain.py`
- 예: `text_domain.py`, `customer_domain.py`, `retail_domain.py`, `security_domain.py`

### Router 변수명
- `{domain}_router`
- 예: `text_router`, `customer_router`, `retail_router`, `security_router`

### 초기화 함수명
- `initialize_{domain}_module()`
- 예: `initialize_text_module()`, `initialize_customer_module()`

### 전역 변수명
- 메인 컴포넌트: `{domain}_{component}`
- 예: `sentiment_model`, `customer_analyzer`, `retail_analyzer`, `security_detector`
- 상태 변수: `{domain}_module_available`, `{domain}_import_error`

### 엔드포인트 함수명
- `{action}_{domain}()`
- 예: `analyze_sentiment()`, `segment_customers()`, `analyze_retail()`, `detect_anomalies()`

### Helper 함수명
- `_simple_{action}()` - 폴백 함수
- `_sanitize_for_json()` - JSON 정제
- `_calculate_{metric}()` - 계산 함수

---

## ✅ 10. 응답 형식 표준

### 성공 응답 (ML 모델 사용)
```json
{
  "status": "success",
  "mode": "ml_model",
  "result": {...},
  "message": "Analysis completed successfully"
}
```

### 성공 응답 (폴백 사용)
```json
{
  "status": "success",
  "mode": "fallback",
  "result": {...},
  "message": "Using rule-based fallback (ML model not available)"
}
```

### 부분 성공 (에러 후 폴백)
```json
{
  "status": "partial",
  "mode": "fallback",
  "result": {...},
  "error": "error message",
  "error_type": "ValueError",
  "traceback": "...",
  "message": "ML failed, using fallback"
}
```

### 경고 (데이터 부족 등)
```json
{
  "status": "warning",
  "mode": "insufficient_data",
  "message": "Need more samples: 3 samples for 3 clusters",
  "recommendation": "Reduce n_clusters or provide more data",
  "fallback_result": {...}
}
```

### 에러 응답 (HTTPException)
```json
{
  "detail": "Module not loaded"
}
```

---

## 🔍 11. 에러 처리 체크리스트

### 모든 엔드포인트에서 확인
- [ ] 모델/analyzer가 None인 경우 처리
- [ ] import 실패 시 HTTPException
- [ ] ML 실행 실패 시 폴백
- [ ] 데이터 검증 (크기, 형식)
- [ ] NaN/Infinity 처리 (JSON 직렬화)
- [ ] 로깅 (logger.exception)
- [ ] traceback 포함 (마지막 500자)

---

## 📊 12. 전체 체크리스트 (도메인 생성 시)

### 파일 생성
- [ ] `api/domains/{domain}_domain.py` 생성
- [ ] Docstring 추가

### Imports
- [ ] fastapi, pydantic, typing, logging import
- [ ] logger 설정

### Pydantic Models
- [ ] 입력 데이터 Model
- [ ] 요청 파라미터 Model (2-3개)
- [ ] Field validation

### Router Setup
- [ ] APIRouter 생성 (prefix, tags)

### Global State
- [ ] 전역 변수 선언 (모델, 상태, 에러)

### Initialization
- [ ] `initialize_{domain}_module()` 함수
- [ ] try-except import
- [ ] 로깅 (✅/❌)

### Helper Functions
- [ ] 폴백 함수 (1-2개)
- [ ] `_sanitize_for_json()` (필요 시)
- [ ] 기타 유틸리티

### Endpoints
- [ ] `GET /{domain}/health`
- [ ] `POST /{domain}/{action1}` (메인 분석)
- [ ] `POST /{domain}/{action2}` (예측/추가 분석)
- [ ] `GET /{domain}/{resource}` (리소스 조회)

### api_main.py 통합
- [ ] import router, init function
- [ ] app.include_router()
- [ ] startup에 init 추가

### 테스트
- [ ] health endpoint 확인
- [ ] 각 엔드포인트 curl 테스트
- [ ] Swagger UI 확인
- [ ] 에러 케이스 테스트

---

## 🎯 끝! Copilot이 이 패턴을 따라 Retail, Security를 자동 생성할 수 있어야 함!
