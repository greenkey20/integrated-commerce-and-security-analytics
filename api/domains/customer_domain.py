"""Customer Analytics Domain API

고객 세그멘테이션 및 분석 관련 API 엔드포인트
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
import logging

logger = logging.getLogger(__name__)

# ==================== Pydantic Models ====================

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
    clusters: List[int] = Field(description="각 고객의 클러스터 할당 (segment 결과)")


# ==================== Router Setup ====================

customer_router = APIRouter(prefix="/customer", tags=["customer"])


# ==================== Global State ====================

customer_analyzer: Optional[Any] = None
customer_module_available = False
customer_import_error: Optional[str] = None


# ==================== Initialization ====================

async def initialize_customer_module():
    """Customer Analytics 모듈 초기화
    
    ClusterAnalyzer를 로드하고 전역 변수에 할당
    """
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


# ==================== Helper Functions ====================

def _simple_customer_segment(data: CustomerData, n_clusters: int) -> List[str]:
    """규칙 기반 간단 세그멘테이션 (폴백용)
    
    Returns: List of segment labels for each customer
    """
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


# ==================== API Endpoints ====================

@customer_router.get("/health")
async def customer_health():
    """Customer domain health check"""
    return {
        "status": "ok",
        "domain": "customer",
        "module_available": customer_module_available,
        "analyzer_loaded": customer_analyzer is not None
    }


@customer_router.post("/segment")
async def segment_customers(request: SegmentRequest):
    """고객 세그멘테이션 분석
    
    ML 모델이 없으면 규칙 기반 폴백 사용
    
    Args:
        request: 고객 데이터, 클러스터 개수, 프로필 포함 여부
        
    Returns:
        클러스터 할당 + (옵션) 프로필 + 품질 지표
    """
    # 모델 없으면 폴백
    if customer_analyzer is None:
        segments = _simple_customer_segment(request.data, request.n_clusters)
        return {
            "status": "success",
            "mode": "fallback",
            "clusters": segments,
            "message": "Using rule-based segmentation (ML model not available)"
        }
    
    try:
        # DataFrame 생성
        import pandas as pd
        import numpy as np
        import math
        
        df = pd.DataFrame({
            'CustomerID': request.data.customer_ids,
            'Age': request.data.ages,
            'Annual Income (k$)': request.data.incomes,
            'Spending Score (1-100)': request.data.spending_scores
        })
        
        # 데이터 개수 확인
        n_samples = len(df)
        if n_samples <= request.n_clusters:
            return {
                "status": "warning",
                "mode": "insufficient_data",
                "message": f"Need more samples: {n_samples} samples for {request.n_clusters} clusters. Minimum: {request.n_clusters + 1}",
                "recommendation": "Reduce n_clusters or provide more customer data",
                "fallback_clusters": _simple_customer_segment(request.data, request.n_clusters)
            }
        
        # 클러스터링 수행
        clusters, kmeans, scaler, silhouette = customer_analyzer.perform_clustering(
            df, n_clusters=request.n_clusters
        )
        
        # 기본 응답 (클러스터 할당만)
        response = {
            "status": "success",
            "mode": "ml_model",
            "clusters": clusters.tolist(),
            "n_clusters": request.n_clusters,
            "silhouette_score": float(silhouette) if not (math.isnan(silhouette) or math.isinf(silhouette)) else 0.0,
        }
        
        # 프로필 포함 옵션
        if request.include_profiles:
            df['Cluster'] = clusters
            profiles = customer_analyzer.analyze_cluster_characteristics(
                df, request.n_clusters
            )
            response["profiles"] = _sanitize_for_json(profiles)
            response["message"] = "ML-based clustering with profile analysis completed"
        else:
            response["message"] = "ML-based clustering completed (use /analyze for profiles)"
        
        return response
        
    except Exception as e:
        logger.exception(f"Segmentation failed: {e}")
        import traceback
        
        # 에러시 폴백
        segments = _simple_customer_segment(request.data, request.n_clusters)
        return {
            "status": "partial",
            "mode": "fallback",
            "clusters": segments,
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc()[-500:],
            "message": "ML model failed, using fallback segmentation"
        }


@customer_router.post("/analyze")
async def analyze_clusters(request: AnalyzeRequest):
    """클러스터 특성 분석 (Opus 원안)
    
    이미 할당된 클러스터에 대한 상세 프로필 분석
    /segment에서 include_profiles=false로 호출한 후 사용
    
    Args:
        request: 고객 데이터 + 할당된 클러스터 번호
        
    Returns:
        각 클러스터의 상세 프로필 (avg, std, labels 등)
    """
    if customer_analyzer is None:
        raise HTTPException(
            status_code=503,
            detail="Customer analyzer not loaded"
        )
    
    try:
        import pandas as pd
        
        # 데이터 준비
        df = pd.DataFrame({
            'CustomerID': request.data.customer_ids,
            'Age': request.data.ages,
            'Annual Income (k$)': request.data.incomes,
            'Spending Score (1-100)': request.data.spending_scores,
            'Cluster': request.clusters
        })
        
        # 클러스터 개수 확인
        n_clusters = len(set(request.clusters))
        
        # 프로필 분석
        profiles = customer_analyzer.analyze_cluster_characteristics(
            df, n_clusters
        )
        
        return {
            "status": "success",
            "n_clusters": n_clusters,
            "profiles": _sanitize_for_json(profiles),
            "message": "Cluster profile analysis completed"
        }
        
    except Exception as e:
        logger.exception(f"Cluster analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@customer_router.post("/predict")
async def predict_customer_cluster(customer: dict):
    """단일 고객의 클러스터 예측
    
    기존에 학습된 모델이 있어야 사용 가능
    """
    if customer_analyzer is None:
        raise HTTPException(
            status_code=503, 
            detail="Customer analyzer not loaded. Please run /segment first to train a model."
        )
    
    try:
        import pandas as pd
        df = pd.DataFrame([customer])
        
        # 필요한 경우 ClusterAnalyzer에 메서드 추가 필요
        return {
            "status": "success",
            "message": "Prediction endpoint - implementation requires trained model storage",
            "customer_data": customer
        }
        
    except Exception as e:
        logger.exception(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@customer_router.get("/profiles")
async def get_cluster_profiles():
    """저장된 클러스터 프로필 조회
    
    마지막으로 분석한 프로필 반환 (구현 필요)
    """
    if customer_analyzer is None:
        raise HTTPException(
            status_code=503,
            detail="Customer analyzer not loaded"
        )
    
    return {
        "status": "success",
        "message": "Profile storage not yet implemented",
        "available": False
    }
