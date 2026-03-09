"""
FastAPI Main Entrypoint - Integrated Analytics API

통합 분석 플랫폼의 메인 진입점
각 도메인별 라우터를 통합하여 제공

Usage:
    uvicorn api_main:app --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# Domain routers import
from api.domains.text_domain import text_router, initialize_text_module
from api.domains.customer_domain import customer_router, initialize_customer_module
from api.domains.retail_domain import retail_router, initialize_retail_module
from api.domains.security_domain import security_router, initialize_security_module

# ==================== FastAPI App Setup ====================

app = FastAPI(
    title="Integrated Analytics API",
    description="통합 커머스 & 보안 분석 플랫폼 API",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 로깅 설정
logger = logging.getLogger("api_main")
logging.basicConfig(level=logging.INFO)


# ==================== Include Routers ====================

app.include_router(text_router)
app.include_router(customer_router)
app.include_router(retail_router)
app.include_router(security_router)


# ==================== Startup Event ====================

@app.on_event("startup")
async def startup():
    """애플리케이션 시작 시 모든 도메인 모듈 초기화"""
    logger.info("🚀 Starting Integrated Analytics API...")
    
    # Text Analytics 초기화
    await initialize_text_module()
    
    # Customer Analytics 초기화
    await initialize_customer_module()
    
    # Retail Analytics 초기화
    await initialize_retail_module()

    # Security Analytics 초기화
    await initialize_security_module()

    logger.info("✅ All modules initialized successfully!")


# ==================== Root Endpoints ====================

@app.get("/", tags=["health"])
async def root():
    """API 루트 엔드포인트"""
    return {
        "status": "ok",
        "message": "Integrated Analytics API",
        "version": "1.0.0",
        "domains": {
            "text": "Text Analytics (Sentiment Analysis)",
            "customer": "Customer Segmentation",
            "retail": "Retail Analytics",
            "security": "Security Analytics"
        }
    }


@app.get("/health", tags=["health"])
async def health_check():
    """헬스 체크 엔드포인트"""
    from datetime import datetime
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/debug/routes", tags=["debug"])
async def debug_routes():
    """모든 등록된 라우트 확인용 디버그 엔드포인트"""
    routes = []
    for route in app.routes:
        if hasattr(route, 'methods'):
            routes.append({
                "path": route.path,
                "methods": list(route.methods),
                "name": route.name
            })
    return {"routes": routes, "total": len(routes)}


# ==================== Main ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_main:app", host="0.0.0.0", port=8000, log_level="info", reload=True)
