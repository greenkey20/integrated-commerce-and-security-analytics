from datetime import datetime, timezone
from typing import Dict, List

from fastapi import APIRouter, FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


# TODO: 내일 TextAnalyticsModels 연동
# TODO: 모레 Customer 도메인 추가
# TODO: 글피 Retail, Security 도메인 추가
# TODO: 비동기 배치 처리 추가


class TextInput(BaseModel):
    text: str = Field(..., description="Input text to analyze", examples=["I loved this movie"])


class TextAnalysisResponse(BaseModel):
    domain: str
    text: str
    sentiment: str
    confidence: float


class RootResponse(BaseModel):
    domain: str
    message: str
    domains: List[str]
    active_domains: List[str]
    note: str


class HealthResponse(BaseModel):
    domain: str
    status: str
    timestamp: str


# App metadata for multi-domain integration
app = FastAPI(
    title="Integrated Analytics API",
    description="Multi-domain/integrated commerce and security analytics API server",
    version="1.0.0",
)

# --- Text domain router ---
text_router = APIRouter(prefix="/text", tags=["text"])


@app.get("/", response_model=RootResponse)
async def root() -> RootResponse:
    """Root endpoint listing domains and active domains."""
    return RootResponse(
        domain="system",
        message="Integrated Analytics API",
        domains=["text", "customer", "retail", "security"],
        active_domains=["text"],
        note="Other domains coming soon",
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Healthcheck with timezone-aware UTC timestamp."""
    return HealthResponse(domain="system", status="healthy", timestamp=datetime.now(timezone.utc).isoformat())


@text_router.post("/analyze", response_model=TextAnalysisResponse)
async def analyze_text(payload: TextInput) -> TextAnalysisResponse:
    """Analyze text and return a dummy sentiment result for the text domain.

    This is a placeholder; real model integration will follow.
    """
    text = payload.text or ""
    lowered = text.lower()
    positive_keywords = ("good", "great", "love", "excellent", "best", "awesome")
    if any(k in lowered for k in positive_keywords):
        sentiment = "positive"
        confidence = 0.95
    else:
        sentiment = "negative"
        confidence = 0.65

    return TextAnalysisResponse(domain="text", text=text, sentiment=sentiment, confidence=float(confidence))


# --- Placeholder routers for future domains ---
def _not_implemented_response(domain: str, path: str) -> JSONResponse:
    return JSONResponse(
        status_code=501,
        content={"domain": domain, "message": f"{domain} API not implemented yet", "path": path},
    )


customer_router = APIRouter(prefix="/customer", tags=["customer"])
retail_router = APIRouter(prefix="/retail", tags=["retail"])
security_router = APIRouter(prefix="/security", tags=["security"])


@customer_router.api_route("/", methods=["GET", "POST"])
async def customer_root() -> JSONResponse:
    return _not_implemented_response("customer", "/customer")


@customer_router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def customer_catchall(path: str) -> JSONResponse:
    return _not_implemented_response("customer", f"/customer/{path}")


@retail_router.api_route("/", methods=["GET", "POST"])
async def retail_root() -> JSONResponse:
    return _not_implemented_response("retail", "/retail")


@retail_router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def retail_catchall(path: str) -> JSONResponse:
    return _not_implemented_response("retail", f"/retail/{path}")


@security_router.api_route("/", methods=["GET", "POST"])
async def security_root() -> JSONResponse:
    return _not_implemented_response("security", "/security")


@security_router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def security_catchall(path: str) -> JSONResponse:
    return _not_implemented_response("security", f"/security/{path}")


# include routers
app.include_router(text_router)
app.include_router(customer_router)
app.include_router(retail_router)
app.include_router(security_router)


# If run directly: uvicorn api_main:app --reload
