from datetime import datetime, timezone
from typing import Dict

from fastapi import APIRouter, FastAPI
from pydantic import BaseModel, Field


# TODO: 내일 core/text/sentiment_models.py 연동
# TODO: 비동기 배치 처리 추가


class TextInput(BaseModel):
    text: str = Field(..., description="Input text to analyze", examples=["I loved this movie"])


class TextAnalysisResponse(BaseModel):
    domain: str
    text: str
    sentiment: str
    confidence: float


app = FastAPI(title="Text Analytics API")

text_router = APIRouter(prefix="/text", tags=["text"])


@app.get("/", response_model=Dict[str, str])
async def root() -> Dict[str, str]:
    """Root endpoint for quick info about this API."""
    return {"message": "Text Analytics API v1.0", "status": "running"}


@app.get("/health", response_model=Dict[str, str])
async def health() -> Dict[str, str]:
    """Simple healthcheck with timestamp."""
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


@text_router.post("/analyze", response_model=TextAnalysisResponse)
async def analyze_text(payload: TextInput) -> TextAnalysisResponse:
    """Analyze text and return a dummy sentiment result.

    This currently returns a heuristic/dummy response. Real model
    integration will be added later.
    """
    text = payload.text or ""

    # very small heuristic to make dummy responses slightly sensible
    lowered = text.lower()
    positive_keywords = ("good", "great", "love", "excellent", "best", "awesome")
    if any(k in lowered for k in positive_keywords):
        sentiment = "positive"
        confidence = 0.95
    else:
        sentiment = "negative"
        confidence = 0.65

    return TextAnalysisResponse(domain="text", text=text, sentiment=sentiment, confidence=float(confidence))


# include routers (prepared for multi-domain expansion)
app.include_router(text_router)


# If run directly (not required), you can use: `uvicorn api_main:app --reload`
