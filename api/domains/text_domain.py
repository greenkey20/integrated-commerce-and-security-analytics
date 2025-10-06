"""Text Analytics Domain API

감정 분석(Sentiment Analysis) 관련 API 엔드포인트
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Any, Dict
import logging
import re

logger = logging.getLogger(__name__)

# ==================== Pydantic Models ====================

class TextInput(BaseModel):
    text: str


# ==================== Router Setup ====================

text_router = APIRouter(prefix="/text", tags=["text"])


# ==================== Global State ====================

sentiment_model: Optional[Any] = None
keras_model: Optional[Any] = None
text_import_exception: Optional[Exception] = None


# ==================== Initialization ====================

async def initialize_text_module():
    """Text Analytics 모듈 초기화
    
    TextAnalyticsModels를 로드하고 LSTM 모델 생성
    """
    global sentiment_model, keras_model, text_import_exception
    
    # Import 시도
    try:
        from core.text.sentiment_models import TextAnalyticsModels
    except Exception as e:
        text_import_exception = e
        logger.error(f"❌ Failed to import TextAnalyticsModels: {e}")
        return
    
    # 모델 생성 시도
    try:
        wrapper = TextAnalyticsModels()
        model_obj, err = wrapper.create_sentiment_lstm()
        
        if model_obj is None:
            keras_model = None
            sentiment_model = wrapper
            logger.warning(f"⚠️ create_sentiment_lstm did not create a model: {err}")
        else:
            keras_model = model_obj
            sentiment_model = wrapper
            try:
                sentiment_model.sentiment_model = keras_model
            except Exception:
                pass
            logger.info("✅ Keras sentiment model created successfully.")
            
    except Exception as e:
        sentiment_model = None
        keras_model = None
        logger.error(f"❌ Failed to create keras model: {e}")


# ==================== Helper Functions ====================

POSITIVE_WORDS = {
    "love", "great", "excellent", "good", "fantastic", "happy", 
    "like", "enjoy", "awesome", "amazing"
}
NEGATIVE_WORDS = {
    "hate", "bad", "terrible", "awful", "worst", "sad", 
    "dislike", "disappointed", "poor", "angry"
}


def _simple_rule_sentiment(text: str) -> Dict[str, object]:
    """규칙 기반 감정 분석 (폴백용)
    
    Returns: {label: 'positive'|'negative'|'neutral', score: float in [-1,1]}
    """
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


# ==================== API Endpoints ====================

@text_router.get("/health")
async def text_health():
    """Text domain health check"""
    return {
        "status": "ok",
        "domain": "text",
        "model_loaded": sentiment_model is not None,
        "keras_model_available": keras_model is not None
    }


@text_router.post("/analyze")
async def analyze_sentiment(payload: TextInput):
    """감정 분석 수행
    
    Keras 모델이 있으면 사용, 없으면 규칙 기반 폴백
    
    Args:
        payload: 분석할 텍스트
        
    Returns:
        감정 레이블 및 신뢰도
    """
    if sentiment_model is None:
        if text_import_exception is not None:
            raise HTTPException(
                status_code=503, 
                detail=f"Model import failed: {text_import_exception}"
            )
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Keras 모델 사용
    if keras_model is not None:
        try:
            text = (payload.text or "").lower()
            tokens = re.findall(r"\w+", text)

            try:
                from config.settings import TextAnalyticsConfig
                vocab_size = getattr(TextAnalyticsConfig, "IMDB_VOCAB_SIZE", 10000)
                maxlen = getattr(TextAnalyticsConfig, "IMDB_MAX_LENGTH", 500)
            except Exception:
                vocab_size = 10000
                maxlen = 500

            seq = [abs(hash(w)) % vocab_size for w in tokens]
            if len(seq) < maxlen:
                seq = seq + [0] * (maxlen - len(seq))
            else:
                seq = seq[:maxlen]

            import numpy as np
            input_arr = np.array([seq])

            raw = keras_model.predict(input_arr)
            try:
                score = float(raw[0][0])
            except Exception:
                try:
                    score = float(np.asarray(raw).reshape(-1)[0])
                except Exception:
                    score = 0.0

            label = "positive" if score >= 0.5 else "negative"
            return {
                "sentiment": {"label": label, "score": score}, 
                "used_model": "keras_model"
            }

        except Exception as e:
            logger.exception(f"Keras model prediction failed: {e}")
            # 폴백으로 진행

    # 폴백: 규칙 기반
    fallback_result = _simple_rule_sentiment(payload.text)
    return {"sentiment": fallback_result, "used_model": "fallback"}
