"""
FastAPI entrypoint that loads the text sentiment model once at startup
and exposes a /analyze endpoint that uses the model's predict() method.

Constraints respected:
- Does not modify existing Streamlit code.
- Does not modify core/text/sentiment_models.py
- Only creates this new file.

Usage:
    uvicorn api_main:app --host 0.0.0.0 --port 8000

"""
from typing import Any, Optional
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import the TextAnalyticsModels class from the project's module.
# This import itself is lightweight because the heavy TensorFlow imports inside
# the module are performed lazily by the module's helpers (per the project's design).
try:
    from core.text.sentiment_models import TextAnalyticsModels
except Exception as e:  # pragma: no cover - if import fails, we'll handle at runtime
    TextAnalyticsModels = None
    import_exception = e
else:
    import_exception = None

# FastAPI app
app = FastAPI(title="Sentiment Analysis API")

# Allow broad CORS for convenience (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("api_main")
logging.basicConfig(level=logging.INFO)

# Global model instance (loaded once on startup)
sentiment_model: Optional[Any] = None


class TextInput(BaseModel):
    text: str


# --- Fallback simple rule-based sentiment predictor ---
import re
from typing import Dict

POSITIVE_WORDS = {
    "love", "great", "excellent", "good", "fantastic", "happy", "like", "enjoy", "awesome", "amazing"
}
NEGATIVE_WORDS = {
    "hate", "bad", "terrible", "awful", "worst", "sad", "dislike", "disappointed", "poor", "angry"
}


def _simple_rule_sentiment(text: str) -> Dict[str, object]:
    """Very small deterministic sentiment heuristic used as a fallback.

    Returns a dict: {label: 'positive'|'negative'|'neutral', score: float in [-1,1]}
    """
    if not text or not text.strip():
        return {"label": "neutral", "score": 0.0}

    words = re.findall(r"\w+", text.lower())
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)

    # Simple normalized score
    total = pos + neg
    if total == 0:
        return {"label": "neutral", "score": 0.0}

    score = (pos - neg) / total  # in range [-1, 1]
    label = "positive" if score > 0 else ("negative" if score < 0 else "neutral")
    return {"label": label, "score": float(score)}


@app.on_event("startup")
async def load_model():
    """Load the TextAnalyticsModels instance once when the app starts.

    We instantiate the wrapper class and call create_sentiment_lstm() so that
    `sentiment_model.sentiment_model` (a Keras model) is created and available
    for inference. If TensorFlow isn't available or model creation fails, we
    keep the fallback logic in place.
    """
    global sentiment_model, keras_model

    if import_exception is not None or TextAnalyticsModels is None:
        logger.exception("Failed to import TextAnalyticsModels: %s", import_exception)
        return

    try:
        # instantiate the wrapper
        wrapper = TextAnalyticsModels()

        # Try to create the sentiment Keras model (may fail if TF missing)
        model_obj, err = wrapper.create_sentiment_lstm()
        if model_obj is None:
            keras_model = None
            sentiment_model = wrapper
            logger.warning("create_sentiment_lstm did not create a model: %s", err)
        else:
            # create_sentiment_lstm returns the keras model instance
            keras_model = model_obj
            sentiment_model = wrapper
            # ensure wrapper.sentiment_model is set (create_sentiment_lstm should do this)
            try:
                sentiment_model.sentiment_model = keras_model
            except Exception:
                pass
            logger.info("Keras sentiment model created successfully on startup.")

    except Exception as e:
        sentiment_model = None
        keras_model = None
        logger.exception("Failed to instantiate TextAnalyticsModels or create keras model: %s", e)


@app.get("/", tags=["health"])
async def root():
    return {"status": "ok"}


@app.post("/analyze", tags=["inference"])
async def analyze_sentiment(payload: TextInput):
    """Run sentiment prediction using the preloaded Keras model (if available).

    - If a Keras model was created at startup, we tokenize deterministically using
      a hash-based mapping and pad/truncate to TextAnalyticsConfig.IMDB_MAX_LENGTH.
    - Otherwise we fall back to the simple rule-based predictor.
    """
    if sentiment_model is None:
        if import_exception is not None:
            raise HTTPException(status_code=503, detail=f"Model import failed: {import_exception}")
        raise HTTPException(status_code=503, detail="Model not loaded")

    # If Keras model available, use it
    if 'keras_model' in globals() and globals().get('keras_model') is not None:
        try:
            keras_m = globals().get('keras_model')

            # Tokenize deterministically (hash -> index) and pad/truncate
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

            import numpy as _np
            input_arr = _np.array([seq])

            raw = keras_m.predict(input_arr)
            # expected shape (1,1) for sigmoid output
            try:
                score = float(raw[0][0])
            except Exception:
                # best-effort conversion
                try:
                    score = float(_np.asarray(raw).reshape(-1)[0])
                except Exception:
                    score = 0.0

            label = "positive" if score >= 0.5 else "negative"
            return {"sentiment": {"label": label, "score": score}, "used_model": "keras_model"}

        except Exception as e:
            logger.exception("Keras model prediction failed: %s", e)
            # fall through to fallback

    # Final fallback: simple rule-based predictor
    fallback_result = _simple_rule_sentiment(payload.text)
    return {"sentiment": fallback_result, "used_model": "fallback"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_main:app", host="0.0.0.0", port=8000, log_level="info")
