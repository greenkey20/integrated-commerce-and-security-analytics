"""
Security Analytics Domain API

Provides endpoints for anomaly detection and monitoring using a UnifiedDetectionEngine
from core.security.detection_engine. Implements ML + rule-based fallback behavior and
JSON sanitization helpers.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Any
import logging

logger = logging.getLogger(__name__)

# ==================== Pydantic Models ====================

class NetworkTraffic(BaseModel):
    source_ips: List[str]
    dest_ips: List[str]
    ports: List[int]
    protocols: List[str]
    packet_sizes: List[int]
    timestamps: List[str]

class DetectRequest(BaseModel):
    data: NetworkTraffic
    # optional thresholds or metadata can be added later

class MonitorRequest(BaseModel):
    data: NetworkTraffic
    interval_seconds: Optional[int] = Field(default=60, ge=1, description="monitoring interval")

# ==================== Router Setup ====================

security_router = APIRouter(prefix="/security", tags=["security"])

# ==================== Global State ====================

security_detector: Optional[Any] = None
security_module_available: bool = False
security_import_error: Optional[str] = None

# ==================== Initialization Function ====================

async def initialize_security_module():
    """Security Analytics 모듈 초기화"""
    global security_detector, security_module_available, security_import_error

    try:
        from core.security.detection_engine import UnifiedDetectionEngine
        security_detector = UnifiedDetectionEngine()
        security_module_available = True
        logger.info("✅ Security UnifiedDetectionEngine loaded successfully.")
    except Exception as e:
        security_module_available = False
        security_import_error = str(e)
        security_detector = None
        logger.error(f"❌ Failed to load UnifiedDetectionEngine: {e}")

# ==================== Helper Functions ====================

def _sanitize_for_json(obj):
    """NaN/Infinity -> None, numpy types -> native python types for JSON serialization"""
    try:
        import numpy as np
        import math
    except Exception:
        np = None
        import math

    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(item) for item in obj]
    elif np is not None and isinstance(obj, (np.integer,)):
        return int(obj)
    elif np is not None and isinstance(obj, (np.floating,)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


def _simple_anomaly_detection(data: NetworkTraffic) -> List[bool]:
    """Rule-based anomaly detection fallback.

    Simple heuristics used as a fallback:
    - packet_size > 10000 -> anomaly
    - destination port in suspicious ports -> anomaly
    - protocol value not in common set -> anomaly
    """
    suspicious_ports = {22, 23, 3389}  # SSH, Telnet, RDP as examples
    common_protocols = {"tcp", "udp", "icmp"}

    anomalies: List[bool] = []
    n = len(data.source_ips)
    for i in range(n):
        try:
            pkt = data.packet_sizes[i]
        except Exception:
            pkt = 0
        try:
            port = int(data.ports[i])
        except Exception:
            port = -1
        try:
            proto = (data.protocols[i] or "").lower()
        except Exception:
            proto = ""

        is_anomaly = False
        if pkt > 10000:
            is_anomaly = True
        if port in suspicious_ports:
            is_anomaly = True
        if proto and proto not in common_protocols:
            # unknown protocol seen
            is_anomaly = True
        anomalies.append(is_anomaly)

    return anomalies

# ==================== API Endpoints ====================

@security_router.get("/health")
async def security_health():
    """Security domain health check"""
    return {
        "status": "ok",
        "domain": "security",
        "module_available": security_module_available,
        "detector_loaded": security_detector is not None,
        "import_error": security_import_error,
    }


@security_router.post("/detect")
async def detect_anomalies(request: DetectRequest):
    """네트워크 이상 탐지"""
    if security_detector is None:
        # fallback
        simple_anomalies = _simple_anomaly_detection(request.data)
        return {
            "status": "success",
            "mode": "fallback",
            "anomalies": simple_anomalies,
            "message": "Rule-based detection (ML model not available)"
        }

    try:
        import pandas as pd
        df = pd.DataFrame({
            'src_ip': request.data.source_ips,
            'dst_ip': request.data.dest_ips,
            'port': request.data.ports,
            'protocol': request.data.protocols,
            'packet_size': request.data.packet_sizes,
            'timestamp': request.data.timestamps
        })

        # Try to call a common detection interface; defensively handle API differences
        if hasattr(security_detector, 'detect_anomalies'):
            anomalies, scores = security_detector.detect_anomalies(df)
        elif hasattr(security_detector, 'detect'):
            result = security_detector.detect(df)
            # expect dict-like with 'anomalies' and 'scores'
            anomalies = result.get('anomalies') if isinstance(result, dict) else None
            scores = result.get('scores') if isinstance(result, dict) else None
        else:
            # detector present but no standard method; fallback to simple rule
            raise RuntimeError("Security detector missing detect interface")

        # Normalize outputs
        try:
            anomalies_list = anomalies.tolist() if hasattr(anomalies, 'tolist') else list(anomalies)
        except Exception:
            anomalies_list = anomalies if isinstance(anomalies, list) else []

        return {
            "status": "success",
            "mode": "ml_model",
            "anomalies": anomalies_list,
            "anomaly_scores": _sanitize_for_json(scores),
            "total_anomalies": int(sum(1 for a in anomalies_list if a)),
            "message": "ML anomaly detection completed"
        }

    except Exception as e:
        logger.exception(f"Anomaly detection failed: {e}")
        # fallback
        simple_anomalies = _simple_anomaly_detection(request.data)
        return {
            "status": "partial",
            "mode": "fallback",
            "anomalies": simple_anomalies,
            "error": str(e),
            "message": "ML failed, using rule-based detection"
        }


@security_router.post("/monitor")
async def monitor_traffic(request: MonitorRequest):
    """실시간 트래픽 모니터링"""
    if security_detector is None:
        raise HTTPException(status_code=503, detail="Security detector not loaded")

    try:
        import pandas as pd
        df = pd.DataFrame({
            'src_ip': request.data.source_ips,
            'dst_ip': request.data.dest_ips,
            'port': request.data.ports,
            'protocol': request.data.protocols,
            'packet_size': request.data.packet_sizes,
            'timestamp': request.data.timestamps
        })

        # If the detector exposes a monitoring method, use it; otherwise provide a simple status
        if hasattr(security_detector, 'monitor'):
            monitor_result = security_detector.monitor(df, interval_seconds=request.interval_seconds)
            return {
                "status": "success",
                "mode": "ml_model",
                "monitor_result": _sanitize_for_json(monitor_result),
                "message": "Monitoring executed"
            }
        elif hasattr(security_detector, 'monitor_traffic'):
            monitor_result = security_detector.monitor_traffic(df, interval_seconds=request.interval_seconds)
            return {
                "status": "success",
                "mode": "ml_model",
                "monitor_result": _sanitize_for_json(monitor_result),
                "message": "Monitoring executed"
            }
        else:
            # No monitor API; return a lightweight ML-driven snapshot detection
            if hasattr(security_detector, 'detect_anomalies'):
                anomalies, scores = security_detector.detect_anomalies(df)
                try:
                    anomalies_list = anomalies.tolist() if hasattr(anomalies, 'tolist') else list(anomalies)
                except Exception:
                    anomalies_list = anomalies if isinstance(anomalies, list) else []
                return {
                    "status": "success",
                    "mode": "ml_model",
                    "anomalies": anomalies_list,
                    "anomaly_scores": _sanitize_for_json(scores),
                    "message": "Snapshot detection executed (no continuous monitor API)"
                }
            return {
                "status": "success",
                "mode": "ml_model",
                "message": "Monitor API not implemented on detector; call detect instead"
            }

    except Exception as e:
        logger.exception(f"Monitoring failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
