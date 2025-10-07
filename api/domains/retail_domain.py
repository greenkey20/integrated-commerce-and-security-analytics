"""
Retail Analytics Domain API

Retail analytics endpoints and simple ML fallbacks following the project's
API pattern guide. This module intentionally avoids expensive imports at
module-import time and tries to fail gracefully when core.retail modules
are unavailable (so the API stays responsive in minimal environments).
"""
# 1. Imports
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
import logging

logger = logging.getLogger(__name__)

# 2. Pydantic Models
class RetailData(BaseModel):
    invoice_ids: List[str]
    descriptions: List[Optional[str]] = Field(default_factory=list)
    quantities: List[int]
    unit_prices: List[float]
    customer_ids: List[Optional[str]] = Field(default_factory=list)
    countries: List[Optional[str]] = Field(default_factory=list)

class AnalyzeRequest(BaseModel):
    data: RetailData
    analysis_type: str = Field(default="sales", description="sales|product|customer")

class PredictRequest(BaseModel):
    data: RetailData
    horizon: int = Field(default=7, description="Forecast horizon (days)")
    method: Optional[str] = Field(default=None, description="Optional forecasting method")

class ReportRequest(BaseModel):
    period: str = Field(default="monthly", description="daily|weekly|monthly")

# 3. Router Setup
retail_router = APIRouter(prefix="/retail", tags=["retail"])

# 4. Global State
retail_analyzer: Optional[Any] = None
retail_modeler: Optional[Any] = None
retail_module_available: bool = False
retail_import_error: Optional[str] = None

# 5. Initialization Function
async def initialize_retail_module():
    """Retail Analytics 모듈 초기화"""
    global retail_analyzer, retail_modeler, retail_module_available, retail_import_error

    try:
        # Import lazily so missing optional deps don't break server import
        from core.retail.analysis import RetailAnalyzer
        from core.retail.modeling import RetailModeling

        retail_analyzer = RetailAnalyzer()
        retail_modeler = RetailModeling()
        retail_module_available = True
        logger.info("✅ Retail modules loaded successfully.")
    except Exception as e:
        retail_module_available = False
        retail_import_error = str(e)
        retail_analyzer = None
        retail_modeler = None
        logger.error(f"❌ Failed to load retail modules: {e}")

# 6. Helper Functions
def _sanitize_for_json(obj: Any) -> Any:
    """Convert numpy/pandas types and NaN/Inf into JSON-serializable values."""
    try:
        import numpy as np
        import math
    except Exception:
        np = None
        import math

    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if np is not None and isinstance(obj, (np.integer, np.floating)):
        try:
            if np.isnan(obj) or np.isinf(obj):
                return None
        except Exception:
            pass
        return float(obj)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    # pandas Series/DataFrame -> convert to dict/list
    try:
        import pandas as pd
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if isinstance(obj, pd.Series):
            return obj.tolist()
    except Exception:
        pass

    return obj


def _calculate_basic_metrics(data: RetailData) -> Dict[str, Any]:
    """Calculate basic retail metrics from raw RetailData.

    Returns a dictionary with total_sales, total_orders, avg_order_value,
    revenue_per_customer (dict) and unique_customers_count.
    """
    # defensive lengths
    n = max(1, len(data.quantities))
    total_sales = 0.0
    order_counts = {}
    revenue_per_customer = {}

    for i in range(len(data.quantities)):
        qty = data.quantities[i]
        price = data.unit_prices[i] if i < len(data.unit_prices) else 0.0
        cust = data.customer_ids[i] if i < len(data.customer_ids) else None

        line = (qty or 0) * (price or 0.0)
        total_sales += line

        if cust:
            revenue_per_customer[cust] = revenue_per_customer.get(cust, 0.0) + line
            order_counts[cust] = order_counts.get(cust, 0) + 1

    total_orders = sum(order_counts.values()) if order_counts else len(data.invoice_ids) if len(data.invoice_ids) > 0 else 0
    avg_order_value = (total_sales / total_orders) if total_orders > 0 else (total_sales / n)

    return {
        "total_sales": float(total_sales),
        "total_orders": int(total_orders),
        "avg_order_value": float(avg_order_value),
        "unique_customers_count": int(len(revenue_per_customer)),
        "revenue_per_customer": {k: float(v) for k, v in revenue_per_customer.items()}
    }


def _simple_sales_forecast(data: RetailData, horizon: int = 7) -> Dict[str, Any]:
    """A very simple rule-based forecast: use average daily sales per invoice
    and project forward by horizon days. This is purely fallback logic.
    """
    # Sum total sales per invoice id (best-effort)
    sales_by_invoice = {}
    for i in range(len(data.invoice_ids)):
        inv = data.invoice_ids[i]
        qty = data.quantities[i]
        price = data.unit_prices[i] if i < len(data.unit_prices) else 0.0
        sales_by_invoice[inv] = sales_by_invoice.get(inv, 0.0) + (qty or 0) * (price or 0.0)

    if len(sales_by_invoice) == 0:
        avg_order = 0.0
    else:
        avg_order = sum(sales_by_invoice.values()) / len(sales_by_invoice)

    # naive forecast: horizon * avg_order
    forecast_total = avg_order * horizon

    # produce simple per-day series
    per_day = [round(forecast_total / max(1, horizon), 2) for _ in range(horizon)]

    return {
        "horizon": horizon,
        "total_forecast": float(round(forecast_total, 2)),
        "daily": per_day,
        "method": "simple_avg_per_invoice"
    }

# 7. API Endpoints
@retail_router.get("/health")
async def retail_health():
    """Retail domain health check"""
    return {
        "status": "ok",
        "domain": "retail",
        "module_available": retail_module_available,
        "analyzer_loaded": retail_analyzer is not None
    }


@retail_router.post("/analyze")
async def analyze_retail(request: AnalyzeRequest):
    """리테일 데이터 분석

    Returns: sales/product/customer analysis depending on analysis_type.
    """
    if retail_analyzer is None:
        basic_metrics = _calculate_basic_metrics(request.data)
        return {
            "status": "success",
            "mode": "fallback",
            "metrics": _sanitize_for_json(basic_metrics),
            "message": "Basic metrics only (ML model not available)"
        }

    try:
        import pandas as pd

        df = pd.DataFrame({
            'InvoiceNo': request.data.invoice_ids,
            'Description': request.data.descriptions,
            'Quantity': request.data.quantities,
            'UnitPrice': request.data.unit_prices,
            'CustomerID': request.data.customer_ids,
            'Country': request.data.countries
        })

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
            "metrics": _sanitize_for_json(basic_metrics),
            "error": str(e),
            "message": "ML failed, using basic metrics"
        }


@retail_router.post("/predict")
async def predict_retail(request: PredictRequest):
    """매출 예측 (간단한 예측 API)"""
    if retail_modeler is None:
        simple_forecast = _simple_sales_forecast(request.data, horizon=request.horizon)
        return {
            "status": "success",
            "mode": "fallback",
            "forecast": _sanitize_for_json(simple_forecast),
            "message": "Simple forecast (ML model not available)"
        }

    try:
        # Assume retail_modeler has a `forecast` method that accepts pandas DataFrame and horizon
        import pandas as pd

        df = pd.DataFrame({
            'InvoiceNo': request.data.invoice_ids,
            'Quantity': request.data.quantities,
            'UnitPrice': request.data.unit_prices,
            'CustomerID': request.data.customer_ids
        })

        forecast = retail_modeler.forecast(df, horizon=request.horizon, method=request.method)
        return {
            "status": "success",
            "mode": "ml_model",
            "forecast": _sanitize_for_json(forecast),
            "message": "Forecast completed"
        }
    except Exception as e:
        logger.exception(f"Retail forecast failed: {e}")
        simple_forecast = _simple_sales_forecast(request.data, horizon=request.horizon)
        return {
            "status": "partial",
            "mode": "fallback",
            "forecast": _sanitize_for_json(simple_forecast),
            "error": str(e),
            "message": "ML failed, using simple forecast"
        }


@retail_router.get("/report")
async def get_retail_report(period: str = "monthly"):
    """리테일 리포트 생성

    period: daily|weekly|monthly
    """
    if retail_analyzer is None:
        raise HTTPException(status_code=503, detail="Retail analyzer not loaded")

    try:
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

