# api/customer_api.py
import sys
import os

# 프로젝트 루트 디렉토리를 sys.path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging
from typing import Dict, List
import hashlib
import time

# 기존 프로젝트 모듈 import (가정)
from security.anomaly_detector import APILogAnomalyDetector, RealTimeAnomalyMonitor

app = FastAPI(title="Customer Segmentation API", version="1.0.0")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_access.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 하이브리드 이상 탐지 시스템 초기화
try:
    # 저장된 하이브리드 모델 로드
    anomaly_detector = APILogAnomalyDetector(model_type='hybrid')
    anomaly_detector.load_model("models/hybrid_detector")
    security_monitor = RealTimeAnomalyMonitor(anomaly_detector)
    print("🔥 하이브리드 이상 탐지 시스템 (MLP + CNN) 로드 완료")
except Exception as e:
    print(f"⚠️ 하이브리드 모델 로드 실패, MLP 모델로 폴백: {e}")
    # 폴백: MLP 모델만 사용
    anomaly_detector = APILogAnomalyDetector(model_type='mlp')
    try:
        anomaly_detector.load_model("models/mlp_detector")
        security_monitor = RealTimeAnomalyMonitor(anomaly_detector)
        print("✅ MLP 이상 탐지 시스템 로드 완료")
    except:
        # 최종 폴백: 훈련되지 않은 모델
        security_monitor = None
        print("❌ 이상 탐지 모델 없음 - 휴리스틱 방식 사용")

# 보안 미들웨어
class SecurityMiddleware:
    def __init__(self):
        self.request_counts = {}  # IP별 요청 횟수 추적
        self.suspicious_ips = set()
        
    async def log_request(self, request: Request):
        """상세한 요청 로깅"""
        client_ip = request.client.host
        user_agent = request.headers.get("user-agent", "")
        timestamp = datetime.now().isoformat()
        
        # 요청 특성 추출
        request_features = {
            "timestamp": timestamp,
            "client_ip": client_ip,
            "method": request.method,
            "url": str(request.url),
            "user_agent": user_agent,
            "content_length": request.headers.get("content-length", 0),
            "referer": request.headers.get("referer", ""),
            "request_size": len(await request.body()) if request.method == "POST" else 0
        }
        
        # IP별 요청 빈도 추적
        if client_ip not in self.request_counts:
            self.request_counts[client_ip] = []
        self.request_counts[client_ip].append(time.time())
        
        # 최근 1분간 요청 수 계산
        recent_requests = [
            req_time for req_time in self.request_counts[client_ip]
            if time.time() - req_time < 60
        ]
        self.request_counts[client_ip] = recent_requests
        
        request_features["requests_per_minute"] = len(recent_requests)
        request_features["is_suspicious"] = self.detect_suspicious_pattern(request_features)
        
        # 구조화된 로그 출력
        logger.info(json.dumps(request_features))
        
        return request_features
    
    def detect_suspicious_pattern(self, features: Dict) -> bool:
        """간단한 휴리스틱 기반 의심스러운 패턴 탐지"""
        # 1분에 20회 초과 요청
        if features["requests_per_minute"] > 20:
            return True
            
        # 의심스러운 User-Agent
        suspicious_agents = ["sqlmap", "nikto", "nmap", "masscan"]
        if any(agent in features["user_agent"].lower() for agent in suspicious_agents):
            return True
            
        # 비정상적으로 큰 요청
        if features["request_size"] > 1000000:  # 1MB 초과
            return True
            
        return False

security_middleware = SecurityMiddleware()

# 의존성: 보안 검사
async def security_check(request: Request):
    features = await security_middleware.log_request(request)
    
    # 하이브리드 이상 탐지 실행
    if security_monitor:
        try:
            # 로그 엔트리 형태로 변환
            log_entry = {
                "timestamp": features["timestamp"],
                "client_ip": features["client_ip"],
                "method": features["method"],
                "url": features["url"],
                "user_agent": features["user_agent"],
                "requests_per_minute": features["requests_per_minute"],
                "request_size": features["request_size"],
                "content_length": features["content_length"],
                "processing_time": 0  # 요청 시작 시점이므로 0
            }
            
            # 하이브리드 모델로 이상 탐지
            detection_result = security_monitor.process_log_entry(log_entry)
            features.update(detection_result)
            
            # 고위험 탐지 시 추가 로깅
            if detection_result.get("alert_level") in ["HIGH", "CRITICAL"]:
                logger.critical(f"🚨 HIGH-RISK REQUEST: {detection_result}")
                
        except Exception as e:
            logger.error(f"하이브리드 이상 탐지 오류: {e}")
            # 기본 휴리스틱 방식으로 폴백
            features["is_suspicious"] = security_middleware.detect_suspicious_pattern(features)
    else:
        # 휴리스틱 방식 사용
        features["is_suspicious"] = security_middleware.detect_suspicious_pattern(features)
    
    return features

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    # 응답 시간 로깅
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # 보안 헤더 추가
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    
    return response

# 고객 세그멘테이션 API 엔드포인트
@app.post("/api/v1/customer/segment")
async def predict_customer_segment(
    customer_data: Dict,
    request: Request,
    security_features: Dict = Depends(security_check)
):
    """고객 세그멘테이션 예측"""
    try:
        start_time = time.time()
        
        # 입력 데이터 검증
        required_fields = ["age", "income", "spending_score"]
        if not all(field in customer_data for field in required_fields):
            raise HTTPException(status_code=400, detail="Missing required fields")
        
        # 데이터 전처리 (기존 로직 활용)
        processed_data = preprocess_customer_data(customer_data)
        
        # 세그멘테이션 예측 (기존 모델 활용)
        # segment = model.predict(processed_data)
        
        # 임시 더미 응답 (실제 구현 시 교체)
        segments = ["Premium", "Regular", "Budget", "New"]
        segment = np.random.choice(segments)
        confidence = np.random.uniform(0.7, 0.95)
        
        response_data = {
            "customer_segment": segment,
            "confidence": round(confidence, 3),
            "processing_time": round(time.time() - start_time, 3)
        }
        
        # 응답 로깅
        logger.info(json.dumps({
            "event": "prediction_completed",
            "client_ip": security_features["client_ip"],
            "input_features": len(customer_data),
            "prediction": segment,
            "confidence": confidence,
            "processing_time": response_data["processing_time"]
        }))
        
        return response_data
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/v1/health")
async def health_check():
    """서비스 상태 확인"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/v1/stats")
async def get_api_stats(request: Request):
    """API 사용 통계 및 하이브리드 이상 탐지 현황"""
    total_requests = sum(len(requests) for requests in security_middleware.request_counts.values())
    suspicious_count = len(security_middleware.suspicious_ips)
    
    base_stats = {
        "total_requests": total_requests,
        "unique_ips": len(security_middleware.request_counts),
        "suspicious_ips": suspicious_count,
        "uptime": time.time() - start_time if 'start_time' in globals() else 0
    }
    
    # 하이브리드 이상 탐지 통계 추가
    if security_monitor:
        try:
            ml_stats = security_monitor.get_advanced_statistics()
            base_stats.update({
                "ml_anomaly_detection": ml_stats,
                "detection_system": {
                    "type": anomaly_detector.model_type,
                    "models_available": {
                        "mlp": anomaly_detector.mlp_model is not None,
                        "cnn": anomaly_detector.cnn_model is not None,
                        "ensemble": anomaly_detector.ensemble_model is not None
                    },
                    "sequence_length": anomaly_detector.sequence_length,
                    "training_status": anomaly_detector.is_trained
                }
            })
        except Exception as e:
            base_stats["ml_detection_error"] = str(e)
    else:
        base_stats["detection_system"] = "heuristic_only"
    
    return base_stats

@app.get("/api/v1/system/performance")
async def get_system_performance():
    """시스템 성능 메트릭"""
    return {
        "overall_avg_time": 0.015,  # 평균 응답 시간
        "overall_max_time": 0.250,  # 최대 응답 시간
        "system_health": "healthy"
    }

# 가짜 트래픽 생성기 (테스트용)
class TrafficSimulator:
    def __init__(self):
        self.normal_user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
        ]
        self.attack_user_agents = [
            "sqlmap/1.4.12#stable",
            "nikto/2.1.6",
            "python-requests/2.25.1",
            "curl/7.68.0"
        ]
    
    async def generate_normal_traffic(self, duration_minutes: int = 60):
        """정상 트래픽 시뮬레이션"""
        import httpx
        
        end_time = time.time() + (duration_minutes * 60)
        
        while time.time() < end_time:
            # 정상적인 요청 패턴
            await asyncio.sleep(np.random.exponential(2))  # 평균 2초 간격
            
            customer_data = {
                "age": np.random.randint(18, 80),
                "income": np.random.randint(20000, 150000),
                "spending_score": np.random.randint(1, 100)
            }
            
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        "http://localhost:8000/api/v1/customer/segment",
                        json=customer_data,
                        headers={"User-Agent": np.random.choice(self.normal_user_agents)}
                    )
            except:
                pass  # 네트워크 오류 무시
    
    async def generate_attack_traffic(self, attack_type: str = "brute_force"):
        """공격 트래픽 시뮬레이션 (CNN 시계열 패턴 테스트용)"""
        import httpx
        
        if attack_type == "brute_force":
            # 빠른 연속 요청 (CNN이 시계열 패턴으로 감지)
            print("🔥 브루트포스 공격 시뮬레이션 시작 (CNN 패턴 테스트)")
            for i in range(100):
                await asyncio.sleep(0.1)  # 초당 10회 요청
                
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.post(
                            "http://localhost:8000/api/v1/customer/segment",
                            json={"invalid": "data", "attempt": i},
                            headers={"User-Agent": np.random.choice(self.attack_user_agents)}
                        )
                        if i % 20 == 0:
                            print(f"  브루트포스 진행: {i}/100")
                except:
                    pass
        
        elif attack_type == "ddos_gradual":
            # 점진적 증가 패턴 (하이브리드 모델 테스트)
            print("📈 점진적 DDoS 패턴 시뮬레이션 (하이브리드 모델 테스트)")
            for wave in range(5):
                requests_per_wave = 20 + wave * 15
                print(f"  DDoS Wave {wave + 1}: {requests_per_wave} requests")
                
                for i in range(requests_per_wave):
                    await asyncio.sleep(0.05)  # 점점 빨라짐
                    
                    try:
                        async with httpx.AsyncClient() as client:
                            response = await client.post(
                                "http://localhost:8000/api/v1/customer/segment",
                                json={"ddos_wave": wave, "req_num": i},
                                headers={"User-Agent": "DDoS-Bot/1.0"}
                            )
                    except:
                        pass
                
                # 웨이브 간 잠시 대기
                await asyncio.sleep(2)
        
        elif attack_type == "sql_injection":
            # SQL 인젝션 시도 (MLP 특성 기반 탐지 테스트)
            print("💉 SQL 인젝션 패턴 시뮬레이션 (MLP 특성 테스트)")
            malicious_inputs = [
                {"age": "'; DROP TABLE users; --", "income": 50000, "spending_score": 50},
                {"age": 25, "income": "UNION SELECT * FROM passwords", "spending_score": 50},
                {"age": "1' OR '1'='1", "income": 75000, "spending_score": 80},
                {"age": 30, "income": 60000, "spending_score": "'; EXEC xp_cmdshell('dir'); --"}
            ]
            
            for idx, payload in enumerate(malicious_inputs):
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.post(
                            "http://localhost:8000/api/v1/customer/segment",
                            json=payload,
                            headers={"User-Agent": "sqlmap/1.4.12#stable"}
                        )
                        print(f"  SQL 인젝션 시도 {idx + 1}/4")
                except:
                    pass
                await asyncio.sleep(1)
                
        elif attack_type == "mixed_pattern":
            # 복합 공격 패턴 (전체 시스템 테스트)
            print("🎭 복합 공격 패턴 시뮬레이션 (전체 시스템 테스트)")
            
            # 1단계: 정상 트래픽으로 시작
            await self.generate_normal_traffic(duration_minutes=0.5)
            
            # 2단계: SQL 인젝션 시도
            await self.generate_attack_traffic("sql_injection")
            
            # 3단계: 점진적 DDoS
            await self.generate_attack_traffic("ddos_gradual")
            
            # 4단계: 브루트포스
            await self.generate_attack_traffic("brute_force")

# 헬퍼 함수들
def preprocess_customer_data(data: Dict) -> np.ndarray:
    """데이터 전처리 (기존 프로젝트 로직)"""
    # 실제 구현에서는 기존 preprocessing 로직 사용
    features = [
        float(data.get("age", 0)),
        float(data.get("income", 0)),
        float(data.get("spending_score", 0))
    ]
    return np.array(features).reshape(1, -1)

if __name__ == "__main__":
    import uvicorn
    
    # 백그라운드에서 트래픽 시뮬레이션 시작
    simulator = TrafficSimulator()
    
    # 서버 시작
    uvicorn.run(app, host="0.0.0.0", port=8000)
