# run_hybrid_system.py
"""
하이브리드 API 로그 이상 탐지 시스템 실행 스크립트
07-2.ipynb (MLP) + 08-2.ipynb (CNN) 결합 버전
"""

import asyncio
import subprocess
import sys
import time
import requests
import json
from pathlib import Path
import multiprocessing as mp
from datetime import datetime
import signal
import os

class HybridSystemLauncher:
    """하이브리드 이상 탐지 시스템 런처"""
    
    def __init__(self):
        self.processes = []
        self.base_dir = Path.cwd()
        self.models_dir = self.base_dir / "models"
        self.data_dir = self.base_dir / "data"
        
    def check_requirements(self):
        """필요 패키지 및 파일 확인"""
        print("🔍 시스템 요구사항 확인 중...")
        
        required_packages = [
            'fastapi', 'uvicorn', 'tensorflow', 'scikit-learn', 
            'pandas', 'numpy', 'httpx', 'joblib'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
                print(f"  ✅ {package}")
            except ImportError:
                missing_packages.append(package)
                print(f"  ❌ {package}")
        
        if missing_packages:
            print(f"\n🚨 누락된 패키지: {missing_packages}")
            print("다음 명령어로 설치하세요:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
        
        # 디렉토리 생성
        self.models_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        
        print("✅ 모든 요구사항이 충족되었습니다!")
        return True
    
    def setup_models(self):
        """모델 초기 설정 및 훈련"""
        print("\n🤖 하이브리드 모델 설정 중...")
        print("📚 07-2.ipynb MLP + 08-2.ipynb CNN 아키텍처")
        
        # 데이터 준비 및 모델 훈련 스크립트 실행
        setup_script = """
import sys
sys.path.append('.')

from data.cicids_data_loader import setup_complete_system

print("🔥 하이브리드 시스템 훈련 시작...")
detector = setup_complete_system()
print("✅ 모델 훈련 및 저장 완료!")
"""
        
        try:
            with open('temp_setup.py', 'w') as f:
                f.write(setup_script)
            
            result = subprocess.run([sys.executable, 'temp_setup.py'], 
                                  capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✅ 하이브리드 모델 설정 완료!")
                print("  - MLP 모델 (07-2.ipynb 스타일): 개별 특성 분석")
                print("  - CNN 모델 (08-2.ipynb 스타일): 시계열 패턴 분석") 
                print("  - 하이브리드 모델: 두 방식 결합")
                return True
            else:
                print(f"❌ 모델 설정 실패: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("⏰ 모델 훈련 시간 초과 (5분)")
            return False
        except Exception as e:
            print(f"❌ 모델 설정 오류: {e}")
            return False
        finally:
            # 임시 파일 정리
            if Path('temp_setup.py').exists():
                Path('temp_setup.py').unlink()
    
    def start_api_server(self):
        """FastAPI 서버 시작"""
        print("\n🚀 하이브리드 API 서버 시작 중...")
        
        try:
            # 비동기로 서버 시작
            server_process = subprocess.Popen([
                sys.executable, '-m', 'uvicorn', 
                'api.customer_api:app',
                '--host', '0.0.0.0',
                '--port', '8000',
                '--reload'
            ])
            
            self.processes.append(server_process)
            
            # 서버 시작 대기
            print("⏳ 서버 시작 대기 중...")
            for i in range(30):
                try:
                    response = requests.get('http://localhost:8000/api/v1/health', timeout=2)
                    if response.status_code == 200:
                        print("✅ API 서버 시작 완료!")
                        print("🔗 서버 주소: http://localhost:8000")
                        print("📋 API 문서: http://localhost:8000/docs")
                        return True
                except:
                    time.sleep(1)
                    print(f"  대기 중... ({i+1}/30)")
            
            print("❌ 서버 시작 실패 (30초 초과)")
            return False
            
        except Exception as e:
            print(f"❌ 서버 시작 오류: {e}")
            return False
    
    def start_traffic_simulation(self):
        """트래픽 시뮬레이션 시작"""
        print("\n🌊 트래픽 시뮬레이션 시작...")
        
        simulation_script = """
import asyncio
import sys
sys.path.append('.')

from api.customer_api import TrafficSimulator

async def run_demo_traffic():
    simulator = TrafficSimulator()
    
    print("🟢 정상 트래픽 생성 중...")
    # 백그라운드에서 정상 트래픽 계속 생성
    normal_task = asyncio.create_task(
        simulator.generate_normal_traffic(duration_minutes=60)
    )
    
    # 5초 후 공격 패턴 시작
    await asyncio.sleep(5)
    
    print("🔥 공격 패턴 시뮬레이션...")
    print("  1. SQL 인젝션 (MLP 탐지)")
    await simulator.generate_attack_traffic("sql_injection")
    
    await asyncio.sleep(3)
    print("  2. 점진적 DDoS (CNN 시계열 탐지)")
    await simulator.generate_attack_traffic("ddos_gradual")
    
    await asyncio.sleep(3)
    print("  3. 브루트포스 (하이브리드 탐지)")
    await simulator.generate_attack_traffic("brute_force")
    
    print("✅ 공격 시뮬레이션 완료!")
    
    # 정상 트래픽 계속 실행
    await normal_task

if __name__ == "__main__":
    asyncio.run(run_demo_traffic())
"""
        
        try:
            with open('temp_simulation.py', 'w') as f:
                f.write(simulation_script)
            
            sim_process = subprocess.Popen([sys.executable, 'temp_simulation.py'])
            self.processes.append(sim_process)
            
            print("✅ 트래픽 시뮬레이션 시작!")
            print("  - 정상 패턴: 지속적 생성")
            print("  - 공격 패턴: SQL 인젝션 → DDoS → 브루트포스")
            
            return True
            
        except Exception as e:
            print(f"❌ 시뮬레이션 시작 오류: {e}")
            return False
    
    def monitor_system(self):
        """시스템 모니터링"""
        print("\n📊 실시간 모니터링 시작...")
        print("Ctrl+C를 눌러 시스템을 종료하세요.\n")
        
        try:
            while True:
                try:
                    # 시스템 상태 확인
                    health_response = requests.get('http://localhost:8000/api/v1/health', timeout=5)
                    stats_response = requests.get('http://localhost:8000/api/v1/stats', timeout=5)
                    
                    if health_response.status_code == 200 and stats_response.status_code == 200:
                        health_data = health_response.json()
                        stats_data = stats_response.json()
                        
                        # 현재 시간
                        current_time = datetime.now().strftime("%H:%M:%S")
                        
                        # 기본 통계
                        total_requests = stats_data.get('total_requests', 0)
                        unique_ips = stats_data.get('unique_ips', 0)
                        
                        # ML 이상 탐지 통계
                        ml_stats = stats_data.get('ml_anomaly_detection', {})
                        if isinstance(ml_stats, dict) and 'total_anomalies' in ml_stats:
                            anomalies = ml_stats['total_anomalies']
                            high_risk = ml_stats.get('high_risk_count', 0)
                            detection_method = stats_data.get('detection_system', {}).get('type', 'unknown')
                        else:
                            anomalies = 0
                            high_risk = 0
                            detection_method = 'heuristic'
                        
                        # 상태 출력
                        print(f"\r[{current_time}] "
                              f"요청: {total_requests:,} | "
                              f"IP: {unique_ips} | "
                              f"이상탐지: {anomalies} | "
                              f"고위험: {high_risk} | "
                              f"모델: {detection_method.upper()}", end="")
                        
                        # 고위험 알림
                        if high_risk > 0:
                            print(f"\n🚨 HIGH RISK DETECTED! {high_risk} 건의 고위험 패턴 감지됨")
                            
                            # 고위험 IP 목록
                            high_risk_ips = ml_stats.get('high_risk_ips', [])
                            if high_risk_ips:
                                print(f"   위험 IP: {', '.join(high_risk_ips[:5])}")
                    
                except requests.RequestException:
                    print(f"\r[{datetime.now().strftime('%H:%M:%S')}] ❌ 서버 연결 실패", end="")
                
                time.sleep(2)
                
        except KeyboardInterrupt:
            print("\n\n🛑 사용자에 의해 시스템 종료 요청됨")
    
    def cleanup(self):
        """프로세스 정리"""
        print("\n🧹 시스템 정리 중...")
        
        for process in self.processes:
            if process.poll() is None:  # 프로세스가 아직 실행 중
                print(f"  프로세스 종료 중: PID {process.pid}")
                process.terminate()
                
                # 강제 종료 대기
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"  강제 종료: PID {process.pid}")
                    process.kill()
        
        # 임시 파일 정리
        temp_files = ['temp_setup.py', 'temp_simulation.py']
        for temp_file in temp_files:
            if Path(temp_file).exists():
                Path(temp_file).unlink()
                print(f"  임시 파일 삭제: {temp_file}")
        
        print("✅ 시스템 정리 완료!")
    
    def run_complete_system(self):
        """전체 시스템 실행"""
        print("=" * 60)
        print("🔥 하이브리드 API 로그 이상 탐지 시스템")
        print("📚 07-2.ipynb (MLP) + 08-2.ipynb (CNN) 통합 버전")
        print("=" * 60)
        
        try:
            # 1. 요구사항 확인
            if not self.check_requirements():
                return False
            
            # 2. 모델 설정
            if not self.setup_models():
                print("❌ 모델 설정에 실패했습니다.")
                return False
            
            # 3. API 서버 시작
            if not self.start_api_server():
                print("❌ API 서버 시작에 실패했습니다.")
                return False
            
            # 4. 트래픽 시뮬레이션 시작
            if not self.start_traffic_simulation():
                print("⚠️ 트래픽 시뮬레이션 시작에 실패했지만 계속 진행합니다.")
            
            # 5. 시스템 모니터링
            self.monitor_system()
            
        except Exception as e:
            print(f"\n❌ 시스템 실행 중 오류 발생: {e}")
        finally:
            self.cleanup()
    
    def show_demo_results(self):
        """데모 결과 요약"""
        print("\n" + "=" * 60)
        print("📊 하이브리드 이상 탐지 시스템 데모 결과")
        print("=" * 60)
        
        try:
            # 최종 통계 가져오기
            stats_response = requests.get('http://localhost:8000/api/v1/stats', timeout=5)
            perf_response = requests.get('http://localhost:8000/api/v1/system/performance', timeout=5)
            
            if stats_response.status_code == 200:
                stats = stats_response.json()
                
                print(f"🔢 전체 요청 수: {stats.get('total_requests', 0):,}")
                print(f"🌐 고유 IP 수: {stats.get('unique_ips', 0)}")
                print(f"⚠️ 휴리스틱 탐지: {stats.get('suspicious_ips', 0)}")
                
                ml_stats = stats.get('ml_anomaly_detection', {})
                if isinstance(ml_stats, dict) and 'total_anomalies' in ml_stats:
                    print(f"🤖 ML 이상 탐지: {ml_stats['total_anomalies']}")
                    print(f"🚨 고위험 이벤트: {ml_stats.get('high_risk_count', 0)}")
                    print(f"📈 평균 이상 확률: {ml_stats.get('average_anomaly_probability', 0):.3f}")
                
                detection_system = stats.get('detection_system', {})
                if isinstance(detection_system, dict):
                    print(f"\n🧠 탐지 시스템 정보:")
                    print(f"  - 모델 타입: {detection_system.get('type', 'unknown').upper()}")
                    models = detection_system.get('models_available', {})
                    if models:
                        print(f"  - MLP: {'✅' if models.get('mlp') else '❌'}")
                        print(f"  - CNN: {'✅' if models.get('cnn') else '❌'}")
                        print(f"  - 하이브리드: {'✅' if models.get('ensemble') else '❌'}")
            
            if perf_response.status_code == 200:
                perf = perf_response.json()
                print(f"\n⚡ 성능 통계:")
                print(f"  - 평균 응답 시간: {perf.get('overall_avg_time', 0):.3f}초")
                print(f"  - 최대 응답 시간: {perf.get('overall_max_time', 0):.3f}초")
                print(f"  - 시스템 상태: {perf.get('system_health', 'unknown').upper()}")
        
        except:
            print("📊 통계 수집 실패 - 서버가 종료되었을 수 있습니다.")
        
        print("\n🎯 주요 성과:")
        print("✅ 07-2.ipynb MLP 모델: 개별 특성 기반 빠른 탐지")
        print("✅ 08-2.ipynb CNN 모델: 시계열 패턴 기반 순차 공격 탐지")
        print("✅ 하이브리드 모델: 두 방식의 장점을 결합한 최적 성능")
        print("✅ 실시간 모니터링: 즉각적인 위협 대응")
        
        print("\n🚀 다음 단계:")
        print("1. 실제 프로덕션 환경에 배포")
        print("2. 더 많은 공격 패턴 학습 데이터 추가")
        print("3. 알림 시스템 연동 (Slack, Email 등)")
        print("4. 대시보드 및 시각화 도구 구축")

def main():
    """메인 실행 함수"""
    launcher = HybridSystemLauncher()
    
    # 시그널 핸들러 등록 (Ctrl+C 처리)
    def signal_handler(signum, frame):
        print("\n🛑 종료 신호 감지됨...")
        launcher.cleanup()
        launcher.show_demo_results()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 시스템 실행
    launcher.run_complete_system()

if __name__ == "__main__":
    main()
