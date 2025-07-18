# data/cicids_data_loader.py
import pandas as pd
import numpy as np
import requests
import zipfile
import os
from pathlib import Path
import logging
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

class CICIDSDataLoader:
    """CICIDS2017 데이터셋 다운로드 및 전처리"""
    
    def __init__(self, data_dir: str = "data/cicids2017"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # CICIDS2017 데이터셋 URL (공식)
        self.dataset_urls = {
            "Monday": "https://www.unb.ca/cic/datasets/ids-2017.html",
            # 실제 다운로드는 수동으로 진행 (약 2.8GB)
        }
        
        # 컬럼 이름 정리 (공백 제거)
        self.column_mapping = {
            ' Destination Port': 'Destination_Port',
            ' Flow Duration': 'Flow_Duration',
            ' Total Fwd Packets': 'Total_Fwd_Packets',
            ' Total Backward Packets': 'Total_Backward_Packets',
            ' Label': 'Label'
        }
        
    def download_sample_data(self):
        """샘플 데이터 생성 (실제 CICIDS 패턴 시뮬레이션)"""
        print("CICIDS2017 스타일 샘플 데이터 생성 중...")
        
        np.random.seed(42)
        
        # 정상 트래픽 패턴 (80%)
        normal_samples = 8000
        normal_data = {
            'Flow_Duration': np.random.exponential(1000000, normal_samples),  # 마이크로초
            'Total_Fwd_Packets': np.random.poisson(10, normal_samples),
            'Total_Backward_Packets': np.random.poisson(8, normal_samples),
            'Total_Length_Fwd_Packets': np.random.normal(500, 200, normal_samples),
            'Total_Length_Bwd_Packets': np.random.normal(400, 150, normal_samples),
            'Fwd_Packet_Length_Max': np.random.normal(800, 300, normal_samples),
            'Fwd_Packet_Length_Min': np.random.normal(50, 20, normal_samples),
            'Fwd_Packet_Length_Mean': np.random.normal(200, 80, normal_samples),
            'Bwd_Packet_Length_Max': np.random.normal(700, 250, normal_samples),
            'Bwd_Packet_Length_Min': np.random.normal(40, 15, normal_samples),
            'Bwd_Packet_Length_Mean': np.random.normal(180, 70, normal_samples),
            'Flow_Bytes_s': np.random.normal(1000, 500, normal_samples),
            'Flow_Packets_s': np.random.normal(10, 5, normal_samples),
            'Flow_IAT_Mean': np.random.exponential(100000, normal_samples),
            'Flow_IAT_Std': np.random.exponential(50000, normal_samples),
            'Fwd_IAT_Total': np.random.exponential(500000, normal_samples),
            'Fwd_IAT_Mean': np.random.exponential(50000, normal_samples),
            'Bwd_IAT_Total': np.random.exponential(400000, normal_samples),
            'Bwd_IAT_Mean': np.random.exponential(45000, normal_samples),
            'Label': ['BENIGN'] * normal_samples
        }
        
        # DDoS 공격 패턴 (10%)
        ddos_samples = 1000
        ddos_data = {
            'Flow_Duration': np.random.exponential(100000, ddos_samples),  # 짧은 지속시간
            'Total_Fwd_Packets': np.random.poisson(100, ddos_samples),    # 많은 패킷
            'Total_Backward_Packets': np.random.poisson(2, ddos_samples), # 적은 응답
            'Total_Length_Fwd_Packets': np.random.normal(5000, 1000, ddos_samples),
            'Total_Length_Bwd_Packets': np.random.normal(100, 50, ddos_samples),
            'Fwd_Packet_Length_Max': np.random.normal(1500, 200, ddos_samples),
            'Fwd_Packet_Length_Min': np.random.normal(40, 10, ddos_samples),
            'Fwd_Packet_Length_Mean': np.random.normal(60, 20, ddos_samples),
            'Bwd_Packet_Length_Max': np.random.normal(100, 30, ddos_samples),
            'Bwd_Packet_Length_Min': np.random.normal(20, 5, ddos_samples),
            'Bwd_Packet_Length_Mean': np.random.normal(30, 10, ddos_samples),
            'Flow_Bytes_s': np.random.normal(10000, 3000, ddos_samples),   # 높은 바이트율
            'Flow_Packets_s': np.random.normal(100, 30, ddos_samples),     # 높은 패킷율
            'Flow_IAT_Mean': np.random.exponential(1000, ddos_samples),    # 짧은 간격
            'Flow_IAT_Std': np.random.exponential(500, ddos_samples),
            'Fwd_IAT_Total': np.random.exponential(10000, ddos_samples),
            'Fwd_IAT_Mean': np.random.exponential(100, ddos_samples),
            'Bwd_IAT_Total': np.random.exponential(50000, ddos_samples),
            'Bwd_IAT_Mean': np.random.exponential(5000, ddos_samples),
            'Label': ['DDoS'] * ddos_samples
        }
        
        # 웹 공격 패턴 (5%)
        web_attack_samples = 500
        web_attack_data = {
            'Flow_Duration': np.random.exponential(200000, web_attack_samples),
            'Total_Fwd_Packets': np.random.poisson(20, web_attack_samples),
            'Total_Backward_Packets': np.random.poisson(15, web_attack_samples),
            'Total_Length_Fwd_Packets': np.random.normal(2000, 500, web_attack_samples),
            'Total_Length_Bwd_Packets': np.random.normal(800, 200, web_attack_samples),
            'Fwd_Packet_Length_Max': np.random.normal(1200, 300, web_attack_samples),
            'Fwd_Packet_Length_Min': np.random.normal(100, 30, web_attack_samples),
            'Fwd_Packet_Length_Mean': np.random.normal(300, 100, web_attack_samples),
            'Bwd_Packet_Length_Max': np.random.normal(600, 150, web_attack_samples),
            'Bwd_Packet_Length_Min': np.random.normal(50, 15, web_attack_samples),
            'Bwd_Packet_Length_Mean': np.random.normal(150, 50, web_attack_samples),
            'Flow_Bytes_s': np.random.normal(2000, 800, web_attack_samples),
            'Flow_Packets_s': np.random.normal(15, 8, web_attack_samples),
            'Flow_IAT_Mean': np.random.exponential(80000, web_attack_samples),
            'Flow_IAT_Std': np.random.exponential(40000, web_attack_samples),
            'Fwd_IAT_Total': np.random.exponential(300000, web_attack_samples),
            'Fwd_IAT_Mean': np.random.exponential(15000, web_attack_samples),
            'Bwd_IAT_Total': np.random.exponential(200000, web_attack_samples),
            'Bwd_IAT_Mean': np.random.exponential(12000, web_attack_samples),
            'Label': ['Web Attack'] * web_attack_samples
        }
        
        # 브루트포스 공격 패턴 (3%)
        brute_force_samples = 300
        brute_force_data = {
            'Flow_Duration': np.random.exponential(50000, brute_force_samples),
            'Total_Fwd_Packets': np.random.poisson(50, brute_force_samples),
            'Total_Backward_Packets': np.random.poisson(3, brute_force_samples),
            'Total_Length_Fwd_Packets': np.random.normal(1000, 200, brute_force_samples),
            'Total_Length_Bwd_Packets': np.random.normal(200, 50, brute_force_samples),
            'Fwd_Packet_Length_Max': np.random.normal(500, 100, brute_force_samples),
            'Fwd_Packet_Length_Min': np.random.normal(30, 10, brute_force_samples),
            'Fwd_Packet_Length_Mean': np.random.normal(50, 15, brute_force_samples),
            'Bwd_Packet_Length_Max': np.random.normal(200, 50, brute_force_samples),
            'Bwd_Packet_Length_Min': np.random.normal(20, 5, brute_force_samples),
            'Bwd_Packet_Length_Mean': np.random.normal(40, 10, brute_force_samples),
            'Flow_Bytes_s': np.random.normal(5000, 1000, brute_force_samples),
            'Flow_Packets_s': np.random.normal(50, 15, brute_force_samples),
            'Flow_IAT_Mean': np.random.exponential(2000, brute_force_samples),
            'Flow_IAT_Std': np.random.exponential(1000, brute_force_samples),
            'Fwd_IAT_Total': np.random.exponential(20000, brute_force_samples),
            'Fwd_IAT_Mean': np.random.exponential(500, brute_force_samples),
            'Bwd_IAT_Total': np.random.exponential(30000, brute_force_samples),
            'Bwd_IAT_Mean': np.random.exponential(3000, brute_force_samples),
            'Label': ['Brute Force'] * brute_force_samples
        }
        
        # 포트 스캔 패턴 (2%)
        port_scan_samples = 200
        port_scan_data = {
            'Flow_Duration': np.random.exponential(10000, port_scan_samples),
            'Total_Fwd_Packets': np.random.poisson(5, port_scan_samples),
            'Total_Backward_Packets': np.random.poisson(1, port_scan_samples),
            'Total_Length_Fwd_Packets': np.random.normal(200, 50, port_scan_samples),
            'Total_Length_Bwd_Packets': np.random.normal(50, 20, port_scan_samples),
            'Fwd_Packet_Length_Max': np.random.normal(100, 20, port_scan_samples),
            'Fwd_Packet_Length_Min': np.random.normal(20, 5, port_scan_samples),
            'Fwd_Packet_Length_Mean': np.random.normal(40, 10, port_scan_samples),
            'Bwd_Packet_Length_Max': np.random.normal(80, 15, port_scan_samples),
            'Bwd_Packet_Length_Min': np.random.normal(15, 3, port_scan_samples),
            'Bwd_Packet_Length_Mean': np.random.normal(25, 5, port_scan_samples),
            'Flow_Bytes_s': np.random.normal(500, 100, port_scan_samples),
            'Flow_Packets_s': np.random.normal(20, 5, port_scan_samples),
            'Flow_IAT_Mean': np.random.exponential(5000, port_scan_samples),
            'Flow_IAT_Std': np.random.exponential(2000, port_scan_samples),
            'Fwd_IAT_Total': np.random.exponential(5000, port_scan_samples),
            'Fwd_IAT_Mean': np.random.exponential(1000, port_scan_samples),
            'Bwd_IAT_Total': np.random.exponential(10000, port_scan_samples),
            'Bwd_IAT_Mean': np.random.exponential(8000, port_scan_samples),
            'Label': ['PortScan'] * port_scan_samples
        }
        
        # 모든 데이터 결합
        all_data = {}
        for key in normal_data.keys():
            all_data[key] = (
                list(normal_data[key]) + 
                list(ddos_data[key]) + 
                list(web_attack_data[key]) + 
                list(brute_force_data[key]) + 
                list(port_scan_data[key])
            )
        
        df = pd.DataFrame(all_data)
        
        # 데이터 섞기
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # 음수 값 제거
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].abs()
        
        # 무한대 값 처리
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        # 저장
        output_path = self.data_dir / "cicids2017_sample.csv"
        df.to_csv(output_path, index=False)
        
        print(f"샘플 데이터 생성 완료: {output_path}")
        print(f"총 샘플 수: {len(df)}")
        print(f"라벨 분포:\n{df['Label'].value_counts()}")
        
        return df
    
    def load_real_cicids_data(self, file_path: str = None) -> pd.DataFrame:
        """실제 CICIDS2017 데이터 로드"""
        if file_path is None:
            # 일반적인 CICIDS2017 파일명들
            possible_files = [
                self.data_dir / "Monday-WorkingHours.pcap_ISCX.csv",
                self.data_dir / "Tuesday-WorkingHours.pcap_ISCX.csv",
                self.data_dir / "Wednesday-workingHours.pcap_ISCX.csv",
                self.data_dir / "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
                self.data_dir / "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
                self.data_dir / "Friday-WorkingHours-Morning.pcap_ISCX.csv",
                self.data_dir / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
                self.data_dir / "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
            ]
            
            for file_path in possible_files:
                if file_path.exists():
                    break
            else:
                print("실제 CICIDS2017 파일을 찾을 수 없습니다.")
                print("다음 중 하나를 다운로드하세요:")
                print("1. https://www.unb.ca/cic/datasets/ids-2017.html")
                print("2. Kaggle: https://www.kaggle.com/cicdataset/cicids2017")
                return None
        
        try:
            print(f"CICIDS2017 데이터 로드 중: {file_path}")
            df = pd.read_csv(file_path)
            
            # 컬럼명 정리
            df.columns = df.columns.str.strip()
            
            # 무한대 값 처리
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(0)
            
            print(f"데이터 로드 완료: {len(df)} 샘플")
            print(f"라벨 분포:\n{df['Label'].value_counts()}")
            
            return df
            
        except Exception as e:
            print(f"데이터 로드 오류: {e}")
            return None
    
    def preprocess_for_api_logs(self, df: pd.DataFrame) -> pd.DataFrame:
        """API 로그 분석에 맞게 전처리"""
        
        # API 로그 관련 특성만 선택
        api_relevant_features = [
            'Flow_Duration',
            'Total_Fwd_Packets', 
            'Total_Backward_Packets',
            'Flow_Bytes_s',
            'Flow_Packets_s', 
            'Flow_IAT_Mean',
            'Fwd_IAT_Mean',
            'Bwd_IAT_Mean',
            'Fwd_Packet_Length_Mean',
            'Bwd_Packet_Length_Mean'
        ]
        
        # 사용 가능한 컬럼만 선택
        available_features = [col for col in api_relevant_features if col in df.columns]
        
        if len(available_features) == 0:
            print("사용 가능한 특성이 없습니다.")
            return None
        
        # 특성 데이터
        X = df[available_features].copy()
        
        # 라벨 이진화 (BENIGN = 0, 나머지 = 1)
        y = (df['Label'] != 'BENIGN').astype(int)
        
        # 이상치 제거 (IQR 방법)
        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)
        IQR = Q3 - Q1
        
        # 극단적인 이상치만 제거
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        for col in X.columns:
            X.loc[X[col] < lower_bound[col], col] = lower_bound[col]
            X.loc[X[col] > upper_bound[col], col] = upper_bound[col]
        
        # 결과 DataFrame 생성
        result_df = X.copy()
        result_df['Label'] = y
        result_df['Original_Label'] = df['Label']
        
        print(f"전처리 완료:")
        print(f"- 사용된 특성: {available_features}")
        print(f"- 정상 샘플: {(y == 0).sum()}")
        print(f"- 공격 샘플: {(y == 1).sum()}")
        
        return result_df

# 실제 사용 예시
def demo_data_loading():
    """데이터 로딩 데모"""
    
    # 데이터 로더 초기화
    loader = CICIDSDataLoader()
    
    # 방법 1: 샘플 데이터 생성 (즉시 사용 가능)
    print("=== 방법 1: 샘플 데이터 생성 ===")
    sample_df = loader.download_sample_data()
    processed_sample = loader.preprocess_for_api_logs(sample_df)
    
    # 방법 2: 실제 CICIDS2017 데이터 사용 (다운로드 후)
    print("\n=== 방법 2: 실제 CICIDS2017 데이터 ===")
    real_df = loader.load_real_cicids_data()
    if real_df is not None:
        processed_real = loader.preprocess_for_api_logs(real_df)
    
    return processed_sample

# 통합 실행 스크립트
def setup_complete_system():
    """완전한 시스템 설정"""
    from core.anomaly_detection import APILogAnomalyDetector
    
    print("🚀 API 로그 이상 탐지 시스템 설정 시작")
    
    # 1. 데이터 준비
    print("\n1️⃣ 데이터 준비...")
    loader = CICIDSDataLoader()
    df = loader.download_sample_data()
    processed_df = loader.preprocess_for_api_logs(df)
    
    # 2. 모델 훈련
    print("\n2️⃣ 모델 훈련...")
    detector = APILogAnomalyDetector(model_type='hybrid')
    
    # 특성과 라벨 분리
    feature_columns = [col for col in processed_df.columns if col not in ['Label', 'Original_Label']]
    X = processed_df[feature_columns].values
    y = processed_df['Label'].values
    
    # 훈련 실행
    history = detector.train(X, y, epochs=20)
    
    # 3. 모델 저장
    print("\n3️⃣ 모델 저장...")
    detector.save_model()
    
    # 4. 테스트
    print("\n4️⃣ 시스템 테스트...")
    
    # 테스트 데이터 생성
    test_normal = {
        "timestamp": "2025-07-12T15:30:00",
        "client_ip": "192.168.1.100",
        "method": "POST",
        "url": "/api/v1/customer/segment",
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "requests_per_minute": 3,
        "request_size": 256,
        "content_length": 128,
        "processing_time": 0.12
    }
    
    test_suspicious = {
        "timestamp": "2025-07-12T15:30:00",
        "client_ip": "10.0.0.1",
        "method": "POST",
        "url": "/api/v1/customer/segment?id=1' UNION SELECT * FROM users--",
        "user_agent": "sqlmap/1.4.12#stable",
        "requests_per_minute": 150,
        "request_size": 5000,
        "content_length": 2048,
        "processing_time": 5.0
    }
    
    # 예측 테스트
    prob_normal, is_anomaly_normal = detector.predict(test_normal)
    prob_suspicious, is_anomaly_suspicious = detector.predict(test_suspicious)
    
    print(f"정상 요청 - 이상 확률: {prob_normal:.3f}, 이상 여부: {is_anomaly_normal}")
    print(f"의심 요청 - 이상 확률: {prob_suspicious:.3f}, 이상 여부: {is_anomaly_suspicious}")
    
    print("\n✅ 시스템 설정 완료!")
    print("\n다음 단계:")
    print("1. FastAPI 서버 실행: uvicorn api.customer_api:app --reload")
    print("2. 트래픽 시뮬레이션 실행")
    print("3. 실시간 모니터링 대시보드 확인")
    
    return detector

if __name__ == "__main__":
    # 시스템 전체 설정
    detector = setup_complete_system()
