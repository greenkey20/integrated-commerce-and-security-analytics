"""
CICIDS2017 보안 데이터 로딩 및 전처리 모듈

실제 CICIDS2017 데이터셋 로딩, 전처리, 샘플 데이터 생성을 담당
화면 코드와 분리된 순수 로직 모듈
"""

import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class CICIDSDataLoader:
    """CICIDS2017 데이터 로딩 및 전처리 클래스"""
    
    def __init__(self, data_dir="/Users/greenpianorabbit/Documents/Development/customer-segmentation/data/cicids2017"):
        self.data_dir = data_dir
        self.supported_encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        
    def check_data_availability(self):
        """CICIDS2017 데이터 파일 존재 확인 (Monday 파일 제외)"""
        patterns = ["*.csv", "*ISCX.csv", "*cicids*.csv", "*CIC*.csv"]
        files = []
        
        for pattern in patterns:
            files.extend(glob.glob(os.path.join(self.data_dir, pattern)))
        
        # Monday 파일 제외 (공격 데이터 거의 0%)
        filtered_files = []
        for file_path in files:
            filename = os.path.basename(file_path)
            if not filename.startswith('Monday'):
                filtered_files.append(file_path)
        
        return {
            "available": len(filtered_files) > 0,
            "files": filtered_files,
            "count": len(filtered_files)
        }
    
    def find_label_column(self, df):
        """데이터프레임에서 라벨 컬럼 찾기"""
        possible_names = [
            'Label', ' Label', 'Label ', ' Label ',
            'label', ' label', 'LABEL', ' LABEL',
            'class', 'Class', ' Class', 'target', 'Target'
        ]
        
        # 정확히 일치하는 컬럼 찾기
        for col_name in possible_names:
            if col_name in df.columns:
                return col_name
        
        # 부분 일치하는 컬럼 찾기
        for col in df.columns:
            if 'label' in col.lower() or 'class' in col.lower():
                return col
        
        return None
    
    def standardize_labels(self, labels):
        """라벨 표준화"""
        standardized = labels.str.strip().str.upper()
        
        label_mapping = {
            'BENIGN': 'BENIGN', 'NORMAL': 'BENIGN', 
            'DDOS': 'DDoS', 'DOS': 'DoS',
            'WEB ATTACK': 'Web Attack',
            'WEB ATTACK – BRUTE FORCE': 'Web Attack - Brute Force',
            'WEB ATTACK – XSS': 'Web Attack - XSS',
            'WEB ATTACK – SQL INJECTION': 'Web Attack - SQL Injection',
            'BRUTE FORCE': 'Brute Force',
            'SSH-PATATOR': 'Brute Force',
            'FTP-PATATOR': 'Brute Force',
            'PORTSCAN': 'PortScan',
            'INFILTRATION': 'Infiltration',
            'BOT': 'Botnet',
            'HEARTBLEED': 'Heartbleed'
        }
        
        for old_label, new_label in label_mapping.items():
            standardized = standardized.replace(old_label, new_label)
        
        return standardized
    
    def load_file_with_encoding(self, file_path, max_rows=10000):
        """여러 인코딩을 시도하여 파일 로드"""
        for encoding in self.supported_encodings:
            try:
                # 헤더 먼저 확인
                df_header = pd.read_csv(file_path, nrows=0, encoding=encoding)
                columns = df_header.columns.tolist()
                
                # 컬럼명 정리
                cleaned_columns = [col.strip() for col in columns]
                
                # 실제 데이터 로드
                df = pd.read_csv(file_path, nrows=max_rows, encoding=encoding)
                df.columns = cleaned_columns
                
                # 라벨 컬럼 찾기 및 표준화
                label_column = self.find_label_column(df)
                if label_column and label_column != 'Label':
                    df = df.rename(columns={label_column: 'Label'})
                
                if 'Label' in df.columns:
                    df['Label'] = self.standardize_labels(df['Label'])
                
                return df, encoding
                
            except Exception as e:
                continue
        
        raise ValueError(f"모든 인코딩 방법으로 {file_path} 파일을 읽을 수 없습니다")
    
    def load_attack_files(self, max_rows_per_file=3000):
        """공격 데이터가 포함된 파일들 로드"""
        attack_files = [
            "Tuesday-WorkingHours.pcap_ISCX.csv",
            "Wednesday-workingHours.pcap_ISCX.csv", 
            "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
            "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
        ]
        
        combined_data = []
        successful_files = []
        
        for filename in attack_files:
            file_path = os.path.join(self.data_dir, filename)
            
            if os.path.exists(file_path):
                try:
                    df, encoding = self.load_file_with_encoding(file_path, max_rows_per_file)
                    
                    if 'Label' in df.columns:
                        attack_count = (df['Label'] != 'BENIGN').sum()
                        attack_ratio = attack_count / len(df) * 100
                        
                        combined_data.append(df)
                        successful_files.append({
                            'filename': filename,
                            'records': len(df),
                            'attacks': attack_count,
                            'attack_ratio': attack_ratio,
                            'encoding': encoding
                        })
                        
                except Exception as e:
                    continue
        
        if combined_data:
            final_data = pd.concat(combined_data, ignore_index=True)
            final_data = self.clean_numeric_data(final_data)
            return final_data, successful_files
        else:
            return None, []
    
    def clean_numeric_data(self, df):
        """수치형 데이터 정리 (무한대값, NaN 처리)"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        df[numeric_cols] = df[numeric_cols].abs()  # 음수값 처리
        return df
    
    def generate_sample_data(self, n_samples=10000, attack_ratio=0.3):
        """CICIDS2017 스타일 샘플 데이터 생성"""
        np.random.seed(42)
        
        normal_samples = int(n_samples * (1 - attack_ratio))
        attack_samples = n_samples - normal_samples
        
        # 네트워크 특성 생성
        features = self._generate_network_features(normal_samples, attack_samples)
        
        # 데이터프레임 생성
        df = pd.DataFrame(features)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df = self.clean_numeric_data(df)
        
        return df
    
    def _generate_network_features(self, normal_samples, attack_samples):
        """네트워크 특성 시뮬레이션"""
        # 기본 네트워크 특성들
        features = {
            # 플로우 기본 정보
            'Flow_Duration': (
                list(np.random.exponential(100000, normal_samples)) +
                list(np.random.exponential(10000, attack_samples))
            ),
            'Total_Fwd_Packets': (
                list(np.random.poisson(15, normal_samples)) +
                list(np.random.poisson(200, attack_samples))
            ),
            'Total_Backward_Packets': (
                list(np.random.poisson(12, normal_samples)) +
                list(np.random.poisson(5, attack_samples))
            ),
            
            # 패킷 길이 특성
            'Total_Length_of_Fwd_Packets': (
                list(np.random.normal(800, 300, normal_samples)) +
                list(np.random.normal(10000, 2000, attack_samples))
            ),
            'Total_Length_of_Bwd_Packets': (
                list(np.random.normal(600, 200, normal_samples)) +
                list(np.random.normal(200, 100, attack_samples))
            ),
            
            # 패킷 길이 통계
            'Fwd_Packet_Length_Max': (
                list(np.random.normal(1200, 400, normal_samples)) +
                list(np.random.normal(1500, 100, attack_samples))
            ),
            'Fwd_Packet_Length_Min': (
                list(np.random.normal(60, 20, normal_samples)) +
                list(np.random.normal(64, 10, attack_samples))
            ),
            'Fwd_Packet_Length_Mean': (
                list(np.random.normal(400, 150, normal_samples)) +
                list(np.random.normal(80, 20, attack_samples))
            ),
            
            'Bwd_Packet_Length_Max': (
                list(np.random.normal(1000, 300, normal_samples)) +
                list(np.random.normal(150, 50, attack_samples))
            ),
            'Bwd_Packet_Length_Min': (
                list(np.random.normal(50, 15, normal_samples)) +
                list(np.random.normal(40, 10, attack_samples))
            ),
            'Bwd_Packet_Length_Mean': (
                list(np.random.normal(300, 100, normal_samples)) +
                list(np.random.normal(60, 20, attack_samples))
            ),
            
            # 플로우 속도 특성 (핵심 구별 지표)
            'Flow_Bytes/s': (
                list(np.random.normal(2000, 1000, normal_samples)) +
                list(np.random.normal(50000, 15000, attack_samples))
            ),
            'Flow_Packets/s': (
                list(np.random.normal(20, 10, normal_samples)) +
                list(np.random.normal(500, 150, attack_samples))
            ),
            
            # IAT (Inter-Arrival Time) 특성
            'Flow_IAT_Mean': (
                list(np.random.exponential(50000, normal_samples)) +
                list(np.random.exponential(1000, attack_samples))
            ),
            'Flow_IAT_Std': (
                list(np.random.exponential(25000, normal_samples)) +
                list(np.random.exponential(500, attack_samples))
            ),
            
            'Fwd_IAT_Total': (
                list(np.random.exponential(200000, normal_samples)) +
                list(np.random.exponential(5000, attack_samples))
            ),
            'Fwd_IAT_Mean': (
                list(np.random.exponential(20000, normal_samples)) +
                list(np.random.exponential(50, attack_samples))
            ),
            
            'Bwd_IAT_Total': (
                list(np.random.exponential(150000, normal_samples)) +
                list(np.random.exponential(20000, attack_samples))
            ),
            'Bwd_IAT_Mean': (
                list(np.random.exponential(15000, normal_samples)) +
                list(np.random.exponential(2000, attack_samples))
            ),
            
            # 라벨
            'Label': (['BENIGN'] * normal_samples + ['DDoS'] * attack_samples)
        }
        
        return features
    
    def generate_enhanced_sample_data(self, n_samples=10000):
        """다양한 공격 유형을 포함한 향상된 샘플 데이터"""
        np.random.seed(42)
        
        # 공격 비율 60%로 증가
        normal_samples = int(n_samples * 0.4)
        ddos_samples = int(n_samples * 0.25)
        web_attack_samples = int(n_samples * 0.15)
        brute_force_samples = int(n_samples * 0.10)
        port_scan_samples = n_samples - normal_samples - ddos_samples - web_attack_samples - brute_force_samples
        
        # 각 공격 유형별 특성 생성
        all_data = {}
        feature_names = [
            'Flow_Duration', 'Total_Fwd_Packets', 'Total_Backward_Packets',
            'Total_Length_of_Fwd_Packets', 'Total_Length_of_Bwd_Packets',
            'Fwd_Packet_Length_Max', 'Fwd_Packet_Length_Min', 'Fwd_Packet_Length_Mean',
            'Bwd_Packet_Length_Max', 'Bwd_Packet_Length_Min', 'Bwd_Packet_Length_Mean',
            'Flow_Bytes/s', 'Flow_Packets/s', 'Flow_IAT_Mean', 'Flow_IAT_Std',
            'Fwd_IAT_Total', 'Fwd_IAT_Mean', 'Bwd_IAT_Total', 'Bwd_IAT_Mean'
        ]
        
        # 정상 트래픽
        normal_data = self._generate_normal_traffic(normal_samples)
        
        # 각 공격 유형별 데이터
        ddos_data = self._generate_ddos_traffic(ddos_samples)
        web_data = self._generate_web_attack_traffic(web_attack_samples)
        brute_data = self._generate_brute_force_traffic(brute_force_samples)
        port_data = self._generate_port_scan_traffic(port_scan_samples)
        
        # 모든 데이터 결합
        for feature in feature_names:
            all_data[feature] = (
                list(normal_data[feature]) + list(ddos_data[feature]) +
                list(web_data[feature]) + list(brute_data[feature]) +
                list(port_data[feature])
            )
        
        all_data['Label'] = (
            ['BENIGN'] * normal_samples + ['DDoS'] * ddos_samples +
            ['Web Attack'] * web_attack_samples + ['Brute Force'] * brute_force_samples +
            ['PortScan'] * port_scan_samples
        )
        
        df = pd.DataFrame(all_data)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df = self.clean_numeric_data(df)
        
        return df
    
    def _generate_normal_traffic(self, n_samples):
        """정상 트래픽 패턴 생성"""
        return {
            'Flow_Duration': np.random.exponential(100000, n_samples),
            'Total_Fwd_Packets': np.random.poisson(15, n_samples),
            'Total_Backward_Packets': np.random.poisson(12, n_samples),
            'Total_Length_of_Fwd_Packets': np.random.normal(800, 300, n_samples),
            'Total_Length_of_Bwd_Packets': np.random.normal(600, 200, n_samples),
            'Fwd_Packet_Length_Max': np.random.normal(1200, 400, n_samples),
            'Fwd_Packet_Length_Min': np.random.normal(60, 20, n_samples),
            'Fwd_Packet_Length_Mean': np.random.normal(400, 150, n_samples),
            'Bwd_Packet_Length_Max': np.random.normal(1000, 300, n_samples),
            'Bwd_Packet_Length_Min': np.random.normal(50, 15, n_samples),
            'Bwd_Packet_Length_Mean': np.random.normal(300, 100, n_samples),
            'Flow_Bytes/s': np.random.normal(2000, 1000, n_samples),
            'Flow_Packets/s': np.random.normal(20, 10, n_samples),
            'Flow_IAT_Mean': np.random.exponential(50000, n_samples),
            'Flow_IAT_Std': np.random.exponential(25000, n_samples),
            'Fwd_IAT_Total': np.random.exponential(200000, n_samples),
            'Fwd_IAT_Mean': np.random.exponential(20000, n_samples),
            'Bwd_IAT_Total': np.random.exponential(150000, n_samples),
            'Bwd_IAT_Mean': np.random.exponential(15000, n_samples)
        }
    
    def _generate_ddos_traffic(self, n_samples):
        """DDoS 공격 패턴 생성"""
        return {
            'Flow_Duration': np.random.exponential(10000, n_samples),
            'Total_Fwd_Packets': np.random.poisson(200, n_samples),
            'Total_Backward_Packets': np.random.poisson(5, n_samples),
            'Total_Length_of_Fwd_Packets': np.random.normal(10000, 2000, n_samples),
            'Total_Length_of_Bwd_Packets': np.random.normal(200, 100, n_samples),
            'Fwd_Packet_Length_Max': np.random.normal(1500, 100, n_samples),
            'Fwd_Packet_Length_Min': np.random.normal(64, 10, n_samples),
            'Fwd_Packet_Length_Mean': np.random.normal(80, 20, n_samples),
            'Bwd_Packet_Length_Max': np.random.normal(150, 50, n_samples),
            'Bwd_Packet_Length_Min': np.random.normal(40, 10, n_samples),
            'Bwd_Packet_Length_Mean': np.random.normal(60, 20, n_samples),
            'Flow_Bytes/s': np.random.normal(50000, 15000, n_samples),
            'Flow_Packets/s': np.random.normal(500, 150, n_samples),
            'Flow_IAT_Mean': np.random.exponential(1000, n_samples),
            'Flow_IAT_Std': np.random.exponential(500, n_samples),
            'Fwd_IAT_Total': np.random.exponential(5000, n_samples),
            'Fwd_IAT_Mean': np.random.exponential(50, n_samples),
            'Bwd_IAT_Total': np.random.exponential(20000, n_samples),
            'Bwd_IAT_Mean': np.random.exponential(2000, n_samples)
        }
    
    def _generate_web_attack_traffic(self, n_samples):
        """웹 공격 패턴 생성"""
        return {
            'Flow_Duration': np.random.exponential(150000, n_samples),
            'Total_Fwd_Packets': np.random.poisson(30, n_samples),
            'Total_Backward_Packets': np.random.poisson(25, n_samples),
            'Total_Length_of_Fwd_Packets': np.random.normal(3000, 800, n_samples),
            'Total_Length_of_Bwd_Packets': np.random.normal(1500, 400, n_samples),
            'Fwd_Packet_Length_Max': np.random.normal(1400, 200, n_samples),
            'Fwd_Packet_Length_Min': np.random.normal(200, 50, n_samples),
            'Fwd_Packet_Length_Mean': np.random.normal(500, 100, n_samples),
            'Bwd_Packet_Length_Max': np.random.normal(800, 150, n_samples),
            'Bwd_Packet_Length_Min': np.random.normal(100, 30, n_samples),
            'Bwd_Packet_Length_Mean': np.random.normal(250, 80, n_samples),
            'Flow_Bytes/s': np.random.normal(4000, 1500, n_samples),
            'Flow_Packets/s': np.random.normal(25, 10, n_samples),
            'Flow_IAT_Mean': np.random.exponential(30000, n_samples),
            'Flow_IAT_Std': np.random.exponential(15000, n_samples),
            'Fwd_IAT_Total': np.random.exponential(100000, n_samples),
            'Fwd_IAT_Mean': np.random.exponential(8000, n_samples),
            'Bwd_IAT_Total': np.random.exponential(80000, n_samples),
            'Bwd_IAT_Mean': np.random.exponential(6000, n_samples)
        }
    
    def _generate_brute_force_traffic(self, n_samples):
        """브루트포스 공격 패턴 생성"""
        return {
            'Flow_Duration': np.random.exponential(30000, n_samples),
            'Total_Fwd_Packets': np.random.poisson(80, n_samples),
            'Total_Backward_Packets': np.random.poisson(8, n_samples),
            'Total_Length_of_Fwd_Packets': np.random.normal(2000, 500, n_samples),
            'Total_Length_of_Bwd_Packets': np.random.normal(400, 150, n_samples),
            'Fwd_Packet_Length_Max': np.random.normal(800, 200, n_samples),
            'Fwd_Packet_Length_Min': np.random.normal(40, 15, n_samples),
            'Fwd_Packet_Length_Mean': np.random.normal(80, 30, n_samples),
            'Bwd_Packet_Length_Max': np.random.normal(300, 100, n_samples),
            'Bwd_Packet_Length_Min': np.random.normal(30, 10, n_samples),
            'Bwd_Packet_Length_Mean': np.random.normal(60, 20, n_samples),
            'Flow_Bytes/s': np.random.normal(8000, 2000, n_samples),
            'Flow_Packets/s': np.random.normal(80, 20, n_samples),
            'Flow_IAT_Mean': np.random.exponential(3000, n_samples),
            'Flow_IAT_Std': np.random.exponential(1500, n_samples),
            'Fwd_IAT_Total': np.random.exponential(15000, n_samples),
            'Fwd_IAT_Mean': np.random.exponential(300, n_samples),
            'Bwd_IAT_Total': np.random.exponential(25000, n_samples),
            'Bwd_IAT_Mean': np.random.exponential(2500, n_samples)
        }
    
    def _generate_port_scan_traffic(self, n_samples):
        """포트스캔 공격 패턴 생성"""
        return {
            'Flow_Duration': np.random.exponential(5000, n_samples),
            'Total_Fwd_Packets': np.random.poisson(10, n_samples),
            'Total_Backward_Packets': np.random.poisson(2, n_samples),
            'Total_Length_of_Fwd_Packets': np.random.normal(400, 150, n_samples),
            'Total_Length_of_Bwd_Packets': np.random.normal(100, 50, n_samples),
            'Fwd_Packet_Length_Max': np.random.normal(200, 60, n_samples),
            'Fwd_Packet_Length_Min': np.random.normal(40, 10, n_samples),
            'Fwd_Packet_Length_Mean': np.random.normal(60, 20, n_samples),
            'Bwd_Packet_Length_Max': np.random.normal(100, 30, n_samples),
            'Bwd_Packet_Length_Min': np.random.normal(20, 5, n_samples),
            'Bwd_Packet_Length_Mean': np.random.normal(40, 15, n_samples),
            'Flow_Bytes/s': np.random.normal(1000, 300, n_samples),
            'Flow_Packets/s': np.random.normal(30, 10, n_samples),
            'Flow_IAT_Mean': np.random.exponential(8000, n_samples),
            'Flow_IAT_Std': np.random.exponential(4000, n_samples),
            'Fwd_IAT_Total': np.random.exponential(3000, n_samples),
            'Fwd_IAT_Mean': np.random.exponential(800, n_samples),
            'Bwd_IAT_Total': np.random.exponential(8000, n_samples),
            'Bwd_IAT_Mean': np.random.exponential(4000, n_samples)
        }
    
    def get_data_quality_report(self, df):
        """데이터 품질 보고서 생성"""
        report = {
            'total_records': len(df),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns)
        }
        
        if 'Label' in df.columns:
            label_counts = df['Label'].value_counts()
            report['labels'] = {
                'unique_count': len(label_counts),
                'distribution': label_counts.to_dict(),
                'attack_ratio': (df['Label'] != 'BENIGN').sum() / len(df) * 100
            }
        
        return report


# 편의 함수들 (하위 호환성)
def check_cicids_data_availability():
    """하위 호환성을 위한 편의 함수"""
    loader = CICIDSDataLoader()
    return loader.check_data_availability()

def generate_cicids_sample_data():
    """하위 호환성을 위한 편의 함수"""
    loader = CICIDSDataLoader()
    return loader.generate_sample_data()

def generate_enhanced_sample_data():
    """하위 호환성을 위한 편의 함수"""
    loader = CICIDSDataLoader()
    return loader.generate_enhanced_sample_data()
