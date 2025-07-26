"""
개선된 현실적인 보안 데이터 생성기

기존 unified_security_loader.py의 문제점을 해결한 버전
- 패턴 간 겹침 증가
- 노이즈 추가  
- 더 현실적인 분포
"""

import numpy as np
import pandas as pd
from typing import Dict, List

class RealisticSecurityDataGenerator:
    """현실적인 보안 데이터 생성기"""
    
    def __init__(self):
        self.noise_level = 0.3  # 노이즈 레벨 증가
        
    def generate_realistic_sample_data(self, total_samples: int = 10000, attack_ratio: float = 0.3) -> pd.DataFrame:
        """현실적인 샘플 데이터 생성"""
        
        np.random.seed(42)
        
        normal_samples = int(total_samples * (1 - attack_ratio))
        attack_samples = total_samples - normal_samples
        
        # 공격 유형별 분배 (더 균등하게)
        ddos_samples = int(attack_samples * 0.3)
        web_attack_samples = int(attack_samples * 0.3) 
        brute_force_samples = int(attack_samples * 0.25)
        port_scan_samples = attack_samples - ddos_samples - web_attack_samples - brute_force_samples
        
        # 현실적인 패턴으로 데이터 생성
        normal_data = self._generate_realistic_normal_traffic(normal_samples)
        ddos_data = self._generate_realistic_ddos_traffic(ddos_samples)
        web_attack_data = self._generate_realistic_web_attack_traffic(web_attack_samples)
        brute_force_data = self._generate_realistic_brute_force_traffic(brute_force_samples)
        port_scan_data = self._generate_realistic_port_scan_traffic(port_scan_samples)
        
        # 데이터 결합
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
        
        # 섞기
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"현실적인 데이터 생성 완료:")
        print(f"라벨 분포:\n{df['Label'].value_counts()}")
        
        return df
    
    def _add_realistic_noise(self, values: np.ndarray, base_noise: float = None) -> np.ndarray:
        """현실적인 노이즈 추가"""
        if base_noise is None:
            base_noise = self.noise_level
            
        # 가우시안 노이즈 + 이상치
        gaussian_noise = np.random.normal(0, base_noise * np.std(values), len(values))
        
        # 5% 확률로 이상치 추가 (실제 네트워크에서 발생)
        outlier_mask = np.random.random(len(values)) < 0.05
        outlier_noise = np.random.normal(0, 2 * np.std(values), len(values))
        
        noisy_values = values + gaussian_noise
        noisy_values[outlier_mask] += outlier_noise[outlier_mask]
        
        return np.abs(noisy_values)  # 음수 제거
    
    def _generate_realistic_normal_traffic(self, samples: int) -> Dict[str, List]:
        """현실적인 정상 트래픽 생성"""
        base_data = {
            'Flow_Duration': np.random.exponential(800000, samples),
            'Total_Fwd_Packets': np.random.poisson(12, samples),
            'Total_Backward_Packets': np.random.poisson(10, samples),
            'Total_Length_of_Fwd_Packets': np.random.normal(750, 400, samples),
            'Total_Length_of_Bwd_Packets': np.random.normal(550, 300, samples),
            'Fwd_Packet_Length_Max': np.random.normal(1100, 500, samples),
            'Fwd_Packet_Length_Min': np.random.normal(55, 25, samples),
            'Fwd_Packet_Length_Mean': np.random.normal(380, 200, samples),
            'Bwd_Packet_Length_Max': np.random.normal(900, 400, samples),
            'Bwd_Packet_Length_Min': np.random.normal(45, 20, samples),
            'Bwd_Packet_Length_Mean': np.random.normal(280, 150, samples),
            'Flow_Bytes_s': np.random.normal(1800, 1200, samples),
            'Flow_Packets_s': np.random.normal(18, 12, samples),
            'Flow_IAT_Mean': np.random.exponential(45000, samples),
            'Flow_IAT_Std': np.random.exponential(22000, samples),
            'Fwd_IAT_Total': np.random.exponential(180000, samples),
            'Fwd_IAT_Mean': np.random.exponential(18000, samples),
            'Bwd_IAT_Total': np.random.exponential(140000, samples),
            'Bwd_IAT_Mean': np.random.exponential(14000, samples),
        }
        
        # 모든 특성에 노이즈 추가
        for key, values in base_data.items():
            base_data[key] = self._add_realistic_noise(values)
        
        base_data['Label'] = ['BENIGN'] * samples
        return base_data
    
    def _generate_realistic_ddos_traffic(self, samples: int) -> Dict[str, List]:
        """현실적인 DDoS 공격 생성 (겹침 영역 포함)"""
        
        # 기본 DDoS 패턴
        base_data = {
            'Flow_Duration': np.random.exponential(15000, samples),
            'Total_Fwd_Packets': np.random.poisson(120, samples),  # 200→120 (겹침 증가)
            'Total_Backward_Packets': np.random.poisson(4, samples),
            'Total_Length_of_Fwd_Packets': np.random.normal(8000, 3000, samples),
            'Total_Length_of_Bwd_Packets': np.random.normal(180, 120, samples),
            'Fwd_Packet_Length_Max': np.random.normal(1400, 200, samples),
            'Fwd_Packet_Length_Min': np.random.normal(60, 15, samples),
            'Fwd_Packet_Length_Mean': np.random.normal(75, 25, samples),
            'Bwd_Packet_Length_Max': np.random.normal(140, 60, samples),
            'Bwd_Packet_Length_Min': np.random.normal(35, 12, samples),
            'Bwd_Packet_Length_Mean': np.random.normal(55, 25, samples),
            'Flow_Bytes_s': np.random.normal(35000, 20000, samples),  # 50000→35000
            'Flow_Packets_s': np.random.normal(300, 200, samples),   # 500→300
            'Flow_IAT_Mean': np.random.exponential(1500, samples),
            'Flow_IAT_Std': np.random.exponential(750, samples),
            'Fwd_IAT_Total': np.random.exponential(8000, samples),
            'Fwd_IAT_Mean': np.random.exponential(80, samples),
            'Bwd_IAT_Total': np.random.exponential(25000, samples),
            'Bwd_IAT_Mean': np.random.exponential(2500, samples),
        }
        
        # 높은 노이즈로 변동성 증가
        for key, values in base_data.items():
            base_data[key] = self._add_realistic_noise(values, 0.5)
        
        # 일부 샘플을 정상과 유사하게 만들기 (약한 공격 시뮬레이션)
        weak_attack_ratio = 0.15  # 15%는 약한 공격
        weak_indices = np.random.choice(samples, int(samples * weak_attack_ratio), replace=False)
        
        for idx in weak_indices:
            # 정상 트래픽에 가까운 값으로 조정
            base_data['Total_Fwd_Packets'][idx] *= 0.3  # 대폭 감소
            base_data['Flow_Packets_s'][idx] *= 0.2
            base_data['Flow_Bytes_s'][idx] *= 0.3
        
        base_data['Label'] = ['DDoS'] * samples
        return base_data
    
    def _generate_realistic_web_attack_traffic(self, samples: int) -> Dict[str, List]:
        """현실적인 웹 공격 생성"""
        base_data = {
            'Flow_Duration': np.random.exponential(120000, samples),
            'Total_Fwd_Packets': np.random.poisson(25, samples),  # 30→25
            'Total_Backward_Packets': np.random.poisson(20, samples),
            'Total_Length_of_Fwd_Packets': np.random.normal(2500, 1000, samples),
            'Total_Length_of_Bwd_Packets': np.random.normal(1200, 500, samples),
            'Fwd_Packet_Length_Max': np.random.normal(1300, 300, samples),
            'Fwd_Packet_Length_Min': np.random.normal(150, 60, samples),
            'Fwd_Packet_Length_Mean': np.random.normal(450, 150, samples),
            'Bwd_Packet_Length_Max': np.random.normal(750, 200, samples),
            'Bwd_Packet_Length_Min': np.random.normal(80, 40, samples),
            'Bwd_Packet_Length_Mean': np.random.normal(220, 100, samples),
            'Flow_Bytes_s': np.random.normal(3200, 2000, samples),  # 4000→3200
            'Flow_Packets_s': np.random.normal(22, 15, samples),    # 25→22
            'Flow_IAT_Mean': np.random.exponential(25000, samples),
            'Flow_IAT_Std': np.random.exponential(12000, samples),
            'Fwd_IAT_Total': np.random.exponential(80000, samples),
            'Fwd_IAT_Mean': np.random.exponential(6000, samples),
            'Bwd_IAT_Total': np.random.exponential(65000, samples),
            'Bwd_IAT_Mean': np.random.exponential(5000, samples),
        }
        
        # 노이즈 추가
        for key, values in base_data.items():
            base_data[key] = self._add_realistic_noise(values, 0.4)
        
        base_data['Label'] = ['Web Attack'] * samples
        return base_data
    
    def _generate_realistic_brute_force_traffic(self, samples: int) -> Dict[str, List]:
        """현실적인 브루트포스 공격 생성"""
        base_data = {
            'Flow_Duration': np.random.exponential(25000, samples),
            'Total_Fwd_Packets': np.random.poisson(60, samples),  # 80→60
            'Total_Backward_Packets': np.random.poisson(6, samples),
            'Total_Length_of_Fwd_Packets': np.random.normal(1600, 600, samples),
            'Total_Length_of_Bwd_Packets': np.random.normal(350, 200, samples),
            'Fwd_Packet_Length_Max': np.random.normal(700, 250, samples),
            'Fwd_Packet_Length_Min': np.random.normal(35, 18, samples),
            'Fwd_Packet_Length_Mean': np.random.normal(70, 35, samples),
            'Bwd_Packet_Length_Max': np.random.normal(250, 120, samples),
            'Bwd_Packet_Length_Min': np.random.normal(25, 12, samples),
            'Bwd_Packet_Length_Mean': np.random.normal(50, 25, samples),
            'Flow_Bytes_s': np.random.normal(6000, 3000, samples),  # 8000→6000
            'Flow_Packets_s': np.random.normal(60, 30, samples),    # 80→60
            'Flow_IAT_Mean': np.random.exponential(2500, samples),
            'Flow_IAT_Std': np.random.exponential(1200, samples),
            'Fwd_IAT_Total': np.random.exponential(12000, samples),
            'Fwd_IAT_Mean': np.random.exponential(250, samples),
            'Bwd_IAT_Total': np.random.exponential(20000, samples),
            'Bwd_IAT_Mean': np.random.exponential(2000, samples),
        }
        
        # 노이즈 추가
        for key, values in base_data.items():
            base_data[key] = self._add_realistic_noise(values, 0.4)
        
        base_data['Label'] = ['Brute Force'] * samples
        return base_data
    
    def _generate_realistic_port_scan_traffic(self, samples: int) -> Dict[str, List]:
        """현실적인 포트스캔 공격 생성"""
        base_data = {
            'Flow_Duration': np.random.exponential(4000, samples),
            'Total_Fwd_Packets': np.random.poisson(8, samples),  # 10→8
            'Total_Backward_Packets': np.random.poisson(1, samples),
            'Total_Length_of_Fwd_Packets': np.random.normal(320, 180, samples),
            'Total_Length_of_Bwd_Packets': np.random.normal(80, 60, samples),
            'Fwd_Packet_Length_Max': np.random.normal(160, 80, samples),
            'Fwd_Packet_Length_Min': np.random.normal(35, 12, samples),
            'Fwd_Packet_Length_Mean': np.random.normal(50, 25, samples),
            'Bwd_Packet_Length_Max': np.random.normal(80, 40, samples),
            'Bwd_Packet_Length_Min': np.random.normal(15, 8, samples),
            'Bwd_Packet_Length_Mean': np.random.normal(30, 20, samples),
            'Flow_Bytes_s': np.random.normal(800, 400, samples),   # 1000→800
            'Flow_Packets_s': np.random.normal(25, 15, samples),   # 30→25
            'Flow_IAT_Mean': np.random.exponential(6000, samples),
            'Flow_IAT_Std': np.random.exponential(3000, samples),
            'Fwd_IAT_Total': np.random.exponential(2500, samples),
            'Fwd_IAT_Mean': np.random.exponential(600, samples),
            'Bwd_IAT_Total': np.random.exponential(6000, samples),
            'Bwd_IAT_Mean': np.random.exponential(3000, samples),
        }
        
        # 노이즈 추가
        for key, values in base_data.items():
            base_data[key] = self._add_realistic_noise(values, 0.3)
        
        base_data['Label'] = ['PortScan'] * samples
        return base_data

# 사용 예시
if __name__ == "__main__":
    generator = RealisticSecurityDataGenerator()
    realistic_data = generator.generate_realistic_sample_data(total_samples=10000, attack_ratio=0.4)
    
    print(f"생성된 데이터 크기: {realistic_data.shape}")
    print(f"라벨 분포:\n{realistic_data['Label'].value_counts()}")
    
    # 간단한 분리도 테스트
    normal_packets = realistic_data[realistic_data['Label'] == 'BENIGN']['Total_Fwd_Packets']
    ddos_packets = realistic_data[realistic_data['Label'] == 'DDoS']['Total_Fwd_Packets']
    
    print(f"\n패킷 수 비교:")
    print(f"정상: 평균 {normal_packets.mean():.1f} ± {normal_packets.std():.1f}")
    print(f"DDoS: 평균 {ddos_packets.mean():.1f} ± {ddos_packets.std():.1f}")
    print(f"비율: {ddos_packets.mean()/normal_packets.mean():.1f}x")
