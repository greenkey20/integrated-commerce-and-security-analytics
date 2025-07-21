"""
CICIDS2017 데이터 로더 (마이그레이션됨)

기존 core/security/cicids_data_loader.py에서 새로운 데이터 계층 구조로 마이그레이션됨.
새로운 data/base 클래스들을 활용하여 더 체계적인 데이터 처리 제공.
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from typing import Tuple, Optional

# 새로운 데이터 계층 import
from data.base import DataValidator, DataCleaner
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class SecurityDataLoader:
    """
    CICIDS2017 데이터셋 로딩 및 전처리를 담당하는 클래스
    
    새로운 데이터 계층 구조를 활용하여 더 체계적인 데이터 처리를 제공합니다.
    기존 core/security/cicids_data_loader.py에서 마이그레이션됨.
    """
    
    def __init__(self, data_dir: str = "data/cicids2017"):
        """초기화 메서드"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 새로운 검증 및 정제 클래스 활용
        self.validator = DataValidator()
        self.cleaner = DataCleaner()
        
        # 컬럼 매핑 정의
        self.column_mapping = {
            ' Destination Port': 'Destination_Port',
            ' Flow Duration': 'Flow_Duration',
            ' Total Fwd Packets': 'Total_Fwd_Packets',
            ' Total Backward Packets': 'Total_Backward_Packets',
            ' Label': 'Label'
        }
        
        logger.info(f"SecurityDataLoader 초기화 완료: {self.data_dir}")
        
    def generate_sample_data(self, total_samples: int = 10000) -> pd.DataFrame:
        """
        CICIDS2017 스타일 샘플 데이터 생성
        
        Args:
            total_samples: 생성할 총 샘플 수
            
        Returns:
            pd.DataFrame: 생성된 샘플 데이터
        """
        logger.info(f"CICIDS2017 스타일 샘플 데이터 생성 중... (총 {total_samples}개)")
        
        np.random.seed(42)
        
        # 데이터 분포 설정
        normal_ratio = 0.8
        ddos_ratio = 0.1
        web_attack_ratio = 0.05
        brute_force_ratio = 0.03
        port_scan_ratio = 0.02
        
        normal_samples = int(total_samples * normal_ratio)
        ddos_samples = int(total_samples * ddos_ratio)
        web_attack_samples = int(total_samples * web_attack_ratio)
        brute_force_samples = int(total_samples * brute_force_ratio)
        port_scan_samples = total_samples - (normal_samples + ddos_samples + web_attack_samples + brute_force_samples)
        
        # 정상 트래픽 패턴 생성
        normal_data = self._generate_normal_traffic(normal_samples)
        
        # 공격 패턴들 생성
        ddos_data = self._generate_ddos_traffic(ddos_samples)
        web_attack_data = self._generate_web_attack_traffic(web_attack_samples)
        brute_force_data = self._generate_brute_force_traffic(brute_force_samples)
        port_scan_data = self._generate_port_scan_traffic(port_scan_samples)
        
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
        
        # 새로운 데이터 정제 시스템 활용
        df = self._clean_generated_data(df)
        
        # 기본 검증
        self.validator.validate_dataframe(df)
        
        logger.info(f"샘플 데이터 생성 완료: {len(df)}개 레코드")
        logger.info(f"라벨 분포:\n{df['Label'].value_counts()}")
        
        return df
    
    def _generate_normal_traffic(self, samples: int) -> dict:
        """정상 트래픽 패턴 생성"""
        return {
            'Flow_Duration': np.random.exponential(1000000, samples),
            'Total_Fwd_Packets': np.random.poisson(10, samples),
            'Total_Backward_Packets': np.random.poisson(8, samples),
            'Flow_Bytes_s': np.random.normal(1000, 500, samples),
            'Flow_Packets_s': np.random.normal(10, 5, samples),
            'Flow_IAT_Mean': np.random.exponential(100000, samples),
            'Fwd_IAT_Mean': np.random.exponential(50000, samples),
            'Bwd_IAT_Mean': np.random.exponential(45000, samples),
            'Fwd_Packet_Length_Mean': np.random.normal(200, 80, samples),
            'Bwd_Packet_Length_Mean': np.random.normal(180, 70, samples),
            'Label': ['BENIGN'] * samples
        }
    
    def _generate_ddos_traffic(self, samples: int) -> dict:
        """DDoS 공격 패턴 생성"""
        return {
            'Flow_Duration': np.random.exponential(100000, samples),
            'Total_Fwd_Packets': np.random.poisson(100, samples),
            'Total_Backward_Packets': np.random.poisson(2, samples),
            'Flow_Bytes_s': np.random.normal(10000, 3000, samples),
            'Flow_Packets_s': np.random.normal(100, 30, samples),
            'Flow_IAT_Mean': np.random.exponential(1000, samples),
            'Fwd_IAT_Mean': np.random.exponential(100, samples),
            'Bwd_IAT_Mean': np.random.exponential(5000, samples),
            'Fwd_Packet_Length_Mean': np.random.normal(60, 20, samples),
            'Bwd_Packet_Length_Mean': np.random.normal(30, 10, samples),
            'Label': ['DDoS'] * samples
        }
    
    def _generate_web_attack_traffic(self, samples: int) -> dict:
        """웹 공격 패턴 생성"""
        return {
            'Flow_Duration': np.random.exponential(200000, samples),
            'Total_Fwd_Packets': np.random.poisson(20, samples),
            'Total_Backward_Packets': np.random.poisson(15, samples),
            'Flow_Bytes_s': np.random.normal(2000, 800, samples),
            'Flow_Packets_s': np.random.normal(15, 8, samples),
            'Flow_IAT_Mean': np.random.exponential(80000, samples),
            'Fwd_IAT_Mean': np.random.exponential(15000, samples),
            'Bwd_IAT_Mean': np.random.exponential(12000, samples),
            'Fwd_Packet_Length_Mean': np.random.normal(300, 100, samples),
            'Bwd_Packet_Length_Mean': np.random.normal(150, 50, samples),
            'Label': ['Web Attack'] * samples
        }
    
    def _generate_brute_force_traffic(self, samples: int) -> dict:
        """브루트포스 공격 패턴 생성"""
        return {
            'Flow_Duration': np.random.exponential(50000, samples),
            'Total_Fwd_Packets': np.random.poisson(50, samples),
            'Total_Backward_Packets': np.random.poisson(3, samples),
            'Flow_Bytes_s': np.random.normal(5000, 1000, samples),
            'Flow_Packets_s': np.random.normal(50, 15, samples),
            'Flow_IAT_Mean': np.random.exponential(2000, samples),
            'Fwd_IAT_Mean': np.random.exponential(500, samples),
            'Bwd_IAT_Mean': np.random.exponential(3000, samples),
            'Fwd_Packet_Length_Mean': np.random.normal(50, 15, samples),
            'Bwd_Packet_Length_Mean': np.random.normal(40, 10, samples),
            'Label': ['Brute Force'] * samples
        }
    
    def _generate_port_scan_traffic(self, samples: int) -> dict:
        """포트 스캔 패턴 생성"""
        return {
            'Flow_Duration': np.random.exponential(10000, samples),
            'Total_Fwd_Packets': np.random.poisson(5, samples),
            'Total_Backward_Packets': np.random.poisson(1, samples),
            'Flow_Bytes_s': np.random.normal(500, 100, samples),
            'Flow_Packets_s': np.random.normal(20, 5, samples),
            'Flow_IAT_Mean': np.random.exponential(5000, samples),
            'Fwd_IAT_Mean': np.random.exponential(1000, samples),
            'Bwd_IAT_Mean': np.random.exponential(8000, samples),
            'Fwd_Packet_Length_Mean': np.random.normal(40, 10, samples),
            'Bwd_Packet_Length_Mean': np.random.normal(25, 5, samples),
            'Label': ['PortScan'] * samples
        }
    
    def _clean_generated_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        생성된 데이터 정제 (새로운 cleaner 활용)
        """
        logger.info("생성된 데이터 정제 중...")
        
        # 음수 값 제거
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].abs()
        
        # 무한대 값 처리
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # 새로운 데이터 정제 시스템 활용
        df_cleaned = self.cleaner.handle_missing_values(df, strategy={
            col: 'median' for col in numeric_columns
        })
        
        # 데이터 섞기
        df_cleaned = df_cleaned.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info("데이터 정제 완료")
        return df_cleaned
    
    def load_real_data(self, file_path: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        실제 CICIDS2017 데이터 로드
        
        Args:
            file_path: 데이터 파일 경로 (None이면 자동 탐색)
            
        Returns:
            pd.DataFrame: 로드된 데이터 또는 None
        """
        if file_path is None:
            # 일반적인 CICIDS2017 파일명들
            possible_files = [
                self.data_dir / "Monday-WorkingHours.pcap_ISCX.csv",
                self.data_dir / "Tuesday-WorkingHours.pcap_ISCX.csv",
                self.data_dir / "Wednesday-workingHours.pcap_ISCX.csv",
                self.data_dir / "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
            ]
            
            for file_path in possible_files:
                if file_path.exists():
                    break
            else:
                logger.warning("실제 CICIDS2017 파일을 찾을 수 없습니다.")
                logger.info("다음 중 하나를 다운로드하세요:")
                logger.info("1. https://www.unb.ca/cic/datasets/ids-2017.html")
                logger.info("2. Kaggle: https://www.kaggle.com/cicdataset/cicids2017")
                return None
        
        try:
            logger.info(f"CICIDS2017 데이터 로드 중: {file_path}")
            df = pd.read_csv(file_path)
            
            # 컬럼명 정리
            df.columns = df.columns.str.strip()
            
            # 기본 검증
            self.validator.validate_dataframe(df)
            
            # 데이터 정제
            df = self._clean_real_data(df)
            
            logger.info(f"데이터 로드 완료: {len(df)} 샘플")
            logger.info(f"라벨 분포:\n{df['Label'].value_counts()}")
            
            return df
            
        except Exception as e:
            logger.error(f"데이터 로드 오류: {e}")
            return None
    
    def _clean_real_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """실제 데이터 정제"""
        logger.info("실제 데이터 정제 중...")
        
        # 무한대 값 처리
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # 새로운 정제 시스템 활용
        df_cleaned = self.cleaner.handle_missing_values(df)
        
        logger.info("실제 데이터 정제 완료")
        return df_cleaned
    
    def preprocess_for_security_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        보안 분석용 전처리
        
        Args:
            df: 원본 데이터프레임
            
        Returns:
            pd.DataFrame: 전처리된 데이터
        """
        logger.info("보안 분석용 전처리 시작...")
        
        # API 로그 관련 특성 선택
        security_features = [
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
        available_features = [col for col in security_features if col in df.columns]
        
        if len(available_features) == 0:
            logger.error("사용 가능한 보안 분석 특성이 없습니다.")
            return None
        
        # 특성 데이터 추출
        X = df[available_features].copy()
        
        # 라벨 이진화 (BENIGN = 0, 나머지 = 1)
        y = (df['Label'] != 'BENIGN').astype(int)
        
        # 새로운 이상치 제거 시스템 활용 (더 관대한 기준)
        X_cleaned = self.cleaner.remove_outliers(
            X, 
            columns=available_features,
            method='iqr',
            threshold=3.0  # 더 관대한 기준
        )
        
        # 결과 DataFrame 생성
        result_df = X_cleaned.copy()
        result_df['Label'] = y
        result_df['Original_Label'] = df['Label']
        
        logger.info("보안 분석용 전처리 완료:")
        logger.info(f"- 사용된 특성: {available_features}")
        logger.info(f"- 정상 샘플: {(y == 0).sum()}")
        logger.info(f"- 공격 샘플: {(y == 1).sum()}")
        
        return result_df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = None) -> str:
        """
        처리된 데이터를 security 도메인 폴더에 저장
        
        새로운 데이터 계층 구조를 활용한 데이터 저장 기능
        """
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"security_data_{timestamp}.csv"
        
        save_path = f"data/processed/security/{filename}"
        df.to_csv(save_path, index=False)
        
        logger.info(f"보안 데이터 저장 완료: {save_path}")
        return save_path
