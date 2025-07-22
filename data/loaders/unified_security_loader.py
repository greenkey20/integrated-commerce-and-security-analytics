"""
통합 보안 데이터 로더 (Unified Security Data Loader)

기존 core/security/data_loader.py와 data/loaders/security_loader.py를 통합한 모듈.
retail 패턴에 맞춘 체계적 구조와 CICIDS2017 전용 상세 기능을 결합.

주요 기능:
- 실제 CICIDS2017 데이터셋 로딩 및 전처리
- 다양한 공격 유형별 시뮬레이션 데이터 생성
- 새로운 데이터 계층 구조 활용 (DataValidator, DataCleaner)
- Type hints, 로깅, 하위 호환성 지원
"""

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
import logging

# 새로운 데이터 계층 import
try:
    from data.base import DataValidator, DataCleaner
except ImportError:
    # 하위 호환성을 위한 fallback
    DataValidator = None
    DataCleaner = None

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class UnifiedSecurityLoader:
    """
    통합 보안 데이터 로더

    CICIDS2017 데이터셋의 로딩, 전처리, 샘플 데이터 생성을 담당하는 통합 클래스.
    기존 core와 data/loaders 모듈의 장점을 결합하여 구현.
    """

    def __init__(self, data_dir: str = "data/cicids2017"):
        """
        초기화 메서드

        Args:
            data_dir: CICIDS2017 데이터 디렉토리 경로
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # 파일 인코딩 지원
        self.supported_encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']

        # 새로운 데이터 계층 활용 (사용 가능한 경우)
        self.validator = DataValidator() if DataValidator else None
        self.cleaner = DataCleaner() if DataCleaner else None

        # CICIDS2017 컬럼 매핑 정의
        self.column_mapping = {
            ' Destination Port': 'Destination_Port',
            ' Flow Duration': 'Flow_Duration',
            ' Total Fwd Packets': 'Total_Fwd_Packets',
            ' Total Backward Packets': 'Total_Backward_Packets',
            ' Total Length of Fwd Packets': 'Total_Length_of_Fwd_Packets',
            ' Total Length of Bwd Packets': 'Total_Length_of_Bwd_Packets',
            ' Fwd Packet Length Max': 'Fwd_Packet_Length_Max',
            ' Fwd Packet Length Min': 'Fwd_Packet_Length_Min',
            ' Fwd Packet Length Mean': 'Fwd_Packet_Length_Mean',
            ' Bwd Packet Length Max': 'Bwd_Packet_Length_Max',
            ' Bwd Packet Length Min': 'Bwd_Packet_Length_Min',
            ' Bwd Packet Length Mean': 'Bwd_Packet_Length_Mean',
            ' Flow Bytes/s': 'Flow_Bytes_s',
            ' Flow Packets/s': 'Flow_Packets_s',
            ' Flow IAT Mean': 'Flow_IAT_Mean',
            ' Flow IAT Std': 'Flow_IAT_Std',
            ' Fwd IAT Total': 'Fwd_IAT_Total',
            ' Fwd IAT Mean': 'Fwd_IAT_Mean',
            ' Bwd IAT Total': 'Bwd_IAT_Total',
            ' Bwd IAT Mean': 'Bwd_IAT_Mean',
            ' Label': 'Label'
        }

        # 핵심 보안 특성 정의 (19개 → 10개 핵심)
        self.core_security_features = [
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

        logger.info(f"UnifiedSecurityLoader 초기화 완료: {self.data_dir}")

    def check_data_availability(self) -> Dict[str, Union[bool, List[str], int]]:
        """
        CICIDS2017 데이터 파일 존재 확인

        Returns:
            Dict: 파일 가용성 정보
        """
        patterns = ["*.csv", "*ISCX.csv", "*cicids*.csv", "*CIC*.csv"]
        files = []

        for pattern in patterns:
            files.extend(glob.glob(str(self.data_dir / pattern)))

        # Monday 파일 제외 (공격 데이터 거의 없음)
        filtered_files = []
        for file_path in files:
            filename = os.path.basename(file_path)
            if not filename.startswith('Monday'):
                filtered_files.append(file_path)

        logger.info(f"CICIDS2017 파일 스캔 완료: {len(filtered_files)}개 발견")

        return {
            "available": len(filtered_files) > 0,
            "files": filtered_files,
            "count": len(filtered_files),
            "data_dir": str(self.data_dir)
        }

    def find_label_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        데이터프레임에서 라벨 컬럼 찾기

        Args:
            df: 입력 데이터프레임

        Returns:
            str: 라벨 컬럼명 또는 None
        """
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

    def standardize_labels(self, labels: pd.Series) -> pd.Series:
        """
        라벨 표준화

        Args:
            labels: 원본 라벨 시리즈

        Returns:
            pd.Series: 표준화된 라벨
        """
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

    def load_file_with_encoding(self, file_path: Union[str, Path], max_rows: int = 10000) -> Tuple[pd.DataFrame, str]:
        """
        여러 인코딩을 시도하여 파일 로드

        Args:
            file_path: 파일 경로
            max_rows: 최대 로드할 행 수

        Returns:
            Tuple[pd.DataFrame, str]: (데이터프레임, 사용된 인코딩)
        """
        file_path = Path(file_path)

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

                # 컬럼명 매핑 적용
                df = df.rename(columns=self.column_mapping)

                # 라벨 컬럼 찾기 및 표준화
                label_column = self.find_label_column(df)
                if label_column and label_column != 'Label':
                    df = df.rename(columns={label_column: 'Label'})

                if 'Label' in df.columns:
                    df['Label'] = self.standardize_labels(df['Label'])

                logger.info(f"파일 로드 성공: {file_path} (인코딩: {encoding})")
                return df, encoding

            except Exception as e:
                logger.debug(f"인코딩 {encoding} 실패: {e}")
                continue

        raise ValueError(f"모든 인코딩 방법으로 {file_path} 파일을 읽을 수 없습니다")

    def load_real_data(self, file_path: Optional[Union[str, Path]] = None, max_rows_per_file: int = 3000) -> Optional[
        Tuple[pd.DataFrame, List[Dict]]]:
        """
        실제 CICIDS2017 데이터 로드

        Args:
            file_path: 특정 파일 경로 (None이면 자동 탐색)
            max_rows_per_file: 파일당 최대 로드 행 수

        Returns:
            Tuple[pd.DataFrame, List[Dict]]: (통합 데이터, 파일 정보) 또는 None
        """
        if file_path is not None:
            # 특정 파일 로드
            try:
                df, encoding = self.load_file_with_encoding(file_path, max_rows_per_file)
                if self.validator:
                    self.validator.validate_dataframe(df)
                df = self._clean_real_data(df)

                file_info = [{
                    'filename': Path(file_path).name,
                    'records': len(df),
                    'attacks': (df['Label'] != 'BENIGN').sum() if 'Label' in df.columns else 0,
                    'encoding': encoding
                }]

                return df, file_info

            except Exception as e:
                logger.error(f"파일 로드 실패: {e}")
                return None

        # 자동 탐색 및 다중 파일 로드
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
            file_path = self.data_dir / filename

            if file_path.exists():
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

                        logger.info(f"파일 로드: {filename} - {len(df)}개 레코드, 공격 {attack_ratio:.1f}%")

                except Exception as e:
                    logger.warning(f"파일 로드 실패 ({filename}): {e}")
                    continue

        if combined_data:
            final_data = pd.concat(combined_data, ignore_index=True)
            final_data = self._clean_real_data(final_data)

            logger.info(f"실제 데이터 로드 완료: {len(final_data)}개 레코드")
            return final_data, successful_files
        else:
            logger.warning("로드 가능한 CICIDS2017 파일이 없습니다.")
            return None

    def generate_sample_data(self, total_samples: int = 10000, attack_ratio: float = 0.3) -> pd.DataFrame:
        """
        CICIDS2017 스타일 샘플 데이터 생성

        Args:
            total_samples: 생성할 총 샘플 수
            attack_ratio: 전체 공격 비율 (0.0 - 1.0)

        Returns:
            pd.DataFrame: 생성된 샘플 데이터
        """
        logger.info(f"CICIDS2017 스타일 샘플 데이터 생성 중... (총 {total_samples}개, 공격 비율 {attack_ratio:.1%})")

        np.random.seed(42)

        # 샘플 분배
        normal_samples = int(total_samples * (1 - attack_ratio))
        attack_samples = total_samples - normal_samples

        # 공격 유형별 분배
        ddos_samples = int(attack_samples * 0.4)
        web_attack_samples = int(attack_samples * 0.25)
        brute_force_samples = int(attack_samples * 0.2)
        port_scan_samples = attack_samples - ddos_samples - web_attack_samples - brute_force_samples

        # 각 트래픽 패턴 생성
        normal_data = self._generate_normal_traffic(normal_samples)
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
        df = self._clean_generated_data(df)

        # 기본 검증
        if self.validator:
            self.validator.validate_dataframe(df)

        logger.info(f"샘플 데이터 생성 완료:")
        logger.info(f"- 정상: {normal_samples}, DDoS: {ddos_samples}, 웹공격: {web_attack_samples}")
        logger.info(f"- 브루트포스: {brute_force_samples}, 포트스캔: {port_scan_samples}")
        logger.info(f"라벨 분포:\n{df['Label'].value_counts()}")

        return df

    def _generate_normal_traffic(self, samples: int) -> Dict[str, List]:
        """정상 트래픽 패턴 생성"""
        return {
            'Flow_Duration': np.random.exponential(1000000, samples),
            'Total_Fwd_Packets': np.random.poisson(15, samples),
            'Total_Backward_Packets': np.random.poisson(12, samples),
            'Total_Length_of_Fwd_Packets': np.random.normal(800, 300, samples),
            'Total_Length_of_Bwd_Packets': np.random.normal(600, 200, samples),
            'Fwd_Packet_Length_Max': np.random.normal(1200, 400, samples),
            'Fwd_Packet_Length_Min': np.random.normal(60, 20, samples),
            'Fwd_Packet_Length_Mean': np.random.normal(400, 150, samples),
            'Bwd_Packet_Length_Max': np.random.normal(1000, 300, samples),
            'Bwd_Packet_Length_Min': np.random.normal(50, 15, samples),
            'Bwd_Packet_Length_Mean': np.random.normal(300, 100, samples),
            'Flow_Bytes_s': np.random.normal(2000, 1000, samples),
            'Flow_Packets_s': np.random.normal(20, 10, samples),
            'Flow_IAT_Mean': np.random.exponential(50000, samples),
            'Flow_IAT_Std': np.random.exponential(25000, samples),
            'Fwd_IAT_Total': np.random.exponential(200000, samples),
            'Fwd_IAT_Mean': np.random.exponential(20000, samples),
            'Bwd_IAT_Total': np.random.exponential(150000, samples),
            'Bwd_IAT_Mean': np.random.exponential(15000, samples),
            'Label': ['BENIGN'] * samples
        }

    def _generate_ddos_traffic(self, samples: int) -> Dict[str, List]:
        """DDoS 공격 패턴 생성"""
        return {
            'Flow_Duration': np.random.exponential(10000, samples),
            'Total_Fwd_Packets': np.random.poisson(200, samples),
            'Total_Backward_Packets': np.random.poisson(5, samples),
            'Total_Length_of_Fwd_Packets': np.random.normal(10000, 2000, samples),
            'Total_Length_of_Bwd_Packets': np.random.normal(200, 100, samples),
            'Fwd_Packet_Length_Max': np.random.normal(1500, 100, samples),
            'Fwd_Packet_Length_Min': np.random.normal(64, 10, samples),
            'Fwd_Packet_Length_Mean': np.random.normal(80, 20, samples),
            'Bwd_Packet_Length_Max': np.random.normal(150, 50, samples),
            'Bwd_Packet_Length_Min': np.random.normal(40, 10, samples),
            'Bwd_Packet_Length_Mean': np.random.normal(60, 20, samples),
            'Flow_Bytes_s': np.random.normal(50000, 15000, samples),
            'Flow_Packets_s': np.random.normal(500, 150, samples),
            'Flow_IAT_Mean': np.random.exponential(1000, samples),
            'Flow_IAT_Std': np.random.exponential(500, samples),
            'Fwd_IAT_Total': np.random.exponential(5000, samples),
            'Fwd_IAT_Mean': np.random.exponential(50, samples),
            'Bwd_IAT_Total': np.random.exponential(20000, samples),
            'Bwd_IAT_Mean': np.random.exponential(2000, samples),
            'Label': ['DDoS'] * samples
        }

    def _generate_web_attack_traffic(self, samples: int) -> Dict[str, List]:
        """웹 공격 패턴 생성"""
        return {
            'Flow_Duration': np.random.exponential(150000, samples),
            'Total_Fwd_Packets': np.random.poisson(30, samples),
            'Total_Backward_Packets': np.random.poisson(25, samples),
            'Total_Length_of_Fwd_Packets': np.random.normal(3000, 800, samples),
            'Total_Length_of_Bwd_Packets': np.random.normal(1500, 400, samples),
            'Fwd_Packet_Length_Max': np.random.normal(1400, 200, samples),
            'Fwd_Packet_Length_Min': np.random.normal(200, 50, samples),
            'Fwd_Packet_Length_Mean': np.random.normal(500, 100, samples),
            'Bwd_Packet_Length_Max': np.random.normal(800, 150, samples),
            'Bwd_Packet_Length_Min': np.random.normal(100, 30, samples),
            'Bwd_Packet_Length_Mean': np.random.normal(250, 80, samples),
            'Flow_Bytes_s': np.random.normal(4000, 1500, samples),
            'Flow_Packets_s': np.random.normal(25, 10, samples),
            'Flow_IAT_Mean': np.random.exponential(30000, samples),
            'Flow_IAT_Std': np.random.exponential(15000, samples),
            'Fwd_IAT_Total': np.random.exponential(100000, samples),
            'Fwd_IAT_Mean': np.random.exponential(8000, samples),
            'Bwd_IAT_Total': np.random.exponential(80000, samples),
            'Bwd_IAT_Mean': np.random.exponential(6000, samples),
            'Label': ['Web Attack'] * samples
        }

    def _generate_brute_force_traffic(self, samples: int) -> Dict[str, List]:
        """브루트포스 공격 패턴 생성"""
        return {
            'Flow_Duration': np.random.exponential(30000, samples),
            'Total_Fwd_Packets': np.random.poisson(80, samples),
            'Total_Backward_Packets': np.random.poisson(8, samples),
            'Total_Length_of_Fwd_Packets': np.random.normal(2000, 500, samples),
            'Total_Length_of_Bwd_Packets': np.random.normal(400, 150, samples),
            'Fwd_Packet_Length_Max': np.random.normal(800, 200, samples),
            'Fwd_Packet_Length_Min': np.random.normal(40, 15, samples),
            'Fwd_Packet_Length_Mean': np.random.normal(80, 30, samples),
            'Bwd_Packet_Length_Max': np.random.normal(300, 100, samples),
            'Bwd_Packet_Length_Min': np.random.normal(30, 10, samples),
            'Bwd_Packet_Length_Mean': np.random.normal(60, 20, samples),
            'Flow_Bytes_s': np.random.normal(8000, 2000, samples),
            'Flow_Packets_s': np.random.normal(80, 20, samples),
            'Flow_IAT_Mean': np.random.exponential(3000, samples),
            'Flow_IAT_Std': np.random.exponential(1500, samples),
            'Fwd_IAT_Total': np.random.exponential(15000, samples),
            'Fwd_IAT_Mean': np.random.exponential(300, samples),
            'Bwd_IAT_Total': np.random.exponential(25000, samples),
            'Bwd_IAT_Mean': np.random.exponential(2500, samples),
            'Label': ['Brute Force'] * samples
        }

    def _generate_port_scan_traffic(self, samples: int) -> Dict[str, List]:
        """포트스캔 공격 패턴 생성"""
        return {
            'Flow_Duration': np.random.exponential(5000, samples),
            'Total_Fwd_Packets': np.random.poisson(10, samples),
            'Total_Backward_Packets': np.random.poisson(2, samples),
            'Total_Length_of_Fwd_Packets': np.random.normal(400, 150, samples),
            'Total_Length_of_Bwd_Packets': np.random.normal(100, 50, samples),
            'Fwd_Packet_Length_Max': np.random.normal(200, 60, samples),
            'Fwd_Packet_Length_Min': np.random.normal(40, 10, samples),
            'Fwd_Packet_Length_Mean': np.random.normal(60, 20, samples),
            'Bwd_Packet_Length_Max': np.random.normal(100, 30, samples),
            'Bwd_Packet_Length_Min': np.random.normal(20, 5, samples),
            'Bwd_Packet_Length_Mean': np.random.normal(40, 15, samples),
            'Flow_Bytes_s': np.random.normal(1000, 300, samples),
            'Flow_Packets_s': np.random.normal(30, 10, samples),
            'Flow_IAT_Mean': np.random.exponential(8000, samples),
            'Flow_IAT_Std': np.random.exponential(4000, samples),
            'Fwd_IAT_Total': np.random.exponential(3000, samples),
            'Fwd_IAT_Mean': np.random.exponential(800, samples),
            'Bwd_IAT_Total': np.random.exponential(8000, samples),
            'Bwd_IAT_Mean': np.random.exponential(4000, samples),
            'Label': ['PortScan'] * samples
        }

    def _clean_real_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """실제 데이터 정제"""
        logger.info("실제 데이터 정제 중...")

        # 수치형 컬럼 식별
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # 무한대 값 및 NaN 처리
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        df[numeric_cols] = df[numeric_cols].abs()  # 음수값 처리

        # 새로운 정제 시스템 활용 (사용 가능한 경우)
        if self.cleaner:
            df = self.cleaner.handle_missing_values(df, strategy={
                col: 'median' for col in numeric_cols
            })

        logger.info("실제 데이터 정제 완료")
        return df

    def _clean_generated_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """생성된 데이터 정제"""
        logger.info("생성된 데이터 정제 중...")

        # 수치형 컬럼 처리
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].abs()  # 음수 제거
        df = df.replace([np.inf, -np.inf], np.nan)  # 무한대 처리

        # 새로운 정제 시스템 활용
        if self.cleaner:
            df = self.cleaner.handle_missing_values(df, strategy={
                col: 'median' for col in numeric_columns
            })
        else:
            # Fallback: 기본 NaN 처리
            df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

        # 데이터 섞기
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        logger.info("생성된 데이터 정제 완료")
        return df

    def preprocess_for_security_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        보안 분석용 전처리

        Args:
            df: 원본 데이터프레임

        Returns:
            pd.DataFrame: 전처리된 데이터 (X + y + original_labels)
        """
        logger.info("보안 분석용 전처리 시작...")

        # 사용 가능한 핵심 특성 선택
        available_features = [col for col in self.core_security_features if col in df.columns]

        if len(available_features) == 0:
            logger.error("사용 가능한 보안 분석 특성이 없습니다.")
            raise ValueError("보안 분석에 필요한 특성이 데이터에 없습니다.")

        # 특성 데이터 추출
        X = df[available_features].copy()

        # 라벨 이진화 (BENIGN = 0, 나머지 = 1)
        if 'Label' in df.columns:
            y = (df['Label'] != 'BENIGN').astype(int)
            original_labels = df['Label']
        else:
            logger.warning("Label 컬럼이 없어 임의 라벨을 생성합니다.")
            y = np.zeros(len(df))
            original_labels = ['UNKNOWN'] * len(df)

        # 이상치 제거 (새로운 시스템 활용)
        if self.cleaner:
            X_cleaned = self.cleaner.remove_outliers(
                X,
                columns=available_features,
                method='iqr',
                threshold=3.0  # 관대한 기준
            )
        else:
            # Fallback: 기본 처리
            X_cleaned = X.copy()

        # 결과 DataFrame 생성
        result_df = X_cleaned.copy()
        result_df['Label'] = y
        result_df['Original_Label'] = original_labels

        logger.info("보안 분석용 전처리 완료:")
        logger.info(f"- 사용된 특성: {available_features}")
        logger.info(f"- 정상 샘플: {(y == 0).sum()}")
        logger.info(f"- 공격 샘플: {(y == 1).sum()}")

        return result_df

    def get_data_quality_report(self, df: pd.DataFrame) -> Dict:
        """
        데이터 품질 보고서 생성

        Args:
            df: 분석할 데이터프레임

        Returns:
            Dict: 품질 보고서
        """
        report = {
            'basic_info': {
                'total_records': len(df),
                'total_columns': len(df.columns),
                'missing_values': df.isnull().sum().sum(),
                'duplicate_rows': df.duplicated().sum(),
                'numeric_columns': len(df.select_dtypes(include=[np.number]).columns)
            }
        }

        if 'Label' in df.columns:
            label_counts = df['Label'].value_counts()
            report['label_analysis'] = {
                'unique_labels': len(label_counts),
                'distribution': label_counts.to_dict(),
                'attack_ratio': (df['Label'] != 'BENIGN').sum() / len(df) * 100 if len(df) > 0 else 0
            }

        # 수치형 특성 요약
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            report['numeric_summary'] = {
                'mean_values': numeric_df.mean().to_dict(),
                'std_values': numeric_df.std().to_dict(),
                'zero_variance_features': (numeric_df.std() == 0).sum()
            }

        return report

    def save_processed_data(self, df: pd.DataFrame, filename: Optional[str] = None) -> str:
        """
        처리된 데이터를 저장

        Args:
            df: 저장할 데이터프레임
            filename: 파일명 (None이면 자동 생성)

        Returns:
            str: 저장된 파일 경로
        """
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"unified_security_data_{timestamp}.csv"

        # 저장 디렉토리 생성
        save_dir = Path("data/processed/security")
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / filename
        df.to_csv(save_path, index=False)

        logger.info(f"보안 데이터 저장 완료: {save_path}")
        return str(save_path)


# ============================================================================
# 하위 호환성을 위한 편의 함수들 (기존 코드 호환성)
# ============================================================================

def check_cicids_data_availability() -> Dict:
    """하위 호환성을 위한 편의 함수"""
    loader = UnifiedSecurityLoader()
    return loader.check_data_availability()


def generate_cicids_sample_data(n_samples: int = 10000) -> pd.DataFrame:
    """하위 호환성을 위한 편의 함수"""
    loader = UnifiedSecurityLoader()
    return loader.generate_sample_data(total_samples=n_samples)


def generate_enhanced_sample_data(n_samples: int = 10000) -> pd.DataFrame:
    """하위 호환성을 위한 편의 함수"""
    loader = UnifiedSecurityLoader()
    return loader.generate_sample_data(total_samples=n_samples, attack_ratio=0.6)


# 클래스 별칭 (하위 호환성)
CICIDSDataLoader = UnifiedSecurityLoader
SecurityDataLoader = UnifiedSecurityLoader


# ============================================================================
# 사용 예시 및 테스트 함수
# ============================================================================

def demo_unified_security_loader():
    """
    통합 보안 로더 사용 예시
    """
    print("=== 통합 보안 데이터 로더 데모 ===")

    # 로더 초기화
    loader = UnifiedSecurityLoader()

    # 1. 데이터 가용성 확인
    availability = loader.check_data_availability()
    print(f"실제 데이터 가용성: {availability['available']}")
    print(f"발견된 파일 수: {availability['count']}")

    # 2. 샘플 데이터 생성
    print("\n샘플 데이터 생성 중...")
    sample_data = loader.generate_sample_data(total_samples=1000, attack_ratio=0.3)
    print(f"생성된 샘플 수: {len(sample_data)}")
    print(f"라벨 분포:\n{sample_data['Label'].value_counts()}")

    # 3. 보안 분석용 전처리
    print("\n보안 분석용 전처리...")
    processed_data = loader.preprocess_for_security_analysis(sample_data)
    print(f"전처리된 데이터 형태: {processed_data.shape}")

    # 4. 품질 보고서
    print("\n데이터 품질 보고서...")
    quality_report = loader.get_data_quality_report(sample_data)
    print(f"총 레코드: {quality_report['basic_info']['total_records']}")
    print(f"공격 비율: {quality_report['label_analysis']['attack_ratio']:.1f}%")

    # 5. 데이터 저장
    saved_path = loader.save_processed_data(processed_data, "demo_security_data.csv")
    print(f"\n데이터 저장됨: {saved_path}")

    print("\n=== 데모 완료 ===")


if __name__ == "__main__":
    demo_unified_security_loader()