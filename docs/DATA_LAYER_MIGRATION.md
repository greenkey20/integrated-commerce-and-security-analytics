# 🚀 데이터 계층 리팩토링 마이그레이션 가이드

## 📅 생성일: 2025-07-20
## 🎯 목표: ML 계층 통합을 위한 데이터 계층 재구성

---

## 📊 새로운 데이터 계층 구조

```
data/
├── __init__.py                    # 통합 DataProcessor 클래스
├── base/                          # 기본 데이터 처리 클래스들 (도메인 중립)
│   ├── __init__.py
│   ├── validator.py              # DataValidator - 데이터 검증
│   ├── cleaner.py                # DataCleaner - 데이터 정제  
│   ├── engineer.py               # FeatureEngineer - 특성 공학
│   └── splitter.py               # DataSplitter - 데이터 분할
├── loaders/                       # 도메인별 데이터 로더들
│   ├── __init__.py
│   ├── retail_loader.py          # 리테일 데이터 로더 (core/retail/data_loader.py에서 이동)
│   └── security_loader.py        # 보안 데이터 로더 (core/security/cicids_data_loader.py에서 이동)
├── processors/                    # 도메인별 특화 처리기들
│   ├── __init__.py
│   ├── retail_processor.py       # 리테일 특화 전처리
│   └── security_processor.py     # 보안 특화 전처리
├── raw/                          # 원본 데이터 (기존 유지)
│   ├── Mall_Customers.csv
│   └── cicids2017/
└── processed/                    # 처리된 데이터 저장소 (이제 활용 가능!)
    ├── retail/                   # 리테일 도메인 처리된 데이터
    ├── security/                 # 보안 도메인 처리된 데이터
    └── segmentation/             # 세그멘테이션 처리된 데이터
```

---

## 🔄 마이그레이션 단계

### 1단계: 기존 코드 백업 ✅ 완료
```bash
# 이미 생성된 새 구조
data/base/validator.py            # common/data_processing.py의 DataValidator 클래스
data/base/cleaner.py              # common/data_processing.py의 DataCleaner 클래스
data/base/engineer.py             # common/data_processing.py의 FeatureEngineer 클래스
data/base/splitter.py             # common/data_processing.py의 DataSplitter 클래스
data/loaders/retail_loader.py     # core/retail/data_loader.py 마이그레이션 예시
```

### 2단계: 남은 데이터 로더들 마이그레이션 (TODO)
```bash
# 수행해야 할 작업
mv core/security/cicids_data_loader.py → data/loaders/security_loader.py
# + 새로운 base 클래스들 활용하도록 코드 수정

# 백업
mv core/retail/data_loader.py → docs/retail_data_loader_backup.py
mv core/security/cicids_data_loader.py → docs/security_data_loader_backup.py
```

### 3단계: 공통 폴더 정리 (TODO)
```bash
# common/data_processing.py는 이제 data/base/로 분산되었으므로 제거
mv common/data_processing.py → docs/common_data_processing_backup.py
rmdir common/  # 비어있으면 제거
```

### 4단계: import 경로 업데이트 (TODO)
기존 코드에서 다음과 같이 수정:
```python
# 기존
from common.data_processing import DataValidator, DataCleaner

# 새로운 방식
from data.base import DataValidator, DataCleaner
# 또는 통합 사용
from data import DataProcessor
```

---

## 💡 새로운 사용법 예시

### 기본 클래스들 개별 사용
```python
from data.base import DataValidator, DataCleaner, FeatureEngineer
from data.loaders.retail_loader import RetailDataLoader

# 데이터 로딩
loader = RetailDataLoader()
df = loader.load_data()

# 검증
validator = DataValidator()
validator.validate_dataframe(df, required_columns=['CustomerID', 'InvoiceNo'])

# 정제
cleaner = DataCleaner()
df_clean = cleaner.handle_missing_values(df)
df_clean = cleaner.remove_outliers(df_clean)

# 특성 공학
engineer = FeatureEngineer()
df_feat = engineer.create_datetime_features(df_clean, 'InvoiceDate')
```

### 통합 데이터 처리기 사용 (권장)
```python
from data import DataProcessor

# 파이프라인 설정
pipeline_config = {
    'validation': {
        'required_columns': ['CustomerID', 'InvoiceNo']
    },
    'missing_values': {
        'CustomerID': 'mode',
        'Quantity': 'median'
    },
    'outliers': {
        'columns': ['Quantity', 'UnitPrice'],
        'method': 'iqr',
        'threshold': 1.5
    },
    'feature_engineering': {
        'datetime_features': ['InvoiceDate'],
        'interaction_features': [('Quantity', 'UnitPrice')]
    },
    'normalization': {
        'method': 'standard',
        'columns': ['Quantity', 'UnitPrice']
    }
}

# 통합 처리
processor = DataProcessor()
df_processed = processor.process_pipeline(df, pipeline_config)

# 처리된 데이터 저장 (이제 processed 폴더 활용!)
save_path = processor.save_processed_data(
    df_processed, 
    domain='retail', 
    filename='processed_retail_data.csv'
)
```

---

## 🎯 이 리팩토링의 장점

### 1. **명확한 책임 분리**
- `data/base/` → 도메인 중립적 기본 처리 로직
- `data/loaders/` → 도메인별 데이터 로딩 로직  
- `data/processors/` → 도메인별 특화 처리 로직

### 2. **재사용성 극대화**
- 모든 도메인이 동일한 기본 클래스들 활용
- 새로운 도메인 추가 시 기본 클래스만 조합하면 됨

### 3. **데이터 저장소 활용**
- `data/processed/` 폴더가 이제 실제로 활용됨
- 도메인별로 처리된 데이터 체계적 관리

### 4. **테스트 용이성**
- 각 클래스가 독립적으로 테스트 가능
- 기본 클래스들은 도메인에 무관하게 테스트

### 5. **ML 계층 통합 준비 완료**
- 데이터 처리가 체계화되어 ML 모델 통합이 쉬워짐
- 모든 도메인이 동일한 데이터 처리 인터페이스 사용

---

## ⚠️ 주의사항

### 1. **점진적 마이그레이션**
- 한 번에 모든 코드를 변경하지 말고 단계적으로 진행
- 각 단계마다 테스트 확인

### 2. **기존 코드 호환성**
- 기존 `common/data_processing.py`를 사용하는 코드들 확인
- 새로운 구조로 점진적 이전

### 3. **import 경로 업데이트**
- 모든 파일에서 import 경로 수정 필요
- IDE의 전역 검색/치환 활용 권장

---

## 🚀 다음 단계: ML 계층 통합

이 데이터 계층 리팩토링이 완료되면:
1. `core/ml/` 폴더 생성 및 ML 모델 통합
2. 각 도메인의 `model_trainer.py`, `models.py` 등을 `core/ml/`로 이동
3. 통합된 ML 파이프라인 구축

데이터 계층이 체계화되어 ML 계층 통합이 훨씬 수월해질 것입니다!
