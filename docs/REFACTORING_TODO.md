# 🔧 프로젝트 리팩토링 TODO

## 📅 생성일: 2025-07-20
## 🎯 목표: core + src 구조 완전 정리

---

## ✅ 완료된 작업

- [x] 기존 최상위 폴더들 (`segmentation/`, `retail_analysis/`, `security/`) 제거
- [x] 중요 파일들을 `core/` 구조로 통합
- [x] 백업 파일들을 `temp_refactor/`로 이동
- [x] 로그 파일들을 `logs/`로 통합

---

## 🚨 긴급 정리 작업 (High Priority)

### 1. Security 모듈 중복 제거
- [x] **파일 정리**
  - [x] `core/security/data_loader.py` 삭제 (→ `docs/security_data_loader_backup.py`로 백업)
  - [x] `core/security/cicids_data_loader.py`만 유지
  - [x] 관련 import 경로 확인 완료 (참조하는 코드 없음)

- [x] **기능 통합** (완료)
  - [x] `anomaly_detector.py` + `attack_detector.py` → `detection_engine.py`로 통합 완료
  - [x] 중복된 클래스/함수들 정리 완료
  - [x] 통합된 UnifiedDetectionEngine, RealTimeSecurityMonitor 클래스 생성
  - [x] 기존 파일들 `docs/` 폴더로 백업 완료

### 2. Retail 백업 파일 정리
- [x] **불필요한 파일 삭제**
  - [x] `src/pages/retail/analysis_backup.py` 삭제 (→ `docs/retail_analysis_backup.py`로 백업)
  - [x] `src/pages/retail/analysis_end.py` 삭제 (→ `docs/retail_analysis_end.py`로 백업)
  - [x] `analysis.py`만 메인으로 유지

- [x] **기능 통합 재평가**
  - [x] `data_loading.py` + `data_cleaning.py` 통합 **불필요** 확인
  - [x] → 두 파일은 서로 다른 UI 페이지로서 유지 필요
  - [x] → Streamlit UI 코드는 기능별 분리가 올바름

### 3. temp_refactor 폴더 처리
- [x] **백업 확인 후 삭제**
  - [x] temp_refactor 폴더 없음 확인 (이미 삭제되었거나 최초부터 없었음)
  - [x] 모든 백업 파일들이 `docs/` 폴더로 이동 완료

---

## ⚠️ 중요 정리 작업 (Medium Priority)

### 4. 데이터 처리 계층 통합
- [x] **새로운 데이터 계층 생성**
  - [x] `shared/` 폴더 생성 (계층 중립적 위치)
  - [x] `shared/data_processing.py` 생성 (통합 데이터 처리 클래스)
  - [x] `shared/__init__.py` 생성

- [x] **분산된 로직 통합**
  - [x] 공통 데이터 검증, 정제, 특성공학 클래스 통합
  - [x] DataValidator, DataCleaner, FeatureEngineer, DataSplitter 클래스 생성
  - [x] 통합 DataProcessor 클래스로 파이프라인 구성
  - [x] `core/retail/data_processor.py` → `docs/retail_data_processor_backup.py`로 백업
  - [x] `core/segmentation/data_processing.py` → `docs/segmentation_data_processing_backup.py`로 백업

### 5. 공통 설정 통합
- [x] **공통 계층 확장** (기존 config/, utils/ 활용)
  - [x] `config/settings.py`에 SecurityConfig, LoggingConfig 추가
  - [x] `config/logging.py` 생성 (로깅 설정 통합)
  - [x] `utils/exceptions.py` 생성 (커스텀 예외들)

- [x] **설정 중앙화**
  - [x] 하드코딩된 설정값들 SecurityConfig로 이동 (완료)
  - [x] 로깅 설정 LoggingConfig로 통합 (완료)
  - [x] 커스텀 예외들 utils/exceptions.py로 정리 (완료)

---

## 📈 개선 작업 (Low Priority)

### 6. ML 계층 통합
- [ ] **ML 전용 계층 생성**
  ```
  mkdir -p core/ml
  touch core/ml/__init__.py
  touch core/ml/base_model.py
  touch core/ml/trainers.py
  touch core/ml/evaluators.py
  touch core/ml/persistence.py
  ```

- [ ] **모델 코드 통합**
  - [ ] `core/segmentation/models.py` → `core/ml/`
  - [ ] `core/retail/model_trainer.py` → `core/ml/trainers.py`
  - [ ] `core/security/model_builder.py` → `core/ml/trainers.py`

### 7. 테스트 코드 정리
- [ ] **중복 테스트 제거**
- [ ] **새 구조에 맞는 테스트 작성**
- [ ] **CI/CD 파이프라인 업데이트**

---

## 🔍 주요 중복 파일 목록 (완료)

### Security 모듈
```
✅ core/security/data_loader.py (삭제 완료 - 백업됨)
✅ core/security/cicids_data_loader.py (유지)
✅ core/security/anomaly_detector.py (통합 완료 - detection_engine.py)
✅ core/security/attack_detector.py (통합 완룼 - detection_engine.py)
✨ core/security/detection_engine.py (새로 생성된 통합 모듈)
```

### Retail 모듈
```
✅ src/pages/retail/analysis_backup.py (삭제 완료 - 백업됨)
✅ src/pages/retail/analysis_end.py (삭제 완룼 - 백업됨)
✅ src/pages/retail/analysis.py (유지)
✅ src/pages/retail/data_loading.py (유지 - UI 페이지)
✅ src/pages/retail/data_cleaning.py (유지 - UI 페이지)
```

### 데이터 처리
```
✅ core/retail/data_processor.py (백업 완료)
✅ core/segmentation/data_processing.py (백업 완료)
✅ src/pages/retail/data_cleaning.py (유지 - UI 페이지)
✅ src/pages/retail/feature_engineering.py (유지)
✨ shared/data_processing.py (새로 생성된 통합 모듈)
```

---

## 📋 체크리스트 사용법

1. **작업 시작 전**: 해당 브랜치 생성
   ```bash
   git checkout -b refactor/cleanup-duplicates
   ```

2. **각 작업 완료 후**: 체크박스 업데이트
   ```markdown
   - [x] 완료된 작업
   ```

3. **작업 완료 후**: 커밋 및 푸시
   ```bash
   git add .
   git commit -m "refactor: remove duplicated security modules"
   git push origin refactor/cleanup-duplicates
   ```

---

## 🚀 예상 효과

### 코드 품질 개선
- **중복 코드 제거**: 약 30-40% 코드량 감소 예상
- **의존성 정리**: 순환 참조 문제 해결
- **테스트 용이성**: 계층별 독립적 테스트 가능

### 개발 생산성 향상
- **명확한 책임 분리**: 어디에 무엇을 구현해야 할지 명확
- **재사용성 증대**: core 모듈을 다양한 UI에서 활용 가능
- **유지보수성**: 기능별 모듈화로 수정 범위 최소화

### 프로젝트 확장성
- **새 도메인 추가 용이**: marketing, inventory 등 쉽게 확장
- **다양한 UI 지원**: React, Flutter 등 다른 프론트엔드 연동 가능
- **마이크로서비스 전환**: 필요시 각 도메인을 독립 서비스로 분리 가능
