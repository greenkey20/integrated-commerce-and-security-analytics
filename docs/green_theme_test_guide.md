# 🌿 Green Theme UI 테스트 가이드

## 🚀 실행 방법
```bash
cd C:\keydev\integrated-commerce-and-security-analytics
streamlit run main_app.py
```

## ✅ 테스트할 기능들

### 1. 🌙 Dark Mode 토글
- 사이드바 맨 위의 "🌙 Dark Mode" 토글 클릭
- 실시간으로 전체 테마가 바뀌는지 확인
- 메트릭 카드, 차트, 배경색 모두 변경되는지 확인

### 2. 📊 Customer Analytics 페이지 
**네비게이션:** Business Intelligence → Customer Analytics → 1️⃣ 데이터 로딩 및 개요

**확인할 요소들:**
- ✨ Green Theme 섹션 헤더 (그라데이션 배경 + 아이콘)
- 📊 4개 메트릭 카드 (green/teal/lime/green 색상별)
- 🎨 히스토그램 (Green Spectrum 색상)
- 🥧 파이 차트 (초록 색상 팔레트)
- 📊 막대 차트 (초록 계열)
- 🔥 상관관계 히트맵 (Greens 색상 스케일)

### 3. 🎨 색상 일관성 테스트
**Light Mode:**
- 배경: 밝은 초록 그라데이션
- 메트릭 카드: 초록/청록/라임 색상
- 차트: Green Spectrum (#22C55E, #10B981, #14B8A6...)

**Dark Mode:**
- 배경: 어두운 배경 
- 메트릭 카드: 에메랄드 계열
- 차트: 밝은 Green 색상들 (#34D399, #A7F3D0...)

## 🐛 예상 이슈 & 해결

### ❌ UI 컴포넌트 로딩 실패시
```
⚠️ UI 컴포넌트를 로드할 수 없습니다. 기본 UI를 사용합니다.
```
→ utils/ui_components.py 파일 확인 필요

### ❌ 색상이 적용되지 않는 경우
→ 브라우저 캐시 삭제 (Ctrl+F5)

### ❌ Dark Mode 토글이 작동하지 않는 경우  
→ 페이지 새로고침 후 재시도

## 📈 성과 지표

### ✅ 성공 기준
- [ ] Dark/Light 모드 완벽 전환
- [ ] 모든 차트가 Green Spectrum 색상 적용
- [ ] 메트릭 카드 4가지 색상 적용
- [ ] 섹션 헤더 Green Theme 적용
- [ ] 전체 UI 일관성 유지

### 🎯 다음 단계 개선 포인트
1. 다른 페이지들(클러스터링, PCA 등)에도 동일하게 적용
2. Retail Analytics 페이지 Green Theme 적용  
3. Security Analytics 페이지 Green Theme 적용
4. 애니메이션 효과 추가 (hover, transition)
5. 반응형 디자인 최적화

## 💡 팁
- **Dark Mode**는 눈의 피로 감소와 배터리 절약 효과
- **Green Theme**은 성장, 안정성, 신뢰성의 브랜딩 효과
- **일관된 색상 팔레트**로 전문적인 비즈니스 인텔리전스 플랫폼 느낌

---

**작업 완료일:** 2025-07-27  
**버전:** v3.0 - Green Spectrum Edition  
**담당:** Claude AI Assistant
