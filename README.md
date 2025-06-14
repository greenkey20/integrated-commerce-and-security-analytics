# 🛍️ Mall Customer Segmentation Analysis

이 프로젝트는 K-means 클러스터링 알고리즘을 활용하여 쇼핑몰 고객 데이터를 분석하고, 고객을 의미 있는 그룹으로 세분화하는 Streamlit 웹 애플리케이션입니다. 분석된 고객 그룹별 특성을 바탕으로 맞춤형 마케팅 전략 수립을 지원합니다.

## ✨ 주요 기능

- **데이터 개요**: 업로드된 고객 데이터의 기본 정보 및 통계 요약 제공
- **탐색적 데이터 분석 (EDA)**: 고객 속성(성별, 연령, 소득, 지출 점수) 분포 및 상관관계 시각화
- **클러스터링 분석**:
    - 엘보우 방법(Elbow Method) 및 실루엣 점수(Silhouette Score)를 이용한 최적 클러스터 개수 제안
    - K-means 클러스터링 수행 및 결과 시각화 (3D, 2D 산점도, 상세 클러스터 맵)
    - 동적으로 생성된 클러스터별 특성(평균 소득, 지출, 연령 등) 및 라벨 제공
    - 클러스터 해석 가이드 제공
- **고객 예측**: 새로운 고객 정보를 입력받아 해당 고객이 속할 클러스터 예측
- **마케팅 전략**: 분석된 클러스터별 특성에 기반한 동적 마케팅 전략 제안

## 🚀 실행 방법

1.  **저장소 복제 (이미 하셨다면 생략):**
    ```bash
    git clone https://github.com/greenkey20/customer-segmentation-analysis.git
    cd customer-segmentation-analysis
    ```

2.  **가상 환경 생성 및 활성화 (권장):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # macOS/Linux
    # venv\Scripts\activate    # Windows
    ```

3.  **필요한 라이브러리 설치:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Streamlit 애플리케이션 실행:**
    ```bash
    streamlit run customer_segmentation_app.py
    ```
    실행 후 웹 브라우저에서 `http://localhost:8501` 주소로 접속합니다.

## 📝 한글 폰트 설정 (macOS)

`customer_segmentation_app.py` 내의 `setup_korean_font_for_streamlit` 함수는 macOS 환경에서 한글 폰트를 자동으로 찾아 설정하려고 시도합니다. 다른 OS 환경에서는 해당 함수를 수정하거나 시스템에 맞는 폰트 설정을 추가해야 할 수 있습니다.

## 📊 데이터 출처

이 애플리케이션은 기본적으로 다음 URL의 "Mall Customer Segmentation Dataset"을 사용합니다:
`https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Datasets/Mall_Customers.csv`

데이터 로드에 실패할 경우, 애플리케이션 내에서 샘플 데이터를 생성하여 데모를 진행합니다.

## 🛠️ 기술 스택

-   **언어**: Python
-   **웹 프레임워크**: Streamlit
-   **데이터 분석**: Pandas, NumPy
-   **머신러닝**: Scikit-learn (KMeans, StandardScaler, silhouette_score)
-   **시각화**: Matplotlib, Seaborn, Plotly Express, Plotly Graph Objects

## 🔮 향후 개선 사항

-   다양한 클러스터링 알고리즘 (예: DBSCAN, 계층적 클러스터링) 추가 옵션 제공
-   사용자 데이터 업로드 기능 강화
-   클러스터링 결과 저장 및 보고서 생성 기능
-   더욱 정교한 마케팅 전략 추천 로직 개발
-   다국어 지원

---

이 프로젝트가 고객 세분화 분석에 도움이 되기를 바랍니다!
