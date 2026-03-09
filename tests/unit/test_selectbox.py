import streamlit as st

# 세션 상태 초기화
if 'current_focus_tab' not in st.session_state:
    st.session_state.current_focus_tab = 'retail'
if 'current_focus_header' not in st.session_state:
    st.session_state.current_focus_header = 'retail'

st.title("🔀 UX 옵션 비교")

# ==================== Option 1: 탭 스타일 버튼 ====================
st.markdown("## Option 1: 탭 스타일 버튼")
st.markdown("*상단에 탭처럼 배치된 버튼들*")

with st.sidebar:
    st.markdown("### 📋 Navigation (Tab Style)")

    # 탭 스타일 버튼들 (3개 열로)
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("💰\nRetail", key="tab_retail",
                     type="primary" if st.session_state.current_focus_tab == 'retail' else "secondary"):
            st.session_state.current_focus_tab = 'retail'

    with col2:
        if st.button("👥\nCustomer", key="tab_customer",
                     type="primary" if st.session_state.current_focus_tab == 'customer' else "secondary"):
            st.session_state.current_focus_tab = 'customer'

    with col3:
        if st.button("🔒\nSecurity", key="tab_security",
                     type="primary" if st.session_state.current_focus_tab == 'security' else "secondary"):
            st.session_state.current_focus_tab = 'security'

    st.markdown("---")

    # 현재 포커스된 섹션의 selectbox만 표시
    if st.session_state.current_focus_tab == 'retail':
        st.markdown("**💰 Retail Analytics**")
        retail_step = st.selectbox("단계 선택:", [
            "1️⃣ 데이터 로딩", "2️⃣ 데이터 정제", "3️⃣ 특성공학", "📊 전체 분석"
        ], key="tab_retail_step")

    elif st.session_state.current_focus_tab == 'customer':
        st.markdown("**👥 Customer Segmentation**")
        customer_step = st.selectbox("단계 선택:", [
            "1️⃣ 데이터 개요", "2️⃣ 탐색적 분석", "3️⃣ 클러스터링", "7️⃣ 마케팅 전략"
        ], key="tab_customer_step")

    elif st.session_state.current_focus_tab == 'security':
        st.markdown("**🔒 Security Analytics**")
        security_step = st.selectbox("단계 선택:", [
            "1️⃣ 데이터 로딩", "2️⃣ 탐색적 분석", "4️⃣ 딥러닝", "7️⃣ 종합 평가"
        ], key="tab_security_step")

# 메인 화면 표시
st.info(f"**현재 포커스 (Tab Style)**: {st.session_state.current_focus_tab}")

st.markdown("---")

# ==================== Option 2: 클릭 가능한 헤더 ====================
st.markdown("## Option 2: 클릭 가능한 헤더")
st.markdown("*헤더 자체가 버튼이 되어 클릭으로 포커스 변경*")

with st.sidebar:
    st.markdown("### 📋 Navigation (Clickable Headers)")

    # A. Business Intelligence 섹션
    st.markdown("**📊 A. Business Intelligence**")

    # 1. Retail - 헤더를 버튼으로
    retail_active = st.session_state.current_focus_header == 'retail'
    if st.button(f"💰 **1. Retail Prediction** {'✅' if retail_active else ''}",
                 key="header_retail", use_container_width=True):
        st.session_state.current_focus_header = 'retail'

    if retail_active:
        retail_step = st.selectbox("단계 선택:", [
            "1️⃣ 데이터 로딩", "2️⃣ 데이터 정제", "3️⃣ 특성공학", "📊 전체 분석"
        ], key="header_retail_step")

    # 2. Customer - 헤더를 버튼으로
    customer_active = st.session_state.current_focus_header == 'customer'
    if st.button(f"👥 **2. Customer Segmentation** {'✅' if customer_active else ''}",
                 key="header_customer", use_container_width=True):
        st.session_state.current_focus_header = 'customer'

    if customer_active:
        customer_step = st.selectbox("단계 선택:", [
            "1️⃣ 데이터 개요", "2️⃣ 탐색적 분석", "3️⃣ 클러스터링", "7️⃣ 마케팅 전략"
        ], key="header_customer_step")

    # B. Security Analytics 섹션
    st.markdown("**🛡️ B. Security Analytics**")

    security_active = st.session_state.current_focus_header == 'security'
    if st.button(f"🔒 **1. 네트워크 보안 분석** {'✅' if security_active else ''}",
                 key="header_security", use_container_width=True):
        st.session_state.current_focus_header = 'security'

    if security_active:
        security_step = st.selectbox("단계 선택:", [
            "1️⃣ 데이터 로딩", "2️⃣ 탐색적 분석", "4️⃣ 딥러닝", "7️⃣ 종합 평가"
        ], key="header_security_step")

# 메인 화면 표시
st.info(f"**현재 포커스 (Clickable Header)**: {st.session_state.current_focus_header}")

# ==================== 비교 분석 ====================
st.markdown("---")
st.markdown("## 🔍 UX 비교")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 💡 Option 1: 탭 스타일")
    st.markdown("""
    **장점:**
    - 직관적이고 친숙한 UI
    - 현재 활성 탭이 명확함
    - 빠른 전환 가능

    **단점:**
    - 사이드바 공간 많이 차지
    - 3개 이상 탭 시 좁아짐
    """)

with col2:
    st.markdown("### 💡 Option 2: 클릭 가능한 헤더")
    st.markdown("""
    **장점:**
    - 기존 구조 거의 그대로 유지
    - 계층 구조 명확함
    - 확장성 좋음

    **단점:**
    - 버튼인지 텍스트인지 애매할 수 있음
    - 시각적 피드백 필요
    """)

st.markdown("---")
st.markdown("**어떤 방식이 더 직관적이고 사용하기 편한가요?**")