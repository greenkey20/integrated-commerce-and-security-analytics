"""
🌿 Green Spectrum UI 컴포넌트
통합 커머스 & 보안 분석 플랫폼용 재사용 가능한 UI 컴포넌트들

작성자: AI Assistant (Claude)
버전: v3.0 - Green Theme Edition
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Optional, Dict, Any

# 🎨 Green Spectrum 색상 팔레트
GREEN_PALETTE = [
    '#22C55E',  # Forest Green (메인)
    '#10B981',  # Emerald
    '#14B8A6',  # Teal  
    '#84CC16',  # Lime
    '#059669',  # Emerald Dark
    '#0D9488',  # Teal Dark
    '#65A30D',  # Lime Dark
    '#16A34A',  # Green
    '#15803D',  # Green Dark
    '#166534',  # Green Darker
]

# 상태별 색상
STATUS_COLORS = {
    'success': '#22C55E',
    'warning': '#F59E0B', 
    'error': '#EF4444',
    'info': '#64748B',
    'neutral': '#94A3B8'
}

# Dark Mode용 색상
DARK_GREEN_PALETTE = [
    '#34D399',  # Emerald 300
    '#A7F3D0',  # Emerald 200  
    '#6EE7B7',  # Emerald 300
    '#10B981',  # Emerald 500
    '#059669',  # Emerald 600
    '#047857',  # Emerald 700
    '#065F46',  # Emerald 800
    '#064E3B',  # Emerald 900
]

def get_green_colors(dark_mode: bool = False) -> List[str]:
    """
    Plotly 차트용 Green Spectrum 팔레트 반환
    
    Args:
        dark_mode: Dark Mode인지 여부
        
    Returns:
        색상 리스트
    """
    return DARK_GREEN_PALETTE if dark_mode else GREEN_PALETTE

def create_metric_card(
    title: str, 
    value: str, 
    delta: Optional[str] = None, 
    color: str = "green",
    dark_mode: bool = False
) -> None:
    """
    초록색 테마 메트릭 카드 생성
    
    Args:
        title: 메트릭 제목
        value: 메트릭 값
        delta: 변화량 (선택)
        color: 색상 테마 (green, teal, lime)
        dark_mode: Dark Mode 여부
    """
    
    if dark_mode:
        # Dark Mode 색상
        colors = {
            "green": "#34D399",
            "teal": "#14B8A6", 
            "lime": "#84CC16"
        }
        bg_gradient = "linear-gradient(135deg, #374151, #1F2937)"
        text_color = "#A7F3D0"
        value_color = colors[color]
        border_color = colors[color]
    else:
        # Light Mode 색상
        colors = {
            "green": "#22C55E",
            "teal": "#14B8A6", 
            "lime": "#84CC16"
        }
        bg_gradient = "linear-gradient(135deg, #F0FDF4, #F0FDFA)"
        text_color = "#166534"
        value_color = colors[color]
        border_color = colors[color]
    
    st.markdown(f"""
    <div style="
        background: {bg_gradient};
        border-left: 4px solid {border_color};
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        box-shadow: 0 4px 6px -1px rgba(34, 197, 94, 0.15);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    " onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 8px 25px -5px rgba(34, 197, 94, 0.25)'" 
       onmouseout="this.style.transform='translateY(0px)'; this.style.boxShadow='0 4px 6px -1px rgba(34, 197, 94, 0.15)'">
        <h3 style="color: {text_color}; margin: 0; font-size: 1rem; font-weight: 500;">{title}</h3>
        <h1 style="color: {value_color}; margin: 0.5rem 0; font-size: 2.5rem; font-weight: 700;">{value}</h1>
        {f'<p style="color: {value_color}; margin: 0; font-size: 0.9rem; font-weight: 500;">{delta}</p>' if delta else ''}
    </div>
    """, unsafe_allow_html=True)

def create_section_header(
    title: str, 
    subtitle: Optional[str] = None, 
    icon: str = "🌿",
    dark_mode: bool = False
) -> None:
    """
    섹션 헤더 생성
    
    Args:
        title: 섹션 제목
        subtitle: 부제목 (선택)
        icon: 아이콘 이모지
        dark_mode: Dark Mode 여부
    """
    
    if dark_mode:
        bg_gradient = "linear-gradient(135deg, #374151, #1F2937)"
        border_color = "#34D399"
        title_color = "#A7F3D0"
        subtitle_color = "#D1D5DB"
    else:
        bg_gradient = "linear-gradient(135deg, #F0FDF4, #F0FDFA)"
        border_color = "#22C55E"
        title_color = "#166534"
        subtitle_color = "#059669"
    
    st.markdown(f"""
    <div style="
        background: {bg_gradient};
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        border-left: 6px solid {border_color};
        box-shadow: 0 4px 6px -1px rgba(34, 197, 94, 0.1);
    ">
        <h1 style="color: {title_color}; margin: 0; display: flex; align-items: center; font-weight: 700;">
            <span style="margin-right: 1rem; font-size: 2rem;">{icon}</span>
            {title}
        </h1>
        {f'<p style="color: {subtitle_color}; margin: 1rem 0 0 3.5rem; font-size: 1.2rem; font-weight: 400;">{subtitle}</p>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)

def create_info_box(
    content: str,
    box_type: str = "info",
    icon: Optional[str] = None,
    dark_mode: bool = False
) -> None:
    """
    정보 박스 생성 (알림, 경고, 오류 등)
    
    Args:
        content: 내용
        box_type: 박스 타입 (info, success, warning, error)
        icon: 아이콘 (선택)
        dark_mode: Dark Mode 여부
    """
    
    if dark_mode:
        type_colors = {
            'info': {'bg': '#0C4A6E', 'border': '#3B82F6', 'text': '#DBEAFE'},
            'success': {'bg': '#064E3B', 'border': '#22C55E', 'text': '#A7F3D0'},
            'warning': {'bg': '#451A03', 'border': '#F59E0B', 'text': '#FDE68A'},
            'error': {'bg': '#450A0A', 'border': '#EF4444', 'text': '#FECACA'}
        }
    else:
        type_colors = {
            'info': {'bg': '#F0F9FF', 'border': '#3B82F6', 'text': '#0C4A6E'},
            'success': {'bg': '#F0FDF4', 'border': '#22C55E', 'text': '#166534'},
            'warning': {'bg': '#FFFBEB', 'border': '#F59E0B', 'text': '#92400E'},
            'error': {'bg': '#FEF2F2', 'border': '#EF4444', 'text': '#991B1B'}
        }
    
    default_icons = {
        'info': '💡',
        'success': '✅', 
        'warning': '⚠️',
        'error': '🚨'
    }
    
    colors = type_colors.get(box_type, type_colors['info'])
    display_icon = icon or default_icons.get(box_type, '📝')
    
    st.markdown(f"""
    <div style="
        background: {colors['bg']};
        border: 2px solid {colors['border']};
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: {colors['text']};
        display: flex;
        align-items: flex-start;
        gap: 1rem;
    ">
        <span style="font-size: 1.5rem; flex-shrink: 0;">{display_icon}</span>
        <div style="font-size: 1rem; line-height: 1.6;">{content}</div>
    </div>
    """, unsafe_allow_html=True)

def create_progress_card(
    title: str,
    current: int,
    total: int,
    description: Optional[str] = None,
    color: str = "green",
    dark_mode: bool = False
) -> None:
    """
    진행률 카드 생성
    
    Args:
        title: 카드 제목
        current: 현재 값
        total: 전체 값  
        description: 설명 (선택)
        color: 색상 테마
        dark_mode: Dark Mode 여부
    """
    
    percentage = min(100, max(0, (current / total) * 100)) if total > 0 else 0
    
    if dark_mode:
        bg_color = "#374151"
        text_color = "#D1D5DB"
        progress_bg = "#1F2937"
        progress_color = "#34D399"
    else:
        bg_color = "#FFFFFF"
        text_color = "#374151"
        progress_bg = "#F3F4F6"
        progress_color = "#22C55E"
    
    st.markdown(f"""
    <div style="
        background: {bg_color};
        border: 1px solid rgba(34, 197, 94, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    ">
        <h3 style="color: {text_color}; margin: 0 0 0.5rem 0; font-weight: 600;">{title}</h3>
        <div style="
            background: {progress_bg};
            border-radius: 8px;
            height: 12px;
            overflow: hidden;
            margin: 1rem 0;
        ">
            <div style="
                background: {progress_color};
                height: 100%;
                width: {percentage}%;
                transition: width 0.3s ease;
            "></div>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="color: {text_color}; font-size: 0.9rem;">{current}/{total}</span>
            <span style="color: {progress_color}; font-weight: 600;">{percentage:.1f}%</span>
        </div>
        {f'<p style="color: {text_color}; margin: 0.5rem 0 0 0; font-size: 0.85rem; opacity: 0.8;">{description}</p>' if description else ''}
    </div>
    """, unsafe_allow_html=True)

def style_plotly_chart(
    fig: go.Figure,
    dark_mode: bool = False,
    title: Optional[str] = None
) -> go.Figure:
    """
    Plotly 차트에 Green Theme 스타일 적용
    
    Args:
        fig: Plotly 차트 객체
        dark_mode: Dark Mode 여부
        title: 차트 제목 (선택)
        
    Returns:
        스타일 적용된 차트 객체
    """
    
    if dark_mode:
        # Dark Mode 스타일
        fig.update_layout(
            plot_bgcolor='rgba(31, 41, 55, 0.9)',
            paper_bgcolor='rgba(31, 41, 55, 0.9)',
            font_color='#D1D5DB',
            title_font_color='#A7F3D0',
            legend=dict(
                bgcolor='rgba(55, 65, 81, 0.8)',
                bordercolor='#34D399',
                borderwidth=1
            ),
            xaxis=dict(
                gridcolor='rgba(55, 65, 81, 0.5)',
                linecolor='#34D399'
            ),
            yaxis=dict(
                gridcolor='rgba(55, 65, 81, 0.5)',
                linecolor='#34D399'
            )
        )
    else:
        # Light Mode 스타일
        fig.update_layout(
            plot_bgcolor='rgba(255, 255, 255, 0.9)',
            paper_bgcolor='rgba(255, 255, 255, 0.9)',
            font_color='#374151',
            title_font_color='#166534',
            legend=dict(
                bgcolor='rgba(240, 253, 244, 0.8)',
                bordercolor='#22C55E',
                borderwidth=1
            ),
            xaxis=dict(
                gridcolor='rgba(229, 231, 235, 0.5)',
                linecolor='#22C55E'
            ),
            yaxis=dict(
                gridcolor='rgba(229, 231, 235, 0.5)',
                linecolor='#22C55E'
            )
        )
    
    # 공통 스타일
    fig.update_layout(
        title=dict(
            text=title,
            font_size=20,
            font_weight='bold',
            x=0.5
        ) if title else {},
        margin=dict(t=60, l=50, r=50, b=50),
        height=400,
        showlegend=True
    )
    
    return fig

def create_stats_grid(
    stats: List[Dict[str, Any]],
    columns: int = 3,
    dark_mode: bool = False
) -> None:
    """
    통계 그리드 생성
    
    Args:
        stats: 통계 데이터 리스트 [{'title': '제목', 'value': '값', 'delta': '변화', 'color': '색상'}]
        columns: 열 개수
        dark_mode: Dark Mode 여부
    """
    
    # 그리드 레이아웃 생성
    cols = st.columns(columns)
    
    for i, stat in enumerate(stats):
        with cols[i % columns]:
            create_metric_card(
                title=stat.get('title', '제목'),
                value=stat.get('value', '0'),
                delta=stat.get('delta'),
                color=stat.get('color', 'green'),
                dark_mode=dark_mode
            )

def create_action_button(
    label: str,
    icon: str = "🚀",
    button_type: str = "primary",
    dark_mode: bool = False,
    key: Optional[str] = None
) -> bool:
    """
    커스텀 액션 버튼 생성
    
    Args:
        label: 버튼 라벨
        icon: 아이콘 이모지
        button_type: 버튼 타입 (primary, secondary, danger)
        dark_mode: Dark Mode 여부
        key: Streamlit 키
        
    Returns:
        버튼 클릭 여부
    """
    
    if dark_mode:
        type_colors = {
            'primary': {'bg': '#22C55E', 'hover': '#16A34A', 'text': '#FFFFFF'},
            'secondary': {'bg': '#374151', 'hover': '#4B5563', 'text': '#D1D5DB'},
            'danger': {'bg': '#EF4444', 'hover': '#DC2626', 'text': '#FFFFFF'}
        }
    else:
        type_colors = {
            'primary': {'bg': '#22C55E', 'hover': '#16A34A', 'text': '#FFFFFF'},
            'secondary': {'bg': '#F9FAFB', 'hover': '#F3F4F6', 'text': '#374151'},
            'danger': {'bg': '#EF4444', 'hover': '#DC2626', 'text': '#FFFFFF'}
        }
    
    colors = type_colors.get(button_type, type_colors['primary'])
    
    # CSS 스타일 추가
    st.markdown(f"""
    <style>
    .custom-button-{key or 'default'} {{
        background: {colors['bg']};
        color: {colors['text']};
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        text-decoration: none;
        width: 100%;
        justify-content: center;
    }}
    .custom-button-{key or 'default'}:hover {{
        background: {colors['hover']};
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }}
    </style>
    """, unsafe_allow_html=True)
    
    return st.button(f"{icon} {label}", key=key)

# 사용 예시 함수들
def demo_components(dark_mode: bool = False):
    """UI 컴포넌트 데모"""
    
    create_section_header(
        "🌿 Green Theme UI 컴포넌트 데모", 
        "통합 커머스 & 보안 분석 플랫폼용 디자인 시스템",
        dark_mode=dark_mode
    )
    
    # 메트릭 카드 데모
    st.subheader("📊 메트릭 카드")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        create_metric_card("총 고객수", "1,234", "+12% 증가", "green", dark_mode)
    with col2:
        create_metric_card("월 매출", "₩5.6M", "+8.2% 증가", "teal", dark_mode)  
    with col3:
        create_metric_card("전환율", "4.8%", "+0.3%p 증가", "lime", dark_mode)
    
    # 정보 박스 데모
    st.subheader("💡 정보 박스")
    create_info_box("데이터 분석이 성공적으로 완료되었습니다!", "success", dark_mode=dark_mode)
    create_info_box("새로운 보안 위협이 감지되었습니다. 확인이 필요합니다.", "warning", dark_mode=dark_mode)
    
    # 진행률 카드 데모  
    st.subheader("📈 진행률 카드")
    col1, col2 = st.columns(2)
    
    with col1:
        create_progress_card("모델 훈련", 750, 1000, "딥러닝 모델 학습 중...", dark_mode=dark_mode)
    with col2:
        create_progress_card("데이터 처리", 8, 10, "전처리 단계", dark_mode=dark_mode)

if __name__ == "__main__":
    st.set_page_config(page_title="UI Components Demo", page_icon="🌿", layout="wide")
    
    # Dark Mode 토글
    dark_mode = st.sidebar.toggle("🌙 Dark Mode", False)
    
    # 데모 실행
    demo_components(dark_mode)
