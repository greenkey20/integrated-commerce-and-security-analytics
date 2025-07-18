"""
Online Retail 시각화 전담 모듈

이 모듈은 Online Retail 데이터 분석 결과의 
시각화를 전담하는 클래스들을 제공합니다.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings("ignore")


class RetailVisualizer:
    """Online Retail 데이터 시각화 전담 클래스"""
    
    @staticmethod
    def create_data_quality_dashboard(quality_report: Dict) -> go.Figure:
        """데이터 품질 리포트 시각화"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('결측값 분포', '데이터 타입 분포', '이상치 현황', '메모리 사용량'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "indicator"}]]
        )
        
        # 결측값 분포
        missing_data = quality_report['missing_values']
        cols = list(missing_data.keys())
        missing_pcts = [missing_data[col]['percentage'] for col in cols]
        
        fig.add_trace(
            go.Bar(x=cols, y=missing_pcts, name="결측값 %", marker_color='lightcoral'),
            row=1, col=1
        )
        
        # 데이터 타입 분포
        type_counts = {}
        for col, info in quality_report['data_types'].items():
            dtype = info['current_type']
            type_counts[dtype] = type_counts.get(dtype, 0) + 1
        
        fig.add_trace(
            go.Pie(labels=list(type_counts.keys()), values=list(type_counts.values()), name="데이터 타입"),
            row=1, col=2
        )
        
        # 이상치 현황
        outlier_data = quality_report['outliers']
        if outlier_data:
            outlier_cols = list(outlier_data.keys())
            outlier_pcts = [outlier_data[col]['outlier_percentage'] for col in outlier_cols]
            
            fig.add_trace(
                go.Bar(x=outlier_cols, y=outlier_pcts, name="이상치 %", marker_color='orange'),
                row=2, col=1
            )
        
        # 메모리 사용량
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=quality_report['memory_usage_mb'],
                title={'text': "메모리 (MB)"},
                gauge={'axis': {'range': [None, quality_report['memory_usage_mb'] * 1.5]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, quality_report['memory_usage_mb'] * 0.5], 'color': "lightgray"},
                                {'range': [quality_report['memory_usage_mb'] * 0.5, quality_report['memory_usage_mb']], 'color': "gray"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75, 'value': quality_report['memory_usage_mb'] * 1.2}}
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="📊 데이터 품질 종합 대시보드",
            showlegend=False,
            height=600
        )
        
        return fig
    
    @staticmethod  
    def create_customer_distribution_plots(customer_features: pd.DataFrame) -> go.Figure:
        """고객 특성 분포 시각화"""
        
        key_metrics = ['total_amount', 'frequency', 'recency_days', 'unique_products']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f'{metric} 분포' for metric in key_metrics]
        )
        
        for i, metric in enumerate(key_metrics):
            row = i // 2 + 1
            col = i % 2 + 1
            
            if metric in customer_features.columns:
                fig.add_trace(
                    go.Histogram(x=customer_features[metric], name=metric, nbinsx=30),
                    row=row, col=col
                )
        
        fig.update_layout(
            title_text="👥 고객 특성 분포 분석",
            showlegend=False,
            height=600
        )
        
        return fig
    
    @staticmethod
    def create_rfm_analysis_plots(customer_features: pd.DataFrame) -> go.Figure:
        """RFM 분석 시각화"""
        
        if not all(col in customer_features.columns for col in ['recency_days', 'frequency', 'monetary']):
            # 기본 RFM 컬럼이 없는 경우 대체 컬럼 사용
            rfm_cols = []
            for col in ['recency_days', 'frequency', 'monetary']:
                if col in customer_features.columns:
                    rfm_cols.append(col)
                elif col == 'frequency' and 'unique_invoices' in customer_features.columns:
                    rfm_cols.append('unique_invoices')
                elif col == 'monetary' and 'total_amount' in customer_features.columns:
                    rfm_cols.append('total_amount')
            
            if len(rfm_cols) < 2:
                return go.Figure().add_annotation(text="RFM 분석에 필요한 데이터가 부족합니다.")
        else:
            rfm_cols = ['recency_days', 'frequency', 'monetary']
        
        fig = make_subplots(
            rows=1, cols=len(rfm_cols),
            subplot_titles=[f'{col.replace("_", " ").title()}' for col in rfm_cols]
        )
        
        colors = ['lightcoral', 'lightblue', 'lightgreen']
        
        for i, col in enumerate(rfm_cols):
            fig.add_trace(
                go.Histogram(x=customer_features[col], name=col, marker_color=colors[i % len(colors)]),
                row=1, col=i+1
            )
        
        fig.update_layout(
            title_text="📊 RFM 분석 분포",
            showlegend=False,
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_customer_segment_analysis(customer_features: pd.DataFrame) -> go.Figure:
        """고객 세그먼트 분석 시각화"""
        
        if 'customer_segment' not in customer_features.columns:
            return go.Figure().add_annotation(text="고객 세그먼트 데이터가 없습니다.")
        
        segment_counts = customer_features['customer_segment'].value_counts()
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['고객 세그먼트 분포', '세그먼트별 평균 구매 금액'],
            specs=[[{"type": "pie"}, {"type": "bar"}]]
        )
        
        # 세그먼트 분포 파이차트
        fig.add_trace(
            go.Pie(labels=segment_counts.index, values=segment_counts.values, name="세그먼트"),
            row=1, col=1
        )
        
        # 세그먼트별 평균 구매 금액
        if 'total_amount' in customer_features.columns:
            segment_avg = customer_features.groupby('customer_segment')['total_amount'].mean().sort_values(ascending=False)
            
            fig.add_trace(
                go.Bar(x=segment_avg.index, y=segment_avg.values, name="평균 구매 금액", marker_color='lightgreen'),
                row=1, col=2
            )
        
        fig.update_layout(
            title_text="🎯 고객 세그먼트 분석",
            showlegend=False,
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_model_performance_plots(evaluation_results: Dict) -> go.Figure:
        """모델 성능 시각화"""
        
        if 'y_test' not in evaluation_results or 'y_test_pred' not in evaluation_results:
            return go.Figure().add_annotation(text="모델 평가 결과가 없습니다.")
        
        y_test = evaluation_results['y_test']
        y_test_pred = evaluation_results['y_test_pred']
        residuals = evaluation_results['residuals']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['예측값 vs 실제값', '잔차 분포', '잔차 vs 예측값', '성능 지표'],
            specs=[[{"type": "scatter"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "indicator"}]]
        )
        
        # 예측값 vs 실제값 산점도
        fig.add_trace(
            go.Scatter(x=y_test, y=y_test_pred, mode='markers', name='예측값', marker=dict(color='blue', opacity=0.6)),
            row=1, col=1
        )
        
        # 완벽한 예측선 추가
        min_val = min(y_test.min(), y_test_pred.min())
        max_val = max(y_test.max(), y_test_pred.max())
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='완벽한 예측', 
                      line=dict(color='red', dash='dash')),
            row=1, col=1
        )
        
        # 잔차 히스토그램
        fig.add_trace(
            go.Histogram(x=residuals, name='잔차', marker_color='lightcoral'),
            row=1, col=2
        )
        
        # 잔차 vs 예측값
        fig.add_trace(
            go.Scatter(x=y_test_pred, y=residuals, mode='markers', name='잔차', marker=dict(color='green', opacity=0.6)),
            row=2, col=1
        )
        
        # 기준선 추가
        fig.add_trace(
            go.Scatter(x=[y_test_pred.min(), y_test_pred.max()], y=[0, 0], mode='lines', name='기준선',
                      line=dict(color='red', dash='dash')),
            row=2, col=1
        )
        
        # R² 점수 게이지
        r2_score = evaluation_results.get('r2_test', 0)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=r2_score,
                title={'text': "R² Score"},
                gauge={'axis': {'range': [None, 1]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, 0.5], 'color': "lightgray"},
                                {'range': [0.5, 0.8], 'color': "gray"},
                                {'range': [0.8, 1], 'color': "lightgreen"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75, 'value': 0.9}}
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="📈 모델 성능 종합 분석",
            showlegend=False,
            height=800
        )
        
        return fig
    
    @staticmethod
    def create_feature_importance_plot(feature_importance: Dict) -> go.Figure:
        """특성 중요도 시각화"""
        
        if 'top_10_features' not in feature_importance:
            return go.Figure().add_annotation(text="특성 중요도 데이터가 없습니다.")
        
        top_features = feature_importance['top_10_features']
        
        features = [f['feature'] for f in top_features]
        coefficients = [f['coefficient'] for f in top_features]
        abs_coefficients = [f['abs_coefficient'] for f in top_features]
        
        # 색상 설정 (양수는 파란색, 음수는 빨간색)
        colors = ['blue' if coef > 0 else 'red' for coef in coefficients]
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                x=abs_coefficients,
                y=features,
                orientation='h',
                marker_color=colors,
                text=[f'{coef:.3f}' for coef in coefficients],
                textposition='auto'
            )
        )
        
        fig.update_layout(
            title_text="📊 상위 10개 특성 중요도",
            xaxis_title="계수 절댓값",
            yaxis_title="특성명",
            yaxis={'categoryorder': 'total ascending'},
            height=600
        )
        
        return fig
    
    @staticmethod
    def create_prediction_confidence_plot(customer_features: pd.DataFrame, predictions: pd.Series) -> go.Figure:
        """예측 신뢰도 시각화"""
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['예측 금액 분포', '예측 신뢰도 구간'],
            specs=[[{"type": "histogram"}, {"type": "scatter"}]]
        )
        
        # 예측 금액 분포
        fig.add_trace(
            go.Histogram(x=predictions, name='예측 금액', marker_color='lightblue'),
            row=1, col=1
        )
        
        # 예측 신뢰도 구간 (간단한 버전)
        if 'total_amount' in customer_features.columns:
            historical_amount = customer_features['total_amount']
            
            # 예측값과 실제값의 관계
            fig.add_trace(
                go.Scatter(
                    x=historical_amount, 
                    y=predictions, 
                    mode='markers',
                    name='예측 vs 실제',
                    marker=dict(color='green', opacity=0.6)
                ),
                row=1, col=2
            )
        
        fig.update_layout(
            title_text="🎯 예측 결과 분석",
            showlegend=False,
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_business_insights_dashboard(customer_features: pd.DataFrame, evaluation_results: Dict) -> go.Figure:
        """비즈니스 인사이트 대시보드"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['고객 가치 분포', '구매 패턴 분석', '리텐션 분석', '모델 신뢰도'],
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "indicator"}]]
        )
        
        # 고객 가치 분포
        if 'customer_value_category' in customer_features.columns:
            value_counts = customer_features['customer_value_category'].value_counts()
            fig.add_trace(
                go.Pie(labels=value_counts.index, values=value_counts.values, name="고객 가치"),
                row=1, col=1
            )
        
        # 구매 패턴 분석
        if 'frequency' in customer_features.columns and 'total_amount' in customer_features.columns:
            fig.add_trace(
                go.Bar(
                    x=customer_features['frequency'].value_counts().index[:10],
                    y=customer_features['frequency'].value_counts().values[:10],
                    name="구매 빈도",
                    marker_color='lightgreen'
                ),
                row=1, col=2
            )
        
        # 리텐션 분석 (recency vs frequency)
        if 'recency_days' in customer_features.columns and 'frequency' in customer_features.columns:
            fig.add_trace(
                go.Scatter(
                    x=customer_features['recency_days'],
                    y=customer_features['frequency'],
                    mode='markers',
                    name='리텐션 패턴',
                    marker=dict(color='orange', opacity=0.6)
                ),
                row=2, col=1
            )
        
        # 모델 신뢰도
        r2_score = evaluation_results.get('r2_test', 0)
        confidence_score = min(r2_score * 100, 100)
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=confidence_score,
                title={'text': "모델 신뢰도 (%)"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkgreen"},
                       'steps': [{'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "lightgreen"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75, 'value': 90}}
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="💼 비즈니스 인사이트 대시보드",
            showlegend=False,
            height=800
        )
        
        return fig
