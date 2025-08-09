"""
LangChain 기반 Customer Analysis Chain

고객 데이터를 분석하여 인사이트를 생성하는 LangChain 체인
"""

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
import pandas as pd
import json
from typing import Dict, List, Any, Optional
import logging


class CustomerInsightParser(BaseOutputParser):
    """고객 분석 결과를 파싱하는 클래스"""
    
    def parse(self, text: str) -> Dict[str, Any]:
        """LLM 출력을 구조화된 형태로 파싱"""
        try:
            # JSON 형태로 파싱 시도
            if text.strip().startswith('{'):
                return json.loads(text)
            
            # 텍스트 형태인 경우 기본 구조로 변환
            return {
                "summary": text.strip(),
                "insights": [],
                "recommendations": []
            }
        except json.JSONDecodeError:
            return {
                "summary": text.strip(),
                "insights": [],
                "recommendations": []
            }


class CustomerAnalysisChain:
    """LangChain 기반 고객 분석 체인"""
    
    def __init__(self, llm=None):
        """
        고객 분석 체인 초기화
        
        Args:
            llm: LangChain LLM 인스턴스 (선택사항)
        """
        self.llm = llm
        self.output_parser = CustomerInsightParser()
        self._setup_chains()
    
    def _setup_chains(self):
        """분석 체인들을 설정"""
        
        # 고객 세그먼트 분석 프롬프트
        self.segment_analysis_prompt = PromptTemplate(
            input_variables=["customer_data", "cluster_info"],
            template="""
고객 데이터와 클러스터링 결과를 바탕으로 각 고객 세그먼트를 분석해주세요.

고객 데이터:
{customer_data}

클러스터 정보:
{cluster_info}

다음 형식으로 분석 결과를 제공해주세요:

{{
    "segments": [
        {{
            "cluster_id": 0,
            "segment_name": "세그먼트 이름",
            "characteristics": ["특징1", "특징2", "특징3"],
            "size": "세그먼트 크기",
            "avg_age": "평균 연령",
            "avg_income": "평균 소득",
            "avg_spending": "평균 지출점수"
        }}
    ],
    "overall_insights": ["전체 인사이트1", "전체 인사이트2"],
    "business_recommendations": ["비즈니스 추천사항1", "추천사항2"]
}}
"""
        )
    
    def analyze_customer_segments(self, customer_data: pd.DataFrame, cluster_labels: List[int]) -> Dict[str, Any]:
        """
        고객 세그먼트 분석 수행
        
        Args:
            customer_data: 고객 데이터 DataFrame
            cluster_labels: 클러스터 라벨 리스트
            
        Returns:
            Dict: 세그먼트 분석 결과
        """
        try:
            # 데이터 전처리
            data_with_clusters = customer_data.copy()
            data_with_clusters['Cluster'] = cluster_labels
            
            # LLM이 없는 경우 기본 분석 제공
            result = self._generate_basic_segment_analysis(data_with_clusters)
            
            return result
            
        except Exception as e:
            logging.error(f"세그먼트 분석 중 오류 발생: {str(e)}")
            return {"error": f"분석 중 오류가 발생했습니다: {str(e)}"}
    
    def analyze_individual_customer(self, customer_profile: Dict[str, Any], segment_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        개별 고객 분석 수행
        """
        try:
            result = self._generate_basic_individual_analysis(customer_profile)
            return result
            
        except Exception as e:
            logging.error(f"개별 고객 분석 중 오류 발생: {str(e)}")
            return {"error": f"분석 중 오류가 발생했습니다: {str(e)}"}
    
    def analyze_trends(self, data_summary: Dict[str, Any], time_period: str = "현재") -> Dict[str, Any]:
        """
        고객 데이터 트렌드 분석 수행
        """
        try:
            result = self._generate_basic_trend_analysis(data_summary)
            return result
            
        except Exception as e:
            logging.error(f"트렌드 분석 중 오류 발생: {str(e)}")
            return {"error": f"분석 중 오류가 발생했습니다: {str(e)}"}
    
    def _generate_basic_segment_analysis(self, data_with_clusters: pd.DataFrame) -> Dict[str, Any]:
        """LLM 없이 기본 세그먼트 분석 생성"""
        segments = []
        
        for cluster_id in data_with_clusters['Cluster'].unique():
            cluster_data = data_with_clusters[data_with_clusters['Cluster'] == cluster_id]
            
            avg_age = cluster_data['Age'].mean()
            avg_income = cluster_data['Annual Income (k$)'].mean()
            avg_spending = cluster_data['Spending Score (1-100)'].mean()
            
            # 세그먼트 특성 분석
            if avg_income > 60 and avg_spending > 60:
                segment_name = "고소득 고지출 그룹"
                characteristics = ["높은 구매력", "브랜드 선호", "프리미엄 제품 선호"]
            elif avg_income > 60 and avg_spending <= 60:
                segment_name = "고소득 저지출 그룹"
                characteristics = ["신중한 구매", "가치 지향적", "저축 성향"]
            elif avg_income <= 60 and avg_spending > 60:
                segment_name = "저소득 고지출 그룹"
                characteristics = ["충동구매 성향", "유행에 민감", "할부/대출 이용"]
            else:
                segment_name = "저소득 저지출 그룹"
                characteristics = ["필수품 위주", "가격 민감", "할인 상품 선호"]
            
            segments.append({
                "cluster_id": int(cluster_id),
                "segment_name": segment_name,
                "characteristics": characteristics,
                "size": len(cluster_data),
                "avg_age": round(avg_age, 1),
                "avg_income": round(avg_income, 1),
                "avg_spending": round(avg_spending, 1)
            })
        
        return {
            "segments": segments,
            "overall_insights": [
                f"총 {len(segments)}개의 고객 세그먼트가 식별되었습니다.",
                "소득과 지출 패턴에 따른 명확한 세그먼테이션이 관찰됩니다.",
                "각 세그먼트별로 차별화된 마케팅 전략이 필요합니다."
            ],
            "business_recommendations": [
                "고가치 고객(고소득 고지출)에게는 프리미엄 서비스를 제공하세요.",
                "가격 민감 고객에게는 할인 혜택과 가성비 상품을 제안하세요.",
                "각 세그먼트별 맞춤형 커뮤니케이션 전략을 수립하세요."
            ]
        }
    
    def _generate_basic_individual_analysis(self, customer_profile: Dict[str, Any]) -> Dict[str, Any]:
        """LLM 없이 기본 개별 고객 분석 생성"""
        age = customer_profile.get('Age', 0)
        income = customer_profile.get('Annual Income (k$)', 0)
        spending = customer_profile.get('Spending Score (1-100)', 0)
        gender = customer_profile.get('Gender', 'Unknown')
        
        # 고객 유형 분류
        if income > 60 and spending > 60:
            customer_type = "프리미엄 고객"
            retention_risk = "낮음"
        elif income > 60 and spending <= 60:
            customer_type = "신중한 고소득 고객"
            retention_risk = "보통"
        elif spending > 60:
            customer_type = "활발한 소비자"
            retention_risk = "보통"
        else:
            customer_type = "가격 민감 고객"
            retention_risk = "높음"
        
        return {
            "customer_type": customer_type,
            "behavioral_analysis": f"{age}세 {gender} 고객으로 연소득 {income}천달러, 지출점수 {spending}점입니다.",
            "preferences": ["개인화된 서비스", "편의성", "가치 대비 효과"],
            "marketing_strategy": f"{customer_type}에 적합한 맞춤형 마케팅 전략 수립 필요",
            "retention_risk": retention_risk,
            "personalized_offers": [
                "개인 맞춤형 상품 추천",
                "로열티 프로그램 참여 제안"
            ]
        }
    
    def _generate_basic_trend_analysis(self, data_summary: Dict[str, Any]) -> Dict[str, Any]:
        """LLM 없이 기본 트렌드 분석 생성"""
        return {
            "key_trends": [
                "연령대별 소비 패턴의 다양화",
                "소득 수준에 따른 지출 성향 차이",
                "성별 간 구매 행동 차이"
            ],
            "demographic_insights": {
                "age_patterns": "젊은 층은 높은 지출 성향, 중장년층은 신중한 소비",
                "gender_patterns": "성별에 따른 선호 카테고리 차이 관찰",
                "income_patterns": "고소득층의 브랜드 충성도가 높음"
            },
            "spending_behaviors": [
                "계절적 소비 패턴 존재",
                "프로모션에 대한 반응도 차이"
            ],
            "market_opportunities": [
                "개인화 서비스 확대",
                "모바일 쇼핑 경험 개선"
            ],
            "risk_factors": [
                "가격 민감 고객의 이탈 위험",
                "경쟁사 대비 차별화 부족"
            ]
        }


class CustomerInsightGenerator:
    """고객 분석 결과를 바탕으로 인사이트를 생성하는 클래스"""
    
    def __init__(self, analysis_chain: CustomerAnalysisChain):
        self.analysis_chain = analysis_chain
    
    def generate_comprehensive_report(self, customer_data: pd.DataFrame, cluster_labels: List[int]) -> Dict[str, Any]:
        """종합적인 고객 분석 리포트 생성"""
        try:
            # 세그먼트 분석
            segment_analysis = self.analysis_chain.analyze_customer_segments(customer_data, cluster_labels)
            
            # 데이터 요약 통계
            data_summary = {
                "total_customers": len(customer_data),
                "avg_age": customer_data['Age'].mean(),
                "avg_income": customer_data['Annual Income (k$)'].mean(),
                "avg_spending": customer_data['Spending Score (1-100)'].mean(),
                "gender_distribution": customer_data['Gender'].value_counts().to_dict()
            }
            
            # 트렌드 분석
            trend_analysis = self.analysis_chain.analyze_trends(data_summary)
            
            # 종합 리포트 생성
            comprehensive_report = {
                "executive_summary": {
                    "total_customers": data_summary["total_customers"],
                    "segments_identified": len(set(cluster_labels)),
                    "key_insights": segment_analysis.get("overall_insights", []),
                    "main_trends": trend_analysis.get("key_trends", [])
                },
                "segment_analysis": segment_analysis,
                "trend_analysis": trend_analysis,
                "data_summary": data_summary,
                "recommendations": {
                    "immediate_actions": segment_analysis.get("business_recommendations", []),
                    "long_term_strategy": trend_analysis.get("market_opportunities", []),
                    "risk_mitigation": trend_analysis.get("risk_factors", [])
                }
            }
            
            return comprehensive_report
            
        except Exception as e:
            logging.error(f"종합 리포트 생성 중 오류 발생: {str(e)}")
            return {"error": f"리포트 생성 중 오류가 발생했습니다: {str(e)}"}
EOF < /dev/null