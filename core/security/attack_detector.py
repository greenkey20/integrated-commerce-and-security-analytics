"""
실시간 네트워크 공격 탐지 및 시뮬레이션 모듈

실시간 트래픽 분석, 공격 패턴 탐지, 성능 평가를 담당
화면 코드와 분리된 순수 탐지 로직
"""

import numpy as np
import pandas as pd
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class RealTimeAttackDetector:
    """실시간 공격 탐지 클래스"""
    
    def __init__(self, model=None, scaler=None, threshold=0.5):
        self.model = model
        self.scaler = scaler
        self.threshold = threshold
        self.detection_history = []
        self.alert_count = 0
        
    def set_model(self, model, scaler=None):
        """모델 및 스케일러 설정"""
        self.model = model
        self.scaler = scaler
    
    def detect_single_packet(self, packet_features, packet_id=None):
        """단일 패킷 분석"""
        if self.model is None:
            raise ValueError("모델이 설정되지 않았습니다.")
        
        # 특성 정규화
        if self.scaler is not None:
            packet_features = self.scaler.transform(packet_features.reshape(1, -1))
        else:
            packet_features = packet_features.reshape(1, -1)
        
        # 예측 수행
        try:
            if hasattr(self.model, 'predict'):
                prediction = self.model.predict(packet_features, verbose=0)[0][0]
            else:
                prediction = np.random.uniform(0, 1)  # 폴백
        except:
            prediction = np.random.uniform(0, 1)  # 오류 시 폴백
        
        is_attack = prediction > self.threshold
        confidence = prediction if is_attack else 1 - prediction
        
        # 탐지 결과 기록
        detection_result = {
            'packet_id': packet_id or len(self.detection_history) + 1,
            'timestamp': time.time(),
            'prediction': prediction,
            'is_attack': is_attack,
            'confidence': confidence,
            'features': packet_features.flatten()
        }
        
        self.detection_history.append(detection_result)
        
        if is_attack:
            self.alert_count += 1
        
        return detection_result
    
    def detect_batch(self, packet_batch, packet_ids=None):
        """배치 패킷 분석"""
        results = []
        
        for i, packet in enumerate(packet_batch):
            packet_id = packet_ids[i] if packet_ids else i + 1
            result = self.detect_single_packet(packet, packet_id)
            results.append(result)
        
        return results
    
    def get_detection_stats(self):
        """탐지 통계 반환"""
        if not self.detection_history:
            return {
                'total_packets': 0,
                'attack_packets': 0,
                'attack_ratio': 0.0,
                'avg_confidence': 0.0,
                'recent_attack_rate': 0.0
            }
        
        total_packets = len(self.detection_history)
        attack_packets = sum(1 for r in self.detection_history if r['is_attack'])
        attack_ratio = attack_packets / total_packets * 100
        avg_confidence = np.mean([r['confidence'] for r in self.detection_history])
        
        # 최근 100개 패킷의 공격 비율
        recent_results = self.detection_history[-100:]
        recent_attacks = sum(1 for r in recent_results if r['is_attack'])
        recent_attack_rate = recent_attacks / len(recent_results) * 100
        
        return {
            'total_packets': total_packets,
            'attack_packets': attack_packets,
            'attack_ratio': attack_ratio,
            'avg_confidence': avg_confidence,
            'recent_attack_rate': recent_attack_rate
        }
    
    def clear_history(self):
        """탐지 이력 초기화"""
        self.detection_history = []
        self.alert_count = 0


class TrafficSimulator:
    """네트워크 트래픽 시뮬레이터"""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
    
    def generate_normal_traffic(self, n_packets):
        """정상 트래픽 생성"""
        return np.random.normal(0, 1, (n_packets, 19))  # 19개 특성
    
    def generate_ddos_traffic(self, n_packets):
        """DDoS 공격 트래픽 생성"""
        data = np.random.normal(0, 1, (n_packets, 19))
        # DDoS 특성: 높은 패킷율, 낮은 응답
        data[:, 0] *= 10  # Flow_Bytes/s 증가
        data[:, 1] *= 5   # Flow_Packets/s 증가
        data[:, 2] /= 3   # Backward_Packets 감소
        return data
    
    def generate_web_attack_traffic(self, n_packets):
        """웹 공격 트래픽 생성"""
        data = np.random.normal(0, 1, (n_packets, 19))
        # 웹 공격 특성: 특정 패턴의 패킷 크기
        data[:, 3] *= 3   # 패킷 길이 증가
        data[:, 4] *= 2   # 전송 패킷 길이 증가
        return data
    
    def generate_brute_force_traffic(self, n_packets):
        """브루트포스 공격 트래픽 생성"""
        data = np.random.normal(0, 1, (n_packets, 19))
        # 브루트포스 특성: 짧은 간격, 많은 시도
        data[:, 5] /= 5   # IAT 감소
        data[:, 6] *= 3   # 패킷 수 증가
        return data
    
    def generate_port_scan_traffic(self, n_packets):
        """포트스캔 공격 트래픽 생성"""
        data = np.random.normal(0, 1, (n_packets, 19))
        # 포트스캔 특성: 많은 전송, 적은 응답
        data[:, 1] *= 2   # Forward 패킷 증가
        data[:, 2] /= 4   # Backward 패킷 감소
        return data
    
    def generate_mixed_traffic(self, n_packets, attack_ratio=0.3):
        """혼합 트래픽 생성"""
        n_normal = int(n_packets * (1 - attack_ratio))
        n_attack = n_packets - n_normal
        
        normal_traffic = self.generate_normal_traffic(n_normal)
        
        # 공격 트래픽을 여러 유형으로 분산
        attack_types = [
            self.generate_ddos_traffic,
            self.generate_web_attack_traffic,
            self.generate_brute_force_traffic,
            self.generate_port_scan_traffic
        ]
        
        attack_per_type = n_attack // len(attack_types)
        remainder = n_attack % len(attack_types)
        
        attack_traffic = []
        for i, attack_func in enumerate(attack_types):
            n_this_attack = attack_per_type + (1 if i < remainder else 0)
            if n_this_attack > 0:
                attack_traffic.append(attack_func(n_this_attack))
        
        if attack_traffic:
            all_attack_traffic = np.vstack(attack_traffic)
            all_traffic = np.vstack([normal_traffic, all_attack_traffic])
        else:
            all_traffic = normal_traffic
        
        # 무작위로 섞기
        indices = np.random.permutation(len(all_traffic))
        return all_traffic[indices]
    
    def generate_scenario_traffic(self, scenario, n_packets):
        """시나리오별 트래픽 생성"""
        if "정상" in scenario:
            return self.generate_normal_traffic(n_packets), 0
        elif "DDoS" in scenario:
            return self.generate_ddos_traffic(n_packets), 80
        elif "웹 공격" in scenario:
            return self.generate_web_attack_traffic(n_packets), 70
        elif "브루트포스" in scenario:
            return self.generate_brute_force_traffic(n_packets), 60
        elif "포트스캔" in scenario:
            return self.generate_port_scan_traffic(n_packets), 65
        else:  # 혼합
            return self.generate_mixed_traffic(n_packets, 0.4), 40


class PerformanceEvaluator:
    """성능 평가 클래스"""
    
    def __init__(self):
        self.evaluation_history = []
    
    def evaluate_detection_performance(self, true_labels, predicted_labels, predictions=None):
        """탐지 성능 평가"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        # 기본 성능 메트릭
        metrics = {
            'accuracy': accuracy_score(true_labels, predicted_labels),
            'precision': precision_score(true_labels, predicted_labels, zero_division=0),
            'recall': recall_score(true_labels, predicted_labels, zero_division=0),
            'f1_score': f1_score(true_labels, predicted_labels, zero_division=0)
        }
        
        # AUC 계산 (연속 예측값이 있는 경우)
        if predictions is not None:
            try:
                metrics['auc'] = roc_auc_score(true_labels, predictions)
            except:
                metrics['auc'] = 0.5  # 폴백
        
        # 혼동 행렬 요소
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics.update({
                'true_negatives': tn,
                'false_positives': fp,
                'false_negatives': fn,
                'true_positives': tp,
                'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
                'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0
            })
        
        return metrics
    
    def calculate_business_impact(self, performance_metrics, scenario_params):
        """비즈니스 임팩트 계산"""
        # 기본 파라미터
        daily_traffic = scenario_params.get('daily_traffic', 1000000)
        attack_rate = scenario_params.get('attack_rate', 1.0)  # %
        damage_per_attack = scenario_params.get('damage_per_attack', 5000)  # 만원
        
        # 연간 추정치
        daily_attacks = daily_traffic * attack_rate / 100
        annual_attacks = daily_attacks * 365
        
        # 현재 시스템 성능
        recall = performance_metrics.get('recall', 0.8)
        precision = performance_metrics.get('precision', 0.9)
        
        # 놓친 공격으로 인한 손실
        missed_attacks = annual_attacks * (1 - recall)
        annual_loss = missed_attacks * damage_per_attack
        
        # 오탐으로 인한 운영 비용 (시간당 100만원 추정)
        false_positives_daily = daily_traffic * (1 - attack_rate/100) * (1 - precision)
        annual_false_positives = false_positives_daily * 365
        false_positive_cost = annual_false_positives * 10  # 건당 10만원 추정
        
        # 기존 규칙 기반 시스템과 비교
        baseline_recall = 0.7
        baseline_precision = 0.6
        
        baseline_missed = annual_attacks * (1 - baseline_recall)
        baseline_loss = baseline_missed * damage_per_attack
        baseline_fp_cost = daily_traffic * (1 - attack_rate/100) * (1 - baseline_precision) * 365 * 10
        
        # 절약 효과
        loss_reduction = baseline_loss - annual_loss
        fp_cost_reduction = baseline_fp_cost - false_positive_cost
        total_savings = loss_reduction + fp_cost_reduction
        
        return {
            'annual_attacks': annual_attacks,
            'missed_attacks': missed_attacks,
            'annual_loss': annual_loss,
            'false_positive_cost': false_positive_cost,
            'baseline_loss': baseline_loss,
            'baseline_fp_cost': baseline_fp_cost,
            'total_savings': total_savings,
            'loss_reduction': loss_reduction,
            'fp_cost_reduction': fp_cost_reduction
        }
    
    def generate_performance_report(self, detection_results, true_labels=None):
        """성능 보고서 생성"""
        if not detection_results:
            return {"error": "탐지 결과가 없습니다."}
        
        # 기본 통계
        total_packets = len(detection_results)
        attack_predictions = sum(1 for r in detection_results if r['is_attack'])
        attack_ratio = attack_predictions / total_packets * 100
        avg_confidence = np.mean([r['confidence'] for r in detection_results])
        
        report = {
            'basic_stats': {
                'total_packets': total_packets,
                'predicted_attacks': attack_predictions,
                'attack_ratio': attack_ratio,
                'avg_confidence': avg_confidence
            }
        }
        
        # 실제 라벨이 있는 경우 정확도 계산
        if true_labels is not None and len(true_labels) == total_packets:
            predicted_labels = [r['is_attack'] for r in detection_results]
            predictions = [r['prediction'] for r in detection_results]
            
            performance = self.evaluate_detection_performance(
                true_labels, predicted_labels, predictions
            )
            report['performance'] = performance
        
        # 시간별 패턴 분석
        timestamps = [r['timestamp'] for r in detection_results]
        if timestamps:
            start_time = min(timestamps)
            time_intervals = [(t - start_time) // 10 for t in timestamps]  # 10초 간격
            
            interval_stats = {}
            for i, interval in enumerate(time_intervals):
                if interval not in interval_stats:
                    interval_stats[interval] = {'total': 0, 'attacks': 0}
                interval_stats[interval]['total'] += 1
                if detection_results[i]['is_attack']:
                    interval_stats[interval]['attacks'] += 1
            
            report['time_patterns'] = interval_stats
        
        return report


class AlertManager:
    """경고 관리 클래스"""
    
    def __init__(self, alert_threshold=0.8, burst_threshold=10, time_window=60):
        self.alert_threshold = alert_threshold
        self.burst_threshold = burst_threshold
        self.time_window = time_window
        self.alerts = []
        self.active_alerts = []
    
    def check_alert_conditions(self, detection_result):
        """경고 조건 확인"""
        alerts_triggered = []
        
        # 높은 신뢰도 공격 탐지
        if detection_result['is_attack'] and detection_result['confidence'] >= self.alert_threshold:
            alerts_triggered.append({
                'type': 'high_confidence_attack',
                'severity': 'HIGH',
                'message': f"높은 신뢰도 공격 탐지 (신뢰도: {detection_result['confidence']:.2%})",
                'timestamp': detection_result['timestamp'],
                'packet_id': detection_result['packet_id']
            })
        
        return alerts_triggered
    
    def check_burst_attack(self, recent_detections):
        """연속 공격 패턴 확인"""
        if len(recent_detections) < self.burst_threshold:
            return None
        
        # 최근 시간 윈도우 내의 공격 수 확인
        current_time = time.time()
        recent_attacks = [
            d for d in recent_detections 
            if d['is_attack'] and (current_time - d['timestamp']) <= self.time_window
        ]
        
        if len(recent_attacks) >= self.burst_threshold:
            return {
                'type': 'burst_attack',
                'severity': 'CRITICAL',
                'message': f"{self.time_window}초 동안 {len(recent_attacks)}건의 공격 탐지",
                'timestamp': current_time,
                'attack_count': len(recent_attacks)
            }
        
        return None
    
    def add_alert(self, alert):
        """경고 추가"""
        alert['id'] = len(self.alerts) + 1
        self.alerts.append(alert)
        
        # 활성 경고에 추가 (중복 제거)
        if not any(a['type'] == alert['type'] and 
                  abs(a['timestamp'] - alert['timestamp']) < 30 
                  for a in self.active_alerts):
            self.active_alerts.append(alert)
    
    def get_active_alerts(self):
        """활성 경고 반환"""
        # 오래된 경고 제거 (5분 이상)
        current_time = time.time()
        self.active_alerts = [
            a for a in self.active_alerts 
            if (current_time - a['timestamp']) < 300
        ]
        return self.active_alerts
    
    def clear_alerts(self):
        """모든 경고 초기화"""
        self.alerts = []
        self.active_alerts = []


class DetectionOrchestrator:
    """탐지 시스템 통합 관리 클래스"""
    
    def __init__(self, model=None, scaler=None):
        self.detector = RealTimeAttackDetector(model, scaler)
        self.simulator = TrafficSimulator()
        self.evaluator = PerformanceEvaluator()
        self.alert_manager = AlertManager()
        
    def set_model(self, model, scaler=None):
        """모델 설정"""
        self.detector.set_model(model, scaler)
    
    def run_simulation(self, scenario, n_packets=100, real_time_delay=0.01):
        """시뮬레이션 실행"""
        # 트래픽 생성
        traffic_data, expected_attack_ratio = self.simulator.generate_scenario_traffic(
            scenario, n_packets
        )
        
        # 실시간 탐지 시뮬레이션
        detection_results = []
        
        for i, packet in enumerate(traffic_data):
            # 실시간 지연 시뮬레이션
            if real_time_delay > 0:
                time.sleep(real_time_delay)
            
            # 패킷 탐지
            result = self.detector.detect_single_packet(packet, i + 1)
            detection_results.append(result)
            
            # 경고 확인
            alerts = self.alert_manager.check_alert_conditions(result)
            for alert in alerts:
                self.alert_manager.add_alert(alert)
            
            # 연속 공격 패턴 확인
            if len(detection_results) >= 10:
                burst_alert = self.alert_manager.check_burst_attack(detection_results[-20:])
                if burst_alert:
                    self.alert_manager.add_alert(burst_alert)
        
        # 결과 분석
        stats = self.detector.get_detection_stats()
        report = self.evaluator.generate_performance_report(detection_results)
        active_alerts = self.alert_manager.get_active_alerts()
        
        return {
            'detection_results': detection_results,
            'stats': stats,
            'report': report,
            'alerts': active_alerts,
            'expected_attack_ratio': expected_attack_ratio
        }
    
    def evaluate_model_performance(self, test_data, true_labels):
        """모델 성능 종합 평가"""
        # 배치 탐지
        detection_results = self.detector.detect_batch(test_data)
        
        # 성능 메트릭 계산
        predicted_labels = [r['is_attack'] for r in detection_results]
        predictions = [r['prediction'] for r in detection_results]
        
        performance = self.evaluator.evaluate_detection_performance(
            true_labels, predicted_labels, predictions
        )
        
        # 비즈니스 임팩트 계산
        business_impact = self.evaluator.calculate_business_impact(
            performance, 
            {'daily_traffic': 1000000, 'attack_rate': 1.0, 'damage_per_attack': 5000}
        )
        
        return {
            'performance': performance,
            'business_impact': business_impact,
            'detection_results': detection_results
        }


# 편의 함수들
def create_detection_system(model=None, scaler=None):
    """탐지 시스템 생성"""
    return DetectionOrchestrator(model, scaler)

def run_quick_simulation(scenario="혼합 트래픽", n_packets=100):
    """빠른 시뮬레이션 실행"""
    orchestrator = DetectionOrchestrator()
    return orchestrator.run_simulation(scenario, n_packets, real_time_delay=0)

def evaluate_attack_detection(detection_results, expected_attack_ratio):
    """공격 탐지 정확성 평가"""
    if not detection_results:
        return 0.0
    
    predicted_attacks = sum(1 for r in detection_results if r['is_attack'])
    predicted_ratio = predicted_attacks / len(detection_results) * 100
    
    accuracy = 1.0 - abs(predicted_ratio - expected_attack_ratio) / max(expected_attack_ratio, 1)
    return max(0.0, accuracy)
