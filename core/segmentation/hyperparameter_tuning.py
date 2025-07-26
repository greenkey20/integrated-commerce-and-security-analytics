"""
딥러닝 하이퍼파라미터 튜닝 모듈

"혼자 공부하는 머신러닝, 딥러닝" 방식을 딥러닝에 적용한 실습 코드
학습 목적으로 단계별 하이퍼파라미터 튜닝을 수행
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 프로젝트 내부 모듈 import
from core.segmentation.data_processing import CustomerDataProcessor
from core.segmentation.models import DeepLearningModels
from config.settings import DeepLearningConfig
import streamlit as st


class HyperparameterTuner:
    """하이퍼파라미터 튜닝을 담당하는 클래스"""
    
    def __init__(self):
        self.data_processor = CustomerDataProcessor()
        self.dl_models = DeepLearningModels()
        self.results_history = []
        
    def prepare_data(self):
        """데이터 로드 및 전처리"""
        print("📊 데이터 로드 및 전처리 중...")
        
        # 데이터 로드
        data = self.data_processor.load_data()
        print(f"데이터 크기: {data.shape}")
        
        # 특성 추출 및 스케일링
        features = data[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
        scaler = StandardScaler()
        X = scaler.fit_transform(features)
        
        # 임시 레이블 생성 (K-means로 클러스터링 수행)
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        y = kmeans.fit_predict(X)
        
        # 훈련/테스트 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
        )
        
        print(f"훈련 세트: {X_train.shape}, 테스트 세트: {X_test.shape}")
        print(f"클러스터 분포: {np.bincount(y)}")
        
        return X_train, X_test, y_train, y_test
    
    def tune_learning_rate(self, X_train, y_train, X_test, y_test):
        """학습률 튜닝 - 가장 중요한 하이퍼파라미터"""
        
        learning_rates = [0.1, 0.01, 0.001, 0.0001]  # 10배씩 줄여가며 테스트
        results = []
        
        print("\\n=== 1단계: Learning Rate 튜닝 ===")
        print("학습률은 모델 성능에 가장 큰 영향을 미치는 하이퍼파라미터야!")
        
        for lr in learning_rates:
            print(f"\\n🔧 학습률 {lr} 테스트 중...")
            
            try:
                # 기본 설정으로 모델 생성 (다른 하이퍼파라미터는 고정)
                model, error = self.dl_models.create_safe_classification_model(
                    input_dim=3, 
                    n_clusters=5,
                    hidden_units=64,     # 고정값
                    dropout_rate=0.2,    # 고정값  
                    learning_rate=lr     # 테스트할 값
                )
                
                if error:
                    print(f"❌ 모델 생성 실패: {error}")
                    continue
                    
                # 짧은 에포크로 빠르게 테스트 (30 에포크)
                history, error = self.dl_models.train_model_with_progress(
                    model, X_train, y_train, X_test, y_test, 
                    epochs=30, progress_bar=None, status_text=None
                )
                
                if error:
                    print(f"❌ 훈련 실패: {error}")
                    continue
                    
                # 최종 검증 정확도 기록
                final_val_accuracy = max(history.history['val_accuracy'])
                
                results.append({
                    'learning_rate': lr,
                    'val_accuracy': final_val_accuracy,
                    'history': history.history
                })
                
                print(f"✅ 학습률 {lr}: 최고 검증 정확도 = {final_val_accuracy:.4f}")
                
                # 학습 과정 간단 분석
                if lr >= 0.1:
                    print("   → 학습률이 너무 높을 수 있음 (발산 위험)")
                elif lr <= 0.0001:
                    print("   → 학습률이 너무 낮을 수 있음 (학습 속도 느림)")
                else:
                    print("   → 적절한 학습률 범위")
                    
            except Exception as e:
                print(f"❌ 예외 발생: {str(e)}")
                continue
        
        if not results:
            print("❌ 모든 학습률 테스트 실패")
            return None, []
            
        # 최적 학습률 선택
        best_lr_result = max(results, key=lambda x: x['val_accuracy'])
        best_lr = best_lr_result['learning_rate']
        
        print(f"\\n🎯 최적 학습률: {best_lr} (정확도: {best_lr_result['val_accuracy']:.4f})")
        
        return best_lr, results

    def tune_hidden_units(self, X_train, y_train, X_test, y_test, best_lr):
        """은닉층 뉴런 수 튜닝"""
        
        hidden_units_list = [16, 32, 64, 128, 256]  # 2배씩 늘려가며 테스트
        results = []
        
        print(f"\\n=== 2단계: Hidden Units 튜닝 (최적 학습률 {best_lr} 사용) ===")
        print("은닉층 크기는 모델의 표현력을 결정해!")
        
        for units in hidden_units_list:
            print(f"\\n🔧 은닉층 {units}개 뉴런 테스트 중...")
            
            try:
                model, error = self.dl_models.create_safe_classification_model(
                    input_dim=3,
                    n_clusters=5, 
                    hidden_units=units,     # 테스트할 값
                    dropout_rate=0.2,       # 고정값
                    learning_rate=best_lr   # 1단계에서 찾은 최적값
                )
                
                if error:
                    print(f"❌ 모델 생성 실패: {error}")
                    continue
                    
                history, error = self.dl_models.train_model_with_progress(
                    model, X_train, y_train, X_test, y_test,
                    epochs=50, progress_bar=None, status_text=None
                )
                
                if error:
                    print(f"❌ 훈련 실패: {error}")
                    continue
                    
                final_val_accuracy = max(history.history['val_accuracy'])
                results.append({
                    'hidden_units': units,
                    'val_accuracy': final_val_accuracy,
                    'history': history.history
                })
                
                print(f"✅ 은닉층 {units}개: 최고 검증 정확도 = {final_val_accuracy:.4f}")
                
                # 모델 복잡도 분석
                if units <= 32:
                    print("   → 작은 모델: 빠르지만 표현력 제한적")
                elif units >= 256:
                    print("   → 큰 모델: 표현력 높지만 과적합 위험")
                else:
                    print("   → 적절한 크기: 균형 잡힌 모델")
                    
            except Exception as e:
                print(f"❌ 예외 발생: {str(e)}")
                continue
        
        if not results:
            print("❌ 모든 은닉층 크기 테스트 실패")
            return None, []
            
        best_units_result = max(results, key=lambda x: x['val_accuracy'])
        best_units = best_units_result['hidden_units']
        
        print(f"\\n🎯 최적 은닉층 크기: {best_units} (정확도: {best_units_result['val_accuracy']:.4f})")
        
        return best_units, results

    def tune_dropout_rate(self, X_train, y_train, X_test, y_test, best_lr, best_units):
        """드롭아웃 비율 튜닝"""
        
        dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]  # 0%부터 50%까지
        results = []
        
        print(f"\\n=== 3단계: Dropout Rate 튜닝 (학습률 {best_lr}, 은닉층 {best_units} 사용) ===")
        print("드롭아웃은 과적합을 방지하는 핵심 기법이야!")
        
        for dropout in dropout_rates:
            print(f"\\n🔧 드롭아웃 {dropout:.1f} 테스트 중...")
            
            try:
                model, error = self.dl_models.create_safe_classification_model(
                    input_dim=3,
                    n_clusters=5,
                    hidden_units=best_units,  # 2단계에서 찾은 최적값
                    dropout_rate=dropout,     # 테스트할 값
                    learning_rate=best_lr     # 1단계에서 찾은 최적값
                )
                
                if error:
                    print(f"❌ 모델 생성 실패: {error}")
                    continue
                    
                history, error = self.dl_models.train_model_with_progress(
                    model, X_train, y_train, X_test, y_test,
                    epochs=100, progress_bar=None, status_text=None
                )
                
                if error:
                    print(f"❌ 훈련 실패: {error}")
                    continue
                    
                final_val_accuracy = max(history.history['val_accuracy'])
                final_train_accuracy = max(history.history['accuracy'])
                
                results.append({
                    'dropout_rate': dropout,
                    'val_accuracy': final_val_accuracy,
                    'train_accuracy': final_train_accuracy,
                    'overfitting_gap': final_train_accuracy - final_val_accuracy,
                    'history': history.history
                })
                
                print(f"✅ 드롭아웃 {dropout:.1f}: 검증 정확도 = {final_val_accuracy:.4f}")
                print(f"   훈련 정확도 = {final_train_accuracy:.4f}, 과적합 갭 = {final_train_accuracy - final_val_accuracy:.4f}")
                
                # 과적합 분석
                if dropout == 0.0:
                    print("   → 드롭아웃 없음: 과적합 위험 있음")
                elif dropout >= 0.4:
                    print("   → 높은 드롭아웃: 과적합 방지 강화, 표현력 감소")
                else:
                    print("   → 적절한 드롭아웃: 균형 잡힌 정규화")
                    
            except Exception as e:
                print(f"❌ 예외 발생: {str(e)}")
                continue
        
        if not results:
            print("❌ 모든 드롭아웃 비율 테스트 실패")
            return None, []
            
        best_dropout_result = max(results, key=lambda x: x['val_accuracy'])
        best_dropout = best_dropout_result['dropout_rate']
        
        print(f"\\n🎯 최적 드롭아웃: {best_dropout} (정확도: {best_dropout_result['val_accuracy']:.4f})")
        
        return best_dropout, results

    def final_validation(self, X_train, y_train, X_test, y_test, 
                        best_lr, best_units, best_dropout):
        """최적 하이퍼파라미터로 최종 모델 훈련 및 검증"""
        
        print(f"\\n=== 🏆 최종 검증 ===")
        print(f"최적 하이퍼파라미터:")
        print(f"- Learning Rate: {best_lr}")
        print(f"- Hidden Units: {best_units}")  
        print(f"- Dropout Rate: {best_dropout}")
        
        try:
            # 최적 설정으로 모델 생성
            final_model, error = self.dl_models.create_safe_classification_model(
                input_dim=3,
                n_clusters=5,
                hidden_units=best_units,
                dropout_rate=best_dropout,
                learning_rate=best_lr
            )
            
            if error:
                print(f"❌ 최종 모델 생성 실패: {error}")
                return None, None
                
            print("\\n🚀 최종 모델 훈련 시작 (충분한 에포크로)...")
            
            # 충분한 에포크로 최종 훈련
            final_history, error = self.dl_models.train_model_with_progress(
                final_model, X_train, y_train, X_test, y_test,
                epochs=200, progress_bar=None, status_text=None
            )
            
            if error:
                print(f"❌ 최종 훈련 실패: {error}")
                return None, None
                
            # 최종 성능 평가
            test_loss, test_accuracy = final_model.evaluate(X_test, y_test, verbose=0)
            
            print(f"\\n🏆 최종 테스트 정확도: {test_accuracy:.4f}")
            print(f"🏆 최종 테스트 손실: {test_loss:.4f}")
            
            # 훈련 과정 요약
            final_train_acc = max(final_history.history['accuracy'])
            final_val_acc = max(final_history.history['val_accuracy'])
            print(f"🏆 최고 훈련 정확도: {final_train_acc:.4f}")
            print(f"🏆 최고 검증 정확도: {final_val_acc:.4f}")
            print(f"🏆 과적합 갭: {final_train_acc - final_val_acc:.4f}")
            
            return final_model, final_history
            
        except Exception as e:
            print(f"❌ 최종 검증 중 예외 발생: {str(e)}")
            return None, None

    def save_results_to_csv(self, results_summary, filename=None):
        """결과를 CSV 파일로 저장"""
        if filename is None:
            timestamp = int(time.time())
            filename = f"hyperparameter_tuning_results_{timestamp}.csv"
            
        try:
            df = pd.DataFrame(results_summary)
            df.to_csv(filename, index=False, encoding='utf-8')
            print(f"\\n💾 결과가 {filename}에 저장되었습니다.")
        except Exception as e:
            print(f"❌ 결과 저장 실패: {str(e)}")

    def run_complete_tuning(self, save_results=True):
        """전체 하이퍼파라미터 튜닝 프로세스 실행"""
        
        print("🌱 딥러닝 하이퍼파라미터 튜닝 시작!")
        print("=" * 60)
        print("이 과정은 '혼자 공부하는 머신러닝, 딥러닝' 방식을 딥러닝에 적용한 것입니다.")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # 1. 데이터 준비
            X_train, X_test, y_train, y_test = self.prepare_data()
            
            # 2. 학습률 튜닝
            best_lr, lr_results = self.tune_learning_rate(X_train, y_train, X_test, y_test)
            if best_lr is None:
                print("❌ 학습률 튜닝 실패, 중단합니다.")
                return None
            
            # 3. 은닉층 크기 튜닝  
            best_units, units_results = self.tune_hidden_units(X_train, y_train, X_test, y_test, best_lr)
            if best_units is None:
                print("❌ 은닉층 튜닝 실패, 중단합니다.")
                return None
            
            # 4. 드롭아웃 튜닝
            best_dropout, dropout_results = self.tune_dropout_rate(X_train, y_train, X_test, y_test, best_lr, best_units)
            if best_dropout is None:
                print("❌ 드롭아웃 튜닝 실패, 중단합니다.")
                return None
            
            # 5. 최종 검증
            final_model, final_history = self.final_validation(
                X_train, y_train, X_test, y_test, best_lr, best_units, best_dropout
            )
            
            end_time = time.time()
            print(f"\\n⏰ 총 튜닝 시간: {end_time - start_time:.2f}초")
            
            # 결과 요약
            results_summary = {
                'best_learning_rate': best_lr,
                'best_hidden_units': best_units, 
                'best_dropout_rate': best_dropout,
                'final_test_accuracy': final_model.evaluate(X_test, y_test, verbose=0)[1] if final_model else None,
                'tuning_time_seconds': end_time - start_time
            }
            
            # 결과 저장
            if save_results and final_model:
                self.save_results_to_csv([results_summary])
            
            print("\\n🎉 하이퍼파라미터 튜닝 완료!")
            
            return {
                'best_learning_rate': best_lr,
                'best_hidden_units': best_units, 
                'best_dropout_rate': best_dropout,
                'final_model': final_model,
                'final_history': final_history,
                'lr_results': lr_results,
                'units_results': units_results,
                'dropout_results': dropout_results
            }
            
        except Exception as e:
            print(f"❌ 튜닝 프로세스 중 예외 발생: {str(e)}")
            return None


def run_grid_search_experiment(limited_combinations=True):
    """그리드 서치 실험 실행 (사이킷런 GridSearchCV 방식과 유사)"""
    
    print("\\n🔍 그리드 서치 실험 시작!")
    print("=" * 40)
    
    tuner = HyperparameterTuner()
    X_train, X_test, y_train, y_test = tuner.prepare_data()
    
    # 하이퍼파라미터 그리드 정의
    if limited_combinations:
        # 제한된 조합 (빠른 테스트용)
        param_grid = {
            'learning_rate': [0.01, 0.001],
            'hidden_units': [32, 64],
            'dropout_rate': [0.1, 0.2]
        }
    else:
        # 전체 조합 (상세 분석용)
        param_grid = {
            'learning_rate': [0.01, 0.001, 0.0001],
            'hidden_units': [32, 64, 128],
            'dropout_rate': [0.1, 0.2, 0.3]
        }
    
    # 모든 조합 생성 (itertools.product 사용)
    combinations = list(product(
        param_grid['learning_rate'],
        param_grid['hidden_units'], 
        param_grid['dropout_rate']
    ))
    
    print(f"총 {len(combinations)}개 조합 테스트 예정...")
    
    results = []
    
    for i, (lr, units, dropout) in enumerate(combinations, 1):
        print(f"\\n[{i}/{len(combinations)}] lr={lr}, units={units}, dropout={dropout}")
        
        try:
            model, error = tuner.dl_models.create_safe_classification_model(
                input_dim=3, n_clusters=5,
                hidden_units=units, dropout_rate=dropout, learning_rate=lr
            )
            
            if error:
                print(f"❌ 모델 생성 실패: {error}")
                continue
                
            history, error = tuner.dl_models.train_model_with_progress(
                model, X_train, y_train, X_test, y_test,
                epochs=50, progress_bar=None, status_text=None
            )
            
            if error:
                print(f"❌ 훈련 실패: {error}")
                continue
                
            val_accuracy = max(history.history['val_accuracy'])
            train_accuracy = max(history.history['accuracy'])
            
            results.append({
                'learning_rate': lr,
                'hidden_units': units,
                'dropout_rate': dropout,
                'val_accuracy': val_accuracy,
                'train_accuracy': train_accuracy,
                'overfitting_gap': train_accuracy - val_accuracy
            })
            
            print(f"✅ 검증 정확도: {val_accuracy:.4f}, 과적합 갭: {train_accuracy - val_accuracy:.4f}")
            
        except Exception as e:
            print(f"❌ 예외 발생: {str(e)}")
            continue
    
    if not results:
        print("❌ 모든 조합 테스트 실패")
        return None
        
    # 최적 조합 찾기
    best_result = max(results, key=lambda x: x['val_accuracy'])
    print(f"\\n🏆 그리드 서치 최적 조합:")
    print(f"Learning Rate: {best_result['learning_rate']}")
    print(f"Hidden Units: {best_result['hidden_units']}")  
    print(f"Dropout Rate: {best_result['dropout_rate']}")
    print(f"검증 정확도: {best_result['val_accuracy']:.4f}")
    print(f"과적합 갭: {best_result['overfitting_gap']:.4f}")
    
    # 결과를 DataFrame으로 정리
    results_df = pd.DataFrame(results)
    print(f"\\n📊 상위 5개 조합:")
    print(results_df.nlargest(5, 'val_accuracy')[['learning_rate', 'hidden_units', 'dropout_rate', 'val_accuracy', 'overfitting_gap']])
    
    return best_result, results_df


# 실행 함수들
def main():
    """메인 실행 함수"""
    print("🎯 하이퍼파라미터 튜닝 실험을 선택하세요:")
    print("1. 단계별 튜닝 (권장)")
    print("2. 그리드 서치 (제한된 조합)")  
    print("3. 그리드 서치 (전체 조합)")
    
    choice = input("\\n선택 (1/2/3): ").strip()
    
    if choice == "1":
        tuner = HyperparameterTuner()
        results = tuner.run_complete_tuning()
        
    elif choice == "2":
        results = run_grid_search_experiment(limited_combinations=True)
        
    elif choice == "3":
        results = run_grid_search_experiment(limited_combinations=False)
        
    else:
        print("잘못된 선택입니다.")
        return
    
    if results:
        print("\\n✅ 실험 완료!")
    else:
        print("\\n❌ 실험 실패!")


if __name__ == "__main__":
    # TensorFlow 경고 메시지 줄이기
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    main()
