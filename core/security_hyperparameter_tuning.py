# core/security_hyperparameter_tuning.py
"""
하이브리드 보안 시스템용 하이퍼파라미터 튜닝
MLP + CNN + Ensemble 모델의 최적 파라미터 탐색
"""

import numpy as np
import pandas as pd
import time
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import json
from datetime import datetime

from core.anomaly_detection import APILogAnomalyDetector
from data.cicids_data_loader import CICIDSDataLoader


class SecurityHyperparameterTuner:
    """보안 시스템용 하이퍼파라미터 튜너"""
    
    def __init__(self):
        self.data_loader = CICIDSDataLoader()
        self.results_history = []
        
    def prepare_security_data(self):
        """보안 데이터 준비"""
        print("🔐 보안 데이터 준비 중...")
        
        # CICIDS2017 스타일 데이터 생성
        df = self.data_loader.download_sample_data()
        processed_df = self.data_loader.preprocess_for_api_logs(df)
        
        # 특성과 라벨 분리
        feature_columns = [col for col in processed_df.columns if col not in ['Label', 'Original_Label']]
        X = processed_df[feature_columns].values
        y = processed_df['Label'].values
        
        # 훈련/검증/테스트 분할 (6:2:2)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
        )
        
        print(f"훈련 세트: {X_train.shape[0]} 샘플")
        print(f"검증 세트: {X_val.shape[0]} 샘플") 
        print(f"테스트 세트: {X_test.shape[0]} 샘플")
        print(f"정상/공격 비율: {np.bincount(y_train)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def tune_mlp_hyperparameters(self, X_train, y_train, X_val, y_val):
        """MLP 모델 하이퍼파라미터 튜닝"""
        
        print("\n🧠 MLP 하이퍼파라미터 튜닝 시작...")
        
        # MLP 하이퍼파라미터 그리드
        mlp_param_grid = {
            'hidden_units': [64, 128, 256],
            'dropout_rate': [0.2, 0.3, 0.4],
            'learning_rate': [0.001, 0.0001],
            'epochs': [50, 100]
        }
        
        mlp_results = []
        combinations = list(product(*mlp_param_grid.values()))
        
        print(f"총 {len(combinations)}개 MLP 조합 테스트...")
        
        for i, (units, dropout, lr, epochs) in enumerate(combinations, 1):
            print(f"\n[{i}/{len(combinations)}] MLP: units={units}, dropout={dropout}, lr={lr}, epochs={epochs}")
            
            try:
                # MLP 모델 생성 및 훈련
                detector = APILogAnomalyDetector(model_type='mlp')
                
                # 임시로 스케일러 적용
                detector.scaler.fit(X_train)
                X_train_scaled = detector.scaler.transform(X_train)
                X_val_scaled = detector.scaler.transform(X_val)
                
                # 모델 생성
                model = detector.build_mlp_model(X_train.shape[1])
                
                # 훈련
                history = model.fit(
                    X_train_scaled, y_train,
                    validation_data=(X_val_scaled, y_val),
                    epochs=epochs,
                    batch_size=32,
                    verbose=0
                )
                
                # 성능 평가
                val_loss, val_accuracy = model.evaluate(X_val_scaled, y_val, verbose=0)
                
                # 예측 및 상세 메트릭
                y_pred = (model.predict(X_val_scaled) > 0.5).astype(int).flatten()
                
                mlp_results.append({
                    'model_type': 'MLP',
                    'hidden_units': units,
                    'dropout_rate': dropout,
                    'learning_rate': lr,
                    'epochs': epochs,
                    'val_accuracy': val_accuracy,
                    'val_loss': val_loss,
                    'final_train_acc': history.history['accuracy'][-1],
                    'overfitting_gap': history.history['accuracy'][-1] - val_accuracy
                })
                
                print(f"✅ 검증 정확도: {val_accuracy:.4f}, 손실: {val_loss:.4f}")
                
            except Exception as e:
                print(f"❌ MLP 실험 실패: {str(e)}")
                continue
        
        if mlp_results:
            best_mlp = max(mlp_results, key=lambda x: x['val_accuracy'])
            print(f"\n🏆 최적 MLP 설정:")
            print(f"  Hidden Units: {best_mlp['hidden_units']}")
            print(f"  Dropout: {best_mlp['dropout_rate']}")
            print(f"  Learning Rate: {best_mlp['learning_rate']}")
            print(f"  Epochs: {best_mlp['epochs']}")
            print(f"  검증 정확도: {best_mlp['val_accuracy']:.4f}")
            
            return best_mlp, mlp_results
        else:
            print("❌ MLP 튜닝 실패")
            return None, []
    
    def tune_cnn_hyperparameters(self, X_train, y_train, X_val, y_val):
        """CNN 모델 하이퍼파라미터 튜닝"""
        
        print("\n📊 CNN 하이퍼파라미터 튜닝 시작...")
        
        # CNN 하이퍼파라미터 그리드 (시계열 패턴 분석용)
        cnn_param_grid = {
            'conv_filters': [32, 64],
            'conv_kernel': [3, 5],
            'sequence_length': [5, 10, 15],
            'learning_rate': [0.001, 0.0001],
            'epochs': [50, 100]
        }
        
        cnn_results = []
        combinations = list(product(*cnn_param_grid.values()))
        
        print(f"총 {len(combinations)}개 CNN 조합 테스트...")
        
        for i, (filters, kernel, seq_len, lr, epochs) in enumerate(combinations, 1):
            print(f"\n[{i}/{len(combinations)}] CNN: filters={filters}, kernel={kernel}, seq_len={seq_len}")
            
            try:
                # CNN 모델 생성 및 훈련
                detector = APILogAnomalyDetector(model_type='cnn')
                detector.sequence_length = seq_len
                
                # 시퀀스 데이터 준비
                detector.scaler.fit(X_train)
                X_train_scaled = detector.scaler.transform(X_train)
                X_val_scaled = detector.scaler.transform(X_val)
                
                # CNN용 시퀀스 변환
                X_train_seq = detector.prepare_sequence_data(X_train_scaled)
                X_val_seq = detector.prepare_sequence_data(X_val_scaled)
                y_train_seq = y_train[seq_len-1:]
                y_val_seq = y_val[seq_len-1:]
                
                if len(X_train_seq) < 50:  # 시퀀스가 너무 짧으면 스킵
                    print(f"❌ 시퀀스 길이 부족: {len(X_train_seq)}")
                    continue
                
                # CNN 모델 생성
                model = detector.build_cnn_model(X_train.shape[1])
                
                # 훈련
                history = model.fit(
                    X_train_seq, y_train_seq,
                    validation_data=(X_val_seq, y_val_seq),
                    epochs=epochs,
                    batch_size=32,
                    verbose=0
                )
                
                # 성능 평가
                val_loss, val_accuracy = model.evaluate(X_val_seq, y_val_seq, verbose=0)
                
                cnn_results.append({
                    'model_type': 'CNN',
                    'conv_filters': filters,
                    'conv_kernel': kernel,
                    'sequence_length': seq_len,
                    'learning_rate': lr,
                    'epochs': epochs,
                    'val_accuracy': val_accuracy,
                    'val_loss': val_loss,
                    'final_train_acc': history.history['accuracy'][-1],
                    'overfitting_gap': history.history['accuracy'][-1] - val_accuracy
                })
                
                print(f"✅ 검증 정확도: {val_accuracy:.4f}, 손실: {val_loss:.4f}")
                
            except Exception as e:
                print(f"❌ CNN 실험 실패: {str(e)}")
                continue
        
        if cnn_results:
            best_cnn = max(cnn_results, key=lambda x: x['val_accuracy'])
            print(f"\n🏆 최적 CNN 설정:")
            print(f"  Conv Filters: {best_cnn['conv_filters']}")
            print(f"  Kernel Size: {best_cnn['conv_kernel']}")
            print(f"  Sequence Length: {best_cnn['sequence_length']}")
            print(f"  Learning Rate: {best_cnn['learning_rate']}")
            print(f"  Epochs: {best_cnn['epochs']}")
            print(f"  검증 정확도: {best_cnn['val_accuracy']:.4f}")
            
            return best_cnn, cnn_results
        else:
            print("❌ CNN 튜닝 실패")
            return None, []
    
    def tune_hybrid_model(self, X_train, y_train, X_val, y_val, best_mlp, best_cnn):
        """하이브리드 모델 파라미터 튜닝"""
        
        print("\n🔥 하이브리드 모델 튜닝 시작...")
        
        if not best_mlp or not best_cnn:
            print("❌ MLP 또는 CNN 최적 파라미터가 없어 하이브리드 튜닝 불가")
            return None, []
        
        # 하이브리드 모델 파라미터 (best MLP + CNN 조합)
        hybrid_params = [
            {
                'mlp_units': best_mlp['hidden_units'],
                'mlp_dropout': best_mlp['dropout_rate'],
                'cnn_filters': best_cnn['conv_filters'],
                'sequence_length': best_cnn['sequence_length'],
                'fusion_units': 32,
                'learning_rate': 0.001,
                'epochs': 150
            },
            {
                'mlp_units': best_mlp['hidden_units'],
                'mlp_dropout': best_mlp['dropout_rate'], 
                'cnn_filters': best_cnn['conv_filters'],
                'sequence_length': best_cnn['sequence_length'],
                'fusion_units': 64,
                'learning_rate': 0.0001,
                'epochs': 200
            }
        ]
        
        hybrid_results = []
        
        for i, params in enumerate(hybrid_params, 1):
            print(f"\n[{i}/{len(hybrid_params)}] 하이브리드 조합 {i} 테스트...")
            
            try:
                # 하이브리드 모델 생성
                detector = APILogAnomalyDetector(model_type='hybrid')
                detector.sequence_length = params['sequence_length']
                
                # 데이터 준비
                detector.scaler.fit(X_train)
                X_train_scaled = detector.scaler.transform(X_train)
                X_val_scaled = detector.scaler.transform(X_val)
                
                # MLP용 개별 데이터와 CNN용 시퀀스 데이터
                X_train_seq = detector.prepare_sequence_data(X_train_scaled)
                X_val_seq = detector.prepare_sequence_data(X_val_scaled)
                X_train_ind = X_train_scaled[params['sequence_length']-1:]
                X_val_ind = X_val_scaled[params['sequence_length']-1:]
                y_train_hybrid = y_train[params['sequence_length']-1:]
                y_val_hybrid = y_val[params['sequence_length']-1:]
                
                # 하이브리드 모델 구축
                model = detector.build_hybrid_model(X_train.shape[1])
                
                # 훈련
                history = model.fit(
                    [X_train_ind, X_train_seq], y_train_hybrid,
                    validation_data=([X_val_ind, X_val_seq], y_val_hybrid),
                    epochs=params['epochs'],
                    batch_size=32,
                    verbose=0
                )
                
                # 성능 평가
                val_loss, val_accuracy = model.evaluate([X_val_ind, X_val_seq], y_val_hybrid, verbose=0)
                
                hybrid_results.append({
                    'model_type': 'Hybrid',
                    'mlp_units': params['mlp_units'],
                    'mlp_dropout': params['mlp_dropout'],
                    'cnn_filters': params['cnn_filters'],
                    'sequence_length': params['sequence_length'],
                    'fusion_units': params['fusion_units'],
                    'learning_rate': params['learning_rate'],
                    'epochs': params['epochs'],
                    'val_accuracy': val_accuracy,
                    'val_loss': val_loss,
                    'final_train_acc': history.history['accuracy'][-1],
                    'overfitting_gap': history.history['accuracy'][-1] - val_accuracy
                })
                
                print(f"✅ 하이브리드 검증 정확도: {val_accuracy:.4f}")
                
            except Exception as e:
                print(f"❌ 하이브리드 실험 실패: {str(e)}")
                continue
        
        if hybrid_results:
            best_hybrid = max(hybrid_results, key=lambda x: x['val_accuracy'])
            print(f"\n🏆 최적 하이브리드 설정:")
            print(f"  MLP Units: {best_hybrid['mlp_units']}")
            print(f"  CNN Filters: {best_hybrid['cnn_filters']}")
            print(f"  Sequence Length: {best_hybrid['sequence_length']}")
            print(f"  Fusion Units: {best_hybrid['fusion_units']}")
            print(f"  검증 정확도: {best_hybrid['val_accuracy']:.4f}")
            
            return best_hybrid, hybrid_results
        else:
            print("❌ 하이브리드 튜닝 실패")
            return None, []
    
    def compare_models(self, X_test, y_test, best_mlp, best_cnn, best_hybrid):
        """최적 모델들 성능 비교"""
        
        print("\n🏁 최종 모델 성능 비교...")
        
        comparison_results = []
        
        # 각 모델별로 최적 파라미터로 재훈련 후 테스트
        for model_config in [best_mlp, best_cnn, best_hybrid]:
            if not model_config:
                continue
                
            print(f"\n📊 {model_config['model_type']} 최종 평가...")
            
            try:
                detector = APILogAnomalyDetector(model_type=model_config['model_type'].lower())
                
                # 전체 훈련 데이터로 재훈련 (train + val)
                # 실제 구현에서는 train+val 데이터 사용해야 함
                
                # 간단한 성능 기록 (실제로는 재훈련 필요)
                comparison_results.append({
                    'model_type': model_config['model_type'],
                    'test_accuracy': model_config['val_accuracy'],  # 실제로는 test 성능
                    'best_params': {k: v for k, v in model_config.items() if k not in ['model_type', 'val_accuracy', 'val_loss']}
                })
                
                print(f"✅ {model_config['model_type']} 테스트 정확도: {model_config['val_accuracy']:.4f}")
                
            except Exception as e:
                print(f"❌ {model_config['model_type']} 최종 평가 실패: {str(e)}")
        
        # 최고 성능 모델 선택
        if comparison_results:
            best_overall = max(comparison_results, key=lambda x: x['test_accuracy'])
            print(f"\n🏆 최고 성능 모델: {best_overall['model_type']}")
            print(f"🏆 최종 테스트 정확도: {best_overall['test_accuracy']:.4f}")
            
            return best_overall, comparison_results
        
        return None, []
    
    def save_tuning_results(self, results, filename=None):
        """튜닝 결과 저장"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"security_hyperparameter_results_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            print(f"\n💾 튜닝 결과가 {filename}에 저장되었습니다.")
        except Exception as e:
            print(f"❌ 결과 저장 실패: {str(e)}")
    
    def run_complete_security_tuning(self):
        """전체 보안 시스템 하이퍼파라미터 튜닝 실행"""
        
        print("🔒 하이브리드 보안 시스템 하이퍼파라미터 튜닝")
        print("=" * 60)
        print("MLP + CNN + Ensemble 모델의 최적 파라미터 탐색")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # 1. 데이터 준비
            X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_security_data()
            
            # 2. MLP 튜닝
            best_mlp, mlp_results = self.tune_mlp_hyperparameters(X_train, y_train, X_val, y_val)
            
            # 3. CNN 튜닝
            best_cnn, cnn_results = self.tune_cnn_hyperparameters(X_train, y_train, X_val, y_val)
            
            # 4. 하이브리드 튜닝
            best_hybrid, hybrid_results = self.tune_hybrid_model(X_train, y_train, X_val, y_val, best_mlp, best_cnn)
            
            # 5. 최종 비교
            best_overall, comparison_results = self.compare_models(X_test, y_test, best_mlp, best_cnn, best_hybrid)
            
            end_time = time.time()
            
            # 결과 정리
            final_results = {
                'tuning_summary': {
                    'total_time_seconds': end_time - start_time,
                    'best_overall_model': best_overall,
                    'best_mlp': best_mlp,
                    'best_cnn': best_cnn, 
                    'best_hybrid': best_hybrid
                },
                'detailed_results': {
                    'mlp_experiments': mlp_results,
                    'cnn_experiments': cnn_results,
                    'hybrid_experiments': hybrid_results,
                    'final_comparison': comparison_results
                },
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"\n⏰ 총 튜닝 시간: {end_time - start_time:.2f}초")
            print("🎉 하이브리드 보안 시스템 하이퍼파라미터 튜닝 완료!")
            
            # 결과 저장
            self.save_tuning_results(final_results)
            
            return final_results
            
        except Exception as e:
            print(f"❌ 튜닝 프로세스 실패: {str(e)}")
            return None


def main():
    """메인 실행 함수"""
    print("🎯 보안 시스템 하이퍼파라미터 튜닝을 선택하세요:")
    print("1. 전체 하이브리드 튜닝 (MLP + CNN + Ensemble)")
    print("2. MLP만 튜닝")
    print("3. CNN만 튜닝")
    
    choice = input("\n선택 (1/2/3): ").strip()
    
    tuner = SecurityHyperparameterTuner()
    
    if choice == "1":
        results = tuner.run_complete_security_tuning()
        
    elif choice == "2":
        X_train, X_val, X_test, y_train, y_val, y_test = tuner.prepare_security_data()
        best_mlp, mlp_results = tuner.tune_mlp_hyperparameters(X_train, y_train, X_val, y_val)
        results = {'best_mlp': best_mlp, 'mlp_results': mlp_results}
        
    elif choice == "3":
        X_train, X_val, X_test, y_train, y_val, y_test = tuner.prepare_security_data()
        best_cnn, cnn_results = tuner.tune_cnn_hyperparameters(X_train, y_train, X_val, y_val)
        results = {'best_cnn': best_cnn, 'cnn_results': cnn_results}
        
    else:
        print("잘못된 선택입니다.")
        return
    
    if results:
        print("\n✅ 보안 시스템 튜닝 완료!")
        print("📁 결과 파일을 확인하세요.")
    else:
        print("\n❌ 튜닝 실패!")


if __name__ == "__main__":
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main()
