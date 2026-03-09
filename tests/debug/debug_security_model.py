"""
보안 모델 성능 이상 진단 도구

정확도 1.0이 나오는 원인을 분석하는 스크립트
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from data.loaders.unified_security_loader import UnifiedSecurityLoader

def diagnose_perfect_performance():
    """완벽한 성능의 원인 분석"""
    print("=== 보안 모델 성능 이상 진단 ===")
    
    # 1. 샘플 데이터 생성
    loader = UnifiedSecurityLoader()
    data = loader.generate_sample_data(total_samples=1000, attack_ratio=0.6)
    
    print(f"데이터 크기: {data.shape}")
    print(f"라벨 분포:\n{data['Label'].value_counts()}")
    
    # 2. 특성별 분리 정도 분석
    analyze_feature_separation(data)
    
    # 3. 클러스터 분석
    analyze_clustering_patterns(data)
    
    # 4. 데이터 복잡성 측정
    measure_data_complexity(data)
    
    # 5. 개선 방안 제안
    suggest_improvements()

def analyze_feature_separation(data):
    """특성별 분리 정도 분석"""
    print("\n=== 특성별 분리 정도 분석 ===")
    
    numeric_features = data.select_dtypes(include=[np.number]).columns[:5]
    
    separation_scores = []
    for feature in numeric_features:
        normal_data = data[data['Label'] == 'BENIGN'][feature]
        attack_data = data[data['Label'] != 'BENIGN'][feature]
        
        # 평균 차이 비율
        if normal_data.mean() != 0:
            ratio = abs(attack_data.mean() / normal_data.mean())
        else:
            ratio = float('inf')
        
        # 분포 겹침 정도 (Wasserstein distance 근사)
        overlap = calculate_distribution_overlap(normal_data, attack_data)
        
        separation_scores.append({
            'feature': feature,
            'mean_ratio': ratio,
            'overlap': overlap,
            'normal_mean': normal_data.mean(),
            'attack_mean': attack_data.mean(),
            'normal_std': normal_data.std(),
            'attack_std': attack_data.std()
        })
        
        print(f"{feature}:")
        print(f"  정상 평균: {normal_data.mean():.2f} ± {normal_data.std():.2f}")
        print(f"  공격 평균: {attack_data.mean():.2f} ± {attack_data.std():.2f}")
        print(f"  비율: {ratio:.2f}x, 겹침: {overlap:.3f}")
    
    # 가장 분리가 잘 되는 특성들
    separation_df = pd.DataFrame(separation_scores)
    separation_df = separation_df.sort_values('mean_ratio', ascending=False)
    
    print("\n가장 분리가 잘 되는 특성들:")
    print(separation_df[['feature', 'mean_ratio', 'overlap']].head())
    
    return separation_df

def calculate_distribution_overlap(dist1, dist2):
    """두 분포의 겹침 정도 계산 (0: 완전 분리, 1: 완전 겹침)"""
    try:
        # 히스토그램 기반 겹침 계산
        min_val = min(dist1.min(), dist2.min())
        max_val = max(dist1.max(), dist2.max())
        
        bins = np.linspace(min_val, max_val, 50)
        hist1, _ = np.histogram(dist1, bins=bins, density=True)
        hist2, _ = np.histogram(dist2, bins=bins, density=True)
        
        # 겹치는 영역 계산
        overlap = np.sum(np.minimum(hist1, hist2)) / np.sum(np.maximum(hist1, hist2))
        return overlap
    except:
        return 0.0

def analyze_clustering_patterns(data):
    """클러스터링 패턴 분석"""
    print("\n=== 클러스터링 패턴 분석 ===")
    
    # 수치형 특성만 선택
    numeric_features = data.select_dtypes(include=[np.number]).columns
    X = data[numeric_features].values
    
    # PCA로 차원 축소
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # 라벨별 분리도 측정
    labels = data['Label'].values
    unique_labels = np.unique(labels)
    
    print("PCA 설명 가능한 분산:")
    print(f"PC1: {pca.explained_variance_ratio_[0]:.3f}")
    print(f"PC2: {pca.explained_variance_ratio_[1]:.3f}")
    print(f"총합: {pca.explained_variance_ratio_.sum():.3f}")
    
    # 클러스터 간 거리 측정
    cluster_distances = {}
    for label in unique_labels:
        mask = labels == label
        centroid = X_pca[mask].mean(axis=0)
        cluster_distances[label] = centroid
    
    # 클러스터 중심 간 거리
    distance_matrix = {}
    for label1 in unique_labels:
        for label2 in unique_labels:
            if label1 != label2:
                dist = np.linalg.norm(
                    cluster_distances[label1] - cluster_distances[label2]
                )
                distance_matrix[f"{label1} vs {label2}"] = dist
    
    print("\n클러스터 중심 간 거리:")
    for pair, distance in distance_matrix.items():
        print(f"{pair}: {distance:.3f}")
    
    return X_pca, cluster_distances

def measure_data_complexity(data):
    """데이터 복잡성 측정"""
    print("\n=== 데이터 복잡성 측정 ===")
    
    # 1. 클래스 불균형
    label_counts = data['Label'].value_counts()
    imbalance_ratio = label_counts.max() / label_counts.min()
    print(f"클래스 불균형 비율: {imbalance_ratio:.2f}")
    
    # 2. 특성 간 상관관계
    numeric_data = data.select_dtypes(include=[np.number])
    correlation_matrix = numeric_data.corr().abs()
    high_corr_pairs = (correlation_matrix > 0.9).sum().sum() - len(correlation_matrix)
    print(f"높은 상관관계 특성 쌍 수 (>0.9): {high_corr_pairs}")
    
    # 3. 특성 분산
    feature_variances = numeric_data.var()
    low_variance_features = (feature_variances < 0.01).sum()
    print(f"낮은 분산 특성 수 (<0.01): {low_variance_features}")
    
    # 4. 이상치 비율
    outlier_counts = []
    for column in numeric_data.columns:
        Q1 = numeric_data[column].quantile(0.25)
        Q3 = numeric_data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = (numeric_data[column] < lower_bound) | (numeric_data[column] > upper_bound)
        outlier_counts.append(outliers.sum())
    
    total_outliers = sum(outlier_counts)
    outlier_ratio = total_outliers / (len(data) * len(numeric_data.columns))
    print(f"이상치 비율: {outlier_ratio:.3f}")
    
    return {
        'imbalance_ratio': imbalance_ratio,
        'high_corr_pairs': high_corr_pairs,
        'low_variance_features': low_variance_features,
        'outlier_ratio': outlier_ratio
    }

def suggest_improvements():
    """개선 방안 제안"""
    print("\n=== 🚀 개선 방안 제안 ===")
    
    improvements = [
        "1. 더 현실적인 데이터 생성:",
        "   - 정상과 공격 패턴 간 겹치는 영역 추가",
        "   - 노이즈와 변동성 증가",
        "   - 다양한 공격 강도 모델링",
        "",
        "2. 데이터 품질 개선:",
        "   - 샘플 수 증가 (10K → 100K)",
        "   - 더 복잡한 특성 관계 모델링",
        "   - 실제 CICIDS2017 데이터 활용",
        "",
        "3. 모델 검증 강화:",
        "   - 교차검증 추가",
        "   - 홀드아웃 검증셋 분리",
        "   - 다양한 메트릭 활용 (AUC, PR-AUC)",
        "",
        "4. 정칙화 및 복잡성 제어:",
        "   - Dropout 비율 증가",
        "   - L1/L2 정규화 추가",
        "   - 더 작은 모델 아키텍처 시도"
    ]
    
    for improvement in improvements:
        print(improvement)

def create_realistic_sample_data(n_samples=10000):
    """더 현실적인 샘플 데이터 생성"""
    print("\n=== 개선된 현실적 데이터 생성 ===")
    
    # 더 복잡하고 현실적인 패턴으로 데이터 생성
    loader = UnifiedSecurityLoader()
    
    # 원본 generate_sample_data를 오버라이드하는 방법
    # (실제 구현에서는 클래스를 상속받아 메서드를 재정의)
    
    print("기존 데이터와 개선된 데이터의 차이:")
    print("- 패턴 간 겹침 증가")
    print("- 노이즈 레벨 상승")
    print("- 더 다양한 공격 강도")
    
    return None

if __name__ == "__main__":
    diagnose_perfect_performance()
