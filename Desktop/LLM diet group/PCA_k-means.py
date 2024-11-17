# 匯入必要的庫
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. 讀取 CSV 數據
data = pd.read_csv('diet_data.csv')  # 請將 'your_data.csv' 替換為您的 CSV 文件名

# 假設第一列是 ID，其餘列是特徵
# 如果 ID 列無用，請將其刪除
# data = data.drop('ID', axis=1)

# 2. 處理缺失值

# 將缺失值填充為零
data_filled = data.fillna(0)

# 3. 資料標準化

# 提取特徵列
features = data_filled.columns[1:]  # 假設第一列是 ID
X = data_filled[features]

# 應用 Z 分數標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. 使用 PCA 進行降維

# 設定保留的主成分數量，例如保留解釋 95% 變異的主成分
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA 降維後的特徵數量：{X_pca.shape[1]}")

# 5. 分群分析

# 嘗試不同的 K 值，並計算 Silhouette Score
silhouette_scores = []
K = range(2, 11)  # 從 2 到 10 個群組
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    silhouette_scores.append(score)
    print(f"K={k} 的 Silhouette Score：{score}")

# 找出最高的 Silhouette Score 對應的 K 值
optimal_k = K[np.argmax(silhouette_scores)]
print(f"\n最佳的群組數量為 K={optimal_k}")

# 使用最佳 K 值進行 K-均值聚類
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
labels = kmeans.fit_predict(X_pca)

# 將群組標籤添加到原始數據中
data_filled['Cluster'] = labels

# 6. 生成聚類細節

# 計算每個群組的樣本數量
cluster_counts = data_filled['Cluster'].value_counts().sort_index()
print("\n每個群組的樣本數量：")
print(cluster_counts)

# 計算每個群組的特徵平均值
cluster_profiles = data_filled.groupby('Cluster').mean()
print("\n群組特徵平均值：")
print(cluster_profiles)

# 如果需要更詳細的統計，例如中位數或標準差，可以計算
cluster_median = data_filled.groupby('Cluster').median()
cluster_std = data_filled.groupby('Cluster').std()

print("\n群組特徵中位數：")
print(cluster_median)

print("\n群組特徵標準差：")
print(cluster_std)
