# 匯入必要的庫
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# 1. 讀取 CSV 數據
data = pd.read_csv('diet_data.csv')  # 請將 'diet_data.csv' 替換為您的 CSV 文件名

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

# 設定保留的主成分數量，例如保留解釋 90% 變異的主成分
pca = PCA(n_components=0.90)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA 降維後的特徵數量：{X_pca.shape[1]}")

# 5. 執行層次式聚類

# 5.1 使用 scipy 計算連結矩陣
linked = linkage(X_pca, method='ward')

# 5.2 繪製樹狀圖（可選）
plt.figure(figsize=(10, 7))
dendrogram(linked, truncate_mode='level', p=5)  # 只顯示層級較高的結點
plt.title('樹狀圖')
plt.xlabel('樣本編號')
plt.ylabel('距離')
plt.show()

# 5.3 確定群組數量，並進行聚類
# 假設選擇 4 個群組（您可以根據樹狀圖調整此值）
optimal_k = 4
cluster = AgglomerativeClustering(n_clusters=optimal_k, affinity='euclidean', linkage='ward')
labels = cluster.fit_predict(X_pca)

# 6. 評估聚類效果

sil_score = silhouette_score(X_pca, labels)
print(f"\n最佳群組數量 K={optimal_k} 的 Silhouette Score：{sil_score}")

# 7. 將群組標籤添加到原始數據中
data_filled['Cluster'] = labels

# 8. 生成聚類細節

# 8.1 計算每個群組的樣本數量
cluster_counts = data_filled['Cluster'].value_counts().sort_index()
print("\n每個群組的樣本數量：")
print(cluster_counts)

# 8.2 計算每個群組的特徵平均值
cluster_profiles = data_filled.groupby('Cluster').mean()
print("\n群組特徵平均值：")
print(cluster_profiles)

# 8.3 如果需要更詳細的統計，例如中位數或標準差，可以計算
cluster_median = data_filled.groupby('Cluster').median()
cluster_std = data_filled.groupby('Cluster').std()

print("\n群組特徵中位數：")
print(cluster_median)

print("\n群組特徵標準差：")
print(cluster_std)
