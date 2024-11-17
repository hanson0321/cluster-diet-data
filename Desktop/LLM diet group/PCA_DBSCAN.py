# 匯入必要的庫
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# 1. 讀取 CSV 資料
data = pd.read_csv('diet_data.csv')  # 請將 'diet_data.csv' 替換為您的 CSV 檔案名

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

# 5. 自動測試超參數，找到較大的 eps 值

from sklearn.neighbors import NearestNeighbors

# 定義超參數的測試範圍
min_samples_range = [5, 7, 10]
eps_range = np.arange(5.0, 7.5, 0.5)  # 測試 eps=5.0, 5.5, 6.0, 6.5, 7.0

best_silhouette = -1
best_eps = None
best_min_samples = None
best_labels = None

for min_samples in min_samples_range:
    # 計算最近鄰
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(X_pca)
    distances, indices = neighbors_fit.kneighbors(X_pca)
    
    # 對距離進行排序
    distances = np.sort(distances[:, min_samples - 1], axis=0)
    
    # 根據膝點法則，這裡我們略過繪圖，直接嘗試不同的 eps 值
    for eps_value in eps_range:
        # 執行 DBSCAN 聚類
        dbscan = DBSCAN(eps=eps_value, min_samples=min_samples)
        labels = dbscan.fit_predict(X_pca)
        
        # 計算生成的群組數量
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        # 排除只有一個群組或全部都是噪音的情況
        if n_clusters <= 1 or n_clusters >= len(X_pca):
            continue
        
        # 計算 Silhouette Score（排除噪音點）
        if n_clusters > 1 and np.count_nonzero(labels != -1) > 1:
            sil_score = silhouette_score(X_pca[labels != -1], labels[labels != -1])
            print(f"min_samples={min_samples}, eps={eps_value}, 群組數量={n_clusters}, Silhouette Score={sil_score}")
            
            # 更新最佳參數
            if sil_score > best_silhouette:
                best_silhouette = sil_score
                best_eps = eps_value
                best_min_samples = min_samples
                best_labels = labels.copy()

# 輸出最佳結果
print(f"\n最佳參數：min_samples={best_min_samples}, eps={best_eps}, Silhouette Score={best_silhouette}")

# 使用最佳參數進行最終的聚類
dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
labels = dbscan.fit_predict(X_pca)

# 8. 將群組標籤添加到原始資料中
data_filled['Cluster'] = labels

# 9. 生成聚類細節

# 9.1 計算每個群組的樣本數量
cluster_counts = data_filled['Cluster'].value_counts().sort_index()
print("\n每個群組的樣本數量（-1 表示噪音點）：")
print(cluster_counts)

# 9.2 計算每個群組的特徵平均值（排除噪音點）
cluster_profiles = data_filled[data_filled['Cluster'] != -1].groupby('Cluster').mean()
print("\n群組特徵平均值（不包括噪音點）：")
print(cluster_profiles)

# 9.3 如果需要更詳細的統計，例如中位數或標準差，可以計算
cluster_median = data_filled[data_filled['Cluster'] != -1].groupby('Cluster').median()
cluster_std = data_filled[data_filled['Cluster'] != -1].groupby('Cluster').std()

print("\n群組特徵中位數（不包括噪音點）：")
print(cluster_median)

print("\n群組特徵標準差（不包括噪音點）：")
print(cluster_std)

# 10. 可視化聚類結果

# 使用 PCA 的前兩個主成分繪製散點圖
plt.figure(figsize=(10, 6))
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for k, col in zip(unique_labels, colors):
    class_member_mask = (labels == k)
    if k == -1:
        # 噪音點顏色設為黑色
        col = 'black'
        label_name = '噪音點'
    else:
        label_name = f'群組 {k}'
    
    xy = X_pca[class_member_mask]
    plt.scatter(xy[:, 0], xy[:, 1], c=[col], label=label_name, edgecolors='k', s=20)

plt.title('DBSCAN 聚類結果')
plt.xlabel('主成分 1')
plt.ylabel('主成分 2')
plt.legend()
plt.show()
