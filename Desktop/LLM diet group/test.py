# 匯入必要的庫
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# 新增匯入 UMAP 的庫
import umap.umap_ as umap

# 忽略無法顯示中文字的警告
import warnings
warnings.filterwarnings("ignore", message="Glyph.*missing from current font")

# 1. 讀取 CSV 資料
data = pd.read_csv('diet_data.csv')  # 請將 'diet_data.csv' 替換為您的 CSV 檔案名

# 2. 處理缺失值
data_filled = data.fillna(0)

# 3. 資料標準化
features = data_filled.columns[1:]  # 假設第一列是 ID
X = data_filled[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. 定義 UMAP 的 n_components 範圍
n_components_list = [5, 10, 15, 20, 25]  # 您可以根據需要調整

# 固定 UMAP 和 DBSCAN 的其他參數
n_neighbors = 15
min_dist = 0.1
eps_value = 7
min_samples = 15

best_score = -1
best_n_components = None
best_labels = None
best_embedding = None

# 5. 測試不同的 n_components
for n_components in n_components_list:
    # 使用 UMAP 進行降維
    umap_reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42
    )
    X_umap = umap_reducer.fit_transform(X_scaled)
    
    # 使用固定的 DBSCAN 參數進行聚類
    dbscan = DBSCAN(eps=eps_value, min_samples=min_samples)
    labels = dbscan.fit_predict(X_umap)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    # 只考慮有超過一個群組的結果
    if n_clusters > 1 and n_clusters < len(X_umap):
        # 計算 Silhouette Score（排除噪音點）
        if np.count_nonzero(labels != -1) > 1:
            score = silhouette_score(X_umap[labels != -1], labels[labels != -1])
            
            print(f"n_components={n_components}; 群組數量={n_clusters}, Silhouette Score={score}")
            
            if score > best_score:
                best_score = score
                best_n_components = n_components
                best_labels = labels
                best_embedding = X_umap

# 6. 輸出最佳結果
print("\n最佳 n_components 值：", best_n_components)
print(f"最高的 Silhouette Score: {best_score}")

# 7. 使用最佳 n_components 的結果進行分析
labels = best_labels
X_umap = best_embedding

# 將群組標籤添加到原始資料中
data_filled['Cluster'] = labels

# 計算每個群組的樣本數量
cluster_counts = data_filled['Cluster'].value_counts().sort_index()
print("\n每個群組的樣本數量（-1 表示噪音點）：")
print(cluster_counts)

# 計算每個群組的特徵平均值（排除噪音點）
cluster_profiles = data_filled[data_filled['Cluster'] != -1].groupby('Cluster').mean()
print("\n群組特徵平均值（不包括噪音點）：")
print(cluster_profiles)

# 8. 可視化聚類結果（如果 n_components >= 2）
if best_embedding.shape[1] >= 2:
    plt.figure(figsize=(10, 6))
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for k, col in zip(unique_labels, colors):
        class_member_mask = (labels == k)
        if k == -1:
            col = 'black'
            label_name = '噪音點'
        else:
            label_name = f'群組 {k}'
        
        xy = X_umap[class_member_mask]
        plt.scatter(xy[:, 0], xy[:, 1], c=[col], label=label_name, edgecolors='k', s=20)
    
    plt.title('DBSCAN 聚類結果（使用 UMAP 降維）')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend()
    plt.show()
else:
    print("無法可視化，因為降維後的維度小於 2。")
