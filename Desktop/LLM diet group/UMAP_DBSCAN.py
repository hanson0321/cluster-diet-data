
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# 新增匯入 UMAP 的庫
import umap.umap_ as umap

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

# 4. 使用 UMAP 進行降維

# 設定 UMAP 的參數，例如降至 2 維
umap_reducer = umap.UMAP(n_components=10, random_state=42)
X_umap = umap_reducer.fit_transform(X_scaled)
print(f"UMAP 降維後的形狀：{X_umap.shape}")

# 5. 執行 DBSCAN 聚類

# 固定超參數
min_samples = 10
eps_value = 7.0

# 執行 DBSCAN 聚類
dbscan = DBSCAN(eps=eps_value, min_samples=min_samples)
labels = dbscan.fit_predict(X_umap)

# 6. 評估聚類效果

# 計算生成的群組數量
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"\n生成的群組數量：{n_clusters}")

# 計算 Silhouette Score（排除噪音點）
if n_clusters > 1 and np.count_nonzero(labels != -1) > 1:
    sil_score = silhouette_score(X_umap[labels != -1], labels[labels != -1])
    print(f"聚類的 Silhouette Score：{sil_score}")
else:
    print("無法計算 Silhouette Score，因為群組數量少於 2 或沒有足夠的非噪音點。")

# 7. 將群組標籤添加到原始資料中
data_filled['Cluster'] = labels

# 8. 生成聚類細節

# 8.1 計算每個群組的樣本數量
cluster_counts = data_filled['Cluster'].value_counts().sort_index()
print("\n每個群組的樣本數量（-1 表示噪音點）：")
print(cluster_counts)

# 8.2 計算每個群組的特徵平均值（排除噪音點）
cluster_profiles = data_filled[data_filled['Cluster'] != -1].groupby('Cluster').mean()
print("\n群組特徵平均值（不包括噪音點）：")
print(cluster_profiles)

# 8.3 如果需要更詳細的統計，例如中位數或標準差，可以計算
cluster_median = data_filled[data_filled['Cluster'] != -1].groupby('Cluster').median()
cluster_std = data_filled[data_filled['Cluster'] != -1].groupby('Cluster').std()

print("\n群組特徵中位數（不包括噪音點）：")
print(cluster_median)

print("\n群組特徵標準差（不包括噪音點）：")
print(cluster_std)

# 9. 可視化聚類結果

# 使用 UMAP 降維後的資料繪製散點圖
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
    
    xy = X_umap[class_member_mask]
    plt.scatter(xy[:, 0], xy[:, 1], c=[col], label=label_name, edgecolors='k', s=20)

plt.title('DBSCAN 聚類結果（使用 UMAP 降維）')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.legend()
plt.show()
