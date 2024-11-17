# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# 新增导入 UMAP 的库
import umap.umap_ as umap

# 忽略无法显示中文字的警告
import warnings
warnings.filterwarnings("ignore", message="Glyph.*missing from current font")

# 1. 读取 CSV 数据
data = pd.read_csv('diet_data.csv')  # 请将 'diet_data.csv' 替换为您的 CSV 文件名

# 2. 处理缺失值
data_filled = data.fillna(0)

# 3. 数据标准化
features = data_filled.columns[1:]  # 假设第一列是 ID
X = data_filled[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. 定义 UMAP 和 DBSCAN 的超参数范围
n_components_list = [5, 10, 15]  # 降维维度数列表
n_neighbors_list = [10, 15, 30]  # UMAP 的邻居数
min_dist_list = [0.0, 0.1, 0.5]  # UMAP 的最小距离

eps_range = np.arange(2, 10, 2)  # DBSCAN 的 eps 参数
min_samples_range = [5, 10, 15]  # DBSCAN 的 min_samples 参数

best_score = -1
best_params = {}
best_labels = None
best_embedding = None

# 5. 自动调参
for n_components in n_components_list:
    for n_neighbors in n_neighbors_list:
        for min_dist in min_dist_list:
            # 使用 UMAP 进行降维
            umap_reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=42
            )
            X_umap = umap_reducer.fit_transform(X_scaled)
            
            # 对于每个降维后的结果，测试不同的 DBSCAN 参数
            for eps_value in eps_range:
                for min_samples in min_samples_range:
                    dbscan = DBSCAN(eps=eps_value, min_samples=min_samples)
                    labels = dbscan.fit_predict(X_umap)
                    
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    
                    # 只考虑有超过一个群组的结果
                    if n_clusters > 1 and n_clusters < len(X_umap):
                        # 计算 Silhouette Score（排除噪声点）
                        if np.count_nonzero(labels != -1) > 1:
                            score = silhouette_score(X_umap[labels != -1], labels[labels != -1])
                            
                            print(f"UMAP: n_components={n_components}, n_neighbors={n_neighbors}, min_dist={min_dist}; "
                                  f"DBSCAN: eps={eps_value}, min_samples={min_samples}; "
                                  f"群组数量={n_clusters}, Silhouette Score={score}")
                            
                            if score > best_score:
                                best_score = score
                                best_params = {
                                    'n_components': n_components,
                                    'n_neighbors': n_neighbors,
                                    'min_dist': min_dist,
                                    'eps': eps_value,
                                    'min_samples': min_samples
                                }
                                best_labels = labels
                                best_embedding = X_umap

# 6. 输出最佳结果
print("\n最佳参数组合：")
print(f"UMAP 参数: n_components={best_params['n_components']}, "
      f"n_neighbors={best_params['n_neighbors']}, min_dist={best_params['min_dist']}")
print(f"DBSCAN 参数: eps={best_params['eps']}, min_samples={best_params['min_samples']}")
print(f"最高的 Silhouette Score: {best_score}")

# 7. 使用最佳参数的结果进行分析
labels = best_labels
X_umap = best_embedding

# 将群组标签添加到原始数据中
data_filled['Cluster'] = labels

# 计算每个群组的样本数量
cluster_counts = data_filled['Cluster'].value_counts().sort_index()
print("\n每个群组的样本数量（-1 表示噪声点）：")
print(cluster_counts)

# 计算每个群组的特征平均值（排除噪声点）
cluster_profiles = data_filled[data_filled['Cluster'] != -1].groupby('Cluster').mean()
print("\n群组特征平均值（不包括噪声点）：")
print(cluster_profiles)

# 8. 可视化聚类结果
plt.figure(figsize=(10, 6))
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

for k, col in zip(unique_labels, colors):
    class_member_mask = (labels == k)
    if k == -1:
        col = 'black'
        label_name = '噪声点'
    else:
        label_name = f'群组 {k}'
    
    xy = X_umap[class_member_mask]
    plt.scatter(xy[:, 0], xy[:, 1], c=[col], label=label_name, edgecolors='k', s=20)

plt.title('DBSCAN 聚类结果（使用 UMAP 降维）')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.legend()
plt.show()
