import numpy as np
import umap
from sklearn.preprocessing import StandardScaler
import pandas as pd

# 加载三个算法的数据
# 1. Staff 1024D 特征
staff_data = np.load('anime_similarity_reduced_1024d.npz')
staff_features = staff_data['features']
staff_ids = staff_data['anime_ids']

# 2. Word2Vec 特征
word2vec_data = np.load('anime_vectors.npz')
word2vec_features = word2vec_data['vectors']
word2vec_ids = word2vec_data['ids']

# 3. User-based CF 特征 (假设使用 all_vectors.npz)
user_cf_data = np.load('all_vectors.npz')
user_cf_ids = np.array([int(uid) for uid in user_cf_data.files])
user_cf_features = np.array([user_cf_data[uid] for uid in user_cf_data.files])

# 找到共同 ID
common_ids = np.intersect1d(staff_ids, word2vec_ids)
common_ids = np.intersect1d(common_ids, user_cf_ids)

# 对齐特征
def align_features(source_ids, source_features, target_ids):
  idx = np.where(np.isin(source_ids, target_ids))[0]
  sorted_idx = idx[np.argsort(source_ids[idx])]
  return source_features[sorted_idx]

staff_aligned = align_features(staff_ids, staff_features, common_ids)
word2vec_aligned = align_features(word2vec_ids, word2vec_features, common_ids)
user_cf_aligned = align_features(user_cf_ids, user_cf_features, common_ids)

# 拼接特征
combined_features = np.concatenate([staff_aligned, word2vec_aligned, user_cf_aligned], axis=1)
# Check and fill invalid values
if np.isnan(combined_features).any():
  print("Detected invalid values, filling with column means...")
  col_means = np.nanmean(combined_features, axis=0)
  # Replace NaN means with 0 for columns that are entirely NaN
  col_means = np.where(np.isnan(col_means), 0, col_means)
  inds = np.where(np.isnan(combined_features))
  combined_features[inds] = np.take(col_means, inds[1])

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(combined_features)

# UMAP dimensionality reduction
reducer = umap.UMAP(
    random_state=2501,
    n_neighbors=15,  # 调整邻域大小
    min_dist=0.1,    # 增大最小距离
    densmap=True     # 启用密度均衡
)
embeddings = reducer.fit_transform(scaled_features)

# Save the results
df = pd.DataFrame({
  'id': np.sort(common_ids),
  'x': embeddings[:, 0],
  'y': embeddings[:, 1]
})
df.to_csv('anime_umap_coordinates.csv', index=False)
print("Coordinates saved to anime_umap_coordinates.csv")