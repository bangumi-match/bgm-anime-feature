#%%
import json
import numpy as np
from tqdm import tqdm

# 读取动画元数据
print("读取动画元数据...")
with open('../data/anime.json', 'r', encoding='utf-8') as f:
  anime_data = json.load(f)

# 构建完整subject映射
subject_id_map = {a['id']: idx for idx, a in enumerate(anime_data)}
all_subjects = set(subject_id_map.keys())
num_subjects = len(subject_id_map)

# 读取用户数据
print("读取用户数据...")
with open('../data/user.json', 'r', encoding='utf-8') as f:
  user_data = json.load(f)

# 构建用户映射
user_id_map = {}
for user in user_data:
  if user['user_id'] not in user_id_map:
    user_id_map[user['user_id']] = len(user_id_map)
num_users = len(user_id_map)

# 初始化评分矩阵
print(f"\n初始化评分矩阵 ({num_users} users x {num_subjects} subjects)")
rating_matrix = np.full((num_users, num_subjects), np.nan, dtype=np.float32)

# 填充有效评分
print("填充评分数据...")
seen_subjects = set()
for user in tqdm(user_data):
  user_idx = user_id_map[user['user_id']]
  for item in user.get('collect', []) or []:
    sid = item['subject_id']
    if sid not in subject_id_map:
      continue
    if item['rate'] == 0:
      continue
    subj_idx = subject_id_map[sid]
    rating_matrix[user_idx, subj_idx] = item['rate']
    seen_subjects.add(sid)

# 识别未被收藏的动画
unseen_subjects = all_subjects - seen_subjects
unseen_indices = [subject_id_map[sid] for sid in unseen_subjects]
print(f"发现 {len(unseen_subjects)} 个未被收藏的动画")

# 计算相似度
def safe_cosine(matrix):
  # 均值中心化
  mean = np.nanmean(matrix, axis=1, keepdims=True)
  centered = matrix - mean

  # 处理全NaN行
  valid_rows = ~np.all(np.isnan(centered), axis=1)
  norm_matrix = np.zeros_like(centered)
  norm_matrix[valid_rows] = np.nan_to_num(centered[valid_rows])

  # 计算范数
  norms = np.linalg.norm(norm_matrix, axis=1, keepdims=True)
  norms[norms == 0] = 1e-6

  # 计算相似度
  sim = np.dot(norm_matrix, norm_matrix.T) / (norms * norms.T)

  # 标记无效行
  invalid_rows = ~valid_rows
  sim[invalid_rows, :] = np.nan
  sim[:, invalid_rows] = np.nan

  return sim

print("\n计算相似度矩阵...")
similarity_matrix = safe_cosine(rating_matrix.T)  # (subjects x users) 转置

# 强制设置未被收藏动画的相似度为NaN
similarity_matrix[unseen_indices, :] = np.nan
similarity_matrix[:, unseen_indices] = np.nan

# 验证维度
assert similarity_matrix.shape == (num_subjects, num_subjects)
print(f"最终矩阵维度验证通过: {similarity_matrix.shape}")

# 保存结果
subject_ids = [a['id'] for a in anime_data]  # 保持与anime_lite.json相同顺序
# 保存整体相似度矩阵
np.savez("../data/cf_similarity.npz",
         similarity=similarity_matrix,
         subject_ids=subject_ids)

# 保存所有动画的相似度向量到一个文件
print("\n保存所有动画的相似度向量到一个文件...")
vector_dict = {str(subject_id): similarity_matrix[idx] for idx, subject_id in enumerate(subject_ids)}
np.savez("../data/cf_vectors.npz", **vector_dict)

print("所有动画的相似度向量已保存！")

print("\n示例检查未被收藏动画的相似度:")
test_unseen = next(iter(unseen_subjects)) if unseen_subjects else None
if test_unseen:
  idx = subject_id_map[test_unseen]
  print(f"subject_id {test_unseen} 的相似度均值: {np.nanmean(similarity_matrix[idx])}")
else:
  print("所有动画都有收藏记录")

print("操作完成！")