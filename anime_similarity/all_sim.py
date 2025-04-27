import numpy as np
import json
from pathlib import Path

# 配置参数
npz_paths = ['../data/cf_similarity.npz', '../data/staff_similarity.npz','../data/tag_similarity.npz']
json_path = '../data/anime.json'
weights = [0.4, 0.3, 0.3]
output_paths = {
  "cf": 'top100_similar_animes_cf.json',
  "staff": 'top100_similar_animes_staff.json',
  "tag": 'top100_similar_animes_tag.json',
  "combined": 'top100_similar_animes_combined.json'
}
check_id = 50

def load_fill_and_normalize(npz_path, key):
  """逐行处理：行均值填充NaN+逐行归一化"""
  data = np.load(npz_path)
  matrix = data[key]
  filled_matrix = np.zeros_like(matrix, dtype=np.float32)
  global_mean = np.nanmean(matrix)  # 全局均值用于全NaN行

  for i in range(matrix.shape[0]):
    row = matrix[i]

    # Check if the row is entirely NaN
    if np.all(np.isnan(row)):
      row_mean = np.nan  # Explicitly set row_mean to NaN
    else:
      row_mean = np.nanmean(row)

    # Handle NaN row_mean
    if np.isnan(row_mean):
      fill_value = global_mean * 0.2
    else:
      fill_value = row_mean * 0.2

    # Fill and normalize the row
    filled_row = np.where(np.isnan(row), fill_value, row)
    row_min = filled_row.min()
    row_max = filled_row.max()

    if row_max == row_min:
      norm_row = np.zeros_like(filled_row)
    else:
      norm_row = (filled_row - row_min) / (row_max - row_min)

    filled_matrix[i] = norm_row.astype(np.float32)

  return filled_matrix

# 读取并处理矩阵
matrix_cf = load_fill_and_normalize(npz_paths[0], 'similarity')
matrix_staff = load_fill_and_normalize(npz_paths[1], 'matrix')
matrix_tag = load_fill_and_normalize(npz_paths[2], 'matrix')

# 确保尺寸一致
assert matrix_cf.shape == matrix_staff.shape == matrix_tag.shape, "矩阵尺寸不一致"

# 加权融合（此处保持不变）
matrix_combined = (matrix_cf * weights[0] + matrix_staff * weights[1] + matrix_tag * weights[2])


# 读取动画信息
with open(json_path, 'r', encoding='utf-8') as f:
  json_data = json.load(f)
  id_to_idx = {item['id']: idx for idx, item in enumerate(json_data)}
  idx_to_info = {idx: (item['name'], item['name_cn']) for idx, item in enumerate(json_data)}

def generate_top100(matrix, output_path):
  """生成Top100相似列表并保存到文件"""
  top100_results = {}
  for idx in range(len(json_data)):
    sim_scores = matrix[idx]
    sim_scores[idx] = -np.inf  # 排除自身
    top_indices = np.argsort(-sim_scores)[:100]
    top_similar = [{"id": json_data[i]['id'], "score": float(sim_scores[i])} for i in top_indices]
    top100_results[json_data[idx]['id']] = top_similar

  # 保存结果
  with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(top100_results, f, ensure_ascii=False, indent=2)
  return top100_results

# 分别生成并保存结果
top100_cf = generate_top100(matrix_cf, output_paths["cf"])
top100_staff = generate_top100(matrix_staff, output_paths["staff"])
top100_tag = generate_top100(matrix_tag, output_paths["tag"])
top100_combined = generate_top100(matrix_combined, output_paths["combined"])

def print_top30_results(top100_results, check_id, method_name):
  """打印指定ID的前30个相似动画"""
  if check_id not in id_to_idx:
    print(f"ID {check_id} 不在数据中")
    return
  idx = id_to_idx[check_id]
  similar_items = top100_results.get(check_id, [])
  print(f"\n方法: {method_name}")
  print(f"动画ID {check_id} 的相似动画（名称中/日）：")
  print(f"原始动画：{idx_to_info[idx][1]} / {idx_to_info[idx][0]}")
  for similar_item in similar_items[:30]:
    similar_id = similar_item["id"]
    score = similar_item["score"]
    s_idx = id_to_idx.get(similar_id, None)
    if s_idx is None:
      continue
    name, name_cn = idx_to_info[s_idx]
    print(f"  - {name_cn} / {name} (相似度: {score:.4f})")

# 打印动画ID 50的前30个相似动画
print_top30_results(top100_cf, check_id, "CF")
print_top30_results(top100_staff, check_id, "Staff")
print_top30_results(top100_tag, check_id, "Tag")
print_top30_results(top100_combined, check_id, "Combined")