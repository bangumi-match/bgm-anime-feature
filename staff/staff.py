import json
from collections import defaultdict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# 修改后的代码（主要新增PCA降维部分）
from sklearn.decomposition import PCA  # 新增导入


# ...其余代码保持不变...
class AnimeStaffSimilarity:
  def __init__(self):
    self.config = {
      "core_positions": {'导演', '系列构成', '角色设计', '总作画监督', '音乐',
                         '音响监督'},
      "position_weights": {
        '导演': 5.0, '系列构成': 4.0, '角色设计': 4.0,
        '总作画监督': 3.5, '音乐': 3.0, '音响监督': 3.0,
        'default_core': 2.0, 'default_common': 1.2
      }
    }
    self.encoder = self.StaffEncoder(self.config)
    self.anime_ids = []
    self.pca = PCA(n_components=100)  # 新增PCA实例

  class StaffEncoder:
    def __init__(self, config):
      self.config = config
      self.feature_dict = {}
      self.counter = 0

    def _get_feature_key(self, person):
      if person['relation'] in self.config['core_positions']:
        return f"{person['relation']}_{person['id']}"
      return f"staff_{person['id']}"

    def fit(self, data):
      for project in data:
        for person in project["persons"]:
          key = self._get_feature_key(person)
          if key not in self.feature_dict:
            self.feature_dict[key] = self.counter
            self.counter += 1
      return self

    def transform(self, data):
      vectors = []
      for project in data:
        vec = np.zeros(len(self.feature_dict))
        common_weights = defaultdict(float)

        for person in project["persons"]:
          key = self._get_feature_key(person)
          pos_type = person['relation']

          # 计算权重
          if pos_type in self.config['position_weights']:
            weight = self.config['position_weights'][pos_type]
          elif pos_type in self.config['core_positions']:
            weight = self.config['position_weights']['default_core']
          else:
            weight = self.config['position_weights']['default_common']

          # 核心职位直接记录
          if key.startswith(tuple(self.config['core_positions'])):
            if key in self.feature_dict:
              vec[self.feature_dict[key]] += weight
          # 普通职位累计
          else:
            common_weights[key] += weight

        # 应用普通职位权重
        for key, w in common_weights.items():
          if key in self.feature_dict:
            vec[self.feature_dict[key]] += w

        vectors.append(vec)
      return np.array(vectors)

  def load_data(self, json_path):
    with open(json_path, encoding='utf-8') as f:
      self.raw_data = json.load(f)
    self.anime_ids = [item['id'] for item in self.raw_data]

  def build_model(self):
    """构建特征和相似度模型"""
    self.encoder.fit(self.raw_data)
    self.feature_vectors = self.encoder.transform(self.raw_data)

    # 新增降维步骤
    self.reduced_features = self.pca.fit_transform(
      self.feature_vectors)  # 降维到100维

    # 保持原有相似度计算（基于原始特征）
    self.similarity_matrix = cosine_similarity(self.feature_vectors)

  def save_artifacts(self, prefix="anime_similarity"):
    """保存所有计算结果"""
    # 保存原始特征（保持兼容性）
    np.savez_compressed(
        f"{prefix}_features.npz",
        features=self.feature_vectors,
        anime_ids=self.anime_ids
    )

    # 新增保存降维特征
    np.savez_compressed(
        f"{prefix}_reduced_100d.npz",
        features=self.reduced_features,
        anime_ids=self.anime_ids
    )

    # 保持原有相似度矩阵保存
    np.savez_compressed(
        f"{prefix}_matrix_staff.npz",
        matrix=self.similarity_matrix,
        anime_ids=self.anime_ids
    )

  def get_top_similar(self, anime_id, top_n=10):
    """获取最相似的动画ID列表"""
    idx = self.anime_ids.index(anime_id)
    similarities = self.similarity_matrix[idx]
    sorted_indices = np.argsort(similarities)[::-1][1:top_n + 1]  # 排除自己
    return [(self.anime_ids[i], similarities[i]) for i in sorted_indices]


# 使用示例
if __name__ == "__main__":
  system = AnimeStaffSimilarity()

  # 加载数据（假设数据文件为 anime_staff.json）
  system.load_data("subject_persons.json")

  # 构建模型
  system.build_model()

  # 保存结果
  system.save_artifacts()

  # 示例查询（假设存在动画ID为309）
  similar_list = system.get_top_similar(50)
  print("最相似的动画：", similar_list)

  # 查看方差解释率
  print(f"累计方差解释率: {system.pca.explained_variance_ratio_.sum():.2%}")

  # 查看降维后特征形状
  print(system.reduced_features.shape)  # 应为 (动画数量, 100)
