import json
from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MaxAbsScaler
from sklearn.pipeline import make_pipeline

class AnimeStaffSimilarity:
  def __init__(self):
    self.config = {
      "core_positions": {
        "副本": 0, "导演": 1, "系列构成": 2, "脚本 ": 3, "原作": 4, "原案": 5, "人物设定": 6,
        "作画监督": 7, "原画": 8, "角色设计": 9, "总作画监督": 10, "音乐": 11,
        "音响监督": 12, "美术监督": 13, "色彩设计": 14, "摄影": 15, "摄影监督": 16,
        "演出": 17, "分镜": 18, "机械设定": 19
      },
      "position_weights": {
        "导演": 6.0, "系列构成": 4.0, "脚本 ": 4.0, "角色设计": 4.0, "原作": 2.0,
        "原案": 2.0, "人物设定": 2.0, "作画监督": 2.0, "原画": 2.0, "总作画监督": 3.5,
        "音乐": 3.0, "音响监督": 3.0, "美术监督": 2.0, "色彩设计": 2.0,
        "摄影": 2.0, "摄影监督": 2.0, "default_core": 2.0
      },
      "model_params": {
        "emb_dims": [1024],  # 只保留1024维度
        "gat_heads": 4,
        "epochs": 100,
        "learning_rate": 0.001
      }
    }
    self.encoder = self.StaffEncoder(self.config)
    self.anime_ids = []
    self.svd_pipelines = {
      dim: make_pipeline(
          MaxAbsScaler(),
          TruncatedSVD(n_components=dim, random_state=42)
      ) for dim in self.config["model_params"]["emb_dims"]
    }
    self.reduced_features_dict = {}
    self.reduced_similarity_dict = {}

  class StaffEncoder:
    def __init__(self, config):
      self.config = config
      self.feature_dict = {}

    def _get_feature_key(self, person):
      position = person['relation'].strip()
      pid = person['id']
      if position in self.config['core_positions']:
        return f"{position}_{pid}"
      else:
        return f"{pid}"

    def fit(self, data):
      seen = set()
      for project in data:
        for person in project["persons"]:
          key = self._get_feature_key(person)
          seen.add(key)
      self.feature_dict = {f: i for i, f in enumerate(sorted(seen))}
      return self

    def transform(self, data):
      vectors = []
      for project in data:
        vec = np.zeros(len(self.feature_dict))
        weight_acc = defaultdict(float)

        for person in project["persons"]:
          position = person['relation'].strip()
          pid = person['id']
          key = f"{position}_{pid}" if position in self.config['core_positions'] else pid
          weight = self.config['position_weights'].get(
              position,
              self.config['position_weights'].get('default_core', 1.0)
          )
          weight_acc[key] += weight

        for key, weight in weight_acc.items():
          if key in self.feature_dict:
            vec[self.feature_dict[key]] = weight

        vectors.append(vec)
      return np.array(vectors)

  def load_data(self, json_path):
    with open(json_path, encoding='utf-8') as f:
      self.raw_data = json.load(f)
    self.anime_ids = [item['id'] for item in self.raw_data]

  def build_model(self):
    self.encoder.fit(self.raw_data)
    self.feature_vectors = self.encoder.transform(self.raw_data)
    self.orig_similarity = cosine_similarity(self.feature_vectors)

    for dim, pipeline in self.svd_pipelines.items():
      reduced = pipeline.fit_transform(self.feature_vectors)
      similarity = cosine_similarity(reduced)

      self.reduced_features_dict[dim] = reduced
      self.reduced_similarity_dict[dim] = similarity

      ratio = pipeline.named_steps['truncatedsvd'].explained_variance_ratio_.sum()
      print(f"[{dim}维] 精密度总和: {ratio:.4f}")

  def save_artifacts(self):
    # 按照 anime_ids 升序排序
    sorted_indices = np.argsort(self.anime_ids)
    sorted_anime_ids = [self.anime_ids[i] for i in sorted_indices]

    # 只处理1024维数据
    dim = 1024
    features = self.reduced_features_dict[dim]
    similarity = self.reduced_similarity_dict[dim]

    # 保存特征向量
    np.savez_compressed(
        "../data/staff_vectors.npz",
        features=features[sorted_indices],
        anime_ids=sorted_anime_ids
    )

    # 保存相似度矩阵
    np.savez_compressed(
        "../data/staff_similarity.npz",
        matrix=similarity[sorted_indices][:, sorted_indices],
        anime_ids=sorted_anime_ids
    )

  def get_top_similar(self, anime_id, top_n=10, use_reduced=False, dim=1024):
    idx = self.anime_ids.index(anime_id)
    if use_reduced:
      matrix = self.reduced_similarity_dict.get(dim)
      if matrix is None:
        raise ValueError(f"{dim}维精准矩阵未生成")
    else:
      matrix = self.orig_similarity

    similarities = matrix[idx]
    sorted_indices = np.argsort(similarities)[::-1][1:top_n + 1]
    return [(self.anime_ids[i], similarities[i]) for i in sorted_indices]

if __name__ == "__main__":
  system = AnimeStaffSimilarity()
  system.load_data("../data/anime_staffs.json")
  system.build_model()

  print(f"实际特征维度: {system.feature_vectors.shape[1]}")
  print(f"示例特征名称: {list(system.encoder.feature_dict.keys())[:15]}")

  anime_id = 50
  print("\n原始特征相似结果:")
  for aid, score in system.get_top_similar(anime_id)[:15]:
    print(f"{aid}: {score:.3f}")

  # 只显示1024维结果
  dim = 1024
  print(f"\n{dim}D 降维后结果:")
  for aid, score in system.get_top_similar(anime_id, use_reduced=True, dim=dim)[:15]:
    print(f"{aid}: {score:.3f}")

  system.save_artifacts()