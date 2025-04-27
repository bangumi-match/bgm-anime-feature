import numpy as np

class AnimeSimilarityQuery:
  def __init__(self, matrix_path, ids_path):
    """
    初始化查询器
    :param matrix_path: 相似度矩阵.npz文件路径
    :param ids_path: 动画ID列表.npz文件路径
    """
    # 加载相似度矩阵
    matrix_data = np.load(matrix_path)
    self.similarity_matrix = matrix_data["matrix"]

    # 加载动画ID列表
    ids_data = np.load(ids_path)
    self.anime_ids = ids_data["anime_ids"]

    # 创建ID到索引的映射
    self.id_to_index = {aid: idx for idx, aid in enumerate(self.anime_ids)}

  def find_similar_animes(self, anime_id, top_n=10):
    """
    查找相似动画
    :param anime_id: 要查询的动画ID
    :param top_n: 返回结果数量
    :return: 包含(动画ID, 相似度)的列表
    """
    # 获取索引
    try:
      idx = self.id_to_index[anime_id]
    except KeyError:
      raise ValueError(f"动画ID {anime_id} 不存在于数据集中")

    # 获取相似度行
    similarities = self.similarity_matrix[idx]

    # 排除自身并排序
    sorted_indices = np.argsort(similarities)[::-1][1:top_n+1]

    # 组合结果
    return [(self.anime_ids[i], similarities[i]) for i in sorted_indices]

# 使用示例
if __name__ == "__main__":
  # 初始化查询器（使用之前保存的文件）
  query = AnimeSimilarityQuery(
      matrix_path="../data/staff_similarity.npz",
      ids_path="../data/staff_vectors.npz"
  )

  # 查询动画ID=50的相似作品
  target_id = 2
  try:
    results = query.find_similar_animes(target_id, top_n=5)
    print(f"与动画 {target_id} 最相似的5部作品：")
    for rank, (aid, score) in enumerate(results, 1):
      print(f"TOP{rank}: 动画ID={aid}, 相似度={score:.4f}")
  except ValueError as e:
    print(str(e))