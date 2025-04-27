import json
import os
import numpy as np
from gensim.models import Word2Vec
from collections import defaultdict


def prepare_data(file_path):
  invalid_meta_tags = {"TV", "剧场版", "WEB", "OVA"}
  with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

  sentences = []
  animes = []
  for item in data:
    # 过滤无效meta_tags
    valid_meta_tags = [tag for tag in item['meta_tags'] if
                       tag not in invalid_meta_tags]
    # 过滤普通tags（count>100）
    valid_normal_tags = [tag for tag in item['tags'] if
                         tag.get('count', 0) > 100]

    # 构建训练数据（仅名称）
    combined_names = valid_meta_tags + [tag['name'] for tag in
                                        valid_normal_tags]
    sentences.append(combined_names)

    # 构建带权重的tag数据
    weighted_tags = []
    for mt in valid_meta_tags:
      weighted_tags.append({'name': mt, 'weight': 1})  # meta_tags权重为1

    total_tags_count = sum(
        tag['count'] for tag in valid_normal_tags)  # 计算总标签数量
    for nt in valid_normal_tags:
      weight = nt['count'] / total_tags_count * 5  # 计算权重
      weighted_tags.append({'name': nt['name'], 'weight': weight})

    animes.append({
      'id': item['id'],
      'name': item['name'],
      'namecn': item['name_cn'],
      'tags': weighted_tags
    })
  animes_sorted = sorted(animes, key=lambda x: x['id'])
  return sentences, animes_sorted


def train_and_save_model(sentences, save_path='word2vec.model'):
  model = Word2Vec(
      sentences=sentences,
      vector_size=400,
      window=5,
      min_count=100,
      workers=8,
      sg=1
  )
  model.save(save_path)
  return model


def load_model(model_path='word2vec.model'):
  return Word2Vec.load(model_path)


def compute_anime_vectors(animes, model):
  anime_vectors = {}
  for anime in animes:
    weighted_sum = np.zeros(model.vector_size)
    total_weight = 0

    for tag_info in anime['tags']:
      tag = tag_info['name']
      weight = tag_info['weight']
      if tag in model.wv:
        vector = model.wv[tag] * weight
        if weighted_sum is None:
          weighted_sum = vector
        else:
          weighted_sum += vector
        total_weight += weight

    if weighted_sum is not None and total_weight > 0:
      avg_vector = weighted_sum / total_weight
      norm = np.linalg.norm(avg_vector)
      unit_vector = avg_vector / norm if norm != 0 else avg_vector
      anime_vectors[anime['id']] = {
        'name': anime['name'],
        'name_cn': anime['namecn'],
        'vector': unit_vector
      }
    else:
      anime_vectors[anime['id']] = None
  return anime_vectors


class EnhancedAnimeQuery:
  def __init__(self, model_path, anime_vectors):
    self.model = Word2Vec.load(model_path)
    self.word_vectors = self.model.wv
    sorted_aids = sorted([aid for aid, ainfo in anime_vectors.items() if ainfo is not None and ainfo['vector'] is not None])

    # 预处理动画数据
    self.anime_ids = []
    self.anime_names = []
    self.anime_namecns = []
    self.anime_vecs = []
    self.name_to_id = {}
    self.id_to_info = {}

    for aid in sorted_aids:
      ainfo = anime_vectors[aid]
      self.anime_ids.append(aid)
      self.anime_names.append(ainfo['name'])
      self.anime_namecns.append(ainfo['name_cn'])
      self.anime_vecs.append(ainfo['vector'])
      self.name_to_id[ainfo['name']] = aid
      self.id_to_info[aid] = ainfo

    self.anime_vecs = np.array(self.anime_vecs)

  def get_similar_tags(self, tag, topn=50):
    try:
      results = self.word_vectors.most_similar(tag, topn=topn)
      return [{"tag": item[0], "similarity": round(item[1], 4)} for item in
              results]
    except KeyError:
      return f"Tag '{tag}' not found in vocabulary"

  def get_anime_info(self, identifier):
    # 解析标识符
    if isinstance(identifier, str) and identifier in self.name_to_id:
      anime_id = self.name_to_id[identifier]
    elif identifier in self.anime_ids:
      anime_id = identifier
    else:
      return {"error": f"Identifier '{identifier}' not found"}

    # 获取向量信息
    if anime_id not in self.id_to_info:
      return {"error": "No vector available for this anime"}

    anime_info = self.id_to_info[anime_id]
    vector = anime_info['vector']

    # 获取相似动画
    similar_animes = self.get_similar_animes(identifier)

    return {
      'id': anime_id,
      'name': anime_info['name'],
      'vector': vector.tolist(),  # 转换为Python list
      'similar_animes': similar_animes
    }

  def get_similar_animes(self, identifier, topn=10):
    # 根据输入标识符获取动画ID
    if isinstance(identifier, str) and identifier in self.name_to_id:
      anime_id = self.name_to_id[identifier]
    elif identifier in self.anime_ids:
      anime_id = identifier
    else:
      return f"Identifier '{identifier}' not found"

    try:
      idx = self.anime_ids.index(anime_id)
    except ValueError:
      return f"Anime '{identifier}' has no vector representation"

    target_vec = self.anime_vecs[idx]
    similarities = np.dot(self.anime_vecs, target_vec)

    # 获取排序结果（排除自己）
    sorted_indices = np.argsort(-similarities)
    results = []
    for i in sorted_indices:
      if self.anime_ids[i] == anime_id:
        continue
      results.append({
        'id': self.anime_ids[i],
        'name': self.anime_names[i],
        'similarity': float(similarities[i])
      })
      if len(results) >= topn:
        break

    return results[:topn]


# ...（前面的prepare_data, train_and_save_model等函数保持不变）...

if __name__ == "__main__":
  # 数据准备
  sentences, animes = prepare_data('../data/anime.json')
  all_anime_ids = [anime['id'] for anime in animes]

  # 创建一个全 NaN 的相似度矩阵
  full_similarity_matrix = np.full((len(all_anime_ids), len(all_anime_ids)), np.nan, dtype=np.float32)


  # 模型训练/加载
  model_path = 'word2vec.model'
  # if not os.path.exists(model_path):
  #   model = train_and_save_model(sentences, model_path)
  # else:
  #   model = load_model(model_path)
  model = train_and_save_model(sentences, model_path)


  # 计算特征向量
  anime_vectors = compute_anime_vectors(animes, model)

  # 初始化查询器
  query = EnhancedAnimeQuery(model_path, anime_vectors)

  # 批量生成所有动画的相似关系和特征向量
  print("\n生成批量数据...")

  # 获取所有有效动画数据
  valid_anime_ids = query.anime_ids
  anime_vectors_matrix = query.anime_vecs

  # 构建id到索引的映射
  anime_id_to_index = {aid: idx for idx, aid in enumerate(all_anime_ids)}
  valid_id_to_index = {aid: idx for idx, aid in enumerate(valid_anime_ids)}

  # 计算全局相似度矩阵
  similarity_matrix = np.dot(anime_vectors_matrix, anime_vectors_matrix.T)

  # 填充有效的相似度值
  for i, aid_i in enumerate(valid_anime_ids):
    for j, aid_j in enumerate(valid_anime_ids):
      full_i = anime_id_to_index[aid_i]
      full_j = anime_id_to_index[aid_j]
      full_similarity_matrix[full_i][full_j] = similarity_matrix[i][j]

  # 准备存储结构
  top10_output = {}
  top100_output = {}
  least100_output = {}
  vector_output = {}

  # 遍历所有有效动画
  for idx, anime_id in enumerate(valid_anime_ids):
    # 获取基础信息
    name = query.anime_names[idx]
    name_cn = query.anime_namecns[idx]
    vector = anime_vectors_matrix[idx]

    # 存储特征向量（使用单精度浮点数节省空间）
    vector_output[anime_id] = vector.astype(np.float32).tolist()

    # 处理相似度数据
    similarities = similarity_matrix[idx]
    sorted_indices = np.argsort(-similarities)

    # 排除自己并获取top和least结果
    valid_indices = [i for i in sorted_indices if i != idx]
    top100_indices = valid_indices[:100]
    least100_indices = valid_indices[-100:]

    # 构建top10数据
    top10_list = []
    for sim_idx in top100_indices[:10]:
      top10_list.append({
        "id": valid_anime_ids[sim_idx],
        "name": query.anime_names[sim_idx],
        "name_cn": query.anime_namecns[sim_idx],
        "similarity": float(similarities[sim_idx])
      })

    # 构建top100的压缩格式（[id, 相似度]）
    top100_compressed = [
      [valid_anime_ids[sim_idx], round(float(similarities[sim_idx]), 4)]
      for sim_idx in top100_indices
    ]

    # 构建least100的压缩格式（[id, 相似度]）
    least100_compressed = [
      [valid_anime_ids[sim_idx], round(float(similarities[sim_idx]), 4)]
      for sim_idx in least100_indices
    ]

    # 存储结果
    top10_output[anime_id] = {
      "name": name,
      "name_cn": name_cn,
      "top10": top10_list
    }
    top100_output[anime_id] = top100_compressed
    least100_output[anime_id] = least100_compressed

  # 保存结果文件
  print("\n保存输出文件...")

  # # 1. 每个动画的top10相似动画（带完整信息）
  # with open("anime_top10.json", "w", encoding="utf-8") as f:
  #   json.dump(top10_output, f, ensure_ascii=False, indent=2)
  #
  # # 2. 每个动画的top100相似动画（压缩格式）
  # with open("anime_top100.compact.json", "w", encoding="utf-8") as f:
  #   json.dump(top100_output, f, ensure_ascii=False, separators=(",", ":"))
  #
  # # 3. 每个动画的least100不相似动画（压缩格式）
  # with open("anime_least100.compact.json", "w", encoding="utf-8") as f:
  #   json.dump(least100_output, f, ensure_ascii=False, separators=(",", ":"))

  # 4. 保存相似度矩阵
  np.savez_compressed(
      "../data/tag_similarity.npz",
      ids=valid_anime_ids,
      matrix=full_similarity_matrix.astype(np.float32)
  )

  # 5. 特征向量存储（使用二进制格式优化）
  np.savez_compressed(
      "../data/tag_vectors.npz",
      ids=valid_anime_ids,
      vectors=anime_vectors_matrix.astype(np.float32)
  )

  # 附加元数据文件
  # with open("anime_vector_metadata.json", "w", encoding="utf-8") as f:
  #   metadata = {
  #     str(anime_id): {"name": query.id_to_info[anime_id]["name"]}
  #     for anime_id in valid_anime_ids
  #   }
  #   json.dump(metadata, f, ensure_ascii=False, indent=2)

  print("处理完成！")