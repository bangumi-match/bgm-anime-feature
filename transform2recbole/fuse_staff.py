import numpy as np

# 加载.npz文件
data = np.load('anime_similarity_reduced_128d.npz')
anime_ids = data['anime_ids']
features = data['features']

# 创建anime_id到特征的映射字典
id_to_feature = {aid: feature for aid, feature in zip(anime_ids, features)}

input_filename = 'bangumi.item'
output_filename = 'bangumi_new.item'

with open(input_filename, 'r', encoding='utf-8') as fin, \
    open(output_filename, 'w', encoding='utf-8') as fout:

  for line in fin:
    line = line.strip()
    if not line:
      fout.write('\n')
      continue

    fields = line.split(',')
    first_field = fields[0].strip()

    # 处理标题行
    if first_field == 'subject_id:token':
      # 添加staff_embed列头
      fout.write(','.join(fields) + ',staff_embed:float_seq\n')
      continue

    # 处理特殊行
    if first_field == '[PAD]':
      fout.write(line + '\n')
      continue

    # 处理数据行
    try:
      subject_id = int(fields[0].strip())
    except ValueError:
      fout.write(line + '\n')
      continue

    # 查找特征
    if subject_id not in id_to_feature:
      fout.write(line + '\n')
      continue

    # 构造新字段
    feature = id_to_feature[subject_id]
    feature_str = ' '.join(f"{x:.12f}" for x in feature)
    staff_field = f"{feature_str}"

    # 更新或添加staff字段
    staff_added = False
    new_fields = []
    for field in fields:
      if field.startswith('staff_embed:float_seq'):
        new_fields.append(staff_field)
        staff_added = True
      else:
        new_fields.append(field)

    if not staff_added:
      new_fields.append(staff_field)

    fout.write(','.join(new_fields) + '\n')