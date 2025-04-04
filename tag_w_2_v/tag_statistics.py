import json

# 读取JSON文件
with open('anime_lite.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 统计所有meta_tags种类数量
meta_tags_count = {}
for item in data:
    for tag in item['meta_tags']:
        if tag in meta_tags_count:
            meta_tags_count[tag] += 1
        else:
            meta_tags_count[tag] = 1

# 统计所有总标记数量大于100的tag的种类数量
tags_count = {}
for item in data:
    for tag in item['tags']:
        if tag['count'] > 0:
            if tag['name'] in tags_count:
                tags_count[tag['name']] += 1
            else:
                tags_count[tag['name']] = 1
# 打印所有meta_tags
print("All meta_tags:")
for tag in meta_tags_count:
    print(tag)
# 输出结果
print(f"Total meta_tags types: {len(meta_tags_count)}")
print(f"Total tags with count > 100: {len(tags_count)}")