import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set font to SimHei to support Chinese characters
rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei for Chinese
rcParams['axes.unicode_minus'] = False   # Ensure minus signs are displayed correctly

# Load coordinate data
df = pd.read_csv('anime_umap_coordinates.csv')
id_to_coord = dict(zip(df['id'], zip(df['x'], df['y'])))

# User input
input_ids = list(map(int, input("请输入动画ID（用逗号分隔）: ").split(',')))

# Extract coordinates
valid_ids = []
coords = []
for aid in input_ids:
  if aid in id_to_coord:
    valid_ids.append(aid)
    coords.append(id_to_coord[aid])
  else:
    print(f"警告: ID {aid} 不存在")

if not valid_ids:
  print("无有效ID，退出")
  exit()

# Plot
plt.figure(figsize=(12, 10))
plt.scatter(df['x'], df['y'], c='gray', s=5, alpha=0.3, label='所有动画')
xs, ys = zip(*coords) if coords else ([], [])
plt.scatter(xs, ys, c='red', s=50, edgecolor='black', label='选中动画')

# Annotate IDs
for aid, x, y in zip(valid_ids, xs, ys):
  plt.text(x, y, str(aid), fontsize=8, ha='right', va='bottom')

plt.title('动画项目UMAP可视化')
plt.xlabel('UMAP维度1')
plt.ylabel('UMAP维度2')
plt.legend()
plt.show()

# 8,237,50,324,9622,25961,253,10380,3375,1728,1428,4583,464376