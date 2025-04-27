import numpy as np

# 加载 .npz 文件
data = np.load('cf/all_vectors.npz')

# 查看包含的键（每个键对应一个数组）
print("Keys in the .npz file:", data.files)

# 遍历每个数组并打印一些信息
for key in data.files:
  array = data[key]
  print(f"\nKey: '{key}'")
  print(f"  Shape: {array.shape}")
  print(f"  Dtype: {array.dtype}")
  # 显示前几项
  print(f"  First 5 values:\n{array[:10]}")
