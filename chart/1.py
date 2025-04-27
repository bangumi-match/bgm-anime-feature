import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set font to SimHei (or another Chinese-compatible font)
rcParams['font.sans-serif'] = ['SimHei']  # Use SimHei for Chinese characters
rcParams['axes.unicode_minus'] = False   # Ensure minus signs are displayed correctly

# 实验结果
embedding_dims = [64, 128, 256, 512, 1024]
precision = [0.259108, 0.276388, 0.28, 0.2773, 0.2730]
recall = [0.24626092, 0.2561, 0.26641013, 0.2637, 0.2593]
ndcg = [0.34093032, 0.3572, 0.37294214, 0.3699, 0.3644]

# 使用等距索引代替实际数值
indices = range(len(embedding_dims))

plt.figure(figsize=(8, 6))
plt.plot(indices, precision, marker='o', label='Precision@50')
plt.plot(indices, recall, marker='s', label='Recall@50')
plt.plot(indices, ndcg, marker='^', label='NDCG@50')

# 设置 x 轴刻度为实际的 embedding_dims 值
plt.xticks(indices, embedding_dims)

plt.xlabel('嵌入维度')
plt.ylabel('评价指标')
plt.title('LightGCN嵌入维度对预测结果的影响')
plt.legend()
plt.grid(True)
plt.tight_layout()

# 保存图片为高清格式
plt.savefig('output.png', dpi=300)  # 保存为 PNG 格式，分辨率为 300 DPI
plt.show()