import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set font for Chinese characters (if needed)
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# Data
layers = [2, 3, 4]
precision = [0.265584, 0.27728, 0.276448]
recall = [0.25733993, 0.26371371, 0.26025142]
ndcg = [0.35399297, 0.36985122, 0.36669807]

# Plot
plt.figure(figsize=(8, 6))
plt.plot(layers, precision, marker='o', label='Precision')
plt.plot(layers, recall, marker='s', label='Recall')
plt.plot(layers, ndcg, marker='^', label='NDCG')

# Labels and title
plt.xlabel('聚合层数')
plt.ylabel('评价指标')
plt.title('LightGCN层数对预测结果的影响')
plt.xticks(layers)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save and show the plot
plt.savefig('layers_vs_metrics.png', dpi=300)
plt.show()