import json
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)

# Read JSON file
with open('anime_lite.json', 'r', encoding='utf-8') as f:
  data = json.load(f)

# Extract summary and generate feature vectors
def extract_features(summary):
  inputs = tokenizer(summary, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
  outputs = model(**inputs)
  return outputs.last_hidden_state.mean(dim=1).cpu().detach().numpy()

# Extract features for all anime summaries
features = []
subject_ids = []
for anime in data:
  summary = anime['summary']
  subject_id = anime['id']
  feature = extract_features(summary)
  features.append(feature)
  subject_ids.append(subject_id)

# Convert to numpy array and reduce to 100 dimensions
features = np.vstack(features)
features_100d = features[:, :100]

# Compute similarity matrix
similarity_matrix = cosine_similarity(features_100d)

# Save feature vectors and similarity matrix
np.savez_compressed('anime_features_similarity.npz',
                    subject_ids=subject_ids,
                    features=features_100d,
                    similarity=similarity_matrix)

# Get similar anime for a given subject_id
def get_top_similar(subject_id, top_k=30):
  try:
    idx = subject_ids.index(subject_id)
    sim_scores = similarity_matrix[idx]
    sorted_indices = np.argsort(-sim_scores)
    results = [(subject_ids[i], sim_scores[i]) for i in sorted_indices[1:top_k+1]]
    return results
  except ValueError:
    print(f"Error: subject_id {subject_id} not found")
    return []

# Check similar items for subject_id=50
test_id = 50
similar_items = get_top_similar(test_id, 30)

if similar_items:
  print(f"Found {len(similar_items)} similar items:")
  for sid, score in similar_items[:5]:  # Print top 5 results
    print(f"subject_id: {sid} similarity: {score:.4f}")
else:
  print("No similar items found")