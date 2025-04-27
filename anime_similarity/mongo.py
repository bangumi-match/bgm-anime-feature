import json
import os

from pymongo import MongoClient, InsertOne

# MongoDB connection string
connection_string = os.environ.get("MONGODB_CONNECTION_STRING")
if not connection_string or "mongodb://" not in connection_string:
    raise ValueError("Invalid MongoDB connection string")

# Connect to MongoDB
client = MongoClient(connection_string)
db = client.bangumi
collection = db.similarity
collection.delete_many({})

# Load data from JSON file
with open('top100_similar_animes_combined.json', 'r', encoding='utf-8') as file:
  data = json.load(file)

# Prepare bulk operations
operations = []
for anime_id, similar_animes in data.items():
  operations.append(InsertOne({
    "anime_id": int(anime_id),
    "similar_animes": [
      {"id": similar_anime["id"], "score": similar_anime["score"]}
      for similar_anime in similar_animes
    ]
  }))

# Execute bulk operations
if operations:
  result = collection.bulk_write(operations)
  print(f"Inserted {result.inserted_count} documents.")

# Close the connection
client.close()