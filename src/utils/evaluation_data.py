import jsonlines
import os

BASE_DIR = "/home/murad/Documents/self-study/contextual_embeddings/data/evaluation_openai"
SAVE_DIR = "/home/murad/Documents/self-study/contextual_embeddings/data/"

# Whole corpus generation
corpus = []

for index, path in enumerate(os.listdir(BASE_DIR)):
    if not path.startswith("doc"):
        continue
    with jsonlines.open(os.path.join(BASE_DIR, path), "r") as reader:
        document = {}
        text = ""
        for obj in reader:
            text += ''.join(str(obj["pos"]))
        
        document["id"] = index
        document["text"] = text
        
        corpus.append(document)

with jsonlines.open(os.path.join(BASE_DIR, "corpus.jsonl"), "w") as writer:
    writer.write_all(corpus)

# Queries corpus generation
query_corpus = []
id = 100
for index, path in enumerate(os.listdir(BASE_DIR)):
    if not path.startswith("doc"):
        continue
    with jsonlines.open(os.path.join(BASE_DIR, path), "r") as reader:
        corp = {}
        text = ""
        for obj in reader:
            corp["id"] = id
            corp["text"] = obj["query"]
            id += 1
        
        query_corpus.append(corp)

with jsonlines.open(os.path.join(BASE_DIR, "test_queries.jsonl"), "w") as writer:
    writer.write_all(query_corpus)
