import jsonlines
import os

BASE_DIR = "/home/murad/Documents/self-study/contextual_embeddings/data/evaluation_openai"
SAVE_DIR = "/home/murad/Documents/self-study/contextual_embeddings/data/evaluation_data"

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

with jsonlines.open(os.path.join(SAVE_DIR, "corpus.jsonl"), "w") as writer:
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

with jsonlines.open(os.path.join(SAVE_DIR, "test_queries.jsonl"), "w") as writer:
    writer.write_all(query_corpus)

# Test qrels generation
corpus_list = []
with jsonlines.open(os.path.join(SAVE_DIR, "corpus.jsonl"), "r") as corpus_reader:
    for obj in corpus_reader:
        corpus_list.append(obj)

queries = []
with jsonlines.open(os.path.join(SAVE_DIR, "test_queries.jsonl"), "r") as queries_reader:
    for obj in queries_reader:
        queries.append(obj)

whole_docs = []

for path in os.listdir(BASE_DIR):
    with jsonlines.open(os.path.join(BASE_DIR, path), "r") as reader:
        for obj in reader:
            whole_docs.append(obj)

final_document = []
for query in queries:
    document = {}
    if type(query["text"]) == str:
        text = query["text"]
    else:
        text = query["text"][0]

    qid = query["id"]
    for i in range (len(whole_docs)):
        if text in whole_docs[i]["query"]:
            answer = whole_docs[i]["pos"][0]
            break

    for i in range(len(corpus_list)):
        if answer in corpus_list[i]["text"]:
            docid = corpus_list[i]["id"]
            break

    document["qid"] = qid
    document["docid"] = docid
    document["relevance"] = 1

    final_document.append(document)

with jsonlines.open(os.path.join(SAVE_DIR, "test_qrels.jsonl"), "w") as writer:
    writer.write_all(final_document)
