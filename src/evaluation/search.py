import torch
import jsonlines
from FlagEmbedding import FlagModel
from sentence_transformers import util

# --- 1. CONFIGURE YOUR SETTINGS HERE ---

# Path to the directory containing your fine-tuned model
# MODEL_PATH = '/home/murad/Documents/self-study/contextual_embeddings/finetuned_bge_m3_v4'
MODEL_PATH = "BAAI/bge-m3"

# Path to your corpus file
CORPUS_PATH = '/home/murad/Documents/self-study/contextual_embeddings/data/evaluation_data/custom_evaluation_dataset/corpus.jsonl'

# The query you want to test
QUERY = "Bələdiyyə hansı məqsədlər üçün sifarişçi kimi çıxış edə bilər?"

# Number of top results to retrieve
TOP_K = 5

# --- END OF CONFIGURATION ---


def manual_search():
    """
    Loads a fine-tuned model and a corpus to perform a manual search.
    """
    print("--- Starting Manual Search ---")

    # --- 2. Load the Fine-Tuned Model ---
    print(f"Loading model from: {MODEL_PATH}")
    # Using FlagModel which is suitable for BGE models
    model = FlagModel(MODEL_PATH, use_fp16=True)
    print("Model loaded successfully.")

    # --- 3. Load the Corpus Data ---
    print(f"Loading corpus from: {CORPUS_PATH}")
    corpus_texts = []
    corpus_ids = []
    with jsonlines.open(CORPUS_PATH, mode='r') as reader:
        for doc in reader:
            # Handle both 'id' and '_id' keys for flexibility
            doc_id = doc.get('id', doc.get('_id'))
            if doc_id is not None:
                corpus_ids.append(str(doc_id))
                corpus_texts.append(doc.get('text', ''))
            
    if not corpus_texts:
        print("Error: No documents found in the corpus file. Please check the path and file format.")
        return
        
    print(f"Loaded {len(corpus_texts)} documents from the corpus.")

    # --- 4. Embed the Corpus ---
    print("Embedding all documents in the corpus... (This might take a while)")
    corpus_embeddings = model.encode(corpus_texts, batch_size=32)
    # Convert to tensor for faster similarity calculation
    corpus_embeddings_tensor = torch.tensor(corpus_embeddings)
    print("Corpus embedding complete.")

    # --- 5. Embed the Query ---
    print(f"\nSearching for query: '{QUERY}'")
    query_embedding = model.encode([QUERY])
    query_embedding_tensor = torch.tensor(query_embedding)

    # --- 6. Calculate Cosine Similarity ---
    # Compute cosine similarity between the query and all corpus documents
    cos_scores = util.cos_sim(query_embedding_tensor, corpus_embeddings_tensor)[0]

    # --- 7. Find and Display Top Results ---
    # Use torch.topk to find the highest scores and their indices
    top_results = torch.topk(cos_scores, k=min(TOP_K, len(corpus_texts)))

    print(f"\n--- Top {TOP_K} Most Relevant Documents ---")
    for i, (score, idx) in enumerate(zip(top_results.values, top_results.indices)):
        doc_id = corpus_ids[idx]
        doc_text = corpus_texts[idx]
        
        print(f"\n{i+1}. Rank: {i+1}")
        print(f"   Score: {score.item():.4f}")
        print(f"   Doc ID: {doc_id}")
        print(f"   Text: {doc_text[:]}...") # Print first 500 characters
        
    print("\n--- Search Finished ---")


if __name__ == "__main__":
    manual_search()