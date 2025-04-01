import torch
from transformers import XLMRobertaTokenizer, XLMRobertaModel
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

def get_contextual_word_embeddings_improved(model, tokenizer, sentences, target_word, layer_num=-1):
    """
    Extract embeddings for a specific word in different contexts with improved token detection.
    
    Args:
        model: The transformer model
        tokenizer: The tokenizer for the model
        sentences: List of sentences containing the target word
        target_word: The word to extract embeddings for
        layer_num: Which layer to extract embeddings from (-1 for last layer)
    
    Returns:
        Dictionary mapping sentence index to word embedding
    """
    model.eval()
    word_embeddings = {}
    detailed_info = {}
    
    for i, sentence in enumerate(sentences):
        # Tokenize the sentence
        encoded_input = tokenizer(sentence, return_tensors='pt')
        
        # Get all tokens for debugging
        input_ids = encoded_input['input_ids'][0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        
        print(f"\nSentence {i}: '{sentence}'")
        print(f"All tokens: {tokens}")
        
        # Find token(s) that might contain our target word
        matched_indices = []
        
        # First try exact match on full word with word boundary marker
        for j, token in enumerate(tokens):
            if token == f'▁{target_word}':
                matched_indices.append(j)
                print(f"Exact match for '▁{target_word}' at position {j}")
        
        # If no exact match, look for the token without the word boundary marker
        if not matched_indices:
            for j, token in enumerate(tokens):
                if token == target_word:
                    matched_indices.append(j)
                    print(f"Match for '{target_word}' at position {j}")
        
        # If still no match, the word might be split into subword tokens
        # Try a more fuzzy matching approach
        if not matched_indices:
            # Reconstruct the full text from tokens to find offsets
            full_text = ""
            offset_mapping = []
            
            for token in tokens:
                # Remove the leading '▁' if present (marks word boundary in XLM-RoBERTa)
                if token.startswith('▁'):
                    token_text = token[1:]
                    is_word_start = True
                else:
                    token_text = token
                    is_word_start = False
                
                start = len(full_text)
                full_text += token_text
                end = len(full_text)
                
                offset_mapping.append((start, end, is_word_start))
            
            # Find the target word in the reconstructed text
            word_start = 0
            while True:
                word_start = full_text.find(target_word, word_start)
                if word_start == -1:
                    break
                
                # Find which token(s) this corresponds to
                target_token_indices = []
                for j, (start, end, _) in enumerate(offset_mapping):
                    if start <= word_start < end or start < word_start + len(target_word) <= end:
                        target_token_indices.append(j)
                
                # Only use if it's a complete word (preceded by space or start of text)
                if word_start == 0 or full_text[word_start-1].isspace():
                    matched_indices.extend(target_token_indices)
                    print(f"Fuzzy match for '{target_word}' at text position {word_start}, tokens {target_token_indices}")
                
                word_start += 1
        
        if not matched_indices:
            print(f"WARNING: Target word '{target_word}' not found in sentence {i}")
            continue
            
        # Now get the embeddings from the model
        with torch.no_grad():
            # Get hidden states from all layers
            outputs = model(**encoded_input, output_hidden_states=True, output_attentions=True)
            
            # Use the specified layer (default: last layer)
            hidden_states = outputs.hidden_states[layer_num][0]
            
            # Get attention weights from the last layer
            attentions = outputs.attentions[-1][0]  # Shape: [num_heads, seq_len, seq_len]
            avg_attention = attentions.mean(dim=0)  # Average across heads
            
            # If we found multiple tokens that make up our word, average their embeddings
            if matched_indices:
                # Get embeddings for each matched token
                token_embeddings = [hidden_states[idx].numpy() for idx in matched_indices]
                
                # Average the embeddings if there are multiple tokens
                word_embedding = np.mean(token_embeddings, axis=0)
                
                # Store the embedding
                word_embeddings[i] = word_embedding
                
                # Store detailed information for analysis
                detailed_info[i] = {
                    'tokens': [tokens[idx] for idx in matched_indices],
                    'indices': matched_indices,
                    'embedding_shape': word_embedding.shape,
                    'first_5_values': word_embedding[:5],
                    'attention': [avg_attention[idx].numpy() for idx in matched_indices]
                }
                
                print(f"Embedding shape: {word_embedding.shape}")
                print(f"First 5 values: {word_embedding[:5]}")
    
    return word_embeddings, detailed_info

def analyze_embeddings_multiple_layers(model, tokenizer, sentences, target_word, layers=[1, 4, 8, 12]):
    """
    Analyze embeddings from different layers of the model.
    
    Args:
        model: The transformer model
        tokenizer: The tokenizer for the model
        sentences: List of sentences containing the target word
        target_word: The word to extract embeddings for
        layers: List of layer indices to analyze
    """
    results = {}
    
    for layer in layers:
        if layer >= len(model.encoder.layer):
            actual_layer = -1  # Use last layer
        else:
            actual_layer = layer
            
        print(f"\n\n===== ANALYSIS FOR LAYER {layer} =====")
        embeddings, detailed_info = get_contextual_word_embeddings_improved(
            model, tokenizer, sentences, target_word, layer_num=actual_layer)
        
        if len(embeddings) < 2:
            print(f"Not enough embeddings found in layer {layer}. Skipping analysis.")
            continue
            
        # Create a similarity matrix
        indices = sorted(embeddings.keys())
        n = len(indices)
        sim_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                idx_i = indices[i]
                idx_j = indices[j]
                emb_i = embeddings[idx_i].reshape(1, -1)
                emb_j = embeddings[idx_j].reshape(1, -1)
                sim_matrix[i, j] = cosine_similarity(emb_i, emb_j)[0][0]
        
        # Visualize similarity matrix as heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(sim_matrix, annot=True, fmt=".4f", 
                   xticklabels=[f"Sent {idx}" for idx in indices],
                   yticklabels=[f"Sent {idx}" for idx in indices],
                   cmap="YlGnBu")
        plt.title(f"Cosine Similarity of '{target_word}' Embeddings from Layer {layer}")
        plt.tight_layout()
        plt.show()
        
        # Dimensionality reduction with PCA
        if len(embeddings) >= 2:  # Need at least 2 samples for PCA
            # Stack embeddings
            X = np.vstack([embeddings[idx] for idx in sorted(embeddings.keys())])
            
            # Apply PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            
            # Plot
            plt.figure(figsize=(12, 10))
            plt.scatter(X_pca[:, 0], X_pca[:, 1], s=100)
            
            # Add labels
            for i, idx in enumerate(sorted(embeddings.keys())):
                plt.annotate(f"Sent {idx}: {sentences[idx][:30]}...", 
                           xy=(X_pca[i, 0], X_pca[i, 1]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=12)
                
            plt.title(f"PCA of '{target_word}' Embeddings from Layer {layer}")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            
            # Calculate explained variance
            print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
            print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.4f}")
        
        # Store results
        results[layer] = {
            'embeddings': embeddings,
            'detailed_info': detailed_info,
            'similarity_matrix': sim_matrix
        }
    
    return results

def extract_attention_patterns(model, tokenizer, sentences, target_word):
    """
    Extract and analyze attention patterns for the target word.
    
    Args:
        model: The transformer model
        tokenizer: The tokenizer for the model
        sentences: List of sentences containing the target word
        target_word: The word to extract embeddings for
    """
    model.eval()
    attention_patterns = {}
    
    for i, sentence in enumerate(sentences):
        # Tokenize the sentence
        encoded_input = tokenizer(sentence, return_tensors='pt')
        
        # Find the target word token
        input_ids = encoded_input['input_ids'][0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        
        # Find the most likely token for our target word
        target_idx = None
        for j, token in enumerate(tokens):
            if token == f'▁{target_word}' or token == target_word:
                target_idx = j
                break
        
        if target_idx is None:
            print(f"Target word '{target_word}' not found in sentence {i}. Skipping.")
            continue
        
        # Get attention patterns
        with torch.no_grad():
            outputs = model(**encoded_input, output_attentions=True)
            
            # Get attention from all layers
            all_attentions = outputs.attentions  # Tuple of tensors, one per layer
            
            # Store attention patterns
            attention_patterns[i] = {
                'sentence': sentence,
                'tokens': tokens,
                'target_idx': target_idx,
                'attention': [layer_attn[0] for layer_attn in all_attentions]  # Get first batch item
            }
    
    # Visualize attention patterns for each sentence
    for idx, data in attention_patterns.items():
        # Get the last layer attention
        last_layer_attention = data['attention'][-1]  # Shape: [num_heads, seq_len, seq_len]
        
        # Average across heads
        avg_attention = last_layer_attention.mean(dim=0)
        
        # Get attention from target word to all other tokens
        target_idx = data['target_idx']
        target_attention = avg_attention[target_idx, :].cpu().numpy()
        
        # Visualize
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(target_attention)), target_attention)
        plt.xticks(range(len(target_attention)), data['tokens'], rotation=90)
        plt.xlabel('Tokens')
        plt.ylabel('Attention Weight')
        plt.title(f"Attention from '{target_word}' to Other Tokens in Sentence {idx}")
        plt.tight_layout()
        plt.show()
    
    return attention_patterns

def compare_models(sentences, target_word, model_names=["xlm-roberta-base", "bert-base-multilingual-cased"]):
    """
    Compare embeddings from different models.
    
    Args:
        sentences: List of sentences containing the target word
        target_word: The word to extract embeddings for
        model_names: List of model names to compare
    """
    results = {}
    
    for model_name in model_names:
        print(f"\n\n===== ANALYSIS FOR MODEL {model_name} =====")
        
        # Load model and tokenizer
        if "bert-base-multilingual-cased" in model_name:
            from transformers import BertTokenizer, BertModel
            tokenizer = BertTokenizer.from_pretrained(model_name)
            model = BertModel.from_pretrained(model_name)
        else:
            tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
            model = XLMRobertaModel.from_pretrained(model_name)
        
        # Get embeddings
        embeddings, detailed_info = get_contextual_word_embeddings_improved(
            model, tokenizer, sentences, target_word)
        
        if len(embeddings) < 2:
            print(f"Not enough embeddings found for model {model_name}. Skipping analysis.")
            continue
        
        # Create a similarity matrix
        indices = sorted(embeddings.keys())
        n = len(indices)
        sim_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                idx_i = indices[i]
                idx_j = indices[j]
                emb_i = embeddings[idx_i].reshape(1, -1)
                emb_j = embeddings[idx_j].reshape(1, -1)
                sim_matrix[i, j] = cosine_similarity(emb_i, emb_j)[0][0]
        
        # Visualize similarity matrix as heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(sim_matrix, annot=True, fmt=".4f", 
                   xticklabels=[f"Sent {idx}" for idx in indices],
                   yticklabels=[f"Sent {idx}" for idx in indices],
                   cmap="YlGnBu")
        plt.title(f"Cosine Similarity of '{target_word}' Embeddings from {model_name}")
        plt.tight_layout()
        plt.show()
        
        # Store results
        results[model_name] = {
            'embeddings': embeddings,
            'detailed_info': detailed_info,
            'similarity_matrix': sim_matrix
        }
    
    return results

# Main execution
if __name__ == "__main__":
    # Load model
    model_name = "xlm-roberta-base"
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
    model = XLMRobertaModel.from_pretrained(model_name)
    
    # Define Azerbaijani sentences with the target word in different contexts
    target_word = "bal"
    sentences = [
        "Mən bal yeməyi çox sevirəm.",  # "bal" as honey
        "İmtahanda topladığım bal 95 idi.",  # "bal" as score
        "Arılar bu il çox keyfiyyətli bal veriblər.",  # "bal" as honey
        "Onun konsertdə göstərdiyi performans üçün maksimum bal verdilər."  # "bal" as score
    ]
    
    # 1. Analyze embeddings from different layers
    layer_results = analyze_embeddings_multiple_layers(
        model, tokenizer, sentences, target_word, layers=[1, 4, 8, 12])
    
    # 2. Extract and analyze attention patterns
    attention_patterns = extract_attention_patterns(model, tokenizer, sentences, target_word)
    
    # 3. Compare with another model (optional)
    model_comparison = compare_models(sentences, target_word, 
                                     model_names=["xlm-roberta-base", "bert-base-multilingual-cased"])