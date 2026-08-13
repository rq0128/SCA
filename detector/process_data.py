import pandas as pd
import random
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from code_ast import SemanticCodeParser
from desc_parser import DescriptionParser  # Import the NLP parser

# Configuration
MODEL_NAME = "microsoft/unixcoder-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_embeddings(text_list, tokenizer, model):
    """Batch-compute UniXcoder embeddings."""
    if not text_list:
        return None
    # Features are usually short, so max_length=128 is sufficient and faster
    inputs = tokenizer(text_list, padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        # Obtain the [CLS] vector
        embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings

def generate_feature_aligned_dataset(input_csv, output_csv):
    print("[INFO] Initializing Feature Alignment Engine...")
    
    # 1. Prepare model and parsers
    print("   - Loading UniXcoder model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    
    print("   - Initializing parsers...")
    code_parser = SemanticCodeParser()
    desc_parser = DescriptionParser()  
    
    df = pd.read_csv(input_csv)
    print(f"[INFO] Reading data from {input_csv} ({len(df)} rows)...")
    
    # 2. Build a global feature pool for negative sampling
    print("[INFO] Building global feature pool...")
    global_feature_pool = []
    
    # Pre-extract features of all code snippets
    for code in tqdm(df['code'], desc="Parsing Code Features"):
        if isinstance(code, str):
            feats = code_parser.parse_features(code)
            global_feature_pool.extend(feats)
    
    # Cap the pool size to avoid excessive memory usage
    if len(global_feature_pool) > 10000:
        global_feature_pool = random.sample(global_feature_pool, 10000)
    
    new_data = []
    print("[INFO] Aligning sentences to code features...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing Tools"):
        description = str(row.get('description', ''))
        code = str(row.get('code', ''))
        label = int(row.get('label', 1))
        
        # Only use benign tools to build positive samples (ground truth)
        if label == 0:
            continue
            
        # === Use the NLP parser to split the description ===
        # This handles docstring format, merges multi-line text, and splits compound sentences
        sents = desc_parser.parse_sentences(description)
        
        # Extract code features
        code_features = code_parser.parse_features(code)
        
        if not sents or not code_features:
            continue
            
        # Compute the similarity matrix
        sent_embs = get_embeddings(sents, tokenizer, model)
        feat_embs = get_embeddings(code_features, tokenizer, model)
        
        if sent_embs is None or feat_embs is None:
            continue

        # [Num_Sents, Num_Features]
        sim_matrix = torch.matmul(sent_embs, feat_embs.T)
        best_scores, best_indices = sim_matrix.max(dim=1)
        
        for i in range(len(sents)):
            sent = sents[i]
            best_idx = best_indices[i].item()
            score = best_scores[i].item()
            
            target_feature = code_features[best_idx]
            
            # --- Positive sample ---
            if score > 0.1:
                new_data.append({
                    'description': sent,
                    'code': target_feature,
                    'label': 1,
                    'alignment_score': score
                })
                
                # --- Hard negative sample ---
                # Set the desired negative-sample ratio, e.g. negative_ratio = 3
                negative_ratio = 3 
                neg_count = 0
                
                # Allow more attempts to ensure enough unique negatives are selected
                for _ in range(10): 
                    if neg_count >= negative_ratio:
                        break
                        
                    random_feat = random.choice(global_feature_pool)
                    
                    # Ensure the randomly chosen code feature differs from the true feature of this description
                    if random_feat != target_feature:
                        new_data.append({
                            'description': sent,
                            'code': random_feat,
                            'label': 0,
                            'alignment_score': 0.0
                        })
                        neg_count += 1

    new_df = pd.DataFrame(new_data)
    # Shuffle the order
    new_df = new_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"[OK] Generated {len(new_df)} feature-grained training pairs.")
    new_df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    # Make sure the paths are correct
    generate_feature_aligned_dataset('../data/train_data_clean.csv', '../data/aligned_train_data_new.csv')