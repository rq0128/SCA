import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from code_parser import MMDEdgeCalibratedSegmenter
from desc_parser import DescriptionParser  
from config import Config
from model import ConsistencyCrossEncoder 

class PoisonDetector:
    def __init__(self, model_path='../model/best_cross_encoder.pth'):
        self.device = Config.DEVICE
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        
        print(f"[INFO] Initializing Poison Detector...")
        print(f"   - Model: {Config.MODEL_NAME}")
        print(f"   - Loading weights from: {model_path}")
        
        # 1. Load model architecture
        self.model = ConsistencyCrossEncoder(model_name=Config.MODEL_NAME)
        
        # 2. Load weights
        try:
            if torch.cuda.is_available():
                state_dict = torch.load(model_path)
            else:
                state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            self.model.load_state_dict(state_dict)
        except FileNotFoundError:
            print(f"[WARNING] Model file not found at {model_path}. Running with random weights (for testing only).")
        
        self.model.to(self.device)
        self.model.eval()
        
        # 3. Initialize the two parsers
        print("   - Initializing Code & Description Parsers...")
        self.code_parser = MMDEdgeCalibratedSegmenter()
        self.desc_parser = DescriptionParser() # Handles docstrings and complex syntax

    def detect(self, description, code, threshold=0.5, verbose=True):
        """
        Add a verbose parameter; set to False during batch testing to avoid log flooding.
        """
        # 1. Extract code features (ground truth)
        code_features = self.code_parser.segment_code_mmd_calibrated(code)
        
        if not code_features:
            if verbose: print("[WARNING] Code is empty or unparsable. Cannot verify description.")
            return {"is_poisoned": False, "reason": "No code features"}

        # 2. Split the description (using the NLP parser)
        desc_sentences = self.desc_parser.parse_sentences(description)

        suspicious_sentences =[]
        verified_sentences =[]

        # 3. Verify sentence by sentence (verification loop)
        for sent in desc_sentences:
            pairs = [[sent, feat] for feat in code_features]
            
            encoded = self.tokenizer(
                [p[0] for p in pairs], 
                [p[1] for p in pairs],
                padding=True, truncation=True, max_length=128, return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(encoded['input_ids'], encoded['attention_mask'])
                logits = outputs['logits'] # [Num_Features, 2]
                
                # Probability of label 1 (consistent)
                probs = torch.softmax(logits, dim=1)[:, 1]
            
            # Core logic: find the maximum support score this sentence has in the code
            max_support_score = probs.max().item()
            best_feature_idx = probs.argmax().item()
            best_feature = code_features[best_feature_idx]
            
            if verbose:
                print(f"\n   [SENTENCE] '{sent}'")
                print(f"      Best Support: '{best_feature}' (Score: {max_support_score:.4f})")
            
            # 4. Decision
            if max_support_score < threshold:
                if verbose: print(f"      [ALERT] No code evidence found! Potential poisoning.")
                suspicious_sentences.append({
                    'sentence': sent,
                    'best_feature': best_feature,
                    'support_score': max_support_score
                })
            else:
                if verbose: print(f"      [OK] Verified.")
                verified_sentences.append(sent)

        # 5. Final report
        is_poisoned = len(suspicious_sentences) > 0
        
        if verbose:
            print("\n" + "="*60)
            if is_poisoned:
                print(f"MALICIOUS POISONING DETECTED!")
                print(f"Found {len(suspicious_sentences)} sentences with no code backing:")
                for item in suspicious_sentences:
                    print(f"  - \"{item['sentence']}\"")
                    print(f"    (Max support: {item['support_score']:.4f} from '{item['best_feature']}')")
            else:
                print(f"CLEAN. All description sentences are supported by code logic.")
            print("="*60)
        
        return {
            "is_poisoned": is_poisoned,
            "suspicious_items": suspicious_sentences,
            "verified_count": len(verified_sentences)
        }

def evaluate_detector(detector, csv_path, threshold=0.5):
    print(f"\n[INFO] Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Metric counters
    TP = 0  # Actually malicious (0), predicted poisoned (True)
    TN = 0  # Actually benign (1), predicted clean (False)
    FP = 0  # Actually benign (1), predicted poisoned (True) -> false positive
    FN = 0  # Actually malicious (0), predicted clean (False) -> false negative
    
    # Used to record details of false positives
    fp_details = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Scanning Test Dataset"):
        desc = str(row['description'])
        code = str(row['code'])
        label = int(row['label']) 
        
        # Get the tool id; prefer the 'id' or 'tool_id' column, otherwise use the row index
        tool_id = row.get('id', row.get('tool_id', f"Index_{index}"))
        
        # Disable verbose printing during batch evaluation
        result = detector.detect(desc, code, threshold=threshold, verbose=False)
        is_poisoned = result["is_poisoned"]
        
        if label == 0:  # Actually malicious poisoning
            if is_poisoned:
                TP += 1
            else:
                FN += 1
        elif label == 1:  # Actually benign code
            if is_poisoned:
                FP += 1
                # Record false-positive info: tool id, the sentence that caused it, and its best match score
                fp_details.append({
                    "id": tool_id,
                    "suspicious_items": result["suspicious_items"]
                })
            else:
                TN += 1

    # Compute metrics
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # 2. Print the formatted evaluation report
    print(f"Total Samples: {len(df)}")
    print(f"Precision  : {precision:.4f}  ({precision*100:.2f}%)")
    print(f"Recall  : {recall:.4f}  ({recall*100:.2f}%)")
    print(f"F1-Score  : {f1_score:.4f}  ({f1_score*100:.2f}%)")

    
def demonstrate_case_study(detector, description, code, label, threshold=0.1, index=0):
    # 1. Preprocessing
    code_chunks = detector.code_parser.segment_code_mmd_calibrated(code)
    desc_sents = detector.desc_parser.parse_sentences(description)
    
    print("-" * 80)
    # Header design
    print(f"{'meta action':<33} | {'code block':<26} | {'score'}")
    print("-" * 80)

    for sent in desc_sents:
        # Compute the score between this intent and all code chunks
        pairs = [[sent, feat] for feat in code_chunks]
        encoded = detector.tokenizer([p[0] for p in pairs], [p[1] for p in pairs], 
                                   padding=True, truncation=True, max_length=128, return_tensors='pt').to(detector.device)
        
        with torch.no_grad():
            outputs = detector.model(encoded['input_ids'], encoded['attention_mask'])
            probs = torch.softmax(outputs['logits'], dim=1)[:, 1]
            
        max_score = probs.max().item()
        best_idx = probs.argmax().item()
        
        # Prepare the sentence and code summary for display
        display_sent = (sent[:32] + '..') if len(sent) > 32 else sent
        
        if max_score >= threshold:
            # Brief representation of the code chunk (take the first non-empty line and collapse whitespace)
            best_chunk_raw = code_chunks[best_idx].strip().split('\n')[0]
            if len(best_chunk_raw) > 28:
                evidence = best_chunk_raw[:25] + "..."
            else:
                evidence = best_chunk_raw
            status_prefix = "[OK]"
        else:
            evidence = "Support code not found"
            status_prefix = "[ALERT]"

        # Print one formatted row
        print(f"{status_prefix} {display_sent:<35} | {evidence:<30} | {max_score:.4f}")

    print("-" * 80)
    
    # 5. Final conclusion
    result = detector.detect(description, code, threshold=threshold, verbose=False)
    print(f"\n[RESULT]")
    if result["is_poisoned"]:
        print(">>> poisoned")
    else:
        print(">>> benign")

if __name__ == "__main__":
    # Initialize detector
    detector = PoisonDetector(model_path='../model/best_cross_encoder_final.pth')
    test_csv_path = "/home/lrq/MCPDetector_new/data/test_data_mcpzoo_new.csv"
    
    df = pd.read_csv(test_csv_path)
    
    threshold = 0.1       # Decision threshold
    num_fps_to_show = 10   # Number of false-positive cases to inspect
    count = 0

    evaluate_detector(detector, test_csv_path)