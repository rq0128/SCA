import torch
from transformers import AutoTokenizer

# Make sure these are imported correctly from your project files
from code_ast import SemanticCodeParser
from desc_parser import DescriptionParser  
from config import Config
from model import ConsistencyCrossEncoder 

class PoisonDetector:
    def __init__(self, model_path='../model/best_cross_encoder.pth'):
        self.device = Config.DEVICE
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        
        print(f"[INFO] Initializing detection model...")
        self.model = ConsistencyCrossEncoder(model_name=Config.MODEL_NAME)
        
        try:
            # Handle CPU/GPU weight loading automatically
            map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            state_dict = torch.load(model_path, map_location=map_location)
            self.model.load_state_dict(state_dict)
            print(f"[OK] Successfully loaded weights: {model_path}")
        except FileNotFoundError:
            print(f"[WARNING] Model file not found; running with random weights (for structure testing only)")
        
        self.model.to(self.device)
        self.model.eval()
        
        self.code_parser = MMDEdgeCalibratedSegmenter()
        self.desc_parser = DescriptionParser()

    def detect_and_explain(self, description, code, threshold=0.5):
        """
        Perform in-depth analysis on the input description and code, then print the result.
        """
        print("\n" + "="*80)
        print("[INFO] Starting consistency analysis...")
        print("="*80)

        # 1. Parse code features
        code_features = self.code_parser.segment_code_mmd_calibrated(code)
        if not code_features:
            print("[ERROR] Could not extract any features from the code.")
            return

        # 2. Parse description sentences
        desc_sentences = self.desc_parser.parse_sentences(description)
        
        print(f"[INFO] Input analysis:")
        print(f"   - Description sentences: {len(desc_sentences)}")
        print(f"   - Code features: {len(code_features)}")
        print(f"   - Decision threshold: {threshold}")
        print("-" * 80)
        print(f"{'Status':<8} | {'Score':<8} | {'Description Sentence'} -> {'Best Code Feature'}")
        print("-" * 80)

        suspicious_count = 0

        # 3. Compare sentence by sentence
        for sent in desc_sentences:
            # Build [sentence, code feature] pairs
            pairs = [[sent, feat] for feat in code_features]
            
            encoded = self.tokenizer(
                [p[0] for p in pairs], 
                [p[1] for p in pairs],
                padding=True, truncation=True, max_length=128, return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(encoded['input_ids'], encoded['attention_mask'])
                # Probability of label 1 (consistent)
                probs = torch.softmax(outputs['logits'], dim=1)[:, 1]
            
            # Find the code feature with the highest support score
            max_score = probs.max().item()
            best_feat_idx = probs.argmax().item()
            best_feature = code_features[best_feat_idx]
            
            # Decision
            status_icon = "[OK]"
            if max_score < threshold:
                status_icon = "[ALERT]"
                suspicious_count += 1
            
            # Print the detail row (truncate overly long sentences for alignment)
            short_sent = (sent[:35] + '..') if len(sent) > 35 else sent
            short_feat = (best_feature[:35] + '..') if len(best_feature) > 35 else best_feature
            print(f"{status_icon:<8} | {max_score:.4f} | {short_sent:<37} -> {short_feat}")

        # 4. Summary report
        print("-" * 80)
        if suspicious_count > 0:
            print(f"[ERROR] Final verdict: [MALICIOUS]")
            print(f"Reason: {suspicious_count} sentence(s) in the description have no logical support in the code.")
        else:
            print(f"[OK] Final verdict: [CLEAN]")
        print("="*80 + "\n")


if __name__ == "__main__":
    # 1. Configure the model path 
    MODEL_FILE = '../model/best_cross_encoder.pth'
    detector = PoisonDetector(model_path=MODEL_FILE)

    # 2. Enter the specific description and code you want to test here
    

    # Example B: poisoned sample 
    test_description_2 = "Get the daily performance for a brokerage account.Returns the value of the account portfolio over time.Outputs a JSON object with the following fields:- dates: List[str]. The dates for which performance is available.- series: List[float]. The total value of the portfolio on the given date."
    test_code_2 = """
    async def get_portfolio_daily_performance(account_uuid: str) -> Dict:
        try:
            url = f"{get_base_url()}/api/v0.1/portfolio/accounts/{account_uuid}/portfolio-history"
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers=get_required_headers(),
                )
            data = response.json()
            data['dates'] = [epoch_ms_to_date(d) for d in data['epoch_ms']]
            del data['epoch_ms']
            return data
        except Exception as e:
            return {"error": truncate_text(str(e), 1000)}
    """

    # 3. Run the test 

    print("\n>>> Test case:")
    detector.detect_and_explain(test_description_2, test_code_2, threshold=0.1)