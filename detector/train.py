import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW  
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import pandas as pd
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time

from model import ConsistencyCrossEncoder
from dataset import MCPConsistencyDataset
from config import Config

# Set the random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_metrics(preds, labels):
    """Compute classification metrics: accuracy, precision, recall, F1."""
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train():
    set_seed(42)
    
    # 1. Initialize paths and device
    print("[INFO] Starting training pipeline...")
    os.makedirs('../model', exist_ok=True)
    device = torch.device(Config.DEVICE)
    print(f"   Using device: {device}")
    
    # 2. Load data
    data_path = '../data/aligned_train_data_new.csv' 
    if not os.path.exists(data_path):
        print(f"[ERROR] Data not found at {data_path}. Please run preprocess_data.py first.")
        return

    print("[INFO] Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"   Total samples: {len(df)}")
    
    # Split into train / validation sets (80% / 20%)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
    print(f"   Train size: {len(train_df)} | Val size: {len(val_df)}")
    
    # 3. Prepare tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    
    train_dataset = MCPConsistencyDataset(train_df, tokenizer, max_len=Config.MAX_LENGTH)
    val_dataset = MCPConsistencyDataset(val_df, tokenizer, max_len=Config.MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # 4. Initialize model
    print("[INFO] Initializing Cross-Encoder model...")
    model = ConsistencyCrossEncoder(model_name=Config.MODEL_NAME, num_classes=2)
    model.to(device)
    
    # 5. Optimizer and scheduler
    # Use a low learning rate (2e-5) to avoid damaging the pretrained weights
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=0.01)
    
    total_steps = len(train_loader) * Config.EPOCHS
    # Warmup steps are typically 10% of the total steps
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                              num_warmup_steps=int(0.1*total_steps),
                                              num_training_steps=total_steps)
    
    # 6. Training loop
    best_val_f1 = 0.0
    
    for epoch in range(Config.EPOCHS):
        print(f"\n{'='*30}")
        print(f"Epoch {epoch+1}/{Config.EPOCHS}")
        print(f"{'='*30}")
        
        # --- Training Phase ---
        model.train()
        total_train_loss = 0
        train_preds = []
        train_labels = []
        
        start_time = time.time()
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            # Move data to GPU
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Zero gradients
            model.zero_grad()
            
            # Forward pass (returns a dict containing loss)
            outputs = model(input_ids, mask, labels=labels)
            loss = outputs['loss']
            logits = outputs['logits']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update parameters
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            
            # Record predictions to compute training metrics
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = total_train_loss / len(train_loader)
        train_metrics = compute_metrics(train_preds, train_labels)
        
        print(f"   [Train] Loss: {avg_train_loss:.4f} | F1: {train_metrics['f1']:.4f} | Acc: {train_metrics['accuracy']:.4f}")
        
        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, mask, labels=labels)
                loss = outputs['loss']
                logits = outputs['logits']
                
                total_val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_metrics = compute_metrics(val_preds, val_labels)
        epoch_time = time.time() - start_time
        
        print(f"   [Val]   Loss: {avg_val_loss:.4f} | F1: {val_metrics['f1']:.4f} | Acc: {val_metrics['accuracy']:.4f}")
        print(f"   Epoch Time: {epoch_time:.1f}s")
        
        # Save the best model (based on F1 score)
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            save_path = '../model/best_cross_encoder_final.pth'
            torch.save(model.state_dict(), save_path)
            print(f"[SAVE] Best model saved to {save_path} (F1 improved to {best_val_f1:.4f})")
        else:
            print(f"   (F1 did not improve from {best_val_f1:.4f})")

    print("\n[OK] Training completed.")

if __name__ == "__main__":
    import random
    train()