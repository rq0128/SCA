import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig

class ConsistencyCrossEncoder(nn.Module):
    """
    UniXcoder-based Cross-Encoder model.
    Used to judge the semantic consistency between a [description sentence]
    and a [code snippet].
    """
    def __init__(self, model_name="microsoft/unixcoder-base", num_classes=2):
        super().__init__()
        
        print(f"[INFO] Loading UniXcoder Cross-Encoder from: {model_name}...")
        
        # 1. Load the pretrained UniXcoder
        self.encoder = AutoModel.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        
        # 2. Classification head
        # Standard RoBERTa classification head: Dropout -> Linear -> Tanh -> Linear
        hidden_size = self.config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(), # Non-linear activation to help fit complex logical relations
            nn.Linear(hidden_size, num_classes) 
        )
        
        # 3. Loss function
        # Use CrossEntropyLoss to train the model to distinguish "consistent (1)" from "inconsistent (0)"
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass.
        :param input_ids: [Batch, Seq_Len] (format: <s> Desc </s></s> Code </s>)
        :param attention_mask: [Batch, Seq_Len]
        :param labels: [Batch] (optional, 0 or 1)
        """
        # 1. Extract features via UniXcoder
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # 2. Obtain the [CLS] vector
        # For UniXcoder (RoBERTa-based), the CLS token is typically the first token of the sequence
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # 3. Produce logits through the classification head
        logits = self.classifier(cls_embedding) # [Batch, 2]
        
        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)
            
        return {
            'loss': loss,
            'logits': logits,
            'cls_embedding': cls_embedding
        }