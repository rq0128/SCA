import torch

class Config:
    # Model configuration
    MODEL_NAME = "microsoft/unixcoder-base"
    MAX_LENGTH = 512  # Maximum input sequence length (description + code)

    # Label definitions
    LABEL_CONSISTENT = 1    # Code supports the description
    LABEL_INCONSISTENT = 0  # Code does not support the description (possible hallucination or malicious injection)

    # Training configuration
    BATCH_SIZE = 8
    LEARNING_RATE = 2e-5
    LEARNING_RATE_2 = 5e-6
    EPOCHS = 10
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"