TEXT_LEN = 2048
UNIQUE_HOMOPHONE_COUNT = 500
UNIQUE_LETTER_COUNT = 26
TOTAL_SEQ = (TEXT_LEN * 2) + 1
OUTPUT_DIR = "./outputs"


class Config:
    unique_homophones: int = UNIQUE_HOMOPHONE_COUNT
    unique_letters: int = UNIQUE_LETTER_COUNT
    vocab_size: int = unique_homophones + unique_letters + 1
    max_context = TOTAL_SEQ
    dims: int = 512
    layers: int = 12
    att_heads: int = 8

    # TRAINING
    batch_size: int = 1
    grad_accum: int = 16
    learning_rate: float = 3e-4
    epochs: int = 1
    weight_decay: float = 0.01
    grad_checkpoint: bool = True
    log_steps: int = 10
    save_steps: int = 500

    # SYSTEM
    output_dir: str = OUTPUT_DIR
