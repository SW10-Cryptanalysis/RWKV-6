import os

os.environ["TRITON_DISABLE_CACHE"] = "1"
os.environ["TRITON_CACHE_DIR"] = "/tmp/triton"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from model import get_model
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
from config import Config

# Speed up training on NVIDIA GPUs (Ampere and newer)
torch.backends.cuda.matmul.allow_tf32 = True

class CipherPlainData(Dataset):
    def __init__(self):
        # Placeholder for your specific data loading logic
        print("Loading RWKV-v6 test dataset...")
        self.data_len = 1000

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        # Mock data: RWKV-6 expects standard input_ids and labels
        seq = torch.randint(0, Config.vocab_size, (Config.max_context,))
        return {
            "input_ids": seq,
            "labels": seq.clone(), # Standard Causal LM training
        }

def train():
    # Load the RWKV-6 Finch model we defined earlier
    model = get_model()

    args = TrainingArguments(
        output_dir=Config.output_dir,
        num_train_epochs=Config.epochs,
        per_device_train_batch_size=Config.batch_size,
        gradient_accumulation_steps=Config.grad_accum,
        learning_rate=Config.learning_rate,
        weight_decay=Config.weight_decay,
        gradient_checkpointing=Config.grad_checkpoint,
        logging_steps=Config.log_steps,
        save_steps=Config.save_steps,
        bf16=True, # RWKV-6 performs best in bfloat16
        optim="adamw_torch_fused",
        remove_unused_columns=False,
        # FLA models support efficient report logging
        report_to="none", 
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=CipherPlainData(),
    )

    print(f"Training RWKV-v6 (Finch) on {torch.cuda.get_device_name(0)}...")
    
    # Note: No 'sdpa_kernel' needed here; FLA handles its own Triton kernels.
    trainer.train()

    trainer.save_model(f"{Config.output_dir}/model")

if __name__ == "__main__":
    train()