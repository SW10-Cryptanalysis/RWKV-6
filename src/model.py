import torch
from fla.models.rwkv6 import RWKV6Config, RWKV6ForCausalLM
from config import Config
import torch.nn as nn

def replace_layer_norms(module):
    for name, child in module.named_children():
        # Detect any FLA / custom norm
        if "norm" in child.__class__.__name__.lower():
            if hasattr(child, "weight") and child.weight is not None:
                shape = child.weight.shape
                new_ln = nn.LayerNorm(shape, eps=1e-5)
            else:
                # fallback if no weight (rare)
                new_ln = nn.LayerNorm(Config.dims, eps=1e-5)

            setattr(module, name, new_ln)
        else:
            replace_layer_norms(child)

def get_model():
    conf = RWKV6Config(
        vocab_size=Config.vocab_size,
        hidden_size=Config.dims,
        num_hidden_layers=Config.layers,
        num_attention_heads=Config.att_heads,
        intermediate_size=Config.dims * 4,
        max_position_embeddings=Config.max_context,
        use_cache=False,
    )

    model = RWKV6ForCausalLM(conf)

    replace_layer_norms(model)

    return model.to(torch.bfloat16).cuda()