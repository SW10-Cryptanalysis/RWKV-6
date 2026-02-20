import torch
from fla.models.rwkv6 import RWKV6Config, RWKV6ForCausalLM
from config import Config
import torch.nn as nn

def replace_layer_norms(model):
    for name, module in model.named_children():
        if module.__class__.__name__.lower().startswith("layernorm"):
            setattr(model, name, nn.LayerNorm(module.normalized_shape))
        else:
            replace_layer_norms(module)

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

    replace_layer_norms(model)  # 👈 important

    return model.to(torch.bfloat16).cuda()