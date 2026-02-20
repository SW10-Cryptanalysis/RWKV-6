import torch
from fla.models.rwkv6 import RWKV6Config, RWKV6ForCausalLM
from config import Config


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
    return model.to(torch.bfloat16).cuda()
