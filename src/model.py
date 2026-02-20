import torch
import torch.nn as nn
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

    # Collect candidate norm modules first (avoid mutating while iterating)
    candidates = [(name, module) for name, module in model.named_modules() if "norm" in name.lower()]

    for name, module in candidates:
        # Skip if already a native nn.LayerNorm
        if isinstance(module, nn.LayerNorm):
            continue

        # Determine normalized shape
        if hasattr(module, "normalized_shape") and module.normalized_shape is not None:
            norm_shape = module.normalized_shape
        elif hasattr(module, "weight") and module.weight is not None:
            norm_shape = module.weight.shape[-1]
        else:
            norm_shape = Config.dims

        eps = getattr(module, "eps", 1e-5)

        new_ln = nn.LayerNorm(norm_shape, eps=eps)

        # Try to copy weight/bias if present on the original module
        try:
            if hasattr(module, "weight") and module.weight is not None:
                with torch.no_grad():
                    new_ln.weight.data = module.weight.data.to(new_ln.weight.dtype).clone()
            if hasattr(module, "bias") and module.bias is not None:
                with torch.no_grad():
                    new_ln.bias.data = module.bias.data.to(new_ln.bias.dtype).clone()
        except Exception:
            # If copying fails, continue with freshly initialized LayerNorm
            pass

        # Assign new LayerNorm into the model by locating its parent
        parent = model
        parts = name.split('.')
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], new_ln)

    return model.to(torch.bfloat16).cuda()