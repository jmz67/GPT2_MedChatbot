import torch.nn as nn
from transformers import GPT2LMHeadModel
from lora import LoRALayer

class GPT2WithLoRA(nn.Module):
    def __init__(self, model_name = 'gpt2', r=8, alpha=1.0):
        super(GPT2WithLoRA, self).__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name)

        # 获取模型中的注意力层
        for module in self.gpt2.transformer.h:
            # 替换自注意力机制中的q_proj, k_proj, v_proj层
            module.attn.c_attn = LoRALayer(module.attn.c_attn.in_features, module.attn.c_attn.out_features, r, alpha)

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.gpt2(input_ids, attention_mask=attention_mask, labels=labels)
