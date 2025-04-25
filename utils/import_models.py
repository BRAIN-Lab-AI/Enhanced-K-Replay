# 根据命令行构建模型

import os
import yaml
from pathlib import Path
from models.Transformer.transformer import Transformer_Cap
from models.OFA.ofa import OFA


def construct_model(config):
    if config.model == 'Transformer':
        model = Transformer_Cap(config)
    elif config.model == 'OFA':
        model = OFA(config, use_patch_self_attn=config.use_patch_self_attn)
    else:
        print("model "+str(config.model)+" not found")
        return None
    return model