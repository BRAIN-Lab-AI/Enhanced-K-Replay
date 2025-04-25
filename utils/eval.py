# 测试模型
# 为验证、测试集生成句子并保存为可用pycoco直接计算指标的格式
# 用保存的句子计算指标

import os
import torch
import pickle
import json
import numpy as np

from data_load import data_load
from tqdm import tqdm
from pycocoevalcap.eval import COCOEvalCap
from evaluation import Cider

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_captions(config, model, step, mode, final_test=False):
    print("Generating captions...")

    log_path = config.log_dir.format(config.id)
    # Print the current working directory
    print("Current Working Directory:", os.getcwd())
    result_dir = os.path.join(log_path, 'generated')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    # Print the current working directory
    gen_pycoco_path = os.path.join(result_dir, mode+'_'+str(step)+'.json')
    data_dir = os.path.join(config.data_dir, mode+'.json')
    print("config.data_dir:", config.data_dir)
    print("data_dir:", data_dir)
    print("mode:", mode)


    eval_loader = data_load(config, data_dir, mode)
    model.eval()
    gen_pycoco = {}

    for i, (image_id, image_feature) in tqdm(enumerate(eval_loader)):
        patch_image = image_feature['patch_image']
        patch_image = patch_image.to(device)
        batch_size = len(image_id)
        if not final_test:
            captions, _ = model.greedy_search(patch_image)
        else:
            captions = model.generate_caption_batchbs(patch_image)
        for j, cap_id in enumerate(captions):
            if config.model == 'OFA':
                gen = cap_id.unsqueeze(0)
                caption = model.tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()
            elif config.model == 'BLIP':
                caption = model.tokenizer.decode(cap_id, skip_special_tokens=True)
                caption = caption[len(model.prompt):]
            elif config.model == 'GIT':
                gen = cap_id.unsqueeze(0)
                caption = model.tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()
            refs = []
            ref = {'image_id': image_id[j], 'id': i * batch_size + j, 'caption': caption}
            refs.append(ref)
            gen_pycoco[i * batch_size + j] = refs
        if not final_test:
            if len(gen_pycoco) >= 200:
                break

    json.dump(gen_pycoco, open(gen_pycoco_path, 'w'), ensure_ascii=False)

    return gen_pycoco_path




def eval_pycoco(config, gen_pycoco_path, mode):
    print("Calculating pycoco...")
    ref_pycoco_path = os.path.join(config.data_dir, mode+'_pycoco.json')
    ref_pycoco = json.load(open(ref_pycoco_path, 'r'))
    gen_pycoco = json.load(open(gen_pycoco_path, 'r'))
    
    # Ensure that the number of generated captions matches the reference captions
    num = len(gen_pycoco)
    ref_pycoco = {int(k): v for k, v in ref_pycoco.items() if int(k) < num}  # Ensure keys are integers
    gen_pycoco = {int(k): v for k, v in gen_pycoco.items() if int(k) < num}

    cocoEval = COCOEvalCap('diy', 'diy')
    pycoco_results = cocoEval.evaluate_diy(ref_pycoco, gen_pycoco)

  
    return pycoco_results