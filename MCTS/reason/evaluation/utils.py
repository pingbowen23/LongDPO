import json
from typing import Optional
import random
import numpy as np
import os
import torch
from dataclasses import dataclass
import argparse
from argparse import ArgumentParser
from config.config_utils import str2bool

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--LM", type=str, )
    parser.add_argument("--RM", type=str, default="dummy")
    parser.add_argument("--len_scale", type=int, default=10, required=True)
    parser.add_argument("--controller_addr", type=str, default="http://0.0.0.0:28778")
    # task config
    parser.add_argument("--task_name", type=str, default="gsm8k")
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--test", type=str2bool, default=True)
    parser.add_argument("--is_few_shot", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=0)
    # method config
    parser.add_argument("--method", type=str, )
    parser.add_argument("--num_sequence", type=int, default=1)
    # LM gen config
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    # Tree construction config
    parser.add_argument("--tree_max_depth", type=int, default=None)
    parser.add_argument("--tree_max_width", type=int, default=None)
    # ckpg config
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--save_step_dir", type=str, default=None)
    parser.add_argument("--save_refine_dir", type=str, default=None)
    parser.add_argument("--group_init_index", type=int, default=0)
    parser.add_argument("--resume_dir", type=str, default=None)
    # parallel config
    parser.add_argument("--local", action="store_true", default=False)
    parser.add_argument("--num_worker", type=int, default=32)
    parser.add_argument("--reward_model_addr", type=int, default=8000)
    return parser.parse_args()


def write_to_jsonl(data, output_file):
    cnt = 0
    with open(output_file, "w") as outfile:
        for item in data:
            outfile.write(json.dumps(item) + "\n")
            cnt += len(item["answer"])
        print("Write {} items into {}".format(cnt, output_file))


def load_jsonl(file_path):
    data_list = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line.strip())
            data_list.append(data)
    return data_list


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
