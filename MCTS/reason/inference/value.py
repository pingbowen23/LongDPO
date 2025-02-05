import torch
from typing import Union, List
from transformers import AutoTokenizer
import re
import numpy as np
import requests
import json
from typing import Optional
from openai import OpenAI
import concurrent.futures
from ..evaluation.utils import get_args
# Set OpenAI's API key and API base to use vLLM's API server.

def score(x, y, len_scale=10):
    if y > x:
        return 100 * max(0, 1. - (y / x - 1) / 3) / len_scale
    else:
        return 100 * max(0, 1. - (x / y - 1) / 2) / len_scale

def count_words(text):
    chinese_characters = re.findall(r'[\u4e00-\u9fff]', text)
    english_words = re.findall(r'\b[a-zA-Z]+\b', text)
    
    chinese_char_count = len(chinese_characters)
    english_word_count = len(english_words)
    
    total_count = chinese_char_count + english_word_count
    
    return total_count

def filter_history(history):
    history = history.split("[/INST]", 1)
    return history[-1]

def get_len_score(input_str):
    history = input_str[0][0]
    
    history_len = count_words(history)
    lens = []
    for _ , output in input_str:
        output_len = count_words(output)
        lens.append(output_len + history_len)
    return lens    

def extract_info(pattern, text):
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return None

def get_mcts_score(completion):
    total_scores = []
    # dims = ["Relevance", "Accuracy", "Coherence", "Clarity", "Breadth and Depth", "Reading Experience"]
    dims = ["principle1","principle2","principle3","principle4","principle5","principle6","principle7"]
    for idx,output in enumerate(completion.choices):
        try:
            output = extract_info(r'\{(.*?)\}', output.text)
            scores = json.loads('{' + output + '}')
            scores = {key.lower(): value for key, value in scores.items()}
            total_score = dict()
            for dim in dims:
                if dim not in scores:
                    total_score[dim] = 2.5
                else:
                    total_score[dim] = scores[dim]
                    
            total_scores.append([sum(total_score.values()) / len(total_score)])        
        except Exception as e:
            total_scores.append([2.5])
    return total_scores

def add_generation_prompt(prompt):
    return f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    
def _value_inference_fastchat(
    model_name: str,
    input_str: Union[List[str], str],
    controller_addr="http://0.0.0.0:28777",
    depth: Optional[int] = None,
    reward_model_addr: Optional[int] = 8000,
    target_length: Optional[int] = None,
):
    openai_api_key = "EMPTY"
    openai_api_base = f"http://localhost:{reward_model_addr}/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    prompt_template = open("/mnt/gemininjceph2/geminicephfs/pr-others-prctrans/pingbowen/workspace/MCTS_DPO/MCTS-dpo/templates/judge_text.txt", "r", encoding="utf-8").read()
    prompts = [
        add_generation_prompt(prompt_template.replace('$INST$', question).replace('$RESPONSE$', answer))
        for question, answer in input_str
    ]
    
    completion = client.completions.create(
        model="Llama3.1_70B_awq_int4",
        prompt=prompts,
        temperature=0.7,
        max_tokens=1024,
        # add_generation_prompt=True
    )
    value = get_mcts_score(completion)
    if target_length is not None:
        lens = get_len_score(input_str)
        lens_scores = [[score(l, target_length)] for l in lens]
        value = [[v[0] + ls[0]] for v, ls in zip(value, lens_scores)]
    
    # ret = requests.post(
    #     controller_addr + "/get_worker_address", json={"model": model_name}
    # )
    # worker_addr = ret.json()["address"]
    # if not worker_addr:
    #     raise ValueError("Value Model name {} does not exist.".format(model_name))

    # headers = {"User-Agent": "FastChat Client"}
    # gen_params = {"input_str": input_str}
    # response = requests.post(
    #     worker_addr + "/worker_value_inference",
    #     headers=headers,
    #     json=gen_params,
    #     stream=True,
    # )
    # results = response.json()
    # value = results["value"]

    return value
