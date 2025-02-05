import json
import os
import copy
import argparse
import random
import re
from itertools import groupby
# from vllm import LLM, SamplingParams

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str)
parser.add_argument('--output_file', type=str)
parser.add_argument('--fix_assistant', action='store_true')
parser.add_argument('--add_qwen_special_token', action='store_true')
parser.add_argument('--global_pair', action='store_true')
args = parser.parse_args()

def count_words(text):
    chinese_characters = re.findall(r'[\u4e00-\u9fff]', text)
    english_words = re.findall(r'\b[a-zA-Z]+\b', text)
    
    chinese_char_count = len(chinese_characters)
    english_word_count = len(english_words)
    
    total_count = chinese_char_count + english_word_count
    
    return total_count

def read_file(file_path):
    data_list = []
    with open(os.path.join(file_path), 'r',encoding='utf-8') as lines:
        for line in lines:
            data_list.append(json.loads(line))
    return data_list

def add_template(inst, word):
    template1 = f"At least {word} words."
    template2 = f"No less than {word} words."
    template3 = f"Approximately {word} words."

    chosen_template = random.choice([template1, template2, template3])

    # inst = inst.split("[/INST]",1)[0].rstrip() + " " + chosen_template + "[/INST]" + inst.split("[/INST]",1)[1]
    inst = inst.strip() + " " + chosen_template
    return inst


if args.fix_assistant:

    data_list = []

    with open(args.input_file, 'r') as lines:
        for line in lines:
            data_list.append(json.loads(line))

    data_list_copy = copy.deepcopy(data_list)
    # 按 group_id 排序，确保 groupby 可以正确分组
    data_list_sorted = sorted(data_list, key=lambda x: x['group_id'])

    # 使用 groupby 进行分组
    grouped_data = {group_id: sorted(items, key=lambda x: x['depth'])
                    for group_id, items in groupby(data_list_sorted, key=lambda x: x['group_id'])}

    # 输出分组并排序后的数据
    cnt = 0
    filter_group_id = []
    for group_id, items in grouped_data.items():
        if items[-1]['answer_role'] != "assistant":
            for i in range(len(data_list_copy)):
                if data_list_copy[i]['group_id'] == group_id and data_list_copy[i]['depth'] == items[-1]['depth']:
                    data_list_copy[i]['answer_role'] = "assistant"
    
    with open(args.output_file, 'w') as f:
        for data in data_list_copy:
            f.write(json.dumps(data,ensure_ascii=False) + "\n")

elif args.add_qwen_special_token:

    data_list = []

    with open(args.input_file, 'r') as lines:
        for line in lines:
            data_list.append(json.loads(line))

    for data in data_list:
        data["instruction"] = data["instruction"].replace("[INST]", "<|im_start|>user\n").replace("[/INST]", "<|im_end|>\n")

    with open(args.output_file, 'w') as f:
        for data in data_list:
            f.write(json.dumps(data,ensure_ascii=False) + "\n")
elif args.global_pair:
    generation_params = {
        "temperature":0.7,
        "top_p":0.8,
        "top_k":50,
        "max_tokens":4096,
        "repetition_penalty":1,
        "n":8,
    }

    model = LLM(
        model= "",
        dtype="auto",
        trust_remote_code=True,
        tensor_parallel_size=8,
    )
 
    data_list = read_file(args.input_file)
    inputs = [f"[INST]{data['instruction']}[/INST]" for data in data_list]
    outputs = model.generate(inputs,sampling_params=SamplingParams(**generation_params))

    with open(args.output_file, 'a',encoding="utf-8") as f:
        for idx,output in enumerate(outputs):
            
            max_word_count = 0
            for i in range(generation_params["n"]):
                tmp_word_count = count_words(output.outputs[i].text)
                
                if tmp_word_count > max_word_count:
                    max_word_count = tmp_word_count
                    rejected = output.outputs[i].text

            data_list[idx]["rejected"] = rejected
            f.write(json.dumps(data_list[idx],ensure_ascii=False) + "\n")

            if max_word_count < 2000 and max_word_count > 1500:
                print(f"The {idx} chosen length is {max_word_count}, which is less than 2000 but greater than 1500.")

else:
    inst_2_4k = read_file("")
    filtered_inst = []

    filtered_inst = list(set(filtered_inst))

    unique_group_ids = set()

    # 遍历 data_list，提取并存储 group_id
    for item in inst_2_4k:
        group_id = item.get("group_id")
        if group_id is not None:
            unique_group_ids.add(group_id)

    group_id_lengths = {}

    # 遍历 data_list，统计每个 group_id 对应的 chosen 字段的长度
    for item in inst_2_4k:
        group_id = item.get("group_id")
        chosen_length = item.get("chosen", "")  # 获取 "chosen" 的长度，若没有 "chosen" 则为 0

    unique_group_ids_list = list(unique_group_ids)

    data_list_sorted = sorted(inst_2_4k, key=lambda x: x['group_id'])

    # 使用 groupby 进行分组
    grouped_data = {group_id: sorted(items, key=lambda x: x['depth'])
                    for group_id, items in groupby(data_list_sorted, key=lambda x: x['group_id'])}
    
    sample_size = 1000
    sampled_ids = random.sample(list(grouped_data.keys()), sample_size)

    with open("", 'w') as f:
        for group_id in sampled_ids:
            for item in grouped_data[group_id]:
                f.write(json.dumps(item,ensure_ascii=False) + "\n")
