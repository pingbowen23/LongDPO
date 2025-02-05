# import torch
# from transformers import AutoTokenizer,AutoModelForCausalLM
import json
import re
import os
# from datasets import load_dataset
from vllm import LLM, SamplingParams
from openai import OpenAI
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, help="Path to" )
parser.add_argument("--batch_size", type=int, help="Path to" )
parser.add_argument("--input_file", type=str, default="Llama3.1_70B_awq_int4", help="Path to" )
parser.add_argument("--output_file", type=str, default="Llama3.1_70B_awq_int4", help="Path to" )
parser.add_argument("--prompt_file", type=str, default="Llama3.1_70B_awq_int4", help="Path to" )
parser.add_argument("--candidate_file", type=str, default="Llama3.1_70B_awq_int4", help="Path to" )
parser.add_argument("--reward_file", type=str, default="Llama3.1_70B_awq_int4", help="Path to" )
parser.add_argument("--chosen_file", type=str, default="Llama3.1_70B_awq_int4", help="Path to" )
parser.add_argument("--analysis_file", type=str, default="Llama3.1_70B_awq_int4", help="Path to" )
parser.add_argument("--refine_output_file", type=str, default="Llama3.1_70B_awq_int4", help="Path to" )
parser.add_argument("--new_template", action="store_true",  help="Path to" )
parser.add_argument("--analysis", action="store_true",  help="Path to" )
parser.add_argument("--self_refine", action="store_true",  help="Path to" )
parser.add_argument("--generate_candidates", action="store_true",  help="Path to" )
parser.add_argument("--generate_suggestions", action="store_true",  help="Path to" )
parser.add_argument("--eval_candidates", action="store_true",  help="Path to" )
parser.add_argument("--refine_candidates", action="store_true",  help="Path to" )
parser.add_argument('--model_path', type=str)
parser.add_argument('--tp_size', type=int)
parser.add_argument('--n', type=int,default=1)
parser.add_argument('--max_tokens', type=int)
args = parser.parse_args()


def count_words(text):
    chinese_characters = re.findall(r'[\u4e00-\u9fff]', text)
    english_words = re.findall(r'\b[a-zA-Z]+\b', text)
    
    chinese_char_count = len(chinese_characters)
    english_word_count = len(english_words)
    
    total_count = chinese_char_count + english_word_count
    
    return total_count

def count_non_length_stop_reason(file_path):
    count = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            sample = json.loads(line)
            if sample.get("stop_reason") == "stop":
                count += 1
    return count

def save_to_jsonl(filename, data_list):
    with open(filename, 'a') as f:
        for data in data_list:
            data = {"instruction": data}
            f.write(json.dumps(data,ensure_ascii=False) + '\n')

def extract_info(pattern, text):
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return None

def add_generation_prompt(prompt):
    return f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

def add_generator_prompt(prompt):
    return f"[INST]{prompt}[/INST]"

# def get_mcts_score(outputs):
#     total_scores = []
#     if args.new_template:
#         dims = ["principle1","principle2","principle3","principle4","principle5","principle6","principle7"]
#     else:
#         dims = ["Relevance", "Accuracy", "Coherence", "Clarity", "Breadth and Depth", "Reading Experience"]

#     for idx,output in enumerate(outputs):        
#         try:
#             try:
#                 output = output.outputs[0].text
#             except:
#                 output = output.choices[0].text
                
#             output = extract_info(r'\{(.*?)\}', output)
#             scores = json.loads('{' + output + '}')
#             if args.new_template:
#                 scores = {key.lower(): value for key, value in scores.items()}
#             total_score = dict()
#             for dim in dims:
#                 if dim not in scores:
#                     total_score[dim] = 2.5
#                 else:
#                     total_score[dim] = scores[dim]
#             total_score["average_score"] = sum(total_score.values()) / len(total_score)
#             total_scores.append(total_score)        
#         except Exception as e:
#             total_scores.append({"average_score": 2.5})
#     return total_scores

def get_mcts_score(outputs):
    total_scores = []
    if args.new_template:
        dims = ["principle1","principle2","principle3","principle4","principle5","principle6","principle7"]
    else:
        dims = ["Relevance", "Accuracy", "Coherence", "Clarity", "Breadth and Depth", "Reading Experience"]

    for idx,output in enumerate(outputs):        
        try:
            try:
                output = output.outputs[0].text
            except:
                output = output.choices[0].text
                
            output = extract_info(r'\{(.*?)\}', output)
            scores = json.loads('{' + output + '}')
            if args.new_template:
                scores = {key.lower(): value for key, value in scores.items()}
            total_score = dict()
            for dim in dims:
                if isinstance(scores[dim],list) and len(scores[dim]) == 5:
                    total_score[dim] = sum(a * b for a, b in zip(scores[dim], [1,2,3,4,5]))
                else:
                    total_score[dim] = 2.5
            total_score["average_score"] = sum(total_score.values()) / len(total_score)
            total_scores.append(total_score)        
        except Exception as e:
            total_scores.append({"average_score": 2.5})
    return total_scores


def get_reward_model(model_path):
    model = LLM(
        model= args.model_path,
        # dtype="auto",
        trust_remote_code=True,
        tensor_parallel_size=args.tp_size,
        quantization="awq_marlin"
    )
    return model

def get_generator(model_path):
    model = LLM(
        model= args.model_path,
        # dtype="auto",
        trust_remote_code=True,
        tensor_parallel_size=args.tp_size,
        # max_model_len=32768,
        # gpu_memory_utilization=1,
    )
    return model

def read_input_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

# def save_output_file(file_path, data_list):
#     with open(file_path, 'w', encoding='utf-8') as f:
#         for data in data_list:
#             f.write(json.dumps(data, ensure_ascii=False) + '\n')
            
prompt_template = open(args.prompt_file, "r", encoding="utf-8").read()

principles = ["The response meets the user’s purpose and needs.","The response meets the user’s purpose and needs.","The response is non-toxic and safe.","The response meets the user’s formatting requirements and maintains logical consistency.","The response contains diverse and comprehensive information with minimal repetition.","The response provides an excellent reading experience.","The response is insightful and provides the user with additional avenues for thought."]

generation_params = {
        "temperature":1.0,
        "top_p":0.8,
        "top_k":50,
        "max_tokens":args.max_tokens,
        "repetition_penalty":1,
        # stop_token_ids=stop_token_ids,
        "n":args.n,
    }
    
# analysis reward
if args.analysis:
    
    file_path = args.input_file
    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                # 解析每一行的 JSON 对象
                data.append(json.loads(line.strip()))
    except json.JSONDecodeError as e:
        print(f"JSON 解码错误：{e}")
    
    chosen_prompts = [
        add_generation_prompt(prompt_template.replace('$INST$', sample['instruction']).replace('$RESPONSE$', sample['chosen'])) for sample in data
    ][:1000]
    
    model = get_reward_model(args.model_path)
    
    outputs = model.generate(chosen_prompts,sampling_params=SamplingParams(**generation_params))
    chosen_values = get_mcts_score(outputs)
    with open(args.output_file, "a", encoding="utf-8") as file:
        for idx in range(len(chosen_values)):
            try:
                chosen_value = chosen_values[idx]
            except:
                chosen_value = {}
            chosen_value["group_id"] = data[idx]["group_id"]
            chosen_value["depth"] = data[idx]["depth"]
            file.write(json.dumps(chosen_value, ensure_ascii=False) + "\n")

    # with open(args.output_file, "a", encoding="utf-8") as file:
    #     for chosen_value, sample in zip(chosen_values,data):
    #         chosen_value["chosen_reward"] = round(chosen_value["average_score"], 2)
    #         # chosen_value["rejected_reward"] = round(rejected_value["average_score"], 2)
    #         chosen_value["group_id"] = sample["group_id"]
    #         chosen_value["depth"] = sample["depth"]
    #         del chosen_value["average_score"]
    #         file.write(json.dumps(chosen_value, ensure_ascii=False) + "\n")

elif args.generate_candidates:
    model = get_generator(args.model_path)
    
    tokenizer = model.get_tokenizer()

    stop_token_ids = [tokenizer.eos_token_id]

    data = read_input_file(args.input_file)
        
    inputs = [sample['instruction'] for sample in data]
    
    outputs = model.generate(inputs,
            sampling_params=SamplingParams(**generation_params))
    with open(args.output_file, 'a', encoding='utf-8') as f:
        for sample, output in zip(data, outputs):
            item = {"instruction": sample['instruction'], "candidates": [output.outputs[i].text for i in range(args.n)],"group_id": sample['group_id'], "depth": sample['depth']}
            f.write(json.dumps(item, ensure_ascii=False) + '\n')         
elif args.eval_candidates:
    model = get_reward_model(args.model_path)
    
    lines = read_input_file(args.input_file)
    
    batch = [add_generation_prompt(prompt_template.replace('$INST$', line['instruction']).replace('$RESPONSE$', candidate)) for line in lines for candidate in line['candidates']]
    
    candidates = [line['candidates'][i] for line in lines for i in range(args.n)]
    
    generation_params["n"] = 1
    
    outputs = model.generate(batch,sampling_params=SamplingParams(**generation_params))
    
    chosen_values = get_mcts_score(outputs)
    
    with open(args.output_file, "a", encoding="utf-8") as file:
        for chosen_value,candidate in zip(chosen_values,candidates):
            chosen_value["candidate"] = candidate
            file.write(json.dumps(chosen_value, ensure_ascii=False) + "\n")
elif args.generate_suggestions:
    def get_principle_index(reward,candidate):
        dims = ["principle1","principle2","principle3","principle4","principle5","principle6","principle7"]
        indices = []
        for idx,dim in enumerate(dims):
            candidate_score = candidate.get(dim, None)
            chosen_score = reward.get(dim, None)
            
            if candidate_score is None or chosen_score is None:
                continue
            
            if candidate_score > chosen_score:
                indices.append(idx)
        return indices
    
    def process_batch(candidates, rewards, chosens, candidate_per_sample):
        # instruction , pricinple, candidate1, candidate2
        batch , group_ids, depths = [], [], []
        for candidate, reward, chosen in zip(candidates, rewards, chosens):
            indices = get_principle_index(reward,candidate)
            for index in indices:
                if not args.self_refine:
                    data = add_generation_prompt(prompt_template.replace('$INST$', chosen['instruction']).replace('$PRINCIPLE$', principles[index]).replace('$CANDIDATE1$', candidate["candidate"])).replace('$CANDIDATE2$', chosen["chosen"])
                else:
                    data = add_generator_prompt(prompt_template.replace('$INST$', chosen['instruction']).replace('$PRINCIPLE$', principles[index]).replace('$CANDIDATE1$', candidate["candidate"])).replace('$CANDIDATE2$', chosen["chosen"])
                batch.append(data)
                group_ids.append(chosen["group_id"])
                depths.append(chosen["depth"])
             
        return batch, group_ids, depths
        
    candidates = read_input_file(args.candidate_file)
    
    rewards = read_input_file(args.reward_file)
    
    chosens = read_input_file(args.chosen_file)
    
    candidate_per_sample = len(candidates) // len(chosens)
    
    repeated_candidates = [candidates[i] for i in range(len(candidates)) for _ in range(candidate_per_sample)]
    repeated_rewards = [rewards[i] for i in range(len(rewards)) for _ in range(candidate_per_sample)]
    
    batch , group_ids, depths = process_batch(repeated_candidates, repeated_rewards, chosens, candidate_per_sample)
    # model = get_reward_model(args.model_path)
    model = get_generator(args.model_path)
    
    outputs = model.generate(batch,sampling_params=SamplingParams(**generation_params))
    outs = []
    for output in outputs:
        output = output.outputs[0].text
        
        try:
            output = extract_info(r'\{(.*?)\}', output)
            output = json.loads('{' + output + '}')
        except:
            continue
        
        outs.append(output)
    with open(args.refine_output_file, "a", encoding="utf-8") as file:
        for group_id, depth, out in zip(group_ids, depths, outs):
            out["group_id"] = group_id
            out["depth"] = depth
            file.write(json.dumps(out, ensure_ascii=False) + "\n")
else:
    
    def process_special_tokens(instruction,suggestions):
        prefix = "Here are some suggestions for you to improve your writing quality:"
        start_tag = "[INST]"
        end_tag = "[/INST]"

        # 找到开始标签和结束标签的索引
        start_index = instruction.find(start_tag) + len(start_tag)  # 找到 [INST] 结束位置
        end_index = instruction.find(end_tag)  # 找到 [/INST] 开始位置
        
        if prefix in instruction:
            instruction = instruction[:start_index] + instruction[start_index:end_index].strip() + " " + suggestions.strip() + instruction[end_index:]
        else:
            instruction = instruction[:start_index] + instruction[start_index:end_index].strip() + " " + prefix + " " + suggestions.strip() + instruction[end_index:]
        return instruction
    
    def process_batch(chosens, suggestions):
        from collections import defaultdict

        grouped_suggestions = defaultdict(list)
        for suggestion in suggestions:
            grouped_suggestions[suggestion["group_id"]].append(suggestion)
            
        for chosen in chosens:
            instruction = chosen["instruction"]
            max_suggestions = random.randint(1, 3)
            matching_suggestions = grouped_suggestions.get(chosen["group_id"], [])

            for suggestion in matching_suggestions: #  遍历matching_suggestions列表
                if "Confidence Score" in suggestion:
                    suggestion["Confidence Score"] = int(suggestion["Confidence Score"])

            matching_suggestions = sorted(matching_suggestions, key=lambda x: x.get("Confidence Score",0), reverse=True)[:max_suggestions]
            
            for suggestion in matching_suggestions:
                instruction = process_special_tokens(instruction, suggestion.get("Writing Suggestion",""))
                chosen["instruction"] = instruction
                chosen["has_suggestion"] = True

        with_suggestion = []
        without_suggestion = []
        for item in chosens:
            if "has_suggestion" in item:
                with_suggestion.append(item)
            else:
                without_suggestion.append(item)
        
        return with_suggestion, without_suggestion
    
    chosens = read_input_file(args.chosen_file)
    
    suggestions = read_input_file(args.analysis_file)
    
    chosen_with_suggestion , chosen_without_suggestion = process_batch(chosens, suggestions)
    inputs = [item["instruction"] for item in chosen_with_suggestion]
    model = get_generator(args.model_path)
    outputs = model.generate(inputs,sampling_params=SamplingParams(**generation_params))
    for tmp_chosen, output in zip(chosen_with_suggestion,outputs):
        output = output.outputs[0].text
        tmp_chosen["chosen"] = output
    
    chosen_with_suggestion.extend(chosen_without_suggestion)
    
    with open(args.output_file, 'a', encoding='utf-8') as f:
        for data in chosen_with_suggestion:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')