from vllm import LLM, SamplingParams
import json
import argparse
import re
import random
from transformers import AutoTokenizer
argparser = argparse.ArgumentParser()
argparser.add_argument('--model_path', type=str)
argparser.add_argument('--tp_size', type=int)
argparser.add_argument('--batch_size', type=int, default=1)
argparser.add_argument('--seed', type=int, default=1)
argparser.add_argument('--input_file', type=str, default=1)
argparser.add_argument('--output_file', type=str, default=1)
argparser.add_argument('--temp', type=float, default=1)
argparser.add_argument('--top_p', type=float, default=1)
argparser.add_argument('--top_k', type=int, default=1)
argparser.add_argument('--longwriter', action='store_true')
argparser.add_argument('--fast_inference', action='store_true')
args = argparser.parse_args()
random.seed(42)

def count_words(text):
    chinese_characters = re.findall(r'[\u4e00-\u9fff]', text)
    english_words = re.findall(r'\b[a-zA-Z]+\b', text)
    
    chinese_char_count = len(chinese_characters)
    english_word_count = len(english_words)
    
    total_count = chinese_char_count + english_word_count
    
    return total_count

def save_to_jsonl(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for item in data:
            json.dump(item, outfile, ensure_ascii=False)
            outfile.write('\n')  # 每个 JSON 对象分隔为单独的一行

model = LLM(
    model= args.model_path,
    # dtype="auto",
    trust_remote_code=True,
    tensor_parallel_size=args.tp_size,
    max_model_len=32768,
    gpu_memory_utilization=0.9,
)
tokenizer = model.get_tokenizer()

stop_token_ids = [tokenizer.eos_token_id]
generation_params = SamplingParams(
    temperature=args.temp,
    max_tokens=32768,
    repetition_penalty=1.1,
    stop_token_ids=stop_token_ids,
)

with open(args.input_file, 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

if args.longwriter:
    inputs = [tokenizer(f"[INST]{d['prompt']}[/INST]").input_ids for d in data]

    if not args.fast_inference:
        for i in range(0, len(data), args.batch_size):

            outputs = model.generate(
                sampling_params=generation_params,
                prompt_token_ids=inputs[i:i+args.batch_size],
            )
            
            for idx,output in enumerate(outputs):
                data[i + idx]['response'] = output.outputs[0].text
                data[i + idx]['response_length'] = count_words(output.outputs[0].text)
    else:
        outputs = model.generate(
                sampling_params=generation_params,
                prompt_token_ids=inputs,
            )
        
        for idx,output in enumerate(outputs):
            data[idx]['response'] = output.outputs[0].text
            data[idx]['response_length'] = count_words(output.outputs[0].text)
            
        
else:
    inputs = [tokenizer.apply_chat_template([{"role": "user", "content": d['prompt']}],tokenize=False,add_generation_prompt=True) for d in data]
    
    outputs = model.generate(
            inputs,
            sampling_params=generation_params
        )
    
    for idx,output in enumerate(outputs):
        data[idx]['response'] = output.outputs[0].text
        data[idx]['response_length'] = count_words(output.outputs[0].text)
            

save_to_jsonl(data, args.output_file)