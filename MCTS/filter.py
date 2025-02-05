import json
import argparse
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--filter', action="store_true")
arg_parser.add_argument('--merge', action="store_true")
arg_parser.add_argument('--threshold', type=float)
arg_parser.add_argument('--scale', type=str)
arg_parser.add_argument('--filter_data_input_file', type=str)
arg_parser.add_argument('--filter_reward_input_file', type=str)
arg_parser.add_argument('--filter_data_output_file', type=str)
arg_parser.add_argument('--filter_reward_output_file', type=str)
arg_parser.add_argument('--refined_file', type=str)
arg_parser.add_argument('--original_file', type=str)
arg_parser.add_argument('--merged_file', type=str)

args = arg_parser.parse_args()

def read_file(file_path):
    data_list = []
    with open(file_path, 'r') as infile:
        for line in infile:
            data = json.loads(line)
            data_list.append(data)
    return data_list

if args.filter:
    # 定义输入文件
    reward_file = args.filter_reward_input_file
    data_file = args.filter_data_input_file
    data_output_file = args.filter_data_output_file
    reward_output_file = args.filter_reward_output_file

    # 读取数据并统计 chosen_reward 小于 8.0 的数量
    
    # lines = []
    # with open(leaf_reward_file, 'r') as infile:
    #     for line in infile:
    #         data = json.loads(line)
    #         lines.append(data)

    reward_list = []
    with open(reward_file, 'r') as infile:
        for line in infile:
            data = json.loads(line)
            reward_list.append(data)

    low_reward_indices = [index for index, item in enumerate(reward_list) if item.get("chosen_reward", float('inf')) < args.threshold]
    # low_reward_indices = [index for index, item in enumerate(reward_list) for line in lines if item.get("group_id", "") == line.get("group_id", "") and item.get("depth", "") != line.get("depth", "")  and item.get("chosen_reward", float('inf')) < 8.0]

    data_list = []
    with open(data_file, 'r') as infile:
        for line in infile:
            data = json.loads(line)
            data_list.append(data)
            
    low_reward_samples = [data_list[index] for index in low_reward_indices]
    low_rewards = [reward_list[index] for index in low_reward_indices]
   
    with open(reward_output_file, 'w') as outfile:
        for sample in low_rewards:
            json.dump(sample, outfile,ensure_ascii=False)
            outfile.write('\n')
    
    with open(data_output_file, 'w') as outfile:
        for sample in low_reward_samples:
            json.dump(sample, outfile,ensure_ascii=False)
            outfile.write('\n')
elif args.merge:
    refined_data = read_file(args.refined_file)
    final_data = read_file(args.original_file)
    
    for data in refined_data:
        for final_item in final_data:
            if final_item["group_id"] == data["group_id"] and final_item["depth"] == data["depth"]:
                final_item["chosen"] = data["chosen"]
    
    with open(args.merged_file, 'w') as outfile:
        for item in final_data:
            json.dump(item, outfile, ensure_ascii=False)
            outfile.write('\n')