file=$3
# output_dir=./data/test_output.jsonl
output_dir=$4
group_init_index=$8

python reason/evaluation/evaluate.py \
    --LM LongWriter-llama3.1-8b \
    --local \
    --RM Llama3.1_70B_awq_int4 \
    --task_name MATH \
    --temperature 0.7 \
    --max_new_tokens $7 \
    --num_sequence 1 \
    --tree_max_width 4 \
    --tree_max_depth $6 \
    --method vanila_mcts \
    --num_worker $5 \
    --controller_addr http://127.0.0.1:$1 \
    --input_file $file \
    --save_step_dir $output_dir \
    --reward_model_addr $2\
    --group_init_index $group_init_index \
    --len_scale 10

# len_scale 长度分数在[0,100], 而quality在[0,10]，控制在同一个区间

# Qwen2-0.5B
# math-shepherd-mistral-7b-prm