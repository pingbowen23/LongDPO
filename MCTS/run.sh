export PYTHONPATH=$(pwd)

CONTROLER_PORTS=(28677 38778 48889)
WORKER_BASE_PORTS=(30010 40010 50010)
SESSION_NAMES=(2k 4k 16k)
reward_model_addrs=(9010 9011 9012)
num_workers=(4 1 4)

input_files=()
output_files=()

tree_depth=(4 4 4) 
max_tokens=(512 2048 3000 8192)
group_init_index=(66666 100000 15000)

# for i in {0..0}
# do
#     sh reason/llm_service/generator_service.sh ${CONTROLER_PORTS[$i]} ${WORKER_BASE_PORTS[$i]} ${SESSION_NAMES[$i]} $i

#     # sleep 60
# done


for i in {0..0}
do
    # check --len_scale参数
    sh scripts/eval/vanila_mcts.sh ${CONTROLER_PORTS[$i]} ${reward_model_addrs[$i]} ${input_files[$i]} ${output_files[$i]} ${num_workers[$i]} ${tree_depth[$i]} ${max_tokens[$i]} ${group_init_index[$i]} #&
done
wait


# tmux kill-session -t 2k 4k
# tmux attach -t 2k