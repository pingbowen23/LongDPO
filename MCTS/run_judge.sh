reward_model_addrs=(9010 9011 9012)

for i in {0..0}
do
    if [ $i -eq 0 ]; then
        GPU_1=3
        GPU_2=4
        tp=2
        max_model_len=131072
        export CUDA_VISIBLE_DEVICES=$GPU_1,$GPU_2
    elif [ $i -eq 1 ]; then
        GPU_1=5
        GPU_2=6  # 使用 GPU 4 和 5
        tp=2
        export CUDA_VISIBLE_DEVICES=$GPU_1,$GPU_2
        max_model_len=131072
    elif [ $i -eq 2 ]; then
        GPU_1=7
        tp=1
        export CUDA_VISIBLE_DEVICES=$GPU_1
        max_model_len=65536
    fi
    
    vllm serve models/Llama3.1-70B-awq  --uvicorn-log-level info -tp $tp -q awq_marlin --served-model-name Llama3.1_70B_awq_int4 --port ${reward_model_addrs[$i]}  --max-model-len $max_model_len &
done
wait