set -e

HOST_ADDR=0.0.0.0
CONTROLER_PORT=$1
WORKER_BASE_PORT=$2
SESSION_NAME=$3
CUDA_DEVICE_BASE=$4

echo PYTHON_EXECUTABLE=$(which python3)
PYTHON_EXECUTABLE=$(which python3)

MODEL_BASE=/mnt/gemininjceph2/geminicephfs/pr-others-prctrans/pingbowen/models
POLICY_MODEL_NAME=LongWriter-llama3.1-8b
MODEL_PATH=$MODEL_BASE/$POLICY_MODEL_NAME

LOGDIR=logs_fastchat

tmux start-server
tmux new-session -s $SESSION_NAME -n controller -d
tmux send-keys "export LOGDIR=${LOGDIR}" Enter
tmux send-keys "$PYTHON_EXECUTABLE -m fastchat.serve.controller --port ${CONTROLER_PORT} --host $HOST_ADDR" Enter

NUM_LM_WORKER=1
echo "Wait 5 seconds ..."
sleep 5

echo "Starting workers"
for i in $(seq 0 $((NUM_LM_WORKER-1)))
do
  WORKER_PORT=$((WORKER_BASE_PORT+i))
  tmux new-window -n policy_worker_$i
  tmux send-keys "export LOGDIR=${LOGDIR}" Enter
  tmux send-keys "CUDA_VISIBLE_DEVICES=$((i+CUDA_DEVICE_BASE)) $PYTHON_EXECUTABLE -m reason.llm_service.workers.vllm_worker --model-path $MODEL_PATH --controller-address http://$HOST_ADDR:$CONTROLER_PORT --dtype bfloat16 --host $HOST_ADDR --port $WORKER_PORT --worker-address http://$HOST_ADDR:$WORKER_PORT" Enter
done

echo "All done!"