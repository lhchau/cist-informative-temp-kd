# Shared settings, sourced by the other scripts. Override any of these from the environment.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH=$PWD TOKENIZERS_PARALLELISM=false TF_CPP_MIN_LOG_LEVEL=3 WANDB_DISABLED=True
GPUS=${GPUS:-4}
PORT=${PORT:-2012}
LAUNCH=${LAUNCH:-"torchrun --nproc_per_node $GPUS --nnodes 1 --master_addr localhost --master_port $PORT"}

FAMILY=$1
case "$FAMILY" in
  gpt2)
    STUDENT=${STUDENT:-gpt2}                                   # GPT-2 Base (0.1B)
    TEACHER=${TEACHER:-checkpoints/teacher-gpt2-1.5B}         # SFT GPT-2 XL (1.5B)
    LR=${LR:-5e-4}
    MODEL_OPTS="--model-type gpt2"
    TRAIN_OPTS=""
    ;;
  opt)
    STUDENT=${STUDENT:-facebook/opt-1.3b}
    TEACHER=${TEACHER:-checkpoints/teacher-opt-6.7B}          # SFT OPT-6.7B
    LR=${LR:-5e-5}
    MODEL_OPTS="--model-type opt"
    TRAIN_OPTS="--gradient-checkpointing"
    ;;
  *)
    echo "usage: $0 <gpt2|opt> ..." >&2; exit 1 ;;
esac
shift

# Sampling used both for dev-set checkpoint selection and for the final evaluation (decoding temperature 1).
GEN_OPTS="--do-sample --top-k 0 --top-p 1.0 --temperature 1.0"
DS_OPTS="--deepspeed --deepspeed_config configs/ds_config.json"
