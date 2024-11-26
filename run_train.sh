#!/bin/bash

# Settings
EXPERIMENT="resnet_imagenet"
MODE="train_and_eval"
MODEL_DIR="results"
CONFIG_FILE="official/vision/configs/experiments/image_classification/imagenet_resnet50v1.5.yaml"
NUM_GPUS=8
MIXED_PRECISION_DTYPE="float16"
PARAMS_OVERRIDE="runtime.num_gpus=$NUM_GPUS"
LOG_FILE="${MODEL_DIR}/${EXPERIMENT}_${MODE}.log"
#DATA_FORMAT=${DATA_FORMAT:-"channels_last"} # channels_last
DATA_FORMAT=${DATA_FORMAT:-"channels_first"} # channels_last
RANDOM_SEED=301

# Environment
unset XLA_FLAGS TF_CPP_MAX_VLOG_LEVEL TF_FORCE_GPU_ALLOW_GROWTH
export TF_CPP_MAX_VLOG_LEVEL=-1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CUDNN_WORKSPACE_LIMIT_IN_MB=16384


#if [ "$DATA_FORMAT" = "channels_last" ]; then
#   export XLA_FLAGS=--xla_gpu_force_conv_nhwc
#fi

# Parser commands
while [ $# -gt 0 ]; do
  case "$1" in
    --experiment=*)
      EXPERIMENT="${1#*=}"
      ;;
    --mode=*)
      MODE="${1#*=}"
      ;;
    --model_dir=*)
      MODEL_DIR="${1#*=}"
      ;;
    --config_file=*)
      CONFIG_FILE="${1#*=}"
      ;;
    --num_gpus=*)
      NUM_GPUS="${1#*=}"
      ;;
    --mixed_precision_dtype=*)
      MIXED_PRECISION_DTYPE="${1#*=}"
      ;;
    --enable_xla=*)
      ENABLE_XLA="${1#*=}"
      ;;
    --log_file=*)
      LOG_FILE="${1#*=}"
      ;;
    *)
      echo "Invalid Argument: $1"
      exit 1
  esac
  shift
done

# Ensure model directory exists
mkdir -p $MODEL_DIR

# Run commands
python3 -m official.vision.train \
  --experiment=$EXPERIMENT \
  --mode=$MODE \
  --model_dir=$MODEL_DIR \
  --config_file=$CONFIG_FILE \
  --params_override="$PARAMS_OVERRIDE" \
  --data_format=$DATA_FORMAT \
  --seed=$RANDOM_SEED \
  2>&1 | tee $LOG_FILE
