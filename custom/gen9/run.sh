#!/bin/bash

# Set the name of your Python script
PYTHON_SCRIPT="custom/gen9/train.py"

# --- Set your required arguments here ---
# These arguments are required by the Python script
RUN_NAME="HRM_Pokemon_Gen9"
SAVE_DIRECTORY="results"
PARSED_REPLAY_DIR="parsed-replays" 
MODEL_CONFIG="./custom/gen9/configs/hrm_agent.gin"
TRAIN_CONFIG="./custom/gen9/configs/train.gin"
FORMATS="gen9ou"

ACTION_SPACE="DefaultActionSpace"
TOKENIZER="DefaultObservationSpace-v1"
OBSERVATION_SPACE="TeamPreviewObservationSpace"
REWARD_FUNCTION="DefaultShapedReward"
CKPT_EPOCH=-1   
CKPT_INTERVAL=1
EPOCHS=20
BATCH_SIZE=64
STEPS_PER_EPOCH=2000
GRADIENT_ACCUMULATION=2
# Set a flag for the --log argument
ENABLE_WANDB_LOGGING=true
LOG_INTERVAL=100
VERBOSE=true
WORKERS=10
# --- Run the Python script with the arguments ---
# The backslash character "\" is used to continue the command on a new line
# for readability.
echo "Starting ${RUN_NAME} training with MODAL..."


modal run --detach  "$PYTHON_SCRIPT" \
  --run_name "$RUN_NAME" \
  --save_dir "$SAVE_DIRECTORY" \
  --model_gin_config "$MODEL_CONFIG" \
  --train_gin_config "$TRAIN_CONFIG" \
  --obs_space "$OBSERVATION_SPACE" \
  --reward_function "$REWARD_FUNCTION" \
  --epochs "$EPOCHS" \
  --batch_size_per_gpu "$BATCH_SIZE" \
  --grad_accum "$GRADIENT_ACCUMULATION" \
  --ckpt_interval "$CKPT_INTERVAL" \
  --dloader_workers "$WORKERS" \
  --async_env_mp_context "forkserver" \
  --eval_gens 1 \
  --formats "$FORMATS" \
  --verbose "$VERBOSE" \
  --log_interval "$LOG_INTERVAL" \
  --steps_per_epoch "$STEPS_PER_EPOCH" \
  --parsed_replay_dir "$PARSED_REPLAY_DIR" \
  --tokenizer "$TOKENIZER" \
  --action_space "$ACTION_SPACE" \
  $( [ "$ENABLE_WANDB_LOGGING" = true ] && echo "--log" )

echo "Script finished."

# To run this script, first make it executable in your terminal:
# chmod +x run_script.sh

#  