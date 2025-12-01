#!/bin/bash

# Set the name of your Python script
PYTHON_SCRIPT="custom/gen1/train_traj_encoder_KD.py"

# --- Set your required arguments here ---
# These arguments are required by the Python script
RUN_NAME="TrajEncoder_KD_Syn-v2_Gen-1"
SAVE_DIRECTORY="results"
PARSED_REPLAY_DIR="parsed-replays" 
MODEL_CONFIG="./custom/gen1/configs/traj_encoder_KD.gin"
TRAIN_CONFIG="./custom/gen1/configs/train.gin"
FORMATS="gen1ou"

ACTION_SPACE="MinimalActionSpace"
TOKENIZER="allreplays-v3"
OBSERVATION_SPACE="DefaultObservationSpace"
REWARD_FUNCTION="DefaultShapedReward"
CKPT_EPOCH=-1   
CKPT_INTERVAL=1
EPOCHS=3
BATCH_SIZE=64
STEPS_PER_EPOCH=500
GRADIENT_ACCUMULATION=4
# Set a flag for the --log argument
ENABLE_WANDB_LOGGING=true
LOG_INTERVAL=40
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