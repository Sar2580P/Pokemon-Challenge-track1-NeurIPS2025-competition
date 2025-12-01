#!/bin/bash

# ==============================================================================
#                      CONFIGURATION SECTION
# ==============================================================================

# Name of the Python script to execute
PYTHON_SCRIPT="custom/gen1/evaluate.py"

AGENT="SyntheticRLV2"   # "HRM_gen1ou_OpponentModeling"  "HRM_gen1ou"   "SmallRL"  "LargeRL"  "SyntheticRLV2"
EVAL_TYPE="pokeagent"    # Choices: "heuristic", "il", "ladder", "pokeagent"
GENS="1" 
FORMATS="ou"
TOTAL_BATTLES=10
TEAM_SET="competitive"
AVATAR="lucas"

CHECKPOINTS=44  #"results/TrajEncoder_KD_Syn-v2_Gen-1/ckpts/policy_weights/policy_epoch_2.pt"

BATTLE_BACKEND="metamon"    # Choices: "poke-env", "metamon"
SAVE_TRAJECTORIES_TO="eval_replays"
SAVE_TEAM_RESULTS_TO="team_results.csv"
LOG_TO_WANDB="false"



# ==============================================================================
#                      SCRIPT EXECUTION LOGIC
# ==============================================================================

# # Start building the command with the python script and required args
# CMD="python $PYTHON_SCRIPT \
#     --agent $AGENT \
#     --eval_type $EVAL_TYPE"

# # --- Append optional arguments only if they have been set ---
# if [ -n "$GENS" ]; then
#     CMD="$CMD --gens $GENS"
# fi

# if [ -n "$FORMATS" ]; then
#     CMD="$CMD --formats $FORMATS"
# fi

# if [ -n "$TOTAL_BATTLES" ]; then
#     CMD="$CMD --total_battles $TOTAL_BATTLES"
# fi

# if [ -n "$CHECKPOINTS" ]; then
#     CMD="$CMD --checkpoints $CHECKPOINTS"
# fi

# if [ -n "$USERNAME" ]; then
#     CMD="$CMD --username $USERNAME"
# fi

# if [ -n "$PASSWORD" ]; then
#     CMD="$CMD --password $PASSWORD"
# fi

# if [ -n "$AVATAR" ]; then
#     CMD="$CMD --avatar $AVATAR"
# fi

# if [ -n "$TEAM_SET" ]; then
#     CMD="$CMD --team_set $TEAM_SET"
# fi

# if [ -n "$BATTLE_BACKEND" ]; then
#     CMD="$CMD --battle_backend $BATTLE_BACKEND"
# fi

# if [ -n "$SAVE_TRAJECTORIES_TO" ]; then
#     CMD="$CMD --save_trajectories_to $SAVE_TRAJECTORIES_TO"
# fi

# if [ -n "$SAVE_TEAM_RESULTS_TO" ]; then
#     CMD="$CMD --save_team_results_to $SAVE_TEAM_RESULTS_TO"
# fi

# # Add the --log_to_wandb flag only if the variable is set to "true"
# if [ "$LOG_TO_WANDB" = "true" ]; then
#     CMD="$CMD --log_to_wandb"
# fi

# # Print the final command to the console for verification
# echo "🤖 Executing the following command:"
# echo "---------------------------------"
# echo $CMD
# echo "---------------------------------"

# # Execute the command
# eval $CMD

# echo "✅ Script execution finished."


# 1. Start building the command in an array. This is the safest method.
CMD=(
    "modal" "run" "$PYTHON_SCRIPT"
    "--agent" "$AGENT"
    "--eval_type" "$EVAL_TYPE"
    "--gens" "$GENS"
    "--formats" "$FORMATS"
    "--total_battles" "$TOTAL_BATTLES"
    "--checkpoints" "$CHECKPOINTS"
    "--avatar" "$AVATAR"
    "--team_set" "$TEAM_SET"
    "--battle_backend" "$BATTLE_BACKEND"
    "--save_trajectories_to" "$SAVE_TRAJECTORIES_TO"
    "--save_team_results_to" "$SAVE_TEAM_RESULTS_TO"
)

# 2. Conditionally add the boolean flag ONLY if the variable is true.
#    This adds the flag without any value, which is what argparse expects.
if [ "$LOG_TO_WANDB" = "true" ]; then
    CMD+=("--log_to_wandb")
fi

# 3. Print and execute the command safely from the array.
echo "---------------------------------"
echo "🤖 Executing Command:"
# The 'printf' command below is a safe way to see exactly what will be run
printf "%q " "${CMD[@]}"
printf "\n"
echo "---------------------------------"

"${CMD[@]}"

echo "✅ Script finished."