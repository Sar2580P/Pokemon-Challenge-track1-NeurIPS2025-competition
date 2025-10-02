#!/bin/bash

# ==============================================================================
#                      CONFIGURATION SECTION
# ==============================================================================

# Name of the Python script to execute
PYTHON_SCRIPT="custom/evaluate.py"

AGENT="HRM_gen1ou"
EVAL_TYPE="heuristic"    # Choices: "heuristic", "il", "ladder", "pokeagent"
GENS="1" 
FORMATS="ou"
TOTAL_BATTLES=50
TEAM_SET="competitive"
AVATAR="lucas"

CHECKPOINTS=""

BATTLE_BACKEND="metamon"    # Choices: "poke-env", "metamon"
SAVE_TRAJECTORIES_TO="./eval_replays"
SAVE_TEAM_RESULTS_TO="./team_results.csv"
LOG_TO_WANDB=true


# ==============================================================================
#                      SCRIPT EXECUTION LOGIC
# ==============================================================================

# Start building the command with the python script and required args
CMD="python $PYTHON_SCRIPT \
    --agent $AGENT \
    --eval_type $EVAL_TYPE"

# --- Append optional arguments only if they have been set ---
if [ -n "$GENS" ]; then
    CMD="$CMD --gens $GENS"
fi

if [ -n "$FORMATS" ]; then
    CMD="$CMD --formats $FORMATS"
fi

if [ -n "$TOTAL_BATTLES" ]; then
    CMD="$CMD --total_battles $TOTAL_BATTLES"
fi

if [ -n "$CHECKPOINTS" ]; then
    CMD="$CMD --checkpoints $CHECKPOINTS"
fi

if [ -n "$USERNAME" ]; then
    CMD="$CMD --username $USERNAME"
fi

if [ -n "$PASSWORD" ]; then
    CMD="$CMD --password $PASSWORD"
fi

if [ -n "$AVATAR" ]; then
    CMD="$CMD --avatar $AVATAR"
fi

if [ -n "$TEAM_SET" ]; then
    CMD="$CMD --team_set $TEAM_SET"
fi

if [ -n "$BATTLE_BACKEND" ]; then
    CMD="$CMD --battle_backend $BATTLE_BACKEND"
fi

if [ -n "$SAVE_TRAJECTORIES_TO" ]; then
    CMD="$CMD --save_trajectories_to $SAVE_TRAJECTORIES_TO"
fi

if [ -n "$SAVE_TEAM_RESULTS_TO" ]; then
    CMD="$CMD --save_team_results_to $SAVE_TEAM_RESULTS_TO"
fi

# Add the --log_to_wandb flag only if the variable is set to "true"
if [ "$LOG_TO_WANDB" = "true" ]; then
    CMD="$CMD --log_to_wandb"
fi

# Print the final command to the console for verification
echo "🤖 Executing the following command:"
echo "---------------------------------"
echo $CMD
echo "---------------------------------"

# Execute the command
eval $CMD

echo "✅ Script execution finished."