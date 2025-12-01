#!/bin/bash

# ==============================================================================
#                      CONFIGURATION SECTION
# ==============================================================================

# Name of the Python script to execute
PYTHON_SCRIPT="inference/play_battle/gen1.py"

AGENT="SyntheticRLV2"   # DO NOT CHANGE THIS...
EVAL_TYPE="pokeagent"    # Choices: "heuristic", "il", "ladder", "pokeagent"
GENS="1" 
FORMATS="ou"
TOTAL_BATTLES=10
TEAM_SET="competitive"
AVATAR="lucas"

BATTLE_BACKEND="metamon"    # Choices: "poke-env", "metamon"
SAVE_TRAJECTORIES_TO="eval_replays"
SAVE_TEAM_RESULTS_TO="team_results.csv"
LOG_TO_WANDB="false"

# 1. Start building the command in an array. This is the safest method.
CMD=(
    "modal" "run" "$PYTHON_SCRIPT"
    "--agent" "$AGENT"
    "--eval_type" "$EVAL_TYPE"
    "--gens" "$GENS"
    "--formats" "$FORMATS"
    "--total_battles" "$TOTAL_BATTLES"
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