from metamon.env import get_metamon_teams, PokeAgentLadder, QueueOnLocalLadder
from metamon.interface import DefaultObservationSpace, DefaultShapedReward, DefaultActionSpace
from custom.utils import get_env_var 
from metamon.env import BattleAgainstBaseline
from metamon.baselines import get_baseline

USERNAME, PASSWD=get_env_var("player_username"), get_env_var("player_password")

team_set = get_metamon_teams("gen9ou", "modern_replays")

obs_space = DefaultObservationSpace()
reward_fn = DefaultShapedReward()
action_space = DefaultActionSpace()



# env = BattleAgainstBaseline(
#     battle_format="gen1ou",
#     observation_space=obs_space,
#     action_space=action_space,
#     reward_function=reward_fn,
#     team_set=team_set,
#     opponent_type=get_baseline("Gen1BossAI"),
# )

'''
equest battles on our local Showdown server and battle 
anyone else who is online (humans, pretrained agents, or other Pokémon AI projects). 
If it plays Showdown, we can battle against it!
'''

# env = QueueOnLocalLadder(
#     battle_format="gen1ou",
#     player_username=USERNAME,
#     num_battles=10,
#     observation_space=obs_space,
#     action_space=action_space,
#     reward_function=reward_fn,
#     player_team_set=team_set,
# )

'''
PokéAgent Challenge
'''

env = PokeAgentLadder(
    battle_format="gen9ou",
    player_username=USERNAME,
    player_password=PASSWD, 
    num_battles=1,
    observation_space=obs_space,
    action_space=action_space,
    reward_function=reward_fn,
    player_team_set=team_set,
)

obs, info = env.reset()
import time
# standard `gymnasium` environment
start=time.time()
terminated=False
while (not terminated) and time.time()-start<600:
    next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print(f"next-obs ==> {next_obs}")
    print("_"*50)
    print(f"{reward}  |  {terminated}  |  {truncated}   |  {info} \n\n")