# model gins -->
SYN_AGENT="inference/configs/models/synthetic_agent.gin"
SYN_MULTI_TASK_AGENT = "inference/configs/models/synthetic_multitaskagent.gin"

# train gins-->
BINARY_MAXQ_RL = "inference/configs/training/binary_maxq_rl.gin"
BINARY_RL= "inference/configs/training/binary_rl.gin"
EXP_RL="inference/configs/training/exp_rl.gin"

GEN1_Models=[
    # ("SyntheticRLV0" , {'checkpoint': 46 , 'model_gin_config':SYN_AGENT , 'train_gin_config':BINARY_RL}), 
    # ("SyntheticRLV1" , {'checkpoint': 40 , 'model_gin_config':SYN_AGENT , 'train_gin_config':BINARY_RL}),
    # ("SyntheticRLV1_SelfPlay" , {'checkpoint': 40 , 'model_gin_config':SYN_AGENT , 'train_gin_config':BINARY_RL}),
    # ("SyntheticRLV1_PlusPlus" , {'checkpoint': 40 , 'model_gin_config':SYN_AGENT , 'train_gin_config':BINARY_MAXQ_RL}),
    ("SyntheticRLV2" , {'checkpoint': 40 , 'model_gin_config':SYN_MULTI_TASK_AGENT , 'train_gin_config':BINARY_RL}),
    #     ("SyntheticRLV2" , {'checkpoint': 46 , 'model_gin_config':SYN_MULTI_TASK_AGENT , 'train_gin_config':BINARY_RL}),

    # ("SyntheticRLV2" , {'checkpoint': 44 , 'model_gin_config':SYN_MULTI_TASK_AGENT , 'train_gin_config':BINARY_RL}),

    # ("SyntheticRLV2" , {'checkpoint': 48 , 'model_gin_config':SYN_MULTI_TASK_AGENT , 'train_gin_config':BINARY_RL}),


]