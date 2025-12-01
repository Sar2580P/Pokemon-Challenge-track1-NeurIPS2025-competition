# 🎮 Adaptive Regret Minimization for Multi-Task Agents via Reactive Hedge

<p align="center">
  <strong>Team: PAC-srsk-1729 | NeurIPS 2025 - Pokémon Challenge Track 1</strong>
</p>

<br>

<div align="center">
    <img src="media/metamon_banner.png" alt="Metamon Banner" width="720">
</div>

<br>

<p align="center">
  <strong>👁️U👁️</strong>
</p>


<p align="center">
  <a href="https://github.com/Sar2580P/Pokemon-Challenge-track1-NeurIPS2025-competition">
    <img src="https://img.shields.io/badge/GitHub-Repository-blue?style=for-the-badge&logo=github" alt="GitHub">
  </a>
  <a href="https://huggingface.co/jakegrigsby/metamon/tree/main">
    <img src="https://img.shields.io/badge/🤗-Model_Checkpoints-yellow?style=for-the-badge" alt="HuggingFace">
  </a>
</p>

---

## 📋 Abstract

We introduce a **"Tribe of Experts"** architecture for decision-making in multi-task reinforcement learning environments. To effectively ensemble a diverse population of agents (Entities) trained with varying hyperparameters, we propose a **Reactive Hedge** algorithm. This method minimizes cumulative regret during inference by dynamically weighting experts based on a normalized Temporal Difference (TD) error proxy. Furthermore, we implement an adaptive learning rate mechanism and a refined nucleus sampling strategy to balance exploration and stability.

---

## 🏆 Competition Results

| Track Category | Rank | Notes |
|:---------------|:----:|:------|
| **Track 1 - Gen1ou** | 🥉 **3rd** | High stability observed in expert ensemble |
| **Track 1 - Gen9ou** | **7th** | Impacted by team composition meta-game |

---

## 🚀 Quick Start

### Prerequisites

- **CUDA-capable GPU** (recommended)
- **Python 3.10**
- **Conda** (for environment management)
- **Node.js** (for Pokémon Showdown server)

### 1. Clone the Repository

```bash
git clone https://github.com/Sar2580P/Pokemon-Challenge-track1-NeurIPS2025-competition.git
cd Pokemon-Challenge-track1
```

### 2. Create Environment

```bash
conda env create -f environment.yml
conda activate metamon
```

### 3. Set Environment Variables

```bash
export PYTHONPATH=$PYTHONPATH:.
export METAMON_CACHE_DIR="PAC-dataset"
```

### 4. Download Model Checkpoints

Model weights are available on HuggingFace: [jakegrigsby/metamon](https://huggingface.co/jakegrigsby/metamon/tree/main)

```bash
# Example: Download a specific checkpoint
modal volume get pokemon-showdown-gen1 results/HRM_Pokemon_Gen1/ckpts/latest/policy.pt model_weights.pt
```

---

## 📁 Project Structure

```
Pokemon-Challenge-track1/
│
├── 📂 custom/                      # Custom training components
│   ├── 📂 gen1/                    # Gen1-specific training scripts
│   │   ├── configs/                # Gin configuration files
│   │   ├── scripts/                # Shell scripts for training
│   │   ├── train.py                # Main training script
│   │   ├── train_opponent_modeling.py
│   │   ├── train_traj_encoder_KD.py
│   │   ├── train_vae_prior.py
│   │   └── evaluate.py
│   │
│   ├── 📂 gen9/                    # Gen9-specific training scripts
│   │   ├── configs/
│   │   ├── scripts/
│   │   ├── train.py
│   │   ├── train_opponent_modeling.py
│   │   ├── train_vae_prior.py
│   │   └── evaluate.py
│   │
│   ├── 📂 hrm_utils/               # HRM utility modules
│   │   ├── layers.py               # Custom attention & neural network layers
│   │   ├── common.py               # Common utilities
│   │   ├── losses.py               # Loss functions
│   │   ├── modules.py              # Neural network modules
│   │   └── sparse_embedding.py     # Sparse embedding utilities
│   │
│   ├── hrm_agent.py                # HRM Multi-Task Agent implementation
│   ├── traj_encoder.py             # Trajectory Encoder (HRM-based)
│   ├── experiment.py               # Custom experiment class
│   ├── evaluate.py                 # Evaluation utilities
│   └── utils.py                    # General utilities
│
├── 📂 inference/                   # Inference-time components
│   ├── 📂 configs/                 # Inference configurations
│   │   ├── models/                 # Model architecture configs
│   │   └── training/               # Training hyperparameter configs
│   │
│   ├── 📂 play_battle/             # Battle playing scripts
│   │   └── gen1.py, gen9.py
│   │
│   ├── 📂 scripts/                 # Evaluation shell scripts
│   │   ├── eval_gen1.sh
│   │   └── eval_gen9.sh
│   │
│   ├── agent_tribe.py              # 🌟 Tribe of Experts Implementation
│   ├── population_config.py        # Expert population configuration
│   ├── experiment_tribe.py         # Tribe experiment manager
│   └── evaluate.py                 # Inference evaluation
│
├── 📂 metamon/                     # Core metamon library (forked)
├── 📂 amago/                       # AMAGO framework (forked)
├── 📂 server/                      # Pokémon Showdown server
├── 📂 team_design/                 # Team composition analysis
│
├── environment.yml                 # Conda environment specification
├── pyproject.toml                  # Python project configuration
└── pokemon_writeup.pdf             # Competition report
```

---

## 🧠 Architecture Overview

### The Reactive Hedge Algorithm

The core mechanism maintains a probability distribution (weights) over the expert population. At each time step, expert weights are updated based on TD-error:

```
w_{i,t+1} = (w_{i,t} · exp(-η_{i,t} · L_{i,t})) / Σ_j(w_{j,t} · exp(-η_{j,t} · L_{j,t}))
```

### Loss Design: Normalized TD-Error Proxy

```python
δ_i,t = (r_t + γ_i · V_i(s_{t+1})) - V_i(s_t)   # TD Error
z_i,t = δ_i,t / (σ_popart + ε)                   # Normalize
L_i,t = -tanh(z_i,t)                             # Proxy Loss
```

### Dynamic Ensembling Pipeline

1. **Per-Gamma Refinement**: Each expert refines its policy using internal advantage estimates
2. **Top-K Masking**: Filters poor-performing experts
3. **Nucleus Sampling**: Final action selection via Top-p sampling

---

## 🏋️ Training

### Standard Training (Gen1)

```bash
cd custom/gen1
bash scripts/train.sh
```

### Training with Opponent Modeling

```bash
python custom/gen1/train_opponent_modeling.py
```

### Knowledge Distillation for Trajectory Encoder

```bash
python custom/gen1/train_traj_encoder_KD.py
```

### VAE Prior Training

```bash
python custom/gen1/train_vae_prior.py
```

### Configuration (Gin)

Training configurations use Google's Gin library. Key config files:

| Config Type | Location |
|-------------|----------|
| Model Architecture | `custom/gen1/configs/` or `custom/gen9/configs/` |
| Training Hyperparameters | `inference/configs/training/` |
| Agent Configuration | `inference/configs/models/` |

---

## 🎯 Inference & Evaluation

### Running Evaluation (Gen1)

```bash
bash inference/scripts/eval_gen1.sh
```

### Running Evaluation (Gen9)

```bash
bash inference/scripts/eval_gen9.sh
```

### Evaluation Options

Edit the shell scripts to configure:

| Parameter | Description | Options |
|-----------|-------------|---------|
| `EVAL_TYPE` | Opponent type | `heuristic`, `il`, `ladder`, `pokeagent` |
| `TOTAL_BATTLES` | Number of battles | Integer |
| `TEAM_SET` | Team composition | `competitive` |
| `BATTLE_BACKEND` | Backend system | `metamon`, `poke-env` |

### Expert Population Configuration

Configure the expert ensemble in `inference/population_config.py`:

```python
GEN1_Models = [
    ("SyntheticRLV2", {'checkpoint': 40, 'model_gin_config': ..., 'train_gin_config': ...}),
    ("SyntheticRLV2", {'checkpoint': 46, ...}),
    # Add more experts...
]
```

---

## 🔧 Key Components

### `custom/hrm_agent.py`

The **HRM_MultiTaskAgent** class implements:
- Multi-gamma policy learning
- Advantage-weighted policy refinement
- Nucleus sampling for action selection
- Component-wise checkpoint initialization

### `custom/traj_encoder.py`

The **HRMTrajEncoder** implements a Hierarchical Reasoning Model with:
- Chunked attention for long sequences
- Temporal summary aggregation
- VAE-based opponent modeling (optional)

### `inference/agent_tribe.py`

The **InferenceTribeMTA** class implements:
- Reactive Hedge weight updates
- RMS-Prop style adaptive learning rates
- Top-K expert filtering
- Dynamic policy ensembling

---

## 📊 Evaluation Metrics

The framework tracks several diagnostic metrics:

| Metric | Description |
|--------|-------------|
| **Policy Entropy (H_π)** | Uncertainty of the ensembled policy |
| **Weight Entropy (H_w)** | Expert diversity / democratic ensemble indicator |
| **Confidence Margin** | Probability gap between top two actions |
| **TD Error** | Temporal difference error for weight updates |

---

## 🛠️ Modal Deployment (Cloud)

For cloud-based training/evaluation using Modal:

```bash
modal shell --volume my-volume
```

---

## 📚 Dependencies

Key dependencies from `environment.yml`:

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | CUDA 12.x | Deep Learning |
| `gymnasium` | 0.29.1 | RL environments |
| `einops` | 0.8.1 | Tensor operations |
| `gin-config` | 0.5.0 | Configuration |
| `wandb` | 0.21.1 | Experiment tracking |
| `accelerate` | 1.10.0 | Distributed training |

---

## 📖 References

1. **Van Erven, T., et al.** (2011). *Adaptive hedge*. NeurIPS.
2. **Freund, Y., & Schapire, R. E.** (1997). *A decision-theoretic generalization of on-line learning*. JCSS.
3. **Ren, Y., et al.** (2024). *HRM: Hierarchical Reasoning Model*. ICLR.

---

## 🙏 Acknowledgments

Special thanks to:
- **Competition Organizers** for hosting this challenging event
- **Jake Grigsby** for detailed guidance and support throughout the competition
- The **metamon** and **AMAGO** framework developers

---

## 📄 License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

---
