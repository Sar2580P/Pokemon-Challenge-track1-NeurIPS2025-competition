

import modal
from pathlib import Path # Add this import
from omegaconf import OmegaConf 
# ... other imports

# --- Define the paths once at the top ---
HOME_DIR = Path.home()/"Pokemon-Challenge-track1"

print(HOME_DIR)