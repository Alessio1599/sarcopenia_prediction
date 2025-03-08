import yaml
from pathlib import Path

# Automatically find the config file in the correct location
CONFIG_PATH = Path(__file__).parent.parent / "configs/config.yml"


if not CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"Configuration file not found: {CONFIG_PATH}. Please create it from `config.example.yml`."
    )


# Load the general configuration
with open(CONFIG_PATH, "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

