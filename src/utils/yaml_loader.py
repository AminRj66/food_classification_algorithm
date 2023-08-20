import yaml
from typing import Dict


# load training hyperparameters
def load_params(yaml_file: str) -> Dict:
    with open(yaml_file, 'r') as stream:
        params = yaml.safe_load(stream)
    return params
