from pathlib import Path
import torch
import argparse
import pprint
import numpy as np

# OWN MODULES
from src.training import get_orchestration_func
from utils import LoadConfig
from paths import *
# Ensure deterministic behavior
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--run-name', type=str, help='Name of the run', required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    config = LoadConfig(dir_name = SETUPS_STAGE2, args_name = args.run_name)
    config['network']['checkpoint'] = args.checkpoint
    
    print(f"\n{'- '*10}CONFIGURATION{' -'*10}\n")
    pprint.pprint(config, indent=10, width=1)
    print("\n\n")
    
    # We should firts do stage-1 if the graphs are not already created with the feat features
        
    orchestration_func = get_orchestration_func(config['train_method']) # Load orchestration function
    model = orchestration_func(config)

    print("Ja s'ha acabat")