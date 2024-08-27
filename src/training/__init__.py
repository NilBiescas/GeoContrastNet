import sys
import importlib
from pathlib import Path
import os
# List files in the current directory

from pathlib import Path
CUR_DIR = Path(__file__).resolve().parent

module_names = [module_name for module_name in os.listdir(CUR_DIR) if module_name.endswith(".py") and not module_name.startswith("__")]

#MAIN_PROJECT_PATH = '/home/nbiescas/Desktop/CVC/CVC_internship'  # Replace with the actual path
#sys.path.append(MAIN_PROJECT_PATH)

def get_orchestration_func(training_method: str):
    for module_name in module_names:
        module = importlib.import_module('src.training.' + module_name[:-3])
        function = getattr(module, training_method, None)
        if function is not None:
            return function
    raise ValueError(f"Training method {training_method} not found")
