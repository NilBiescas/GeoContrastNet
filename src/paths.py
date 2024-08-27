from pathlib import Path

import os

# ROOT
ROOT = Path(os.path.dirname(os.path.abspath(__file__)))

# PROJECT TREE

CHECKPOINTS = Path('/home/nbiescas/Desktop/CVC/CVC_internship/CheckPoints')
DATA = ROOT / 'datasets'
TRAINING = ROOT / 'training'

TRAIN_GRAPH = ROOT / 'train_graph.pkl'
VAL_GRAPH   = ROOT / 'val_graph.pkl'

TASKS_YAML = ROOT / 'tasks.yaml'