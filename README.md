# GeoContrastNet

This repository provides the implementation of GeoContrastNet, as detailed in our paper: GeoContrastNet: Contrastive Key-Value Edge Learning for Language-Agnostic Document Understanding. GeoContrastNet is a two-stage Graph Neural Network designed for Named Entity Recognition and Link Prediction in document entities identified by YOLO or any other object detector. The first stage learns geometric representations, which, alongside visual features, are utilized in the second stage for the tasks mentioned.

## Features
- Code for training and testing both stages of the model.
- Pretrained weights for immediate model evaluation.
- Graphs for both stages, including how to generate your own.
- Instructions for replicating paper experiments.

## Getting Started

### Prerequisites
- Git
- Conda
- Python

### Setup Instructions

1. **Clone the Repository**
   ```
   git clone https://github.com/NilBiescas/CVC_internship.git
   ```

4. **Create and Activate a Conda Environment**
   ```
   conda create --name GeoNet python=3.9
   conda activate GeoNet
   ```

5. **Install the dependencies**
   ```
   conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
   conda install -c dglteam/label/cu118 dgl
   pip install pandas segmentation-models-pytorch scikit-learn seaborn wget torchdata pydantic
   ```
6. **Download the datasets**
   ```
   #Enter to src/data
   cd src/data
   # And exectue download.py
   python download.py
   ```

### Configuration Files
- YAML configuration files for all experiments are provided. You may use these directly or as templates for custom experiments.
- Match YAML files with their corresponding pretrained weights for experimentation. Each yaml file goes with their matching pretrained weights, they share the same name.

### Training and Testing
## Stage 1
- **Training**
  ```
  python build_graphs.py --run-name <your_yaml_file>
  ```
- **Testing**
  ```
  python build_graphs.py --run-name <your_yaml_file> --checkpoint <path_to_pretrained_weights>
  ```
## Stage 2
- **Training on FUNSD**
  ```
  python main.py --run-name <your_yaml_file>
  ```
- **Testing**
  ```
  python main.py --run-name <your_yaml_file> --checkpoint <path_to_pretrained_weights>
  ```

### Training with your own graphs

Training is done in two stages, each requiring its own YAML file. Example YAML files for each stage can be found in the setups_stage1 and setups_stage2 folders.

In the first stage, the expected input is a dgl.graph that can be created from various domains. This graph should have geometric features in its nodes and edges. The graph for the second stage will be the output graph from the first stage.

The YAML file for the firts stage defines various hyperparameters for the contrastive setting. In this stage, we use geometric features. The yaml contains a key named features, that contains two sub keys: node and edge. This fields define the geometric features of the nodes and edges used in this module. The node field defines the features in g.ndata that will be used and the same in the edge field and g.edata. Additionally, the YAML file has a model key, which refers to the model definition in models/contrastive_model.py.

Once your dgl.graph includes g.ndata and g.edata as described, run the following command:
```
python V2_contrastive_datasets.py --run-name <your_yaml_file_stage1>
```
When the training completes, three graphs will be created: one each for training, validation, and testing data.

Next, create a second YAML file for the second stage, specifying the paths to the graphs created in the first stage. Then run:

```
python main.py --run-name <your_yaml_file_stage2>
```

## Additional Resources
- **CHEKPOINTS**: https://drive.google.com/drive/folders/1UlbQZPdrphr-qdF64EveORM31zBWTMuj?usp=drive_link
- **Stage 1 YAML files**: Located in `setups_stage1` folder.
- **Stage 2 YAML files**: Located in `setups_stage2` folder.
- **Stage 1 Graphs**: Located in `graphs_stage1` folder.
- **Stage 2 Graphs**: Located in `graphs_stage2` folder.

## Authors
