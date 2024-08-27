from typing import Any
from torch.utils.data import Dataset, dataloader
import dgl
import torch
from sklearn.model_selection import train_test_split
import pickle
import pprint
from functools import partial
from paths import HERE

class Unet_Dataset(Dataset):
    def __init__(self, graphs, path):
        self.graphs = graphs
        with open(path, 'rb') as file:
            self.imgs_paths = pickle.load(file)
        
        self.imgs_paths = [path.replace('/data2/users/sbiswas/nil_biescas', HERE.__str__()).replace('data/doc2_graph/DATA', 'data/datasets') 
                           for path in self.imgs_paths] 

    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, index):
        return self.graphs[index], self.imgs_paths[index]  
    
class Dataset_Kmeans_Graphs(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):

        return self.graphs[idx], self.graphs[idx].ndata['label']
    
class Partition_DatasetGraphsFUNSD(Dataset):
    def __init__(self, graphs, n, m):
        self.graphs = graphs
        self.n      = n
        self.m      = m
        self.new_graphs = self.create_graphs()

    def create_graphs(self):
        return [dgl.batch(self.get_sub_graphs(n = self.n, m = self.m, graph = graph)) 
                for graph in self.graphs]
    
    def obtain_partitions(self, n, m, boxes):
        def center(rect):
            new_x = rect[:, [0, 2]].sum(1).unsqueeze(1) / 2
            new_y = rect[:, [1, 3]].sum(1).unsqueeze(1) / 2

            return torch.cat([new_x, new_y], dim=1)
        w, h = 1, 1
        w_chunk = w / n
        h_chunk = h / m

        centers = center(boxes)

        # Compute the indices of the partition for each bounding box
        x_indices = (centers[:, 0] / w_chunk).long()
        y_indices = (centers[:, 1] / h_chunk).long()
        indices = x_indices + y_indices * n
        # Group the bounding boxes by partition
        partitions = []
        for i in range(n * m):
            partition_boxes = torch.where(indices == i)[0].tolist()
            partitions.append(partition_boxes)

        return partitions
    
    def get_sub_graphs(self, n, m, graph):
        partitions = self.obtain_partitions(n, m, graph.ndata['geom'])
        subgraphs = []
        for partion in partitions:
            if len(partion) > 0:
                subgraphs.append(graph.subgraph(partion))
        return subgraphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.new_graphs[idx]

def vector_func(config, edges):

    node_feat_list = []
    for node_feat in config['features']['node']:
        node_feat = edges.src[node_feat]
        if len(node_feat.size()) == 1:
            node_feat = node_feat.unsqueeze(1)
        node_feat_list.append(node_feat)

    for edge_feat in config['features']['edge']:
        edge_feat = edges.data[edge_feat]
        if len(edge_feat.size()) == 1:
            edge_feat = edge_feat.unsqueeze(1)
        node_feat_list.append(edge_feat)
    
    msg = torch.cat(node_feat_list, dim=1)
    return {'m': msg}


def edgesAggregation_kmeans_graphs(train, config):
    vector_func_partial = partial(vector_func, config)
    if train:
        print("-> Loading V2 kmeans partitioned graphs for training")
        with open(config['pickle_path_train_kmeans'], 'rb') as kmeans_train:
            train_graphs = pickle.load(kmeans_train)
        
        batch = dgl.batch(train_graphs)
        batch.apply_edges(vector_func_partial)
        train_graphs = dgl.unbatch(batch)

        train_graphs, val_graphs, _, _ = train_test_split(train_graphs, torch.ones(len(train_graphs), 1), test_size=config['val_size'], random_state=42)
        # Datasets
        print("-> Number of training graphs: ", len(train_graphs))
        print("-> Number of validation graphs: ", len(val_graphs))

        if config['loader']:
            train_loader = torch.utils.data.DataLoader(train_graphs, batch_size=config['batch_size'], collate_fn = dgl.batch, shuffle=True)
            validation_loader = torch.utils.data.DataLoader(val_graphs, batch_size=config['batch_size'], collate_fn = dgl.batch, shuffle=False)
            return train_loader, validation_loader
        
        train_graphs = dgl.batch(train_graphs)
        val_graphs = dgl.batch(val_graphs)

        # Creation of the features vector
        return train_graphs, val_graphs
    else:
        print("-> Loading kmeans partitioned graphs for testing")
        with open(config['pickle_path_test_kmeans'], 'rb') as kmeans_test:
            test_graphs = pickle.load(kmeans_test)
        print("-> Number of test graphs: ", len(test_graphs))
        
        batch = dgl.batch(test_graphs)
        batch.apply_edges(vector_func_partial)
        test_graphs = dgl.unbatch(batch)
        
        if config['loader']:
            test_loader = torch.utils.data.DataLoader(test_graphs, batch_size=config['batch_size'], collate_fn = dgl.batch, shuffle=False)
            return test_loader
        
        test_graphs = dgl.batch(test_graphs)

        return test_graphs

def kmeans_graphs(train, config):
    if train:
        print("-> Loading kmeans partitioned graphs for training")
        with open(config['pickle_path_train_kmeans'], 'rb') as kmeans_train:
            train_graphs = pickle.load(kmeans_train)
        
        train_graphs, val_graphs, _, _ = train_test_split(train_graphs, torch.ones(len(train_graphs), 1), test_size=config['val_size'], random_state=42)
        # Datasets
        print("-> Number of training graphs: ", len(train_graphs))
        print("-> Number of validation graphs: ", len(val_graphs))

        if config['loader']:

            train_loader = torch.utils.data.DataLoader(train_graphs, batch_size=config['batch_size'], collate_fn = dgl.batch, shuffle=True)
            validation_loader = torch.utils.data.DataLoader(val_graphs, batch_size=config['batch_size'], collate_fn = dgl.batch, shuffle=False)
            return train_loader, validation_loader
        
        train_graphs = dgl.batch(train_graphs)
        val_graphs = dgl.batch(val_graphs)

        # Creation of the features vector
        return train_graphs, val_graphs
    else:
        print("-> Loading kmeans partitioned graphs for testing")
        with open(config['pickle_path_test_kmeans'], 'rb') as kmeans_test:
            test_graphs = pickle.load(kmeans_test)
        print("-> Number of test graphs: ", len(test_graphs))
        test_loader = torch.utils.data.DataLoader(test_graphs, batch_size=config['batch_size'], collate_fn = dgl.batch, shuffle=False)
        return test_loader


def dataloaders_funsd(train, config):
    if train:
        # Load train grahps from disk
        with open(config['pickle_path_train'], 'rb') as path_train:
            train_graphs = pickle.load(path_train)

        train_graphs, val_graphs, _, _ = train_test_split(train_graphs, torch.ones(len(train_graphs), 1), test_size=config['val_size'], random_state=42)
        # Datasets
        print("-> Number of training graphs: ", len(train_graphs))
        print("-> Number of validation graphs: ", len(val_graphs))
        train_dataset      = Partition_DatasetGraphsFUNSD(train_graphs, config['Dataset_partitions']['n'], config['Dataset_partitions']['m'])
        validation_dataset = Partition_DatasetGraphsFUNSD(val_graphs, config['Dataset_partitions']['n'], config['Dataset_partitions']['m'])
        # Loaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], collate_fn = dgl.batch, shuffle=True)
        validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=config['batch_size'], collate_fn = dgl.batch, shuffle=False)
        
        return train_loader, validation_loader
    
    else:
        # Load test graphs from disk
        with open(config['pickle_path_test'], 'rb') as path_test:
            test_graphs = pickle.load(path_test)

        test_dataset = Partition_DatasetGraphsFUNSD(test_graphs, config['Dataset_partitions']['n'], config['Dataset_partitions']['m'])
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], collate_fn = dgl.batch, shuffle=False)
        return test_loader

#from .doc2_graph.data.dataloader import Document2Graph
#from .doc2_graph.paths import FUNSD_TRAIN, FUNSD_TEST, TRAIN_SAMPLES, TEST_SAMPLES
#from .doc2_graph.paths import PAU_TRAIN, PAU_TEST
#from .doc2_graph.utils import get_config

def FUNSD_loader(train = True, name = 'FUNSD'):
    config = get_config('preprocessing')

    pprint.pprint(config, indent=4, width=1)
    print("\n")
    if name == 'FUNSD':
        if train:
            print("TRAIN")
            return Document2Graph(name='FUNSD TRAIN', src_path=FUNSD_TRAIN, device = "cuda:0", output_dir=TRAIN_SAMPLES)
        else:
            print("TEST")
            return Document2Graph(name='FUNSD TEST', src_path=FUNSD_TEST, device = "cuda:0", output_dir=TEST_SAMPLES)
    elif name == 'PAU':
        if train:
            print("TRAIN")
            return Document2Graph(name='PAU TRAIN', src_path=PAU_TRAIN, device = "cuda:0", output_dir=TRAIN_SAMPLES)
        else:
            print("TEST")
            return Document2Graph(name='PAU TEST', src_path=PAU_TEST, device = "cuda:0", output_dir=TEST_SAMPLES)