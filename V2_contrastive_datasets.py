import torch
import pickle
import dgl
import os
import matplotlib.pyplot as plt
from functools import partial
import argparse
import torch.optim as optim
import numpy as np

from src.models import get_model_2
from pytorch_metric_learning import losses, miners, distances, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from src.training.utils import get_model, get_activation, get_scheduler
from src.training.utils_contrastive import obtain_embeddings, create_plots
from src.data.Dataset import Dataset_Kmeans_Graphs
from utils import LoadConfig, createDir
from paths import *

torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
np.random.seed(42)

def train(model, loss_func, mining_func, train_loader, optimizer, epoch):    
    model.train()
    total_loss = 0
    
    for batch_idx, (graph, labels) in enumerate(train_loader):

        graph = graph.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        embeddings = model(graph)

        indices_tuple = mining_func(embeddings, labels)
        loss = loss_func(embeddings, labels, indices_tuple)

        total_loss += loss.item()
        loss.backward()

        optimizer.step()
        if batch_idx % 20 == 0:
            print("Epoch {} Iteration {}: Loss = {}".format(epoch, batch_idx, loss))

    return total_loss / (batch_idx + 1)

def get_all_embeddings(dataset, model):
    tester = testers.BaseTester(dataloader_num_workers=2)
    return tester.get_all_embeddings(dataset, model, collate_fn=collate)

def accuracy_at_1(train_set, val_set, model, accuracy_calculator):
    model.eval()
    with torch.no_grad():
        train_embeddings, train_labels = get_all_embeddings(train_set, model)
        val_embeddings, val_labels = get_all_embeddings(val_set, model)
        accuracies = accuracy_calculator.get_accuracy(query = train_embeddings, reference = val_embeddings, query_labels = train_labels, reference_labels = val_labels)
    
    return accuracies["precision_at_1"]

def collate(batch):
    graphs, labels = map(list, zip(*batch))
    graphs = dgl.batch(graphs)

    return graphs, torch.cat(labels, dim=0)

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

def contrastive_features(loader: list, model, config, path_dir):
    """
    Obtain the embeddings obtained after the contrastive learning
    """
    createDir(config['root_dir'] / 'graphs_contrastive')
    with torch.no_grad():
        model.eval()
        graphs = []
        for graph, _ in loader:
            graph = graph.to(device)

            # Obtain the new embeddings
            embeddings = model(graph)

            graph.ndata['feat'] = embeddings
            graphs.extend(dgl.unbatch(graph))

    dgl.save_graphs(str(config['root_dir'] / 'graphs_contrastive' / path_dir), graphs)

"""
RECORDA QUE HAS CAMBIAT UNA PART DEL CODI FONT DE LA LLIBRERIA DELS TRIPLETS A LA FUNCIO compute_all_embeddings
xmin, ymin, xmax, ymax = box
"""
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--run-name', type=str, default='run172')
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()


    config = LoadConfig(dir = SETUPS_STAGE1, args_name = args.run_name)
    config['pretrainedWeights'] = args.checkpoint

    device = 'cuda' if torch.cuda.is_available() else "cpu"

    print("Distance metric {}".format(config['contrastive_learning']['distance_metric']))

    distance = getattr(distances, config['contrastive_learning']['distance_metric'])()

    mining_func = miners.TripletMarginMiner(margin = config['contrastive_learning']['margin'],
                                                distance = distance,
                                                type_of_triplets=config['contrastive_learning']['type_of_triplets'])

    loss_func = losses.TripletMarginLoss(margin=config['contrastive_learning']['margin'],
                                             distance = distance,
                                             triplets_per_anchor=config['contrastive_learning']['type_of_triplets'])

    accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k = 1)

    if config['pickle_path_train_kmeans'].endswith(".bin"):
        train_graphs, _ = dgl.load_graphs(config['pickle_path_train_kmeans'])
        val_graphs, _   = dgl.load_graphs(config['pickle_path_val_kmeans'])
        test_graphs, _  = dgl.load_graphs(config['pickle_path_test_kmeans'])
    else:
        with open(config['pickle_path_train_kmeans'], 'rb') as kmeans_train,\
             open(config['pickle_path_val_kmeans'], 'rb')   as kmeans_val,\
             open(config['pickle_path_test_kmeans'], 'rb')  as kmeans_test:
             train_graphs = pickle.load(kmeans_train)
             val_graphs = pickle.load(kmeans_val)
             test_graphs = pickle.load(kmeans_test)
             
    vector_func_partial = partial(vector_func, config) # Partial function to use with apply_edges
    def update_features(graphs: dgl.graph):
        batch = dgl.batch(graphs)
        batch.apply_edges(vector_func_partial)
        graphs = dgl.unbatch(batch)
        return graphs

    train_graphs = update_features(train_graphs)
    val_graphs   = update_features(val_graphs)
    test_graphs  = update_features(test_graphs)

        # Datasets
    train_dataset = Dataset_Kmeans_Graphs(train_graphs) # Train
    val_dataset = Dataset_Kmeans_Graphs(val_graphs)     # Val
    test_dataset = Dataset_Kmeans_Graphs(test_graphs)   # Test

    # Loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], collate_fn = collate, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], collate_fn = collate, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], collate_fn = collate, shuffle=False)

    config['activation'] = get_activation(config['activation'])
    model = get_model_2(config['model_name'], config).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    loss_evolution = []
    precision_evolution = []

    # Scheduler
    scheduler = get_scheduler(optimizer, config)
    max_precision = 0

    if config['pretrainedWeights'] == None:
        for epoch in range(config['epochs']):
            loss = train(model, loss_func, mining_func, train_loader, optimizer, epoch)
            precision = accuracy_at_1(train_dataset, val_dataset, model, accuracy_calculator)
            
            precision_evolution.append(precision)   
            loss_evolution.append(loss)
            if precision > max_precision:
                print("Saving model")
                max_precision = precision
                torch.save(model, config["weights_dir"] / f'model_{epoch}.pth')

            scheduler.step()

        plt.plot(loss_evolution)
        plt.savefig(config['root_dir'] / 'loss.png')
        plt.close()
        plt.plot(precision_evolution)
        plt.savefig(config['root_dir'] / 'precision.png')
        plt.close()
        # Load the best model
        model = torch.load(os.path.join(config["weights_dir"], os.listdir(config["weights_dir"])[-1]))
    else:
        statedict = torch.load(config['pretrainedWeights'])
        model.load_state_dict(statedict)
        #model = torch.load(os.path.join(config["weights_dir"], os.listdir(config["weights_dir"])[-1]))
        print("Skipping training")
        # Compute the test accuracy
    print("!!! Computing test accuracy !!!")
    precision = accuracy_at_1(train_dataset, test_dataset, model, accuracy_calculator)
    print("Test precision {}".format(precision))
    model.eval()
    # T-SNE Visualization
    createDir(config['root_dir'] / 'train_and_test_embeddings')
    createDir(config['root_dir'] / 'train_embeddings')
    createDir(config['root_dir'] / 'test_embeddings')

    train_loader.shuffle = False
    contrastive_features(train_loader, model, config, 'train_contrastive.bin')
    contrastive_features(test_loader, model, config, 'test_contrastive.bin')
    contrastive_features(val_loader, model, config, 'val_contrastive.bin')
