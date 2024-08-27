import pickle
import torch
from sklearn.model_selection import train_test_split
import dgl
import PIL
import os
import sys
from tqdm import tqdm
from pathlib import Path

from src.training.masking import add_features
from src.training.utils import weighted_edges, discrete_bin_edges
from src.data.Dataset import FUNSD_loader
from src.data.utils import concat_paragraph2graph_edges
from utils import LoadConfig
from functools import partial

HERE = Path(os.path.dirname(os.path.abspath(__file__)))

def vector_func(config, edges):
    """
    Function to use with apply_edges, 
    it will concatenate the features of the nodes and the edges
    Based on the configuration that we pass
    """
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

def save_graphs(name, type_graphs, 
                imgs_train, imgs_val, imgs_test,
                train_graphs, val_graphs, test_graphs):
    bin_path_train  = HERE / f'PKL_Graphs/{name}/{type_graphs}/train.bin'
    bin_path_val    = HERE / f'PKL_Graphs/{name}/{type_graphs}/val.bin'
    bin_path_test   = HERE / f'PKL_Graphs/{name}/{type_graphs}/test.bin'
    path_imgs_train = HERE / f'PKL_Graphs/{name}/{type_graphs}/imgs_train.pkl'
    path_imgs_val   = HERE / f'PKL_Graphs/{name}/{type_graphs}/imgs_val.pkl'
    path_imgs_test  = HERE / f'PKL_Graphs/{name}/{type_graphs}/imgs_test.pkl'

    # Store the imgs of each document in order to be used with the UNET later
    with open (path_imgs_train, 'wb') as f,\
        open(path_imgs_val, 'wb') as f2,\
        open(path_imgs_test, 'wb') as f3:
        pickle.dump(imgs_train, f)
        pickle.dump(imgs_val, f2)
        pickle.dump(imgs_test, f3)
    # Store the graphs
    dgl.save_graphs(bin_path_train, dgl.unbatch(train_graphs))
    dgl.save_graphs(bin_path_val, dgl.unbatch(val_graphs))
    dgl.save_graphs(bin_path_test, dgl.unbatch(test_graphs))

def store_changed_graphs(name = 'FUNSD', create_new = False):
    """
    This function will store the graphs for doing the ablation study
    We will store the graphs with different features, obtained from the Stage-1 model
    with combination with visual and text.
    """
    # LOCATION WHERE THE GRAPH WILL BE STORED
    bin_path_train = "/home/nbiescas/Desktop/CVC/CVC_internship/PKL_Graphs/VISUAL/train_visual.bin"
    bin_path_val   = "/home/nbiescas/Desktop/CVC/CVC_internship/PKL_Graphs/VISUAL/val_visual.bin"
    bin_path_test  = "/home/nbiescas/Desktop/CVC/CVC_internship/PKL_Graphs/VISUAL/test_visual.bin"

    if create_new:
        
        data_train = FUNSD_loader(train=True, name = name)
        data_test  = FUNSD_loader(train=False,name = name)

        def store_features(graphs, desc = 'adding new features train'):
            for id, graph in enumerate(tqdm(graphs, desc=desc)):
                # Geometric features
                geometric = add_features(graph)
                graph.ndata['area'] = geometric[:, 4]
                graph.ndata['regional_encoding'] = geometric[:, 5:]
                graph.ndata['text_feat'] = graph.ndata['feat'][:, 4:304]
                graph.ndata['Geometric'] = geometric # bounding box, area and region encoding
                # Visual features
                graph.ndata['visual_feat']  = graph.ndata['feat'][:, 304:]
                
                # Edge features
                graph.edata['angle'] = weighted_edges(graph) 
                graph.edata['distance_normalized'] = graph.edata['weights']
                graph.edata['discrete_bin_edges'] = discrete_bin_edges(graph)   
 
            return graphs
        imgs_train_paths, imgs_test = data_train.paths, data_test.paths
        graphs_train, graphs_val, imgs_train, imgs_val = train_test_split(graphs_train, imgs_train_paths, test_size=0.2, random_state=42)

        graphs_train = store_features(graphs_train, desc='adding new features train')
        graphs_val   = store_features(graphs_val,   desc='adding new features validation')
        graphs_test  = store_features(data_test.graphs, desc='adding new features test') # TEST

        # Pretrain model: model obtained at STAGE-1, used for obtaining the embeddings 
        weights_pretrain = '/home/nbiescas/Desktop/CVC/CVC_internship/runs/run109/weights/model_71.pth'
        pretrain_model = torch.load(weights_pretrain).to('cuda:0')
        # Prepare the graphs, use the same config as the pretrain model
        # We need to prepare the graphs in order to be used in the pretrain model
        config_pretrain = LoadConfig('run109')
        message_func = partial(vector_func, config_pretrain) # Use the same config as the pretrain model

        train_graphs = dgl.batch(graphs_train).to('cuda:0')
        val_graphs   = dgl.batch(graphs_val).to('cuda:0')
        test_graphs  = dgl.batch(graphs_test).to('cuda:0')
        # Apply the transformation to each of the features of the graphs
        train_graphs.apply_edges(message_func)
        val_graphs.apply_edges(message_func)
        test_graphs.apply_edges(message_func)

        # Create the different graphs, with the different features
        # Visual, Geometric, Text, Visual+Geometric, Visual+Text, Geometric+Text, Visual+Geometric+Text ...
        pretrain_model.eval()
        with torch.no_grad():
            pretrain_embeddings = pretrain_model(train_graphs)
            preval_embeddings = pretrain_model(val_graphs)
            pretest_embeddings = pretrain_model(test_graphs)
            # ---------- NOW GEOM + TEXT + VISUAL --------------------------------
            train_graphs.ndata['feat'] = torch.cat((pretrain_embeddings,
                                                    train_graphs.ndata['text_feat'],
                                                    train_graphs.ndata['visual_feat']), dim=1)
            
            val_graphs.ndata['feat'] = torch.cat((preval_embeddings,
                                                    val_graphs.ndata['text_feat'],
                                                    val_graphs.ndata['visual_feat']), dim=1)
            
            test_graphs.ndata['feat'] = torch.cat((pretest_embeddings,
                                                    test_graphs.ndata['text_feat'],
                                                    test_graphs.ndata['visual_feat']), dim=1)
            
            save_graphs(name, imgs_train, imgs_val, imgs_test, 
                        train_graphs, val_graphs, test_graphs, type_graphs = 'GEOM_TEXT_VISUAL')
  
            # ----------NOW ONLY GEOM------------------------------
            train_graphs.ndata['feat']  = pretrain_embeddings
            val_graphs.ndata['feat']    = preval_embeddings
            test_graphs.ndata['feat']   = pretest_embeddings
            save_graphs(name, imgs_train, imgs_val, imgs_test,
                        train_graphs, val_graphs, test_graphs, type_graphs = 'GEOM')

            # ----------NOW ONLY GEOM + TEXT------------------------------
            train_graphs.ndata['feat'] = torch.cat((pretrain_embeddings,
                                                    train_graphs.ndata['text_feat']
                                                    ), dim=1)
            val_graphs.ndata['feat'] = torch.cat((preval_embeddings,
                                                    val_graphs.ndata['text_feat']
                                                    ), dim=1)
            test_graphs.ndata['feat'] = torch.cat((pretest_embeddings,
                                                    test_graphs.ndata['text_feat']
                                                    ), dim=1)
            save_graphs(name, imgs_train, imgs_val, imgs_test,
                        train_graphs, val_graphs, test_graphs, type_graphs = 'GEOM_TEXT')
            # ---------- NOW ONLY TEXT -------------------------------------
            train_graphs.ndata['feat'] = train_graphs.ndata['text_feat']
            val_graphs.ndata['feat'] = val_graphs.ndata['text_feat']
            test_graphs.ndata['feat'] = test_graphs.ndata['text_feat']
            save_graphs(name, imgs_train, imgs_val, imgs_test,
                        train_graphs, val_graphs, test_graphs, type_graphs = 'TEXT')
            
            # ----------- NOW ONLY VISUAL----------------------------------------------
            train_graphs.ndata['feat'] = train_graphs.ndata['visual_feat']
            val_graphs.ndata['feat'] = val_graphs.ndata['visual_feat']
            test_graphs.ndata['feat'] = test_graphs.ndata['visual_feat']
            save_graphs(name, imgs_train, imgs_val, imgs_test,
                        train_graphs, val_graphs, test_graphs, type_graphs = 'VISUAL')
            # ------------NOW VISUAL + GEOM ----------------------------------------------
            train_graphs.ndata['feat'] = torch.cat((pretrain_embeddings,
                                                    train_graphs.ndata['visual_feat']), dim=1)
            val_graphs.ndata['feat'] = torch.cat((preval_embeddings,
                                                    val_graphs.ndata['visual_feat']), dim=1)
            test_graphs.ndata['feat'] = torch.cat((pretest_embeddings,
                                                    test_graphs.ndata['visual_feat']), dim=1)
            save_graphs(name, imgs_train, imgs_val, imgs_test,
                        train_graphs, val_graphs, test_graphs, type_graphs = 'GEOM_VISUAL')
            # ------------- NOW VISUAL + TEXT ----------------------------------------------
            train_graphs.ndata['feat'] = torch.cat((train_graphs.ndata['text_feat'],
                                                    train_graphs.ndata['visual_feat']), dim=1)
            val_graphs.ndata['feat'] = torch.cat((val_graphs.ndata['text_feat'],
                                                    val_graphs.ndata['visual_feat']), dim=1)
            test_graphs.ndata['feat'] = torch.cat((test_graphs.ndata['text_feat'],
                                                    test_graphs.ndata['visual_feat']), dim=1)
            save_graphs(name, imgs_train, imgs_val, imgs_test,
                        train_graphs, val_graphs, test_graphs, type_graphs = 'TEXT_VISUAL')

if __name__ == '__main__':
    store_changed_graphs(name = 'FUNSD', create_new = True)
    store_changed_graphs(name = 'PAU',   create_new = True)