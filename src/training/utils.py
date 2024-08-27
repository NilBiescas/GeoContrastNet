#from ..models.autoencoders import GAE, GSage_AE, GIN_AE, GAT_AE
#from ..models.mask_autoencoder_modified_sage import AUTOENCODER_MASK_MODF_SAGE
#from ..models.mask_aut_modifed_edges import V2_AUTOENCODER_MASK_MODF_SAGE
#from ..models.autoencoder_drop_edge import E2E
#from ..models.mask_autoencoders import GAT_masking
#from ..models.autoencoders import device
#from ..models.mask_autoencoders import SELF_supervised
#from ..models.contrastive_models import AUTOENCODER_MASK_MODF_SAGE_CONTRASTIVE


import math
#from ..data.doc2_graph.utils import get_config
from sklearn.utils import class_weight
import numpy as np
import torch

import torch.nn.functional as F
from typing import Tuple
import pickle
import dgl

from sklearn.model_selection import train_test_split
from typing import Tuple
from ..data.Dataset import FUNSD_loader

from ..paths import TRAIN_GRAPH, VAL_GRAPH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def polar(rect_src : list, rect_dst : list) -> Tuple[int, int]:
    """Compute distance and angle from src to dst bounding boxes (poolar coordinates considering the src as the center)
    Args:
        rect_src (list) : source rectangle coordinates
        rect_dst (list) : destination rectangle coordinates
    
    Returns:
        tuple (ints): distance and angle
    """
    
    # check relative position
    center = lambda rect: ((rect[2]+rect[0])/2, (rect[3]+rect[1])/2)

    # evaluate reciprocal position
    sc = center(rect_src)
    ec = center(rect_dst)
    new_ec = (ec[0] - sc[0], ec[1] - sc[1])
    angle = int(math.degrees(math.atan2(new_ec[1], new_ec[0])) % 360)
    # Pasar els angles 
    
    return angle

def discrete_positions_binarize(rect_src : list, rect_dst : list) -> Tuple[int, int]:
    # check relative position
    left = (rect_dst[2] - rect_src[0]) <= 0
    bottom = (rect_src[3] - rect_dst[1]) <= 0
    right = (rect_src[2] - rect_dst[0]) <= 0
    top = (rect_dst[3] - rect_src[1]) <= 0
    
    vp_intersect = (rect_src[0] <= rect_dst[2] and rect_dst[0] <= rect_src[2]) # True if two rects "see" each other vertically, above or under
    hp_intersect = (rect_src[1] <= rect_dst[3] and rect_dst[1] <= rect_src[3]) # True if two rects "see" each other horizontally, right or left
    rect_intersect = vp_intersect and hp_intersect

    return torch.where(torch.tensor([left, bottom, right, top, vp_intersect, hp_intersect, rect_intersect]), 1, 0)

def discrete_positions(rect_src : list, rect_dst : list) -> Tuple[int, int]:
    """Compute distance and angle from src to dst bounding boxes (poolar coordinates considering the src as the center)
    Args:
        rect_src (list) : source rectangle coordinates
        rect_dst (list) : destination rectangle coordinates
    
    Returns:
        tuple (ints): distance and angle
    """
    
    # check relative position
    left = (rect_dst[2] - rect_src[0]) <= 0
    bottom = (rect_src[3] - rect_dst[1]) <= 0
    right = (rect_src[2] - rect_dst[0]) <= 0
    top = (rect_dst[3] - rect_src[1]) <= 0
    
    vp_intersect = (rect_src[0] <= rect_dst[2] and rect_dst[0] <= rect_src[2]) # True if two rects "see" each other vertically, above or under
    hp_intersect = (rect_src[1] <= rect_dst[3] and rect_dst[1] <= rect_src[3]) # True if two rects "see" each other horizontally, right or left
    rect_intersect = vp_intersect and hp_intersect
    
    if rect_intersect:
        return 0 #'intersect'
    elif top and left:
        return 1 #'top_left'
    elif left and bottom:
        return 2 #'bottom_left'
    elif bottom and right:
        return 3 #'bottom_right'
    elif right and top:
        return 4 #'top_right'
    elif left:
        return 5 #'left'
    elif right:
        return 6 #'right'
    elif bottom:
        return 7 #'bottom'
    elif top:
        return 8 #'top'  
    
    #number2_position = {0:'intersect', 1:'top_left', 2:'bottom_left', 3:'bottom_right', 4:'top_right', 5:'left', 6:'right', 7:'bottom', 8:'top'}

def get_relative_positons(data):

    for graph in data.graphs:
        src, dst = graph.edges()
        discret_info = []
        for src_idx, dst_idx in zip(src, dst):
            src_idx = src_idx.item()
            dst_idx = dst_idx.item()
            relative_position = discrete_positions(graph.nodes[src_idx][0]['geom'][0], graph.nodes[dst_idx][0]['geom'][0])
            discret_info.append(relative_position)
        graph.edata['discrete_info'] = torch.tensor(discret_info)
    return data

def region_encoding(graph):
    """
    Encode the region of the bounding box in a 4x4 grid
    """
    def get_encoding(coord):
        limit = 1 # High and with of the image after normalization
        return np.where(coord < limit / 2, np.where(coord < limit / 4, 11, 21), np.where(coord > limit / 4 * 3, 22, 12))
    
    return torch.from_numpy(get_encoding(graph.ndata['geom']))

def spatial_features(graphs):
    graphs_nodes = graphs.batch_num_nodes().tolist()
    start = 0
    features = []
    for len_graphs in range(graphs.batch_num_nodes().__len__()):
        geom_polar = []
        end = start + graphs_nodes[len_graphs]
        for i in range(start, end):
            edges = graphs.out_edges(i)
            edge_features = graphs.edata['feat'][edges[1].long()]
            geom_polar.append(edge_features.sum(dim=0))

        geom_polar = torch.stack(geom_polar)
        # Z score normalzation of the features of each graph
        geom_polar = (geom_polar - geom_polar.mean(dim=0)) / geom_polar.std(dim=0)
        features.append(geom_polar)
        start = end

    return torch.cat(features)

def concat_geom_edge_featurs(graphs):
    geom_polar = []
    for i in range(graphs.num_nodes()):
        edges = graphs.out_edges(i)
        edge_features = graphs.edata['feat'][edges[1].long()]
        geom_polar.append(edge_features.sum(dim=0))

    return torch.stack(geom_polar)

def discrete_bin_edges(graph):
    u, v = graph.edges()
    srcs, dsts =  u.tolist(), v.tolist()
    discrete_positions_bin_list = []
    for pair in zip(srcs, dsts):
        src = graph.ndata['geom'][pair[0]]
        dst = graph.ndata['geom'][pair[1]]

        discrete_positions_bin_list.append(discrete_positions_binarize(src, dst))

    return torch.stack(discrete_positions_bin_list)

def weighted_edges(graph):
    u, v = graph.edges()
    srcs, dsts =  u.tolist(), v.tolist()
    angles = []
    angle_ = lambda x: x if x < 180 else x - 360
    for pair in zip(srcs, dsts):
        src = graph.ndata['geom'][pair[0]]
        dst = graph.ndata['geom'][pair[1]]

        angle_degrees = polar(dst, src)
        angle_degrees = angle_(angle_degrees)
        angle_radians = math.radians(angle_degrees)
        angles.append(angle_radians)

    return torch.tensor(angles)

def load_graphs(load=False):
    if load:
        data = FUNSD_loader(train=True)
        print("SAVING ")
        train_graphs, val_graphs, _, _ = train_test_split(data.graphs, torch.ones(len(data.graphs), 1), test_size=0.2, random_state=42)
        print("-> Number of training graphs: ", len(train_graphs))
        print("-> Number of validation graphs: ", len(val_graphs))

        #Graph for training
        train_graphs = get_relative_positons(train_graphs)
        train_graph = dgl.batch(train_graphs)
        train_graph = train_graph.int()

        #Graph for validating
        val_graphs = get_relative_positons(val_graphs)
        val_graph = dgl.batch(val_graphs)
        val_graph = val_graph.int().to(device)

        with open (TRAIN_GRAPH, 'wb') as train:
            pickle.dump(train_graph, train)

        with open(VAL_GRAPH, 'wb') as val:
            pickle.dump(val_graph, val)
        


    print("LOADING THE TRAINING AND VALIDATION GRAPHS FROM MEMORY")
    with open(TRAIN_GRAPH, 'rb') as train:
        train_graph = pickle.load(train)

    with open(VAL_GRAPH, 'rb') as val:
        val_graph   = pickle.load(val)

    return train_graph, val_graph


def compute_crossentropy_loss(scores : torch.Tensor, labels : torch.Tensor):
    w = class_weight.compute_class_weight(class_weight='balanced', classes= np.unique(labels.cpu().numpy()), y=labels.cpu().numpy())
    return torch.nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32).to('cuda:0'))(scores, labels)

def get_scheduler(optimizer, config):
    if config['schedule_name'] == 'MultiStepLR':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['milestones'], gamma=config['gamma'])
    elif config['schedule_name'] == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['T_max'], eta_min=config['eta_min'])

def get_optimizer(model, config):
    if config['optimizer'] == 'SGD': 
        return torch.optim.SGD(model.parameters(), lr = config['lr'], weight_decay = config['weight_decay'], momentum=config['momentum'])
    elif config['optimizer'] == 'ADAMW':
        return torch.optim.AdamW(model.parameters(), lr = config['lr'], weight_decay = config['weight_decay'])
    else:
        raise NotImplementedError

def get_activation(activation):
    if activation == 'relu':
        return F.relu
    elif activation == 'leaky_relu':
        return F.leaky_relu
    elif activation == 'tanh':
        return torch.tanh
    elif activation == 'elu':
        return F.elu
    elif activation == 'prelu':
        return F.prelu

def get_model(config):
    raise NotImplementedError
#    #Dimensions of the autencoder
#    config['edge_pred_features'] = int((math.log2(get_config('preprocessing').FEATURES.num_polar_bins) + config['node_classes'])*2)
#    activation = get_activation(config['activation'])
#
#    if config['model_name'] == 'SAGE':
#        model = GSage_AE(config['layers_dimensions'] ).to(device)
#    elif config['model_name'] == 'GAE':
#        model = GAE(config['layers_dimensions'] ).to(device)
#    elif config['model_name'] == 'GIN':
#        model = GIN_AE(config['layers_dimensions'] ).to(device)
#    elif config['model_name'] == 'GAT':
#        model = GAT_masking(dimensions_layers           = config['layers_dimensions'] ,
#                            edge_classes                = config['edge_classes'],
#                            dropout                     = config['dropout'],
#                            edge_pred_features          = config['edge_pred_features'],
#                            node_classes    = config['node_classes'],
#                            concat_hidden   = config['concat_hidden'],
#                            mask_rate       = config['mask_rate'],
#                    ).to(device)
#        
#    elif config['model_name'] == 'SELF':
#        model = SELF_supervised(dimensions_layers           = config['layers_dimensions'],
#                                edge_classes                = config['edge_classes'],
#                                dropout                     = config['dropout'],
#                                edge_pred_features          = config['edge_pred_features'],
#                                node_classes    = config['node_classes'],
#                                concat_hidden   = config['concat_hidden'],
#                                mask_rate       = config['mask_rate'],
#                    ).to(device)
#        
#    elif config['model_name'] == 'E2E':
#        model = E2E(node_classes        = config['node_classes'], 
#                    edge_classes        = config['edge_classes'], 
#                    dimensions_layers   = config['layers_dimensions'], 
#                    dropout             = config['dropout'], 
#                    edge_pred_features  = config['edge_pred_features'],
#                    drop_rate           = config['drop_edge'],
#                    discrete_pos        = config['relative_position_classification'],
#                    bounding_box        = config['bounding_box_classification']
#                    ).to(device)
#        
#    elif config['model_name'] == 'AUTOENCODER_MASK_MODF_SAGE':
#        model = AUTOENCODER_MASK_MODF_SAGE( dimensions_layers           = config['layers_dimensions'],
#                                            dropout                     = config['dropout'],
#                                            Tresh_distance              = config['Tresh_distance'],
#                                            node_classes                = config['node_classes'],
#                                            added_features              = config['added_features'],
#                                            concat_hidden               = config['concat_hidden'],
#                                            mask_rate                   = config['mask_rate'],
#                                            ).to(device)
#        
#    elif config['model_name'] == 'V2_AUTOENCODER_MASK_MODF_SAGE':
#        model = V2_AUTOENCODER_MASK_MODF_SAGE( dimensions_layers           = config['layers_dimensions'],
#                                               dropout                     = config['dropout'],
#                                               node_classes                = config['node_classes'],
#                                               concat_hidden               = config['concat_hidden'],
#                                               mask_rate                   = config['mask_rate'],
#                                               activation                  = activation
#                                               ).to(device)
#    
#    elif config['model_name'] == 'Contrastive_nodes':
#        raise NotImplementedError
#        #model = AUTOENCODER_MASK_MODF_SAGE_CONTRASTIVE(**config).to(device)
#    else:
#        raise NotImplementedError
#    
#    return model
#
#
