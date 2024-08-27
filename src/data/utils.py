from math import log
import torch
import os

def createDir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return


def paragraph2graph_edges(geom1, geom2):

    xmin, ymin, xmax, ymax = geom1[:, 0], geom1[:, 1], geom1[:, 2], geom1[:, 3]
    xmin2, ymin2, xmax2, ymax2 = geom2[:, 0], geom2[:, 1], geom2[:, 2], geom2[:, 3]

    xS = (xmin + xmax) / 2
    yS = (ymin + ymax) / 2
    wS = xmax - xmin
    hS = ymax - ymin

    xO = (xmin2 + xmax2) / 2
    yO = (ymin2 + ymax2) / 2
    wO = xmax2 - xmin2
    hO = ymax2 - ymin2

    txSO = (xS - xO) / wS
    tySO = (yS - yO) / hS
    # Relu and adding a small value to avoid log(0) or log(-1)
    wS_wO = torch.relu(wS / wO) + 1e-6
    hS_hO = torch.relu(hS / hO) + 1e-6

    twSO = torch.log(wS_wO)
    thSO = torch.log(hS_hO)
    txOS = (xO - xS) / wO
    tyOS = (yO - yS) / hO

    res = torch.cat([txSO, tySO, twSO, thSO, txOS, tyOS], dim=0)
    res = res.view(txSO.shape[0], -1)
    return res

def concat_paragraph2graph_edges(edges):
    """
    This function is called using dgl.apply_edges(concat_paragraph2graph_edges)
    The function calculates the edges features for the graph. This is from the paper: PARAGRAPH2GRAPH: A GNN-BASED FRAMEWORK FOR LAYOUT PARAGRAPH ANALYSIS
    """
    SD = paragraph2graph_edges(edges.src['geom'], edges.dst['geom'])

    R = lambda src, dst: (torch.min(src[:, 0], dst[:, 0]), torch.min(src[:, 1], dst[:, 1]), torch.max(src[:, 2], dst[:, 2]), torch.max(src[:, 3], dst[:, 3]))

    r = R(edges.src['geom'], edges.dst['geom'])
    r = torch.cat(r, dim = 0).view(r[0].shape[0], -1)

    SR = paragraph2graph_edges(edges.src['geom'], r)
    OR = paragraph2graph_edges(edges.dst['geom'], r)

    res = torch.cat([SD, SR, OR], dim=1)
    res[res == torch.inf] = 0
    res[res == -torch.inf]= 0
    res[torch.isnan(res)] = 0
    return {'F_edge': res}