import dgl
import torch
from torch.nn import functional as F
import numpy as np
from utils import LoadConfig
from src.models.all_models import MaskedGat_contrastive, UNET_MaskedGat_contrastive_UNET
from src.training.utils import get_activation
from src.evaluation import *
from PIL import Image
from statistics import mean
from sklearn.metrics import accuracy_score
import argparse 
import xml.etree.ElementTree as ET
from src.data.doc2_graph.data.preprocessing import match_pred_w_gt

from src.data.doc2_graph.data.dataloader import Document2Graph
from src.data.doc2_graph.paths import TEST_SAMPLES
from src.data.doc2_graph.paths import PAU_TEST
from src.models import get_model_2

import pickle 
def pau_eval(args):
    test_data_path = args.test_data_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = args.checkpoint
    config = LoadConfig(args.run_name)

    model = get_model_2(config['model_name'], config)
    model.load_state_dict(torch.load(weights))
    model.eval()

    imgs = pickle.load(open('/data2/users/sbiswas/nil_biescas/PKL_Graphs/GT_PAU/pau_imgs_test.pkl', 'rb'))
    test_graph.imgs = imgs
    
    test_data = Document2Graph(name='PAU TEST', src_path=PAU_TEST, device = device, output_dir=TEST_SAMPLES)
    
    best_model = ''
    nodes_micro = []
    edges_f1 = []
    test_graph = dgl.batch(dgl.load_graphs(test_data_path)[0]).to(device)

    all_precisions = []
    all_recalls = []
    all_f1 = []

    # Load the model
    with torch.no_grad():

        n, e = model(test_graph, test_graph.ndata['feat'].to(device))
        auc = compute_auc_mc(e.to(device), test_graph.edata['label'].to(device))
        _, epreds = torch.max(F.softmax(e, dim=1), dim=1)
        _, npreds = torch.max(F.softmax(n, dim=1), dim=1)

        accuracy_nodes = accuracy_score(test_graph.ndata['label'].detach().cpu(), npreds.detach().cpu())
        accuracy, f1 = get_binary_accuracy_and_f1(epreds, test_graph.edata['label'])
        _, classes_f1 = get_binary_accuracy_and_f1(epreds, test_graph.edata['label'], per_class=True)
        edges_f1.append(classes_f1[1])

        macro, micro = get_f1(n, test_graph.ndata['label'].to(device))
        nodes_micro.append(micro)

        test_graph.edata['preds'] = epreds
        test_graph.ndata['preds'] = npreds
        t_f1 = 0
        t_precision = 0
        t_recall = 0
        no_table = 0
        tables = 0
            
        for g, graph in enumerate(dgl.unbatch(test_graph)):
            etargets = graph.edata['preds']
            ntargets = graph.ndata['preds']
            kvp_ids = etargets.nonzero().flatten().tolist()

            table_g = dgl.edge_subgraph(graph, torch.tensor(kvp_ids, dtype=torch.int32).to(device))
            table_nodes = table_g.ndata['geom']
            try:
                table_topleft, _ = torch.min(table_nodes, 0)
                table_bottomright, _ = torch.max(table_nodes, 0)
                table = torch.cat([table_topleft[:2], table_bottomright[2:]], 0)
            except:
                table = None
                
            img_path = test_data.paths[g]
            w, h = Image.open(img_path).size
            gt_path = img_path.split(".")[0]

            root = ET.parse(gt_path + '_gt.xml').getroot()
            regions = []
            for parent in root:
                if parent.tag.split("}")[1] == 'Page':
                    for child in parent:
                        # print(file_gt)
                        region_label = child[0].attrib['value']
                        if region_label != 'positions': continue
                        region_bbox = [int(child[1].attrib['points'].split(" ")[0].split(",")[0].split(".")[0]),
                                    int(child[1].attrib['points'].split(" ")[1].split(",")[1].split(".")[0]),
                                    int(child[1].attrib['points'].split(" ")[2].split(",")[0].split(".")[0]),
                                    int(child[1].attrib['points'].split(" ")[3].split(",")[1].split(".")[0])]
                        regions.append([region_label, region_bbox])

            table_regions = [region[1] for region in regions if region[0]=='positions']
            if table is None and len(table_regions) !=0: 
                t_f1 += 0
                t_precision += 0
                t_recall += 0
                tables += len(table_regions)
            elif table is None and len(table_regions) == 0:
                no_table -= 1
                continue
            elif table is not None and len(table_regions) ==0:
                t_f1 += 0
                t_precision += 0
                t_recall += 0
                no_table -= 1
            else:
                table = [[t[0]*w, t[1]*h, t[2]*w, t[3]*h] for t in [table.flatten().tolist()]][0]
                # d = match_pred_w_gt(torch.tensor(boxs_preds[idx]), torch.tensor(gt))
                d = match_pred_w_gt(torch.tensor(table).view(-1, 4), torch.tensor(table_regions).view(-1, 4), [])
                bbox_true_positive = len(d["pred2gt"])
                p = bbox_true_positive / (bbox_true_positive + len(d["false_positive"]))
                r = bbox_true_positive / (bbox_true_positive + len(d["false_negative"]))
                try:
                    t_f1 += (2 * p * r) / (p + r)
                except:
                    t_f1 += 0
                t_precision += p
                t_recall += r
                tables += len(table_regions)

                test_data.print_graph(num=g, node_labels = None, labels_ids=None, name=f'test_{g}', bidirect=False, regions=regions, preds=table)

        # test_data.print_graph(num=g, name=f'test_labels_{g}')
        t_recall = t_recall / (tables + no_table)
        t_precision = t_precision / (tables + no_table)
        t_f1 = (2 * t_precision * t_recall) / (t_precision + t_recall)
        all_precisions.append(t_precision)
        all_recalls.append(t_recall)
        all_f1.append(t_f1)

    ################* STEP 4: RESULTS ################
    #print("\n### RESULTS {} ###".format(m))
    print("Accuracy Nodes {:.4f}".format(accuracy_nodes))
    print("AUC {:.4f}".format(auc))
    print("Accuracy {:.4f}".format(accuracy))
    print("F1 Edges: Macro {:.4f} - Micro {:.4f}".format(f1[0], f1[1]))
    print("F1 Edges: None {:.4f} - Table {:.4f}".format(classes_f1[0], classes_f1[1]))
    print("F1 Nodes: Macro {:.4f} - Micro {:.4f}".format(macro, micro))

    print("\nTABLE DETECTION")
    print("PRECISION [MAX, MEAN, STD]:", max(all_precisions), mean(all_precisions), np.std(all_precisions))
    print("RECALLS [MAX, MEAN, STD]:", max(all_recalls), mean(all_recalls), np.std(all_recalls))
    print("F1s [MAX, MEAN, STD]:", max(all_f1), mean(all_f1), np.std(all_f1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation')
    
    parser.add_argument('--run-name', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--test_data_path', type=str, default=None)
    args = parser.parse_args()
    pau_eval(args = args)