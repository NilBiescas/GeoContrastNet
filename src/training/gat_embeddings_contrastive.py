import torch
import sys
import dgl
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import random
import numpy as np
import json
from paths import HERE

sys.path.append("..") 

from ..models import get_model_2
from ..data.Dataset import (FUNSD_loader,
                            Unet_Dataset,
                            dataloaders_funsd, 
                            kmeans_graphs, 
                            edgesAggregation_kmeans_graphs)


from .utils import (get_model, 
                    compute_crossentropy_loss,
                    get_activation,
                    get_optimizer,
                    get_scheduler,
                    weighted_edges, 
                    region_encoding)

from ..evaluation import (SVM_classifier, 
                          kmeans_classifier, 
                          compute_auc_mc, 
                          get_f1,
                          conf_matrix,
                          plot_predictions,
                          get_accuracy,
                          get_binary_accuracy_and_f1)

from sklearn.metrics import recall_score, precision_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_funsd(model, optimizer, train_loader, config):
    model.train()
    nodes_predictions = []
    nodes_ground_truth = []
    total_train_loss = 0
    for train_graph in train_loader:
        
        train_graph = train_graph.to(device)
        feat = train_graph.ndata['feat'].to(device)
        labels = train_graph.ndata['label'].to(device)
        
        x_pred = model(train_graph, feat, mask_rate = config['mask_rate']).to(device)

        train_loss = compute_crossentropy_loss(x_pred, labels)
        #Reconstruction loss
        total_train_loss += train_loss
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        nodes_predictions.append(x_pred)
        nodes_ground_truth.append(labels)

    nodes_predictions = torch.cat(nodes_predictions, dim = 0)
    nodes_ground_truth = torch.cat(nodes_ground_truth)

    macro, micro, _, _ = get_f1(nodes_predictions, nodes_ground_truth)
    auc = compute_auc_mc(nodes_predictions, nodes_ground_truth)
    accuracy_train = get_accuracy(nodes_predictions, nodes_ground_truth)

    return total_train_loss, macro, auc, accuracy_train

def validation_funsd(model, val_loader):
    model.eval()
    nodes_predictions = []
    nodes_ground_truth = []
    total_validation_loss = 0
    with torch.no_grad():
        for val_graph in val_loader:
            
            val_graph = val_graph.to(device)
            feat = val_graph.ndata['feat'].to(device)
            x_true = val_graph.ndata['label'].to(device)

            x_pred = model(val_graph, feat, mask_rate = 0).to(device)

            val_n_loss = compute_crossentropy_loss(x_pred, x_true)     
            total_validation_loss += val_n_loss

            nodes_predictions.append(x_pred)
            nodes_ground_truth.append(x_true)

        nodes_predictions = torch.cat(nodes_predictions, dim = 0)
        nodes_ground_truth = torch.cat(nodes_ground_truth)

        
        macro_f1, micro_f1, precision_macro, recall_macro = get_f1(nodes_predictions, nodes_ground_truth)
        
        auc = compute_auc_mc(nodes_predictions, nodes_ground_truth)
        accuracy = get_accuracy(nodes_predictions, nodes_ground_truth)
    return total_validation_loss, macro_f1, auc, precision_macro, accuracy

def test_funsd(model, test_loader, config):
    model.eval()
    nodes_predictions = []
    nodes_ground_truth = []
    total_test_loss = 0

    with torch.no_grad():
        for test_graph in test_loader:
            
            test_graph = test_graph.to(device)
            feat = test_graph.ndata['feat'].to(device)
            x_true = test_graph.ndata['label'].to(device)

            x_pred = model(test_graph, feat, mask_rate = 0)

            val_n_loss = compute_crossentropy_loss(x_pred, x_true)     
            total_test_loss += val_n_loss

            nodes_predictions.append(x_pred.to('cpu'))
            nodes_ground_truth.append(x_true.to('cpu'))

    nodes_predictions = torch.cat(nodes_predictions, dim = 0)
    nodes_ground_truth = torch.cat(nodes_ground_truth)
    
    macro_f1, micro_f1, precision_macro, recall_macro = get_f1(nodes_predictions, nodes_ground_truth)
    auc = compute_auc_mc(nodes_predictions, nodes_ground_truth)
    accuracy = get_accuracy(nodes_predictions, nodes_ground_truth)
    # Compute confusion matrix

    pred = torch.argmax(nodes_predictions, dim=1)

    pred = pred.cpu().detach().numpy()
    nodes_ground_truth = nodes_ground_truth.cpu().detach().numpy()

    conf_matrix(nodes_ground_truth, pred, columns=["other", "question->answer"], path = config["output_dir"], title="Confusion Matrix - Test Set - MODEL")
    ################* STEP 4: RESULTS ################
    print("\n### BEST RESULTS ###")
    print("Precision macro: {:.4f}".format(precision_macro))
    print("Recall macro: {:.4f}".format(recall_macro))
    print("Accuracy: {:.4f}".format(accuracy))
    print("AUC: {:.4f}".format(auc))
    print("F1: Macro {:.4f} - Micro {:.4f}".format(macro_f1, micro_f1))
    
    data = {    
            "Test accuracy": accuracy,
            "Test f1_macro": macro_f1,
            "Test f1_micro": micro_f1,
            "Test precision macro": precision_macro,
            "Test recall macro": recall_macro,
            "Test auc": auc
    }
    
    print("Saving metrics.json")
    with open(config['output_dir'] / 'metrics_nodes.json', 'w') as f:
            json.dump(data, f)

    return total_test_loss / len(test_loader.dataset)

def test_evaluation(model, train_loader, test_loader, config):
    data_test = FUNSD_loader(train=False) #Loading test set graphs with Kmeans edges

    test_loss = test_funsd(model, test_loader)
    
    train_graph = dgl.batch(train_loader.dataset)
    test_graph = dgl.batch(test_loader.dataset)
    
    train_graph = train_graph.to(device)
    test_graph = test_graph.to(device)

    pred_kmeans = kmeans_classifier(model, train_graph, test_graph, config)
    pred_svm    = SVM_classifier(model, train_graph, test_graph, config)

    test_graph = dgl.batch(data_test.graphs)
    start, end = config['images']['start'], config['images']['end']
    plot_predictions(data_test, test_graph, pred_svm, path = config['output_svm'], start = start, end = end)
    plot_predictions(data_test, test_graph, pred_kmeans, path = config['output_kmeans'], start = start, end = end)
    return test_loss.item()

def contrastive_training_embeddings(config):
    # Load the learned embeddings

    train_graphs, _ = dgl.load_graphs(config['train_graphs'])
    test_graphs, _ = dgl.load_graphs(config['test_graphs'])
    
    train_graphs, validation_graphs, _, _ = train_test_split(train_graphs, torch.ones(len(train_graphs), 1), test_size=0.1, random_state=42)

    train_loader = torch.utils.data.DataLoader(train_graphs, batch_size=config['batch_size'], collate_fn = dgl.batch, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_graphs, batch_size=config['batch_size'], collate_fn = dgl.batch, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_graphs, batch_size=config['batch_size'], collate_fn = dgl.batch, shuffle=False)

    config['activation'] = F.relu
    try:
        if config['network']['checkpoint'] is None:
            model = get_model_2(config['model_name'], config).to(device)
            optimizer = get_optimizer(model, config)
            scheduler = get_scheduler(optimizer, config)


            total_train_loss = 0
            total_validation_loss = 0
            best_val_auc = 0
            for epoch in range(config['epochs']):

                train_loss, macro, auc, accuracy_train = train_funsd(model, optimizer, train_loader, config)
                val_tot_loss, val_macro, val_auc, precision, accuracy_val = validation_funsd(model, validation_loader)
                scheduler.step()

                total_train_loss += train_loss.item()
                total_validation_loss += val_tot_loss.item()

                if val_auc > best_val_auc:
                    torch.save(model.state_dict(), config['weights_dir'] / f"epoch_{epoch}.pth")
                    best_val_auc = val_auc
                    best_model = model
                
                print("Epoch {:05d} | TrainLoss {:.4f} | TrainF1-MACRO-node {:.4f} | TrainAUC-PR-node {:.4f} | ValLoss {:.4f} | ValF1-MACRO-node {:.4f} | ValAUC-PR-node {:.4f} |"
                        .format(epoch, train_loss.item(), macro, auc, val_tot_loss.item(), val_macro, val_auc))

            total_train_loss /= config['epochs']; total_validation_loss /= config['epochs']
            print("Train Loss: {:.4f} | Validation Loss: {:.4f}".format(total_train_loss, total_validation_loss), end='')
        else:
            print("Loading model from checkpoint")
            best_model = get_model_2(config['model_name'], config)
            best_model.load_state_dict(torch.load(config['network']['checkpoint']))
            best_model = best_model.to(device)
    except KeyboardInterrupt:
        pass
    test_loss = test_funsd(best_model, test_loader, config)

    print(" | Test Loss: {:.4f}".format(test_loss))
    return best_model


def train_edges(model, optimizer, train_loader, config):
    model.train()
    nodes_predictions = []
    nodes_ground_truth = []
    total_train_loss = 0
    for train_graph in train_loader:
        
        train_graph = train_graph.to(device)
        feat = train_graph.ndata['feat'].to(device)
        # Obtaining the key-value ground truth
        labels_edges = train_graph.edata['label'].to(device)
        
        _, edges_pred = model(train_graph, feat, mask_rate = config['mask_rate'])

        train_loss = compute_crossentropy_loss(edges_pred, labels_edges)
        #Reconstruction lossç
        total_train_loss += train_loss
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        nodes_predictions.append(edges_pred)
        nodes_ground_truth.append(labels_edges)

    nodes_predictions = torch.cat(nodes_predictions, dim = 0)
    nodes_ground_truth = torch.cat(nodes_ground_truth)

    macro, micro, _, _ = get_f1(nodes_predictions, nodes_ground_truth)
    auc = compute_auc_mc(nodes_predictions, nodes_ground_truth)
    accuracy_train = get_accuracy(nodes_predictions, nodes_ground_truth)

    return total_train_loss, macro, auc, accuracy_train

def validation_edges(model, val_loader):
    model.eval()
    nodes_predictions = []
    nodes_ground_truth = []
    total_validation_loss = 0
    with torch.no_grad():
        for val_graph in val_loader:
            
            val_graph = val_graph.to(device)
            feat = val_graph.ndata['feat']
            #
            x_true_edges = val_graph.edata['label']

            _, x_pred_edges = model(val_graph, feat, mask_rate = 0)

            val_n_loss = compute_crossentropy_loss(x_pred_edges, x_true_edges)     
            total_validation_loss += val_n_loss

            nodes_predictions.append(x_pred_edges)
            nodes_ground_truth.append(x_true_edges)

        nodes_predictions = torch.cat(nodes_predictions, dim = 0)
        nodes_ground_truth = torch.cat(nodes_ground_truth)

        
        macro, micro, precision_macro, recall_macro = get_f1(nodes_predictions, nodes_ground_truth)
        auc = compute_auc_mc(nodes_predictions, nodes_ground_truth)
        accuracy = get_accuracy(nodes_predictions, nodes_ground_truth)

    return total_validation_loss, macro, auc, precision_macro, accuracy

def test_edges(model, test_loader, config):
    model.eval()
    edges_predictions = []
    edges_ground_truth = []
    total_test_loss = 0

    with torch.no_grad():
        for test_graph in test_loader:
            
            test_graph = test_graph.to(device)
            feat = test_graph.ndata['feat']
            # Ground truth key-value
            x_true_edges = test_graph.edata['label']

            _, x_pred_edges = model(test_graph, feat, mask_rate = 0)

            val_n_loss = compute_crossentropy_loss(x_pred_edges, x_true_edges)     
            total_test_loss += val_n_loss

            edges_predictions.append(x_pred_edges.to('cpu'))
            edges_ground_truth.append(x_true_edges.to('cpu'))

    edges_predictions = torch.cat(edges_predictions, dim = 0)
    edges_ground_truth = torch.cat(edges_ground_truth)
    
    macro_f1, micro, precision, recall = get_f1(edges_predictions, edges_ground_truth)
    auc = compute_auc_mc(edges_predictions, edges_ground_truth)
    accuracy = get_accuracy(edges_predictions, edges_ground_truth)
    # Compute confusion matrix

    _, indices = torch.max(edges_predictions, dim=1)
    indices = indices.cpu().detach().numpy()
    edges_ground_truth = edges_ground_truth.cpu().detach().numpy()

    conf_matrix(edges_ground_truth, indices, config["output_dir"], title="Confusion Matrix - Test Set - MODEL", columns=['no edge', 'edge'])
    ################* STEP 4: RESULTS ################
    print("\n### BEST RESULTS ###")
    print("Precision edges macro: {:.4f}".format(precision))
    print("Recall edges macro: {:.4f}".format(recall))
    print("Accuracy edges: {:.4f}".format(accuracy))
    print("AUC edges: {:.4f}".format(auc))
    print("F1 edges: Macro {:.4f} - Micro {:.4f}".format(macro_f1, micro))
    
    data = {    
            "accuracy": accuracy,
            "f1": macro_f1,
            "precision": precision,
            "recall": recall
        }
    print("Saving metrics.json")
    with open(config['output_dir'] / 'metrics_edges.json', 'w') as f:
            json.dump(data, f)

    return total_test_loss / len(test_loader.dataset)

def predic_edges(config):
    train_graphs, _ = dgl.load_graphs(config['train_graphs'])
    test_graphs, _ = dgl.load_graphs(config['test_graphs'])
    
    train_graphs, validation_graphs, _, _ = train_test_split(train_graphs, torch.ones(len(train_graphs), 1), test_size=0.1, random_state=42)

    train_loader = torch.utils.data.DataLoader(train_graphs, batch_size=config['batch_size'], collate_fn = dgl.batch, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_graphs, batch_size=config['batch_size'], collate_fn = dgl.batch, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_graphs, batch_size=config['batch_size'], collate_fn = dgl.batch, shuffle=False)

    config['activation'] = F.relu

    # Pretrained weights
    model = get_model_2(config['model_name'], config)
    
    model.load_state_dict(torch.load(config['network']['checkpoint']), strict=False)
    model = model.to(device)

    if config.get('freeze_network', False):
        print("Freezing network")
        model.freeze_network(freeze = True)
    
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    total_train_loss = 0
    total_validation_loss = 0
    best_val_auc = 0
    for epoch in range(config['epochs']):

        train_loss, macro, auc, accuracy_train = train_edges(model, optimizer, train_loader, config)
        val_tot_loss, val_macro, val_auc, precision, accuracy_val = validation_edges(model, validation_loader)
        scheduler.step()

        total_train_loss += train_loss.item()
        total_validation_loss += val_tot_loss.item()

        if epoch == config.get('unfreeze_epoch', -1):
            print("Unfreezing network")
            model.freeze_network(freeze = False)

        if val_auc > best_val_auc:
            torch.save(model.state_dict(), config['weights_dir'] / f"epoch_{epoch}.pth")
            best_val_auc = val_auc
            best_model = model

                
        print("Epoch {:05d} | TrainLoss {:.4f} | TrainF1-MACRO-edge {:.4f} | TrainAUC-PR-edge {:.4f} | ValLoss {:.4f} | ValF1-MACRO-edge {:.4f} | ValAUC-PR-edge {:.4f} |"
               .format(epoch, train_loss.item(), macro, auc, val_tot_loss.item(), val_macro, val_auc))


    total_train_loss /= config['epochs']; total_validation_loss /= config['epochs']
    test_loss = test_edges(best_model, test_loader, config)

    print("Train Loss: {:.4f} | Validation Loss: {:.4f} | Test Loss: {:.4f}".format(total_train_loss, total_validation_loss, test_loss))
    return best_model




##############                   BOTH TASKS AT THE SAME TIME            #########
##############                   BOTH TASKS AT THE SAME TIME            #########
##############                   BOTH TASKS AT THE SAME TIME            #########
##############                   BOTH TASKS AT THE SAME TIME            #########
##############                   BOTH TASKS AT THE SAME TIME            ######### 


def train_edges_nodes(model, optimizer, train_loader, config):
    model.train()
    nodes_predictions = []
    nodes_ground_truth = []
    edges_predicitons = []
    edges_ground_truth = []

    total_train_loss = 0
    for train_graph, imgs in train_loader:
        
        train_graph = train_graph.to(device)
        feat = train_graph.ndata['feat']
        labels_nodes = train_graph.ndata['label']
        labels_edges = train_graph.edata['label']

        train_graph.imgs = imgs
        x_pred_nodes, x_pred_edges = model(train_graph, feat, mask_rate = config['mask_rate'])

        train_loss_nodes = compute_crossentropy_loss(x_pred_nodes, labels_nodes)
        train_loss_edges = compute_crossentropy_loss(x_pred_edges, labels_edges)
        train_loss = train_loss_nodes + train_loss_edges

        total_train_loss += train_loss
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        nodes_predictions.append(x_pred_nodes)
        nodes_ground_truth.append(labels_nodes)

        edges_predicitons.append(x_pred_edges)
        edges_ground_truth.append(labels_edges)

    nodes_predictions = torch.cat(nodes_predictions, dim = 0)
    nodes_ground_truth = torch.cat(nodes_ground_truth)

    edges_predicitons = torch.cat(edges_predicitons, dim = 0)
    edges_ground_truth = torch.cat(edges_ground_truth)

    def get_metrics(predictions, ground_truth):
        macro_f1, micro_f1, _, _ = get_f1(predictions, ground_truth)
        auc = compute_auc_mc(predictions, ground_truth)
        accuracy_train = get_accuracy(predictions, ground_truth)
        return macro_f1, auc, accuracy_train
    metrics_nodes = get_metrics(nodes_predictions, nodes_ground_truth)
    metrics_edges = get_metrics(edges_predicitons, edges_ground_truth)
    return total_train_loss, *metrics_nodes, *metrics_edges


def validation_edges_nodes(model, val_loader):
    model.eval()
    nodes_predictions = []
    nodes_ground_truth = []
    edges_predicitons = []
    edges_ground_truth = []

    total_validation_loss = 0
    with torch.no_grad():
        for val_graph, imgs in val_loader:
            
            val_graph = val_graph.to(device)
            feat = val_graph.ndata['feat']
            x_true_nodes = val_graph.ndata['label']
            x_true_edges = val_graph.edata['label']

            val_graph.imgs = imgs
            x_pred_nodes, x_pred_edges = model(val_graph, feat, mask_rate = 0)

            val_n_loss_nodes = compute_crossentropy_loss(x_pred_nodes, x_true_nodes)
            val_n_loss_edges = compute_crossentropy_loss(x_pred_edges, x_true_edges)
            val_n_loss = val_n_loss_nodes + val_n_loss_edges
            total_validation_loss += val_n_loss

            nodes_predictions.append(x_pred_nodes)
            nodes_ground_truth.append(x_true_nodes)
            edges_predicitons.append(x_pred_edges)
            edges_ground_truth.append(x_true_edges)

        nodes_predictions = torch.cat(nodes_predictions, dim = 0)
        nodes_ground_truth = torch.cat(nodes_ground_truth)

        edges_predicitons = torch.cat(edges_predicitons, dim = 0)
        edges_ground_truth = torch.cat(edges_ground_truth)

        def get_metrics(predictions, ground_truth, per_class = False):
            classes_f1 = None
            if per_class:
                edges_predictions = torch.argmax(predictions, dim=1)
                _, classes_f1 = get_binary_accuracy_and_f1(edges_predictions, edges_ground_truth, per_class=True)

            macro_f1, micro_f1, precision, recall = get_f1(predictions, ground_truth)
            auc = compute_auc_mc(predictions, ground_truth)
            accuracy = get_accuracy(predictions, ground_truth)
            return macro_f1, auc, precision, accuracy, micro_f1, classes_f1
        
        
        return total_validation_loss, *get_metrics(nodes_predictions, nodes_ground_truth), *get_metrics(edges_predicitons, edges_ground_truth, per_class=True)


def test_edges_report(edges_predictions, edges_ground_truth, config):

    macro, micro, precision_macro, recall_macro = get_f1(edges_predictions, edges_ground_truth)
    accuracy = get_accuracy(edges_predictions, edges_ground_truth)
    auc = compute_auc_mc(edges_predictions, edges_ground_truth)
    
    edges_predictions = torch.argmax(edges_predictions, dim=1)

    accuracy, f1 = get_binary_accuracy_and_f1(edges_predictions, edges_ground_truth)
    _, classes_f1 = get_binary_accuracy_and_f1(edges_predictions, edges_ground_truth, per_class=True)

    precision = precision_score(edges_ground_truth, edges_predictions)
    recall = recall_score(edges_ground_truth, edges_predictions)

    conf_matrix(edges_ground_truth, edges_predictions, config["output_dir"], title="Test Set - Edges", columns =  ['no edge', 'edge'])
    data = {
        "precision_macro": precision_macro,
        "precision": precision,
        "recall_macro": recall_macro,
        "recall": recall,
        "accuracy": accuracy,
        "macro_f1": macro,
        "micro_f1": micro,
        "f1": f1,
        "None_F1": classes_f1[0],
        "Key_Value_F1": classes_f1[1],
        "AUC_PR": auc,
    }
    print("Saving metrics.json")
    with open(config['output_dir'] / 'metrics_edges_doc2graph.json', 'w') as f:
            json.dump(data, f)

def test_edges_nodes(model, test_loader, config):
    model.eval()
    nodes_predictions = []
    nodes_ground_truth = []
    edges_predicitons = []
    edges_ground_truth = []
    total_test_loss = 0

    with torch.no_grad():
        for test_graph, imgs in test_loader:
            
            test_graph = test_graph.to(device)
            feat = test_graph.ndata['feat']
            x_true_nodes = test_graph.ndata['label']
            x_true_edges = test_graph.edata['label']

            test_graph.imgs = imgs
            x_pred_nodes, x_pred_edges = model(test_graph, feat, mask_rate = 0)

            val_n_loss_nodes = compute_crossentropy_loss(x_pred_nodes, x_true_nodes)
            val_n_loss_edges = compute_crossentropy_loss(x_pred_edges, x_true_edges)
            val_n_loss = val_n_loss_nodes + val_n_loss_edges
            total_test_loss += val_n_loss

            nodes_predictions.append(x_pred_nodes.to('cpu'))
            nodes_ground_truth.append(x_true_nodes.to('cpu'))
            edges_predicitons.append(x_pred_edges.to('cpu'))
            edges_ground_truth.append(x_true_edges.to('cpu'))


    nodes_predictions = torch.cat(nodes_predictions, dim = 0)
    nodes_ground_truth = torch.cat(nodes_ground_truth)
    edges_predicitons = torch.cat(edges_predicitons, dim = 0)
    edges_ground_truth = torch.cat(edges_ground_truth)

    def get_metrics(predictions, ground_truth, columns, title = "Confusion Matrix - Test Set - MODEL", title_json = "metrics_nodes.json"):
        macro_f1, micro, precision, recall = get_f1(predictions, ground_truth)
        auc = compute_auc_mc(predictions, ground_truth)
        accuracy = get_accuracy(predictions, ground_truth)
        _, indices = torch.max(predictions, dim=1)
        indices = indices.cpu().detach().numpy()
        nodes_ground_truth = ground_truth.cpu().detach().numpy()
        #conf_matrix(nodes_ground_truth, indices, config["output_dir"], title=title, columns = columns)
        ################* STEP 4: RESULTS ################
        print("\n### BEST RESULTS ###")
        print("Precision nodes macro: {:.4f}".format(precision))
        print("Recall nodes macro: {:.4f}".format(recall))
        print("Accuracy nodes: {:.4f}".format(accuracy))
        print("AUC Nodes: {:.4f}".format(auc))
        print("F1 Nodes: Macro {:.4f} - Micro {:.4f}".format(macro_f1, micro))
        data = {
            "precision macro": precision,
            "macro f1": macro_f1,
            "micro f1": micro,
            "AUC": auc,
            "accuracy": accuracy,
            "f1": macro_f1,
            "recall macro": recall
        }
        with open(config['output_dir'] / title_json, 'w') as f:
            json.dump(data, f)
        return macro_f1, auc, precision, accuracy
   
    get_metrics(nodes_predictions, nodes_ground_truth, columns = ['answer', 'header', 'other', 'question', "a", "b"], 
                title = "Confusion Matrix - Test Set - Nodes", title_json = "metrics_nodes.json")
    test_edges_report(edges_predicitons, edges_ground_truth, config=config)
    
    return total_test_loss / len(test_loader.dataset)

def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, labels

def contrastiv_node_edge_training(config):
    # Load the learned embeddings
    train_graphs, _      = dgl.load_graphs((HERE / config['train_graphs']).__str__())
    validation_graphs, _ = dgl.load_graphs((HERE / config['validation_graphs']).__str__())
    test_graphs, _       = dgl.load_graphs((HERE / config['test_graphs']).__str__())
    
    dataset_train = Unet_Dataset(train_graphs, (HERE /config['train_img'].__str__()))
    dataset_val   = Unet_Dataset(validation_graphs, (HERE /config['val_img'].__str__()))
    dataset_test = Unet_Dataset(test_graphs, (HERE /config['test_img']).__str__())
#
    train_loader        = torch.utils.data.DataLoader(dataset_train, batch_size=config['batch_size'], collate_fn = collate, shuffle=True)
    validation_loader   = torch.utils.data.DataLoader(dataset_val, batch_size=config['batch_size'],  collate_fn = collate, shuffle=False)
    test_loader         = torch.utils.data.DataLoader(dataset_test, batch_size=config['batch_size'], collate_fn = collate, shuffle=False)

    #train_loader        = torch.utils.data.DataLoader(train_graphs, batch_size=config['batch_size'], collate_fn = dgl.batch, shuffle=True)
    #validation_loader   = torch.utils.data.DataLoader(validation_graphs, batch_size=config['batch_size'], collate_fn = dgl.batch, shuffle=False)
    #test_loader         = torch.utils.data.DataLoader(test_graphs, batch_size=config['batch_size'], collate_fn = dgl.batch, shuffle=False)

    config['activation'] = get_activation(config['activation'])
    try:
        if config['network']['checkpoint'] is None:
            model = get_model_2(config['model_name'], config).to(device)
            optimizer = get_optimizer(model, config)
            scheduler = get_scheduler(optimizer, config)

            total_train_loss = 0
            total_validation_loss = 0
            best_val_f1_micro_key_value = -1
            for epoch in range(config['epochs']):

                train_loss, macro_f1, auc, accuracy_train, _, _, _ = train_edges_nodes(model, optimizer, train_loader, config)
                val_tot_loss, val_macro, val_auc, precision, accuracy_val, micro_f1_nodes, _, macro_f1_edges, auc_edges, precision_edges, accuracy_edges, f1_micro_edges, classes_f1 = validation_edges_nodes(model, validation_loader)
                scheduler.step()

                total_train_loss += train_loss.item()
                total_validation_loss += val_tot_loss.item()
                
                if classes_f1[1] > best_val_f1_micro_key_value:
                    torch.save(model.state_dict(), config['weights_dir'] / f"epoch_{epoch}.pth")
                    best_val_f1_micro_key_value = classes_f1[1]
                    best_model = model

                print("Epoch {:05d} | TrainLoss {:.4f} | TrainF1-MACRO-node {:.4f} | Val-f1-key-val {:.4f} | ValLoss {:.4f} | ValF1-MACRO-node {:.4f} | ValAUC-PR-node {:.4f} |"
                        .format(epoch, train_loss.item(), macro_f1, classes_f1[1], val_tot_loss.item(), val_macro, val_auc))

            total_train_loss /= config['epochs']; total_validation_loss /= config['epochs']
            print("Train Loss: {:.4f} | Validation Loss: {:.4f}".format(total_train_loss, total_validation_loss), end='')
        else:
            print("Loading model from checkpoint")
            best_model = get_model_2(config['model_name'], config)
            state_dict = torch.load(config['network']['checkpoint'])
            #state_dict["gat_layers.0.bias"] = state_dict.pop("gat_layers.0.res_fc.bias")
            #state_dict["gat_layers.0.res_fc.bias"] = state_dict.pop("gat_layers.0.bias")
            best_model.load_state_dict(state_dict)
            best_model = best_model.to(device)
    except KeyboardInterrupt:
        pass
    #test_loss = test_evaluation(best_model, train_loader, config)
    test_loss = test_edges_nodes(best_model, test_loader, config)

    print(" | Test Loss: {:.4f}".format(test_loss))
    return best_model




def obtain_table_doc2graph(config):
    test_graphs, _ = dgl.load_graphs(config['test_graphs'])
    
    test_loader = torch.utils.data.DataLoader(test_graphs, batch_size=config['batch_size'], collate_fn = dgl.batch, shuffle=False)

    config['activation'] = F.relu

    # Pretrained weights
    model = get_model_2(config['model_name'], config)
    
    model.load_state_dict(torch.load(config['network']['checkpoint']))
    model = model.to(device)

    edges_predictions = []
    edges_ground_truth = []
    with torch.no_grad():
        for test_graph in test_loader:
            
            test_graph = test_graph.to(device)
            feat = test_graph.ndata['feat']
            x_true_edges = test_graph.edata['label']

            _, x_pred_edges = model(test_graph, feat, mask_rate = 0)

            # Predicted edges
            edges_predictions.append(x_pred_edges.to('cpu'))
            # Ground Truth
            edges_ground_truth.append(x_true_edges.to('cpu'))
    
    edges_predictions = torch.cat(edges_predictions, dim = 0)
    edges_ground_truth = torch.cat(edges_ground_truth)

    macro, micro, precision_macro, recall_macro = get_f1(edges_predictions, edges_ground_truth)
    accuracy = get_accuracy(edges_predictions, edges_ground_truth)
    auc = compute_auc_mc(edges_predictions, edges_ground_truth)
    

    edges_predictions = torch.argmax(edges_predictions, dim=1)

    accuracy, f1 = get_binary_accuracy_and_f1(edges_predictions, edges_ground_truth)
    _, classes_f1 = get_binary_accuracy_and_f1(edges_predictions, edges_ground_truth, per_class=True)

    precision = precision_score(edges_ground_truth, edges_predictions)
    recall = recall_score(edges_ground_truth, edges_predictions)

    data = {
        "precision_macro": precision_macro,
        "precision": precision,
        "recall_macro": recall_macro,
        "recall": recall,
        "accuracy": accuracy,
        "macro_f1": macro,
        "micro_f1": micro,
        "f1": f1,
        "None_F1": classes_f1[0],
        "Key_Value_F1": classes_f1[1],
        "AUC_PR": auc,
    }
    print("Saving metrics.json")
    with open(config['output_dir'] / 'metrics_edges_doc2graph.json', 'w') as f:
            json.dump(data, f)
    
    return model
