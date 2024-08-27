import torch
import PIL
import json
import pandas as pd
import seaborn as sn
from typing import Tuple
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, confusion_matrix)
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_auc_mc(scores, labels):
    scores = scores.detach().cpu().numpy()
    labels = F.one_hot(labels).cpu().numpy()
    # return roc_auc_score(labels, scores)
    scores = np.array(scores)

    return average_precision_score(labels, scores)


def get_binary_accuracy_and_f1(classes, labels : torch.Tensor, per_class = False) -> Tuple[float, list]:

    correct = torch.sum(classes.flatten() == labels)
    accuracy = correct.item() * 1.0 / len(labels)
    classes = classes.detach().cpu().numpy()
    labels = labels.cpu().numpy()

    if not per_class:
        f1 = f1_score(labels, classes, average='macro'), f1_score(labels, classes, average='micro')
    else:
        f1 = precision_recall_fscore_support(labels, classes, average=None)[2].tolist()
    
    return accuracy, f1

def get_accuracy(logits : torch.Tensor, labels : torch.Tensor) -> float:
    _, indices = torch.max(logits, dim=1)
    indices = indices.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    return accuracy_score(labels, indices)

def get_f1(logits : torch.Tensor, labels : torch.Tensor, per_class = False) -> tuple:
    """Returns Macro and Micro F1-score for given logits / labels.

    Args:
        logits (torch.Tensor): model prediction logits
        labels (torch.Tensor): target labels

    Returns:
        tuple: macro-f1 and micro-f1
    """
    _, indices = torch.max(logits, dim=1)
    indices = indices.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    if not per_class:
        return f1_score(labels, indices, average='macro'), f1_score(labels, indices, average='micro'), precision_score(labels, indices, average='macro', zero_division=0.0), recall_score(labels, indices, average='macro', zero_division=0.0)                                                                                                      
    else:
        return precision_recall_fscore_support(labels, indices, average=None)[2].tolist()


def extract_embeddings(graph, model):
    with torch.no_grad():
        model.eval()
        h = graph.ndata['Geometric'].to(device)
        #h = graph.ndata['Geometric'].to(device)
        for layer in model.encoder:
            h = layer(graph, h)

        embeddings = h.cpu().detach().numpy()
        labels = graph.ndata['label'].cpu().detach().numpy()
        return embeddings, labels


def conf_matrix(y_true, y_pred, path, title, columns = ['answer', 'header', 'other', 'question']):
    data = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(data, columns=np.array(columns), index = np.array(columns))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize = (10,7))
    plt.title(title)
    sn.set(font_scale=1.4) #for label size
    sn.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 16}, fmt='d') # font size
    plt.savefig(path / f'{title}.png')
    #plt.show()
    plt.close()

def kmeans_classifier(model, train_graph, test_graph, config):
    from sklearn.cluster import KMeans
    with torch.no_grad():

        embeddings_train, labels_train = model.extract_embeddings(train_graph)
        embeddings_test, labels_test   = model.extract_embeddings(test_graph)

        kmeans = KMeans(n_clusters=4, random_state=0).fit(embeddings_train)
        groups = kmeans.labels_

        group1_idx = np.where(groups == 0)[0]
        group2_idx = np.where(groups == 1)[0]
        group3_idx = np.where(groups == 2)[0]
        group4_idx = np.where(groups == 3)[0]

        group1_labels, group2_labels, group3_labels, group4_labels = labels_train[group1_idx], labels_train[group2_idx], labels_train[group3_idx], labels_train[group4_idx]

        group1_labels, group2_labels, group3_labels, group4_labels = np.bincount(group1_labels), np.bincount(group2_labels), np.bincount(group3_labels), np.bincount(group4_labels)

        if not all(map(lambda a: len(a) != 0, [group1_labels, group2_labels, group3_labels, group4_labels])):
            return
        
        mapping = {'0': np.argmax(group1_labels), '1': np.argmax(group2_labels), '2': np.argmax(group3_labels), '3': np.argmax(group4_labels)}
        print(mapping)

        pred = kmeans.predict(embeddings_test)
        pred = [mapping[str(i)] for i in pred]

        accuracy = accuracy_score(labels_test, pred)
        f1 = f1_score(labels_test, pred, average='macro')
        precision = precision_score(labels_test, pred, average='macro')
        recall = recall_score(labels_test, pred, average='macro')
        
        accuracy *= 100; f1 *= 100; precision *= 100; recall *= 100 #To percentage
        accuracy, f1, precision, recall = int(accuracy), int(f1), int(precision), int(recall) #To int

        data = {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall
        }
        with open(config['json_kmeans'] / 'metrics_kmeans.json', 'w') as f:
            json.dump(data, f)

        print("Accuracy kmeans: {:.4f} | F1 Score kmeans: {:.4f} | Precision kmeans: {:.4f} | Recall kmeans: {:.4f}".format(accuracy, f1, precision, recall))
        conf_matrix(labels_test, pred, config['output_kmeans'], config['run_name'] + "_kmeans")
        return pred


def plot_predictions(data, graphs, pred, path, start = 0, end = 5):
    import matplotlib.patches as mpatches
    for num_graph in range(start, end):
        start = torch.sum(graphs.batch_num_nodes()[:num_graph])
        end   = torch.sum(graphs.batch_num_nodes()[:num_graph+1])
        pred_features = pred[start:end]

        if len(pred_features) == 0:
            print(start, end)
            continue
        plt.imshow(data.print_graph(num = num_graph, node_labels = pred_features))

        color_dict = {'question': (0.59, 0.29, 0.0), 'answer': (0.0, 0.39, 0.0), 'other':(0.5, 0.5, 0.5), 'header': (1.0, 0.0, 1.0)}
        patches = [mpatches.Patch(color=color_dict[label], label=label) for label in color_dict.keys()]
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper right', borderaxespad=0, fontsize='small', prop =  {'size': 6})
        plt.savefig(path / f"Image_{num_graph}.png", dpi = 600)

def SVM_classifier(model, train_graph, test_graph, config):
    from sklearn.svm import SVC

    with torch.no_grad():

        embeddings_train, labels_train = model.extract_embeddings(train_graph)
        embeddings_test, labels_test   = model.extract_embeddings(test_graph)

        clf = SVC(kernel='rbf', C=1, gamma='auto')
        clf.fit(embeddings_train, labels_train)
        pred = clf.predict(embeddings_test)
        accuracy = accuracy_score(labels_test, pred)
        f1 = f1_score(labels_test, pred, average='macro')
        precision = precision_score(labels_test, pred, average='macro')
        recall = recall_score(labels_test, pred, average='macro')

        # Percentatge
        accuracy *= 100; f1 *= 100; precision *= 100; recall *= 100
        accuracy, f1, precision, recall = int(accuracy), int(f1), int(precision), int(recall)

        data = {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall
        }
        with open(config['json_svm'] / 'metrics_svm.json', 'w') as f:
            json.dump(data, f)

        print("Accuracy SVM: {:.4f} | F1 Score SVM: {:.4f} | Precision SVM: {:.4f} | Recall SVM: {:.4f}".format(accuracy, f1, precision, recall))
        conf_matrix(labels_test, pred, config['output_svm'], config['run_name'] + "_SVM")
        return pred