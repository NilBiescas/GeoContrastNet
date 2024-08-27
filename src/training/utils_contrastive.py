import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_pairwise_distances(embeddings, metric='cosine'):
    metric = {'CosineSimilarity': 'cosine', 'LpDistance': 'euclidean'}[metric]
    pairwise_distances = pdist(embeddings, metric)
    distance_matrix = squareform(pairwise_distances)
    return distance_matrix

def calculate_within_class_distances(labels, class_label, distance_matrix):
    class_indices = np.where(labels == class_label)[0]
    within_class_distances = distance_matrix[class_indices][:, class_indices]
    return within_class_distances

def calculate_between_class_distances(labels, class_label, distance_matrix):
    class_indices = np.where(labels == class_label)[0]
    between_class_distances = []

    for other_class_label in np.unique(labels):
        if other_class_label != class_label:
            other_class_indices = np.where(labels == other_class_label)[0]
            between_class_distances.append(distance_matrix[class_indices][:, other_class_indices])

    return between_class_distances

def plot_distance_histograms(distances, labels, class_indice, dir_path):
    within_class_dis = calculate_within_class_distances(labels, class_indice, distances)
    between_class_dis = calculate_between_class_distances(labels, class_indice, distances)

    triangular_superior = within_class_dis[np.triu_indices(len(within_class_dis), k=1)]
    plt.figure(figsize=(8, 6))
    #plt.hist(triangular_superior, bins=30, alpha=1, color='black', label=f'Within {class_indice} Distances', density=True, stacked = True)
    classes = ["answer", "header", "other", "question"]
    counts, bins = np.histogram(triangular_superior, bins=30, density=False)
    counts = counts / np.sum(counts)
    plt.stairs(counts, bins, fill=True, alpha=0.5, color='black', label=f'Distances within the class {classes[class_indice]}')

    colors = ['red', 'green', 'yellow']
    corresponding = np.unique(labels)[np.unique(labels) != class_indice]
    for idx, i in enumerate(corresponding):
        between_class_dis_flatten = between_class_dis[idx].flatten()
        # Drop na
        between_class_dis_flatten = between_class_dis_flatten[~np.isnan(between_class_dis_flatten)]
        counts, bins = np.histogram(between_class_dis_flatten, bins=30, density=False)
        counts = counts / np.sum(counts)
        plt.stairs(counts, bins, fill=True, alpha=0.5, color=colors[idx], label=f'Distances between {classes[class_indice]} and {classes[i]}')
    
    plt.title(f'Distance Histogram for Class {class_indice}')
    plt.xlabel('Distance')
    plt.ylabel('Percentage')
    plt.legend()
    plt.savefig(dir_path / f'histogram_class_{class_indice}.png')
    plt.close()
  

def visualize_embeddings(embeddings, labels, dir_path):
    classes = ["answer", "header", "other", "question"]
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(embeddings)
    sns.scatterplot(x=tsne_results[:,0], y=tsne_results[:,1], hue=labels)
    plt.savefig(dir_path / 'tsne.png')
    plt.close()

def create_plots(embeddings, labels, dir_path, config):
    visualize_embeddings(embeddings, labels, dir_path)
    distance_matrix = calculate_pairwise_distances(embeddings, metric=config['contrastive_learning']['distance_metric'])
    plot_distance_histograms(distance_matrix, labels, class_indice = 0, dir_path = dir_path)
    plot_distance_histograms(distance_matrix, labels, class_indice = 1, dir_path = dir_path)
    plot_distance_histograms(distance_matrix, labels, class_indice = 2, dir_path = dir_path)
    plot_distance_histograms(distance_matrix, labels, class_indice = 3, dir_path = dir_path)

def obtain_embeddings(train_loader, test_loader, model, train = True, test = True):
    if train and test:
        # Train and test embeddings
        embeddings_train, labels_train = model.get_embeddigns(train_loader)
        embeddings_test, labels_test  = model.get_embeddigns(test_loader)
        embeddings = np.concatenate((embeddings_train, embeddings_test))
        labels = np.concatenate((labels_train, labels_test))
        return embeddings, labels
    elif train:
        # Train embeddings
        embeddings, labels = model.get_embeddigns(train_loader)
        return embeddings, labels
    elif test:
        # Test emebddings
        embeddings, labels  = model.get_embeddigns(test_loader)
        return embeddings, labels
    else:
        raise ValueError("You must choose at least one dataset to obtain embeddings from")


