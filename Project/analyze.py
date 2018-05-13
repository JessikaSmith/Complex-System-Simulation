# data source: http://konect.uni-koblenz.de/networks/cit-HepPh
IMG_PATH = 'Figures/'

import networkx as nx
import matplotlib.pyplot as plt
from processing import *

plt.interactive(True)


def create_graph(data):
    G = nx.DiGraph()
    for i, j in data.values:
        G.add_edge(i, j)
    return G


def plot_in_degree_distribution(graph):
    degs = {}
    for n in graph.nodes():
        deg = graph.in_degree(n)
        if deg not in degs:
            degs[deg] = 0
        degs[deg] += 1
    items = sorted(degs.items())
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([k for (k, v) in items], [v / len(graph.nodes()) for (k, v) in items], 'ro')
    ax.set_xscale('log')
    plt.xlabel('k')
    ax.set_yscale('log')
    plt.ylabel('P(k)')
    plt.title('In-Degree Distribution')
    fig.savefig(IMG_PATH + 'in_degree_distribution.png')


def plot_out_degree_distribution(graph):
    degs = {}
    for n in graph.nodes():
        deg = graph.out_degree(n)
        if deg not in degs:
            degs[deg] = 0
        degs[deg] += 1
    items = sorted(degs.items())
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([k for (k, v) in items], [v / len(graph.nodes()) for (k, v) in items], 'ro')
    ax.set_xscale('log')
    plt.xlabel('k')
    ax.set_yscale('log')
    plt.ylabel('P(k)')
    plt.title('Out-Degree Distribution')
    fig.savefig(IMG_PATH + 'out_degree_distribution.png')


def plot_distances(dist):
    dist_count = {}
    for i in range(len(dist.keys()) - 1):
        for j in range(i, len(dist.keys())):
            if i != j:
                if dist[i][j] not in dist_count:
                    dist_count[dist[i][j]] = 0
                dist_count[dist[i][j]] += 1
    items = sorted(dist_count.items())
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([v for (k, v) in items], [k for (k, v) in items], 'ro')
    plt.xlabel('Distance')
    plt.ylabel('Number of nodes')
    plt.title('Distance Distribution')
    fig.savefig(IMG_PATH + 'distance_distribution.png')

def plot_clustering_coeffs(clust):
    items = sorted(clust.items())
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([v for (k, v) in items], [k for (k, v) in items], 'ro')
    plt.ylabel('Node')
    plt.xlabel('$C_i$')
    plt.title('Clustering Coefficients Distribution')
    fig.savefig('clust_distribution.png')
    pass

def plot_avg_degree_connectivity(graph):
    nx.average_degree_connectivity(graph)


def get_assortativity_coeff(g):
    return nx.degree_pearson_correlation_coefficient(g)


def draw_graph(graph):
    nx.draw(graph, pos=nx.spring_layout(graph), with_labels=False)


def main():
    fname = 'Datasets/Cit-HepPh.txt'
    data = read_txt(fname)
    graph = create_graph(data)
    # draw_graph(graph)
    print(get_assortativity_coeff(graph))
    # plot_in_degree_distribution(graph)
    # plot_out_degree_distribution(graph)
    plot_distances(nx.shortest_path_length(graph))


if __name__ == '__main__':
    main()
