# data source: http://konect.uni-koblenz.de/networks/cit-HepPh

IMG_PATH = 'Figures/'

import networkx as nx
import matplotlib.pyplot as plt
from processing import *
import seaborn as sns
from random import choice
import math

plt.interactive(True)


def create_graph(data):
    G = nx.DiGraph()
    for i, j in data.values:
        G.add_edge(i, j)
    return G


def draw_heatmap(df, title):
    ax = plt.axes()
    sns.heatmap(df, ax=ax, cmap="Greens")
    ax.set_title(title)
    plt.show()


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
    fig.savefig(IMG_PATH + 'clust_distribution.png')


def plot_avg_degree_connectivity(connect):
    items = sorted(connect.items())
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([k for (k, v) in items], [v for (k, v) in items], 'ro')
    plt.xlabel('k')
    plt.ylabel('$<k_{nn}>$')
    plt.title('Assortativity Check')
    fig.savefig(IMG_PATH + 'assortativity.png')


def get_assortativity_coeff(g):
    return nx.degree_pearson_correlation_coefficient(g)


def estimating_power_law(data):
    values = data['Degree']
    cum_sum = 0
    for i in values:
        cum_sum += math.log(i)
    return 1 + len(values) * (cum_sum ** (-1))


def draw_graph(graph):
    nx.draw(graph, pos=nx.spring_layout(graph), with_labels=False)


def plot_betweeness_centrality(data):
    values = data['betweenesscentrality']
    values = [float(i) for i in values]
    counts = {}
    for n in values:
        if n not in counts:
            counts[n] = 0
        counts[n] += 1
    items = sorted(counts.items())
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([k for (k, v) in items], [v for (k, v) in items], 'ro')
    ax.set_xscale('log')
    plt.xlabel('Centrality')
    ax.set_yscale('log')
    plt.ylabel('Count')
    plt.title('Betweeness Centrality Distribution')
    fig.savefig(IMG_PATH + 'betweenesscentrality_distribution.png')


def giant_component_filtering(graph):
    giant = max(nx.strongly_connected_components(graph), key=len)
    H = graph.subgraph(giant)
    print(H)
    nx.write_edgelist(H, "Datasets/giant_component.edgelist")
    return H


def convert_to_csv(fname):
    df = pd.read_csv(fname, sep=" ", header=None)
    df = df.ix[:,[0,1]]
    df.to_csv("Datasets/giant_component_edges.csv", index=False, header=False)


def graph_comparison(graph, giant_comp):
    print(graph.number_of_nodes(), giant_comp.number_of_nodes())
    print(graph.number_of_edges(), giant_comp.number_of_edges())


def main():
    fname = 'Datasets/Datasets/giant_component_edges.csv'
    data = read_csv(fname)
    graph = create_graph(data)
    # component = giant_component_filtering(graph)
    # graph_comparison(graph, component)
    fname = 'Datasets/giant_component.edgelist'
    #convert_to_csv(fname)

    # draw_graph(graph)
    # print(get_assortativity_coeff(graph))
    # plot_in_degree_distribution(graph)
    # plot_out_degree_distribution(graph)
    # plot_distances(nx.shortest_path_length(graph))
    # plot_avg_degree_connectivity(nx.average_degree_connectivity(graph))
    # node_metrics = 'Datasets/node_metrics.csv'
    # metrics = read_csv(node_metrics)
    # plot_betweeness_centrality(metrics)
    # print(estimating_power_law(metrics))


if __name__ == '__main__':
    main()
