# data source: http://konect.uni-koblenz.de/networks/cit-HepPh

IMG_PATH = 'Figures/'

import plotly

plotly.tools.set_credentials_file(username="MKhod", api_key='9eUEqRimNHQfPFX6r8HM')

import plotly.graph_objs as go
from plotly.offline import plot
from plotly.plotly import image

import networkx as nx
import matplotlib.pyplot as plt
from processing import *
import seaborn as sns
from random import choice
import math
import pandas as pd
import numpy as np

plt.interactive(True)


# https://stackoverflow.com/questions/16489655/plotting-log-binned-network-degree-distributions

def create_graph(data):
    G = nx.DiGraph()
    for i, j in data.values:
        G.add_edge(i, j)
    return G


def drop_zeros(a_list):
    return [i for i in a_list if i > 0]


def print_distance_information(graph):
    print("diameter", nx.diameter(graph))
    print("radius", nx.radius(graph))


def components(graph):
    print("Number of strongly connected components", nx.number_strongly_connected_components(graph))
    print("Number of weakly connected components", nx.number_weakly_connected_components(graph))

    condensation = nx.condensation(graph)
    nx.write_edgelist(condensation, "Datasets/condenced_graph.edgelist")


def log_binning(counter_dict, bin_count=35):
    max_x = math.log10(max(counter_dict.keys()))
    max_y = math.log10(max(counter_dict.values()))
    max_base = max([max_x, max_y])

    min_x = math.log10(min(drop_zeros(counter_dict.keys())))

    bins = np.logspace(min_x, max_base, num=bin_count)

    bin_means_y = (np.histogram(list(counter_dict.keys()), bins, weights=list(counter_dict.values()))[0] /
                   np.histogram(list(counter_dict.keys()), bins)[0])
    bin_means_x = (np.histogram(list(counter_dict.keys()), bins, weights=list(counter_dict.keys()))[0] /
                   np.histogram(list(counter_dict.keys()), bins)[0])

    return bin_means_x, bin_means_y


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


def plot_avg_binned_degree_connectivity(connect):
    items = sorted(connect.items())
    x, y = log_binning(connect, 60)
    trace = go.Scatter(
        x=np.array(x),
        y=np.array(y),
        mode='markers',
        line=dict(
            color=('rgb(205, 12, 24)')
        )
    )
    layout = dict(title="Assortativity",
                  xaxis=dict(title='k'),
                  yaxis=dict(title='$<k_{nn}>$')
                  )
    plot_data = [trace]
    fig = go.Figure(data=plot_data, layout=layout)
    image.save_as(fig, filename=IMG_PATH + 'assortativity_binned.jpeg')


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
    # ax.set_ylim(ymin=-0.001)
    ax.set_xscale('log')
    plt.xlabel('Centrality')
    ax.set_yscale('log')
    plt.ylabel('Count')
    plt.title('Betweeness Centrality Distribution')
    fig.savefig(IMG_PATH + 'betweenesscentrality_distribution2.png')


def plot_binned_betweeness_centrality(data):
    values = data['betweenesscentrality']
    values = [float(i) for i in values]
    counts = {}
    for n in values:
        if n not in counts:
            counts[n] = 0
        counts[n] += 1
    x, y = log_binning(counts, 60)
    trace = go.Scatter(
        x=np.array(x),
        y=np.array(y),
        mode='markers',
        line=dict(
            color=('rgb(205, 12, 24)')
        )
    )
    layout = dict(title="Beetweeness Distribution",
                  xaxis=dict(title='Betweenness centrality',
                             type='log'),
                  yaxis=dict(title='Count',
                             type='log')
                  )
    plot_data = [trace]
    fig = go.Figure(data=plot_data, layout=layout)
    image.save_as(fig, filename=IMG_PATH + 'betweenesscentrality_binned2.jpeg')


def plot_distance_distribution(graph):
    count = {}
    for n in list(graph):
        tmp_dict = nx.shortest_path_length(graph, source=n)
        tmp_res = 0
        for j in list(graph):
            if n != j:
                try:
                    tmp_res += tmp_dict[j]
                except KeyError:
                    continue
        count[n] = tmp_res / (len(list(graph)) - 1)
    counts = {}
    for n in count:
        if n not in counts:
            counts[n] = 0
        counts[n] += 1
    x, y = log_binning(count, 150)
    trace = go.Scatter(
        x=np.array(y),
        y=np.array(x),
        mode='markers',
        line=dict(
            color=('rgb(205, 12, 24)')
        )
    )
    layout = dict(title="Distance Distribution",
                  xaxis=dict(title='Average distance',
                             type='log'),
                  yaxis=dict(title='Count',
                             type='log')
                  )
    plot_data = [trace]
    fig = go.Figure(data=plot_data, layout=layout)
    image.save_as(fig, filename=IMG_PATH + 'distance_distribution.jpeg')


def giant_component_filtering(graph):
    giant = max(nx.strongly_connected_components(graph), key=len)
    H = graph.subgraph(giant)
    print(H)
    nx.write_edgelist(H, "Datasets/giant_component.edgelist")
    return H


def read_csv(fname):
    df = pd.read_csv(fname)
    return df


def convert_to_csv(fname):
    df = pd.read_csv(fname, sep=" ", header=None)
    df = df.ix[:, [0, 1]]
    df.to_csv("Datasets/condenced_edges.csv", index=False, header=False)


def graph_comparison(graph, giant_comp):
    print(graph.number_of_nodes(), giant_comp.number_of_nodes())
    print(graph.number_of_edges(), giant_comp.number_of_edges())


def main():
    fname = 'Datasets/Cit-HepPh.csv'
    data = read_csv(fname)
    graph = create_graph(data)
    plot_distance_distribution(graph)
    # print_distance_information(graph)
    # fname = 'Datasets/giant_component_edges.csv'
    # data = read_csv(fname)
    # graph = create_graph(data)
    # print_distance_information(graph)

    # components(graph)

    # # component = giant_component_filtering(graph)
    # graph_comparison(graph, component)

    # fname = 'Datasets/condenced_graph.edgelist'
    # convert_to_csv(fname)

    # draw_graph(graph)
    # print(get_assortativity_coeff(graph))
    # plot_in_degree_distribution(graph)
    # plot_out_degree_distribution(graph)
    # plot_distances(nx.shortest_path_length(graph))
    # plot_avg_degree_connectivity(nx.average_degree_connectivity(graph))
    node_metrics = 'Datasets/node_metrics_giant_component.csv'
    metrics = read_csv(node_metrics)
    # plot_binned_betweeness_centrality(metrics)
    # plot_betweeness_centrality(metrics)
    # print(estimating_power_law(metrics))


if __name__ == '__main__':
    main()
