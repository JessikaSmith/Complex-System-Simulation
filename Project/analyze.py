# data source: http://konect.uni-koblenz.de/networks/cit-HepPh

IMG_PATH = 'Figures/'
import plotly

import plotly.graph_objs as go
from plotly.offline import plot
from plotly.plotly import image
import matplotlib.patches as mpatches
import networkx as nx
import matplotlib.pyplot as plt
from processing import *
import seaborn as sns
from random import choice
import math
import random
import pandas as pd
import numpy as np
from tqdm import tqdm

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

def plot_total_degree_distribution(graph):
    degs = {}
    for n in graph.nodes():
        deg = graph.degree(n)
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
    plt.title('BA Total Degree Distribution')
    fig.savefig(IMG_PATH + 'model_total_distribution_BA.png')


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
    fig.savefig(IMG_PATH + 'out_assortativity.png')


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
    layout = dict(title="Assortativity in",
                  xaxis=dict(title='k'),
                  yaxis=dict(title='$<k_{nn}>$')
                  )
    plot_data = [trace]
    fig = go.Figure(data=plot_data, layout=layout)
    image.save_as(fig, filename=IMG_PATH + 'assortativity_in_binned.jpeg')


def plot_binned_clustering_coeffs(data):
    values = data['clustering']
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
    layout = dict(title="Clustering Distribution",
                  xaxis=dict(title='Clustering Coefficient',
                             type='log'),
                  yaxis=dict(title='Count',
                             type='log')
                  )
    plot_data = [trace]
    fig = go.Figure(data=plot_data, layout=layout)
    image.save_as(fig, filename=IMG_PATH + 'clustering_binned.jpeg')


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


def diameter_ave_path_length(G):
    # We create our own function to do this so things are slightly faster,
    # we can calculate diameter and avg path length at the same time
    max_path_length = 0
    total = 0.0
    for n in G:  # iterate over all nodes
        path_length = nx.single_source_shortest_path_length(G, n)  # generate shortest paths from node n to all others
        total += sum(path_length.values())  # total of all shortest paths from n
        if max(path_length.values()) > max_path_length:  # keep track of longest shortest path we see.
            max_path_length = max(path_length.values())
    try:
        avg_path_length = total / (G.order() * (G.order() - 1))
    except ZeroDivisionError:
        avg_path_length = 0.0
    return max_path_length, avg_path_length


def all_network_statistics(nw_list):
    # a function that takes in a list of networks and returns 3 lists of same length listing the diameter, average
    # path length and giant component size for all the networks
    diameters = []
    path_lengths = []
    S = []
    densities = []
    average_degrees = []
    for n in nw_list:
        d, l, s, e, avd = a_network_statistics(n)
        diameters.append(d)
        path_lengths.append(l)
        S.append(s)
        densities.append(e)
        average_degrees.append(avd)
    return (diameters, path_lengths, S, densities, avd)


def a_network_statistics(graph):
    Gcc = sorted(nx.strongly_connected_component_subgraphs(graph), key=len, reverse=True)
    G0 = Gcc[0]
    d, l = diameter_ave_path_length(G0)
    s = float(G0.order()) / float(graph.number_of_nodes())
    e = nx.density(G0)
    avd = float(G0.size()) / float(graph.number_of_nodes())
    return d, l, s, e, avd


def fail(G):
    n = random.choice(list(G.nodes()))
    G.remove_node(n)


def attack_degree(G):
    degrees = dict(G.degree())
    max_degree = max(degrees.values())
    max_keys = [k for k, v in degrees.itams() if v == max_degree]
    G.remove_node(max_keys[0])


def attack_betweenness(G):
    betweenness = dict(nx.betweenness_centrality(G))
    max_betweenness = max(betweenness.values())
    max_keys = [k for k, v in betweenness.items() if v == max_betweenness]
    G.remove_node(max_keys[0])


def experiments(networks, removals, run_fail=True, measure_every_X_removals=20):
    # the below list will record the average statistic for all networks, a new entry in the list is added after each fail
    ave_diameters = []
    ave_path_lengths = []
    ave_S = []
    print("---- Starting Experiments ---- \n")
    print()
    for x in tqdm(range(removals)):
        for n in networks:
            if run_fail:
                fail(n)
            else:
                attack_degree(n)
        if x % measure_every_X_removals == 0:
            d, l, s, e, avd = all_network_statistics(networks)
            ave_diameters.append(np.mean(d))
            ave_path_lengths.append(np.mean(l))
            ave_S.append(np.mean(s))
    print("---- Experiments Finished ---- \n")
    print()
    return ave_diameters, ave_path_lengths, ave_S


def plot_failure_attack(G):
    NetworkSize = G.order()
    G_c = G.copy()
    anf_ave_diameters, anf_ave_path_lengths, anf_ave_S = experiments([G], int(NetworkSize * 0.8), run_fail=True,
                                                                     measure_every_X_removals=200)
    ana_ave_diameters, ana_ave_path_lengths, ana_ave_S = experiments([G_c], int(NetworkSize * 0.8), run_fail=False,
                                                                     measure_every_X_removals=200)
    fig_size = [18, 13]
    plt.rcParams.update({'font.size': 20, "figure.figsize": fig_size})
    xvalues = [(float(x) / float(NetworkSize)) * 200 for x in range(len(anf_ave_diameters))]

    # Plot diameter
    plt.plot(xvalues, anf_ave_diameters, '--or', xvalues, ana_ave_diameters, '--xb')
    plt.xlabel('f')
    plt.ylabel('diameter')
    plt.title('Attacks & Failures on Airline Network Networks size')
    red_patch = mpatches.Patch(color='red', label='Failures')
    blue_patch = mpatches.Patch(color='blue', label='Attacks')
    plt.legend(handles=[red_patch, blue_patch])
    plt.show()

    # Plot average path length
    plt.plot(xvalues, anf_ave_path_lengths, '--or', xvalues, ana_ave_path_lengths, '--xb')
    red_patch = mpatches.Patch(color='red', label='Failures')
    blue_patch = mpatches.Patch(color='blue', label='Attacks')
    plt.legend(handles=[red_patch, blue_patch])
    plt.xlabel('f')
    plt.ylabel('<l>')
    plt.title('Attacks & Failures on Airline Network Networks size')
    plt.show()

    # Plot fraction of nodes in giant component
    plt.plot(xvalues, anf_ave_S, '--or', xvalues, ana_ave_S, '--xb')
    red_patch = mpatches.Patch(color='red', label='Failures')
    blue_patch = mpatches.Patch(color='blue', label='Attacks')
    plt.legend(handles=[red_patch, blue_patch])
    plt.xlabel('f')
    plt.ylabel('S')
    plt.title('Attacks & Failures on Airline Network Networks size')
    plt.show()


def model_the_dist(graph):
    # graph.number_of_edges()
    # graph.number_of_nodes()
    new_graph = nx.generators.extended_barabasi_albert_graph(int(graph.number_of_edges()/100),int(graph.number_of_nodes()/100), 0.8, 0.01)
    plot_total_degree_distribution(new_graph)

def calculate_gamma(G):
    degrees = []
    for n in G.nodes():
        degrees += [G.out_degree(n)]
    print(degrees)
    #print(degrees[:5])
    degrees = sorted(degrees)
    degrees = [n for n in degrees if n >= 70]
    print(degrees[:5])
    exp = sum(map(lambda degree: math.log(degree / degrees[0]), degrees))
    exp = 1 + len(degrees) / exp
    return exp

def main():
    fname = 'Datasets/giant_component_edges.csv'
    data = read_csv(fname)
    graph = create_graph(data)
    # BA_graph = nx.barabasi_albert_graph(int(graph.number_of_edges()/100),int(graph.number_of_nodes()/100))
    # plot_total_degree_distribution(BA_graph)
    #print(calculate_gamma(graph))
    # plot_failure_attack(graph)
    # plot_distance_distribution(graph)
    # print_distance_information(graph)
    # fname = 'Datasets/giant_component_edges.csv'
    # data = read_csv(fname)
    # graph = create_graph(data)
    # vv = [len(c) for c in sorted(nx.strongly_connected_components(graph), key=len, reverse=True)]
    # print(choice(vv, 10))
    # components(graph)
    # # component = giant_component_filtering(graph)
    # graph_comparison(graph, component)

    # fname = 'Datasets/condenced_graph.edgelist'
    # convert_to_csv(fname)
    model_the_dist(graph)
    # draw_graph(graph)
    print(get_assortativity_coeff(graph))
    # plot_in_degree_distribution(graph)
    # plot_out_degree_distribution(graph)
    # plot_distances(nx.shortest_path_length(graph))
    # plot_avg_binned_degree_connectivity(nx.average_degree_connectivity(graph, source="out"))
    # plot_avg_binned_degree_connectivity(nx.average_degree_connectivity(graph, source="in"))
    # node_metrics = 'Datasets/clustering_modularity.csv'
    # metrics = read_csv(node_metrics)
    # plot_binned_clustering_coeffs(metrics)
    # plot_binned_betweeness_centrality(metrics)
    # plot_betweeness_centrality(metrics)
    # print(estimating_power_law(metrics))


if __name__ == '__main__':
    main()
