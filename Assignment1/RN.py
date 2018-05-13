import networkx as nx
import matplotlib.pyplot as plt

plt.interactive(True)


def generate_small_example(n, m, seed=42):
    graph = nx.gnm_random_graph(n, m, seed=seed)
    nx.draw(graph, pos=nx.spring_layout(graph), with_labels=True)
    derive_params(graph)


def derive_params(graph):
    print('Adjacency matrix')
    print(nx.adjacency_matrix(graph).todense())
    plot_degree_distribution(graph)
    print('Average distance')
    print(nx.average_shortest_path_length(graph))
    print(plot_clustering_coeffs(nx.clustering(graph)))
    print(plot_distances(nx.shortest_path_length(graph)))


def plot_distances(dist):
    dist_count = {}
    for i in range(len(dist.keys())-1):
        for j in range(i + 1, len(dist.keys())):
            if dist[i][j] not in dist_count:
                dist_count[dist[i][j]] = 0
            dist_count[dist[i][j]] += 1
    items = sorted(dist_count.items())
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([k for (k, v) in items], [v for (k, v) in items], 'ro')
    plt.xlabel('Number of nodes')
    ax.set_xlim([0, 5])
    ax.set_ylim([0, 7])
    plt.ylabel('Distance')
    plt.title('Distance Distribution')
    fig.savefig('distance_distribution.png')


def plot_clustering_coeffs(clust):
    items = sorted(clust.items())
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([k for (k, v) in items], [v for (k, v) in items], 'ro')
    plt.xlabel('Node')
    ax.set_xlim([0, 5])
    plt.xticks([k for (k, v) in items], [k for (k, v) in items])
    plt.yticks([v for (k, v) in items], [round(v, 3) for (k, v) in items])
    plt.ylabel('$C_i$')
    plt.title('Clustering Coefficients Distribution')
    fig.savefig('clust_distribution.png')
    pass


def plot_degree_distribution(graph):
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
    plt.xticks([k for (k, v) in items], [k for (k, v) in items])
    plt.yticks([v / len(graph.nodes()) for (k, v) in items], [v / len(graph.nodes()) for (k, v) in items])
    plt.ylabel('P(k)')
    plt.title('Degree Distribution')
    fig.savefig('degree_distribution.png')


# def main():
#     n, m = map(int, input().split())
#     generate_small_example(n, m)
#
#
# if __name__ == '__main__':
#     main()
